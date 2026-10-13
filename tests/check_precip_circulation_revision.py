"""Static contracts and real Matplotlib geometry checks; no science data required."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RevisionChecks(unittest.TestCase):
    def test_render_contracts(self):
        for filename, tag in [("paper_fig7_from_cache.py", "fig7"),
                              ("paper_fig8_from_cache.py", "fig8"),
                              ("paper_fig9_from_cache.py", "fig9")]:
            text = (ROOT / filename).read_text()
            tree = ast.parse(text)
            names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
            self.assertNotIn(f"_{tag}_overlay_z500_significance", names)
            self.assertIn(f"_{tag}_overlay_tp_significance", names)
            self.assertIn(f"_{tag}_contour_diff", names)
            self.assertNotIn("read_csv", text)
            self.assertNotIn("open_mfdataset", text)
            self.assertNotIn("cfgrib", text)
            self.assertNotIn("suptitle", text)
        daily = (ROOT / "paper_fig7_from_cache.py").read_text()
        self.assertIn('plt.subplots(4, 3', daily)
        self.assertIn('labels = list("abcdefghijkl")', daily)
        hov = (ROOT / "paper_fig8_from_cache.py").read_text()
        self.assertIn('labels = iter("bcdefg")', hov)
        self.assertIn('"(a)", "Observed"', hov)
        maps = (ROOT / "cell_leadtime_total_hotcold_composite_maps.py").read_text()
        self.assertIn('ANALYSIS_WINDOW_LABEL = "2023-06-14 to 2023-06-24"', maps)
        self.assertIn('INIT_DATES = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]', maps)

    def test_seven_panel_geometry_from_renderer(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tree = ast.parse((ROOT / "paper_fig8_from_cache.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "plot_fig7_hovmoller_from_cache")
        # Execute the actual GridSpec assignment from the renderer.
        grid = next(n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "gs" for t in n.targets))
        fig = plt.figure(figsize=(6.5, 9.4))
        ns = {"fig": fig}
        exec(compile(ast.Module(body=[grid], type_ignores=[]), "geometry", "exec"), ns)
        gs = ns["gs"]
        obs = fig.add_subplot(gs[0, 1:3])
        axes = [obs] + [fig.add_subplot(gs[r, c*2:c*2+2]) for r in range(1, 4) for c in range(2)]
        self.assertEqual(len(axes), 7)
        for ax in axes:
            self.assertAlmostEqual(ax.get_position().width, obs.get_position().width)
            self.assertAlmostEqual(ax.get_position().height, obs.get_position().height)
        self.assertAlmostEqual(obs.get_position().x0 + obs.get_position().width / 2, (0.145 + 0.985) / 2)
        plt.close(fig)

    def test_daily_fixed_aspect_layout_and_row_label_clearance(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tree = ast.parse((ROOT / "paper_fig7_from_cache.py").read_text())
        names = {"FIG7_SUBPLOT_LAYOUT", "FIG7_CAX_ABS", "FIG7_CAX_DIFF", "FONT_ROW_HEADER"}
        selected = [n for n in tree.body if (
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
        ) or (isinstance(n, ast.FunctionDef) and n.name == "_fig7_apply_daily_layout")]
        ns = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "daily-layout", "exec"), ns)
        fig, axs = plt.subplots(4, 3, figsize=(6.5, 7.1))
        for ax in axs.flat:
            ax.set_xlim(90, 135)
            ax.set_ylim(25, 55)
            ax.set_aspect("equal", adjustable="box")
            ax.set_yticks([25, 35, 45, 55], ["25°N", "35°N", "45°N", "55°N"])
        rows = [(name, None, None) for name in ("Observed", "Hot17", "Cold17", "Hot17 − Cold17")]
        labels = ns["_fig7_apply_daily_layout"](fig, axs, rows)
        renderer = fig.canvas.get_renderer()
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(fig.axes), 12)
        self.assertAlmostEqual(fig.get_figwidth(), 6.5)
        self.assertAlmostEqual(fig.get_figheight(), 7.1)
        self.assertGreaterEqual(fig.subplotpars.left, 0.105)
        self.assertLessEqual(fig.subplotpars.left, 0.120)
        for r, label in enumerate(labels):
            frame = axs[r, 0].get_position()
            text = label.get_window_extent(renderer)
            ticks = axs[r, 0].get_tightbbox(renderer)
            self.assertAlmostEqual(label.get_position()[1], (frame.y0 + frame.y1) / 2)
            self.assertGreaterEqual(text.x0, 0)
            self.assertLess(text.x1, ticks.x0)
            self.assertGreaterEqual(text.y0, 0)
            self.assertLessEqual(text.y1, fig.bbox.height)
            self.assertAlmostEqual(frame.width * 6.5 / (frame.height * 7.1), 45 / 30)
        for name, expected in (("FIG7_CAX_ABS", [0.17, 0.085, 0.28, 0.011]),
                               ("FIG7_CAX_DIFF", [0.60, 0.085, 0.30, 0.011])):
            self.assertEqual(ns[name], expected)
            cax = fig.add_axes(ns[name])
            self.assertAlmostEqual(cax.get_position().y0, 0.085)
            self.assertAlmostEqual(cax.get_position().height, 0.011)
        plt.close(fig)

    def test_both_daily_entries_share_layout(self):
        tree = ast.parse((ROOT / "paper_fig7_from_cache.py").read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        calls = [n.func.id for n in ast.walk(functions["plot_fig6_from_cache"])
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertIn("_fig7_apply_daily_layout", calls)
        calls = [n.func.id for n in ast.walk(functions["plot_figS6_from_cache"])
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertEqual(calls, ["plot_fig6_from_cache"])

    def test_visual_helpers_contour_paths_and_effective_pdf_hatch_width(self):
        import tempfile
        import re
        import zlib
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        names = {"_queue_contour_labels", "_place_contour_labels", "_save_visual_figure"}
        for tag in ("fig7", "fig8", "fig9"):
            tree = ast.parse((ROOT / f"paper_{tag}_from_cache.py").read_text())
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
            ns = {"np": np, "plt": plt, "pe": pe, "FONT_CONTOUR_LABEL": 8.5,
                  "TP_HATCH_LINEWIDTH": .60}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "visual-helpers", "exec"), ns)
            fig, ax = plt.subplots(figsize=(3, 3))
            x = np.linspace(0, 10, 80)
            xx, yy = np.meshgrid(x, x)
            cs = ax.contour(xx, yy, 50 * np.sin(xx / 3) + yy * 5 - 50,
                            levels=[-40, -20, 20, 40])
            before = [[segment.copy() for segment in level] for level in cs.allsegs]
            ax.text(.02, .97, "(a)", transform=ax.transAxes, va="top")
            ax.axvline(4, color="black")
            ns["_queue_contour_labels"](ax, cs)
            original_rc = plt.rcParams["hatch.linewidth"]
            with plt.rc_context({"hatch.color": "black", "hatch.linewidth": .60}):
                hatch = ax.contourf(xx, yy, np.ones_like(xx), levels=[.5, 1.5],
                                    colors="none", hatches=["..."], zorder=12)
            if hasattr(hatch, "get_hatch_linewidth"):
                self.assertAlmostEqual(hatch.get_hatch_linewidth(), .60)
            with tempfile.TemporaryDirectory() as tmp:
                stem = str(Path(tmp) / tag)
                ns["_save_visual_figure"](fig, stem)
                self.assertTrue(Path(stem + ".png").is_file())
                pdf = Path(stem + ".pdf").read_bytes()
                decoded = []
                for stream in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
                    try:
                        decoded.append(zlib.decompress(stream))
                    except zlib.error:
                        pass
                self.assertTrue(any(b"0.6 w" in stream for stream in decoded), "PDF hatch stroke width missing")
            self.assertEqual(plt.rcParams["hatch.linewidth"], original_rc)
            self.assertEqual({label.get_text() for label in cs.labelTexts}, {"-40", "-20", "20", "40"})
            for old, new in zip(before, cs.allsegs):
                for a, b in zip(old, new):
                    np.testing.assert_array_equal(a, b)  # no inline contour cuts
            # Label anchors must be on a segment of their own real contour level.
            for label in cs.labelTexts:
                index = list(cs.levels).index(float(label.get_text()))
                point = np.asarray(label.get_position())
                distances = []
                for segment in cs.allsegs[index]:
                    vectors = np.diff(segment, axis=0)
                    length2 = (vectors * vectors).sum(axis=1)
                    valid = length2 > 0
                    starts, vectors, length2 = segment[:-1][valid], vectors[valid], length2[valid]
                    fraction = np.clip(((point - starts) * vectors).sum(axis=1) / length2, 0, 1)
                    distances.extend(np.linalg.norm(point - starts - fraction[:, None] * vectors, axis=1))
                self.assertLess(min(distances), 1e-6)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
