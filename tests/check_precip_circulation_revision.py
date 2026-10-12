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


if __name__ == "__main__":
    unittest.main()
