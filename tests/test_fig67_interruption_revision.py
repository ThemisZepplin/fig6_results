"""Synthetic scientific/export and rendering checks; no research data."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]


def module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def exporter():
    return module("cell_fig6_interruption_totalgroup_export.py", "fig6_export_test")


@pytest.fixture
def renderer():
    return module("paper_fig6_interruption_from_cache.py", "fig6_render_test")


def science_fixture(exporter):
    # Only remove the historical source's top-level output-directory creation.
    # All production scientific function bodies execute unchanged.
    tree = ast.parse((ROOT / exporter.SOURCE_FILENAME).read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "os" and node.value.func.attr == "makedirs")]
    env = {"__name__": "science_fixture"}
    exec(compile(tree, exporter.SOURCE_FILENAME, "exec"), env)
    env["ANALYSIS_DATES"] = exporter.FIG6_DATES.copy()
    env["ANALYSIS_WINDOW_LABEL"] = exporter.FIG6_WINDOW
    dates = pd.date_range("2023-06-14", "2023-06-24")
    rng = np.random.default_rng(42)
    shape = (51, 11, 3, 3)
    member = np.arange(51)[:, None, None, None]
    day = np.arange(11)[None, :, None, None]
    tp = 1 + member * .02 + day * .03 + rng.normal(0, .025, size=shape)
    z = 5800 - member * .3 + day * .5 + rng.normal(0, .3, size=shape)
    coords = {"member": np.arange(51), "bjt_date": dates,
              "latitude": [30., 39., 50.], "longitude": [100., 116., 130.]}
    fields = {name: xr.DataArray(values, dims=tuple(coords), coords=coords)
              for name, values in (("tp", tp), ("z500_abs", z))}
    calls = []
    def load_fields(init):
        calls.append(init)
        return {name: array.copy(deep=True) for name, array in fields.items()}
    def members(_, init, group):
        rotated = np.roll(np.arange(51), exporter.FIG6_INITS.index(init) * 3)
        return rotated[:17].tolist() if group == "Hot17" else rotated[-17:].tolist()
    def reference():
        assert env["ANALYSIS_DATES"].equals(exporter.FIG6_DATES)
        return {
            "tp": xr.DataArray(np.full((2, 2), 1.5), dims=("latitude", "longitude"),
                               coords={"latitude": [35., 44.], "longitude": [114., 119.]}),
            "z500_abs": xr.DataArray(np.full((3, 3), 5800.), dims=("latitude", "longitude"),
                                     coords={"latitude": [30., 40., 50.], "longitude": [100., 115., 130.]}),
        }
    env.update(build_daily_s2s_member_fields=load_fields,
               load_total_hotcold_groups=lambda _: None,
               get_group_members=members, load_reference_fields=reference)
    return env, fields, calls


@pytest.fixture
def exported(tmp_path, monkeypatch, exporter):
    env, fields, calls = science_fixture(exporter)
    monkeypatch.setattr(exporter, "_fig6_science_namespace", lambda _: env)
    paths = exporter.export_fig6_interruption_cache_and_significance(root=tmp_path)
    with xr.open_dataset(paths["cache"]) as handle:
        cache = handle.load()
    with xr.open_dataset(paths["tp_significance"]) as handle:
        sig = handle.load()
    return paths, cache, sig, env, fields, calls


def test_three_day_member_mean_and_missing_day_rejection(exporter):
    dates = pd.date_range("2023-06-14", "2023-06-24")
    vals = np.broadcast_to(np.arange(11)[None, :, None, None], (34, 11, 2, 2)).astype(float).copy()
    daily = xr.DataArray(vals, dims=("member", "bjt_date", "latitude", "longitude"),
                         coords={"member": range(34), "bjt_date": dates,
                                 "latitude": [35., 44.], "longitude": [114., 119.]})
    mean, qc = exporter._fig6_member_mean(daily, list(range(34)), "2023-06-12", "tp", (90, 135, 25, 55))
    np.testing.assert_allclose(mean, 5)  # indices 4/5/6, not all 11 days
    assert qc["expected_day_count"] == 3
    daily.loc[dict(member=0, bjt_date="2023-06-19", latitude=35., longitude=114.)] = np.nan
    with pytest.raises(ValueError, match="incomplete three-day"):
        exporter._fig6_member_mean(daily, list(range(34)), "2023-06-12", "tp", (90, 135, 25, 55))
    with pytest.raises(ValueError, match="missing BJT dates"):
        exporter._fig6_member_mean(daily.drop_sel(bjt_date="2023-06-18"), list(range(34)), "x", "tp", (90, 135, 25, 55))


def test_export_four_inits_welch_and_netcdf_roundtrip(exported, exporter, renderer):
    paths, cache, sig, env, fields, calls = exported
    assert calls == list(exporter.FIG6_INITS)  # exactly one member-field load per init
    assert cache.tp_hot.shape == (4, 3, 3)
    assert cache.tp_obs_ref.dims == ("mswep_lat", "mswep_lon")
    assert cache.z500_abs_obs_ref.dims == ("era5_lat", "era5_lon")
    assert "member" not in cache.dims and "bjt_date" not in cache.dims
    assert cache.attrs["analysis_window"] == exporter.FIG6_WINDOW
    assert cache.attrs["grouping_window"] == exporter.FIG6_GROUPING_WINDOW
    for index, init in enumerate(exporter.FIG6_INITS):
        hot = env["get_group_members"](None, init, "Hot17")
        cold = env["get_group_members"](None, init, "Cold17")
        values = fields["tp"].sel(bjt_date=exporter.FIG6_DATES).mean("bjt_date")
        np.testing.assert_allclose(cache.tp_hot.isel(init=index), values.sel(member=hot).mean("member"))
        expected = ttest_ind(values.sel(member=hot).values, values.sel(member=cold).values,
                             axis=0, equal_var=False, nan_policy="omit", alternative="two-sided").pvalue
        np.testing.assert_allclose(sig.p_value.isel(init=index), expected)
    renderer._fig6_validate_significance(cache, sig)
    renderer._fig6_validate_products(cache, sig)
    assert all(Path(path).is_file() for path in paths.values())
    qc = pd.read_csv(paths["significance_qc"])
    assert len(qc) == 8
    assert set(qc.expected_day_count) == {3}
    with pytest.raises(FileExistsError):
        exporter.export_fig6_interruption_cache_and_significance(root=Path(paths["cache"]).parents[1])


@pytest.mark.parametrize("key,value", [
    ("analysis_window", "2023-06-14 to 2023-06-24"),
    ("grouping_window", "2023-06-18 to 2023-06-20"),
    ("grouping_stage", "Stage-I"), ("mean_day_count", 11),
    ("z500_type", "anomaly"), ("tp_units", "mm"),
])
def test_reject_wrong_product_identity(exported, renderer, key, value):
    _, cache, sig, *_ = exported
    bad = sig.copy(deep=True)
    bad.attrs[key] = value
    with pytest.raises(ValueError):
        renderer._fig6_validate_products(cache, bad)


def test_reject_two_inits_wrong_mask_grid_and_difference(exported, renderer):
    _, cache, sig, *_ = exported
    with pytest.raises(ValueError):
        renderer._fig6_validate_products(cache.isel(init=[2, 3]), sig.isel(init=[2, 3]))
    bad = sig.copy(deep=True)
    bad["significant_local_p005"] = xr.zeros_like(bad.significant_local_p005)
    with pytest.raises(ValueError, match="local mask"):
        renderer._fig6_validate_products(cache, bad)
    bad = sig.assign_coords(s2s_lon=sig.s2s_lon + .5)
    with pytest.raises(ValueError):
        renderer._fig6_validate_significance(cache, bad)
    bad = cache.copy(deep=True)
    bad["z500_abs_diff"] = bad.z500_abs_diff + 1
    bad.z500_abs_diff.attrs["units"] = "gpm"
    with pytest.raises(ValueError, match="difference"):
        renderer._fig6_validate_products(bad, sig)


def test_no_raw_io_in_new_renderers():
    for filename in ("paper_fig6_interruption_from_cache.py", "paper_fig8_from_cache.py"):
        tree = ast.parse((ROOT / filename).read_text())
        forbidden = {"read_csv", "open_mfdataset", "open_grib", "load_s2s_ensemble_with_cf"}
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Name, ast.Attribute))}
        assert not calls & forbidden
        assert "cfgrib" not in (ROOT / filename).read_text()


def test_closed_positive_label_is_on_own_real_contour(renderer, tmp_path):
    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(4, 3, left=.145, right=.985, top=.965, bottom=.27, hspace=.11, wspace=.10)
    ax = fig.add_subplot(gs[3, 2])
    ax.set_xlim(90, 135); ax.set_ylim(25, 55); ax.set_aspect("equal")
    ax._require_positive_z500_label = True
    x, y = np.linspace(90, 135, 151), np.linspace(25, 55, 101)
    xx, yy = np.meshgrid(x, y)
    cs = ax.contour(xx, yy, 18*np.exp(-((xx-113)/7.5)**2-((yy-39)/5.5)**2), levels=[10, 20], colors="firebrick")
    before = [[segment.copy() for segment in levels] for levels in cs.allsegs]
    renderer._queue_contour_labels(ax, cs, [20, 10])
    import matplotlib.patheffects as pe
    line, = ax.plot([114,119,119,114,114], [35,35,44,44,35], "k--", lw=1.35)
    line.set_path_effects([pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])
    original = plt.rcParams["hatch.linewidth"]
    renderer._save_visual_figure(fig, str(tmp_path/"closed_loop"))
    assert any(label.get_text() == "10" for label in cs.labelTexts)
    for label in cs.labelTexts:
        index = list(cs.levels).index(float(label.get_text()))
        p = ax.transData.transform(label.get_position())
        distance = []
        for segment in cs.allsegs[index]:
            q = ax.transData.transform(segment); v = np.diff(q, axis=0)
            if not len(v): continue
            fraction = np.clip(((p-q[:-1])*v).sum(1)/np.maximum((v*v).sum(1), 1e-20), 0, 1)
            distance.extend(np.linalg.norm(p-q[:-1]-fraction[:,None]*v, axis=1))
        assert min(distance) < 1e-6
    for aa, bb in zip(before, cs.allsegs):
        for a, b in zip(aa, bb):
            np.testing.assert_array_equal(a, b)
    assert plt.rcParams["hatch.linewidth"] == original
    assert (tmp_path/"closed_loop.png").is_file()
    assert (tmp_path/"closed_loop.pdf").is_file()
    plt.close(fig)


def test_hovmoller_headings_dates_and_equal_geometry():
    hov = module("paper_fig8_from_cache.py", "hov_render_test")
    dates = np.arange(np.datetime64("2023-06-14"), np.datetime64("2023-06-25"))
    fn = next(n for n in ast.parse((ROOT/"paper_fig8_from_cache.py").read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "plot_fig7_hovmoller_from_cache")
    grid = next(n for n in fn.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "gs" for t in n.targets))
    fig = plt.figure(figsize=(6.5, 9.4)); ns = {"fig": fig}
    exec(compile(ast.Module(body=[grid], type_ignores=[]), "actual-grid", "exec"), ns)
    gs = ns["gs"]
    axes = [fig.add_subplot(gs[0,1:3])] + [
        fig.add_subplot(gs[r,c*2:c*2+2]) for r in range(1,4) for c in range(2)]
    for ax, letter in zip(axes, "abcdefg"):
        hov._fig8_format(ax, dates, show_xlabel=True)
        hov._fig8_heading(ax, f"({letter})", "June 12 · Hot17 − Cold17")
    fig.canvas.draw()
    boxes = np.array([[a.get_position().width, a.get_position().height] for a in axes])
    np.testing.assert_allclose(boxes, np.broadcast_to(boxes[0], boxes.shape), atol=1e-9)
    assert axes[0].get_yticklabels()[0].get_text() == "6-14"
    assert axes[0].get_yticklabels()[-1].get_text() == "6-24"
    for ax in axes:
        assert ax.title.get_fontweight() == "bold"
        assert ax.title.get_fontsize() == 11
        assert ax.title.get_window_extent(fig.canvas.get_renderer()).y0 > ax.get_window_extent().y1
        assert not any(t.get_text().startswith("(") for t in ax.texts)
    center = (axes[0].get_position().x0 + axes[0].get_position().x1)/2
    np.testing.assert_allclose(center, (.10+.97)/2)
    plt.close(fig)


def test_preserved_source_blobs():
    expected = {
        "cell_leadtime_total_hotcold_composite_maps.py": "1ec1fd65e5db052b2d5e6a4b4e338f63491343e6",
        "cell_hovmoller_tp_z500_obs_s2s_stageII.py": "2eacaeba94585f5401efa3eb95fad1f9caa83742",
        "paper_fig7_from_cache.py": "248ec6cd1750f19160591f6fedc9b81569c20791",
        "paper_fig9_from_cache.py": "b32d7deed1283477f8464d34103277740d60f7ed",
    }
    for filename, sha in expected.items():
        actual = subprocess.check_output(["git", "hash-object", str(ROOT/filename)], text=True).strip()
        assert actual == sha, f"Preserved source changed: {filename}"



def test_full_synthetic_hovmoller_png_pdf(tmp_path):
    hov = module("paper_fig8_from_cache.py", "hov_full_test")
    dates = pd.date_range("2023-06-14", "2023-06-24")
    lon = np.linspace(90, 135, 31)
    day, xx = np.meshgrid(np.arange(11), lon, indexing="ij")
    tp_obs = 2 + np.sin(xx/5 + day/4)
    z_obs = 5800 + 45*np.cos(xx/12) + day*3
    inputs = {}
    for init, suffix, scale in (("2023-06-08", "0608", 1.), ("2023-06-12", "0612", 1.2)):
        hot = 2 + .6*np.sin(xx/5 + day/4)
        cold = hot + scale*.5*np.cos(xx/7 + day/3)
        z_hot = z_obs + 15
        z_cold = z_hot - scale*25*np.sin(xx/9 + day/5)
        values = {
            "tp_obs_hov": tp_obs, "z500_obs_hov": z_obs,
            "tp_hot_hov": hot, "tp_cold_hov": cold, "tp_diff_hov": hot-cold,
            "z500_hot_hov": z_hot, "z500_cold_hov": z_cold, "z500_diff_hov": z_hot-z_cold,
        }
        data_vars = {}
        for name, value in values.items():
            kind = "mswep" if name == "tp_obs_hov" else "era5" if name == "z500_obs_hov" else "s2s"
            data_vars[name] = xr.DataArray(
                value, dims=("time", f"{kind}_lon"), coords={"time": dates, f"{kind}_lon": lon},
                attrs={"units": "mm day^-1" if name.startswith("tp_") else "gpm"})
        data_vars["tp_abs_levels"] = xr.DataArray(np.linspace(0,6,13), dims="tp_abs_level")
        data_vars["tp_diff_levels"] = xr.DataArray(np.linspace(-6,6,13), dims="tp_diff_level")
        cache = xr.Dataset(data_vars, attrs={
            "initialization": init, "analysis_window": "2023-06-14 to 2023-06-24",
            "grouping_stage": "Total", "z500_type": "absolute", "z500_units": "gpm",
            "tp_units": "mm day^-1", "hovmoller_lat_band": "(35.0, 44.0)",
            "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17"})
        p = np.where(hot < cold, .01, .2)
        sig = xr.Dataset({
            "hot_minus_cold": (("time","s2s_lon"), hot-cold),
            "p_value": (("time","s2s_lon"), p),
            "significant_local_p005": (("time","s2s_lon"), (p<.05).astype("int8")),
            "n_hot": (("time","s2s_lon"), np.full_like(p,17)),
            "n_cold": (("time","s2s_lon"), np.full_like(p,17)),
            "degrees_of_freedom": (("time","s2s_lon"), np.full_like(p,30)),
        }, coords={"time": dates, "s2s_lon": lon},
        attrs={"initialization":init, "grouping_stage":"Total",
               "units":"mm day^-1", "difference_direction":"Hot17 minus Cold17"})
        cp, sp = tmp_path/f"cache_{suffix}.nc", tmp_path/f"sig_{suffix}.nc"
        cache.to_netcdf(cp); sig.to_netcdf(sp)
        inputs[f"cache_{suffix}"] = str(cp)
        inputs[f"tp_significance_{suffix}"] = str(sp)
    png, pdf = hov.plot_fig7_hovmoller_from_cache(**inputs, out_dir=str(tmp_path))
    assert Path(png).is_file() and Path(pdf).is_file()


def test_full_synthetic_four_init_map_png_pdf(exported, renderer, monkeypatch, tmp_path):
    paths, cache, sig, *_ = exported
    def setup_without_downloads(ax):
        ax.set_extent(renderer.FIG6_MAP_EXTENT, crs=renderer.ccrs.PlateCarree())
    monkeypatch.setattr(renderer, "_fig6_setup_map", setup_without_downloads)
    save = renderer._save_visual_figure
    def inspect_then_save(fig, stem):
        assert len(fig.axes) == 15  # twelve forecasts + one observation + two bars
        np.testing.assert_allclose(fig.axes[-2].get_position().y0, .160)
        np.testing.assert_allclose(fig.axes[-1].get_position().y0, .085)
        assert [text.get_text() for text in fig.texts] == ["1 Jun", "5 Jun", "8 Jun", "12 Jun"]
        assert all(text.get_position()[0] == .085 for text in fig.texts)
        save(fig, stem)
    monkeypatch.setattr(renderer, "_save_visual_figure", inspect_then_save)
    png, pdf = renderer.plot_fig6_from_cache(
        cache_path=paths["cache"], tp_significance_path=paths["tp_significance"],
        out_dir=str(tmp_path/"rendered"))
    assert Path(png).is_file() and Path(pdf).is_file()
