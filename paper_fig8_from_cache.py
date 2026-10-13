"""Plot paper Fig.8 from a plot-ready NetCDF cache only."""

import os
from datetime import datetime
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from cartopy.mpl.ticker import LongitudeFormatter

FIG8_DEFAULT_CACHE = "/data1/huangy/fig6/NC/v12/paper_figures/cache/fig8_hovmoller_plot_cache.nc"
FIG8_DEFAULT_OUT_DIR = "/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in"
FIG8_TP_SIGNIFICANCE = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig8_hovmoller_significance_tp.nc"
FIG8_Z500_SIGNIFICANCE = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig8_hovmoller_significance_z500.nc"
FIG8_ABS_TICKS = [0, 1.5, 3.0, 4.5, 6.0]
FIG8_DIFF_TICKS = [-6, -3, 0, 3, 6]
FIG_WIDTH_IN = 6.5
FIG8_HEIGHT_IN = 5.3
LON_MAJOR = np.arange(90, 131, 10)
LON_MINOR = np.arange(95, 131, 10)
FONT_PANEL_LETTER = 10.5
FONT_PANEL_TITLE = 9.5
FONT_AXIS_LABEL = 9.5
FONT_TICK = 8.5
FONT_COLORBAR_TICK = 8.5
FONT_COLORBAR_LABEL = 9.0
FONT_CONTOUR_LABEL = 8.5

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42



TP_HATCH_LINEWIDTH = 0.60


def _queue_contour_labels(ax, contour, levels=None):
    # Delay placement until Cartopy aspect and the individual figure layout settle.
    if not hasattr(ax, "_pending_contour_labels"):
        ax._pending_contour_labels = []
    ax._pending_contour_labels.append((contour, contour.levels if levels is None else levels))


def _place_contour_labels(fig):
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Text extents exclude path effects: include half the existing 3.2-pt stroke.
    stroke_pad = (3.2 / 2 + 0.5) * fig.dpi / 72
    separation = 1.5 * fig.dpi / 72
    for ax in fig.axes:
        pending = getattr(ax, "_pending_contour_labels", [])
        if not pending:
            continue
        frame = ax.get_window_extent(renderer)
        inner = Bbox.from_extents(frame.x0 + separation, frame.y0 + separation,
                                 frame.x1 - separation, frame.y1 - separation)
        occupied = []
        for text in ax.texts:
            if text.get_visible():
                box = text.get_bbox_patch()
                occupied.append((box if box is not None else text).get_window_extent(renderer).padded(separation))
        # Deliberate contour intersections are allowed; avoid only drawn reference
        # lines (NCHN box or 114/119E), not the actual contour carrying the number.
        references = [line.get_transform().transform_path(line.get_path())
                      for line in ax.lines if line.get_visible()]
        for contour, levels in pending:
            for level in levels:
                index = int(np.flatnonzero(np.isclose(contour.levels, level))[0])
                segments = sorted(contour.allsegs[index], key=len, reverse=True)
                placed = False
                for segment in segments:
                    if len(segment) < 2:
                        continue
                    points = contour.get_transform().transform(segment)
                    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                    if not np.isfinite(points).all() or distance[-1] == 0:
                        continue
                    # Try center first, then progressively more peripheral sites.
                    for fraction in sorted(np.linspace(.08, .92, 29), key=lambda f: abs(f - .5)):
                        target = fraction * distance[-1]
                        j = min(np.searchsorted(distance, target, side="right") - 1, len(points) - 2)
                        length = distance[j + 1] - distance[j]
                        if length <= 0:
                            continue
                        pixel = points[j] + (points[j + 1] - points[j]) * ((target - distance[j]) / length)
                        if not inner.contains(*pixel):
                            continue
                        position = ax.transData.inverted().transform(pixel)
                        previous = len(contour.labelTexts)
                        ax.clabel(contour, manual=[position],
                                  fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=False)
                        if len(contour.labelTexts) == previous:
                            continue
                        text = contour.labelTexts[-1]
                        text.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
                        bounds = text.get_window_extent(renderer).padded(stroke_pad)
                        clear = (text.get_text() == ("%.0f" % level)
                                 and inner.contains(bounds.x0, bounds.y0)
                                 and inner.contains(bounds.x1, bounds.y1)
                                 and not any(bounds.overlaps(box) for box in occupied)
                                 and not any(path.intersects_bbox(bounds.padded((3.4 / 2 + 1.5) * fig.dpi / 72), filled=False)
                                             for path in references))
                        if clear:
                            text._final_contour_label = True
                            occupied.append(bounds.padded(separation))
                            placed = True
                            break
                        # Non-inline labels never cut the contour, including rejected trials.
                        text.remove()
                        contour.labelTexts.pop()
                        contour.labelCValues.pop()
                    if placed:
                        break
                if not placed:
                    print(f"[Contour labels] level={level:g}: no safe on-contour location; contour retained")
        ax._pending_contour_labels = []
    # Check all accepted labels again after the final canvas draw, including stroke.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        accepted = [text for text in ax.texts if getattr(text, "_final_contour_label", False)]
        occupied = []
        for text in ax.texts:
            if text not in accepted and text.get_visible():
                patch = text.get_bbox_patch()
                occupied.append((patch if patch is not None else text).get_window_extent(renderer).padded(separation))
        references = [line.get_transform().transform_path(line.get_path())
                      for line in ax.lines if line.get_visible()]
        for text in accepted:
            box = text.get_window_extent(renderer).padded(stroke_pad)
            frame = ax.get_window_extent(renderer)
            if not (frame.contains(box.x0, box.y0) and frame.contains(box.x1, box.y1)):
                raise ValueError("Final stroked contour label is clipped")
            if (any(box.overlaps(other) for other in occupied)
                    or any(path.intersects_bbox(box.padded((3.4 / 2 + 1.5) * fig.dpi / 72), filled=False)
                           for path in references)):
                raise ValueError("Final stroked contour label collides with text or reference line")
            occupied.append(box.padded(separation))


def _save_visual_figure(fig, stem):
    # Creation context covers versions caching artist hatch width; this context
    # also covers versions/backends reading rcParams at draw or save time.
    with plt.rc_context({"hatch.color": "black", "hatch.linewidth": TP_HATCH_LINEWIDTH}):
        _place_contour_labels(fig)
        fig.savefig(stem + ".png", dpi=600)
        fig.savefig(stem + ".pdf")


def _initialization_from_attrs(attrs):
    """Resolve a valid date by precedence, never by filename or plotted dates."""
    invalid = []
    for key in ("initialization", "init_date", "main_init_date"):
        value = attrs.get(key)
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        if not text:
            continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            invalid.append(key)
    if invalid:
        raise ValueError(f"No valid initialization metadata in {invalid}")
    return None


def _validate_identity(ds, init_date, start, end, time_coord):
    declared = _initialization_from_attrs(ds.attrs)
    if declared != init_date:
        raise ValueError(f"Cache initialization {declared!r} differs from {init_date}")
    if ds.attrs.get("grouping_stage") != "Total":
        raise ValueError("Cache must use Total-period Tmax groups")
    if ds.attrs.get("analysis_window") != f"{start} to {end}":
        raise ValueError("Cache analysis window mismatch")
    dates = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D"))
    if not np.array_equal(ds[time_coord].values.astype("datetime64[D]"), dates):
        raise ValueError("Cache BJT dates must be complete and ordered")
    if ds.attrs.get("z500_type") != "absolute" or ds.attrs.get("z500_units") != "gpm":
        raise ValueError("Expected absolute Z500 in gpm")
    if ds.attrs.get("tp_units") != "mm day^-1":
        raise ValueError("Expected daily precipitation in mm day^-1")
    for name in ds.data_vars:
        if name.startswith(("tp_", "z500_")) and name not in ("tp_abs_levels", "tp_diff_levels"):
            expected = "mm day^-1" if name.startswith("tp_") else "gpm"
            if ds[name].attrs.get("units") != expected:
                raise ValueError(f"Wrong units for {name}")
    for key in ("hot_members", "cold_members"):
        if key in ds.attrs:
            members = ds.attrs[key].split(",")
            if len(set(members)) != 17:
                raise ValueError(f"Invalid {key} provenance")
    if "hot_members" in ds.attrs and "cold_members" in ds.attrs:
        if set(ds.attrs["hot_members"].split(",")) & set(ds.attrs["cold_members"].split(",")):
            raise ValueError("Hot and Cold members overlap")

def _validate_pair(cache, sig, init_date):
    declared = _initialization_from_attrs(sig.attrs)
    # Preserve historical June 12-only compatibility, never infer June 8.
    if declared is None and init_date == "2023-06-12":
        declared = "2023-06-12"
    if declared != init_date or sig.attrs.get("grouping_stage") != "Total":
        raise ValueError("TP significance initialization/grouping mismatch")
    if sig.attrs.get("units") != "mm day^-1":
        raise ValueError("Expected TP significance, never a Z500 mask")
    for key in ("hot_members", "cold_members"):
        if key in cache.attrs and sig.attrs.get(key) != cache.attrs[key]:
            raise ValueError(f"Cache/significance {key} mismatch")

def _fig8_effects(texts):
    for txt in texts:
        txt.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])


def _fig8_abs_levels(arrays: Sequence[xr.DataArray]) -> np.ndarray:
    pool = np.concatenate([a.values.ravel() for a in arrays])
    pmax = min(max(float(np.nanpercentile(pool, 95)), 1.0), 25.0)
    return np.linspace(0, pmax, 13)


def _fig8_diff_levels(da: xr.DataArray) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    vmax = max(float(np.nanpercentile(np.abs(da.values), 98)), 0.5)
    return np.linspace(-vmax, vmax, 13), mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _fig8_levels_from_cache_or_fallback(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, mcolors.TwoSlopeNorm]:
    if "tp_abs_levels" in ds.data_vars:
        abs_levels = ds["tp_abs_levels"].values
    else:
        print("[WARNING] Fig.8 cache missing tp_abs_levels; recomputing fallback levels from cached fields.")
        abs_levels = _fig8_abs_levels([ds.tp_obs_hov, ds.tp_hot_hov, ds.tp_cold_hov])

    if "tp_diff_levels" in ds.data_vars:
        diff_levels = ds["tp_diff_levels"].values
        diff_norm = mcolors.TwoSlopeNorm(
            vmin=float(np.nanmin(diff_levels)),
            vcenter=0.0,
            vmax=float(np.nanmax(diff_levels)),
        )
    else:
        print("[WARNING] Fig.8 cache missing tp_diff_levels; recomputing fallback levels from cached fields.")
        diff_levels, diff_norm = _fig8_diff_levels(ds.tp_diff_hov)
    return abs_levels, diff_levels, diff_norm


def _fig8_require_cache_vars(ds: xr.Dataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Fig.8 cache is missing required data_vars: {missing}")


def _fig8_check_z500_metadata(ds: xr.Dataset) -> None:
    z500_type = str(ds.attrs.get("z500_type", "")).lower()
    if "absolute" not in z500_type:
        raise ValueError(
            "Fig.8 from-cache contours assume absolute Z500 for obs/hot/cold; "
            f"got z500_type={ds.attrs.get('z500_type')!r}"
        )


def _fig8_pad(values, n):
    y = np.arange(n, dtype=float)
    ypad = np.r_[y[0] - 0.5, y, y[-1] + 0.5]
    return y, ypad, np.vstack([values[0:1, :], values, values[-1:, :]])


def _fig8_format(ax, dates, show_xlabel=False, show_ylabels=True):
    y = np.arange(len(dates), dtype=float)
    ax.set_ylim(y[-1] + 0.5, y[0] - 0.5)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([str(np.datetime_as_string(d, unit="D")) for d in dates])
        for tick, d in zip(ax.get_yticklabels(), dates):
            day = str(np.datetime_as_string(d, unit="D"))
            tick.set_color("#D55E00" if day <= "2023-06-17" else "#0072B2")
            tick.set_fontweight("bold")
            tick.set_fontsize(FONT_TICK)
    ax.tick_params(axis="y", labelleft=show_ylabels)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.axvline(114, color="0.25", ls="--", lw=0.8)
    ax.axvline(119, color="0.25", ls="--", lw=0.8)
    ax.set_xlim(90, 135)
    ax.set_xticks(LON_MAJOR)
    ax.set_xticks(LON_MINOR, minor=True)
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="°"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelbottom=show_xlabel)
    ax.grid(axis="x", which="major", linewidth=0.25, linestyle="--", alpha=0.35)
    ax.grid(axis="x", which="minor", linewidth=0.25, linestyle="--", alpha=0.35)
    if show_xlabel:
        ax.set_xlabel("Longitude (°E)", fontsize=FONT_AXIS_LABEL)


def _fig8_label(ax, text):
    ax.text(0.012, 0.975, text, transform=ax.transAxes, ha="left", va="top", fontsize=FONT_PANEL_LETTER, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.8))


def _fig8_contour_abs(ax, lon, y, z):
    vals = z.values
    lo = np.floor(np.nanmin(vals) / 20) * 20
    hi = np.ceil(np.nanmax(vals) / 20) * 20
    levels = np.arange(lo, hi + 1e-6, 20)
    cs = ax.contour(lon, y, vals, levels=levels, colors="0.12", linewidths=1.25)
    _queue_contour_labels(ax, cs, cs.levels[::2])
    if np.nanmin(vals) <= 5880 <= np.nanmax(vals):
        ax.contour(lon, y, vals, levels=[5880], colors="purple", linewidths=2.2)


def _fig8_contour_diff(ax, lon, y, z):
    base = np.array([20, 40, 60, 80], dtype=float)
    base = base[base <= float(np.nanmax(np.abs(z.values)))]
    if base.size:
        csn = ax.contour(lon, y, z.values, levels=-base[::-1], colors="royalblue", linestyles="--", linewidths=1.15)
        csp = ax.contour(lon, y, z.values, levels=base, colors="firebrick", linewidths=1.15)
        label_levels = base[np.isin(base, [20, 40, 60, 80])]
        if label_levels.size:
            _queue_contour_labels(ax, csn, -label_levels[::-1])
            _queue_contour_labels(ax, csp, label_levels)


def _fig8_plot_abs(ax, tp, z, dates, levels, label, title, show_xlabel=False, show_ylabels=True):
    tp_lon_dim = _fig8_lon_dim(tp)
    z_lon_dim = _fig8_lon_dim(z)
    y, ypad, tppad = _fig8_pad(tp.values, len(dates))
    cf = ax.contourf(tp[tp_lon_dim], ypad, tppad, levels=levels, cmap="Blues", extend="max", alpha=0.92)
    _fig8_contour_abs(ax, z[z_lon_dim], y, z)
    _fig8_format(ax, dates, show_xlabel, show_ylabels)
    _fig8_label(ax, label)
    ax.set_title(title, fontsize=FONT_PANEL_TITLE, pad=3)
    return cf


def _fig8_plot_diff(ax, tp, z, dates, levels, norm, label, title, show_xlabel=False, show_ylabels=True):
    tp_lon_dim = _fig8_lon_dim(tp)
    z_lon_dim = _fig8_lon_dim(z)
    y, ypad, tppad = _fig8_pad(tp.values, len(dates))
    cf = ax.contourf(tp[tp_lon_dim], ypad, tppad, levels=levels, cmap="RdBu_r", norm=norm, extend="both", alpha=0.92)
    _fig8_contour_diff(ax, z[z_lon_dim], y, z)
    _fig8_format(ax, dates, show_xlabel, show_ylabels)
    _fig8_label(ax, label)
    ax.set_title(title, fontsize=FONT_PANEL_TITLE, pad=3)
    return cf


def _fig8_lon_dim(da: xr.DataArray) -> str:
    lon_dims = [dim for dim in da.dims if dim.endswith("_lon") or dim == "lon"]
    if not lon_dims:
        raise ValueError(f"Cannot infer longitude dim for {da.name}: dims={da.dims}")
    return lon_dims[0]


def _fig8_open_significance(path: str, variable: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fig.8 {variable} significance file not found: {path}")
    ds = xr.open_dataset(path)
    _fig8_require_cache_vars(
        ds,
        [
            "hot_minus_cold", "p_value", "significant_local_p005",
            "n_hot", "n_cold", "degrees_of_freedom",
        ],
    )
    return ds


def _fig8_validate_significance(
    cache: xr.Dataset,
    tp_sig: xr.Dataset,
    z500_sig: xr.Dataset,
    time_coord: str,
) -> None:
    expected_time = cache[time_coord].values
    expected_lon = cache["s2s_lon"].values
    for variable, sig, expected in (
        ("tp", tp_sig, cache["tp_diff_hov"]),
    ):
        if str(sig.attrs.get("difference_direction", "")) != "Hot17 minus Cold17":
            raise ValueError(
                f"Fig.8 {variable} significance difference_direction must be "
                f"'Hot17 minus Cold17'; got {sig.attrs.get('difference_direction')!r}"
            )
        if not np.array_equal(sig["time"].values, expected_time):
            raise ValueError(f"Fig.8 {variable} significance time coordinate differs from plot cache")
        if not np.array_equal(sig["s2s_lon"].values, expected_lon):
            raise ValueError(f"Fig.8 {variable} significance longitude differs from plot cache")
        if sig["hot_minus_cold"].shape != expected.shape:
            raise ValueError(
                f"Fig.8 {variable} significance shape {sig['hot_minus_cold'].shape} "
                f"does not match plot-cache difference shape {expected.shape}"
            )
        if not np.allclose(
            sig["hot_minus_cold"].values,
            expected.values,
            rtol=1e-7,
            atol=1e-6,
            equal_nan=True,
        ):
            raise ValueError(f"Fig.8 {variable} significance composite differs from plot cache")
        p_value = sig["p_value"].values
        finite = p_value[np.isfinite(p_value)]
        if finite.size and ((finite < 0).any() or (finite > 1).any()):
            raise ValueError(f"Fig.8 {variable} local p-values fall outside [0, 1]")
        expected_mask = np.isfinite(p_value) & (p_value < 0.05)
        actual_mask = sig["significant_local_p005"].values.astype(bool)
        if not np.array_equal(actual_mask, expected_mask):
            raise ValueError(
                f"Fig.8 {variable} significance mask is inconsistent "
                "with finite local p < 0.05"
            )
        valid = np.isfinite(p_value)
        if np.any(sig["n_hot"].values[valid] != 17):
            raise ValueError(
                f"Fig.8 {variable} contains valid cells with non-17 Hot samples"
            )
        if np.any(sig["n_cold"].values[valid] != 17):
            raise ValueError(
                f"Fig.8 {variable} contains valid cells with non-17 Cold samples"
            )


def _fig8_overlay_tp_significance(ax, mask: xr.DataArray, dates) -> None:
    sig = mask.astype(bool)
    if not bool(sig.any().item()):
        return
    _, ypad, sigpad = _fig8_pad(sig.astype(float).values, len(dates))
    with plt.rc_context({"hatch.color": "black", "hatch.linewidth": TP_HATCH_LINEWIDTH}):
        ax.contourf(
            sig["s2s_lon"],
            ypad,
            sigpad,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["..."],
            zorder=6,
        )



FIG8_0608_CACHE = FIG8_DEFAULT_CACHE.replace("fig8_hovmoller_plot_cache", "fig8_hovmoller_init2023-06-08_plot_cache")
FIG8_0608_TP_SIGNIFICANCE = FIG8_TP_SIGNIFICANCE.replace("fig8_hovmoller", "fig8_init2023-06-08_hovmoller")

def plot_fig7_hovmoller_from_cache(
    cache_0608=FIG8_0608_CACHE,
    cache_0612=FIG8_DEFAULT_CACHE,
    tp_significance_0608=FIG8_0608_TP_SIGNIFICANCE,
    tp_significance_0612=FIG8_TP_SIGNIFICANCE,
    out_dir=FIG8_DEFAULT_OUT_DIR,
):
    """New Fig.7: one shared observation plus six forecast panels (a–g)."""
    os.makedirs(out_dir, exist_ok=True)
    datasets, masks = [], []
    for path, sig_path, init in zip(
        (cache_0608, cache_0612),
        (tp_significance_0608, tp_significance_0612),
        ("2023-06-08", "2023-06-12"),
    ):
        with xr.open_dataset(path) as source:
            ds = source.load()
        _fig8_require_cache_vars(ds, ["tp_obs_hov", "z500_obs_hov", "tp_hot_hov", "z500_hot_hov", "tp_cold_hov", "z500_cold_hov", "tp_diff_hov", "z500_diff_hov"])
        coord = "time" if "time" in ds.coords else "date"
        _validate_identity(ds, init, "2023-06-14", "2023-06-24", coord)
        if ds.attrs.get("hovmoller_lat_band") != "(35.0, 44.0)":
            raise ValueError("Expected 35–44N Hovmoller latitude band")
        sig = _fig8_open_significance(sig_path, "TP")
        _validate_pair(ds, sig, init)
        _fig8_validate_significance(ds, sig, None, coord)
        datasets.append(ds)
        masks.append(sig.significant_local_p005.load())
        tp_count = int(masks[-1].astype(bool).sum().item())
        print(f"[Fig.7 TP local significance] init={init}: TP={tp_count}; two-sided Welch p < 0.05")
        sig.close()
    left, right = datasets
    for key in ("tp_obs_hov", "z500_obs_hov"):
        if not left[key].equals(right[key]):
            raise ValueError("The two caches must share identical observations")
    for key in ("tp_abs_levels", "tp_diff_levels"):
        if key not in left or key not in right or not np.array_equal(left[key], right[key]):
            raise ValueError("Shared color levels missing or inconsistent; regenerate both caches")
    for coord in ("s2s_lon",):
        if not np.array_equal(left[coord], right[coord]):
            raise ValueError("The forecast longitude coordinates differ")
    dates = left["time" if "time" in left.coords else "date"].values
    abs_levels, diff_levels, norm = _fig8_levels_from_cache_or_fallback(left)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 9.4))
    # Four equal GridSpec rows. Half-column offsets center the single top panel.
    gs = fig.add_gridspec(4, 4, left=0.145, right=0.985, top=0.965, bottom=0.14, wspace=0.22, hspace=0.30)
    obs_ax = fig.add_subplot(gs[0, 1:3])
    cf_abs = _fig8_plot_abs(obs_ax, left.tp_obs_hov, left.z500_obs_hov, dates, abs_levels, "(a)", "Observed")
    forecast_axes = []
    labels = iter("bcdefg")
    for r, group in enumerate(("hot", "cold", "diff"), start=1):
        for c, ds in enumerate(datasets):
            ax = fig.add_subplot(gs[r, c*2:c*2+2], sharex=obs_ax, sharey=obs_ax)
            forecast_axes.append(ax)
            title = ("June 8" if c == 0 else "June 12") + " · " + {"hot": "Hot17", "cold": "Cold17", "diff": "Hot17 − Cold17"}[group]
            label = f"({next(labels)})"
            tp, z = ds[f"tp_{group}_hov"], ds[f"z500_{group}_hov"]
            if group == "diff":
                cf_diff = _fig8_plot_diff(ax, tp, z, dates, diff_levels, norm, label, title, True, c == 0)
                _fig8_overlay_tp_significance(ax, masks[c], dates)
            else:
                _fig8_plot_abs(ax, tp, z, dates, abs_levels, label, title, False, c == 0)
    fig.canvas.draw()
    # Verify all seven plot boxes have equal geometry before writing.
    boxes = np.array([[ax.get_position().width, ax.get_position().height] for ax in [obs_ax] + forecast_axes])
    if not np.allclose(boxes, boxes[0], rtol=0, atol=1e-9):
        raise ValueError("Seven-panel axes geometry differs")
    for cf, rect, ticks, title in (
        (cf_abs, [0.17, 0.055, 0.28, 0.011], FIG8_ABS_TICKS, "Precipitation (mm day$^{-1}$)"),
        (cf_diff, [0.60, 0.055, 0.30, 0.011], FIG8_DIFF_TICKS, "Hot17 − Cold17 precipitation (mm day$^{-1}$)"),
    ):
        bar = fig.colorbar(cf, cax=fig.add_axes(rect), orientation="horizontal")
        bar.set_ticks(ticks)
        bar.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
        bar.set_label(title, fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    stem = os.path.join(out_dir, "fig7_hovmoller_total_init2023-06-08_2023-06-12")
    _save_visual_figure(fig, stem)
    plt.close(fig)
    return stem + ".png", stem + ".pdf"

# Historical entry now invokes the final seven-panel Hovmoller layout.
plot_fig8_from_cache = plot_fig7_hovmoller_from_cache
