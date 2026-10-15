"""Plot new four-init Fig.6, June 18–20 means with fixed Total Tmax groups."""

import os
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

FIG6_DEFAULT_CACHE = "/data1/huangy/fig6/NC/v12/paper_figures/cache/fig6_interruption_0618_0620_totalgroup_fourinit_plot_cache.nc"
FIG6_DEFAULT_OUT_DIR = "/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in"
FIG6_SIGNIFICANCE_TP = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig6_interruption_0618_0620_totalgroup_fourinit_significance_tp.nc"
FIG6_SIGNIFICANCE_Z500 = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig6_interruption_0618_0620_totalgroup_fourinit_significance_z500.nc"
FIG6_MAP_EXTENT = (90.0, 135.0, 25.0, 55.0)
FIG6_NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
FIG6_ABS_TICKS = [0, 3, 6, 9, 12, 15]
FIG6_DIFF_TICKS = [-4.5, -3.0, -1.5, 0, 1.5, 3.0, 4.5]
FIG_WIDTH_IN = 6.5
FIG6_HEIGHT_IN = 6.5
FONT_PANEL_LETTER = 10.5
FONT_COLUMN_HEADER = 10.0
FONT_ROW_HEADER = 9.5
FONT_PANEL_TITLE = 9.5
FONT_TICK = 8.5
FONT_COLORBAR_TICK = 8.5
FONT_COLORBAR_LABEL = 9.0
FONT_CONTOUR_LABEL = 8.5
FONT_MAP_TICK = 7.75

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


TP_HATCH_LINEWIDTH = 0.60
CONTOUR_LABEL_SAMPLES = 161


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
                    for fraction in sorted(np.linspace(.01, .99, CONTOUR_LABEL_SAMPLES), key=lambda f: abs(f - .5)):
                        target = fraction * distance[-1]
                        j = min(np.searchsorted(distance, target, side="right") - 1, len(points) - 2)
                        length = distance[j + 1] - distance[j]
                        if length <= 0:
                            continue
                        pixel = points[j] + (points[j + 1] - points[j]) * ((target - distance[j]) / length)
                        if not inner.contains(*pixel):
                            continue
                        previous = len(contour.labelTexts)
                        # Initialise all levels; restrict the nearest-path search to this level.
                        ax.clabel(contour, manual=[], fmt="%.0f",
                                  fontsize=FONT_CONTOUR_LABEL, inline=False)
                        original_indices = contour.labelIndiceList
                        previous_xy = len(getattr(contour, "labelXYs", []))
                        try:
                            contour.labelIndiceList = [index]
                            contour.add_label_near(*pixel, inline=False, transform=False)
                        finally:
                            contour.labelIndiceList = original_indices
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
                            text._contour_level = float(level)
                            occupied.append(bounds.padded(separation))
                            placed = True
                            break
                        # Non-inline labels never cut the contour, including rejected trials.
                        contour.pop_label()
                        if hasattr(contour, "labelXYs"):
                            del contour.labelXYs[previous_xy:]
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


def _fig6_setup_map(ax):
    ax.set_extent(FIG6_MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    ax.add_feature(cfeature.LAKES, linewidth=0.25, edgecolor="0.5", facecolor="none")


def _fig6_box(ax):
    lon1, lon2, lat1, lat2 = FIG6_NCHN_BOX
    lines = ax.plot(
        [lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1],
        color="black", lw=1.35, ls="--", transform=ccrs.PlateCarree(), zorder=30,
    )
    for line in lines:
        line.set_path_effects([pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])


def _fig6_effects(texts):
    for txt in texts:
        txt.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])


def _fig6_label(ax, text):
    ax.text(
        0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left",
        fontsize=FONT_PANEL_LETTER, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.8),
        zorder=20,
    )


def _fig6_format_geo_axes(ax, show_left: bool, show_bottom: bool) -> None:
    ax.gridlines(
        draw_labels=False, linewidth=0.25, linestyle="--", alpha=0.35,
        xlocs=np.arange(90, 136, 5), ylocs=np.arange(25, 56, 5),
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.0,
        xlocs=np.arange(100, 131, 10), ylocs=np.arange(30, 56, 10),
        x_inline=False, y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left
    gl.bottom_labels = show_bottom
    gl.xlabel_style = {"size": FONT_MAP_TICK}
    gl.ylabel_style = {"size": FONT_MAP_TICK}
    gl.xformatter = LongitudeFormatter(degree_symbol="°")
    gl.yformatter = LatitudeFormatter(degree_symbol="°")


def _fig6_abs_levels(arrays: Sequence[xr.DataArray]) -> np.ndarray:
    pool = np.concatenate([a.values.ravel() for a in arrays])
    pmax = min(max(float(np.nanpercentile(pool, 95)), 1.0), 25.0)
    return np.linspace(0, pmax, 13)


def _fig6_diff_levels(da: xr.DataArray) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    vmax = max(float(np.nanpercentile(np.abs(da.values), 98)), 0.5)
    return np.linspace(-vmax, vmax, 13), mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _fig6_levels_from_cache_or_fallback(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, mcolors.TwoSlopeNorm]:
    if "tp_abs_levels" in ds.data_vars:
        abs_levels = ds["tp_abs_levels"].values
    else:
        print("[WARNING] Fig.6 cache missing tp_abs_levels; recomputing fallback levels from cached fields.")
        abs_levels = _fig6_abs_levels([ds.tp_obs_ref, ds.tp_hot, ds.tp_cold])

    if "tp_diff_levels" in ds.data_vars:
        diff_levels = ds["tp_diff_levels"].values
        diff_norm = mcolors.TwoSlopeNorm(
            vmin=float(np.nanmin(diff_levels)),
            vcenter=0.0,
            vmax=float(np.nanmax(diff_levels)),
        )
    else:
        print("[WARNING] Fig.6 cache missing tp_diff_levels; recomputing fallback levels from cached fields.")
        diff_levels, diff_norm = _fig6_diff_levels(ds.tp_diff)
    return abs_levels, diff_levels, diff_norm


def _fig6_require_cache_vars(ds: xr.Dataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Fig.6 cache is missing required data_vars: {missing}")


def _fig6_check_z500_metadata(ds: xr.Dataset) -> None:
    z500_type = str(ds.attrs.get("z500_type", "")).lower()
    if z500_type != "absolute":
        raise ValueError(f"Fig.6 cache must declare z500_type='absolute'; got {ds.attrs.get('z500_type')!r}")


def _fig6_contour_abs(ax, z):
    lon_dim, lat_dim = _fig6_lon_lat_dims(z)
    levels = np.arange(5520, 5921, 40)
    levels = levels[(levels >= float(z.min())) & (levels <= float(z.max()))]
    if levels.size:
        cs = ax.contour(z[lon_dim], z[lat_dim], z, levels=levels, colors="0.12", linewidths=1.1, transform=ccrs.PlateCarree())
        _queue_contour_labels(ax, cs)
    if float(z.min()) <= 5880 <= float(z.max()):
        ax.contour(z[lon_dim], z[lat_dim], z, levels=[5880], colors="purple", linewidths=1.5, transform=ccrs.PlateCarree())


def _fig6_contour_diff(ax, z):
    lon_dim, lat_dim = _fig6_lon_lat_dims(z)
    base = np.array([10, 20, 40, 60, 80], dtype=float)
    base = base[base <= float(np.nanmax(np.abs(z.values)))]
    if base.size:
        csn = ax.contour(z[lon_dim], z[lat_dim], z, levels=-base[::-1], colors="royalblue", linestyles="--", linewidths=1.05, transform=ccrs.PlateCarree())
        csp = ax.contour(z[lon_dim], z[lat_dim], z, levels=base, colors="firebrick", linewidths=1.05, transform=ccrs.PlateCarree())
        labs = base[base >= 20]
        if getattr(ax, "_require_positive_z500_label", False):
            _queue_contour_labels(ax, csp, np.r_[labs, base[base < 20]])
            if labs.size:
                _queue_contour_labels(ax, csn, -labs[::-1])
        elif labs.size:
            _queue_contour_labels(ax, csn, -labs[::-1])
            _queue_contour_labels(ax, csp, labs)


def _fig6_plot_abs(ax, tp, z, levels, title=""):
    tp_lon_dim, tp_lat_dim = _fig6_lon_lat_dims(tp)
    _fig6_setup_map(ax)
    cf = ax.contourf(tp[tp_lon_dim], tp[tp_lat_dim], tp, levels=levels, cmap="Blues", extend="max", alpha=0.92, transform=ccrs.PlateCarree())
    _fig6_contour_abs(ax, z)
    _fig6_box(ax)
    if title:
        ax.set_title(title, fontsize=FONT_PANEL_TITLE)
    return cf


def _fig6_plot_diff(ax, tp, z, levels, norm, title=""):
    tp_lon_dim, tp_lat_dim = _fig6_lon_lat_dims(tp)
    _fig6_setup_map(ax)
    cf = ax.contourf(tp[tp_lon_dim], tp[tp_lat_dim], tp, levels=levels, cmap="RdBu_r", norm=norm, extend="both", alpha=0.92, transform=ccrs.PlateCarree())
    _fig6_contour_diff(ax, z)
    _fig6_box(ax)
    if title:
        ax.set_title(title, fontsize=FONT_PANEL_TITLE)
    return cf


def _fig6_lon_lat_dims(da: xr.DataArray) -> Tuple[str, str]:
    lon_dims = [dim for dim in da.dims if dim.endswith("_lon") or dim == "lon"]
    lat_dims = [dim for dim in da.dims if dim.endswith("_lat") or dim == "lat"]
    if not lon_dims or not lat_dims:
        raise ValueError(f"Cannot infer lon/lat dims for {da.name}: dims={da.dims}")
    return lon_dims[0], lat_dims[0]


def _fig6_open_significance(path: str, variable: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fig.6 {variable} significance file not found: {path}. "
            "Run export_fig6_spatial_significance() in the science/source cell first."
        )
    ds = xr.open_dataset(path)
    _fig6_require_cache_vars(
        ds,
        ["p_fdr_bh", "significant_fdr_005", "significant_local_p005", "hot_minus_cold"],
    )
    if str(ds.attrs.get("difference_direction", "")) != "Hot17 minus Cold17":
        raise ValueError(f"Fig.6 {variable} significance difference direction is invalid")
    if str(ds.attrs.get("grouping_stage", "")) != "Total":
        raise ValueError(f"Fig.6 {variable} significance must use Total grouping")
    finite = ds.p_fdr_bh.values[np.isfinite(ds.p_fdr_bh.values)]
    if np.any((finite < 0) | (finite > 1)):
        raise ValueError(f"Fig.6 {variable} significance p_fdr_bh contains values outside [0, 1]")
    return ds


def _fig6_validate_significance(plot_ds: xr.Dataset, tp_sig: xr.Dataset, z_sig: xr.Dataset = None) -> None:
    if str(plot_ds.attrs.get("sign_convention", "")) != "Hot17_minus_Cold17 = Hot17 - Cold17":
        raise ValueError("Fig.6 plot cache does not declare the required Hot17−Cold17 sign convention")
    for variable, sig in (("TP", tp_sig),):
        if not np.array_equal(plot_ds.init.values.astype(str), sig.init.values.astype(str)):
            raise ValueError(f"Fig.6 {variable} significance initializations do not match plot cache")
        if not np.array_equal(plot_ds.s2s_lat.values, sig.s2s_lat.values):
            raise ValueError(f"Fig.6 {variable} significance latitude does not match native S2S plot grid")
        if not np.array_equal(plot_ds.s2s_lon.values, sig.s2s_lon.values):
            raise ValueError(f"Fig.6 {variable} significance longitude does not match native S2S plot grid")
        if sig.significant_local_p005.shape != plot_ds.tp_diff.shape:
            raise ValueError(f"Fig.6 {variable} significance-mask shape does not match difference panels")
    if not np.allclose(tp_sig.hot_minus_cold.values, plot_ds.tp_diff.values, rtol=1e-7, atol=1e-6, equal_nan=True):
        raise ValueError("Fig.6 TP significance difference does not match the plotted Hot17−Cold17 field")


def _fig6_overlay_tp_significance(ax, mask: xr.DataArray) -> None:
    sig = mask.astype(bool)
    if not bool(sig.any().item()):
        return
    with plt.rc_context({"hatch.color": "black", "hatch.linewidth": TP_HATCH_LINEWIDTH}):
        ax.contourf(
            sig.s2s_lon, sig.s2s_lat, sig.astype(float), levels=[0.5, 1.5],
            colors="none", hatches=["..."], transform=ccrs.PlateCarree(), zorder=12,
        )


def plot_fig6_from_cache(
    cache_path: str = FIG6_DEFAULT_CACHE,
    out_dir: str = FIG6_DEFAULT_OUT_DIR,
    tp_significance_path: str = FIG6_SIGNIFICANCE_TP,
    z500_significance_path: str = FIG6_SIGNIFICANCE_Z500,
):
    os.makedirs(out_dir, exist_ok=True)
    with xr.open_dataset(cache_path) as source:
        ds = source.load()
    _fig6_require_cache_vars(ds, ["tp_hot", "z500_abs_hot", "tp_cold", "z500_abs_cold", "tp_diff", "z500_abs_diff", "tp_obs_ref", "z500_abs_obs_ref"])
    _fig6_check_z500_metadata(ds)
    if ds.attrs.get("grouping_stage") != "Total" or ds.attrs.get("analysis_window") != "2023-06-18 to 2023-06-20":
        raise ValueError("Fig.6 requires Total grouping and BJT June 18–20")
    inits = [str(v)[:10] for v in ds.init.values.astype("datetime64[D]")]
    if inits != ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]:
        raise ValueError("Fig.6 initialization order differs")
    tp_sig = _fig6_open_significance(tp_significance_path, "TP")
    _fig6_validate_significance(ds, tp_sig)
    _fig6_validate_products(ds, tp_sig)
    print("[Fig6 local significance] local two-sided Welch t-test p < 0.05")
    for r, init in enumerate(ds.init.values):
        tp_count = int(tp_sig.significant_local_p005.isel(init=r).astype(bool).sum().item())
        init_label = (
            np.datetime_as_string(init, unit="D")
            if np.issubdtype(np.asarray(init).dtype, np.datetime64)
            else str(init)[:10]
        )
        print(f"{init_label}: TP={tp_count}")
    abs_levels, diff_levels, diff_norm = _fig6_levels_from_cache_or_fallback(ds)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG6_HEIGHT_IN))
    gs = fig.add_gridspec(
        4, 3, left=0.145, right=0.985, top=0.965, bottom=0.27,
        hspace=0.11, wspace=0.10,
    )
    axes = np.empty((4, 3), dtype=object)
    for r in range(4):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
    axes[3, 2]._require_positive_z500_label = True
    ax_ref = fig.add_axes([0.145, 0.035, 0.245, 0.155], projection=ccrs.PlateCarree())
    cax_abs = fig.add_axes([0.43, 0.160, 0.52, 0.012])
    cax_diff = fig.add_axes([0.43, 0.085, 0.52, 0.012])
    cf_abs = cf_diff = None
    titles = ["Hot17", "Cold17", "Hot17 − Cold17"]
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=FONT_COLUMN_HEADER, fontweight="bold", pad=3)
    labels = list("abcdefghijklm")
    for r, init in enumerate(ds.init.values):
        cf_abs = _fig6_plot_abs(axes[r, 0], ds.tp_hot.sel(init=init), ds.z500_abs_hot.sel(init=init), abs_levels)
        _fig6_plot_abs(axes[r, 1], ds.tp_cold.sel(init=init), ds.z500_abs_cold.sel(init=init), abs_levels)
        cf_diff = _fig6_plot_diff(axes[r, 2], ds.tp_diff.sel(init=init), ds.z500_abs_diff.sel(init=init), diff_levels, diff_norm)
        _fig6_overlay_tp_significance(axes[r, 2], tp_sig.significant_local_p005.isel(init=r))
        for c in range(3):
            _fig6_label(axes[r, c], f"({labels[r * 3 + c]})")
            _fig6_format_geo_axes(axes[r, c], show_left=(c == 0), show_bottom=(r == 3))
    _fig6_plot_abs(ax_ref, ds.tp_obs_ref, ds.z500_abs_obs_ref, abs_levels, "Observed")
    _fig6_label(ax_ref, "(m)")
    _fig6_format_geo_axes(ax_ref, show_left=True, show_bottom=True)
    fig.canvas.draw()
    for r, init in enumerate(ds.init.values):
        init_label = (
            np.datetime_as_string(init, unit="D")
            if np.issubdtype(np.asarray(init).dtype, np.datetime64)
            else str(init)[:10]
        )
        short_init = init_label[8:10].lstrip("0") + " Jun"
        pos = axes[r, 0].get_position()
        fig.text(
            0.085, 0.5 * (pos.y0 + pos.y1), short_init,
            rotation=90, ha="center", va="center",
            fontsize=FONT_ROW_HEADER, fontweight="bold",
        )
    cbar_abs = fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_ticks(_fig6_color_ticks(abs_levels))
    cbar_abs.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_abs.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_abs.set_label("Precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_ticks(_fig6_color_ticks(diff_levels))
    cbar_diff.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_diff.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_diff.set_label("Hot17 − Cold17 precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    stem = os.path.join(out_dir, "fig6_interruption_0618_0620_totalgroup_fourinit")
    _save_visual_figure(fig, stem)
    plt.close(fig)
    tp_sig.close()
    return stem + ".png", stem + ".pdf"




def _fig6_color_ticks(levels):
    """Derive ticks from the stored common scale, not old 11-day limits."""
    low, high = float(np.min(levels)), float(np.max(levels))
    ticks = mticker.MaxNLocator(nbins=7, steps=[1, 2, 3, 5, 10]).tick_values(low, high)
    return ticks[(ticks >= low - 1e-9) & (ticks <= high + 1e-9)]


def _fig6_validate_products(cache, sig):
    import json

    inits = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]
    for product in (cache, sig):
        required = {
            "analysis_window": "2023-06-18 to 2023-06-20",
            "grouping_window": "2023-06-14 to 2023-06-24",
            "grouping_stage": "Total", "z500_type": "absolute",
            "z500_units": "gpm", "tp_units": "mm day^-1",
            "target_dates_bjt": "2023-06-18,2023-06-19,2023-06-20",
        }
        for name, expected in required.items():
            if product.attrs.get(name) != expected:
                raise ValueError(f"New Fig.6 requires {name}={expected!r}")
        if int(product.attrs.get("mean_day_count", -1)) != 3:
            raise ValueError("New Fig.6 requires exactly three BJT days")
        actual = np.asarray(product.init.values).astype("datetime64[D]").astype(str).tolist()
        if actual != inits:
            raise ValueError("New Fig.6 requires all four initializations in approved order")
        groups = json.loads(product.attrs.get("group_members_by_init", "{}"))
        if set(groups) != set(inits):
            raise ValueError("Missing Total membership metadata")
        for init in inits:
            hot, cold = groups[init]["Hot17"], groups[init]["Cold17"]
            if (len(hot) != 17 or len(cold) != 17 or len(set(hot)) != 17
                    or len(set(cold)) != 17 or set(hot) & set(cold)
                    or not set(hot + cold).issubset(set(range(51)))):
                raise ValueError(f"{init}: invalid 17/17 Total membership")
    if json.loads(cache.attrs["group_members_by_init"]) != json.loads(sig.attrs["group_members_by_init"]):
        raise ValueError("Cache and significance Total memberships differ")
    if sig.attrs.get("units") != "mm day^-1":
        raise ValueError("TP significance units differ")
    if str(cache.attrs.get("map_extent")) != str(FIG6_MAP_EXTENT):
        raise ValueError("Map extent differs from the approved domain")
    if str(cache.attrs.get("nchn_box")) != str(FIG6_NCHN_BOX):
        raise ValueError("NCHN box differs from the approved definition")
    for grid in ("s2s", "mswep", "era5"):
        for direction in ("lat", "lon"):
            coord = np.asarray(cache[f"{grid}_{direction}"].values)
            delta = np.diff(coord)
            if (len(coord) < 2 or not np.isfinite(coord).all()
                    or not (np.all(delta > 0) or np.all(delta < 0))):
                raise ValueError(f"{grid}_{direction}: invalid native coordinates")
    for variable, units in (("tp", "mm day^-1"), ("z500_abs", "gpm")):
        hot, cold, difference = (cache[f"{variable}_{g}"] for g in ("hot", "cold", "diff"))
        for field in (hot, cold, difference):
            if field.dims != ("init", "s2s_lat", "s2s_lon") or field.attrs.get("units") != units:
                raise ValueError(f"{variable}: unexpected dimensions or units")
        if not np.allclose(hot - cold, difference, rtol=1e-7, atol=1e-6, equal_nan=True):
            raise ValueError(f"{variable}: cache difference is not Hot17 minus Cold17")
    for name in ("tp_abs_levels", "tp_diff_levels"):
        if name not in cache:
            raise ValueError(f"Missing common four-init color scale {name}")
        values = cache[name].values
        if len(values) < 2 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
            raise ValueError(f"Invalid color scale {name}")
    for name, field in (("hot_mean", cache.tp_hot), ("cold_mean", cache.tp_cold)):
        if name not in sig or not np.allclose(sig[name], field, rtol=1e-7, atol=1e-6, equal_nan=True):
            raise ValueError(f"TP significance {name} differs from cache")
    for name in ("p_value", "fdr_domain_mask", "n_hot", "n_cold", "significant_local_p005"):
        if name not in sig or sig[name].dims != cache.tp_diff.dims or sig[name].shape != cache.tp_diff.shape:
            raise ValueError(f"TP significance {name} has wrong dimensions/shape")
    p = sig.p_value.values
    if np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("TP p-values outside [0,1]")
    domain = sig.fdr_domain_mask.values
    mask = sig.significant_local_p005.values
    if not np.isin(domain, [0, 1]).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("TP masks must contain only 0/1")
    lat, lon = np.meshgrid(cache.s2s_lat.values, cache.s2s_lon.values, indexing="ij")
    expected_domain = (lat >= 25) & (lat <= 55) & (lon >= 90) & (lon <= 135)
    if not np.array_equal(domain.astype(bool), np.broadcast_to(expected_domain, domain.shape)):
        raise ValueError("TP significance spatial domain differs")
    if not np.array_equal(mask.astype(bool), domain.astype(bool) & np.isfinite(p) & (p < .05)):
        raise ValueError("TP local mask does not match two-sided local p<0.05")
    for name in ("n_hot", "n_cold"):
        if not np.all(sig[name].values[domain.astype(bool)] == 17):
            raise ValueError("Expected 17 valid members per group in the display domain")
