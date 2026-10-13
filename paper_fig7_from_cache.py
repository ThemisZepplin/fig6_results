"""Final Fig.6/Fig.S6 daily maps; historical filename retained for compatibility."""

import os
from datetime import datetime
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

FIG7_DEFAULT_CACHE = "/data1/huangy/fig6/NC/v12/paper_figures/cache/fig7_daily_synoptic_plot_cache.nc"
FIG7_DEFAULT_OUT_DIR = "/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in"
FIG7_SIGNIFICANCE_TP = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig7_spatial_significance_tp.nc"
FIG7_SIGNIFICANCE_Z500 = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig7_spatial_significance_z500.nc"
FIG7_MAP_EXTENT = (90.0, 135.0, 25.0, 55.0)
FIG7_NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
FIG7_ABS_TICKS = [0, 5, 10, 15, 20]
FIG7_DIFF_TICKS = [-15, -10, -5, 0, 5, 10, 15]
FIG_WIDTH_IN = 6.5
FIG7_HEIGHT_IN = 7.1
FIG7_SUBPLOT_LAYOUT = dict(
    left=0.105, right=0.99, top=0.965, bottom=0.145,
    wspace=0.035, hspace=0.075,
)
FIG7_CAX_ABS = [0.17, 0.085, 0.28, 0.011]
FIG7_CAX_DIFF = [0.60, 0.085, 0.30, 0.011]
FONT_PANEL_LETTER = 10.5
FONT_COLUMN_HEADER = 10.0
FONT_ROW_HEADER = 9.5
FONT_AXIS_LABEL = 9.5
FONT_TICK = 8.5
FONT_COLORBAR_TICK = 8.5
FONT_COLORBAR_LABEL = 9.0
FONT_CONTOUR_LABEL = 8.5

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42



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

def _fig7_setup_map(ax):
    ax.set_extent(FIG7_MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    ax.add_feature(cfeature.LAKES, linewidth=0.25, edgecolor="0.5", facecolor="none")


def _fig7_box(ax):
    lon1, lon2, lat1, lat2 = FIG7_NCHN_BOX
    lines = ax.plot(
        [lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1],
        color="black", lw=1.35, ls="--", transform=ccrs.PlateCarree(), zorder=30,
    )
    for line in lines:
        line.set_path_effects([pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])


def _fig7_label(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=FONT_PANEL_LETTER, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.8), zorder=20)


def _fig7_effects(texts):
    for txt in texts:
        txt.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])


def _fig7_abs_levels(arrays: Sequence[xr.DataArray]) -> np.ndarray:
    pool = np.concatenate([a.values.ravel() for a in arrays])
    pmax = min(max(float(np.nanpercentile(pool, 95)), 1.0), 25.0)
    return np.linspace(0, pmax, 13)


def _fig7_diff_levels(da: xr.DataArray) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    vmax = max(float(np.nanpercentile(np.abs(da.values), 98)), 0.5)
    return np.linspace(-vmax, vmax, 13), mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _fig7_require_cache_vars(ds: xr.Dataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Fig.7 cache is missing required data_vars: {missing}")


def _fig7_check_z500_metadata(ds: xr.Dataset) -> None:
    z500_type = str(ds.attrs.get("z500_type", "")).lower()
    if "absolute" not in z500_type:
        raise ValueError(
            "Fig.7 from-cache contours assume absolute Z500 for obs/hot/cold; "
            f"got z500_type={ds.attrs.get('z500_type')!r}"
        )


def _fig7_levels_from_cache_or_fallback(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, mcolors.TwoSlopeNorm]:
    if "tp_abs_levels" in ds.data_vars:
        abs_levels = ds["tp_abs_levels"].values
    else:
        print("[WARNING] Fig.7 cache missing tp_abs_levels; recomputing fallback levels from cached fields.")
        abs_levels = _fig7_abs_levels([ds.tp_obs, ds.tp_hot, ds.tp_cold])

    if "tp_diff_levels" in ds.data_vars:
        diff_levels = ds["tp_diff_levels"].values
        diff_norm = mcolors.TwoSlopeNorm(
            vmin=float(np.nanmin(diff_levels)),
            vcenter=0.0,
            vmax=float(np.nanmax(diff_levels)),
        )
    else:
        print("[WARNING] Fig.7 cache missing tp_diff_levels; recomputing fallback levels from cached fields.")
        diff_levels, diff_norm = _fig7_diff_levels(ds.tp_diff)
    return abs_levels, diff_levels, diff_norm


def _fig7_contour_abs(ax, z):
    lon_dim, lat_dim = _fig7_lon_lat_dims(z)
    levels = np.arange(5520, 5921, 40)
    levels = levels[(levels >= float(z.min())) & (levels <= float(z.max()))]
    if levels.size:
        cs = ax.contour(z[lon_dim], z[lat_dim], z, levels=levels, colors="0.12", linewidths=1.1, transform=ccrs.PlateCarree())
        _fig7_effects(ax.clabel(cs, fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))
    if float(z.min()) <= 5880 <= float(z.max()):
        ax.contour(z[lon_dim], z[lat_dim], z, levels=[5880], colors="purple", linewidths=1.5, transform=ccrs.PlateCarree())


def _fig7_contour_diff(ax, z):
    lon_dim, lat_dim = _fig7_lon_lat_dims(z)
    base = np.array([10, 20, 40, 60, 80], dtype=float)
    base = base[base <= float(np.nanmax(np.abs(z.values)))]
    if base.size:
        neg = -base[::-1]
        pos = base
        csn = ax.contour(z[lon_dim], z[lat_dim], z, levels=neg, colors="royalblue", linestyles="--", linewidths=1.05, transform=ccrs.PlateCarree())
        csp = ax.contour(z[lon_dim], z[lat_dim], z, levels=pos, colors="firebrick", linewidths=1.05, transform=ccrs.PlateCarree())
        labs = base[base >= 20]
        if labs.size:
            _fig7_effects(ax.clabel(csn, levels=-labs[::-1], fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))
            _fig7_effects(ax.clabel(csp, levels=labs, fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))


def _fig7_format_geo_axes(ax, show_left: bool, show_bottom: bool) -> None:
    ax.gridlines(
        draw_labels=False, linewidth=0.25, linestyle="--", alpha=0.35,
        xlocs=np.arange(90, 136, 5), ylocs=np.arange(25, 56, 5),
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.0,
        xlocs=np.arange(90, 136, 10), ylocs=np.arange(30, 56, 10),
        x_inline=False, y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left
    gl.bottom_labels = show_bottom
    gl.xlabel_style = {"size": FONT_TICK}
    gl.ylabel_style = {"size": FONT_TICK}
    gl.xformatter = LongitudeFormatter(degree_symbol="°")
    gl.yformatter = LatitudeFormatter(degree_symbol="°")


def _fig7_lon_lat_dims(da: xr.DataArray) -> Tuple[str, str]:
    lon_dims = [dim for dim in da.dims if dim.endswith("_lon") or dim == "lon"]
    lat_dims = [dim for dim in da.dims if dim.endswith("_lat") or dim == "lat"]
    if not lon_dims or not lat_dims:
        raise ValueError(f"Cannot infer lon/lat dims for {da.name}: dims={da.dims}")
    return lon_dims[0], lat_dims[0]


def _fig7_open_significance(path: str, variable: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fig.7 {variable} significance file not found: {path}. "
            "Run export_fig7_spatial_significance() in the science/source cell first."
        )
    ds = xr.open_dataset(path)
    _fig7_require_cache_vars(
        ds,
        ["p_fdr_bh", "significant_fdr_005", "significant_local_p005", "hot_minus_cold"],
    )
    if str(ds.attrs.get("difference_direction", "")) != "Hot17 minus Cold17":
        raise ValueError(f"Fig.7 {variable} significance difference direction is invalid")
    if str(ds.attrs.get("grouping_stage", "")) != "Total":
        raise ValueError(f"Fig.7 {variable} significance must use Total grouping")
    finite = ds.p_fdr_bh.values[np.isfinite(ds.p_fdr_bh.values)]
    if np.any((finite < 0) | (finite > 1)):
        raise ValueError(f"Fig.7 {variable} significance p_fdr_bh contains values outside [0, 1]")
    return ds


def _fig7_validate_significance(plot_ds: xr.Dataset, tp_sig: xr.Dataset, z_sig: xr.Dataset = None) -> None:
    if str(plot_ds.attrs.get("sign_convention", "")) != "Hot17_minus_Cold17 = Hot17 - Cold17":
        raise ValueError("Fig.7 plot cache does not declare the required Hot17−Cold17 sign convention")
    for variable, sig in (("TP", tp_sig),):
        if not np.array_equal(plot_ds.date.values, sig.date.values):
            raise ValueError(f"Fig.7 {variable} significance dates do not match plot cache")
        if not np.array_equal(plot_ds.s2s_lat.values, sig.s2s_lat.values):
            raise ValueError(f"Fig.7 {variable} significance latitude does not match native S2S plot grid")
        if not np.array_equal(plot_ds.s2s_lon.values, sig.s2s_lon.values):
            raise ValueError(f"Fig.7 {variable} significance longitude does not match native S2S plot grid")
        if sig.significant_local_p005.shape != plot_ds.tp_diff.shape:
            raise ValueError(f"Fig.7 {variable} significance-mask shape does not match difference panel")
    if not np.allclose(tp_sig.hot_minus_cold.values, plot_ds.tp_diff.values, rtol=1e-7, atol=1e-6, equal_nan=True):
        raise ValueError("Fig.7 TP significance difference does not match the plotted Hot17−Cold17 field")


def _fig7_overlay_tp_significance(ax, mask: xr.DataArray) -> None:
    sig = mask.astype(bool)
    if not bool(sig.any().item()):
        return
    with plt.rc_context({"hatch.color": "black", "hatch.linewidth": 0.45}):
        ax.contourf(
            sig.s2s_lon, sig.s2s_lat, sig.astype(float), levels=[0.5, 1.5],
            colors="none", hatches=["..."], transform=ccrs.PlateCarree(), zorder=12,
        )


def _fig7_apply_daily_layout(fig, axs, rows):
    """Shared Fig.6/S6 layout, measured after fixed-aspect axes settle."""
    fig.subplots_adjust(**FIG7_SUBPLOT_LAYOUT)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Keep a 3-point gap from latitude labels and from the canvas edge.
    gap = 3.0 * fig.dpi / 72.0
    labels = []
    for r, (row_title, _, _) in enumerate(rows):
        pos = axs[r, 0].get_position()
        label = fig.text(
            0, 0.5 * (pos.y0 + pos.y1), row_title.replace("Total ", ""),
            rotation=90, ha="center", va="center",
            fontsize=FONT_ROW_HEADER, fontweight="bold",
        )
        labels.append(label)

    def required_left_padding():
        return max(
            0.0,
            max(label.get_window_extent(renderer).width + 2 * gap
                - axs[r, 0].get_tightbbox(renderer).x0
                for r, label in enumerate(labels)),
        )

    extra = required_left_padding() / fig.bbox.width
    if extra:
        # Only accommodate actual tick/font collisions, without stretching maps.
        if extra > 0.015:
            raise ValueError("Daily row labels need more than a small margin adjustment")
        fig.subplots_adjust(left=FIG7_SUBPLOT_LAYOUT["left"] + extra)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    for r, label in enumerate(labels):
        ax = axs[r, 0]
        pos = ax.get_position()
        width = label.get_window_extent(renderer).width
        x = ax.get_tightbbox(renderer).x0 - gap - width / 2
        label.set_position((fig.transFigure.inverted().transform((x, 0))[0],
                            0.5 * (pos.y0 + pos.y1)))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for r, label in enumerate(labels):
        bounds = label.get_window_extent(renderer)
        if (bounds.x0 < gap - 0.5 or bounds.y0 < 0 or bounds.y1 > fig.bbox.height
                or bounds.x1 > axs[r, 0].get_tightbbox(renderer).x0 - gap + 0.5):
            raise ValueError("Daily row label clips the canvas or overlaps latitude labels")
    return labels


def plot_fig6_from_cache(
    cache_path: str = FIG7_DEFAULT_CACHE,
    out_dir: str = FIG7_DEFAULT_OUT_DIR,
    tp_significance_path: str = FIG7_SIGNIFICANCE_TP,
    z500_significance_path: str = FIG7_SIGNIFICANCE_Z500,
    initialization: str = "2023-06-12",
):
    if initialization not in ("2023-06-08", "2023-06-12"):
        raise ValueError("Daily final figures support June 8 and June 12 only")
    os.makedirs(out_dir, exist_ok=True)
    with xr.open_dataset(cache_path) as source:
        ds = source.load()
    _fig7_require_cache_vars(ds, ["tp_obs", "z500_obs", "tp_hot", "z500_hot", "tp_cold", "z500_cold", "tp_diff", "z500_diff"])
    _fig7_check_z500_metadata(ds)
    _validate_identity(ds, initialization, "2023-06-18", "2023-06-20", "date")
    tp_sig = _fig7_open_significance(tp_significance_path, "TP")
    _validate_pair(ds, tp_sig, initialization)
    _fig7_validate_significance(ds, tp_sig)
    figure_label = "Fig.S6" if initialization == "2023-06-08" else "Fig.6"
    print(f"[{figure_label} TP local significance] two-sided Welch p < 0.05; init={initialization}")
    for c, date in enumerate(ds.date.values):
        tp_count = int(tp_sig.significant_local_p005.isel(date=c).astype(bool).sum().item())
        print(f"{np.datetime_as_string(date, unit='D')}: TP={tp_count}")
    for key in ("tp_abs_levels", "tp_diff_levels"):
        if key not in ds:
            raise ValueError("Shared daily color levels missing; regenerate both caches")
    abs_levels, diff_levels, diff_norm = _fig7_levels_from_cache_or_fallback(ds)
    fig, axs = plt.subplots(4, 3, figsize=(FIG_WIDTH_IN, FIG7_HEIGHT_IN), subplot_kw={"projection": ccrs.PlateCarree()})
    rows = [("Observed", "tp_obs", "z500_obs"), ("Total Hot17", "tp_hot", "z500_hot"), ("Total Cold17", "tp_cold", "z500_cold"), ("Total Hot17 − Cold17", "tp_diff", "z500_diff")]
    labels = list("abcdefghijkl")
    cf_abs = cf_diff = None
    for r, (row_title, tp_name, z_name) in enumerate(rows):
        for c, date in enumerate(ds.date.values):
            ax = axs[r, c]
            _fig7_setup_map(ax)
            tp = ds[tp_name].isel(date=c)
            z = ds[z_name].isel(date=c)
            tp_lon_dim, tp_lat_dim = _fig7_lon_lat_dims(tp)
            if r < 3:
                cf_abs = ax.contourf(tp[tp_lon_dim], tp[tp_lat_dim], tp, levels=abs_levels, cmap="Blues", extend="max", alpha=0.92, transform=ccrs.PlateCarree())
                _fig7_contour_abs(ax, z)
            else:
                cf_diff = ax.contourf(tp[tp_lon_dim], tp[tp_lat_dim], tp, levels=diff_levels, norm=diff_norm, cmap="RdBu_r", extend="both", alpha=0.92, transform=ccrs.PlateCarree())
                _fig7_contour_diff(ax, z)
                _fig7_overlay_tp_significance(ax, tp_sig.significant_local_p005.isel(date=c))
            _fig7_box(ax)
            _fig7_label(ax, f"({labels[r*3+c]})")
            _fig7_format_geo_axes(ax, show_left=(c == 0), show_bottom=(r == 3))
            if r == 0:
                ax.set_title(str(np.datetime_as_string(date, unit="D")), fontsize=FONT_COLUMN_HEADER, pad=3)
    _fig7_apply_daily_layout(fig, axs, rows)
    cax_abs = fig.add_axes(FIG7_CAX_ABS)
    cbar_abs = fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_ticks(FIG7_ABS_TICKS)
    cbar_abs.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_abs.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_abs.set_label("Precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    cax_diff = fig.add_axes(FIG7_CAX_DIFF)
    cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_ticks(FIG7_DIFF_TICKS)
    cbar_diff.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_diff.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_diff.set_label("Hot17 − Cold17 precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    stem = os.path.join(out_dir, ("fig6_daily_synoptic_init2023-06-12" if initialization == "2023-06-12" else "figS6_daily_synoptic_init2023-06-08"))
    fig.savefig(stem + ".png", dpi=600)
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    tp_sig.close()
    return stem + ".png", stem + ".pdf"


def plot_figS6_from_cache(
    cache_path=FIG7_DEFAULT_CACHE.replace("fig7_daily_synoptic_plot_cache", "fig7_daily_synoptic_init2023-06-08_plot_cache"),
    out_dir=FIG7_DEFAULT_OUT_DIR,
    tp_significance_path=FIG7_SIGNIFICANCE_TP.replace("fig7_spatial", "fig7_init2023-06-08_spatial"),
):
    return plot_fig6_from_cache(cache_path, out_dir, tp_significance_path, initialization="2023-06-08")

# Historical filename and entry remain compatible; daily maps now output Fig.6.
plot_fig7_from_cache = plot_fig6_from_cache
