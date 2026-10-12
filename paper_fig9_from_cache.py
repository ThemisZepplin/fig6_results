"""Plot final paper Fig.9 from the historical Fig.9-named plot-ready cache."""

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

FIG9_DEFAULT_CACHE = "/data1/huangy/fig6/NC/v12/paper_figures/cache/fig9_leadtime_tp_z500_plot_cache.nc"
FIG9_DEFAULT_OUT_DIR = "/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in"
FIG9_SIGNIFICANCE_TP = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig9_spatial_significance_tp.nc"
FIG9_SIGNIFICANCE_Z500 = "/data1/huangy/fig6/NC/v12/paper_figures/significance/fig9_spatial_significance_z500.nc"
FIG10_DEFAULT_CACHE = FIG9_DEFAULT_CACHE
FIG10_DEFAULT_OUT_DIR = FIG9_DEFAULT_OUT_DIR
FIG10_SIGNIFICANCE_TP = FIG9_SIGNIFICANCE_TP
FIG10_SIGNIFICANCE_Z500 = FIG9_SIGNIFICANCE_Z500
FIG9_MAP_EXTENT = (90.0, 135.0, 25.0, 55.0)
FIG9_NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
FIG9_ABS_TICKS = [0, 3, 6, 9, 12, 15]
FIG9_DIFF_TICKS = [-4.5, -3.0, -1.5, 0, 1.5, 3.0, 4.5]
FIG_WIDTH_IN = 6.5
FIG10_HEIGHT_IN = 6.5
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


def _fig9_setup_map(ax):
    ax.set_extent(FIG9_MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    ax.add_feature(cfeature.LAKES, linewidth=0.25, edgecolor="0.5", facecolor="none")


def _fig9_box(ax):
    lon1, lon2, lat1, lat2 = FIG9_NCHN_BOX
    lines = ax.plot(
        [lon1, lon2, lon2, lon1, lon1], [lat1, lat1, lat2, lat2, lat1],
        color="black", lw=1.35, ls="--", transform=ccrs.PlateCarree(), zorder=30,
    )
    for line in lines:
        line.set_path_effects([pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])


def _fig9_effects(texts):
    for txt in texts:
        txt.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])


def _fig9_label(ax, text):
    ax.text(
        0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left",
        fontsize=FONT_PANEL_LETTER, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.8),
        zorder=20,
    )


def _fig9_format_geo_axes(ax, show_left: bool, show_bottom: bool) -> None:
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


def _fig9_abs_levels(arrays: Sequence[xr.DataArray]) -> np.ndarray:
    pool = np.concatenate([a.values.ravel() for a in arrays])
    pmax = min(max(float(np.nanpercentile(pool, 95)), 1.0), 25.0)
    return np.linspace(0, pmax, 13)


def _fig9_diff_levels(da: xr.DataArray) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    vmax = max(float(np.nanpercentile(np.abs(da.values), 98)), 0.5)
    return np.linspace(-vmax, vmax, 13), mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _fig9_levels_from_cache_or_fallback(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, mcolors.TwoSlopeNorm]:
    if "tp_abs_levels" in ds.data_vars:
        abs_levels = ds["tp_abs_levels"].values
    else:
        print("[WARNING] Fig.9 cache missing tp_abs_levels; recomputing fallback levels from cached fields.")
        abs_levels = _fig9_abs_levels([ds.tp_obs_ref, ds.tp_hot, ds.tp_cold])

    if "tp_diff_levels" in ds.data_vars:
        diff_levels = ds["tp_diff_levels"].values
        diff_norm = mcolors.TwoSlopeNorm(
            vmin=float(np.nanmin(diff_levels)),
            vcenter=0.0,
            vmax=float(np.nanmax(diff_levels)),
        )
    else:
        print("[WARNING] Fig.9 cache missing tp_diff_levels; recomputing fallback levels from cached fields.")
        diff_levels, diff_norm = _fig9_diff_levels(ds.tp_diff)
    return abs_levels, diff_levels, diff_norm


def _fig9_require_cache_vars(ds: xr.Dataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise KeyError(f"Fig.9 cache is missing required data_vars: {missing}")


def _fig9_check_z500_metadata(ds: xr.Dataset) -> None:
    z500_type = str(ds.attrs.get("z500_type", "")).lower()
    if z500_type != "absolute":
        raise ValueError(f"Fig.9 cache must declare z500_type='absolute'; got {ds.attrs.get('z500_type')!r}")


def _fig9_contour_abs(ax, z):
    lon_dim, lat_dim = _fig9_lon_lat_dims(z)
    levels = np.arange(5520, 5921, 40)
    levels = levels[(levels >= float(z.min())) & (levels <= float(z.max()))]
    if levels.size:
        cs = ax.contour(z[lon_dim], z[lat_dim], z, levels=levels, colors="0.12", linewidths=1.1, transform=ccrs.PlateCarree())
        _fig9_effects(ax.clabel(cs, fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))
    if float(z.min()) <= 5880 <= float(z.max()):
        ax.contour(z[lon_dim], z[lat_dim], z, levels=[5880], colors="purple", linewidths=1.5, transform=ccrs.PlateCarree())


def _fig9_contour_diff(ax, z):
    lon_dim, lat_dim = _fig9_lon_lat_dims(z)
    base = np.array([10, 20, 40, 60, 80], dtype=float)
    base = base[base <= float(np.nanmax(np.abs(z.values)))]
    if base.size:
        csn = ax.contour(z[lon_dim], z[lat_dim], z, levels=-base[::-1], colors="royalblue", linestyles="--", linewidths=1.05, transform=ccrs.PlateCarree())
        csp = ax.contour(z[lon_dim], z[lat_dim], z, levels=base, colors="firebrick", linewidths=1.05, transform=ccrs.PlateCarree())
        labs = base[base >= 20]
        if labs.size:
            _fig9_effects(ax.clabel(csn, levels=-labs[::-1], fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))
            _fig9_effects(ax.clabel(csp, levels=labs, fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))


def _fig9_plot_abs(ax, tp, z, levels, title=""):
    tp_lon_dim, tp_lat_dim = _fig9_lon_lat_dims(tp)
    _fig9_setup_map(ax)
    cf = ax.contourf(tp[tp_lon_dim], tp[tp_lat_dim], tp, levels=levels, cmap="Blues", extend="max", alpha=0.92, transform=ccrs.PlateCarree())
    _fig9_contour_abs(ax, z)
    _fig9_box(ax)
    if title:
        ax.set_title(title, fontsize=FONT_PANEL_TITLE)
    return cf


def _fig9_plot_diff(ax, tp, z, levels, norm, title=""):
    tp_lon_dim, tp_lat_dim = _fig9_lon_lat_dims(tp)
    _fig9_setup_map(ax)
    cf = ax.contourf(tp[tp_lon_dim], tp[tp_lat_dim], tp, levels=levels, cmap="RdBu_r", norm=norm, extend="both", alpha=0.92, transform=ccrs.PlateCarree())
    _fig9_contour_diff(ax, z)
    _fig9_box(ax)
    if title:
        ax.set_title(title, fontsize=FONT_PANEL_TITLE)
    return cf


def _fig9_lon_lat_dims(da: xr.DataArray) -> Tuple[str, str]:
    lon_dims = [dim for dim in da.dims if dim.endswith("_lon") or dim == "lon"]
    lat_dims = [dim for dim in da.dims if dim.endswith("_lat") or dim == "lat"]
    if not lon_dims or not lat_dims:
        raise ValueError(f"Cannot infer lon/lat dims for {da.name}: dims={da.dims}")
    return lon_dims[0], lat_dims[0]


def _fig9_open_significance(path: str, variable: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fig.9 {variable} significance file not found: {path}. "
            "Run export_fig9_spatial_significance() in the science/source cell first."
        )
    ds = xr.open_dataset(path)
    _fig9_require_cache_vars(
        ds,
        ["p_fdr_bh", "significant_fdr_005", "significant_local_p005", "hot_minus_cold"],
    )
    if str(ds.attrs.get("difference_direction", "")) != "Hot17 minus Cold17":
        raise ValueError(f"Fig.9 {variable} significance difference direction is invalid")
    if str(ds.attrs.get("grouping_stage", "")) != "Total":
        raise ValueError(f"Fig.9 {variable} significance must use Total grouping")
    finite = ds.p_fdr_bh.values[np.isfinite(ds.p_fdr_bh.values)]
    if np.any((finite < 0) | (finite > 1)):
        raise ValueError(f"Fig.9 {variable} significance p_fdr_bh contains values outside [0, 1]")
    return ds


def _fig9_validate_significance(plot_ds: xr.Dataset, tp_sig: xr.Dataset, z_sig: xr.Dataset = None) -> None:
    if str(plot_ds.attrs.get("sign_convention", "")) != "Hot17_minus_Cold17 = Hot17 - Cold17":
        raise ValueError("Fig.9 plot cache does not declare the required Hot17−Cold17 sign convention")
    for variable, sig in (("TP", tp_sig),):
        if not np.array_equal(plot_ds.init.values.astype(str), sig.init.values.astype(str)):
            raise ValueError(f"Fig.9 {variable} significance initializations do not match plot cache")
        if not np.array_equal(plot_ds.s2s_lat.values, sig.s2s_lat.values):
            raise ValueError(f"Fig.9 {variable} significance latitude does not match native S2S plot grid")
        if not np.array_equal(plot_ds.s2s_lon.values, sig.s2s_lon.values):
            raise ValueError(f"Fig.9 {variable} significance longitude does not match native S2S plot grid")
        if sig.significant_local_p005.shape != plot_ds.tp_diff.shape:
            raise ValueError(f"Fig.9 {variable} significance-mask shape does not match difference panels")
    if not np.allclose(tp_sig.hot_minus_cold.values, plot_ds.tp_diff.values, rtol=1e-7, atol=1e-6, equal_nan=True):
        raise ValueError("Fig.9 TP significance difference does not match the plotted Hot17−Cold17 field")


def _fig9_overlay_tp_significance(ax, mask: xr.DataArray) -> None:
    sig = mask.astype(bool)
    if not bool(sig.any().item()):
        return
    with plt.rc_context({"hatch.color": "black", "hatch.linewidth": 0.45}):
        ax.contourf(
            sig.s2s_lon, sig.s2s_lat, sig.astype(float), levels=[0.5, 1.5],
            colors="none", hatches=["..."], transform=ccrs.PlateCarree(), zorder=12,
        )


def plot_fig9_from_cache(
    cache_path: str = FIG10_DEFAULT_CACHE,
    out_dir: str = FIG10_DEFAULT_OUT_DIR,
    tp_significance_path: str = FIG10_SIGNIFICANCE_TP,
    z500_significance_path: str = FIG10_SIGNIFICANCE_Z500,
):
    os.makedirs(out_dir, exist_ok=True)
    with xr.open_dataset(cache_path) as source:
        ds = source.load()
    _fig9_require_cache_vars(ds, ["tp_hot", "z500_abs_hot", "tp_cold", "z500_abs_cold", "tp_diff", "z500_abs_diff", "tp_obs_ref", "z500_abs_obs_ref"])
    _fig9_check_z500_metadata(ds)
    if ds.attrs.get("grouping_stage") != "Total" or ds.attrs.get("analysis_window") != "2023-06-14 to 2023-06-24":
        raise ValueError("Fig.9 requires Total grouping and BJT June 14–24")
    inits = [str(v)[:10] for v in ds.init.values.astype("datetime64[D]")]
    if inits != ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]:
        raise ValueError("Fig.9 initialization order differs")
    tp_sig = _fig9_open_significance(tp_significance_path, "TP")
    _fig9_validate_significance(ds, tp_sig)
    print("[Fig9 local significance] local two-sided Welch t-test p < 0.05")
    for r, init in enumerate(ds.init.values):
        tp_count = int(tp_sig.significant_local_p005.isel(init=r).astype(bool).sum().item())
        init_label = (
            np.datetime_as_string(init, unit="D")
            if np.issubdtype(np.asarray(init).dtype, np.datetime64)
            else str(init)[:10]
        )
        print(f"{init_label}: TP={tp_count}")
    abs_levels, diff_levels, diff_norm = _fig9_levels_from_cache_or_fallback(ds)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG10_HEIGHT_IN))
    gs = fig.add_gridspec(
        4, 3, left=0.145, right=0.985, top=0.965, bottom=0.27,
        hspace=0.11, wspace=0.10,
    )
    axes = np.empty((4, 3), dtype=object)
    for r in range(4):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
    ax_ref = fig.add_axes([0.145, 0.035, 0.245, 0.155], projection=ccrs.PlateCarree())
    cax_abs = fig.add_axes([0.43, 0.125, 0.52, 0.012])
    cax_diff = fig.add_axes([0.43, 0.050, 0.52, 0.012])
    cf_abs = cf_diff = None
    titles = ["Hot17", "Cold17", "Hot17 − Cold17"]
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=FONT_COLUMN_HEADER, fontweight="bold", pad=3)
    labels = list("abcdefghijklm")
    for r, init in enumerate(ds.init.values):
        cf_abs = _fig9_plot_abs(axes[r, 0], ds.tp_hot.sel(init=init), ds.z500_abs_hot.sel(init=init), abs_levels)
        _fig9_plot_abs(axes[r, 1], ds.tp_cold.sel(init=init), ds.z500_abs_cold.sel(init=init), abs_levels)
        cf_diff = _fig9_plot_diff(axes[r, 2], ds.tp_diff.sel(init=init), ds.z500_abs_diff.sel(init=init), diff_levels, diff_norm)
        _fig9_overlay_tp_significance(axes[r, 2], tp_sig.significant_local_p005.isel(init=r))
        for c in range(3):
            _fig9_label(axes[r, c], f"({labels[r * 3 + c]})")
            _fig9_format_geo_axes(axes[r, c], show_left=(c == 0), show_bottom=(r == 3))
    _fig9_plot_abs(ax_ref, ds.tp_obs_ref, ds.z500_abs_obs_ref, abs_levels, "Observed")
    _fig9_label(ax_ref, "(m)")
    _fig9_format_geo_axes(ax_ref, show_left=True, show_bottom=True)
    for r, init in enumerate(ds.init.values):
        init_label = (
            np.datetime_as_string(init, unit="D")
            if np.issubdtype(np.asarray(init).dtype, np.datetime64)
            else str(init)[:10]
        )
        short_init = init_label[8:10].lstrip("0") + " Jun"
        pos = axes[r, 0].get_position()
        fig.text(
            0.025, 0.5 * (pos.y0 + pos.y1), short_init,
            rotation=90, ha="center", va="center",
            fontsize=FONT_ROW_HEADER, fontweight="bold",
        )
    cbar_abs = fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_ticks(FIG9_ABS_TICKS)
    cbar_abs.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_abs.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_abs.set_label("Precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_ticks(FIG9_DIFF_TICKS)
    cbar_diff.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_diff.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_diff.set_label("Hot17 − Cold17 precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    stem = os.path.join(out_dir, "fig9_leadtime_tp_z500_composites")
    fig.savefig(stem + ".png", dpi=600)
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    tp_sig.close()
    return stem + ".png", stem + ".pdf"


plot_fig10_from_cache = plot_fig9_from_cache  # historical compatibility
