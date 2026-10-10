"""Plot paper Fig.8 from a plot-ready NetCDF cache only."""

import os
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
    _fig8_effects(ax.clabel(cs, levels=cs.levels[::2], fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))
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
            _fig8_effects(ax.clabel(csn, levels=-label_levels[::-1], fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))
            _fig8_effects(ax.clabel(csp, levels=label_levels, fmt="%.0f", fontsize=FONT_CONTOUR_LABEL, inline=True))


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
        ("z500", z500_sig, cache["z500_diff_hov"]),
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
    with plt.rc_context({"hatch.color": "black", "hatch.linewidth": 0.45}):
        ax.contourf(
            sig["s2s_lon"],
            ypad,
            sigpad,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["..."],
            zorder=6,
        )


def _fig8_overlay_z500_significance(
    ax,
    z500_diff: xr.DataArray,
    mask: xr.DataArray,
    dates,
) -> None:
    sig = mask.astype(bool)
    if not bool(sig.any().item()):
        return
    y = np.arange(len(dates), dtype=float)
    z_sig = z500_diff.where(sig)
    base = np.array([20, 40, 60, 80], dtype=float)
    max_abs = float(np.nanmax(np.abs(z500_diff.values)))
    base = base[base <= max_abs]
    if base.size:
        ax.contour(
            z_sig["s2s_lon"], y, z_sig.values,
            levels=-base[::-1], colors="#000080", linestyles="--",
            linewidths=2.6, zorder=8,
        )
        ax.contour(
            z_sig["s2s_lon"], y, z_sig.values,
            levels=base, colors="#8B0000", linewidths=2.6, zorder=8,
        )


def plot_fig8_from_cache(
    cache_path: str = FIG8_DEFAULT_CACHE,
    out_dir: str = FIG8_DEFAULT_OUT_DIR,
    tp_significance_path: str = FIG8_TP_SIGNIFICANCE,
    z500_significance_path: str = FIG8_Z500_SIGNIFICANCE,
):
    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(cache_path)
    _fig8_require_cache_vars(ds, ["tp_obs_hov", "z500_obs_hov", "tp_hot_hov", "z500_hot_hov", "tp_cold_hov", "z500_cold_hov", "tp_diff_hov", "z500_diff_hov"])
    _fig8_check_z500_metadata(ds)
    if "time" in ds.coords:
        time_coord = "time"
    elif "date" in ds.coords:
        time_coord = "date"
    else:
        raise KeyError("Fig.8 cache must include either a 'time' or 'date' coordinate")
    dates = ds[time_coord].values
    tp_sig = _fig8_open_significance(tp_significance_path, "TP")
    z500_sig = _fig8_open_significance(z500_significance_path, "Z500")
    _fig8_validate_significance(ds, tp_sig, z500_sig, time_coord)
    tp_local_count = int(tp_sig["significant_local_p005"].astype(bool).sum().item())
    z500_local_count = int(z500_sig["significant_local_p005"].astype(bool).sum().item())
    print("[Fig8 local significance] two-sided Welch p < 0.05")
    print(f"TP={tp_local_count}")
    print(f"Z500={z500_local_count}")
    abs_levels, diff_levels, diff_norm = _fig8_levels_from_cache_or_fallback(ds)
    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG8_HEIGHT_IN), sharex=True, sharey=True)
    _fig8_plot_abs(axs[0, 0], ds.tp_obs_hov, ds.z500_obs_hov, dates, abs_levels, "(a)", "Observed")
    _fig8_plot_abs(axs[0, 1], ds.tp_hot_hov, ds.z500_hot_hov, dates, abs_levels, "(b)", "Hot17", show_ylabels=False)
    cf_diff = _fig8_plot_diff(axs[1, 0], ds.tp_diff_hov, ds.z500_diff_hov, dates, diff_levels, diff_norm, "(c)", "Hot17 − Cold17", True)
    _fig8_overlay_tp_significance(
        axs[1, 0], tp_sig["significant_local_p005"], dates
    )
    _fig8_overlay_z500_significance(
        axs[1, 0], ds.z500_diff_hov,
        z500_sig["significant_local_p005"], dates,
    )
    cf_abs = _fig8_plot_abs(axs[1, 1], ds.tp_cold_hov, ds.z500_cold_hov, dates, abs_levels, "(d)", "Cold17", True, False)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.965, bottom=0.27, wspace=0.10, hspace=0.14)
    cbar_diff = fig.colorbar(cf_diff, cax=fig.add_axes([0.17, 0.085, 0.30, 0.012]), orientation="horizontal")
    cbar_diff.set_ticks(FIG8_DIFF_TICKS)
    cbar_diff.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_diff.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_diff.set_label("Hot17 − Cold17 precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    cbar_abs = fig.colorbar(cf_abs, cax=fig.add_axes([0.61, 0.085, 0.27, 0.012]), orientation="horizontal")
    cbar_abs.set_ticks(FIG8_ABS_TICKS)
    cbar_abs.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    cbar_abs.ax.tick_params(labelsize=FONT_COLORBAR_TICK, pad=1)
    cbar_abs.set_label("Precipitation (mm day$^{-1}$)", fontsize=FONT_COLORBAR_LABEL, labelpad=2)
    stem = os.path.join(out_dir, "fig8_hovmoller_propagation")
    fig.savefig(stem + ".png", dpi=600)
    fig.savefig(stem + ".pdf")
    plt.close(fig)
    return stem + ".png", stem + ".pdf"


RUN_FIG8_FROM_CACHE = True

if RUN_FIG8_FROM_CACHE and (__name__ == "__main__" or "get_ipython" in globals()):
    plot_fig8_from_cache()
