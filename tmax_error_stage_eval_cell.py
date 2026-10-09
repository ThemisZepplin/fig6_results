# =============================================================================
# [Independent Cell] Tmax forecast error evaluation by stage
# 依赖提示:
# 运行此 Cell 前，请确保已执行前面的主代码块，使得以下全局变量/函数驻留在内存中：
# - 辅助函数: get_ds, _get_var, _to_time_named, _detect_time_dim, _detect_lat_dim, _detect_lon_dim,
#            safe_slice_region, load_s2s_ensemble_with_cf
# - 全局变量: NCHN_BOX
# =============================================================================

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr

# -------------------------
# 配置
# -------------------------
ERA5_TMAX_FILE = "/data1/huangy/fig6/NC/ai_test/pangu/era5_T/era5_2m_temperature_6_2023.nc"
S2S_PF_FILE = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_mx2t6_2023-06.grb"
S2S_CF_FILE = "/data1/huangy/fig6/NC/cf_2023-6/ecmf_cf_sfc_mx2t6_2023-06.grib"

OUT_ROOT = "/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit"
FIG_DIR = os.path.join(OUT_ROOT, "figures")
TAB_DIR = os.path.join(OUT_ROOT, "tables")
QC_DIR = os.path.join(OUT_ROOT, "qc")
PAPER_FIG_DIR = "/data1/huangy/fig6/NC/v9/paper_figures/fig2/fig2a_tmax_error_divergence"

INITS = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]
PAPER_FIG2A_INITS = ["2023-06-12", "2023-06-08"]
TARGET_DATES = pd.date_range("2023-06-14", "2023-06-24", freq="D")
STAGES = {
    "Stage-I": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17")),
    "Stage-II": (pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24")),
    "Total": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-24")),
}
EXPECTED_NDAYS = {"Stage-I": 4, "Stage-II": 7, "Total": 11}

# -------------------------
# Paper style shared with standalone Fig.3–Fig.6
# -------------------------
PAPER_FIG_WIDTH_IN = 8.6
PAPER_STYLE = {
    "font_family": "Arial",
    "suptitle_size": 15,
    "title_size": 15,
    "panel_title_size": 11.5,
    "axis_label_size": 10.5,
    "tick_label_size": 9.3,
    "legend_size": 9.5,
    "annotation_size": 8.2,
    "panel_label_size": 12,
    "line_width": 1.75,
    "marker_size": 4.8,
    "grid_line_width": 0.45,
    "timeseries_grid_alpha": 0.20,
    "bar_grid_alpha": 0.12,
    "dpi": 300,
    "metric_colors": {
        "Bias": "#3C78D8",
        "RMSE": "#6AA84F",
    },
}

PAPER_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _paper_format_signed(value, digits=2):
    return f"{value:+.{digits}f}".replace("-", "−")


def _paper_apply_axis_base_style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")
    ax.tick_params(axis="both", which="major", width=0.8, length=3.2, labelsize=PAPER_STYLE["tick_label_size"])
    ax.xaxis.label.set_size(PAPER_STYLE["axis_label_size"])
    ax.yaxis.label.set_size(PAPER_STYLE["axis_label_size"])
    ax.title.set_size(PAPER_STYLE["panel_title_size"])


def _paper_apply_timeseries_style(ax):
    _paper_apply_axis_base_style(ax)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=PAPER_STYLE["grid_line_width"], alpha=PAPER_STYLE["timeseries_grid_alpha"])


def _paper_apply_vertical_bar_style(ax):
    _paper_apply_axis_base_style(ax)
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=PAPER_STYLE["grid_line_width"], alpha=PAPER_STYLE["bar_grid_alpha"])
    ax.grid(False, axis="x")


def _paper_apply_boxplot_style(ax):
    _paper_apply_axis_base_style(ax)
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=PAPER_STYLE["grid_line_width"], alpha=PAPER_STYLE["bar_grid_alpha"])
    ax.grid(False, axis="x")


def _save_fixed_canvas_paper_figure(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=PAPER_STYLE["dpi"], facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    width_in, height_in = fig.get_size_inches()
    expected_width_px = int(round(PAPER_FIG_WIDTH_IN * PAPER_STYLE["dpi"]))
    print(f"[Paper Figure QC] fixed width={width_in:.2f} in | height={height_in:.2f} in | no tight bbox")
    try:
        img = plt.imread(png_path)
        height_px, width_px = img.shape[:2]
        print(f"[Paper Figure QC] PNG pixels: width={width_px}, height={height_px}; expected width={expected_width_px}")
    except Exception as exc:
        print(f"[Paper Figure QC][Warning] Could not read PNG pixel dimensions for {png_path}: {exc}")


def _coslat_weighted_mean(da):
    lat_name = _detect_lat_dim(da)
    lon_name = _detect_lon_dim(da)
    w = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(w).mean(dim=[lat_name, lon_name])


def _to_celsius_if_needed(da):
    return da - 273.15 if float(da.mean()) > 150 else da


def _stage_label(ts):
    for lab, (s, e) in STAGES.items():
        if s <= ts <= e:
            return lab
    return "Outside"


def load_era5_tmax_daily():
    ds = get_ds(ERA5_TMAX_FILE)
    da = _get_var(ds, ["t2m"])
    da, _ = _to_time_named(da)
    da = da.rename({_detect_lat_dim(da): "latitude", _detect_lon_dim(da): "longitude"})
    da = safe_slice_region(da, *NCHN_BOX)

    da = da.assign_coords(time=pd.to_datetime(da.time.values) + pd.Timedelta(hours=8))
    da_daily = da.resample(time="1D").max()
    da_daily = _to_celsius_if_needed(da_daily)
    da_daily_reg = _coslat_weighted_mean(da_daily)

    out = da_daily_reg.sel(time=slice(TARGET_DATES.min(), TARGET_DATES.max())).sortby("time")

    print("[QC] ERA5 Tmax daily:")
    print(f"  date range: {pd.to_datetime(out.time.values).min()} -> {pd.to_datetime(out.time.values).max()}")
    print(f"  min/max: {float(out.min()):.2f} / {float(out.max()):.2f} °C")

    return out


def load_s2s_tmax_daily_members(init_date):
    da_merged = load_s2s_ensemble_with_cf(
        S2S_PF_FILE,
        S2S_CF_FILE,
        ["mx2t6", "mx2t", "t2m", "2m_temperature", "t2m_max"],
    )

    tc = _detect_time_dim(da_merged)
    dates_in_ds = pd.to_datetime(da_merged[tc].values).date
    init_d = pd.Timestamp(init_date).date()
    if init_d not in dates_in_ds:
        raise ValueError(f"Init date {init_date} not found in S2S data")

    idx = np.where(dates_in_ds == init_d)[0][0]
    da_init = da_merged.isel({da_merged[tc].dims[0]: idx})
    da_init = da_init.rename({_detect_lat_dim(da_init): "latitude", _detect_lon_dim(da_init): "longitude"})
    da_init = safe_slice_region(da_init, *NCHN_BOX)

    bjt_time = pd.to_datetime(da_init[tc].values) + da_init.step.values + pd.Timedelta(hours=8)
    da_init = da_init.assign_coords(valid_bjt=("step", bjt_time))
    if tc in da_init.coords and tc not in da_init.dims:
        da_init = da_init.drop_vars(tc)
    da_init = da_init.swap_dims({"step": "valid_bjt"}).rename({"valid_bjt": "time"}).sortby("time")

    da_daily = da_init.resample(time="1D").max()
    da_daily = _to_celsius_if_needed(da_daily)
    da_daily = _coslat_weighted_mean(da_daily)
    da_daily = da_daily.sel(time=slice(TARGET_DATES.min(), TARGET_DATES.max())).sortby("time")

    if "number" not in da_daily.dims:
        raise ValueError(f"Init {init_date}: missing 'number' dimension")
    if da_daily.sizes["number"] != 51:
        raise ValueError(f"Init {init_date}: number size != 51, got {da_daily.sizes['number']}")

    present_dates = pd.DatetimeIndex(pd.to_datetime(da_daily.time.values).normalize())
    miss = TARGET_DATES.difference(present_dates)
    if len(miss) > 0:
        print(f"[QC] Missing S2S dates for init {init_date}: {list(miss.strftime('%Y-%m-%d'))}")
        raise ValueError(f"Init {init_date}: missing target dates")

    print(f"[QC] S2S Tmax daily (init={init_date}):")
    print(f"  dims: {da_daily.dims}")
    print(f"  number size: {da_daily.sizes['number']}")
    print(f"  date range: {pd.to_datetime(da_daily.time.values).min()} -> {pd.to_datetime(da_daily.time.values).max()}")
    print(f"  min/max: {float(da_daily.min()):.2f} / {float(da_daily.max()):.2f} °C")

    return da_daily


def compute_daily_error_table(era5_daily, s2s_daily_dict):
    recs = []
    for init, da_mem in s2s_daily_dict.items():
        ens_mean = da_mem.mean("number")
        ens_spread = da_mem.std("number")
        ens_min = da_mem.min("number")
        ens_max = da_mem.max("number")
        q = da_mem.quantile([0.05, 0.25, 0.75, 0.95], dim="number")

        common = pd.DatetimeIndex(pd.to_datetime(da_mem.time.values)).intersection(pd.DatetimeIndex(pd.to_datetime(era5_daily.time.values)))
        miss = TARGET_DATES.difference(common.normalize())
        if len(miss) > 0:
            print(f"[QC] Missing overlap dates for init {init}: {list(miss.strftime('%Y-%m-%d'))}")
            raise ValueError("ERA5/S2S date alignment failed")

        for dt in TARGET_DATES:
            obs = float(era5_daily.sel(time=dt))
            em = float(ens_mean.sel(time=dt))
            vals = da_mem.sel(time=dt).values
            rank = 100.0 * np.mean(vals <= obs)
            recs.append({
                "init": init,
                "date": dt.strftime("%Y-%m-%d"),
                "stage_label": _stage_label(dt),
                "ERA5_Tmax": obs,
                "S2S_ensmean_Tmax": em,
                "daily_error": em - obs,
                "ens_spread": float(ens_spread.sel(time=dt)),
                "ens_min": float(ens_min.sel(time=dt)),
                "ens_p05": float(q.sel(quantile=0.05, time=dt)),
                "ens_p25": float(q.sel(quantile=0.25, time=dt)),
                "ens_p75": float(q.sel(quantile=0.75, time=dt)),
                "ens_p95": float(q.sel(quantile=0.95, time=dt)),
                "ens_max": float(ens_max.sel(time=dt)),
                "era5_in_minmax": bool((obs >= float(ens_min.sel(time=dt))) and (obs <= float(ens_max.sel(time=dt)))),
                "era5_in_p05p95": bool((obs >= float(q.sel(quantile=0.05, time=dt))) and (obs <= float(q.sel(quantile=0.95, time=dt)))),
                "era5_percentile_rank": rank,
            })
    return pd.DataFrame(recs)


def compute_member_stage_metrics(era5_daily, s2s_daily_dict):
    recs = []
    for init, da_mem in s2s_daily_dict.items():
        for stage, (s, e) in STAGES.items():
            dsub = pd.date_range(s, e, freq="D")
            if len(dsub) != EXPECTED_NDAYS[stage]:
                raise ValueError(f"{stage} n_days mismatch")
            obs = era5_daily.sel(time=dsub)
            for m in da_mem.number.values:
                pred = da_mem.sel(number=m, time=dsub)
                err = pred - obs
                recs.append({
                    "init": init,
                    "stage": stage,
                    "member": int(m),
                    "member_type": "cf" if int(m) == 0 else "pf",
                    "bias": float(err.mean()),
                    "mae": float(np.abs(err).mean()),
                    "rmse": float(np.sqrt((err ** 2).mean())),
                    "n_days": len(dsub),
                })
    return pd.DataFrame(recs)


def compute_stage_summary(daily_df, member_df):
    recs = []
    dfx = daily_df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    init_order = [i for i in INITS if i in set(dfx["init"].astype(str))]
    for init in init_order:
        for stage, (s, e) in STAGES.items():
            dsub = dfx[(dfx["init"] == init) & (dfx["date"] >= s) & (dfx["date"] <= e)]
            n = len(dsub)
            if n != EXPECTED_NDAYS[stage]:
                raise ValueError(f"{init} {stage} n_days={n} != {EXPECTED_NDAYS[stage]}")
            mem = member_df[(member_df["init"] == init) & (member_df["stage"] == stage)]
            rmses = mem["rmse"].values
            recs.append({
                "init": init,
                "stage": stage,
                "ensmean_bias": float(dsub["daily_error"].mean()),
                "ensmean_mae": float(np.abs(dsub["daily_error"]).mean()),
                "ensmean_rmse": float(np.sqrt(np.mean(dsub["daily_error"] ** 2))),
                "member_rmse_median": float(np.median(rmses)),
                "member_rmse_iqr": float(np.percentile(rmses, 75) - np.percentile(rmses, 25)),
                "member_rmse_min": float(np.min(rmses)),
                "member_rmse_max": float(np.max(rmses)),
                "mean_spread": float(dsub["ens_spread"].mean()),
                "coverage_minmax": float(dsub["era5_in_minmax"].mean()),
                "coverage_p05p95": float(dsub["era5_in_p05p95"].mean()),
                "n_days": n,
            })
    return pd.DataFrame(recs)


def plot_tmax_daily_evolution(daily_df, s2s_daily_dict):
    plot_inits = [i for i in PAPER_FIG2A_INITS if i in s2s_daily_dict]
    if not plot_inits:
        print("[Plot] Skipping daily evolution plot: no 2023-06-08/2023-06-12 successful init available.")
        return
    fig, axes = plt.subplots(1, len(plot_inits), figsize=(7 * len(plot_inits), 5), sharey=True, squeeze=False)
    axes = axes.ravel()

    for idx, (ax, init) in enumerate(zip(axes, plot_inits)):
        sub = daily_df[daily_df["init"] == init].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.sort_values("date")

        ax.plot(sub["date"], sub["ERA5_Tmax"], color="black", lw=2.2, label="ERA5")
        ax.plot(sub["date"], sub["S2S_ensmean_Tmax"], color="royalblue", lw=2.2, label="S2S ens mean")
        ax.fill_between(sub["date"], sub["ens_p25"], sub["ens_p75"], color="deepskyblue", alpha=0.30, label="p25-p75")
        ax.fill_between(sub["date"], sub["ens_p05"], sub["ens_p95"], color="deepskyblue", alpha=0.16, label="p05-p95")

        ax.axvspan(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17"), color="orange", alpha=0.12)
        ax.axvspan(pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24"), color="green", alpha=0.10)

        metric_lines = []
        for stage_name in ["Stage-I", "Stage-II", "Total"]:
            sdt, edt = STAGES[stage_name]
            stage_sub = sub[(sub["date"] >= sdt) & (sub["date"] <= edt)]
            bias = stage_sub["daily_error"].mean()
            rmse = np.sqrt(np.mean(stage_sub["daily_error"] ** 2))
            metric_lines.append(f"{stage_name}: B={bias:+.2f}, R={rmse:.2f}")
        ax.text(
            0.02, 0.98, "\n".join(metric_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray", boxstyle="round,pad=0.3")
        )

        ax.set_title(f"({chr(97 + idx)}) Tmax evolution (Init: {init})")
        ax.set_xlim(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-24"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.tick_params(axis='x', rotation=35)
        ax.grid(ls=":", alpha=0.45)

    axes[0].set_ylabel("Daily Tmax (°C)")
    axes[0].legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_daily_evolution_stage_error.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_daily_evolution_stage_error.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_member_rmse_boxplot(member_df, stage_df):
    plot_inits = [i for i in INITS if i in set(member_df["init"].astype(str))]
    fig, ax = plt.subplots(figsize=(max(13, 2.0 * len(plot_inits) * 3), 5))
    order = []
    for init in plot_inits:
        for s in ["Stage-I", "Stage-II", "Total"]:
            order.append((init, s))

    data = []
    labels = []
    ens_rmse = []
    for i, (init, stg) in enumerate(order, 1):
        sub = member_df[(member_df.init == init) & (member_df.stage == stg)]
        data.append(sub["rmse"].values)
        labels.append(f"{init[5:7]}{init[8:10]} {stg}")
        ens_rmse.append(float(stage_df[(stage_df.init == init) & (stage_df.stage == stg)]["ensmean_rmse"].iloc[0]))

    bp = ax.boxplot(data, patch_artist=True)
    base_colors = {"Stage-I": "#f4a261", "Stage-II": "#2a9d8f", "Total": "#9e9e9e"}
    colors = [base_colors[stg] for _, stg in order]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, 1):
        x = i + rng.normal(0, 0.05, size=len(vals))
        ax.scatter(x, vals, s=14, alpha=0.45, color="k")
        ax.scatter([i], [ens_rmse[i - 1]], s=45, color="black", marker="D")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("member-wise Tmax RMSE (°C)")
    ax.grid(ls=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_member_rmse_boxplot_by_stage.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_member_rmse_boxplot_by_stage.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_stageI_stageII_scatter(member_df):
    plot_inits = [i for i in PAPER_FIG2A_INITS if i in set(member_df["init"].astype(str))]
    if not plot_inits:
        print("[Plot] Skipping Stage-I vs Stage-II scatter: no 2023-06-08/2023-06-12 successful init available.")
        return
    fig, axes = plt.subplots(1, len(plot_inits), figsize=(5.5 * len(plot_inits), 5), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, init in zip(axes, plot_inits):
        d1 = member_df[(member_df.init == init) & (member_df.stage == "Stage-I")][["member", "rmse"]].rename(columns={"rmse": "rmse1"})
        d2 = member_df[(member_df.init == init) & (member_df.stage == "Stage-II")][["member", "rmse"]].rename(columns={"rmse": "rmse2"})
        dd = d1.merge(d2, on="member")
        dd = dd.sort_values("rmse1").reset_index(drop=True)
        dd["group"] = "Middle17"
        dd.loc[:16, "group"] = "Best17"
        dd.loc[34:, "group"] = "Worst17"

        cmap = {"Best17": "#2a9d8f", "Middle17": "#457b9d", "Worst17": "#e76f51"}
        for g, sdf in dd.groupby("group"):
            ax.scatter(sdf.rmse1, sdf.rmse2, c=cmap[g], label=g, alpha=0.82, s=35)

        pr, pp = pearsonr(dd.rmse1, dd.rmse2)
        sr, sp = spearmanr(dd.rmse1, dd.rmse2)
        mn, mx = min(dd.rmse1.min(), dd.rmse2.min()), max(dd.rmse1.max(), dd.rmse2.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.text(0.02, 0.98, f"Pearson r={pr:.2f} (p={pp:.3f})\nSpearman r={sr:.2f} (p={sp:.3f})", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.text(0.02, 0.12, "Colors indicate Stage-I RMSE-based\nskill groups, not Hot/Cold groups.",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.25"))
        ax.set_title(f"Init: {init}\nStage-I skill group persistence to Stage-II")
        ax.set_xlabel("Stage-I member RMSE (°C)")
        ax.grid(ls=":", alpha=0.5)

    axes[0].set_ylabel("Stage-II member RMSE (°C)")
    axes[0].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_stageI_vs_stageII_member_rmse_scatter.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_stageI_vs_stageII_member_rmse_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_stage_summary_metrics(stage_df):
    p = stage_df.copy()
    p["label"] = p["init"].str[5:7] + p["init"].str[8:10] + " " + p["stage"]
    plot_inits = [i for i in INITS if i in set(stage_df["init"].astype(str))]
    order = [f"{i[5:7]}{i[8:10]} {s}" for i in plot_inits for s in ["Stage-I", "Stage-II", "Total"]]
    p = p.set_index("label").loc[order].reset_index()

    fig, axs = plt.subplots(2, 2, figsize=(max(13, 1.25 * len(order)), 8))
    metrics = [("ensmean_bias", "Ensmean Bias (°C)"), ("ensmean_rmse", "Ensmean RMSE (°C)"), ("member_rmse_median", "Member RMSE median (°C)"), ("coverage_p05p95", "Coverage p05-p95")]
    for ax, (col, ttl) in zip(axs.flat, metrics):
        ax.bar(np.arange(len(p)), p[col].values, color=[{"Stage-I": "#f4a261", "Stage-II": "#2a9d8f", "Total": "#bdbdbd"}[stg] for stg in p["stage"]])
        ax.set_title(ttl)
        ax.set_xticks(np.arange(len(p)))
        ax.set_xticklabels(p["label"], rotation=25, ha="right")
        ax.grid(ls=":", axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_stage_summary_metrics.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_stage_summary_metrics.pdf"), bbox_inches="tight")
    plt.close(fig)


def compute_member_hotcold_groups_by_stage(era5_daily, s2s_daily_dict):
    recs = []
    for init, da_mem in s2s_daily_dict.items():
        for stage, (s, e) in STAGES.items():
            dsub = pd.date_range(s, e, freq="D")
            obs = era5_daily.sel(time=dsub)
            era5_stage_mean = float(obs.mean())
            stage_rows = []
            for m in da_mem.number.values:
                pred = da_mem.sel(number=m, time=dsub)
                err = pred - obs
                stage_rows.append({
                    "init": init,
                    "stage": stage,
                    "member": int(m),
                    "member_type": "cf" if int(m) == 0 else "pf",
                    "member_stage_mean_Tmax": float(pred.mean()),
                    "ERA5_stage_mean_Tmax": era5_stage_mean,
                    "bias": float(err.mean()),
                    "mae": float(np.abs(err).mean()),
                    "rmse": float(np.sqrt((err ** 2).mean())),
                })
            sdf = pd.DataFrame(stage_rows).sort_values("member_stage_mean_Tmax", ascending=False).reset_index(drop=True)
            sdf["rank_by_stage_mean_Tmax"] = np.arange(1, len(sdf) + 1)
            sdf["hotcold_group"] = "Middle17"
            sdf.loc[sdf["rank_by_stage_mean_Tmax"] <= 17, "hotcold_group"] = "Hot17"
            sdf.loc[sdf["rank_by_stage_mean_Tmax"] >= 35, "hotcold_group"] = "Cold17"
            recs.append(sdf)
    return pd.concat(recs, ignore_index=True)


def compute_hotcold_group_summary_by_stage(hotcold_df):
    g = hotcold_df.groupby(["init", "stage", "hotcold_group"], as_index=False)
    out = g.agg(
        n_members=("member", "count"),
        group_mean_Tmax=("member_stage_mean_Tmax", "mean"),
        group_mean_bias=("bias", "mean"),
        group_mean_mae=("mae", "mean"),
        group_mean_rmse=("rmse", "mean"),
        group_median_rmse=("rmse", "median"),
        group_min_Tmax=("member_stage_mean_Tmax", "min"),
        group_max_Tmax=("member_stage_mean_Tmax", "max"),
    )
    return out


def compute_fixed_stageI_groups(hotcold_df):
    refs = hotcold_df[hotcold_df["stage"] == "Stage-I"][["init", "member", "hotcold_group"]].rename(columns={"hotcold_group": "fixed_stageI_group"})
    out = hotcold_df.merge(refs, on=["init", "member"], how="left")
    return out


def compute_fixed_stageI_group_summary(fixed_df):
    g = fixed_df.groupby(["init", "stage", "fixed_stageI_group"], as_index=False)
    return g.agg(
        n_members=("member", "count"),
        group_mean_Tmax=("member_stage_mean_Tmax", "mean"),
        group_mean_bias=("bias", "mean"),
        group_mean_mae=("mae", "mean"),
        group_mean_rmse=("rmse", "mean"),
        group_median_rmse=("rmse", "median"),
    )


def plot_tmax_hotcold_group_stage_mean_boxplot(hotcold_df):
    plot_inits = [i for i in INITS if i in set(hotcold_df["init"].astype(str))]
    fig, ax = plt.subplots(figsize=(max(14, 1.35 * len(plot_inits) * 3), 6))
    order = [(i, s) for i in plot_inits for s in ["Stage-I", "Stage-II", "Total"]]
    base_positions = np.arange(len(order))
    offsets = {"Hot17": -0.22, "Middle17": 0.0, "Cold17": 0.22}
    colors = {"Hot17": "#e76f51", "Middle17": "#457b9d", "Cold17": "#2a9d8f"}
    width = 0.18

    for grp in ["Hot17", "Middle17", "Cold17"]:
        data=[]; pos=[]
        for k,(init,stage) in enumerate(order):
            sub=hotcold_df[(hotcold_df.init==init)&(hotcold_df.stage==stage)&(hotcold_df.hotcold_group==grp)]
            data.append(sub["member_stage_mean_Tmax"].values)
            pos.append(base_positions[k]+offsets[grp])
        bp=ax.boxplot(data, positions=pos, widths=width, patch_artist=True, manage_ticks=False)
        for b in bp['boxes']:
            b.set_facecolor(colors[grp]); b.set_alpha(0.6)

    for k,(init,stage) in enumerate(order):
        y=float(hotcold_df[(hotcold_df.init==init)&(hotcold_df.stage==stage)]["ERA5_stage_mean_Tmax"].iloc[0])
        ax.plot(base_positions[k], y, marker='D', color='black', ms=5)
        ax.hlines(y, base_positions[k]-0.3, base_positions[k]+0.3, color='black', lw=1)

    labels=[f"{i[5:7]}{i[8:10]} {s}" for i,s in order]
    ax.set_xticks(base_positions); ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel("member stage-mean Tmax (°C)")
    ax.grid(ls=':', alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_hotcold_group_stage_mean_boxplot.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_hotcold_group_stage_mean_boxplot.pdf"), bbox_inches='tight')
    plt.close(fig)


def plot_tmax_hotcold_group_bias_rmse_summary(group_summary_df):
    def _auto_text_color(im, value, threshold=0.5):
        rgba = im.cmap(im.norm(value))
        r, g, b, _ = rgba
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if luminance < threshold else "black"

    plot_inits = [i for i in INITS if i in set(group_summary_df["init"].astype(str))]
    rows = [(i, s) for i in plot_inits for s in ["Stage-I", "Stage-II", "Total"]]
    cols = ["Hot17", "Middle17", "Cold17"]

    bias_mat = np.zeros((len(rows), len(cols)))
    rmse_mat = np.zeros((len(rows), len(cols)))
    for r, (init, stage) in enumerate(rows):
        for c, grp in enumerate(cols):
            row = group_summary_df[(group_summary_df.init == init) & (group_summary_df.stage == stage) & (group_summary_df.hotcold_group == grp)].iloc[0]
            bias_mat[r, c] = float(row.group_mean_bias)
            rmse_mat[r, c] = float(row.group_mean_rmse)

    row_labels = [f"{i[5:7]}{i[8:10]} {s}" for i, s in rows]

    fig, axs = plt.subplots(1, 2, figsize=(12, max(6, 0.45 * len(rows))))

    vmax_abs = max(0.1, float(np.nanmax(np.abs(bias_mat))))
    norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
    im0 = axs[0].imshow(bias_mat, cmap="RdBu_r", norm=norm, aspect="auto")
    axs[0].set_title("Bias heatmap (centered at 0)")
    axs[0].set_xticks(np.arange(len(cols))); axs[0].set_xticklabels(cols)
    axs[0].set_yticks(np.arange(len(row_labels))); axs[0].set_yticklabels(row_labels)
    for i in range(bias_mat.shape[0]):
        for j in range(bias_mat.shape[1]):
            v = bias_mat[i, j]
            txt_color = _auto_text_color(im0, v)
            stroke_color = "black" if txt_color == "white" else "white"
            txt = axs[0].text(j, i, f"{v:+.2f}", ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")
            txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground=stroke_color)])
            if abs(v) <= 0.5:
                axs[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec='black', lw=1.2))
    tick_vals = np.linspace(-vmax_abs, vmax_abs, 5)
    cb0 = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, ticks=tick_vals)
    cb0.ax.axhline(0, color="black", lw=1.2)
    cb0.set_label("Bias (°C; 0 = no bias)")

    im1 = axs[1].imshow(rmse_mat, cmap="YlGnBu", vmin=0.0, aspect="auto")
    axs[1].set_title("RMSE heatmap")
    axs[1].set_xticks(np.arange(len(cols))); axs[1].set_xticklabels(cols)
    axs[1].set_yticks(np.arange(len(row_labels))); axs[1].set_yticklabels(row_labels)
    for i in range(rmse_mat.shape[0]):
        for j in range(rmse_mat.shape[1]):
            v = rmse_mat[i, j]
            txt_color = _auto_text_color(im1, v)
            stroke_color = "black" if txt_color == "white" else "white"
            txt = axs[1].text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")
            txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground=stroke_color)])
    cb1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cb1.set_label("RMSE (°C)")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_hotcold_group_bias_rmse_summary.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_hotcold_group_bias_rmse_summary.pdf"), bbox_inches='tight')
    plt.close(fig)


def plot_tmax_stageI_hot17_persistence_to_stageII(fixed_df):
    plot_inits = [i for i in PAPER_FIG2A_INITS if i in set(fixed_df["init"].astype(str))]
    if not plot_inits:
        print("[Plot] Skipping Stage-I Hot17 persistence plot: no 2023-06-08/2023-06-12 successful init available.")
        return
    fig, axes = plt.subplots(1, len(plot_inits), figsize=(5.5 * len(plot_inits), 5), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, init in zip(axes, plot_inits):
        sub = fixed_df[(fixed_df.init==init) & (fixed_df.stage=="Stage-II")]
        data=[sub[sub.fixed_stageI_group==g]["rmse"].values for g in ["Hot17","Middle17","Cold17"]]
        bp=ax.boxplot(data, patch_artist=True)
        for b,c in zip(bp['boxes'], ["#e76f51","#457b9d","#2a9d8f"]):
            b.set_facecolor(c); b.set_alpha(0.65)
        ax.set_xticklabels(["Stage-I Hot17","Stage-I Mid17","Stage-I Cold17"], rotation=20)
        ax.set_title(f"Init: {init}")
        ax.grid(ls=':', alpha=0.4)
    axes[0].set_ylabel("Stage-II member RMSE (°C)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_stageI_hot17_persistence_to_stageII.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, "fig_tmax_stageI_hot17_persistence_to_stageII.pdf"), bbox_inches='tight')
    plt.close(fig)


def _compute_fig2a_summary_for_init(member_df, stage_df, init):
    recs = []
    for stage in ["Stage-I", "Stage-II", "Total"]:
        stg = stage_df[(stage_df["init"] == init) & (stage_df["stage"] == stage)].iloc[0]
        mem = member_df[(member_df["init"] == init) & (member_df["stage"] == stage)]["rmse"].dropna()
        recs.append({
            "init": init,
            "stage": stage,
            "ensmean_bias": float(stg["ensmean_bias"]),
            "ensmean_rmse": float(stg["ensmean_rmse"]),
            "member_rmse_mean": float(mem.mean()),
            "member_rmse_median": float(mem.median()),
            "member_rmse_q25": float(mem.quantile(0.25)),
            "member_rmse_q75": float(mem.quantile(0.75)),
        })
    return pd.DataFrame(recs)


def make_paper_fig2a(daily_df, member_df, stage_df, init, out_dir=PAPER_FIG_DIR):
    with plt.rc_context(PAPER_RC_PARAMS):
        os.makedirs(out_dir, exist_ok=True)
        metric_stage_order = ["Stage-I", "Total"]
        plot_stage_order = ["Stage-I", "Total"]
        stage_colors = {"Stage-I": "#f4a261", "Stage-II": "#2a9d8f", "Total": "#9e9e9e"}

        sub = daily_df[daily_df["init"] == init].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.sort_values("date")
        fig2a_summary = _compute_fig2a_summary_for_init(member_df, stage_df, init)

        fig = plt.figure(figsize=(PAPER_FIG_WIDTH_IN, 5.8), constrained_layout=False)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.42, 1.0])
        ax_ts = fig.add_subplot(gs[0, :])
        ax_bias = fig.add_subplot(gs[1, 0])
        ax_rmse = fig.add_subplot(gs[1, 1])

        # Panel (a): daily Tmax evolution
        stage_boundary = pd.Timestamp("2023-06-17 12:00")
        ax_ts.axvline(stage_boundary, color="0.25", lw=1.0, ls="-", zorder=2)
        ax_ts.fill_between(sub["date"], sub["ens_p05"], sub["ens_p95"], color="deepskyblue", alpha=0.14, lw=0, label="S2S p05–p95", zorder=1)
        ax_ts.fill_between(sub["date"], sub["ens_p25"], sub["ens_p75"], color="deepskyblue", alpha=0.30, lw=0, label="S2S p25–p75", zorder=1)
        ax_ts.plot(sub["date"], sub["ERA5_Tmax"], color="black", lw=1.95, marker="o", ms=PAPER_STYLE["marker_size"], label="ERA5 Obs", zorder=3)
        ax_ts.plot(sub["date"], sub["S2S_ensmean_Tmax"], color="royalblue", lw=PAPER_STYLE["line_width"], marker="s", ms=PAPER_STYLE["marker_size"], label="S2S ensemble mean", zorder=3)

        metric_lines = []
        for stage in metric_stage_order:
            row = fig2a_summary[fig2a_summary["stage"] == stage].iloc[0]
            metric_lines.append(f"{stage}: Bias = {_paper_format_signed(row.ensmean_bias)}°C, RMSE = {row.ensmean_rmse:.2f}°C")
        ax_ts.text(
            0.018, 0.965, "\n".join(metric_lines),
            transform=ax_ts.transAxes, ha="left", va="top", fontsize=PAPER_STYLE["annotation_size"],
            bbox=dict(facecolor="white", alpha=0.86, edgecolor="0.82", boxstyle="round,pad=0.28"),
        )
        ax_ts.set_title("(a) Tmax evolution", loc="left", fontsize=PAPER_STYLE["panel_title_size"])
        ax_ts.set_ylabel("Daily Tmax (°C)", fontsize=PAPER_STYLE["axis_label_size"])
        ax_ts.set_xlim(pd.Timestamp("2023-06-13 12:00"), pd.Timestamp("2023-06-24 12:00"))
        ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax_ts.get_xticklabels(), rotation=30, ha="right", fontsize=PAPER_STYLE["tick_label_size"])
        _paper_apply_timeseries_style(ax_ts)
        handles, labels = ax_ts.get_legend_handles_labels()
        legend_order = ["ERA5 Obs", "S2S ensemble mean", "S2S p25–p75", "S2S p05–p95"]
        ordered_handles = [handles[labels.index(lab)] for lab in legend_order if lab in labels]
        ordered_labels = [lab for lab in legend_order if lab in labels]
        ax_ts.legend(ordered_handles, ordered_labels, loc="lower right", ncol=2, fontsize=PAPER_STYLE["legend_size"], frameon=False)

        # Panel (b): ensemble-mean bias by stage
        x = np.arange(len(plot_stage_order))
        bias_vals = [float(fig2a_summary[fig2a_summary["stage"] == stage]["ensmean_bias"].iloc[0]) for stage in plot_stage_order]
        bars = ax_bias.bar(x, bias_vals, color=[stage_colors[s] for s in plot_stage_order], alpha=0.82, width=0.62, edgecolor="0.25", linewidth=0.6)
        ax_bias.axhline(0, color="black", lw=1.0)
        bias_min = min(0.0, min(bias_vals))
        bias_max = max(0.0, max(bias_vals))
        bias_pad = max(0.15, (bias_max - bias_min) * 0.15)
        ax_bias.set_ylim(bias_min - bias_pad, bias_max + bias_pad)
        for bar, val in zip(bars, bias_vals):
            va = "bottom" if val >= 0 else "top"
            yoff = bias_pad * 0.18 if val >= 0 else -bias_pad * 0.18
            ax_bias.text(bar.get_x() + bar.get_width() / 2, val + yoff, _paper_format_signed(val), ha="center", va=va, fontsize=PAPER_STYLE["annotation_size"])
        ax_bias.set_xticks(x)
        ax_bias.set_xticklabels(plot_stage_order, fontsize=PAPER_STYLE["tick_label_size"])
        ax_bias.set_ylabel("Ensemble-mean bias (°C)", fontsize=PAPER_STYLE["axis_label_size"])
        ax_bias.set_title("(b) Ensemble-mean bias", loc="left", fontsize=PAPER_STYLE["panel_title_size"])
        _paper_apply_vertical_bar_style(ax_bias)

        # Panel (c): member-wise RMSE distribution by stage
        rmse_data = [member_df[(member_df["init"] == init) & (member_df["stage"] == stage)]["rmse"].values for stage in plot_stage_order]
        bp = ax_rmse.boxplot(rmse_data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, stage in zip(bp["boxes"], plot_stage_order):
            patch.set_facecolor(stage_colors[stage])
            patch.set_alpha(0.55)
            patch.set_edgecolor("0.25")
            patch.set_linewidth(0.8)
        for key in ["medians", "whiskers", "caps"]:
            for artist in bp[key]:
                artist.set_color("0.25")
                artist.set_linewidth(0.8)
        rng = np.random.default_rng(20240612)
        for idx, (stage, vals) in enumerate(zip(plot_stage_order, rmse_data), 1):
            jitter = rng.normal(0, 0.032, size=len(vals))
            ax_rmse.scatter(np.full(len(vals), idx) + jitter, vals, s=10, color="0.25", alpha=0.42, linewidths=0)
            ens_rmse = float(fig2a_summary[fig2a_summary["stage"] == stage]["ensmean_rmse"].iloc[0])
            ax_rmse.scatter(idx, ens_rmse, marker="D", s=42, color="black", zorder=5)
        ax_rmse.set_xticks(np.arange(1, len(plot_stage_order) + 1))
        ax_rmse.set_xticklabels(plot_stage_order, fontsize=PAPER_STYLE["tick_label_size"])
        ax_rmse.set_ylabel("Member-wise RMSE (°C)", fontsize=PAPER_STYLE["axis_label_size"])
        ax_rmse.set_title("(c) Member RMSE distribution", loc="left", fontsize=PAPER_STYLE["panel_title_size"])
        _paper_apply_boxplot_style(ax_rmse)

        fig.suptitle(f"Tmax forecast error and member divergence | Init {init}", fontsize=PAPER_STYLE["suptitle_size"], fontweight="normal", y=0.975)
        fig.subplots_adjust(left=0.095, right=0.985, top=0.895, bottom=0.125, wspace=0.28, hspace=0.43)

        safe_init = init
        if init == "2023-06-12":
            base_name = f"fig2a_tmax_error_divergence_{safe_init}_jgrstyle"
        elif init == "2023-06-08":
            base_name = f"figS2a_tmax_error_divergence_{safe_init}_jgrstyle"
        else:
            base_name = f"figS2a_like_tmax_error_divergence_{safe_init}_jgrstyle"
        out_png = os.path.join(out_dir, f"{base_name}.png")
        out_pdf = os.path.join(out_dir, f"{base_name}.pdf")
        out_csv = os.path.join(out_dir, f"fig2a_summary_metrics_{safe_init}.csv")
        _save_fixed_canvas_paper_figure(fig, out_png, out_pdf)
        plt.close(fig)
        fig2a_summary.to_csv(out_csv, index=False)

        print(f"[Paper Fig2a JGR style] Init {init} outputs:")
        print("  Fixed width: 8.6 inch; no tight bbox; panel labels=(a)/(b)/(c)")
        print(f"  Actual height: {5.8:.2f} inch")
        print(f"  PNG: {out_png}")
        print(f"  PDF: {out_pdf}")
        print(f"  CSV: {out_csv}")
        return {"png": out_png, "pdf": out_pdf, "csv": out_csv}

def make_all_paper_fig2a(daily_df, member_df, stage_df, out_dir=PAPER_FIG_DIR):
    outputs = {}
    valid = set(daily_df["init"].astype(str)).intersection(set(member_df["init"].astype(str))).intersection(set(stage_df["init"].astype(str)))
    for init in PAPER_FIG2A_INITS:
        if init not in valid:
            print(f"[Paper Fig2a] Skipping init {init}: not available in successful multi-init results.")
            continue
        outputs[init] = make_paper_fig2a(daily_df, member_df, stage_df, init, out_dir=out_dir)
    return outputs


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(QC_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)

    print(f"[Multi-init QC] requested inits: {INITS}")
    era5_daily = load_era5_tmax_daily()

    s2s_daily_dict = {}
    failed_init_records = []
    availability_records = []
    for init in INITS:
        try:
            da_mem = load_s2s_tmax_daily_members(init)
            dates = pd.DatetimeIndex(pd.to_datetime(da_mem.time.values)).normalize()
            complete_target_dates = bool(TARGET_DATES.difference(dates).empty)
            rec = {
                "requested_init": init,
                "status": "success",
                "error_message": "",
                "n_members": int(da_mem.sizes.get("number", -1)),
                "date_start": str(pd.to_datetime(da_mem.time.values).min())[:10],
                "date_end": str(pd.to_datetime(da_mem.time.values).max())[:10],
                "complete_target_dates": complete_target_dates,
            }
            s2s_daily_dict[init] = da_mem
            availability_records.append(rec)
            print(f"[Multi-init QC][{init}] success | n_members={rec['n_members']} | date_range={rec['date_start']}->{rec['date_end']} | complete_target_dates={complete_target_dates}")
        except Exception as exc:
            msg = repr(exc)
            failed_init_records.append({"requested_init": init, "error_message": msg})
            availability_records.append({
                "requested_init": init,
                "status": "failed",
                "error_message": msg,
                "n_members": np.nan,
                "date_start": "",
                "date_end": "",
                "complete_target_dates": False,
            })
            print(f"[Multi-init QC][{init}] failed: {msg}")

    valid_inits = list(s2s_daily_dict.keys())
    failed_inits = [r["requested_init"] for r in failed_init_records]
    print(f"[Multi-init QC] successful inits: {valid_inits}")
    print(f"[Multi-init QC] failed inits: {failed_inits}")
    qc_available_path = os.path.join(QC_DIR, "qc_available_init_dates.csv")
    pd.DataFrame(availability_records).to_csv(qc_available_path, index=False)
    print(f"[Multi-init QC] availability table: {qc_available_path}")

    if not valid_inits:
        raise RuntimeError("No requested S2S init could be loaded; cannot compute multi-init Tmax diagnostics.")

    daily_df = compute_daily_error_table(era5_daily, s2s_daily_dict)
    member_df = compute_member_stage_metrics(era5_daily, s2s_daily_dict)
    stage_df = compute_stage_summary(daily_df, member_df)

    hotcold_df = compute_member_hotcold_groups_by_stage(era5_daily, s2s_daily_dict)
    hotcold_summary_df = compute_hotcold_group_summary_by_stage(hotcold_df)
    fixed_stageI_df = compute_fixed_stageI_groups(hotcold_df)
    fixed_stageI_summary_df = compute_fixed_stageI_group_summary(fixed_stageI_df)

    daily_df.to_csv(os.path.join(TAB_DIR, "tmax_daily_error_timeseries.csv"), index=False)
    member_df.to_csv(os.path.join(TAB_DIR, "tmax_member_error_by_stage.csv"), index=False)
    stage_df.to_csv(os.path.join(TAB_DIR, "tmax_stage_summary_metrics.csv"), index=False)
    hotcold_df.to_csv(os.path.join(TAB_DIR, "tmax_member_hotcold_group_by_stage.csv"), index=False)
    hotcold_summary_df.to_csv(os.path.join(TAB_DIR, "tmax_hotcold_group_summary_by_stage.csv"), index=False)
    fixed_stageI_df.to_csv(os.path.join(TAB_DIR, "tmax_member_fixed_stageI_group_by_stage.csv"), index=False)
    fixed_stageI_summary_df.to_csv(os.path.join(TAB_DIR, "tmax_fixed_stageI_group_summary_by_stage.csv"), index=False)

    run_summary_lines = [
        f"[Multi-init QC] requested_inits={INITS}",
        f"[Multi-init QC] successful_inits={valid_inits}",
        f"[Multi-init QC] failed_inits={failed_inits}",
        f"[Multi-init QC] qc_available_init_dates_csv={qc_available_path}",
    ]
    for rec in availability_records:
        line = (f"[Init availability][{rec['requested_init']}] status={rec['status']} | "
                f"n_members={rec['n_members']} | date_start={rec['date_start']} | date_end={rec['date_end']} | "
                f"complete_target_dates={rec['complete_target_dates']} | error={rec['error_message']}")
        run_summary_lines.append(line)

    # QC outputs
    qc_basic_recs = []
    qc_basic_recs.append({
        "dataset": "ERA5", "init": "ALL", "n_members": 1,
        "date_start": str(pd.to_datetime(era5_daily.time.values).min())[:10],
        "date_end": str(pd.to_datetime(era5_daily.time.values).max())[:10],
        "value_min": float(era5_daily.min()),
        "value_max": float(era5_daily.max()),
    })
    for init, da_mem in s2s_daily_dict.items():
        qc_basic_recs.append({
            "dataset": "S2S", "init": init, "n_members": int(da_mem.sizes.get("number", np.nan)),
            "date_start": str(pd.to_datetime(da_mem.time.values).min())[:10],
            "date_end": str(pd.to_datetime(da_mem.time.values).max())[:10],
            "value_min": float(da_mem.min()),
            "value_max": float(da_mem.max()),
        })
    pd.DataFrame(qc_basic_recs).to_csv(os.path.join(QC_DIR, "qc_tmax_basic_check.csv"), index=False)

    hotcold_counts = hotcold_df.groupby(["init", "stage", "hotcold_group"])["member"].count().reset_index(name="count")
    hotcold_counts.to_csv(os.path.join(QC_DIR, "qc_hotcold_group_counts.csv"), index=False)
    all_hotcold_17 = bool((hotcold_counts["count"] == 17).all())
    line_counts = f"[QC] tmax_member_hotcold_group_by_stage counts all init×stage×group == 17: {all_hotcold_17}"
    print(line_counts)
    run_summary_lines.append(line_counts)
    for _, row in hotcold_counts.iterrows():
        run_summary_lines.append(f"[HotCold count][{row['init']}][{row['stage']}][{row['hotcold_group']}] n={int(row['count'])}")

    fixed_stageI_df.groupby(["init", "stage", "fixed_stageI_group"])["member"].count().reset_index(name="count").to_csv(
        os.path.join(QC_DIR, "qc_fixed_stageI_group_counts.csv"), index=False
    )

    def _safe_plot(label, func, *args):
        try:
            func(*args)
        except Exception as exc:
            msg = f"[Plot warning] {label} skipped/failed: {repr(exc)}"
            print(msg)
            run_summary_lines.append(msg)

    _safe_plot("daily evolution", plot_tmax_daily_evolution, daily_df, s2s_daily_dict)
    _safe_plot("member RMSE boxplot", plot_member_rmse_boxplot, member_df, stage_df)
    _safe_plot("Stage-I vs Stage-II scatter", plot_stageI_stageII_scatter, member_df)
    _safe_plot("stage summary metrics", plot_stage_summary_metrics, stage_df)
    _safe_plot("Hot/Cold stage mean boxplot", plot_tmax_hotcold_group_stage_mean_boxplot, hotcold_df)
    _safe_plot("Hot/Cold bias/RMSE heatmap", plot_tmax_hotcold_group_bias_rmse_summary, hotcold_summary_df)
    _safe_plot("fixed Stage-I Hot17 persistence", plot_tmax_stageI_hot17_persistence_to_stageII, fixed_stageI_df)

    paper_fig2a_outputs = make_all_paper_fig2a(daily_df, member_df, stage_df)
    for init, paths in paper_fig2a_outputs.items():
        run_summary_lines.append(f"[Paper Fig2a][{init}] PNG={paths['png']}")
        run_summary_lines.append(f"[Paper Fig2a][{init}] PDF={paths['pdf']}")
        run_summary_lines.append(f"[Paper Fig2a][{init}] CSV={paths['csv']}")

    for init in valid_inits:
        s1 = stage_df[(stage_df.init == init) & (stage_df.stage == "Stage-I")].iloc[0]
        s2 = stage_df[(stage_df.init == init) & (stage_df.stage == "Stage-II")].iloc[0]
        d1 = member_df[(member_df.init == init) & (member_df.stage == "Stage-I")][["member", "rmse"]].rename(columns={"rmse": "rmse_stage1"})
        d2 = member_df[(member_df.init == init) & (member_df.stage == "Stage-II")][["member", "rmse"]].rename(columns={"rmse": "rmse_stage2"})
        d12 = d1.merge(d2, on="member", how="inner").sort_values("member")
        sr, sp = spearmanr(d12["rmse_stage1"].values, d12["rmse_stage2"].values)
        bigger = "Stage-I" if s1.ensmean_rmse > s2.ensmean_rmse else "Stage-II"
        line_main = f"[Summary][{init}] Stage-I Bias/RMSE={s1.ensmean_bias:+.2f}/{s1.ensmean_rmse:.2f}; Stage-II Bias/RMSE={s2.ensmean_bias:+.2f}/{s2.ensmean_rmse:.2f}; larger RMSE={bigger}; Spearman r={sr:.2f} (p={sp:.3f})"
        print(line_main)
        run_summary_lines.append(line_main)

        if s1.ensmean_rmse < 0.9 * s2.ensmean_rmse:
            msg = "Stage-I Tmax error is relatively smaller than Stage-II; mechanism analysis may focus more on Stage-II."
        else:
            msg = "Stage-I Tmax error is also substantial; Stage-I mechanism analysis is needed."
        print(msg)
        run_summary_lines.append(msg)

        if abs(sr) < 0.4:
            msg = "Member skill is not stable across stages, suggesting stage-dependent controlling mechanisms."
        else:
            msg = "Member skill is relatively consistent across stages, suggesting stable member-level forecast quality."
        print(msg)
        run_summary_lines.append(msg)

        for stage_name in ["Stage-I", "Stage-II"]:
            hh = hotcold_summary_df[(hotcold_summary_df.init==init)&(hotcold_summary_df.stage==stage_name)&(hotcold_summary_df.hotcold_group=="Hot17")].iloc[0]
            cc = hotcold_summary_df[(hotcold_summary_df.init==init)&(hotcold_summary_df.stage==stage_name)&(hotcold_summary_df.hotcold_group=="Cold17")].iloc[0]
            line_h = f"[{init}][{stage_name}] Hot17 mean Tmax/bias/RMSE = {hh.group_mean_Tmax:.2f}/{hh.group_mean_bias:+.2f}/{hh.group_mean_rmse:.2f}"
            line_c = f"[{init}][{stage_name}] Cold17 mean Tmax/bias/RMSE = {cc.group_mean_Tmax:.2f}/{cc.group_mean_bias:+.2f}/{cc.group_mean_rmse:.2f}"
            print(line_h)
            print(line_c)
            run_summary_lines.extend([line_h, line_c])
            if hh.group_mean_bias < -0.5:
                msg = "Even the hottest members remain colder than ERA5, indicating systematic cold bias."
                print(msg)
                run_summary_lines.append(msg)
            elif abs(hh.group_mean_bias) <= 0.5 and cc.group_mean_bias < -0.5:
                msg = "Hot17 members better capture the observed heat amplitude, while Cold17 members drive the ensemble cold bias."
                print(msg)
                run_summary_lines.append(msg)

    with open(os.path.join(QC_DIR, "qc_run_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(run_summary_lines) + "\n")


main()

# =============================================================================
# [Independent Cell] Fig.2b: lead-time dependence summary for Total-period Tmax error
# 依赖提示:
# 本 Cell 只读取当前 Fig.2 主流程已经输出的 CSV 结果，不重新运行 Tmax 数据读取或 S2S 处理。
# - Bias/RMSE: tables/tmax_stage_summary_metrics.csv 中 Total period 的 ensmean_bias / ensmean_rmse
# - Spread: tables/tmax_member_hotcold_group_by_stage.csv 中每个成员 Total-period stage-mean Tmax 的 std(number)
# =============================================================================


def _leadtime_fig_find_column(df, candidates, label, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Cannot find {label} column. candidates={candidates}; columns={list(df.columns)}")
    return None


def build_fig2b_leadtime_total_summary(
    stage_summary_csv=os.path.join(TAB_DIR, "tmax_stage_summary_metrics.csv"),
    member_stage_csv=os.path.join(TAB_DIR, "tmax_member_hotcold_group_by_stage.csv"),
):
    stage_df = pd.read_csv(stage_summary_csv)
    member_df = pd.read_csv(member_stage_csv)

    init_col = _leadtime_fig_find_column(stage_df, ["init", "init_date", "requested_init"], "init")
    stage_col = _leadtime_fig_find_column(stage_df, ["stage", "stage_label", "period"], "stage")
    bias_col = _leadtime_fig_find_column(stage_df, ["ensmean_bias", "bias", "ensemble_mean_bias"], "bias")
    rmse_col = _leadtime_fig_find_column(stage_df, ["ensmean_rmse", "rmse", "ensemble_mean_rmse"], "rmse")
    ndays_col = _leadtime_fig_find_column(stage_df, ["n_days", "ndays", "num_days"], "n_days", required=False)

    mem_init_col = _leadtime_fig_find_column(member_df, ["init", "init_date", "requested_init"], "member init")
    mem_stage_col = _leadtime_fig_find_column(member_df, ["stage", "stage_label", "period"], "member stage")
    mem_col = _leadtime_fig_find_column(member_df, ["member", "number"], "member")
    stage_mean_col = _leadtime_fig_find_column(member_df, ["member_stage_mean_Tmax", "stage_mean_tmax", "member_stage_mean_tmax"], "member stage-mean Tmax")

    stage_df = stage_df.copy()
    member_df = member_df.copy()
    stage_df["_stage_norm"] = stage_df[stage_col].astype(str).str.lower().str.replace("_", "-")
    member_df["_stage_norm"] = member_df[mem_stage_col].astype(str).str.lower().str.replace("_", "-")

    total_stage_names = {"total", "total-period", "total period"}
    stage_total = stage_df[stage_df["_stage_norm"].isin(total_stage_names)]
    member_total = member_df[member_df["_stage_norm"].isin(total_stage_names)]

    records = []
    for init in INITS:
        srow = stage_total[stage_total[init_col].astype(str) == init]
        mrow = member_total[member_total[mem_init_col].astype(str) == init]
        status = "success"
        error_messages = []

        if srow.empty:
            bias = np.nan
            rmse = np.nan
            n_days = np.nan
            status = "failed"
            error_messages.append("missing Total row in tmax_stage_summary_metrics.csv")
        else:
            bias = float(srow.iloc[0][bias_col])
            rmse = float(srow.iloc[0][rmse_col])
            n_days = int(srow.iloc[0][ndays_col]) if ndays_col is not None and pd.notna(srow.iloc[0][ndays_col]) else EXPECTED_NDAYS["Total"]

        if mrow.empty:
            spread = np.nan
            n_members = 0
            status = "failed"
            error_messages.append("missing Total member rows in tmax_member_hotcold_group_by_stage.csv")
        else:
            vals = mrow.sort_values(mem_col)[stage_mean_col].astype(float).values
            spread = float(np.std(vals, ddof=0))
            n_members = int(len(vals))
            if n_members != 51:
                status = "failed"
                error_messages.append(f"n_members={n_members} != 51")

        records.append({
            "init_date": init,
            "stage": "Total",
            "bias": bias,
            "rmse": rmse,
            "n_days": n_days,
            "ensemble_spread_std": spread,
            "n_members": n_members,
            "status": status,
            "error_message": "; ".join(error_messages),
            "notes": "ensemble_spread_std is retained only for auxiliary QC and is not shown in the main Fig.10 after advisor revision.",
        })

    out = pd.DataFrame(records)
    print(f"[Leadtime Fig QC] Bias/RMSE source: {stage_summary_csv}")
    print(f"[Leadtime Fig QC] Auxiliary spread source: {member_stage_csv}")
    print("[Leadtime Fig QC] Auxiliary spread definition: std across 51 members of each member's Total-period stage-mean regional Tmax (ddof=0); spread is not shown in revised main Fig.10.")
    for _, row in out.iterrows():
        print(f"[Leadtime Fig QC] init={row['init_date']} | status={row['status']} | "
              f"Bias={row['bias']:+.2f} | RMSE={row['rmse']:.2f} | "
              f"Spread(aux)={row['ensemble_spread_std']:.2f} | n={int(row['n_members'])} | {row['error_message']}")
    return out


def audit_fig10_rmse_definition(
    daily_error_csv=os.path.join(TAB_DIR, "tmax_daily_error_timeseries.csv"),
    stage_summary_csv=os.path.join(TAB_DIR, "tmax_stage_summary_metrics.csv"),
):
    daily_df = pd.read_csv(daily_error_csv)
    stage_df = pd.read_csv(stage_summary_csv)

    init_col = _leadtime_fig_find_column(daily_df, ["init", "init_date", "requested_init"], "daily init")
    date_col = _leadtime_fig_find_column(daily_df, ["date", "time", "valid_date"], "date")
    err_col = _leadtime_fig_find_column(daily_df, ["daily_error", "ensmean_error", "error"], "daily error")
    pred_col = _leadtime_fig_find_column(daily_df, ["S2S_ensmean_Tmax", "s2s_ensmean_tmax", "ensmean_tmax"], "S2S ensemble-mean Tmax")
    obs_col = _leadtime_fig_find_column(daily_df, ["ERA5_Tmax", "era5_tmax", "obs_tmax"], "ERA5 Tmax")

    stage_init_col = _leadtime_fig_find_column(stage_df, ["init", "init_date", "requested_init"], "stage init")
    stage_col = _leadtime_fig_find_column(stage_df, ["stage", "stage_label", "period"], "stage")
    bias_col = _leadtime_fig_find_column(stage_df, ["ensmean_bias", "bias", "ensemble_mean_bias"], "stage bias")
    rmse_col = _leadtime_fig_find_column(stage_df, ["ensmean_rmse", "rmse", "ensemble_mean_rmse"], "stage RMSE")

    daily_df = daily_df.copy()
    stage_df = stage_df.copy()
    daily_df[date_col] = pd.to_datetime(daily_df[date_col])
    stage_df["_stage_norm"] = stage_df[stage_col].astype(str).str.lower().str.replace("_", "-")
    stage_total = stage_df[stage_df["_stage_norm"].isin({"total", "total-period", "total period"})]

    records = []
    notes = ("RMSE is computed from daily ensemble-mean errors against ERA5: "
             "sqrt(mean_t((S2S_ensmean_t - ERA5_t)^2)). It is not the absolute "
             "difference between stage-mean S2S Tmax and stage-mean ERA5 Tmax. "
             "Signed Bias can be smaller when positive and negative daily errors compensate, "
             "whereas RMSE measures daily-error magnitude and is not subject to sign cancellation.")
    for init in INITS:
        dsub = daily_df[(daily_df[init_col].astype(str) == init) &
                        (daily_df[date_col] >= STAGES["Total"][0]) &
                        (daily_df[date_col] <= STAGES["Total"][1])].sort_values(date_col)
        srow = stage_total[stage_total[stage_init_col].astype(str) == init]
        if dsub.empty or srow.empty:
            records.append({
                "init_date": init,
                "stage": "Total",
                "n_days": int(len(dsub)),
                "bias_from_stage_summary": np.nan,
                "bias_recomputed_from_daily_error": np.nan,
                "rmse_from_stage_summary": np.nan,
                "rmse_recomputed_from_daily_error": np.nan,
                "abs_stage_mean_difference": np.nan,
                "rmse_minus_abs_stage_mean_difference": np.nan,
                "daily_error_std": np.nan,
                "rmse_definition_pass": False,
                "notes": "missing daily Total rows or stage summary row; " + notes,
            })
            continue
        err = dsub[err_col].astype(float).values
        stage_mean_difference = float(dsub[pred_col].astype(float).mean() - dsub[obs_col].astype(float).mean())
        rmse_recomputed = float(np.sqrt(np.mean(err ** 2)))
        rmse_from_summary = float(srow.iloc[0][rmse_col])
        pass_check = bool(abs(rmse_from_summary - rmse_recomputed) < 1e-6)
        records.append({
            "init_date": init,
            "stage": "Total",
            "n_days": int(len(dsub)),
            "bias_from_stage_summary": float(srow.iloc[0][bias_col]),
            "bias_recomputed_from_daily_error": float(np.mean(err)),
            "rmse_from_stage_summary": rmse_from_summary,
            "rmse_recomputed_from_daily_error": rmse_recomputed,
            "abs_stage_mean_difference": float(abs(stage_mean_difference)),
            "rmse_minus_abs_stage_mean_difference": float(rmse_recomputed - abs(stage_mean_difference)),
            "daily_error_std": float(np.std(err, ddof=0)),
            "rmse_definition_pass": pass_check,
            "notes": notes,
        })

    audit_df = pd.DataFrame(records)
    all_pass = bool(audit_df["rmse_definition_pass"].all())
    print("[Fig10 RMSE QC] RMSE definition: daily error -> square -> time mean -> sqrt")
    for _, row in audit_df.iterrows():
        print(f"[Fig10 RMSE QC] init={row['init_date']} | Bias={row['bias_recomputed_from_daily_error']:+.2f} | "
              f"RMSE={row['rmse_recomputed_from_daily_error']:.2f} | "
              f"abs(stage-mean diff)={row['abs_stage_mean_difference']:.2f} | "
              f"daily_error_std={row['daily_error_std']:.2f} | PASS={row['rmse_definition_pass']}")
    if not all_pass:
        print("[Fig10 RMSE QC][Warning] At least one Total-period RMSE definition check failed.")
    return audit_df


def plot_fig2b_leadtime_total_summary(summary_df, out_dir=PAPER_FIG_DIR):
    with plt.rc_context(PAPER_RC_PARAMS):
        os.makedirs(out_dir, exist_ok=True)
        labels = [pd.Timestamp(init).strftime("%m-%d") for init in summary_df["init_date"]]
        x = np.arange(len(summary_df))

        fig, axs = plt.subplots(1, 2, figsize=(PAPER_FIG_WIDTH_IN, 3.8), sharex=True, constrained_layout=False)
        fig.suptitle("Tmax forecast error by initialization date", fontsize=PAPER_STYLE["suptitle_size"], fontweight="normal", y=0.96)

        panel_specs = [
            ("bias", "Signed bias against ERA5 (°C)", "(a) Signed bias", PAPER_STYLE["metric_colors"]["Bias"]),
            ("rmse", "RMSE against ERA5 (°C)", "(b) RMSE", PAPER_STYLE["metric_colors"]["RMSE"]),
        ]

        for ax, (col, ylabel, title, color) in zip(axs, panel_specs):
            vals = summary_df[col].astype(float).values
            bars = ax.bar(x, vals, width=0.64, color=color, alpha=0.82, edgecolor="0.25", linewidth=0.6)
            if col == "bias":
                ax.axhline(0, color="black", lw=1.0)
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) > 0:
                ymin = min(0.0, float(np.nanmin(finite_vals))) if col == "bias" else 0.0
                ymax = max(0.0, float(np.nanmax(finite_vals)))
                pad = max(0.2, 0.18 * (ymax - ymin if ymax > ymin else abs(ymax) + 1.0))
                ax.set_ylim(ymin - (pad if col == "bias" else 0.0), ymax + pad)
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    va = "bottom" if val >= 0 else "top"
                    yoff = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * (1 if val >= 0 else -1)
                    label = _paper_format_signed(val) if col == "bias" else f"{val:.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, val + yoff, label, ha="center", va=va, fontsize=PAPER_STYLE["annotation_size"])
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "NA", ha="center", va="bottom", fontsize=PAPER_STYLE["annotation_size"], color="crimson")
            ax.set_title(title, loc="left", fontsize=PAPER_STYLE["panel_title_size"])
            ax.set_ylabel(ylabel, fontsize=PAPER_STYLE["axis_label_size"])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=PAPER_STYLE["tick_label_size"])
            _paper_apply_vertical_bar_style(ax)

        fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.20, wspace=0.28)
        out_png = os.path.join(out_dir, "fig10_leadtime_total_tmax_error_bias_rmse_jgrstyle.png")
        out_pdf = os.path.join(out_dir, "fig10_leadtime_total_tmax_error_bias_rmse_jgrstyle.pdf")
        _save_fixed_canvas_paper_figure(fig, out_png, out_pdf)
        plt.close(fig)
        return out_png, out_pdf

def run_fig2b_leadtime_total_summary_cell():
    summary_df = build_fig2b_leadtime_total_summary()
    audit_df = audit_fig10_rmse_definition()
    main_cols = ["init_date", "stage", "bias", "rmse", "n_days", "status", "error_message"]
    main_df = summary_df[main_cols].copy()

    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    out_csv = os.path.join(PAPER_FIG_DIR, "fig2b_leadtime_total_tmax_error_bias_rmse.csv")
    aux_spread_csv = os.path.join(PAPER_FIG_DIR, "fig2b_leadtime_total_tmax_spread_auxiliary.csv")
    audit_csv = os.path.join(PAPER_FIG_DIR, "fig10_rmse_definition_audit.csv")
    main_df.to_csv(out_csv, index=False)
    summary_df[["init_date", "stage", "ensemble_spread_std", "n_members", "status", "error_message", "notes"]].to_csv(aux_spread_csv, index=False)
    audit_df.to_csv(audit_csv, index=False)

    out_png, out_pdf = plot_fig2b_leadtime_total_summary(main_df, out_dir=PAPER_FIG_DIR)
    all_pass = bool(audit_df["rmse_definition_pass"].all())
    print(f"[Fig10 RMSE QC] all Total-period RMSE definition checks passed: {all_pass}")
    print("[Fig10 RMSE QC] RMSE definition = daily error -> square -> time mean -> sqrt")
    print("[Fig10 Revision] panel (c) ensemble spread removed from main figure per advisor suggestion")
    print("[Fig10 Revision] main figure now shows only signed Bias and RMSE with fixed 8.6-inch JGR-style canvas")
    print("[Fig10 Note] Bias and RMSE are evaluated against ERA5 daily Tmax over the Total period (14–24 June 2023). Bias is signed mean daily error; RMSE is sqrt(mean daily squared error). Signed Bias can be smaller when positive and negative daily errors compensate, whereas RMSE measures daily-error magnitude.")
    print(f"[Fig10 Output] PNG: {out_png}")
    print(f"[Fig10 Output] PDF: {out_pdf}")
    print(f"[Fig10 Output] CSV: {out_csv}")
    print(f"[Fig10 Output] Spread auxiliary CSV: {aux_spread_csv}")
    print(f"[Fig10 Output] RMSE audit CSV: {audit_csv}")
    return {"png": out_png, "pdf": out_pdf, "csv": out_csv, "spread_auxiliary_csv": aux_spread_csv, "rmse_audit_csv": audit_csv, "summary": main_df, "audit": audit_df}


run_fig2b_leadtime_total_summary_cell()
