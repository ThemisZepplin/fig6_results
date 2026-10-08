# =============================================================================
# [Standalone Cell] Final JGR Fig.2 / Fig.S2 / Fig.9 plotting-only renderer
#
# This cell reads existing CSV outputs from the upstream Tmax evaluation workflow.
# Raw GRIB/NetCDF read = False; upstream scientific calculation = False;
# kernel-state dependency = False.
# =============================================================================

import os
import struct
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# -------------------------
# Standalone path config
# -------------------------
OUT_ROOT = Path("/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit")
TAB_DIR = OUT_ROOT / "tables"
PAPER_FIG_INPUT_DIR = Path(
    "/data1/huangy/fig6/NC/v12/paper_figures.0/fig2/fig2a_tmax_error_divergence"
)
PAPER_FIG_DIR = Path("/data1/huangy/fig6/NC/v12")

FIG2_PNG = PAPER_FIG_DIR / "fig2_tmax_error_member_divergence_jgr_final.png"
FIG2_PDF = PAPER_FIG_DIR / "fig2_tmax_error_member_divergence_jgr_final.pdf"
FIGS2_PNG = PAPER_FIG_DIR / "figS2_tmax_error_member_divergence_2023-06-08_jgr_final.png"
FIGS2_PDF = PAPER_FIG_DIR / "figS2_tmax_error_member_divergence_2023-06-08_jgr_final.pdf"
FIG9_PNG = PAPER_FIG_DIR / "fig9_tmax_error_by_initialization_jgr_final.png"
FIG9_PDF = PAPER_FIG_DIR / "fig9_tmax_error_by_initialization_jgr_final.pdf"

FIG2_DAILY_CSV = TAB_DIR / "tmax_daily_error_timeseries.csv"
FIG2_MEMBER_CSV = TAB_DIR / "tmax_member_error_by_stage.csv"
FIG2_STAGE_CSV = TAB_DIR / "tmax_stage_summary_metrics.csv"
# Legacy upstream filenames are retained for data compatibility; these inputs now
# support final manuscript Fig.9 and are never overwritten by this renderer.
FIG9_SUMMARY_CSV = PAPER_FIG_INPUT_DIR / "fig2b_leadtime_total_tmax_error_bias_rmse.csv"
FIG9_AUDIT_CSV = PAPER_FIG_INPUT_DIR / "fig10_rmse_definition_audit.csv"

PAPER_FIG2_INITS = ["2023-06-12", "2023-06-08"]
FIG9_INIT_ORDER = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]
TARGET_DATES = pd.date_range("2023-06-14", "2023-06-24", freq="D")
EXPECTED_NDAYS = {"Stage-I": 4, "Total": 11}

# -------------------------
# Final JGR manuscript style (local rc_context only)
# -------------------------
FIG_WIDTH_IN = 6.5
FIG2_HEIGHT_IN = 5.1
FIG9_HEIGHT_IN = 3.0
FIG_DPI = 600
FONT_PANEL_LETTER = 10.8
FONT_PANEL_TITLE = 9.6
FONT_AXIS_LABEL = 9.5
FONT_TICK = 8.7
FONT_LEGEND = 8.4
FONT_ANNOTATION = 8.3
BIAS_COLOR = "#5B8FD9"
RMSE_COLOR = "#80B36A"
BAR_ALPHA = 0.86
EDGE_COLOR = "#4A4A4A"
EDGE_LW = 0.6
GRID_LW = 0.45
PAPER_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _format_signed_without_plus(value, digits=2):
    return f"{value:.{digits}f}".replace("-", "−")


def _apply_axis_style(ax, grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")
    ax.tick_params(axis="both", which="major", width=0.8, length=3.2, labelsize=FONT_TICK)
    ax.xaxis.label.set_size(FONT_AXIS_LABEL)
    ax.yaxis.label.set_size(FONT_AXIS_LABEL)
    if grid_axis:
        ax.grid(
            True,
            which="major",
            axis=grid_axis,
            linestyle=":",
            linewidth=GRID_LW,
            alpha=0.18 if grid_axis == "both" else 0.14,
        )
    ax.set_axisbelow(True)


def _add_panel_heading(ax, letter, title):
    """Place a consistent bold panel letter and regular short title above an axis."""
    ax.text(
        0.0,
        1.025,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_PANEL_LETTER,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )
    ax.text(
        0.105,
        1.025,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_PANEL_TITLE,
        fontweight="normal",
        clip_on=False,
        zorder=20,
    )


def _read_png_size(path):
    try:
        with Path(path).open("rb") as f:
            header = f.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG file")
        return struct.unpack(">II", header[16:24])
    except Exception as exc:
        raise RuntimeError(f"Unable to read PNG dimensions for fixed-canvas QC: {path}: {exc}") from exc


def _save_fixed_canvas(fig, png_path, pdf_path):
    fig.savefig(png_path, dpi=FIG_DPI, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    width_in, height_in = (float(v) for v in fig.get_size_inches())
    width_px, height_px = _read_png_size(png_path)
    expected_width_px = int(round(FIG_WIDTH_IN * FIG_DPI))
    if not np.isclose(width_in, FIG_WIDTH_IN):
        raise RuntimeError(f"Fixed figure width is {width_in:.3f} in; expected {FIG_WIDTH_IN:.3f} in")
    if width_px != expected_width_px:
        raise RuntimeError(f"PNG width is {width_px}px; expected {expected_width_px}px")
    print(
        f"[Fixed-canvas QC] width={width_in:.2f} in | height={height_in:.2f} in | "
        f"PNG={width_px}x{height_px}px | dpi={FIG_DPI} | tight bbox=False"
    )
    return {
        "figure_width_in": width_in,
        "figure_height_in": height_in,
        "dpi": FIG_DPI,
        "png_width_px": width_px,
        "png_height_px": height_px,
        "expected_png_width_px": expected_width_px,
        "bbox_tight_used": False,
        "suptitle_present": False,
    }


# -------------------------
# Validation helpers
# -------------------------
def _require_files(file_map):
    missing = [(label, Path(path)) for label, path in file_map.items() if not Path(path).exists()]
    if missing:
        lines = ["Missing standalone paper-figure inputs:"]
        lines.extend([f"- {label}: {path}" for label, path in missing])
        lines.extend([
            "",
            "Run tmax_error_stage_eval_cell.py once in the full science-processing workflow to create these CSV files.",
            "The plotting-only cell will not rerun upstream calculations.",
        ])
        raise FileNotFoundError("\n".join(lines))


def _require_columns(df, required, table_name):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}; available={list(df.columns)}")


def _assert_finite(series, label):
    values = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in {label}: {series.tolist()}")


def _parse_bool_strict(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean value safely: {value!r}")


def _validate_fig2_inputs(daily_df, member_df, stage_df):
    _require_columns(
        daily_df,
        ["init", "date", "ERA5_Tmax", "S2S_ensmean_Tmax", "ens_p05", "ens_p25", "ens_p75", "ens_p95", "daily_error"],
        "Fig.2 daily table",
    )
    _require_columns(member_df, ["init", "stage", "member", "rmse"], "Fig.2 member table")
    _require_columns(stage_df, ["init", "stage", "ensmean_bias", "ensmean_rmse", "n_days"], "Fig.2 stage summary")

    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    member_df = member_df.copy()
    stage_df = stage_df.copy()
    for init in PAPER_FIG2_INITS:
        dsub = daily_df[daily_df["init"].astype(str) == init].copy()
        if len(dsub) != 11:
            raise ValueError(f"Fig.2 daily table init={init} expected 11 rows, got {len(dsub)}")
        if dsub["date"].duplicated().any():
            duplicates = dsub.loc[dsub["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Fig.2 daily table has duplicated dates for init={init}: {duplicates}")
        got_dates = pd.DatetimeIndex(dsub["date"].dt.normalize().sort_values())
        if not got_dates.equals(TARGET_DATES):
            raise ValueError(
                f"Fig.2 daily dates for init={init} do not exactly equal 2023-06-14–24: "
                f"{got_dates.strftime('%Y-%m-%d').tolist()}"
            )
        _assert_finite(
            dsub[["ERA5_Tmax", "S2S_ensmean_Tmax", "ens_p05", "ens_p25", "ens_p75", "ens_p95", "daily_error"]].stack(),
            f"Fig.2 daily values init={init}",
        )

        for stage, expected_n_days in EXPECTED_NDAYS.items():
            msub = member_df[(member_df["init"].astype(str) == init) & (member_df["stage"].astype(str) == stage)]
            if len(msub) != 51 or msub["member"].nunique() != 51:
                raise ValueError(f"Fig.2 member table init={init} stage={stage} requires 51 unique members")
            if msub["member"].duplicated().any():
                raise ValueError(f"Fig.2 member table has duplicated member IDs for init={init} stage={stage}")
            _assert_finite(msub["rmse"], f"Fig.2 member RMSE init={init} stage={stage}")

            ssub = stage_df[(stage_df["init"].astype(str) == init) & (stage_df["stage"].astype(str) == stage)]
            if len(ssub) != 1:
                raise ValueError(f"Fig.2 stage summary init={init} stage={stage} expected 1 row, got {len(ssub)}")
            if int(ssub.iloc[0]["n_days"]) != expected_n_days:
                raise ValueError(
                    f"Fig.2 stage summary init={init} stage={stage} expected n_days={expected_n_days}, "
                    f"got {ssub.iloc[0]['n_days']}"
                )
            _assert_finite(ssub[["ensmean_bias", "ensmean_rmse"]].iloc[0], f"Fig.2 stage metrics init={init} stage={stage}")
    return daily_df, member_df, stage_df


def _validate_fig9_inputs(summary_df, audit_df):
    _require_columns(summary_df, ["init_date", "stage", "bias", "rmse", "n_days", "status", "error_message"], "Fig.9 summary")
    _require_columns(audit_df, ["init_date", "stage", "n_days", "rmse_definition_pass"], "Fig.9 RMSE audit")

    summary_records = []
    audit_records = []
    for init in FIG9_INIT_ORDER:
        srows = summary_df[summary_df["init_date"].astype(str) == init]
        if len(srows) != 1:
            raise ValueError(f"Fig.9 summary expected exactly one row for init={init}, got {len(srows)}")
        srow = srows.iloc[0]
        if str(srow["stage"]) != "Total":
            raise ValueError(f"Fig.9 summary init={init} stage must be Total, got {srow['stage']}")
        if str(srow["status"]).strip().lower() != "success":
            raise ValueError(f"Fig.9 summary init={init} status must be success: {srow['status']} | {srow['error_message']}")
        if int(srow["n_days"]) != 11:
            raise ValueError(f"Fig.9 summary init={init} n_days must be 11, got {srow['n_days']}")
        _assert_finite(pd.Series([srow["bias"], srow["rmse"]]), f"Fig.9 summary bias/RMSE init={init}")
        summary_records.append(srow)

        arows = audit_df[audit_df["init_date"].astype(str) == init]
        if len(arows) != 1:
            raise ValueError(f"Fig.9 audit expected exactly one row for init={init}, got {len(arows)}")
        arow = arows.iloc[0]
        if str(arow["stage"]) != "Total":
            raise ValueError(f"Fig.9 audit init={init} stage must be Total, got {arow['stage']}")
        if int(arow["n_days"]) != 11:
            raise ValueError(f"Fig.9 audit init={init} n_days must be 11, got {arow['n_days']}")
        if not _parse_bool_strict(arow["rmse_definition_pass"]):
            raise ValueError(f"Fig.9 audit init={init} rmse_definition_pass is not True: {arow['rmse_definition_pass']}")
        audit_records.append(arow)

    if len(summary_df[summary_df["init_date"].astype(str).isin(FIG9_INIT_ORDER)]) != 4:
        raise ValueError("Fig.9 summary contains duplicate requested-init rows")
    if len(audit_df[audit_df["init_date"].astype(str).isin(FIG9_INIT_ORDER)]) != 4:
        raise ValueError("Fig.9 audit contains duplicate requested-init rows")
    return pd.DataFrame(summary_records).reset_index(drop=True), pd.DataFrame(audit_records).reset_index(drop=True)


# -------------------------
# Caption and QC writers
# -------------------------
def _write_text(path, text):
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_fig2_caption(path):
    caption = (
        "Figure 2. Evolution and verification of regional maximum 2-m temperature (Tmax) "
        "for the S2S forecast initialized on 12 June 2023. (a) Daily NCHN regional-mean "
        "ERA5 Tmax and the 51-member S2S ensemble mean for 14–24 June, with the 25th–75th "
        "and 5th–95th ensemble-percentile ranges shown by shading. Dates follow the upstream "
        "Beijing-time (UTC+8) daily processing, the vertical line marks the end of Stage-I "
        "(14–17 June), and the annotations report the RMSE of the ensemble-mean forecast "
        "against ERA5 for Stage-I and the Total period (14–24 June). (b) Signed Bias of the "
        "ensemble-mean forecast against ERA5 for Stage-I and the Total period, where Bias is "
        "the temporal mean of the daily ensemble-mean error. (c) Distributions of the 51 "
        "member-wise RMSE values for Stage-I and the Total period; gray points denote members, "
        "boxes summarize their distributions, and black diamonds denote the RMSE of the "
        "ensemble-mean forecast rather than the mean of member-wise RMSE values."
    )
    _write_text(path, caption)


def _write_fig9_caption(path):
    caption = (
        "Figure 9. Dependence of Total-period Tmax verification metrics on S2S initialization "
        "date. (a) Signed Bias and (b) root-mean-square error (RMSE) of the ensemble-mean "
        "regional Tmax forecast against ERA5 over 14–24 June 2023 for forecasts initialized "
        "on 1, 5, 8, and 12 June. Each metric uses N = 11 daily values. With daily error "
        "e_t defined as ensemble-mean S2S Tmax minus ERA5 Tmax, Bias is (1/N) sum_t e_t and "
        "RMSE is sqrt[(1/N) sum_t e_t^2]; thus, daily errors are squared before temporal "
        "averaging in the RMSE calculation."
    )
    _write_text(path, caption)


def _write_qc(path, records):
    _write_text(path, "\n".join(f"{key} = {value}" for key, value in records.items()))


# -------------------------
# Standalone CSV-driven renderers
# -------------------------
def render_fig2_from_csv(daily_df, member_df, stage_df, init, out_dir=PAPER_FIG_DIR):
    os.makedirs(out_dir, exist_ok=True)
    stages = ["Stage-I", "Total"]
    sub = daily_df[daily_df["init"].astype(str) == init].copy().sort_values("date")
    stage_rows = {
        stage: stage_df[(stage_df["init"].astype(str) == init) & (stage_df["stage"].astype(str) == stage)].iloc[0]
        for stage in stages
    }

    with plt.rc_context(PAPER_RC_PARAMS):
        fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG2_HEIGHT_IN), constrained_layout=False)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.66, wspace=0.34)
        ax_ts = fig.add_subplot(gs[0, :])
        ax_bias = fig.add_subplot(gs[1, 0])
        ax_rmse = fig.add_subplot(gs[1, 1])

        ax_ts.axvline(pd.Timestamp("2023-06-17 12:00"), color="0.25", lw=0.9, zorder=2)
        p0595 = ax_ts.fill_between(
            sub["date"], sub["ens_p05"], sub["ens_p95"], color="deepskyblue", alpha=0.14,
            lw=0, label="S2S p05–p95", zorder=1,
        )
        p2575 = ax_ts.fill_between(
            sub["date"], sub["ens_p25"], sub["ens_p75"], color="deepskyblue", alpha=0.30,
            lw=0, label="S2S p25–p75", zorder=1,
        )
        era5_line, = ax_ts.plot(
            sub["date"], sub["ERA5_Tmax"], color="black", lw=1.75, marker="o", ms=4.1,
            label="ERA5 Obs", zorder=3,
        )
        s2s_line, = ax_ts.plot(
            sub["date"], sub["S2S_ensmean_Tmax"], color="royalblue", lw=1.65, marker="s", ms=4.1,
            label="S2S ensemble mean", zorder=3,
        )
        annotation = "\n".join(
            f"{stage} RMSE = {float(stage_rows[stage]['ensmean_rmse']):.2f}°C" for stage in stages
        )
        ax_ts.text(
            0.015, 0.06, annotation, transform=ax_ts.transAxes, ha="left", va="bottom",
            fontsize=FONT_ANNOTATION,
            bbox=dict(facecolor="white", alpha=0.74, edgecolor="none", boxstyle="round,pad=0.20"),
            zorder=15,
        )
        _add_panel_heading(ax_ts, "a", "Tmax evolution")
        ax_ts.set_ylabel("Daily Tmax (°C)", fontsize=FONT_AXIS_LABEL)
        ax_ts.set_xlim(pd.Timestamp("2023-06-13 12:00"), pd.Timestamp("2023-06-24 12:00"))
        ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax_ts.get_xticklabels(), rotation=30, ha="right", fontsize=FONT_TICK)
        _apply_axis_style(ax_ts, grid_axis="both")
        ax_ts.legend(
            [era5_line, s2s_line, p2575, p0595],
            ["ERA5 Obs", "S2S ensemble mean", "S2S p25–p75", "S2S p05–p95"],
            loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=4,
            fontsize=FONT_LEGEND, frameon=False, borderaxespad=0.0,
            handlelength=1.55, columnspacing=0.9, handletextpad=0.45,
        )

        x = np.arange(len(stages))
        bias_values = np.array([float(stage_rows[stage]["ensmean_bias"]) for stage in stages])
        bars = ax_bias.bar(
            x, bias_values, width=0.62, color=BIAS_COLOR, alpha=BAR_ALPHA,
            edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
        )
        ax_bias.axhline(0, color="0.15", lw=0.9)
        bias_low = min(0.0, float(np.min(bias_values)))
        bias_high = max(0.0, float(np.max(bias_values)))
        span = max(bias_high - bias_low, 0.5)
        pad = 0.16 * span
        ax_bias.set_ylim(bias_low - pad, bias_high + pad)
        label_offset = 0.035 * (ax_bias.get_ylim()[1] - ax_bias.get_ylim()[0])
        for bar, value in zip(bars, bias_values):
            ax_bias.text(
                bar.get_x() + bar.get_width() / 2, value + (label_offset if value >= 0 else -label_offset),
                _format_signed_without_plus(value), ha="center", va="bottom" if value >= 0 else "top",
                fontsize=FONT_ANNOTATION,
            )
        ax_bias.set_xticks(x, stages)
        ax_bias.set_ylabel("Ensemble-mean bias (°C)", fontsize=FONT_AXIS_LABEL)
        _apply_axis_style(ax_bias, grid_axis="y")
        _add_panel_heading(ax_bias, "b", "Ensemble-mean bias")

        rmse_data = [
            member_df[(member_df["init"].astype(str) == init) & (member_df["stage"].astype(str) == stage)]
            .sort_values("member")["rmse"].astype(float).values
            for stage in stages
        ]
        boxplot = ax_rmse.boxplot(rmse_data, patch_artist=True, widths=0.55, showfliers=False)
        for patch in boxplot["boxes"]:
            patch.set_facecolor(RMSE_COLOR)
            patch.set_alpha(0.50)
            patch.set_edgecolor(EDGE_COLOR)
            patch.set_linewidth(0.75)
        for key in ["medians", "whiskers", "caps"]:
            for artist in boxplot[key]:
                artist.set_color(EDGE_COLOR)
                artist.set_linewidth(0.75)
        rng = np.random.default_rng(20240612)
        for index, (stage, values) in enumerate(zip(stages, rmse_data), 1):
            jitter = rng.normal(0, 0.032, size=len(values))
            ax_rmse.scatter(
                np.full(len(values), index) + jitter, values, s=8, color="0.35",
                alpha=0.30, linewidths=0, zorder=3,
            )
            ax_rmse.scatter(
                index, float(stage_rows[stage]["ensmean_rmse"]), marker="D", s=28,
                color="black", edgecolors="none", zorder=5,
            )
        ax_rmse.set_xticks(np.arange(1, len(stages) + 1), stages)
        ax_rmse.set_ylabel("Member-wise RMSE (°C)", fontsize=FONT_AXIS_LABEL)
        _apply_axis_style(ax_rmse, grid_axis="y")
        _add_panel_heading(ax_rmse, "c", "Member RMSE distribution")
        diamond = Line2D([], [], marker="D", linestyle="None", color="black", markersize=4.2)
        ax_rmse.legend(
            [diamond], ["Ensemble-mean RMSE"], loc="upper left",
            bbox_to_anchor=(0.015, 0.95), fontsize=FONT_LEGEND - 0.7,
            frameon=False, handletextpad=0.30, borderaxespad=0.0,
            labelspacing=0.25, markerscale=0.9,
        )

        fig.subplots_adjust(left=0.105, right=0.975, top=0.94, bottom=0.105)
        if init == "2023-06-12":
            png_path, pdf_path = FIG2_PNG, FIG2_PDF
            figure_label = "Fig.2"
        elif init == "2023-06-08":
            png_path, pdf_path = FIGS2_PNG, FIGS2_PDF
            figure_label = "Fig.S2"
        else:
            raise ValueError(f"Final Fig.2 renderer does not support init={init}")
        png_path = Path(out_dir) / png_path.name
        pdf_path = Path(out_dir) / pdf_path.name
        canvas_qc = _save_fixed_canvas(fig, png_path, pdf_path)
        plt.close(fig)

    print(f"[{figure_label} output] PNG: {png_path}")
    print(f"[{figure_label} output] PDF: {pdf_path}")
    return {
        "png": png_path,
        "pdf": pdf_path,
        "canvas_qc": canvas_qc,
        "init": init,
        "stage_rows": stage_rows,
        "member_counts": {stage: len(values) for stage, values in zip(stages, rmse_data)},
    }


def render_fig9_from_csv(summary_df, audit_df, out_dir=PAPER_FIG_DIR):
    summary_df, audit_df = _validate_fig9_inputs(summary_df, audit_df)
    os.makedirs(out_dir, exist_ok=True)
    labels = [pd.Timestamp(init).strftime("%m-%d") for init in summary_df["init_date"]]
    x = np.arange(len(summary_df))

    with plt.rc_context(PAPER_RC_PARAMS):
        fig, axes = plt.subplots(
            1, 2, figsize=(FIG_WIDTH_IN, FIG9_HEIGHT_IN), sharex=True, constrained_layout=False,
        )
        panel_specs = [
            (axes[0], "bias", "a", "Signed bias", "Ensemble-mean bias (°C)", BIAS_COLOR),
            (axes[1], "rmse", "b", "RMSE", "RMSE (°C)", RMSE_COLOR),
        ]
        for ax, column, letter, title, ylabel, color in panel_specs:
            values = summary_df[column].astype(float).values
            bars = ax.bar(
                x, values, width=0.64, color=color, alpha=BAR_ALPHA,
                edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
            )
            if column == "bias":
                ax.axhline(0, color="0.15", lw=0.9)
                lower = min(0.0, float(np.min(values)))
                upper = max(0.0, float(np.max(values)))
                span = max(upper - lower, 0.5)
                ax.set_ylim(lower - 0.18 * span, upper + 0.16 * span)
            else:
                upper = float(np.max(values))
                ax.set_ylim(0.0, upper * 1.20 if upper > 0 else 1.0)
            label_offset = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            for bar, value in zip(bars, values):
                offset = label_offset if value >= 0 else -label_offset
                text = _format_signed_without_plus(value) if column == "bias" else f"{value:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + offset, text,
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=FONT_ANNOTATION,
                )
            ax.set_xticks(x, labels)
            ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
            _apply_axis_style(ax, grid_axis="y")
            _add_panel_heading(ax, letter, title)

        fig.subplots_adjust(left=0.105, right=0.975, top=0.92, bottom=0.18, wspace=0.34)
        png_path = Path(out_dir) / FIG9_PNG.name
        pdf_path = Path(out_dir) / FIG9_PDF.name
        canvas_qc = _save_fixed_canvas(fig, png_path, pdf_path)
        plt.close(fig)

    print(f"[Fig.9 output] PNG: {png_path}")
    print(f"[Fig.9 output] PDF: {pdf_path}")
    return {"png": png_path, "pdf": pdf_path, "canvas_qc": canvas_qc, "summary": summary_df, "audit": audit_df}


def render_all_paper_fig2_fig9_from_csv():
    print("[Standalone renderer] Raw GRIB/NetCDF read = False")
    print("[Standalone renderer] Upstream scientific calculation = False")
    print("[Standalone renderer] Kernel-state dependency = False")

    file_map = {
        "Fig.2 daily table": FIG2_DAILY_CSV,
        "Fig.2 member table": FIG2_MEMBER_CSV,
        "Fig.2 stage summary": FIG2_STAGE_CSV,
        "Fig.9 summary (legacy upstream filename)": FIG9_SUMMARY_CSV,
        "Fig.9 RMSE audit (legacy upstream filename)": FIG9_AUDIT_CSV,
    }
    _require_files(file_map)
    for label, path in file_map.items():
        print(f"[{label} input] {path}")

    daily_df = pd.read_csv(FIG2_DAILY_CSV)
    member_df = pd.read_csv(FIG2_MEMBER_CSV)
    stage_df = pd.read_csv(FIG2_STAGE_CSV)
    daily_df, member_df, stage_df = _validate_fig2_inputs(daily_df, member_df, stage_df)

    outputs = {}
    outputs["fig2"] = render_fig2_from_csv(daily_df, member_df, stage_df, "2023-06-12")
    outputs["figS2"] = render_fig2_from_csv(daily_df, member_df, stage_df, "2023-06-08")

    summary_df = pd.read_csv(FIG9_SUMMARY_CSV)
    audit_df = pd.read_csv(FIG9_AUDIT_CSV)
    ordered_summary, ordered_audit = _validate_fig9_inputs(summary_df, audit_df)
    outputs["fig9"] = render_fig9_from_csv(ordered_summary, ordered_audit)

    fig2_caption = PAPER_FIG_DIR / "fig2_caption_jgr.md"
    fig9_caption = PAPER_FIG_DIR / "fig9_caption_jgr.md"
    fig2_qc_path = PAPER_FIG_DIR / "fig2_jgr_qc_summary.txt"
    fig9_qc_path = PAPER_FIG_DIR / "fig9_jgr_qc_summary.txt"
    _write_fig2_caption(fig2_caption)
    _write_fig9_caption(fig9_caption)

    fig2_result = outputs["fig2"]
    fig2_qc = {
        "figure_id": "Fig.2",
        **fig2_result["canvas_qc"],
        "panel_count": 3,
        "panel_labels": "a,b,c",
        "source_csvs": "; ".join(map(str, [FIG2_DAILY_CSV, FIG2_MEMBER_CSV, FIG2_STAGE_CSV])),
        "init_date": "2023-06-12",
        "Stage-I n_days": int(fig2_result["stage_rows"]["Stage-I"]["n_days"]),
        "Total n_days": int(fig2_result["stage_rows"]["Total"]["n_days"]),
        "member_count Stage-I": fig2_result["member_counts"]["Stage-I"],
        "member_count Total": fig2_result["member_counts"]["Total"],
        "annotation_contains_bias": False,
        "annotation_contains_rmse": True,
        "black_diamond_definition": "RMSE of ensemble-mean forecast",
    }
    _write_qc(fig2_qc_path, fig2_qc)

    fig9_result = outputs["fig9"]
    fig9_qc = {
        "figure_id": "Fig.9",
        "final_figure_number": 9,
        "legacy_upstream_figure_number": 10,
        **fig9_result["canvas_qc"],
        "panel_count": 2,
        "panel_labels": "a,b",
        "source_csvs": f"{FIG9_SUMMARY_CSV}; {FIG9_AUDIT_CSV}",
        "init_order": ",".join(FIG9_INIT_ORDER),
        "Total n_days": 11,
        "rmse_definition_pass_all_four_inits": all(
            _parse_bool_strict(value) for value in fig9_result["audit"]["rmse_definition_pass"]
        ),
        "RMSE calculation verified": "daily error -> square -> time mean -> square root",
        "RMSE_not_equal_to": "abs(stage-mean signed bias)",
        "bias_values": ",".join(f"{value:.8g}" for value in fig9_result["summary"]["bias"].astype(float)),
        "rmse_values": ",".join(f"{value:.8g}" for value in fig9_result["summary"]["rmse"].astype(float)),
    }
    _write_qc(fig9_qc_path, fig9_qc)

    outputs["fig2"]["caption"] = fig2_caption
    outputs["fig2"]["qc"] = fig2_qc_path
    outputs["fig9"]["caption"] = fig9_caption
    outputs["fig9"]["qc"] = fig9_qc_path
    print(f"[Fig.2 caption] {fig2_caption}")
    print(f"[Fig.2 QC] {fig2_qc_path}")
    print(f"[Fig.9 caption] {fig9_caption}")
    print(f"[Fig.9 QC] {fig9_qc_path}")
    print("[Fig.9 RMSE QC] four initialization rows checked; Total n_days=11; rmse_definition_pass=True required")
    print("[Fig.9 RMSE QC] RMSE definition = daily error -> square -> time mean -> square root")
    print("Standalone Fig.2/Fig.S2/Fig.9 rendering completed from existing CSV files.")
    return outputs


render_all_paper_fig2_fig9_from_csv()
