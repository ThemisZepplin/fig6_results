# =============================================================================
# [Standalone Cell] Final merged JGR Fig.2 plotting-only renderer
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
from matplotlib.patches import Patch
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
MERGED_FIG2_PREFIX = "fig2_tmax_error_multiinit_merged_jgr"
MERGED_FIG2_PNG = PAPER_FIG_DIR / f"{MERGED_FIG2_PREFIX}.png"
MERGED_FIG2_PDF = PAPER_FIG_DIR / f"{MERGED_FIG2_PREFIX}.pdf"
MERGED_FIG2_CAPTION = PAPER_FIG_DIR / f"{MERGED_FIG2_PREFIX}_caption.md"
MERGED_FIG2_QC = PAPER_FIG_DIR / f"{MERGED_FIG2_PREFIX}_qc.txt"

FIG2_DAILY_CSV = TAB_DIR / "tmax_daily_error_timeseries.csv"
FIG2_MEMBER_CSV = TAB_DIR / "tmax_member_error_by_stage.csv"
FIG2_STAGE_CSV = TAB_DIR / "tmax_stage_summary_metrics.csv"
# Legacy upstream filenames are retained for data compatibility; these inputs now
# support final manuscript Fig.9 and are never overwritten by this renderer.
FIG9_SUMMARY_CSV = PAPER_FIG_INPUT_DIR / "fig2b_leadtime_total_tmax_error_bias_rmse.csv"
FIG9_AUDIT_CSV = PAPER_FIG_INPUT_DIR / "fig10_rmse_definition_audit.csv"

INIT_ORDER = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]
PAPER_FIG2_INITS = ["2023-06-12", "2023-06-08"]
FIG9_INIT_ORDER = INIT_ORDER.copy()
TARGET_DATES = pd.date_range("2023-06-14", "2023-06-24", freq="D")
EXPECTED_NDAYS = {"Stage-I": 4, "Total": 11}
STAGE_WINDOWS = {
    "Stage-I": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17")),
    "Total": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-24")),
}
INIT_LABELS = {init: pd.Timestamp(init).strftime("%m-%d") for init in INIT_ORDER}
INIT_COLORS = {
    "2023-06-01": "#466F87",
    "2023-06-05": "#7BB6C8",
    "2023-06-08": "#F2D98E",
    "2023-06-12": "#E48578",
}

# Default execution produces only the merged main-text Fig.2. These explicit
# switches preserve the prior supplementary and legacy figure renderers as
# opt-in functionality without making their legacy CSVs mandatory.
GENERATE_OPTIONAL_FIGS2 = False
GENERATE_OPTIONAL_LEGACY_FIG9 = False

# -------------------------
# Final JGR manuscript style (local rc_context only)
# -------------------------
FIG_WIDTH_IN = 6.5
FIG2_HEIGHT_IN = 5.1
MERGED_FIG2_HEIGHT_IN = 6.4
FIG9_HEIGHT_IN = 3.0
FIG_DPI = 600
FONT_PANEL_LETTER = 10.8
FONT_PANEL_TITLE = 9.6
FONT_AXIS_LABEL = 9.5
FONT_TICK = 8.7
FONT_LEGEND = 8.4
FONT_ANNOTATION = 8.3
FONT_BAR_VALUE = 7.0
BAR_VALUE_ROTATION = 0
BAR_VALUE_BASE_OFFSET_PT = 3.0
BAR_VALUE_STAGGER_PT = 12.0
BAR_VALUE_CLOSE_PT = 11.0
FONT_DIAMOND_LEGEND = 7.2
TIMESERIES_COLORS = {"ERA5": "black", "S2S": "royalblue", "interval": "deepskyblue"}
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


def _assert_finite(values, label):
    numeric = np.asarray(pd.to_numeric(np.asarray(values).ravel(), errors="coerce"), dtype=float)
    bad = np.flatnonzero(~np.isfinite(numeric))
    if len(bad):
        raise ValueError(f"Non-finite values found in {label}; flattened indices={bad.tolist()}")


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
    daily_numeric_columns = [
        "ERA5_Tmax", "S2S_ensmean_Tmax", "ens_p05", "ens_p25",
        "ens_p75", "ens_p95", "daily_error",
    ]
    validation_records = []
    reference_era5 = None
    for init in INIT_ORDER:
        dsub = daily_df[daily_df["init"].astype(str) == init].copy().sort_values("date")
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
        numeric_daily = dsub[daily_numeric_columns].apply(pd.to_numeric, errors="coerce")
        _assert_finite(numeric_daily.to_numpy(), f"Fig.2 daily values init={init}")
        expected_daily_error = numeric_daily["S2S_ensmean_Tmax"] - numeric_daily["ERA5_Tmax"]
        if not np.allclose(
            numeric_daily["daily_error"], expected_daily_error, atol=1e-6, rtol=0
        ):
            max_diff = float(np.max(np.abs(numeric_daily["daily_error"] - expected_daily_error)))
            raise ValueError(
                f"Fig.2 daily_error mismatch for init={init}; max_abs_difference={max_diff:.9g} °C"
            )
        quantiles = numeric_daily[["ens_p05", "ens_p25", "ens_p75", "ens_p95"]].to_numpy()
        if np.any(np.diff(quantiles, axis=1) < -1e-12):
            bad_rows = np.flatnonzero(np.any(np.diff(quantiles, axis=1) < -1e-12, axis=1))
            bad_dates = dsub.iloc[bad_rows]["date"].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Ensemble quantiles are not ordered for init={init}; dates={bad_dates}")
        era5_values = numeric_daily["ERA5_Tmax"].to_numpy()
        if reference_era5 is None:
            reference_era5 = era5_values
        elif not np.allclose(era5_values, reference_era5, atol=1e-10, rtol=0):
            max_diff = float(np.max(np.abs(era5_values - reference_era5)))
            raise ValueError(
                f"ERA5 verification sequence differs for init={init}; max_abs_difference={max_diff:.9g} °C"
            )

        for stage, expected_n_days in EXPECTED_NDAYS.items():
            msub = member_df[(member_df["init"].astype(str) == init) & (member_df["stage"].astype(str) == stage)]
            member_ids = pd.to_numeric(msub["member"], errors="coerce")
            if len(msub) != 51 or member_ids.nunique() != 51:
                raise ValueError(
                    f"Fig.2 member table init={init} stage={stage} requires 51 unique members; "
                    f"rows={len(msub)}, unique={member_ids.nunique()}"
                )
            if msub["member"].duplicated().any():
                raise ValueError(f"Fig.2 member table has duplicated member IDs for init={init} stage={stage}")
            if member_ids.isna().any() or not np.allclose(member_ids, np.round(member_ids), atol=0, rtol=0):
                raise ValueError(f"Fig.2 member IDs must be finite integers for init={init} stage={stage}")
            got_members = set(member_ids.astype(int).tolist())
            if got_members != set(range(51)):
                missing = sorted(set(range(51)) - got_members)
                extra = sorted(got_members - set(range(51)))
                raise ValueError(
                    f"Fig.2 member IDs must be 0–50 for init={init} stage={stage}; "
                    f"missing={missing}, extra={extra}"
                )
            member_rmse = pd.to_numeric(msub["rmse"], errors="coerce").to_numpy(dtype=float)
            _assert_finite(member_rmse, f"Fig.2 member RMSE init={init} stage={stage}")
            if np.any(member_rmse < 0):
                raise ValueError(f"Negative member RMSE found for init={init} stage={stage}")

            ssub = stage_df[(stage_df["init"].astype(str) == init) & (stage_df["stage"].astype(str) == stage)]
            if len(ssub) != 1:
                raise ValueError(f"Fig.2 stage summary init={init} stage={stage} expected 1 row, got {len(ssub)}")
            if int(ssub.iloc[0]["n_days"]) != expected_n_days:
                raise ValueError(
                    f"Fig.2 stage summary init={init} stage={stage} expected n_days={expected_n_days}, "
                    f"got {ssub.iloc[0]['n_days']}"
                )
            summary_bias = float(pd.to_numeric(ssub.iloc[0]["ensmean_bias"], errors="coerce"))
            summary_rmse = float(pd.to_numeric(ssub.iloc[0]["ensmean_rmse"], errors="coerce"))
            _assert_finite([summary_bias, summary_rmse], f"Fig.2 stage metrics init={init} stage={stage}")
            if summary_rmse < 0:
                raise ValueError(f"Negative ensemble-mean RMSE found for init={init} stage={stage}")

            start, end = STAGE_WINDOWS[stage]
            stage_daily = dsub[(dsub["date"] >= start) & (dsub["date"] <= end)]
            if len(stage_daily) != expected_n_days:
                raise ValueError(
                    f"Fig.2 daily table init={init} stage={stage} expected {expected_n_days} dates, "
                    f"got {len(stage_daily)}"
                )
            errors = pd.to_numeric(stage_daily["daily_error"], errors="coerce").to_numpy(dtype=float)
            recomputed_bias = float(np.mean(errors))
            recomputed_rmse = float(np.sqrt(np.mean(errors ** 2)))
            bias_difference = recomputed_bias - summary_bias
            rmse_difference = recomputed_rmse - summary_rmse
            if not np.isclose(recomputed_bias, summary_bias, atol=1e-6, rtol=0):
                raise ValueError(
                    f"Stage Bias mismatch for init={init} stage={stage}; "
                    f"summary={summary_bias:.9g}, recomputed={recomputed_bias:.9g}, "
                    f"difference={bias_difference:.9g} °C"
                )
            if not np.isclose(recomputed_rmse, summary_rmse, atol=1e-6, rtol=0):
                raise ValueError(
                    f"Stage RMSE mismatch for init={init} stage={stage}; "
                    f"summary={summary_rmse:.9g}, recomputed={recomputed_rmse:.9g}, "
                    f"difference={rmse_difference:.9g} °C"
                )
            validation_records.append({
                "init": init,
                "stage": stage,
                "n_days": expected_n_days,
                "member_count": 51,
                "ensmean_bias": summary_bias,
                "ensmean_rmse": summary_rmse,
                "recomputed_bias": recomputed_bias,
                "recomputed_rmse": recomputed_rmse,
                "bias_difference": bias_difference,
                "rmse_difference": rmse_difference,
            })
    return daily_df, member_df, stage_df, pd.DataFrame(validation_records)


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
        "Figure 2. Evolution and multi-initialization verification of regional maximum 2-m "
        "temperature (Tmax) over North China. (a) Daily ERA5 Tmax and the 51-member S2S "
        "ensemble mean for the forecast initialized on 12 June 2023, with the 25th–75th and "
        "5th–95th ensemble-percentile ranges shown by shading. The vertical line marks the end "
        "of Stage-I. (b) Ensemble-mean RMSE (upper) and signed Bias (lower) for forecasts "
        "initialized on 1, 5, 8, and 12 June, evaluated for Stage-I (14–17 June) and the Total "
        "period (14–24 June). (c) Member-wise RMSE distributions for Stage-I (upper) and the "
        "Total period (lower) for the same initialization dates. All daily quantities follow "
        "the upstream Beijing-time (UTC+8) processing. Each box contains n = 51 member RMSE "
        "values; boxes span the 25th–75th percentiles, center lines denote medians. "
        "Whiskers extend to the most extreme observations within 1.5 interquartile ranges "
        "below the first quartile and above the third quartile. "
        "Gray points show all members. Black "
        "diamonds denote the RMSE of the ensemble-mean forecast, calculated after first taking "
        "the daily mean across the 51 members, rather than the mean or median of member-wise RMSE."
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
def _add_internal_axis_title(ax, title):
    ax.text(
        0.985, 0.96, title, transform=ax.transAxes,
        ha="right", va="top", fontsize=FONT_PANEL_TITLE, color="0.20",
    )


def _stage_metric_lookup(stage_df, metric):
    values = {}
    for stage in EXPECTED_NDAYS:
        for init in INIT_ORDER:
            row = stage_df[
                (stage_df["init"].astype(str) == init)
                & (stage_df["stage"].astype(str) == stage)
            ].iloc[0]
            values[(stage, init)] = float(row[metric])
    return values


def _draw_grouped_stage_bars(ax, metric_values, metric):
    stage_order = ["Stage-I", "Total"]
    group_centers = np.array([0.0, 1.25])
    bar_width = 0.16
    offsets = (np.arange(len(INIT_ORDER)) - 1.5) * (bar_width + 0.025)
    all_values = np.array(
        [metric_values[(stage, init)] for stage in stage_order for init in INIT_ORDER],
        dtype=float,
    )
    if metric == "rmse":
        upper = max(0.5, float(np.max(all_values)))
        ax.set_ylim(0.0, upper * 1.24)
        ax.set_ylabel("RMSE (°C)", fontsize=FONT_AXIS_LABEL)
    else:
        lower = min(0.0, float(np.min(all_values)))
        upper = max(0.0, float(np.max(all_values)))
        span = max(upper - lower, 0.5)
        ax.set_ylim(lower - 0.20 * span, upper + 0.16 * span)
        ax.axhline(0, color="0.18", lw=0.9)
        ax.set_ylabel("Bias (°C)", fontsize=FONT_AXIS_LABEL)
        ax.text(0.985, 1.025, "Bias", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=FONT_PANEL_TITLE, color="0.20")

    # Estimate label widths without drawing; stagger nearby same-sign endpoints
    # in points, then reserve vertical room for the largest possible offset.
    axis_position = ax.get_position()
    axis_height_pt = axis_position.height * ax.figure.get_figheight() * 72
    axis_width_pt = axis_position.width * ax.figure.get_figwidth() * 72
    x_span = group_centers[-1] - group_centers[0] + 1.04
    max_offset_pt = BAR_VALUE_BASE_OFFSET_PT + len(INIT_ORDER) * BAR_VALUE_STAGGER_PT
    lower_limit, upper_limit = ax.get_ylim()
    reserve_fraction = min(0.35, (max_offset_pt + FONT_BAR_VALUE + 2) / axis_height_pt)
    padding = (upper_limit - lower_limit) * reserve_fraction / (1 - 2 * reserve_fraction)
    ax.set_ylim(lower_limit - (padding if metric != "rmse" else 0), upper_limit + padding)
    value_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_offsets = {}
    for stage in stage_order:
        placed = []
        for init_index, init in enumerate(INIT_ORDER):
            value = metric_values[(stage, init)]
            text = f"{value:.2f}" if metric == "rmse" else _format_signed_without_plus(value)
            direction = 1 if value >= 0 else -1
            width_pt = len(text) * FONT_BAR_VALUE * 0.60
            x_pt = offsets[init_index] * axis_width_pt / x_span
            y_pt = value * axis_height_pt / value_span
            offset_pt = BAR_VALUE_BASE_OFFSET_PT
            for _ in INIT_ORDER:
                endpoint_pt = y_pt + direction * offset_pt
                collisions = [item for item in placed if item[0] == direction
                              and abs(x_pt - item[1]) < (width_pt + item[3]) / 2 + 2
                              and abs(endpoint_pt - item[2]) < BAR_VALUE_CLOSE_PT]
                if not collisions:
                    break
                offset_pt += BAR_VALUE_STAGGER_PT
            label_offsets[(stage, init)] = offset_pt
            placed.append((direction, x_pt, y_pt + direction * offset_pt, width_pt))
    for init_index, init in enumerate(INIT_ORDER):
        positions = group_centers + offsets[init_index]
        values = np.array([metric_values[(stage, init)] for stage in stage_order], dtype=float)
        bars = ax.bar(
            positions, values, width=bar_width, color=INIT_COLORS[init], alpha=BAR_ALPHA,
            edgecolor=EDGE_COLOR, linewidth=EDGE_LW, zorder=3,
        )
        for stage, bar, value in zip(stage_order, bars, values):
            direction = 1 if value >= 0 else -1
            text = f"{value:.2f}" if metric == "rmse" else _format_signed_without_plus(value)
            ax.annotate(
                text, xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, direction * label_offsets[(stage, init)]),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=FONT_BAR_VALUE,
                rotation=BAR_VALUE_ROTATION,
                color="black", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.08"),
                zorder=5,
            )
    ax.set_xlim(group_centers[0] - 0.52, group_centers[-1] + 0.52)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(stage_order)
    ax.tick_params(axis="x", which="both", length=0)
    _apply_axis_style(ax, grid_axis="y")


def _draw_member_rmse_boxes(ax, member_df, stage_df, stage, y_upper, seed):
    data = []
    for init in INIT_ORDER:
        values = member_df[
            (member_df["init"].astype(str) == init)
            & (member_df["stage"].astype(str) == stage)
        ].sort_values("member")["rmse"].astype(float).to_numpy()
        data.append(values)
    positions = np.arange(1, len(INIT_ORDER) + 1)
    boxplot = ax.boxplot(
        data,
        positions=positions,
        widths=0.54,
        whis=1.5,
        patch_artist=True,
        showfliers=False,
    )
    for patch, init in zip(boxplot["boxes"], INIT_ORDER):
        patch.set_facecolor(INIT_COLORS[init])
        patch.set_alpha(0.66)
        patch.set_edgecolor(EDGE_COLOR)
        patch.set_linewidth(0.75)
    for key in ["medians", "whiskers", "caps"]:
        for artist in boxplot[key]:
            artist.set_color(EDGE_COLOR)
            artist.set_linewidth(0.75)

    rng = np.random.default_rng(seed)
    for position, init, values in zip(positions, INIT_ORDER, data):
        jitter = rng.normal(0.0, 0.028, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=6.5,
            color="0.32",
            alpha=0.27,
            linewidths=0,
            zorder=3,
        )
        ensmean_rmse = float(stage_df[
            (stage_df["init"].astype(str) == init)
            & (stage_df["stage"].astype(str) == stage)
        ]["ensmean_rmse"].iloc[0])
        ax.scatter(
            position, ensmean_rmse, marker="D", s=28, color="black",
            edgecolors="none", zorder=6,
        )
    ax.set_ylim(0.0, y_upper)
    ax.set_xticks(positions, [INIT_LABELS[init] for init in INIT_ORDER])
    ax.set_ylabel("")
    _apply_axis_style(ax, grid_axis="y")
    _add_internal_axis_title(ax, stage)
    return data


def render_merged_fig2_from_csv(
    daily_df, member_df, stage_df, validation_df, out_dir=PAPER_FIG_DIR
):
    """Render the merged five-axis Fig.2 exclusively from validated CSV values."""
    os.makedirs(out_dir, exist_ok=True)
    init_for_timeseries = "2023-06-12"
    stages = ["Stage-I", "Total"]
    sub = daily_df[daily_df["init"].astype(str) == init_for_timeseries].copy().sort_values("date")
    stage_rows = {
        stage: stage_df[
            (stage_df["init"].astype(str) == init_for_timeseries)
            & (stage_df["stage"].astype(str) == stage)
        ].iloc[0]
        for stage in stages
    }
    rmse_metrics = _stage_metric_lookup(stage_df, "ensmean_rmse")
    bias_metrics = _stage_metric_lookup(stage_df, "ensmean_bias")
    all_member_rmse = member_df[
        member_df["init"].astype(str).isin(INIT_ORDER)
        & member_df["stage"].astype(str).isin(stages)
    ]["rmse"].astype(float).to_numpy()
    all_ensmean_rmse = np.array(list(rmse_metrics.values()), dtype=float)
    member_y_upper = max(
        10.0,
        float(np.ceil(max(np.max(all_member_rmse), np.max(all_ensmean_rmse)) + 0.5)),
    )

    with plt.rc_context(PAPER_RC_PARAMS):
        fig = plt.figure(figsize=(FIG_WIDTH_IN, MERGED_FIG2_HEIGHT_IN), constrained_layout=False)
        outer_gs = fig.add_gridspec(
            3, 1, height_ratios=[1.42, 0.40, 2.65], hspace=0.25,
        )
        legend_gs = outer_gs[1].subgridspec(2, 1, hspace=0.08)
        lower_gs = outer_gs[2].subgridspec(
            3, 2, height_ratios=[1.0, 0.48, 1.22], width_ratios=[1.0, 1.08],
            hspace=0.0, wspace=0.43,
        )
        ax_ts = fig.add_subplot(outer_gs[0])
        ax_ts_legend = fig.add_subplot(legend_gs[0])
        ax_init_legend = fig.add_subplot(legend_gs[1])
        ax_rmse_bar = fig.add_subplot(lower_gs[0, 0])
        ax_bias_bar = fig.add_subplot(lower_gs[2, 0], sharex=ax_rmse_bar)
        ax_box_stage1 = fig.add_subplot(lower_gs[0, 1])
        ax_box_total = fig.add_subplot(lower_gs[2, 1], sharex=ax_box_stage1, sharey=ax_box_stage1)
        ax_diamond_legend = fig.add_subplot(lower_gs[1, 1])
        # Set final geometry before point-based bar-label spacing is estimated.
        fig.subplots_adjust(left=0.10, right=0.975, top=0.95, bottom=0.075)

        ax_ts.axvline(pd.Timestamp("2023-06-17 12:00"), color="0.25", lw=0.9, zorder=2)
        p0595 = ax_ts.fill_between(
            sub["date"], sub["ens_p05"], sub["ens_p95"], color=TIMESERIES_COLORS["interval"],
            alpha=0.14, lw=0, label="S2S p05–p95", zorder=1,
        )
        p2575 = ax_ts.fill_between(
            sub["date"], sub["ens_p25"], sub["ens_p75"], color=TIMESERIES_COLORS["interval"],
            alpha=0.30, lw=0, label="S2S p25–p75", zorder=1,
        )
        era5_line, = ax_ts.plot(
            sub["date"], sub["ERA5_Tmax"], color=TIMESERIES_COLORS["ERA5"], lw=1.75,
            marker="o", ms=4.1, label="ERA5", zorder=3,
        )
        s2s_line, = ax_ts.plot(
            sub["date"], sub["S2S_ensmean_Tmax"], color=TIMESERIES_COLORS["S2S"], lw=1.65,
            marker="s", ms=4.1, label="S2S ensemble mean (06-12)", zorder=3,
        )
        annotation = "\n".join(
            f"{stage} RMSE = {float(stage_rows[stage]['ensmean_rmse']):.2f}°C"
            for stage in stages
        )
        ax_ts.text(
            0.015, 0.06, annotation, transform=ax_ts.transAxes, ha="left", va="bottom",
            fontsize=FONT_ANNOTATION,
            bbox=dict(facecolor="white", alpha=0.74, edgecolor="none", boxstyle="round,pad=0.20"),
            zorder=15,
        )
        _add_panel_heading(ax_ts, "a", "Tmax evolution (06-12 initialization)")
        ax_ts.set_ylabel("Daily Tmax (°C)", fontsize=FONT_AXIS_LABEL)
        ax_ts.set_xlim(pd.Timestamp("2023-06-13 12:00"), pd.Timestamp("2023-06-24 12:00"))
        ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax_ts.get_xticklabels(), rotation=30, ha="right", fontsize=FONT_TICK)
        _apply_axis_style(ax_ts, grid_axis="both")

        ax_ts_legend.axis("off")
        ax_ts_legend.legend(
            [era5_line, s2s_line, p2575, p0595],
            ["ERA5", "S2S ensemble mean", "S2S p25–p75", "S2S p05–p95"],
            loc="center", ncol=4, fontsize=FONT_LEGEND, frameon=False,
            handlelength=1.55, columnspacing=0.9, handletextpad=0.45,
        )
        ax_init_legend.axis("off")
        init_handles = [
            Patch(facecolor=INIT_COLORS[init], edgecolor=EDGE_COLOR, linewidth=EDGE_LW,
                  label=INIT_LABELS[init])
            for init in INIT_ORDER
        ]
        ax_init_legend.legend(
            handles=init_handles, loc="center", ncol=4,
            fontsize=FONT_LEGEND, frameon=False,
            handlelength=1.25, columnspacing=1.0, handletextpad=0.4,
        )

        _draw_grouped_stage_bars(ax_rmse_bar, rmse_metrics, "rmse")
        _draw_grouped_stage_bars(ax_bias_bar, bias_metrics, "bias")
        ax_rmse_bar.tick_params(axis="x", labelbottom=True)
        ax_bias_bar.tick_params(axis="x", labelbottom=True)
        _add_panel_heading(ax_rmse_bar, "b", "Ensemble-mean error metrics")

        _draw_member_rmse_boxes(
            ax_box_stage1, member_df, stage_df, "Stage-I", member_y_upper, seed=20240612,
        )
        _draw_member_rmse_boxes(
            ax_box_total, member_df, stage_df, "Total", member_y_upper, seed=20240613,
        )
        _add_panel_heading(ax_box_stage1, "c", "Member RMSE distributions")
        diamond = Line2D([], [], marker="D", linestyle="None", color="black", markersize=4.2)
        ax_diamond_legend.axis("off")
        ax_diamond_legend.legend(
            [diamond], ["Ensemble-mean RMSE"], loc="center",
            bbox_to_anchor=(0.5, 0.40), fontsize=FONT_DIAMOND_LEGEND,
            frameon=False, handletextpad=0.3, borderaxespad=0.0,
        )

        box_top = ax_box_stage1.get_position()
        box_bottom = ax_box_total.get_position()
        fig.text(box_top.x0 - 0.067, (box_top.y1 + box_bottom.y0) / 2,
                 "RMSE (°C)", rotation=90, ha="center", va="center",
                 fontsize=FONT_AXIS_LABEL)
        png_path = Path(out_dir) / MERGED_FIG2_PNG.name
        pdf_path = Path(out_dir) / MERGED_FIG2_PDF.name
        canvas_qc = _save_fixed_canvas(fig, png_path, pdf_path)
        plt.close(fig)

    member_counts = {
        f"{init}|{stage}": int(len(member_df[
            (member_df["init"].astype(str) == init)
            & (member_df["stage"].astype(str) == stage)
        ]))
        for init in INIT_ORDER for stage in stages
    }
    print(f"[Merged Fig.2 output] PNG: {png_path}")
    print(f"[Merged Fig.2 output] PDF: {pdf_path}")
    return {
        "png": png_path,
        "pdf": pdf_path,
        "canvas_qc": canvas_qc,
        "validation": validation_df.copy(),
        "member_counts": member_counts,
        "member_y_upper": member_y_upper,
        "panel_count": 3,
        "plot_axis_count": 5,
    }


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


def _write_merged_fig2_qc(path, result):
    validation_df = result["validation"]
    records = {
        "figure_id": "Fig.2",
        **result["canvas_qc"],
        "panel_count": result["panel_count"],
        "plot_axis_count": result["plot_axis_count"],
        "panel_labels": "a,b,c",
        "source_csvs": "; ".join(
            map(str, [FIG2_DAILY_CSV, FIG2_MEMBER_CSV, FIG2_STAGE_CSV])
        ),
        "init_order": ",".join(INIT_ORDER),
        "stage_order": "Stage-I,Total",
        "Stage-I dates": "2023-06-14 to 2023-06-17 (BJT)",
        "Stage-I n_days": EXPECTED_NDAYS["Stage-I"],
        "Total dates": "2023-06-14 to 2023-06-24 (BJT)",
        "Total n_days": EXPECTED_NDAYS["Total"],
        "initialization_colors": "; ".join(
            f"{init}:{INIT_COLORS[init]}" for init in INIT_ORDER
        ),
        "member_rmse_common_y_upper_C": result["member_y_upper"],
        "black_diamond_definition": "RMSE of ensemble-mean forecast",
        "validation_tolerance_C": "atol=1e-6, rtol=0",
        "visual_collision_check": "manual check required after rendering",
        "bar_value_label_qc": "inspect bar-label spacing and possible overlaps after the first actual rendering",
        "timeseries_colors": str(TIMESERIES_COLORS),
        "timeseries_interval_alpha": "p25-p75=0.30; p05-p95=0.14",
        "bar_value_font_pt": FONT_BAR_VALUE,
        "bar_value_rotation": BAR_VALUE_ROTATION,
        "bar_value_style": "black bold; compact white square background, pad=0.08; no border or shadow",
        "bar_value_offset_strategy": "point offsets outside signed endpoints; estimated text-width and same-sign endpoint proximity staggering",
        "bar_value_offset_points": f"base={BAR_VALUE_BASE_OFFSET_PT}; stagger={BAR_VALUE_STAGGER_PT}; proximity={BAR_VALUE_CLOSE_PT}",
        "bar_value_axis_padding": "reserve vertical space for maximum point offset and label height",
        "member_rmse_ylabel": "single RMSE (°C), vertically centered left of both boxplot rows",
        "diamond_legend_location": "dedicated inter-row axis, below Stage-I date labels and above Total",
        "diamond_legend_font_pt": FONT_DIAMOND_LEGEND,
        "central_legends": "two compact horizontal rows; no initialization title; dedicated nested GridSpec",
        "expected_merged_png_size_px": "3900 x 3840",
        "visual_layout_qc": "manual inspection required after actual rendering: bar-value overlap, legend spacing, occlusion and clipping; no automatic visual validation",
    }
    for row in validation_df.itertuples(index=False):
        prefix = f"{row.init}|{row.stage}"
        records[f"{prefix}|member_count"] = int(row.member_count)
        records[f"{prefix}|ensmean_bias_C"] = f"{row.ensmean_bias:.12g}"
        records[f"{prefix}|ensmean_rmse_C"] = f"{row.ensmean_rmse:.12g}"
        records[f"{prefix}|recomputed_bias_C"] = f"{row.recomputed_bias:.12g}"
        records[f"{prefix}|recomputed_rmse_C"] = f"{row.recomputed_rmse:.12g}"
        records[f"{prefix}|bias_difference_C"] = f"{row.bias_difference:.12g}"
        records[f"{prefix}|rmse_difference_C"] = f"{row.rmse_difference:.12g}"
    _write_qc(path, records)


def render_merged_fig2_from_existing_csv(
    generate_optional_figs2=GENERATE_OPTIONAL_FIGS2,
    generate_optional_legacy_fig9=GENERATE_OPTIONAL_LEGACY_FIG9,
):
    """Validate the three science tables and render the merged Fig.2.

    The two boolean options retain the old Fig.S2 and Fig.9 products without
    making either product, or the legacy Fig.9 CSVs, part of the default run.
    """
    print("[Standalone renderer] Raw GRIB/NetCDF read = False")
    print("[Standalone renderer] Upstream scientific calculation = False")
    print("[Standalone renderer] Kernel-state dependency = False")

    file_map = {
        "Merged Fig.2 daily table": FIG2_DAILY_CSV,
        "Merged Fig.2 member table": FIG2_MEMBER_CSV,
        "Merged Fig.2 stage summary": FIG2_STAGE_CSV,
    }
    _require_files(file_map)
    for label, path in file_map.items():
        print(f"[{label} input] {path}")

    daily_df = pd.read_csv(FIG2_DAILY_CSV)
    member_df = pd.read_csv(FIG2_MEMBER_CSV)
    stage_df = pd.read_csv(FIG2_STAGE_CSV)
    daily_df, member_df, stage_df, validation_df = _validate_fig2_inputs(
        daily_df, member_df, stage_df
    )

    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    outputs = {
        "fig2": render_merged_fig2_from_csv(
            daily_df, member_df, stage_df, validation_df
        )
    }
    _write_fig2_caption(MERGED_FIG2_CAPTION)
    _write_merged_fig2_qc(MERGED_FIG2_QC, outputs["fig2"])
    outputs["fig2"]["caption"] = MERGED_FIG2_CAPTION
    outputs["fig2"]["qc"] = MERGED_FIG2_QC
    print(f"[Merged Fig.2 caption] {MERGED_FIG2_CAPTION}")
    print(f"[Merged Fig.2 QC] {MERGED_FIG2_QC}")

    if generate_optional_figs2:
        outputs["figS2"] = render_fig2_from_csv(
            daily_df, member_df, stage_df, "2023-06-08"
        )

    if generate_optional_legacy_fig9:
        legacy_file_map = {
            "Legacy Fig.9 summary": FIG9_SUMMARY_CSV,
            "Legacy Fig.9 RMSE audit": FIG9_AUDIT_CSV,
        }
        _require_files(legacy_file_map)
        for label, path in legacy_file_map.items():
            print(f"[{label} input] {path}")
        summary_df = pd.read_csv(FIG9_SUMMARY_CSV)
        audit_df = pd.read_csv(FIG9_AUDIT_CSV)
        ordered_summary, ordered_audit = _validate_fig9_inputs(summary_df, audit_df)
        outputs["fig9"] = render_fig9_from_csv(ordered_summary, ordered_audit)

    print("Standalone merged Fig.2 rendering completed from existing CSV files.")
    return outputs


render_merged_fig2_from_existing_csv()
