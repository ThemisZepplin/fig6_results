"""Lead-time sensitivity maps for Total-stage Hot17/Cold17 S2S composites.

Notebook-friendly standalone cell/script.  It builds total-period mean maps for
2023-06-14 to 2023-06-24 (BJT) using Total-period Tmax-based Hot17/Cold17
member groups for four S2S initializations.
"""

import glob
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import scipy
import statsmodels
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=FutureWarning)

# ====================================================
# Configuration
# ====================================================

INIT_DATES = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]
ANALYSIS_DATES = pd.date_range("2023-06-14", "2023-06-24", freq="D")
ANALYSIS_WINDOW_LABEL = "2023-06-14 to 2023-06-24"
GROUPING_STAGE = "Total"
BJT_OFFSET = pd.Timedelta(hours=8)
G0 = 9.80665

MAP_EXTENT = (90.0, 135.0, 25.0, 55.0)
DATA_EXTENT = (88.0, 137.0, 23.0, 57.0)
NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
SIGNIFICANCE_DOMAIN = (90.0, 135.0, 25.0, 55.0)

BASE_DATA_DIR = "/data1/huangy/fig6/NC"
CF_DIR = f"{BASE_DATA_DIR}/cf_2023-6"
ERA5_Z500_2023_FILE = f"{BASE_DATA_DIR}/MSE/ERA5/Z500/era5_z500_2023_06.nc"
MSWEP_2023_DIR = f"{BASE_DATA_DIR}/MSE/MSWEP/3hourly-2023"
GROUP_FILE = "/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv"
OUT_DIR = "/data1/huangy/fig6/NC/v11/leadtime_total_composite_maps"
os.makedirs(OUT_DIR, exist_ok=True)

HINDCAST_DIRS = {
    "2023-06-01": "/data1/huangy/fig6/NC/pl_01.6.23",
    "2023-06-05": "/data1/huangy/fig6/NC/pl_05.6.23",
    "2023-06-08": "/data1/huangy/fig6/NC/pl_08.6.23",
    "2023-06-12": "/data1/huangy/fig6/NC/pl_12.6.23",
}

GROUPS = ["Hot17", "Cold17"]
DIFF_GROUP = "Hot17_minus_Cold17"

SIGNIFICANCE_ROOT_FIG9 = "/data1/huangy/fig6/NC/v12/paper_figures/significance"
FIG9_SIGNIFICANCE_TP_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG9, "fig9_spatial_significance_tp.nc")
FIG9_SIGNIFICANCE_Z500_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG9, "fig9_spatial_significance_z500.nc")
FIG9_SIGNIFICANCE_QC_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG9, "fig9_spatial_significance_qc.csv")
SPATIAL_SIGNIFICANCE_AUDIT_PATH = os.path.join(
    SIGNIFICANCE_ROOT_FIG9, "fig7_fig9_spatial_significance_audit.md"
)

HINDCAST_CATALOG = {
    "cf": {
        "pl_500": ["{hindcast_dir}/ecmf_hindcast_cf_pl_500_{init_date}.grib"],
    },
    "pf": {
        "pl_500": ["{hindcast_dir}/ecmf_hindcast_pf_pl_500_{init_date}.grib"],
    },
}

S2S_FILE_CATALOG = {
    "realtime": {
        "pf": {
            "pl_500": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"],
            "sfc_tp": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_tp2023-06.grb"],
        },
        "cf": {
            "pl_500": [
                f"{CF_DIR}/ecmf_cf_pl_500_2023-06.grib",
                f"{CF_DIR}/ecmf_cf_pl500_156_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*pl*zqt*2023-06*.grb*",
            ],
            "sfc_tp": [
                f"{CF_DIR}/ecmf_cf_sfc_tp_2023-6.grib",
                f"{CF_DIR}/ecmf_cf_sfc_tp_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*tp*2023-06*.grb*",
            ],
        },
    },
    "hindcast": HINDCAST_CATALOG,
}

GRIB_SHORTNAME_ORDER = {
    "pl_500": ("gh", "z"),
    "sfc_tp": ("tp",),
}

# ====================================================
# Generic helpers
# ====================================================

def _norm_text(value) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm = {_norm_text(c): c for c in columns}
    for cand in candidates:
        key = _norm_text(cand)
        if key in norm:
            return norm[key]
    return None


def _parse_member_list(value) -> List[int]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        raw = list(value)
    else:
        raw = re.findall(r"\d+", str(value))
    return [int(v) for v in raw]


def _init_match(series: pd.Series, init_date: str) -> pd.Series:
    text_match = series.astype(str) == str(init_date)
    try:
        date_match = pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d") == pd.Timestamp(init_date).strftime("%Y-%m-%d")
        return text_match | date_match.fillna(False)
    except Exception:
        return text_match


def _total_stage_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).map(_norm_text).isin({"total", "totalstage"})


def _open_dataset_safe(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open {path}: {exc}") from exc


def _guess_data_var(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    lower = {v.lower(): v for v in ds.data_vars}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for var in ds.data_vars:
        attrs = ds[var].attrs
        short_name = str(attrs.get("GRIB_shortName", "")).lower()
        standard_name = str(attrs.get("standard_name", "")).lower()
        long_name = str(attrs.get("long_name", "")).lower()
        if any(c.lower() == short_name for c in candidates):
            return var
        if any(c.lower() in standard_name or c.lower() in long_name for c in candidates):
            return var
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not identify variable from {candidates}; available={list(ds.data_vars)}")


def _normalize_time_dim(da: xr.DataArray) -> xr.DataArray:
    for name in ("valid_time", "time", "forecast_time"):
        if name in da.dims and name != "time":
            return da.rename({name: "time"})
    return da


def _standardize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low in ("lat", "latitude") and name != "latitude":
            rename[name] = "latitude"
        if low in ("lon", "longitude") and name != "longitude":
            rename[name] = "longitude"
    return da.rename(rename) if rename else da


def _sel_region(da: xr.DataArray, extent: Tuple[float, float, float, float] = DATA_EXTENT) -> xr.DataArray:
    da = _standardize_lat_lon(da)
    lon_min, lon_max, lat_min, lat_max = extent
    lat_slice = slice(lat_max, lat_min) if float(da.latitude.values[0]) > float(da.latitude.values[-1]) else slice(lat_min, lat_max)
    out = da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    if out.sizes.get("longitude", 0) == 0 and float(da.longitude.max()) > 180 and lon_min < 180:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")
        out = da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    return out


def _select_level(da: xr.DataArray, level_hpa: int) -> xr.DataArray:
    da = _standardize_lat_lon(da)
    for lev in ("isobaricInhPa", "isobaricInPa", "level", "pressure_level"):
        if lev in da.dims:
            vals = np.asarray(da[lev].values)
            target = level_hpa * 100 if np.nanmax(vals) > 2000 else level_hpa
            return da.sel({lev: target}, method="nearest")
        if lev in da.coords:
            vals = np.asarray(da[lev].values)
            target = level_hpa * 100 if np.nanmax(vals) > 2000 else level_hpa
            if vals.ndim == 0 or vals.size == 1:
                val = float(np.ravel(vals)[0])
                if abs(val - target) <= max(1.0, 0.01 * abs(target)):
                    return da.drop_vars(lev, errors="ignore")
                raise ValueError(f"Scalar level {lev}={val}, requested={target}")
            dim = da[lev].dims[0]
            return da.isel({dim: int(np.nanargmin(np.abs(vals - target)))}).drop_vars(lev, errors="ignore")
    return da


def _to_gpm_if_needed(da: xr.DataArray, label: str) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    vmax = float(np.nanpercentile(np.abs(da.values), 95)) if da.size else np.nan
    if "m**2" in units or "m2 s-2" in units or "m^2" in units or vmax > 20000:
        da = da / G0
        print(f"[{label}] converted geopotential to gpm")
    da.attrs["units"] = "gpm"
    return da


def _tp_to_mm(da: xr.DataArray) -> xr.DataArray:
    units_raw = str(da.attrs.get("units", "")).strip()
    units = units_raw.lower().replace(" ", "")
    if units in ("m", "meter", "metre", "meters", "metres"):
        da = da * 1000.0
    elif units in ("kgm**-2", "kgm-2", "kg/m^2", "kgm^-2", "mm"):
        pass
    else:
        warnings.warn(f"[tp] Unknown precipitation units '{units_raw}'; not applying *1000 automatically.")
    da.attrs["units"] = "mm day-1"
    return da


def _to_bjt_daily(da: xr.DataArray, how: str) -> xr.DataArray:
    da = _normalize_time_dim(da)
    if "time" not in da.dims:
        raise ValueError(f"No time dimension in {da.dims}")
    shifted = da.assign_coords(time=pd.to_datetime(da.time.values) + BJT_OFFSET)
    daily = shifted.resample(time="1D").sum() if how == "sum" else shifted.resample(time="1D").mean()
    return daily.rename({"time": "bjt_date"})


def _check_dates(da: xr.DataArray, label: str) -> None:
    have = set(pd.to_datetime(da.bjt_date.values).strftime("%Y-%m-%d"))
    missing = [d.strftime("%Y-%m-%d") for d in ANALYSIS_DATES if d.strftime("%Y-%m-%d") not in have]
    if missing:
        raise ValueError(f"{label} missing BJT dates: {missing}")

# ====================================================
# Grouping helpers
# ====================================================

def load_total_hotcold_groups(group_file: str = GROUP_FILE) -> pd.DataFrame:
    if not os.path.exists(group_file):
        raise FileNotFoundError(f"GROUP_FILE not found: {group_file}")
    df = pd.read_csv(group_file)
    print(f"[Group file] columns={list(df.columns)}")
    init_col = _find_column(df.columns, ["init_date", "init", "initial_date", "forecast_init", "date"])
    stage_col = _find_column(df.columns, ["stage", "grouping_stage", "tmax_stage"])
    if init_col is None or stage_col is None:
        raise KeyError(f"Could not identify init/stage columns; columns={list(df.columns)}")

    group_col = _find_column(df.columns, ["hotcold_group", "group", "hotcold", "tmax_group"])
    member_col = _find_column(df.columns, ["member", "members", "member_id", "number"])
    rows = []
    for init_date in INIT_DATES:
        df_init = df[_init_match(df[init_col], init_date)].copy()
        df_total = df_init[_total_stage_mask(df_init[stage_col])].copy()
        if df_total.empty:
            raise ValueError(f"No Total-stage rows for init={init_date}; columns={list(df.columns)}; init rows={len(df_init)}")
        if group_col is not None and member_col is not None:
            for group in GROUPS:
                sub = df_total[df_total[group_col].astype(str).map(_norm_text) == _norm_text(group)]
                members = sorted(dict.fromkeys(_parse_member_list(sub[member_col])))
                for member in members:
                    rows.append({"init_date": init_date, "grouping_stage": GROUPING_STAGE, "group": group, "member": int(member), "grouping_file": group_file})
        else:
            for group in GROUPS:
                col = _find_column(df_total.columns, [group, f"{group}_members", f"{group.lower()}_members", group.lower()])
                if col is None:
                    raise KeyError(f"Could not identify {group} member column; columns={list(df_total.columns)}; head={df_total.head().to_dict(orient='records')}")
                members = sorted(dict.fromkeys(_parse_member_list(df_total.iloc[0][col])))
                for member in members:
                    rows.append({"init_date": init_date, "grouping_stage": GROUPING_STAGE, "group": group, "member": int(member), "grouping_file": group_file})
    out = pd.DataFrame(rows)
    for init_date in INIT_DATES:
        for group in GROUPS:
            members = get_group_members(out, init_date, group)
            if len(members) != 17:
                raise ValueError(f"Expected 17 members for init={init_date}, group={group}, got {len(members)}")
    return out


def get_group_members(group_df: pd.DataFrame, init_date: str, group: str) -> List[int]:
    sub = group_df[(group_df["init_date"].astype(str) == str(init_date)) & (group_df["grouping_stage"] == GROUPING_STAGE) & (group_df["group"] == group)]
    members = sub["member"].astype(int).tolist()
    if len(members) != 17:
        raise ValueError(f"Expected 17 {GROUPING_STAGE} members for init={init_date}, group={group}, got {len(members)}. members={members}")
    if len(set(members)) != 17:
        raise ValueError(f"Duplicated members for init={init_date}, group={group}: {members}")
    if min(members) < 0 or max(members) > 50:
        raise ValueError(f"Member IDs should be 0..50, got init={init_date}, group={group}, members={members}")
    return members


def build_group_audit(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for init_date in INIT_DATES:
        for group in GROUPS:
            members = get_group_members(group_df, init_date, group)
            rows.append({
                "init_date": init_date,
                "grouping_stage": GROUPING_STAGE,
                "group": group,
                "member_count": len(members),
                "members": ",".join(str(m) for m in members),
                "grouping_file": GROUP_FILE,
            })
            print(f"[Group audit] init={init_date} group={group} n_members={len(members)} members={members}")
    return pd.DataFrame(rows)

# ====================================================
# S2S readers and daily fields
# ====================================================

def _hindcast_dir_for_init(init_date: str) -> str:
    key = pd.Timestamp(init_date).strftime("%Y-%m-%d")
    if key not in HINDCAST_DIRS:
        raise KeyError(f"No hindcast directory configured for init_date={key}")
    return HINDCAST_DIRS[key]


def _format_candidate_paths(patterns: Sequence[str], init_date: str) -> List[str]:
    init = pd.Timestamp(init_date)
    fields = {
        "init_date": init.strftime("%Y-%m-%d"),
        "init_yyyymmdd": init.strftime("%Y%m%d"),
        "init_mmdd": init.strftime("%m%d"),
        "init_yyyy_mm": init.strftime("%Y-%m"),
        "hindcast_dir": _hindcast_dir_for_init(init.strftime("%Y-%m-%d")),
    }
    return [pattern.format(**fields) for pattern in patterns]


def _resolve_patterns(patterns: Sequence[str], init_date: str) -> List[str]:
    paths = []
    for candidate in _format_candidate_paths(patterns, init_date):
        if glob.has_magic(candidate):
            paths.extend(glob.glob(candidate, recursive=True))
        elif os.path.exists(candidate):
            paths.append(candidate)
    return sorted(dict.fromkeys(paths), key=paths.index)


def _open_cfgrib(path: str, short_name: Optional[str] = None) -> xr.Dataset:
    backend_kwargs = {"indexpath": ""}
    if short_name is not None:
        backend_kwargs["filter_by_keys"] = {"shortName": short_name}
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)


def _member_dim(da: xr.DataArray) -> Optional[str]:
    for dim in ("member", "number", "realization", "ensemble", "perturbationNumber"):
        if dim in da.dims:
            return dim
    return None


def _normalize_member_dim(da: xr.DataArray, stream: str) -> xr.DataArray:
    mdim = _member_dim(da)
    if mdim is None:
        da = da.expand_dims(member=[0 if stream == "cf" else 1])
    elif mdim != "member":
        da = da.rename({mdim: "member"})
    offset = 0 if stream == "cf" else 1
    return da.assign_coords(member=np.arange(offset, offset + da.sizes["member"]))


def _concat_cf_pf(cf: Optional[xr.DataArray], pf: Optional[xr.DataArray]) -> xr.DataArray:
    parts = [_normalize_member_dim(da, stream) for stream, da in (("cf", cf), ("pf", pf)) if da is not None]
    if not parts:
        raise FileNotFoundError("Neither CF nor PF field was loaded")
    out = xr.concat(parts, dim="member", coords="minimal", compat="override")
    return out.assign_coords(member=np.arange(out.sizes["member"]))


def _init_coord_name(da: xr.DataArray) -> Optional[str]:
    for name in ("time", "forecast_reference_time", "date"):
        if name in da.coords or name in da.dims:
            return name
    return None


def get_s2s_init_slice(da: xr.DataArray, init_date: str, realtime: bool) -> xr.DataArray:
    coord = _init_coord_name(da)
    if coord is None:
        if realtime:
            raise ValueError(
                f"Realtime initialization coordinate missing: requested={init_date}, "
                f"coords={list(da.coords)}, dims={da.dims}, sizes={dict(da.sizes)}"
            )
        return da
    init = pd.Timestamp(init_date)
    raw_values = np.asarray(da[coord].values).ravel()
    times = pd.to_datetime(raw_values)
    if realtime:
        match_indices = [
            index
            for index, timestamp in enumerate(times)
            if pd.Timestamp(timestamp).date() == init.date()
        ]
        available = [str(pd.Timestamp(timestamp)) for timestamp in times]
        if not match_indices:
            raise ValueError(
                f"Realtime initialization not found: requested={init_date}, coord={coord}, "
                f"available_initializations={available}, dims={da.dims}, sizes={dict(da.sizes)}"
            )
        if len(match_indices) != 1:
            matched = [available[index] for index in match_indices]
            raise ValueError(
                f"Realtime initialization is not unique: requested={init_date}, coord={coord}, "
                f"matched_initializations={matched}, available_initializations={available}, "
                f"dims={da.dims}, sizes={dict(da.sizes)}"
            )
        match_index = match_indices[0]
        matched_value = raw_values[match_index]
        if coord in da.indexes:
            da = da.sel({coord: matched_value})
        elif da[coord].ndim == 1:
            da = da.isel({da[coord].dims[0]: match_index})
        print(
            f"[Init select] requested={init_date} coord={coord} "
            f"matched={available[match_index]} remaining_dims={da.dims}"
        )
    else:
        matches = [timestamp for timestamp in times if pd.Timestamp(timestamp).date() == init.date()]
        if matches:
            da = da.sel({coord: matches})
        else:
            da = da.rename({coord: "hindcast_time"})
    if coord in da.dims and da.sizes.get(coord) == 1:
        da = da.squeeze(coord, drop=True)
    return da


def _load_one_stream(kind: str, stream: str, product: str, init_date: str, candidates: Sequence[str], realtime: bool) -> xr.DataArray:
    if product not in GRIB_SHORTNAME_ORDER:
        raise KeyError(
            f"No product-specific GRIB shortName rule for kind={kind}, "
            f"stream={stream}, product={product}"
        )
    patterns = S2S_FILE_CATALOG[kind][stream].get(product, [])
    paths = _resolve_patterns(patterns, init_date)
    if not paths:
        tried = _format_candidate_paths(patterns, init_date)
        raise FileNotFoundError(
            f"No GRIB file matched: kind={kind}, stream={stream}, product={product}, "
            f"attempted_paths={tried}, attempted_shortNames={GRIB_SHORTNAME_ORDER[product]}"
        )
    errors = []
    for path in paths:
        for sn in GRIB_SHORTNAME_ORDER[product]:
            try:
                ds = _open_cfgrib(path, short_name=sn)
                print(f"[Open] {kind}/{stream}/{product}: {os.path.basename(path)} shortName={sn} vars={list(ds.data_vars)}")
                var = _guess_data_var(ds, candidates)
                da = get_s2s_init_slice(ds[var], init_date, realtime=realtime)
                if realtime:
                    remaining_init_dims = {
                        name: int(da.sizes[name])
                        for name in ("time", "forecast_reference_time", "date")
                        if name in da.dims and da.sizes[name] > 1
                    }
                    if remaining_init_dims:
                        raise ValueError(
                            f"Realtime field retains multiple initializations after selection: "
                            f"requested={init_date}, kind={kind}, stream={stream}, "
                            f"product={product}, path={path}, remaining_init_dims={remaining_init_dims}, "
                            f"dims={da.dims}, sizes={dict(da.sizes)}"
                        )
                print(f"[Loaded] {kind}/{stream}/{product}: {os.path.basename(path)} -> {var}")
                return da
            except Exception as exc:
                print(f"[Skip] {path} shortName={sn}: {exc}")
                errors.append(f"path={path}, shortName={sn}, error={exc!r}")
    raise RuntimeError(
        f"GRIB product load failed: kind={kind}, stream={stream}, product={product}, "
        f"paths={paths}, attempted_shortNames={GRIB_SHORTNAME_ORDER[product]}, "
        f"observed_errors={errors}"
    )


def load_s2s_ensemble_with_cf(kind: str, product: str, init_date: str, candidates: Sequence[str], realtime: bool) -> xr.DataArray:
    cf = _load_one_stream(kind, "cf", product, init_date, candidates, realtime)
    pf = _load_one_stream(kind, "pf", product, init_date, candidates, realtime)
    da = _concat_cf_pf(cf, pf)
    da = _sel_region(da, DATA_EXTENT)
    print(f"[Members] {kind}/{product}/{init_date}: member_count={int(da.sizes.get('member', 0))}")
    return da


def _build_bjt_valid_coord(da: xr.DataArray, init_date: str) -> xr.DataArray:
    if "step" not in da.coords:
        if "valid_time" in da.coords:
            vt = pd.to_datetime(da.valid_time.values) + BJT_OFFSET
            time_dim = "valid_time" if "valid_time" in da.dims else da.valid_time.dims[-1]
            return da.assign_coords(bjt_valid_time=(time_dim, vt)).swap_dims({time_dim: "bjt_valid_time"})
        raise ValueError("S2S field has no step or valid_time coordinate")
    init = pd.Timestamp(init_date)
    bjt = pd.to_datetime([init + pd.Timedelta(s) + BJT_OFFSET for s in da.step.values])
    return da.assign_coords(bjt_valid_time=("step", bjt)).swap_dims({"step": "bjt_valid_time"})


def process_s2s_bjt_daily_mean(da: xr.DataArray, init_date: str) -> xr.DataArray:
    da = _build_bjt_valid_coord(da, init_date)
    return da.resample(bjt_valid_time="1D").mean().rename({"bjt_valid_time": "bjt_date"})


def _interval_day_fractions(start: pd.Timestamp, end: pd.Timestamp):
    if end <= start:
        return []
    total = (end - start).total_seconds()
    pieces = []
    cursor = start
    while cursor < end:
        stop = min(end, cursor.normalize() + pd.Timedelta(days=1))
        pieces.append((cursor.normalize(), (stop - cursor).total_seconds() / total))
        cursor = stop
    return pieces


def _apportion_accum_to_calendar_days(da: xr.DataArray, init_date: str, force_non_negative: bool = False) -> xr.DataArray:
    if "step" not in da.coords:
        raise ValueError("Accumulated S2S field requires a step coordinate")
    init = pd.Timestamp(init_date)
    steps = [pd.Timedelta(s) for s in da.step.values]
    bjt_valid = [init + s + BJT_OFFSET for s in steps]
    inc = da.diff("step")
    if force_non_negative:
        inc = inc.where(inc >= 0.0, 0.0)
    inc = inc.assign_coords(step=da.step.values[1:])
    apportioned = []
    for i in range(1, len(steps)):
        inc_i = inc.isel(step=i - 1).drop_vars("step", errors="ignore")
        for day, frac in _interval_day_fractions(bjt_valid[i - 1], bjt_valid[i]):
            apportioned.append((inc_i * frac).expand_dims(bjt_date=[day]))
    if not apportioned:
        raise ValueError("No accumulated increments available for BJT day apportionment")
    return xr.concat(apportioned, dim="bjt_date").groupby("bjt_date").sum(skipna=True)


def process_s2s_bjt_daily_accum(da: xr.DataArray, init_date: str, output: str, force_non_negative: bool = False) -> xr.DataArray:
    daily_amount = _apportion_accum_to_calendar_days(da, init_date, force_non_negative=force_non_negative)
    daily_amount.attrs.update(da.attrs)
    if output == "tp":
        return _tp_to_mm(daily_amount)
    raise ValueError(output)


def build_daily_s2s_member_fields(init_date: str) -> Dict[str, xr.DataArray]:
    print(f"\n=== Loading S2S realtime/hindcast fields for init={init_date} ===")
    tp_raw = load_s2s_ensemble_with_cf("realtime", "sfc_tp", init_date, ["tp", "total_precipitation"], realtime=True)
    print(f"[TP raw] init={init_date} dims={tp_raw.dims}")
    print(f"[TP raw] init={init_date} sizes={dict(tp_raw.sizes)}")
    if "step" in tp_raw.coords:
        print(f"[TP raw] init={init_date} first_8_steps={tp_raw.step.values[:8]}")
    else:
        print(f"[TP raw] init={init_date} first_8_steps=<no step coordinate>")
    print(f"[TP raw] init={init_date} units={tp_raw.attrs.get('units', '')}")
    z500_raw = load_s2s_ensemble_with_cf("realtime", "pl_500", init_date, ["gh", "z", "geopotential"], realtime=True)
    print(f"[Z500 raw realtime] init={init_date} dims={z500_raw.dims}")
    print(f"[Z500 raw realtime] init={init_date} sizes={dict(z500_raw.sizes)}")

    tp_daily_all = process_s2s_bjt_daily_accum(tp_raw, init_date, output="tp", force_non_negative=True)
    print(
        f"[TP daily] init={init_date} bjt_date_range="
        f"{pd.to_datetime(tp_daily_all.bjt_date.values).min()} to {pd.to_datetime(tp_daily_all.bjt_date.values).max()} "
        f"min/max={float(tp_daily_all.min(skipna=True)):.6g}/{float(tp_daily_all.max(skipna=True)):.6g}"
    )
    tp_daily = tp_daily_all.sel(bjt_date=ANALYSIS_DATES)
    z500_abs_daily = process_s2s_bjt_daily_mean(_to_gpm_if_needed(_select_level(z500_raw, 500), "S2S Z500 realtime"), init_date).sel(bjt_date=ANALYSIS_DATES)
    print(
        f"[Z500 absolute daily] init={init_date} bjt_date_range="
        f"{pd.to_datetime(z500_abs_daily.bjt_date.values).min()} to {pd.to_datetime(z500_abs_daily.bjt_date.values).max()} "
        f"min/max={float(z500_abs_daily.min(skipna=True)):.6g}/{float(z500_abs_daily.max(skipna=True)):.6g}"
    )

    _check_dates(tp_daily, f"S2S TP {init_date}")
    _check_dates(z500_abs_daily, f"S2S Z500 absolute {init_date}")
    return {"tp": tp_daily, "z500_abs": z500_abs_daily}


def _period_mean(da: xr.DataArray) -> xr.DataArray:
    return da.sel(bjt_date=ANALYSIS_DATES).mean("bjt_date", skipna=True)


def group_mean(da: xr.DataArray, members: List[int]) -> xr.DataArray:
    return da.sel(member=members).mean("member", skipna=True)


def build_total_composites_for_init(init_date: str, group_df: pd.DataFrame, fields: Dict[str, xr.DataArray]) -> Dict[str, Dict[str, xr.DataArray]]:
    out: Dict[str, Dict[str, xr.DataArray]] = {}
    for group in GROUPS:
        members = get_group_members(group_df, init_date, group)
        out[group] = {
            "members": members,
            "tp": group_mean(_period_mean(fields["tp"]), members),
            "z500_abs": group_mean(_period_mean(fields["z500_abs"]), members),
        }
    out[DIFF_GROUP] = {
        "members": None,
        "tp": out["Hot17"]["tp"] - out["Cold17"]["tp"],
        "z500_abs": out["Hot17"]["z500_abs"] - out["Cold17"]["z500_abs"],
    }
    for group in ["Hot17", "Cold17", DIFF_GROUP]:
        z = out[group]["z500_abs"]
        if z.ndim != 2:
            raise ValueError(
                f"Composite Z500 absolute field must be 2D before plotting, "
                f"init={init_date}, group={group}, dims={z.dims}, sizes={dict(z.sizes)}"
            )
    return out

# ====================================================
# ERA5/MSWEP reference fields
# ====================================================

def load_mswep_period_mean() -> xr.DataArray:
    files = sorted(glob.glob(os.path.join(MSWEP_2023_DIR, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No MSWEP files found in {MSWEP_2023_DIR}")
    parts = []
    for f in files:
        ds = _open_dataset_safe(f)
        var = _guess_data_var(ds, ["precipitation", "precip", "pr", "tp"])
        da = _standardize_lat_lon(_normalize_time_dim(ds[var]))
        parts.append(_sel_region(da, DATA_EXTENT))
    pr = xr.concat(parts, dim="time").sortby("time")
    pr_daily = _tp_to_mm(_to_bjt_daily(pr, "sum")).sel(bjt_date=ANALYSIS_DATES)
    _check_dates(pr_daily, "MSWEP TP")
    return _period_mean(pr_daily)


def load_era5_z500_daily(path: str, dates: Sequence[pd.Timestamp]) -> xr.DataArray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ds = _open_dataset_safe(path)
    var = _guess_data_var(ds, ["z", "geopotential", "z500", "gh", "hgt"])
    da = _standardize_lat_lon(_normalize_time_dim(_select_level(ds[var], 500)))
    da = _to_gpm_if_needed(da, f"ERA5 Z500 {os.path.basename(path)}")
    daily = _to_bjt_daily(da, "mean").sel(bjt_date=pd.to_datetime(dates))
    return _sel_region(daily, DATA_EXTENT)


def load_reference_fields() -> Dict[str, xr.DataArray]:
    mswep_tp = load_mswep_period_mean()
    z500_2023 = load_era5_z500_daily(ERA5_Z500_2023_FILE, ANALYSIS_DATES)
    _check_dates(z500_2023, "ERA5 Z500 absolute")
    return {"tp": mswep_tp, "z500_abs": _period_mean(z500_2023)}

# ====================================================
# Plotting
# ====================================================

def _setup_map(ax):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    ax.add_feature(cfeature.LAKES, linewidth=0.25, edgecolor="0.5", facecolor="none")
    gl = ax.gridlines(draw_labels=False, linewidth=0.25, linestyle="--", alpha=0.45)
    try:
        gl.xlocator = plt.MultipleLocator(10)
        gl.ylocator = plt.MultipleLocator(5)
    except Exception:
        pass


def _plot_nchn_box(ax):
    lon1, lon2, lat1, lat2 = NCHN_BOX
    xs = [lon1, lon2, lon2, lon1, lon1]
    ys = [lat1, lat1, lat2, lat2, lat1]
    ax.plot(xs, ys, color="black", lw=1.2, ls="--", transform=ccrs.PlateCarree(), zorder=30)


def _apply_label_effects(texts):
    for txt in texts:
        txt.set_color("0.10")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def _ensure_2d_z500(z: xr.DataArray, label: str) -> xr.DataArray:
    z = z.squeeze(drop=True)
    if z.ndim != 2:
        raise ValueError(f"{label} must be 2D, got dims={z.dims}, sizes={dict(z.sizes)}")
    return z


def _contour_z500_abs(ax, z: xr.DataArray):
    z = _ensure_2d_z500(z, "Z500 absolute contour field")
    levels = np.arange(5520, 5921, 40)
    zmin = float(z.min(skipna=True))
    zmax = float(z.max(skipna=True))
    levels = levels[(levels >= zmin) & (levels <= zmax)]
    if levels.size:
        cs = ax.contour(z.longitude, z.latitude, z, levels=levels, colors="0.20", linewidths=0.8, transform=ccrs.PlateCarree())
        _apply_label_effects(ax.clabel(cs, fmt="%.0f", fontsize=6.5, inline=True))
    if zmin <= 5880.0 <= zmax:
        ax.contour(
            z.longitude,
            z.latitude,
            z,
            levels=[5880.0],
            colors="purple",
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
        )


def _contour_z500_diff(ax, z: xr.DataArray):
    z = _ensure_2d_z500(z, "Z500 difference contour field")
    levels = np.array([10, 20, 40, 60, 80], dtype=float)
    vals = np.asarray(z.values)
    vmax = float(np.nanmax(np.abs(vals))) if vals.size else 0.0
    levels = levels[levels <= vmax]
    label_levels = levels[levels >= 20]
    neg = -levels[::-1]
    pos = levels
    if neg.size:
        csn = ax.contour(z.longitude, z.latitude, z, levels=neg, colors="royalblue", linestyles="--", linewidths=0.8, transform=ccrs.PlateCarree())
        label_neg = -label_levels[::-1]
        if label_neg.size:
            _apply_label_effects(ax.clabel(csn, levels=label_neg, fmt="%.0f", fontsize=6.5, inline=True))
    if pos.size:
        csp = ax.contour(z.longitude, z.latitude, z, levels=pos, colors="firebrick", linestyles="-", linewidths=0.8, transform=ccrs.PlateCarree())
        if label_levels.size:
            _apply_label_effects(ax.clabel(csp, levels=label_levels, fmt="%.0f", fontsize=6.5, inline=True))

def _abs_tp_levels(arrays: Sequence[xr.DataArray]) -> np.ndarray:
    pool = np.concatenate([np.asarray(a.values).ravel() for a in arrays])
    pmax = max(float(np.nanpercentile(pool, 95)), 1.0)
    pmax = min(pmax, 25.0)
    return np.linspace(0, pmax, 13)


def _diff_tp_levels(arrays: Sequence[xr.DataArray]) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    pool = np.concatenate([np.asarray(a.values).ravel() for a in arrays])
    vmax = max(float(np.nanpercentile(np.abs(pool), 98)), 0.5)
    levels = np.linspace(-vmax, vmax, 13)
    return levels, mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _plot_abs_panel(ax, tp: xr.DataArray, z: xr.DataArray, levels: np.ndarray, title: str):
    _setup_map(ax)
    cf = ax.contourf(tp.longitude, tp.latitude, tp, levels=levels, cmap="Blues", extend="max", transform=ccrs.PlateCarree())
    _contour_z500_abs(ax, z)
    _plot_nchn_box(ax)
    if title:
        ax.set_title(title, fontsize=9)
    return cf


def _plot_diff_panel(ax, tp: xr.DataArray, z: xr.DataArray, levels: np.ndarray, norm: mcolors.TwoSlopeNorm, title: str):
    _setup_map(ax)
    cf = ax.contourf(tp.longitude, tp.latitude, tp, levels=levels, cmap="RdBu_r", norm=norm, extend="both", transform=ccrs.PlateCarree())
    _contour_z500_diff(ax, z)
    _plot_nchn_box(ax)
    if title:
        ax.set_title(title, fontsize=9)
    return cf


def _save_fig(fig, stem: str) -> Tuple[str, str]:
    png = os.path.join(OUT_DIR, f"{stem}.png")
    pdf = os.path.join(OUT_DIR, f"{stem}.pdf")
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png, pdf


def plot_leadtime_composite_figure(composites: Dict[str, Dict[str, Dict[str, xr.DataArray]]], reference: Dict[str, xr.DataArray]) -> Tuple[str, str]:
    abs_arrays = [reference["tp"]]
    diff_arrays = []
    for init_date in INIT_DATES:
        abs_arrays.extend([composites[init_date]["Hot17"]["tp"], composites[init_date]["Cold17"]["tp"]])
        diff_arrays.append(composites[init_date][DIFF_GROUP]["tp"])
    abs_levels = _abs_tp_levels(abs_arrays)
    diff_levels, diff_norm = _diff_tp_levels(diff_arrays)

    fig = plt.figure(figsize=(13.2, 15.2))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1.0], hspace=0.22, wspace=0.08)
    axes = np.empty((4, 3), dtype=object)
    for r in range(4):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree())
    ax_ref = fig.add_subplot(gs[4, 0], projection=ccrs.PlateCarree())
    cbar_block_gs = gs[4, 1:].subgridspec(
        5,
        3,
        height_ratios=[1, 0.16, 0.60, 0.16, 1],
        width_ratios=[0.12, 0.76, 0.12],
        hspace=0.70,
        wspace=0.0,
    )
    cax_abs = fig.add_subplot(cbar_block_gs[1, 1])
    cax_diff = fig.add_subplot(cbar_block_gs[3, 1])

    col_titles = ["Total Hot17", "Total Cold17", "Total Hot17 − Cold17"]
    cf_abs = None
    cf_diff = None
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10, fontweight="bold")
    for r, init_date in enumerate(INIT_DATES):
        comp = composites[init_date]
        cf_abs = _plot_abs_panel(axes[r, 0], comp["Hot17"]["tp"], comp["Hot17"]["z500_abs"], abs_levels, "")
        _plot_abs_panel(axes[r, 1], comp["Cold17"]["tp"], comp["Cold17"]["z500_abs"], abs_levels, "")
        cf_diff = _plot_diff_panel(axes[r, 2], comp[DIFF_GROUP]["tp"], comp[DIFF_GROUP]["z500_abs"], diff_levels, diff_norm, "")
        axes[r, 0].text(-0.12, 0.5, f"Init {init_date}", transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")

    _plot_abs_panel(ax_ref, reference["tp"], reference["z500_abs"], abs_levels, "MSWEP TP + ERA5 Z500")

    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.09)
    cbar_abs = fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_label("TP (mm day$^{-1}$)")
    cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_label("Hot17 − Cold17 TP (mm day$^{-1}$)")
    fig.suptitle("Lead-time dependence of total-period precipitation and 500 hPa circulation", fontsize=14, y=0.975)
    return _save_fig(fig, "figureX_leadtime_total_hotcold_composite_tp_z500_abs")

# ====================================================
# QC
# ====================================================

def _field_stats(da: xr.DataArray) -> Dict[str, float]:
    return {
        "min": float(da.min(skipna=True)),
        "max": float(da.max(skipna=True)),
        "mean": float(da.mean(skipna=True)),
    }



def _tp_max_location_stats(tp: xr.DataArray) -> Dict[str, float]:
    tp = _standardize_lat_lon(tp).squeeze(drop=True)
    if tp.ndim != 2:
        raise ValueError(f"TP field must be 2D for max-location diagnostics, dims={tp.dims}, sizes={dict(tp.sizes)}")

    vals = np.asarray(tp.values)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return {
            "TP_max_lat": np.nan,
            "TP_max_lon": np.nan,
            "TP_max_value": np.nan,
            "TP_NCHN_mean": np.nan,
            "TP_NCHN_max": np.nan,
        }

    iy, ix = np.unravel_index(int(np.nanargmax(vals)), vals.shape)
    nchn = _sel_region(tp, NCHN_BOX)
    return {
        "TP_max_lat": float(tp.latitude.values[iy]),
        "TP_max_lon": float(tp.longitude.values[ix]),
        "TP_max_value": float(vals[iy, ix]),
        "TP_NCHN_mean": float(nchn.mean(skipna=True)) if nchn.size else np.nan,
        "TP_NCHN_max": float(nchn.max(skipna=True)) if nchn.size else np.nan,
    }

def build_qc(composites: Dict[str, Dict[str, Dict[str, xr.DataArray]]]) -> pd.DataFrame:
    rows = []
    for init_date in INIT_DATES:
        for group in ["Hot17", "Cold17", DIFF_GROUP]:
            fields = composites[init_date][group]
            tp_stats = _field_stats(fields["tp"])
            tp_location_stats = _tp_max_location_stats(fields["tp"])
            z_stats = _field_stats(fields["z500_abs"])
            members = fields.get("members")
            print(
                f"[TP max location] init={init_date} group={group} "
                f"lat={tp_location_stats['TP_max_lat']:.3f} "
                f"lon={tp_location_stats['TP_max_lon']:.3f} "
                f"value={tp_location_stats['TP_max_value']:.3f} "
                f"NCHN_mean={tp_location_stats['TP_NCHN_mean']:.3f} "
                f"NCHN_max={tp_location_stats['TP_NCHN_max']:.3f}"
            )
            rows.append({
                "init_date": init_date,
                "grouping_stage": GROUPING_STAGE,
                "analysis_window": ANALYSIS_WINDOW_LABEL,
                "group": group,
                "member_count": len(members) if members is not None else np.nan,
                "TP_domain_mean": tp_stats["mean"],
                "Z500_abs_domain_mean": z_stats["mean"],
                "TP_min": tp_stats["min"],
                "TP_max": tp_stats["max"],
                "TP_max_lat": tp_location_stats["TP_max_lat"],
                "TP_max_lon": tp_location_stats["TP_max_lon"],
                "TP_max_value": tp_location_stats["TP_max_value"],
                "TP_NCHN_mean": tp_location_stats["TP_NCHN_mean"],
                "TP_NCHN_max": tp_location_stats["TP_NCHN_max"],
                "Z500_abs_min": z_stats["min"],
                "Z500_abs_max": z_stats["max"],
                "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
            })
    return pd.DataFrame(rows)

# ====================================================
# Runner
# ====================================================

def run_leadtime_total_hotcold_composite_maps(make_plots: bool = True):
    group_df = load_total_hotcold_groups(GROUP_FILE)
    audit = build_group_audit(group_df)
    audit_path = os.path.join(OUT_DIR, "figureX_leadtime_total_hotcold_group_member_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(f"[Audit] saved: {audit_path}")

    composites = {}
    for init_date in INIT_DATES:
        fields = build_daily_s2s_member_fields(init_date)
        composites[init_date] = build_total_composites_for_init(init_date, group_df, fields)
        tp_diff = composites[init_date][DIFF_GROUP]["tp"]
        z_diff = composites[init_date][DIFF_GROUP]["z500_abs"]
        print(f"[Diff range] init={init_date} TP_diff min/max={float(tp_diff.min()):.3f}/{float(tp_diff.max()):.3f}")
        print(f"[Diff range] init={init_date} Z500_abs_diff min/max={float(z_diff.min()):.3f}/{float(z_diff.max()):.3f}")

    qc = build_qc(composites)
    qc_path = os.path.join(OUT_DIR, "figureX_leadtime_total_hotcold_composite_z500_abs_qc.csv")
    qc.to_csv(qc_path, index=False)
    print(f"[QC] saved: {qc_path}")

    png_path = pdf_path = None
    if make_plots:
        reference = load_reference_fields()
        png_path, pdf_path = plot_leadtime_composite_figure(composites, reference)
        print(f"[Figure] saved PNG: {png_path}")
        print(f"[Figure] saved PDF: {pdf_path}")

    return {"audit_csv": audit_path, "qc_csv": qc_path, "png": png_path, "pdf": pdf_path}


# ====================================================
# Member-level spatial significance export (Fig.9)
# ====================================================

def _fig9_significance_domain(da: xr.DataArray) -> xr.DataArray:
    """Select the fixed display/FDR domain on the native S2S grid."""
    lon = da["longitude"]
    lat = da["latitude"]
    lon_mask = (lon >= SIGNIFICANCE_DOMAIN[0]) & (lon <= SIGNIFICANCE_DOMAIN[1])
    lat_mask = (lat >= SIGNIFICANCE_DOMAIN[2]) & (lat <= SIGNIFICANCE_DOMAIN[3])
    return da.sel(longitude=lon[lon_mask], latitude=lat[lat_mask])


def _fig9_validate_groups(hot_members: Sequence[int], cold_members: Sequence[int], init_date: str) -> None:
    if len(hot_members) != 17 or len(cold_members) != 17:
        raise ValueError(f"Fig.9 {init_date} significance requires 17 vs 17 members")
    if list(hot_members) == list(cold_members):
        raise ValueError(f"Fig.9 {init_date}: Hot17 and Cold17 member lists are identical")
    overlap = sorted(set(hot_members).intersection(cold_members))
    if overlap:
        raise ValueError(f"Fig.9 {init_date}: Hot17/Cold17 overlap: {overlap}")


def _fig9_member_period_mean_with_completeness_qc(
    daily_field: xr.DataArray,
    selected_members: Sequence[int],
    init_date: str,
    variable: str,
) -> Tuple[xr.DataArray, dict]:
    """Require all 11 BJT days before forming each selected member's period mean."""
    expected_day_count = len(ANALYSIS_DATES)
    if expected_day_count != 11:
        raise ValueError(
            f"Fig.9 {init_date} {variable}: expected configured Total period to contain "
            f"11 BJT days, got {expected_day_count}"
        )
    daily = daily_field.sel(
        bjt_date=ANALYSIS_DATES, member=list(selected_members)
    )
    valid_day_count = daily.count("bjt_date").transpose(
        "member", "latitude", "longitude"
    )
    domain = (
        (valid_day_count.latitude.values[:, None] >= SIGNIFICANCE_DOMAIN[2])
        & (valid_day_count.latitude.values[:, None] <= SIGNIFICANCE_DOMAIN[3])
        & (valid_day_count.longitude.values[None, :] >= SIGNIFICANCE_DOMAIN[0])
        & (valid_day_count.longitude.values[None, :] <= SIGNIFICANCE_DOMAIN[1])
    )
    if not domain.any():
        raise ValueError(
            f"Fig.9 {init_date} {variable}: native grid has no points in the fixed "
            "90–135E, 25–55N completeness-check domain"
        )
    domain_3d = np.broadcast_to(domain, valid_day_count.shape)
    counts = valid_day_count.values
    incomplete = domain_3d & (counts < expected_day_count)
    affected_member_mask = incomplete.any(axis=(1, 2))
    affected_members = valid_day_count.member.values[affected_member_mask].tolist()
    minimum = int(counts[domain_3d].min())
    maximum = int(counts[domain_3d].max())
    affected_member_gridpoints = int(incomplete.sum())
    affected_gridpoints = int(incomplete.any(axis=0).sum())
    completeness_qc = {
        "expected_day_count": expected_day_count,
        "minimum_valid_day_count": minimum,
        "maximum_valid_day_count": maximum,
        "member_gridpoints_with_less_than_11_days": affected_member_gridpoints,
        "affected_member_count": len(affected_members),
        "affected_members": ",".join(str(member) for member in affected_members),
        "affected_gridpoint_count": affected_gridpoints,
    }
    if affected_member_gridpoints:
        member_details = []
        for member in affected_members:
            member_counts = valid_day_count.sel(member=member).values
            member_incomplete = domain & (member_counts < expected_day_count)
            member_details.append(
                f"member={member}, min_valid_days={int(member_counts[domain].min())}, "
                f"affected_member_gridpoints={int(member_incomplete.sum())}"
            )
        raise ValueError(
            f"Fig.9 Total-period completeness failure: init={init_date}, "
            f"variable={variable}, expected_day_count={expected_day_count}; "
            + "; ".join(member_details)
        )
    return daily.mean("bjt_date", skipna=True), completeness_qc


def _fig9_validate_native_grid_exact(
    datasets: Sequence[xr.Dataset], variable: str
) -> dict:
    """Fail before concat unless all initialization grids match exactly."""
    if not datasets:
        raise ValueError(f"Fig.9 {variable}: no significance datasets to concatenate")
    reference = datasets[0]
    ref_init = str(reference.init.values[0])
    ref_lat = reference.s2s_lat.values
    ref_lon = reference.s2s_lon.values
    for candidate in datasets[1:]:
        candidate_init = str(candidate.init.values[0])
        lat = candidate.s2s_lat.values
        lon = candidate.s2s_lon.values
        if ref_lat.shape != lat.shape or not np.array_equal(ref_lat, lat):
            raise ValueError(
                f"Fig.9 {variable}: native latitude grid mismatch before concat; "
                f"reference_init={ref_init}, candidate_init={candidate_init}, "
                f"reference_shape={ref_lat.shape}, candidate_shape={lat.shape}"
            )
        if ref_lon.shape != lon.shape or not np.array_equal(ref_lon, lon):
            raise ValueError(
                f"Fig.9 {variable}: native longitude grid mismatch before concat; "
                f"reference_init={ref_init}, candidate_init={candidate_init}, "
                f"reference_shape={ref_lon.shape}, candidate_shape={lon.shape}"
            )
    return {
        "grid_reference_initialization": ref_init,
        "latitude_shape": str(ref_lat.shape),
        "longitude_shape": str(ref_lon.shape),
        "latitude_min": float(np.min(ref_lat)),
        "latitude_max": float(np.max(ref_lat)),
        "latitude_order": "ascending" if ref_lat[0] < ref_lat[-1] else "descending",
        "longitude_min": float(np.min(ref_lon)),
        "longitude_max": float(np.max(ref_lon)),
        "longitude_order": "ascending" if ref_lon[0] < ref_lon[-1] else "descending",
        "all_four_inits_grid_exact_match": True,
        "xr_concat_join_policy": "exact",
    }


def _fig9_fdr_bh(
    p_value: np.ndarray, correction_domain: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    valid_test_mask = correction_domain & np.isfinite(p_value)
    adjusted = np.full(p_value.shape, np.nan, dtype=float)
    rejected = np.zeros(p_value.shape, dtype=bool)
    if valid_test_mask.any():
        reject_flat, adjusted_flat, _, _ = multipletests(
            p_value[valid_test_mask], alpha=0.05, method="fdr_bh"
        )
        adjusted[valid_test_mask] = adjusted_flat
        rejected[valid_test_mask] = reject_flat
    return adjusted, rejected, valid_test_mask, int(valid_test_mask.sum())


def _fig9_welch_panel(
    member_period_field: xr.DataArray,
    hot_members: Sequence[int],
    cold_members: Sequence[int],
    original_difference: xr.DataArray,
    units: str,
    variable: str,
    init_date: str,
) -> Tuple[xr.Dataset, dict]:
    """Test one init × variable after member-wise Total-period averaging."""
    field = member_period_field.transpose(
        "member", "latitude", "longitude"
    )
    hot = field.sel(member=list(hot_members))
    cold = field.sel(member=list(cold_members))
    hot_values = np.asarray(hot.values, dtype=float)
    cold_values = np.asarray(cold.values, dtype=float)
    n_hot = np.isfinite(hot_values).sum(axis=0).astype(np.int16)
    n_cold = np.isfinite(cold_values).sum(axis=0).astype(np.int16)
    try:
        result = ttest_ind(
            hot_values, cold_values, axis=0, equal_var=False,
            nan_policy="omit", alternative="two-sided",
        )
        alternative_explicit = True
    except TypeError:
        result = ttest_ind(
            hot_values, cold_values, axis=0, equal_var=False, nan_policy="omit"
        )
        alternative_explicit = False
    t_statistic = np.array(result.statistic, dtype=float, copy=True)
    p_value = np.array(result.pvalue, dtype=float, copy=True)
    insufficient = (n_hot < 2) | (n_cold < 2)
    invalid_test = (
        insufficient
        | ~np.isfinite(t_statistic)
        | ~np.isfinite(p_value)
    )
    t_statistic[invalid_test] = np.nan
    p_value[invalid_test] = np.nan
    correction_domain = (
        (field.latitude.values[:, None] >= SIGNIFICANCE_DOMAIN[2])
        & (field.latitude.values[:, None] <= SIGNIFICANCE_DOMAIN[3])
        & (field.longitude.values[None, :] >= SIGNIFICANCE_DOMAIN[0])
        & (field.longitude.values[None, :] <= SIGNIFICANCE_DOMAIN[1])
    )
    if not correction_domain.any():
        raise ValueError("Fig.9 significance native grid has no points in the fixed 90–135E, 25–55N domain")
    p_fdr_bh, significant_fdr, valid_test_mask, n_valid_tests = _fig9_fdr_bh(
        p_value, correction_domain
    )
    local_significant = valid_test_mask & (p_value < 0.05)
    finite_adjusted = p_fdr_bh[np.isfinite(p_fdr_bh)]
    if np.any((finite_adjusted < 0) | (finite_adjusted > 1)):
        raise ValueError(f"Fig.9 {variable} {init_date}: BH-adjusted p-values outside [0, 1]")
    finite_p = p_value[valid_test_mask]
    if finite_p.size and np.all(finite_p == finite_p[0]):
        warnings.warn(f"Fig.9 {variable} {init_date}: finite p-value field is constant")
    fdr_family_mask = np.isfinite(p_fdr_bh)
    if fdr_family_mask.any() and significant_fdr[fdr_family_mask].all():
        warnings.warn(f"Fig.9 {variable} {init_date}: entire FDR mask is significant")

    hot_mean_da = hot.mean("member", skipna=True).transpose(
        "latitude", "longitude"
    )
    cold_mean_da = cold.mean("member", skipna=True).transpose(
        "latitude", "longitude"
    )
    hot_mean = hot_mean_da.values
    cold_mean = cold_mean_da.values
    difference = hot_mean - cold_mean
    expected = original_difference.transpose("latitude", "longitude")
    mismatch = float(np.nanmax(np.abs(difference - expected.values)))
    if not np.allclose(difference, expected.values, rtol=1e-7, atol=1e-6, equal_nan=True):
        raise ValueError(
            f"Fig.9 {variable} {init_date}: member-mean difference does not match original composite; "
            f"max_abs_mismatch={mismatch:.6g}"
        )

    coords = {"s2s_lat": field.latitude.values, "s2s_lon": field.longitude.values}
    dims = ("s2s_lat", "s2s_lon")
    data_vars = {
        "hot_mean": (dims, hot_mean), "cold_mean": (dims, cold_mean),
        "hot_minus_cold": (dims, difference), "t_statistic": (dims, t_statistic),
        "p_value": (dims, p_value), "p_fdr_bh": (dims, p_fdr_bh),
        "significant_local_p005": (dims, local_significant.astype(np.int8)),
        "significant_fdr_005": (dims, significant_fdr.astype(np.int8)),
        "fdr_domain_mask": (dims, correction_domain.astype(np.int8)),
        "fdr_valid_test_mask": (dims, valid_test_mask.astype(np.int8)),
        "n_hot": (dims, n_hot), "n_cold": (dims, n_cold),
    }
    if hasattr(result, "df"):
        degrees_of_freedom = np.array(result.df, dtype=float, copy=True)
        degrees_of_freedom[invalid_test] = np.nan
        data_vars["degrees_of_freedom"] = (dims, degrees_of_freedom)
    ds = xr.Dataset(data_vars, coords=coords).expand_dims(init=[init_date])
    ds["p_fdr_bh"].attrs["description"] = "Benjamini-Hochberg adjusted p-value"
    ds["significant_local_p005"].attrs["description"] = "1 where local two-sided Welch p < 0.05 within the fixed analysis domain"
    ds["significant_fdr_005"].attrs["description"] = "1 where Benjamini-Hochberg adjusted p < 0.05"
    ds["fdr_domain_mask"].attrs.update({
        "description": "Predefined spatial domain used for the multiple-testing assessment",
        "fdr_alpha": 0.05,
        "fdr_method": "Benjamini-Hochberg",
    })
    ds["fdr_valid_test_mask"].attrs.update({
        "description": "Grid points with finite local Welch p-values actually included in the Benjamini-Hochberg adjustment",
        "fdr_alpha": 0.05,
        "fdr_method": "Benjamini-Hochberg",
    })
    ds.attrs["alternative_explicit"] = str(alternative_explicit)
    local_count = int(np.count_nonzero(local_significant))
    fdr_count = int(np.count_nonzero(significant_fdr))
    non_17 = int(np.count_nonzero(correction_domain & ((n_hot != 17) | (n_cold != 17))))
    constant_finite_p = bool(finite_p.size and np.all(finite_p == finite_p[0]))
    all_identical_members = bool(
        np.all(np.nanstd(np.concatenate([hot_values, cold_values], axis=0), axis=0) == 0)
    )
    qc = {
        "figure": "Fig.9", "panel": init_date, "variable": variable,
        "n_hot_min": int(n_hot[correction_domain].min()), "n_hot_max": int(n_hot[correction_domain].max()),
        "n_cold_min": int(n_cold[correction_domain].min()), "n_cold_max": int(n_cold[correction_domain].max()),
        "non_17_vs_17_gridpoints": non_17,
        "invalid_ttest_gridpoints_with_n_ge_2": int(np.count_nonzero(correction_domain & (~insufficient) & invalid_test)),
        "non_17_likely_reason": "finite-value missingness in member fields" if non_17 else "",
        "n_valid_tests": n_valid_tests, "local_p005_count": local_count,
        "local_p005_fraction": local_count / n_valid_tests if n_valid_tests else np.nan,
        "fdr_005_count": fdr_count,
        "fdr_005_fraction": fdr_count / n_valid_tests if n_valid_tests else np.nan,
        "p_value_min": float(np.min(finite_p)) if finite_p.size else np.nan,
        "p_value_max": float(np.max(finite_p)) if finite_p.size else np.nan,
        "p_fdr_bh_min": float(np.nanmin(p_fdr_bh)) if n_valid_tests else np.nan,
        "p_fdr_bh_max": float(np.nanmax(p_fdr_bh)) if n_valid_tests else np.nan,
        "all_nan_p": bool(not finite_p.size),
        "constant_finite_p": constant_finite_p,
        "all_identical_member_field": all_identical_members,
        "all_fdr_significant": bool(fdr_family_mask.any() and significant_fdr[fdr_family_mask].all()),
        "all_fdr_nonsignificant": bool(fdr_family_mask.any() and not significant_fdr[fdr_family_mask].any()),
        "max_absolute_mismatch": mismatch,
        "max_abs_composite_mismatch": mismatch, "units": units,
        "composite_consistency_rtol": 1e-7,
        "composite_consistency_atol": 1e-6,
        "correction_domain_grid_count": int(correction_domain.sum()),
        "fdr_domain_mask_saved": True,
        "fdr_valid_test_mask_saved": True,
        "latitude_order": "ascending" if field.latitude.values[0] < field.latitude.values[-1] else "descending",
        "longitude_min": float(field.longitude.min()), "longitude_max": float(field.longitude.max()),
        "fdr_longitude_range": "90 to 135 E", "fdr_latitude_range": "25 to 55 N",
        "alternative_explicit": alternative_explicit,
    }
    return ds, qc


def _fig9_significance_attrs(units: str, group_rows: Sequence[str]) -> dict:
    return {
        "test": "two-sided Welch independent-samples t test",
        "null_hypothesis": "mean(Hot17) - mean(Cold17) = 0",
        "alternative_hypothesis": "mean(Hot17) - mean(Cold17) != 0",
        "equal_var": "False", "fdr_method": "Benjamini-Hochberg", "fdr_alpha": 0.05,
        "difference_direction": "Hot17 minus Cold17", "grouping_stage": GROUPING_STAGE,
        "grouping_file": GROUP_FILE, "analysis_domain": "90-135E, 25-55N",
        "analysis_window": ANALYSIS_WINDOW_LABEL, "units": units,
        "member_count_expected": 17, "member_sample_dimension": "member",
        "fdr_family": "one initialization and one variable",
        "composite_consistency_rtol": 1e-7,
        "composite_consistency_atol": 1e-6,
        "fdr_domain_mask": "included",
        "fdr_valid_test_mask": "included",
        "group_members_by_init": " | ".join(group_rows),
        "interpretation": "conditional ensemble-group contrast",
        "scipy_version": scipy.__version__, "native_grid": "S2S; no interpolation or regridding",
        "statsmodels_version": statsmodels.__version__,
    }


def _update_spatial_significance_audit_fig9(section_text: str) -> None:
    os.makedirs(os.path.dirname(SPATIAL_SIGNIFICANCE_AUDIT_PATH), exist_ok=True)
    begin, end = "<!-- BEGIN FIG9 -->", "<!-- END FIG9 -->"
    if os.path.exists(SPATIAL_SIGNIFICANCE_AUDIT_PATH):
        with open(SPATIAL_SIGNIFICANCE_AUDIT_PATH, "r", encoding="utf-8") as handle:
            content = handle.read()
    else:
        content = "# Fig.7/Fig.9 spatial-significance runtime audit\n\n"
    block = f"{begin}\n{section_text.rstrip()}\n{end}"
    if begin in content and end in content:
        content = content[:content.index(begin)] + block + content[content.index(end) + len(end):]
    else:
        content = content.rstrip() + "\n\n" + block + "\n"
    with open(SPATIAL_SIGNIFICANCE_AUDIT_PATH, "w", encoding="utf-8") as handle:
        handle.write(content)


def export_fig9_spatial_significance(
    tp_path: str = FIG9_SIGNIFICANCE_TP_PATH,
    z500_path: str = FIG9_SIGNIFICANCE_Z500_PATH,
    qc_path: str = FIG9_SIGNIFICANCE_QC_PATH,
) -> Tuple[str, str]:
    """Export eight independent Fig.9 init × variable FDR families."""
    os.makedirs(os.path.dirname(tp_path), exist_ok=True)
    group_df = load_total_hotcold_groups(GROUP_FILE)
    outputs = {"tp": [], "z500_abs": []}
    qc_rows = []
    audit_groups = []
    attrs_group_rows = []
    completeness_rows = []
    for init_date in INIT_DATES:
        hot_members = get_group_members(group_df, init_date, "Hot17")
        cold_members = get_group_members(group_df, init_date, "Cold17")
        _fig9_validate_groups(hot_members, cold_members, init_date)
        fields = build_daily_s2s_member_fields(init_date)
        selected_members = list(hot_members) + list(cold_members)
        member_period_fields = {}
        completeness_by_variable = {}
        for variable in ("tp", "z500_abs"):
            member_period_fields[variable], completeness_qc = (
                _fig9_member_period_mean_with_completeness_qc(
                    fields[variable], selected_members, init_date, variable
                )
            )
            completeness_by_variable[variable] = completeness_qc
            completeness_rows.append({
                "init_date": init_date,
                "variable": variable,
                **completeness_qc,
            })
        composites = build_total_composites_for_init(init_date, group_df, fields)
        attrs_group_rows.append(f"{init_date}:Hot17={hot_members};Cold17={cold_members}")
        audit_groups.append(
            f"- `{init_date}` Hot17 ({len(hot_members)}): `{hot_members}`; "
            f"Cold17 ({len(cold_members)}): `{cold_members}`; overlap: "
            f"`{sorted(set(hot_members).intersection(cold_members))}`"
        )
        for variable, units in (("tp", "mm day^-1"), ("z500_abs", "gpm")):
            # The day dimension is reduced independently within each member before testing.
            member_period_field = member_period_fields[variable]
            panel, qc = _fig9_welch_panel(
                member_period_field, hot_members, cold_members,
                composites[DIFF_GROUP][variable], units, variable, init_date,
            )
            qc.update(completeness_by_variable[variable])
            outputs[variable].append(panel)
            qc_rows.append(qc)

    tp_grid_qc = _fig9_validate_native_grid_exact(outputs["tp"], "tp")
    z500_grid_qc = _fig9_validate_native_grid_exact(outputs["z500_abs"], "z500_abs")
    if tp_grid_qc != z500_grid_qc:
        raise ValueError("Fig.9 TP and Z500 native-grid QC summaries differ")
    for row in qc_rows:
        row.update(tp_grid_qc)
    tp_ds = xr.concat(outputs["tp"], dim="init", join="exact")
    z500_ds = xr.concat(outputs["z500_abs"], dim="init", join="exact")
    tp_ds.attrs.update(_fig9_significance_attrs("mm day^-1", attrs_group_rows))
    z500_ds.attrs.update(_fig9_significance_attrs("gpm", attrs_group_rows))
    for significance_ds in (tp_ds, z500_ds):
        significance_ds.attrs.update({
            "expected_total_period_day_count": 11,
            "total_period_completeness_required": "11 valid BJT days at every selected-member grid point in the FDR domain",
            "grid_reference_initialization": tp_grid_qc["grid_reference_initialization"],
            "all_four_inits_grid_exact_match": "True",
            "xr_concat_join_policy": "exact",
        })
    tp_p = tp_ds.p_value.values
    z500_p = z500_ds.p_value.values
    if np.array_equal(np.isnan(tp_p), np.isnan(z500_p)) and np.array_equal(
        np.where(np.isnan(tp_p), -1.0, tp_p), np.where(np.isnan(z500_p), -1.0, z500_p)
    ):
        raise ValueError("Fig.9 TP and Z500 p-value fields are identical; possible mask/p-value reuse")
    tp_ds.to_netcdf(tp_path)
    z500_ds.to_netcdf(z500_path)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    audit_table = "```csv\n" + pd.DataFrame(qc_rows).to_csv(index=False) + "```"
    completeness_table = (
        "```csv\n" + pd.DataFrame(completeness_rows).to_csv(index=False) + "```"
    )
    section = f"""## Fig.9 runtime audit

- Source script: `cell_leadtime_total_hotcold_composite_maps.py`; plotting script: `paper_fig9_from_cache.py`.
- Grouping file: `{GROUP_FILE}`; grouping stage: `{GROUPING_STAGE}`.
{chr(10).join(audit_groups)}
- Sample axis: `member`; each member was averaged over {ANALYSIS_WINDOW_LABEL} before testing. Days and initializations were not pooled.
- Total-period completeness: exactly 11 BJT days are required at every selected-member grid point in the fixed FDR domain before `mean("bjt_date", skipna=True)` is evaluated; any shortfall raises `ValueError`.
- Welch parameters: two-sided, `equal_var=False`, `nan_policy=omit`; SciPy `{scipy.__version__}`.
- Main-figure criterion: uncorrected local two-sided Welch `p < 0.05`.
- Robustness criterion: Benjamini-Hochberg-adjusted `p < 0.05`; eight families (four initializations × TP/Z500), each restricted to 90–135 E, 25–55 N and containing only finite local Welch tests.
- Native-grid alignment: S2S latitude/longitude retained; no interpolation or regridding.
- Native-grid exact-match safeguard: reference initialization `{tp_grid_qc['grid_reference_initialization']}`; latitude shape `{tp_grid_qc['latitude_shape']}`, range `{tp_grid_qc['latitude_min']}` to `{tp_grid_qc['latitude_max']}`, order `{tp_grid_qc['latitude_order']}`; longitude shape `{tp_grid_qc['longitude_shape']}`, range `{tp_grid_qc['longitude_min']}` to `{tp_grid_qc['longitude_max']}`, order `{tp_grid_qc['longitude_order']}`; all four exact match `{tp_grid_qc['all_four_inits_grid_exact_match']}`; `xr.concat(join="exact")`.
- Composite consistency tolerance: `rtol=1e-7`, `atol=1e-6`; `max_absolute_mismatch` is recorded for every init × variable.
- FDR masks: `fdr_domain_mask` stores the predefined spatial domain; `fdr_valid_test_mask` stores domain grid points with finite local Welch p-values actually included in BH adjustment. Correction-domain and valid-test counts are recorded separately.
- Default run flags: `RUN_ORIGINAL_FIG9=False`, `RUN_EXPORT_FIG9_CACHE=False`, `RUN_EXPORT_FIG9_SIGNIFICANCE=True`.
- TP and Z500 masks were calculated independently: `True`.
- Output TP: `{tp_path}`
- Output Z500: `{z500_path}`
- QC CSV: `{qc_path}`
- Runtime audit: `{SPATIAL_SIGNIFICANCE_AUDIT_PATH}`
- Expected main figure: `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/fig9_leadtime_tp_z500_composites_from_cache_localp005.png` and `.pdf`.
- The significance markers displayed in the main figure represent uncorrected local Welch `p < 0.05`; BH-FDR-adjusted results remain stored as a robustness assessment.

{audit_table}

### 11-day completeness QC

{completeness_table}

### Interpretation boundary

These p-values describe conditional spatial differences between the predefined Hot17 and Cold17 ensemble-member groups. The groups were defined independently for each initialization by the established Total-period Tmax ranking. The results are statistical support for an ensemble-member contrast within one forecast system, not causal significance, independent confirmation of the grouping, significance of the 2023 event itself, or population inference from independent observations. BH-FDR controls the false-discovery rate across tested grid points within each specified family; it is not a stronger claim of global field significance.

### Fig.9 caption supplement

Hatching indicates grid points where the Hot17−Cold17 precipitation difference is locally significant at p < 0.05 based on a two-sided Welch’s t test. Z500 difference contours do not display significance. Tests were performed separately for precipitation and Z500 and independently for each initialization, using the 17 members in each predefined group after member-level Total-period averaging. BH-FDR-adjusted results are retained separately as a robustness assessment. No multiple-testing correction is applied to the significance markers shown in the main figure.
"""
    _update_spatial_significance_audit_fig9(section)
    print(f"[Fig9 significance TP] saved: {tp_path}")
    print(f"[Fig9 significance Z500] saved: {z500_path}")
    print(f"[Fig9 significance QC] saved: {qc_path}")
    return tp_path, z500_path


# ====================================================
# Plot-ready cache export (Fig.9)
# ====================================================

CACHE_ROOT_FIG9 = "/data1/huangy/fig6/NC/v12/paper_figures/cache"
FIG9_CACHE_PATH = os.path.join(CACHE_ROOT_FIG9, "fig9_leadtime_tp_z500_plot_cache.nc")
FIG9_CACHE_QC_PATH = os.path.join(CACHE_ROOT_FIG9, "fig9_leadtime_tp_z500_cache_qc.csv")


def _cache_map_da_fig9(da: xr.DataArray, source: str, units: str, description: str) -> xr.DataArray:
    lat_dim = f"{source}_lat"
    lon_dim = f"{source}_lon"
    rename_map = {
        old: new
        for old, new in {"latitude": lat_dim, "longitude": lon_dim}.items()
        if (old in da.dims or old in da.coords) and new not in da.dims and new not in da.coords
    }
    out = da.rename(rename_map)
    missing_dims = [dim for dim in (lat_dim, lon_dim) if dim not in out.dims]
    if missing_dims:
        raise ValueError(
            f"Fig.9 cache map field is missing required dims {missing_dims}; "
            f"got dims={out.dims}, sizes={dict(out.sizes)}"
        )
    out = out.transpose(lat_dim, lon_dim)
    out.attrs.update({"units": units, "description": description})
    return out


def _stack_init_cache_fig9(items: Sequence[xr.DataArray], units: str, description: str) -> xr.DataArray:
    out = xr.concat(items, dim=pd.Index(INIT_DATES, name="init"))
    out.attrs.update({"units": units, "description": description})
    return out


def _cache_qc_rows_fig9(ds: xr.Dataset, cache_file: str) -> List[dict]:
    rows = []
    for name, da in ds.data_vars.items():
        vals = da.values
        row = {
            "figure": "Fig.9",
            "cache_file": cache_file,
            "var_name": name,
            "dims": ",".join(da.dims),
            "shape": "x".join(str(da.sizes[d]) for d in da.dims),
            "units": da.attrs.get("units", ""),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "mean": float(np.nanmean(vals)),
            "nan_count": int(np.isnan(vals).sum()),
            "finite_fraction": float(np.isfinite(vals).sum() / vals.size) if vals.size else np.nan,
            "description": da.attrs.get("description", ""),
        }
        lat_dims = [dim for dim in da.dims if dim.endswith("_lat")]
        lon_dims = [dim for dim in da.dims if dim.endswith("_lon")]
        if name.startswith("tp_") and lat_dims and lon_dims:
            lat_dim, lon_dim = lat_dims[0], lon_dims[0]
            if "init" in da.dims:
                # aggregate location diagnostics over each init into a compact string for cache QC
                locs = []
                for init in da.init.values:
                    stats = _tp_max_location_stats(da.sel(init=init).rename({lat_dim: "latitude", lon_dim: "longitude"}))
                    locs.append(
                        f"{init}:lat={stats['TP_max_lat']:.3f},lon={stats['TP_max_lon']:.3f},"
                        f"value={stats['TP_max_value']:.3f},NCHN_mean={stats['TP_NCHN_mean']:.3f},NCHN_max={stats['TP_NCHN_max']:.3f}"
                    )
                row.update({
                    "TP_max_lat": "by_init",
                    "TP_max_lon": "by_init",
                    "TP_max_value": "by_init",
                    "TP_NCHN_mean": "by_init",
                    "TP_NCHN_max": "by_init",
                    "TP_location_detail": "; ".join(locs),
                })
            else:
                stats = _tp_max_location_stats(da.rename({lat_dim: "latitude", lon_dim: "longitude"}))
                row.update(stats)
        rows.append(row)
    return rows


def _validate_fig9_s2s_cache_density(ds: xr.Dataset) -> None:
    for name in ["tp_hot", "z500_abs_hot", "tp_cold", "z500_abs_cold", "tp_diff", "z500_abs_diff"]:
        vals = ds[name].values
        finite_fraction = float(np.isfinite(vals).sum() / vals.size) if vals.size else 0.0
        if finite_fraction < 0.8:
            raise ValueError(
                f"Fig.9 S2S cache field {name} finite_fraction={finite_fraction:.3f} < 0.8; "
                "possible coordinate-union alignment occurred"
            )


def export_fig9_plot_cache(cache_path: str = FIG9_CACHE_PATH, qc_path: str = FIG9_CACHE_QC_PATH) -> str:
    """Export plot-ready Fig.9 fields without changing science-processing logic."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    group_df = load_total_hotcold_groups(GROUP_FILE)
    composites = {}
    for init_date in INIT_DATES:
        fields = build_daily_s2s_member_fields(init_date)
        composites[init_date] = build_total_composites_for_init(init_date, group_df, fields)
    reference = load_reference_fields()

    tp_hot = []
    z_hot = []
    tp_cold = []
    z_cold = []
    tp_diff = []
    z_diff = []
    for init_date in INIT_DATES:
        comp = composites[init_date]
        tp_hot.append(_cache_map_da_fig9(comp["Hot17"]["tp"], "s2s", "mm day^-1", "Total Hot17 period-mean precipitation"))
        z_hot.append(_cache_map_da_fig9(comp["Hot17"]["z500_abs"], "s2s", "gpm", "Total Hot17 period-mean absolute Z500"))
        tp_cold.append(_cache_map_da_fig9(comp["Cold17"]["tp"], "s2s", "mm day^-1", "Total Cold17 period-mean precipitation"))
        z_cold.append(_cache_map_da_fig9(comp["Cold17"]["z500_abs"], "s2s", "gpm", "Total Cold17 period-mean absolute Z500"))
        tp_diff.append(_cache_map_da_fig9(comp[DIFF_GROUP]["tp"], "s2s", "mm day^-1", "Hot17 minus Cold17 period-mean precipitation difference"))
        z_diff.append(_cache_map_da_fig9(comp[DIFF_GROUP]["z500_abs"], "s2s", "gpm", "Hot17 minus Cold17 period-mean absolute Z500 difference"))

    abs_levels = _abs_tp_levels([
        reference["tp"],
        *[composites[init]["Hot17"]["tp"] for init in INIT_DATES],
        *[composites[init]["Cold17"]["tp"] for init in INIT_DATES],
    ])
    diff_levels, _ = _diff_tp_levels([composites[init][DIFF_GROUP]["tp"] for init in INIT_DATES])

    ds = xr.Dataset({
        "tp_hot": _stack_init_cache_fig9(tp_hot, "mm day^-1", "Total Hot17 period-mean precipitation"),
        "z500_abs_hot": _stack_init_cache_fig9(z_hot, "gpm", "Total Hot17 period-mean absolute Z500"),
        "tp_cold": _stack_init_cache_fig9(tp_cold, "mm day^-1", "Total Cold17 period-mean precipitation"),
        "z500_abs_cold": _stack_init_cache_fig9(z_cold, "gpm", "Total Cold17 period-mean absolute Z500"),
        "tp_diff": _stack_init_cache_fig9(tp_diff, "mm day^-1", "Hot17 minus Cold17 precipitation difference"),
        "z500_abs_diff": _stack_init_cache_fig9(z_diff, "gpm", "Hot17 minus Cold17 absolute Z500 difference"),
        "tp_obs_ref": _cache_map_da_fig9(reference["tp"], "mswep", "mm day^-1", "MSWEP total-period precipitation reference"),
        "z500_abs_obs_ref": _cache_map_da_fig9(reference["z500_abs"], "era5", "gpm", "ERA5 total-period absolute Z500 reference"),
        "tp_abs_levels": xr.DataArray(
            abs_levels,
            dims=("tp_abs_level",),
            attrs={
                "units": "mm day^-1",
                "description": "Fig.9 absolute TP colorbar levels computed from reference TP and all init Hot17/Cold17 TP composites",
            },
        ),
        "tp_diff_levels": xr.DataArray(
            diff_levels,
            dims=("tp_diff_level",),
            attrs={
                "units": "mm day^-1",
                "description": "Fig.9 Hot17-minus-Cold17 TP difference colorbar levels computed from all init differences",
            },
        ),
    })
    ds.attrs.update({
        "source_figure": "Fig.9 lead-time total hot/cold composites",
        "source_script": "cell_leadtime_total_hotcold_composite_maps.py",
        "analysis_window": ANALYSIS_WINDOW_LABEL,
        "grouping_file": GROUP_FILE,
        "grouping_stage": GROUPING_STAGE,
        "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
        "z500_type": "absolute",
        "z500_diff_type": "Hot17_minus_Cold17 difference of absolute Z500",
        "z500_units": "gpm",
        "tp_units": "mm day^-1",
        "map_extent": str(MAP_EXTENT),
        "nchn_box": str(NCHN_BOX),
    })
    _validate_fig9_s2s_cache_density(ds)
    ds.to_netcdf(cache_path)
    pd.DataFrame(_cache_qc_rows_fig9(ds, cache_path)).to_csv(qc_path, index=False)
    print(f"[Fig9 cache] saved: {cache_path}")
    print(f"[Fig9 cache QC] saved: {qc_path}")
    return cache_path

# ====================================================
# Notebook-friendly execution block (Fig.9)
# ====================================================

RUN_ORIGINAL_FIG9 = False
RUN_EXPORT_FIG9_CACHE = False
RUN_EXPORT_FIG9_SIGNIFICANCE = False

if (RUN_ORIGINAL_FIG9 or RUN_EXPORT_FIG9_CACHE or RUN_EXPORT_FIG9_SIGNIFICANCE) and (__name__ == "__main__" or "get_ipython" in globals()):
    if RUN_ORIGINAL_FIG9:
        run_leadtime_total_hotcold_composite_maps(make_plots=True)

    if RUN_EXPORT_FIG9_CACHE:
        export_fig9_plot_cache()

    if RUN_EXPORT_FIG9_SIGNIFICANCE:
        export_fig9_spatial_significance()
