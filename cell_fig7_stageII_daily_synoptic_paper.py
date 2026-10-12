"""Paper Fig. 7: Stage-II daily synoptic evolution.

Notebook-friendly standalone cell/script.  This file intentionally does not import
other project cells as Python modules, so it can be pasted into a Jupyter notebook
and run after no prerequisite module imports.
"""

import glob
import os
import re
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
# Paper Fig. 7 configuration
# ====================================================

TARGET_DATES = pd.to_datetime(["2023-06-18", "2023-06-19", "2023-06-20"])
MAIN_INIT_DATE = "2023-06-12"
APPENDIX_INIT_DATE = "2023-06-08"
INIT_DATES = [APPENDIX_INIT_DATE, MAIN_INIT_DATE]

MAP_EXTENT = (90.0, 135.0, 25.0, 55.0)
DATA_EXTENT = (88.0, 137.0, 23.0, 57.0)
NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
SIGNIFICANCE_DOMAIN = (90.0, 135.0, 25.0, 55.0)
BJT_OFFSET = pd.Timedelta(hours=8)
G0 = 9.80665

BASE_DATA_DIR = "/data1/huangy/fig6/NC"
CF_DIR = f"{BASE_DATA_DIR}/cf_2023-6"
ERA5_TP_FILE = f"{BASE_DATA_DIR}/MSE/ERA5/tp/era5_tp_2023_06.nc"
ERA5_Z500_2023_FILE = f"{BASE_DATA_DIR}/MSE/ERA5/Z500/era5_z500_2023_06.nc"
ERA5_UVQ_FILE_NEW = f"{BASE_DATA_DIR}/MSE/ERA5/850uv+q/era5_850_uvq_2023_06.nc"
ERA5_UVQ_FILE_OLD = f"{BASE_DATA_DIR}/MSE/ERA5/850 uv+q/era5_850_uvq_2023_06.nc"
MSWEP_2023_DIR = f"{BASE_DATA_DIR}/MSE/MSWEP/3hourly-2023"
GROUP_FILE = "/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv"
OUT_DIR = "/data1/huangy/fig6/NC/v10/paper_fig7_stageII_daily_synoptic"
os.makedirs(OUT_DIR, exist_ok=True)

VECTOR_COLOR = "#00C853"
ERA5_VECTOR_STRIDE = 24
S2S_VECTOR_STRIDE = 7
WIND_MIN_MS = 1.5
WIND_DIFF_MIN_MS = 0.5

ABS_GROUPS = ["Hot17", "Cold17"]
DIFF_GROUP = "Hot17_minus_Cold17"
GROUP_LABELS = {
    "MSWEP": "MSWEP",
    "Hot17": "Total Hot17",
    "Cold17": "Total Cold17",
    "Hot17_minus_Cold17": "Total Hot17 − Cold17",
}

FIG7_MAIN_STEMS = {
    "novectors": "figure7_stageII_daily_synoptic_evolution_main_totalgroup_init2023-06-12_0618_0620_novectors",
    "vectors": "figure7_stageII_daily_synoptic_evolution_main_totalgroup_init2023-06-12_0618_0620_vectors",
}
FIG7_APPENDIX_STEMS = {
    "novectors": "figureS_stageII_daily_synoptic_evolution_appendix_totalgroup_init2023-06-08_0618_0620_novectors",
    "vectors": "figureS_stageII_daily_synoptic_evolution_appendix_totalgroup_init2023-06-08_0618_0620_vectors",
}

SIGNIFICANCE_ROOT_FIG7 = "/data1/huangy/fig6/NC/v12/paper_figures/significance"
FIG7_SIGNIFICANCE_TP_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG7, "fig7_spatial_significance_tp.nc")
FIG7_SIGNIFICANCE_Z500_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG7, "fig7_spatial_significance_z500.nc")
FIG7_SIGNIFICANCE_QC_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG7, "fig7_spatial_significance_qc.csv")
SPATIAL_SIGNIFICANCE_AUDIT_PATH = os.path.join(
    SIGNIFICANCE_ROOT_FIG7, "fig7_fig9_spatial_significance_audit.md"
)

S2S_FILE_CATALOG = {
    "realtime": {
        "pf": {
            "pl_500": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"],
            "pl_uv": [
                f"{BASE_DATA_DIR}/MSE/ecmf_pf_131_132_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*pf*131*132*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*pf*uv*2023-06*.grb*",
            ],
            "sfc_tp": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_tp2023-06.grb"],
        },
        "cf": {
            "pl_500": [
                f"{CF_DIR}/ecmf_cf_pl_500_2023-06.grib",
                f"{CF_DIR}/ecmf_cf_pl500_156_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*pl*zqt*2023-06*.grb*",
            ],
            "pl_uv": [
                f"{CF_DIR}/ecmf_cf_pl_850_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*131*132*2023-06*.grb*",
                f"{BASE_DATA_DIR}/MSE/*cf*131*132*2023-06*.grb*",
            ],
            "sfc_tp": [
                f"{CF_DIR}/ecmf_cf_sfc_tp_2023-6.grib",
                f"{CF_DIR}/ecmf_cf_sfc_tp_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*tp*2023-06*.grb*",
            ],
        },
    }
}

# ====================================================
# Generic Fig. 7 helpers
# ====================================================

def _safe_float_fig7(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _open_dataset_safe_fig7(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except Exception as exc:
        print(f"[Error] Failed to open dataset: {path}")
        raise RuntimeError(f"Failed to open dataset: {path}. Original error: {repr(exc)}") from exc


def _guess_var_fig7(ds: xr.Dataset, candidates: Sequence[str]) -> Tuple[xr.DataArray, str]:
    lower = {v.lower(): v for v in ds.data_vars}
    for cand in candidates:
        if cand.lower() in lower:
            name = lower[cand.lower()]
            return ds[name], name
    for var in ds.data_vars:
        short_name = str(ds[var].attrs.get("GRIB_shortName", "")).lower()
        standard_name = str(ds[var].attrs.get("standard_name", "")).lower()
        long_name = str(ds[var].attrs.get("long_name", "")).lower()
        if any(c.lower() == short_name for c in candidates):
            return ds[var], var
        if any(c.lower() in standard_name or c.lower() in long_name for c in candidates):
            return ds[var], var
    if len(ds.data_vars) == 1:
        var = list(ds.data_vars)[0]
        return ds[var], var
    raise KeyError(f"Cannot find variable in {list(ds.data_vars)}; candidates={candidates}")


def _normalize_time_dim_fig7(da: xr.DataArray) -> xr.DataArray:
    if "time" in da.dims:
        return da
    if "valid_time" in da.dims:
        return da.rename({"valid_time": "time"})
    if "time" in da.coords:
        return da.expand_dims(time=[pd.to_datetime(da["time"].values).to_datetime64()])
    if "valid_time" in da.coords:
        vt = pd.to_datetime(da["valid_time"].values).to_datetime64()
        return da.drop_vars("valid_time").expand_dims(time=[vt])
    raise KeyError(f"No time/valid_time found. dims={da.dims}, coords={list(da.coords)}")


def _standardize_lat_lon_fig7(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low in ("lat", "latitude") and name != "latitude":
            rename[name] = "latitude"
        if low in ("lon", "longitude") and name != "longitude":
            rename[name] = "longitude"
    if rename:
        da = da.rename(rename)
    if "longitude" in da.coords and float(da.longitude.max()) > 180.0:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")
    if "latitude" in da.coords and da.latitude.size > 1 and float(da.latitude[0]) > float(da.latitude[-1]):
        da = da.sortby("latitude")
    return da


def _sel_region_fig7(da: xr.DataArray, extent: Tuple[float, float, float, float] = DATA_EXTENT) -> xr.DataArray:
    da = _standardize_lat_lon_fig7(da)
    lon_min, lon_max, lat_min, lat_max = extent
    lat_slice = (
        slice(lat_max, lat_min)
        if float(da.latitude.values[0]) > float(da.latitude.values[-1])
        else slice(lat_min, lat_max)
    )
    out = da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    if out.sizes.get("longitude", 0) == 0 and float(da.longitude.max()) > 180 and lon_min < 180:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")
        out = da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    return out


def _select_level_fig7(da: xr.DataArray, level_hpa: float) -> xr.DataArray:
    da = _standardize_lat_lon_fig7(da)
    for lev in ("pressure_level", "level", "isobaricInhPa", "isobaricInPa", "plev"):
        if lev in da.dims:
            vals = np.asarray(da[lev].values)
            target = level_hpa * 100.0 if np.nanmax(vals) > 2000 else level_hpa
            return da.sel({lev: target}, method="nearest").squeeze(drop=True)
        if lev in da.coords:
            vals = np.asarray(da[lev].values)
            target = level_hpa * 100.0 if np.nanmax(vals) > 2000 else level_hpa
            if vals.ndim == 0 or vals.size == 1:
                val = float(np.ravel(vals)[0])
                if np.isfinite(val) and abs(val - target) <= max(1.0, 0.01 * abs(target)):
                    return da.drop_vars(lev, errors="ignore")
                raise ValueError(f"Field has scalar coordinate {lev}={val}, but requested level={target}.")
            if len(da[lev].dims) == 1:
                dim = da[lev].dims[0]
                if dim in da.dims:
                    idx = int(np.nanargmin(np.abs(vals - target)))
                    return da.isel({dim: idx}).drop_vars(lev, errors="ignore")
            raise KeyError(f"Coordinate {lev} exists but is not indexable. dims={da.dims}, coord_dims={da[lev].dims}")
    return da


def _to_bjt_daily_fig7(da: xr.DataArray, how: str) -> xr.DataArray:
    da = da.assign_coords(time=da.time + np.timedelta64(8, "h"))
    return getattr(da.resample(time="1D"), how)()


def _to_bjt_date_dim_fig7(da: xr.DataArray) -> xr.DataArray:
    if "time" in da.dims:
        return da.rename({"time": "bjt_date"})
    return da



def parse_mswep_filename_fig7(path: str) -> pd.Timestamp:
    stem = os.path.basename(path).replace(".nc", "")
    yyyyddd, hh = stem.split(".")
    year = int(yyyyddd[:4])
    doy = int(yyyyddd[4:])
    hour = int(hh)
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1, hours=hour)


def load_mswep_tp_fig7() -> xr.DataArray:
    if not os.path.isdir(MSWEP_2023_DIR):
        raise FileNotFoundError(MSWEP_2023_DIR)
    utc0 = TARGET_DATES.min() - pd.Timedelta(hours=8)
    utc1 = TARGET_DATES.max() + pd.Timedelta(days=1) - pd.Timedelta(hours=8)
    files = []
    for path in sorted(glob.glob(os.path.join(MSWEP_2023_DIR, "*.nc"))):
        ts = parse_mswep_filename_fig7(path)
        if utc0 <= ts <= utc1:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No MSWEP files in window: {utc0} to {utc1}")

    arrs = []
    matched_var = None
    for path in files:
        ds = _open_dataset_safe_fig7(path)
        pr, matched_var = _guess_var_fig7(ds, ["precipitation", "precip", "pr", "rain", "tp"])
        pr = _standardize_lat_lon_fig7(pr)
        if any(x in pr.dims or x in pr.coords for x in ["time", "valid_time"]):
            pr = _normalize_time_dim_fig7(pr)
        else:
            pr = pr.expand_dims(time=[parse_mswep_filename_fig7(path).to_datetime64()])
        arrs.append(pr)
    pr = xr.concat(arrs, dim="time").sortby("time")
    print("MSWEP matched var:", matched_var)
    pr_daily = _to_bjt_daily_fig7(pr, "sum")
    pr_daily.attrs["units"] = "mm day^-1"
    return _to_bjt_date_dim_fig7(_sel_region_fig7(pr_daily.sel(time=TARGET_DATES), DATA_EXTENT)).transpose("bjt_date", "latitude", "longitude")


# ====================================================
# ERA5 Fig. 7 loaders copied from Cell Y logic
# ====================================================

def _find_uvq_file_fig7() -> str:
    if os.path.exists(ERA5_UVQ_FILE_NEW):
        return ERA5_UVQ_FILE_NEW
    if os.path.exists(ERA5_UVQ_FILE_OLD):
        return ERA5_UVQ_FILE_OLD
    raise FileNotFoundError(f"Both uvq paths missing: {ERA5_UVQ_FILE_NEW}, {ERA5_UVQ_FILE_OLD}")


def load_era5_tp_fig7() -> xr.DataArray:
    if not os.path.exists(ERA5_TP_FILE):
        raise FileNotFoundError(ERA5_TP_FILE)
    ds = _open_dataset_safe_fig7(ERA5_TP_FILE)
    tp, name = _guess_var_fig7(ds, ["tp", "total_precipitation", "precip", "precipitation"])
    print("ERA5 TP matched var:", name)
    tp = _standardize_lat_lon_fig7(_normalize_time_dim_fig7(tp))
    units = str(tp.attrs.get("units", "")).lower()
    if units in ("m", "meter", "metre", "meters", "metres"):
        tp = tp * 1000.0
    tp_daily = _to_bjt_daily_fig7(tp, "sum")
    tp_daily.attrs["units"] = "mm day^-1"
    return _to_bjt_date_dim_fig7(_sel_region_fig7(tp_daily.sel(time=TARGET_DATES), DATA_EXTENT)).transpose("bjt_date", "latitude", "longitude")


def load_era5_z500_daily_fig7() -> xr.DataArray:
    if not os.path.exists(ERA5_Z500_2023_FILE):
        raise FileNotFoundError(ERA5_Z500_2023_FILE)
    ds = _open_dataset_safe_fig7(ERA5_Z500_2023_FILE)
    z, name = _guess_var_fig7(ds, ["z", "geopotential", "z500", "gh", "hgt"])
    print("ERA5 Z500 matched var:", name)
    z = _standardize_lat_lon_fig7(_normalize_time_dim_fig7(_select_level_fig7(z, 500.0)))
    units = str(z.attrs.get("units", "")).lower()
    long_name = str(z.attrs.get("long_name", "")).lower()
    vmax = float(np.nanpercentile(np.abs(z.values), 95)) if z.size else np.nan
    if "m**2" in units or "m2 s-2" in units or "m^2 s^-2" in units or units.strip() in ("m2/s2", "m^2/s^2") or "geopotential" in long_name or vmax > 20000:
        z = z / G0
        print("ERA5 Z500 converted geopotential to gpm")
    z.attrs["units"] = "gpm"
    z_daily = _to_bjt_daily_fig7(z, "mean")
    return _to_bjt_date_dim_fig7(_sel_region_fig7(z_daily.sel(time=TARGET_DATES), DATA_EXTENT)).transpose("bjt_date", "latitude", "longitude")


def load_era5_uv850_daily_fig7() -> Tuple[xr.DataArray, xr.DataArray]:
    uvq_file = _find_uvq_file_fig7()
    ds = _open_dataset_safe_fig7(uvq_file)
    u, uname = _guess_var_fig7(ds, ["u", "u_component_of_wind"])
    v, vname = _guess_var_fig7(ds, ["v", "v_component_of_wind"])
    print("ERA5 U/V matched vars:", uname, vname)
    u = _standardize_lat_lon_fig7(_normalize_time_dim_fig7(_select_level_fig7(u, 850.0)))
    v = _standardize_lat_lon_fig7(_normalize_time_dim_fig7(_select_level_fig7(v, 850.0)))
    u_daily = _to_bjt_daily_fig7(u, "mean")
    v_daily = _to_bjt_daily_fig7(v, "mean")
    u_out = _to_bjt_date_dim_fig7(_sel_region_fig7(u_daily.sel(time=TARGET_DATES), DATA_EXTENT)).transpose("bjt_date", "latitude", "longitude")
    v_out = _to_bjt_date_dim_fig7(_sel_region_fig7(v_daily.sel(time=TARGET_DATES), DATA_EXTENT)).transpose("bjt_date", "latitude", "longitude")
    return u_out, v_out


def load_observed_daily_fields_fig7() -> Dict[str, xr.DataArray]:
    """Observed row: MSWEP precipitation with ERA5 Z500 and optional ERA5 UV850."""
    u850, v850 = load_era5_uv850_daily_fig7()
    return {
        "tp": load_mswep_tp_fig7(),
        "z500": load_era5_z500_daily_fig7(),
        "u850": u850,
        "v850": v850,
        "obs_precip_source": "MSWEP",
        "obs_circulation_source": "ERA5",
    }


# ====================================================
# Total-stage grouping helpers
# ====================================================

def _norm_group_text_fig7(value) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _find_column_fig7(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm = {_norm_group_text_fig7(c): c for c in columns}
    for cand in candidates:
        key = _norm_group_text_fig7(cand)
        if key in norm:
            return norm[key]
    return None


def _parse_member_list_fig7(value) -> List[int]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        raw = list(value)
    else:
        raw = re.findall(r"\d+", str(value))
    return [int(v) for v in raw]


def _init_match_fig7(series: pd.Series, init_date: str) -> pd.Series:
    text_match = series.astype(str) == str(init_date)
    try:
        date_match = pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d") == pd.Timestamp(init_date).strftime("%Y-%m-%d")
        return text_match | date_match.fillna(False)
    except Exception:
        return text_match


def _total_stage_mask_fig7(series: pd.Series) -> pd.Series:
    return series.astype(str).map(_norm_group_text_fig7).isin({"total", "totalstage"})


def load_total_hotcold_groups_fig7(group_file: str = GROUP_FILE) -> pd.DataFrame:
    if not os.path.exists(group_file):
        raise FileNotFoundError(f"GROUP_FILE not found: {group_file}")
    df = pd.read_csv(group_file)
    print(f"[Group file] columns={list(df.columns)}")

    init_col = _find_column_fig7(df.columns, ["init_date", "init", "initial_date", "forecast_init", "date"])
    stage_col = _find_column_fig7(df.columns, ["stage", "grouping_stage", "tmax_stage"])
    if init_col is None or stage_col is None:
        raise KeyError(f"Could not identify init/stage columns; columns={list(df.columns)}")

    rows = []
    group_col = _find_column_fig7(df.columns, ["hotcold_group", "group", "hotcold", "tmax_group"])
    member_col = _find_column_fig7(df.columns, ["member", "members", "member_id", "number"])

    for init_date in INIT_DATES:
        df_init = df[_init_match_fig7(df[init_col], init_date)].copy()
        df_total = df_init[_total_stage_mask_fig7(df_init[stage_col])].copy()
        if df_total.empty:
            raise ValueError(
                f"No Total-stage rows found for init={init_date}. "
                f"columns={list(df.columns)}; init rows={len(df_init)}"
            )

        if group_col is not None and member_col is not None:
            for group in ABS_GROUPS:
                sub = df_total[df_total[group_col].astype(str).map(_norm_group_text_fig7) == _norm_group_text_fig7(group)]
                members = sorted(dict.fromkeys(_parse_member_list_fig7(sub[member_col])))
                for member in members:
                    rows.append({
                        "init_date": init_date,
                        "grouping_stage": "Total",
                        "hotcold_group": group,
                        "member": int(member),
                        "source_file": group_file,
                    })
        else:
            for group in ABS_GROUPS:
                col = _find_column_fig7(df_total.columns, [group, f"{group}_members", f"{group.lower()}_members", f"{group.lower()}"])
                if col is None:
                    raise KeyError(
                        f"Could not identify members for {group}; columns={list(df_total.columns)}; "
                        f"head={df_total.head().to_dict(orient='records')}"
                    )
                members = sorted(dict.fromkeys(_parse_member_list_fig7(df_total.iloc[0][col])))
                for member in members:
                    rows.append({
                        "init_date": init_date,
                        "grouping_stage": "Total",
                        "hotcold_group": group,
                        "member": int(member),
                        "source_file": group_file,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No Total-stage Hot17/Cold17 members parsed from {group_file}")
    for init_date in INIT_DATES:
        for group in ABS_GROUPS:
            members = get_hotcold_members_fig7(out, init_date, group)
            if len(members) != 17:
                raise ValueError(f"Expected 17 Total-stage members for init={init_date}, group={group}, got {len(members)}")
    return out


def get_hotcold_members_fig7(group_df: pd.DataFrame, init_date: str, group: str) -> List[int]:
    sub = group_df[
        (group_df["init_date"].astype(str) == str(init_date))
        & (group_df["grouping_stage"].astype(str) == "Total")
        & (group_df["hotcold_group"].astype(str) == str(group))
    ].copy()
    members = sub["member"].astype(int).tolist()
    if len(members) != 17:
        raise ValueError(f"Expected 17 members for init={init_date}, grouping_stage=Total, group={group}, got {len(members)}. members={members}")
    if len(set(members)) != 17:
        raise ValueError(f"Duplicated members for init={init_date}, group={group}: {members}")
    if min(members) < 0 or max(members) > 50:
        raise ValueError(f"Member IDs should be 0..50, got init={init_date}, group={group}, members={members}")
    return members


def build_group_audit_table_fig7(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for init_date in INIT_DATES:
        for group in ABS_GROUPS:
            members = get_hotcold_members_fig7(group_df, init_date, group)
            rows.append({
                "init_date": init_date,
                "grouping_stage": "Total",
                "group": group,
                "member_count": len(members),
                "members": ",".join(str(m) for m in members),
                "source_file": GROUP_FILE,
                "grouping_file": GROUP_FILE,
                "analysis_stage": "Stage-II diagnostic dates",
                "analysis_dates": "2023-06-18, 2023-06-19, 2023-06-20",
                "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
            })
    return pd.DataFrame(rows)


# ====================================================
# S2S readers copied from Cell ZZ logic
# ====================================================

def _format_candidate_paths_fig7(patterns: Sequence[str], init_date: str) -> List[str]:
    init = pd.Timestamp(init_date)
    fields = {
        "init_date": init.strftime("%Y-%m-%d"),
        "init_yyyymmdd": init.strftime("%Y%m%d"),
        "init_mmdd": init.strftime("%m%d"),
        "init_yyyy_mm": init.strftime("%Y-%m"),
    }
    return [pattern.format(**fields) for pattern in patterns]


def _resolve_patterns_fig7(patterns: Sequence[str], init_date: str) -> List[str]:
    paths = []
    for candidate in _format_candidate_paths_fig7(patterns, init_date):
        if glob.has_magic(candidate):
            paths.extend(glob.glob(candidate, recursive=True))
        elif os.path.exists(candidate):
            paths.append(candidate)
    return sorted(dict.fromkeys(paths), key=paths.index)


def _open_cfgrib_fig7(path: str, short_name: Optional[str] = None) -> xr.Dataset:
    backend_kwargs = {"indexpath": ""}
    if short_name is not None:
        backend_kwargs["filter_by_keys"] = {"shortName": short_name}
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)


def _member_dim_fig7(da: xr.DataArray) -> Optional[str]:
    for dim in ("member", "number", "realization", "ensemble", "perturbationNumber"):
        if dim in da.dims:
            return dim
    return None


def _normalize_member_dim_fig7(da: xr.DataArray, stream: str) -> xr.DataArray:
    mdim = _member_dim_fig7(da)
    if mdim is None:
        da = da.expand_dims(member=[0 if stream == "cf" else 1])
    elif mdim != "member":
        da = da.rename({mdim: "member"})
    offset = 0 if stream == "cf" else 1
    return da.assign_coords(member=np.arange(offset, offset + da.sizes["member"]))


def _init_coord_name_fig7(da: xr.DataArray) -> Optional[str]:
    for name in ("time", "forecast_reference_time", "date"):
        if name in da.coords or name in da.dims:
            return name
    return None


def get_s2s_init_slice_fig7(da: xr.DataArray, init_date: str) -> xr.DataArray:
    coord = _init_coord_name_fig7(da)
    if coord is None:
        return da
    init = pd.Timestamp(init_date)
    times = pd.to_datetime(da[coord].values.ravel())
    matches = [t for t in times if pd.Timestamp(t).date() == init.date()]
    if matches:
        da = da.sel({coord: matches})
    else:
        print(f"  [Info] no explicit init coordinate match for {init_date}; keeping all {coord} values")
    if coord in da.dims and da.sizes.get(coord) == 1:
        da = da.squeeze(coord, drop=True)
    return da


def _to_gpm_if_needed_fig7(da: xr.DataArray, var_label: str) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    vmax = float(np.nanpercentile(np.abs(da.values), 95)) if da.size else np.nan
    if "m**2" in units or "m2 s-2" in units or vmax > 20000:
        da = da / G0
        da.attrs["converted_to_gpm"] = "yes: divided by 9.80665"
        print(f"  [{var_label}] converted geopotential to gpm")
    else:
        da.attrs["converted_to_gpm"] = "not needed or already height"
    da.attrs["units"] = "gpm"
    return da


def _tp_to_mm_fig7(da: xr.DataArray) -> xr.DataArray:
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


def _build_bjt_valid_coord_fig7(da: xr.DataArray, init_date: str) -> xr.DataArray:
    if "step" not in da.coords:
        if "valid_time" in da.coords:
            vt = pd.to_datetime(da.valid_time.values) + BJT_OFFSET
            time_dim = "valid_time" if "valid_time" in da.dims else da.valid_time.dims[-1]
            return da.assign_coords(bjt_valid_time=(time_dim, vt)).swap_dims({time_dim: "bjt_valid_time"})
        raise ValueError("S2S field has no step or valid_time coordinate")
    init = pd.Timestamp(init_date)
    bjt = pd.to_datetime([init + pd.Timedelta(s) + BJT_OFFSET for s in da.step.values])
    return da.assign_coords(bjt_valid_time=("step", bjt)).swap_dims({"step": "bjt_valid_time"})


def process_s2s_bjt_daily_mean_fig7(da: xr.DataArray, init_date: str) -> xr.DataArray:
    da = _build_bjt_valid_coord_fig7(da, init_date)
    return da.resample(bjt_valid_time="1D").mean().rename({"bjt_valid_time": "bjt_date"})


def _interval_day_fractions_fig7(start: pd.Timestamp, end: pd.Timestamp):
    if end <= start:
        return []
    total = (end - start).total_seconds()
    pieces = []
    cursor = start
    while cursor < end:
        next_midnight = cursor.normalize() + pd.Timedelta(days=1)
        stop = min(end, next_midnight)
        pieces.append((cursor.normalize(), (stop - cursor).total_seconds() / total))
        cursor = stop
    return pieces


def _apportion_accum_to_calendar_days_fig7(da: xr.DataArray, init_date: str, force_non_negative: bool = False) -> xr.DataArray:
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
        start = bjt_valid[i - 1]
        end = bjt_valid[i]
        inc_i = inc.isel(step=i - 1).drop_vars("step", errors="ignore")
        for day, frac in _interval_day_fractions_fig7(start, end):
            apportioned.append((inc_i * frac).expand_dims(bjt_date=[day]))
    if not apportioned:
        raise ValueError("No accumulated increments available for BJT day apportionment")
    return xr.concat(apportioned, dim="bjt_date").groupby("bjt_date").sum(skipna=True)


def process_s2s_bjt_daily_accum_fig7(da: xr.DataArray, init_date: str, output: str, force_non_negative: bool = False) -> xr.DataArray:
    daily_amount = _apportion_accum_to_calendar_days_fig7(da, init_date, force_non_negative=force_non_negative)
    daily_amount.attrs.update(da.attrs)
    if output == "tp":
        return _tp_to_mm_fig7(daily_amount)
    raise ValueError(output)


def _concat_cf_pf_fig7(cf: Optional[xr.DataArray], pf: Optional[xr.DataArray]) -> xr.DataArray:
    parts = [_normalize_member_dim_fig7(da, stream) for stream, da in (("cf", cf), ("pf", pf)) if da is not None]
    if not parts:
        raise FileNotFoundError("Neither CF nor PF field was loaded")
    out = xr.concat(parts, dim="member", coords="minimal", compat="override")
    return out.assign_coords(member=np.arange(out.sizes["member"]))


def _load_one_stream_fig7(stream: str, product: str, init_date: str, short_name: Optional[str], candidates: Sequence[str]) -> Optional[xr.DataArray]:
    patterns = S2S_FILE_CATALOG["realtime"][stream].get(product, [])
    paths = _resolve_patterns_fig7(patterns, init_date)
    if not paths:
        tried = _format_candidate_paths_fig7(patterns, init_date)
        print(f"  [Missing] realtime/{stream}/{product}: no files matched. Tried:")
        for candidate in tried:
            print("   ", candidate, os.path.exists(candidate))
        return None
    for path in paths:
        if product == "pl_500":
            # The realtime pressure-level file stores geopotential height as
            # shortName="gh".  Try both specific Z500 names before the very
            # large unfiltered dataset.
            short_name_attempts = []
            for candidate_short_name in (short_name, "gh", "z", None):
                if candidate_short_name not in short_name_attempts:
                    short_name_attempts.append(candidate_short_name)
        else:
            short_name_attempts = [short_name] if short_name is None else [short_name, None]
        for sn in short_name_attempts:
            try:
                ds = _open_cfgrib_fig7(path, short_name=sn)
                print(f"  [Open] realtime/{stream}/{product}: {os.path.basename(path)} shortName={sn} data_vars={list(ds.data_vars)}")
                da, var = _guess_var_fig7(ds, candidates)
                da = get_s2s_init_slice_fig7(da, init_date)
                print(f"  [Loaded] realtime/{stream}/{product}: {os.path.basename(path)} -> {var}")
                return da
            except Exception as exc:
                print(f"  [Skip] {path} shortName={sn}: {exc}")
    return None


def load_s2s_ensemble_with_cf_fig7(product: str, init_date: str, short_name: Optional[str], candidates: Sequence[str]) -> xr.DataArray:
    cf = _load_one_stream_fig7("cf", product, init_date, short_name, candidates)
    pf = _load_one_stream_fig7("pf", product, init_date, short_name, candidates)
    da = _concat_cf_pf_fig7(cf, pf)
    da = _sel_region_fig7(da, DATA_EXTENT)
    member_count = int(da.sizes.get("member", 0))
    print(f"  [Members] realtime/{product}/{init_date}: member_count={member_count}")
    if member_count != 51:
        raise ValueError(f"Expected 51 S2S members after CF+PF merge, got {member_count}")
    return da


def build_daily_realtime_member_fields_fig7(init_date: str) -> Dict[str, xr.DataArray]:
    print(f"\n=== Loading S2S realtime fields for init={init_date} ===")
    z500_raw = load_s2s_ensemble_with_cf_fig7("pl_500", init_date, "z", ["gh", "z", "geopotential"])
    u_raw = load_s2s_ensemble_with_cf_fig7("pl_uv", init_date, "u", ["u", "u_component_of_wind"])
    v_raw = load_s2s_ensemble_with_cf_fig7("pl_uv", init_date, "v", ["v", "v_component_of_wind"])
    tp_raw = load_s2s_ensemble_with_cf_fig7("sfc_tp", init_date, "tp", ["tp", "total_precipitation"])

    z500 = _to_gpm_if_needed_fig7(_select_level_fig7(z500_raw, 500), "Z500 realtime")
    u850 = _select_level_fig7(u_raw, 850)
    v850 = _select_level_fig7(v_raw, 850)

    return {
        "tp": process_s2s_bjt_daily_accum_fig7(tp_raw, init_date, output="tp", force_non_negative=True).sel(bjt_date=TARGET_DATES),
        "z500": process_s2s_bjt_daily_mean_fig7(z500, init_date).sel(bjt_date=TARGET_DATES),
        "u850": process_s2s_bjt_daily_mean_fig7(u850, init_date).sel(bjt_date=TARGET_DATES),
        "v850": process_s2s_bjt_daily_mean_fig7(v850, init_date).sel(bjt_date=TARGET_DATES),
    }


def build_daily_realtime_significance_fields_fig7(init_date: str) -> Dict[str, xr.DataArray]:
    """Load only the unchanged member-level TP and Z500 fields needed for significance."""
    print(f"\n=== Loading S2S realtime significance fields for init={init_date} ===")
    z500_raw = load_s2s_ensemble_with_cf_fig7(
        "pl_500", init_date, "z", ["gh", "z", "geopotential"]
    )
    tp_raw = load_s2s_ensemble_with_cf_fig7(
        "sfc_tp", init_date, "tp", ["tp", "total_precipitation"]
    )
    z500 = _to_gpm_if_needed_fig7(
        _select_level_fig7(z500_raw, 500), "Z500 realtime"
    )
    return {
        "tp": process_s2s_bjt_daily_accum_fig7(
            tp_raw, init_date, output="tp", force_non_negative=True
        ).sel(bjt_date=TARGET_DATES),
        "z500": process_s2s_bjt_daily_mean_fig7(z500, init_date).sel(
            bjt_date=TARGET_DATES
        ),
    }


def group_mean_fig7(da: xr.DataArray, members: List[int]) -> xr.DataArray:
    return da.sel(member=members).mean("member", skipna=True)


def build_group_composites_fig7(init_date: str, group_df: pd.DataFrame, fields: Dict[str, xr.DataArray]) -> Dict[str, Dict[str, xr.DataArray]]:
    out: Dict[str, Dict[str, xr.DataArray]] = {}
    for group in ABS_GROUPS:
        members = get_hotcold_members_fig7(group_df, init_date, group)
        out[group] = {
            "members": members,
            "tp": group_mean_fig7(fields["tp"], members),
            "z500": group_mean_fig7(fields["z500"], members),
            "u850": group_mean_fig7(fields["u850"], members),
            "v850": group_mean_fig7(fields["v850"], members),
        }

    out["Hot17_minus_Cold17"] = {
        "members": None,
        "tp": out["Hot17"]["tp"] - out["Cold17"]["tp"],
        "z500": out["Hot17"]["z500"] - out["Cold17"]["z500"],
        "u850": out["Hot17"]["u850"] - out["Cold17"]["u850"],
        "v850": out["Hot17"]["v850"] - out["Cold17"]["v850"],
    }
    return out


# ====================================================
# Plotting helpers
# ====================================================

def _setup_map_fig7(ax):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", lw=0.8)
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    ax.add_feature(cfeature.LAKES.with_scale("110m"), linewidth=0.3, edgecolor="gray", facecolor="none")
    ax.add_feature(cfeature.RIVERS.with_scale("110m"), linewidth=0.25, edgecolor="steelblue")
    ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.35, alpha=0.45)


def _plot_box_fig7(ax, box=NCHN_BOX):
    x0, x1, y0, y1 = box
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="k", lw=1.2, transform=ccrs.PlateCarree())


def _add_panel_label_fig7(ax, label: str):
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.4),
        zorder=30,
    )


def _add_row_label_fig7(ax, label: str):
    ax.text(-0.08, 0.50, label, transform=ax.transAxes, ha="center", va="center", rotation=90, fontsize=11, fontweight="bold")


def _apply_label_path_effects_fig7(texts):
    for txt in texts:
        txt.set_color("0.10")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def _contour_z500_abs_fig7(ax, z500: xr.DataArray):
    cs = ax.contour(
        z500.longitude,
        z500.latitude,
        z500,
        levels=np.arange(5400, 6021, 40),
        colors="0.20",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
    )
    if len(cs.levels):
        _apply_label_path_effects_fig7(ax.clabel(cs, levels=cs.levels[::2], fmt="%.0f", fontsize=6.5, inline=True))
    zmin = _safe_float_fig7(np.nanmin(z500.values))
    zmax = _safe_float_fig7(np.nanmax(z500.values))
    if np.isfinite(zmin) and np.isfinite(zmax) and zmin <= 5880 <= zmax:
        ax.contour(z500.longitude, z500.latitude, z500, levels=[5880], colors="purple", linewidths=1.8, transform=ccrs.PlateCarree())


def _contour_z500_diff_fig7(ax, z500: xr.DataArray):
    zlevels = np.array([-80, -60, -40, -20, 20, 40, 60, 80])
    zmin = _safe_float_fig7(np.nanmin(z500.values))
    zmax = _safe_float_fig7(np.nanmax(z500.values))
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return
    zlevels = zlevels[(zlevels >= zmin) & (zlevels <= zmax)]
    for levels, color, style in ((zlevels[zlevels < 0], "royalblue", "--"), (zlevels[zlevels > 0], "firebrick", "-")):
        if levels.size:
            cs = ax.contour(z500.longitude, z500.latitude, z500, levels=levels, colors=color, linestyles=style, linewidths=1.0, transform=ccrs.PlateCarree())
            _apply_label_path_effects_fig7(ax.clabel(cs, fmt="%.0f", fontsize=6.5, inline=True))


def _prepare_uv_quiver_arrays_fig7(lon, lat, u2d, v2d, stride: int, min_speed: float):
    lon_sub = lon.values[::stride]
    lat_sub = lat.values[::stride]
    u_sub = u2d.values[::stride, ::stride]
    v_sub = v2d.values[::stride, ::stride]
    x2d, y2d = np.meshgrid(lon_sub, lat_sub)
    speed = np.sqrt(u_sub ** 2 + v_sub ** 2)
    mask = speed >= min_speed
    return x2d[mask], y2d[mask], u_sub[mask], v_sub[mask]


def _plot_uv850_quiver_fig7(ax, u850: xr.DataArray, v850: xr.DataArray, *, is_difference: bool, stride: int):
    min_speed = WIND_DIFF_MIN_MS if is_difference else WIND_MIN_MS
    x_plot, y_plot, u_plot, v_plot = _prepare_uv_quiver_arrays_fig7(u850.longitude, u850.latitude, u850, v850, stride, min_speed)
    if u_plot.size == 0:
        return None
    common_kwargs = dict(
        alpha=1.0,
        scale=42 if is_difference else 85,
        pivot="mid",
        headwidth=6.2,
        headlength=7.2,
        headaxislength=6.2,
        transform=ccrs.PlateCarree(),
    )
    ax.quiver(
        x_plot,
        y_plot,
        u_plot,
        v_plot,
        color="white",
        width=0.0045 if is_difference else 0.0048,
        zorder=20,
        **common_kwargs,
    )
    return ax.quiver(
        x_plot,
        y_plot,
        u_plot,
        v_plot,
        color=VECTOR_COLOR,
        width=0.0032 if is_difference else 0.0035,
        zorder=21,
        **common_kwargs,
    )


def _save_fig_fig7(fig, stem: str):
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=250, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _tp_abs_levels_fig7(arrays: Iterable[xr.DataArray]) -> np.ndarray:
    pool = np.concatenate([np.asarray(da.values).ravel() for da in arrays])
    if np.all(np.isnan(pool)):
        raise ValueError("All absolute TP values are NaN; cannot plot")
    pmax = max(float(np.nanpercentile(pool, 95)), 1.0)
    pmax = min(pmax, 25.0)
    return np.linspace(0, pmax, 13)


def _tp_diff_levels_fig7(arrays: Iterable[xr.DataArray]) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    pool = np.concatenate([np.asarray(da.values).ravel() for da in arrays])
    if np.all(np.isnan(pool)):
        raise ValueError("All Hot17-Cold17 TP difference values are NaN; cannot plot")
    vmax = max(float(np.nanpercentile(np.abs(pool), 98)), 0.5)
    levels = np.linspace(-vmax, vmax, 13)
    return levels, mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _plot_abs_panel_fig7(ax, comp: Dict[str, xr.DataArray], date: pd.Timestamp, levels: np.ndarray, label: str, stride: int, show_vectors: bool = True):
    _setup_map_fig7(ax)
    _plot_box_fig7(ax)
    tp = comp["tp"].sel(bjt_date=date)
    z500 = comp["z500"].sel(bjt_date=date)
    u850 = comp["u850"].sel(bjt_date=date)
    v850 = comp["v850"].sel(bjt_date=date)
    cf = ax.contourf(tp.longitude, tp.latitude, tp, levels=levels, cmap="Blues", extend="max", transform=ccrs.PlateCarree())
    _contour_z500_abs_fig7(ax, z500)
    q = _plot_uv850_quiver_fig7(ax, u850, v850, is_difference=False, stride=stride) if show_vectors else None
    _add_panel_label_fig7(ax, label)
    return cf, q


def _plot_diff_panel_fig7(ax, comp: Dict[str, xr.DataArray], date: pd.Timestamp, levels: np.ndarray, norm: mcolors.TwoSlopeNorm, label: str, stride: int, show_vectors: bool = True):
    _setup_map_fig7(ax)
    _plot_box_fig7(ax)
    tp = comp["tp"].sel(bjt_date=date)
    z500 = comp["z500"].sel(bjt_date=date)
    u850 = comp["u850"].sel(bjt_date=date)
    v850 = comp["v850"].sel(bjt_date=date)
    cf = ax.contourf(tp.longitude, tp.latitude, tp, levels=levels, cmap="RdBu_r", norm=norm, extend="both", transform=ccrs.PlateCarree())
    _contour_z500_diff_fig7(ax, z500)
    q = _plot_uv850_quiver_fig7(ax, u850, v850, is_difference=True, stride=stride) if show_vectors else None
    _add_panel_label_fig7(ax, label)
    return cf, q


# ====================================================
# Paper figures
# ====================================================

def plot_figure7_main_fig7(obs_fields: Dict[str, xr.DataArray], comp_0612: Dict[str, Dict[str, xr.DataArray]], abs_levels: np.ndarray, diff_levels: np.ndarray, diff_norm: mcolors.TwoSlopeNorm, show_vectors: bool, suffix: str):
    fig, axs = plt.subplots(4, 3, figsize=(13.8, 12.8), subplot_kw={"projection": ccrs.PlateCarree()})
    panel_labels = list("abcdefghijkl")
    cf_abs = None
    cf_diff = None
    q_abs_key = None
    q_diff_key = None

    rows = ["MSWEP", "Hot17", "Cold17", DIFF_GROUP]
    for row, group in enumerate(rows):
        for col, date in enumerate(TARGET_DATES):
            ax = axs[row, col]
            label = f"({panel_labels[row * 3 + col]})"
            if group == "MSWEP":
                cf, q = _plot_abs_panel_fig7(ax, obs_fields, date, abs_levels, label, ERA5_VECTOR_STRIDE, show_vectors=show_vectors)
                if cf_abs is None:
                    cf_abs = cf
                if q_abs_key is None and q is not None:
                    q_abs_key = q
            elif group == DIFF_GROUP:
                cf, q = _plot_diff_panel_fig7(ax, comp_0612[group], date, diff_levels, diff_norm, label, S2S_VECTOR_STRIDE, show_vectors=show_vectors)
                if cf_diff is None:
                    cf_diff = cf
                if q_diff_key is None and q is not None:
                    q_diff_key = q
            else:
                cf, q = _plot_abs_panel_fig7(ax, comp_0612[group], date, abs_levels, label, S2S_VECTOR_STRIDE, show_vectors=show_vectors)
                if cf_abs is None:
                    cf_abs = cf
                if q_abs_key is None and q is not None:
                    q_abs_key = q
            if row == 0:
                ax.set_title(f"{date.strftime('%Y-%m-%d')} BJT", fontsize=10)
            if col == 0:
                _add_row_label_fig7(ax, GROUP_LABELS[group])

    if show_vectors and q_abs_key is not None:
        axs[0, 0].quiverkey(q_abs_key, X=0.08, Y=-0.08, U=5.0, label="5 m s$^{-1}$", labelpos="E", coordinates="axes")
    if show_vectors and q_diff_key is not None:
        axs[3, 0].quiverkey(q_diff_key, X=0.08, Y=-0.08, U=2.0, label="2 m s$^{-1}$", labelpos="E", coordinates="axes")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.15, wspace=0.06, hspace=0.10)
    if cf_abs is not None:
        cax_abs = fig.add_axes([0.18, 0.065, 0.34, 0.020])
        fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal", label="TP (mm day$^{-1}$)")
    if cf_diff is not None:
        cax_diff = fig.add_axes([0.58, 0.065, 0.30, 0.020])
        fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal", label="Hot17 − Cold17 TP (mm day$^{-1}$)")
    fig.suptitle("Stage-II daily synoptic evolution for the 12 June initialization (Total grouping)", fontsize=15, y=0.965)
    _save_fig_fig7(fig, FIG7_MAIN_STEMS[suffix])


def plot_figureS_appendix_fig7(comp_0608: Dict[str, Dict[str, xr.DataArray]], abs_levels: np.ndarray, diff_levels: np.ndarray, diff_norm: mcolors.TwoSlopeNorm, show_vectors: bool, suffix: str):
    fig, axs = plt.subplots(3, 3, figsize=(13.8, 9.4), subplot_kw={"projection": ccrs.PlateCarree()})
    panel_labels = list("abcdefghi")
    cf_abs = None
    cf_diff = None
    q_abs_key = None
    q_diff_key = None

    rows = ["Hot17", "Cold17", DIFF_GROUP]
    for row, group in enumerate(rows):
        for col, date in enumerate(TARGET_DATES):
            ax = axs[row, col]
            label = f"({panel_labels[row * 3 + col]})"
            if group == DIFF_GROUP:
                cf, q = _plot_diff_panel_fig7(ax, comp_0608[group], date, diff_levels, diff_norm, label, S2S_VECTOR_STRIDE, show_vectors=show_vectors)
                if cf_diff is None:
                    cf_diff = cf
                if q_diff_key is None and q is not None:
                    q_diff_key = q
            else:
                cf, q = _plot_abs_panel_fig7(ax, comp_0608[group], date, abs_levels, label, S2S_VECTOR_STRIDE, show_vectors=show_vectors)
                if cf_abs is None:
                    cf_abs = cf
                if q_abs_key is None and q is not None:
                    q_abs_key = q
            if row == 0:
                ax.set_title(f"{date.strftime('%Y-%m-%d')} BJT", fontsize=10)
            if col == 0:
                _add_row_label_fig7(ax, GROUP_LABELS[group])

    if show_vectors and q_abs_key is not None:
        axs[0, 0].quiverkey(q_abs_key, X=0.08, Y=-0.08, U=5.0, label="5 m s$^{-1}$", labelpos="E", coordinates="axes")
    if show_vectors and q_diff_key is not None:
        axs[2, 0].quiverkey(q_diff_key, X=0.08, Y=-0.08, U=2.0, label="2 m s$^{-1}$", labelpos="E", coordinates="axes")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.15, wspace=0.06, hspace=0.12)
    if cf_abs is not None:
        cax_abs = fig.add_axes([0.18, 0.075, 0.34, 0.020])
        fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal", label="TP (mm day$^{-1}$)")
    if cf_diff is not None:
        cax_diff = fig.add_axes([0.58, 0.075, 0.30, 0.020])
        fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal", label="Hot17 − Cold17 TP (mm day$^{-1}$)")
    fig.suptitle("Stage-II daily synoptic evolution for the 8 June initialization (Total grouping)", fontsize=15, y=0.965)
    _save_fig_fig7(fig, FIG7_APPENDIX_STEMS[suffix])


# ====================================================
# QC
# ====================================================

def _qc_stats_fig7(da: xr.DataArray) -> Tuple[float, float, float]:
    return (_safe_float_fig7(da.min(skipna=True)), _safe_float_fig7(da.max(skipna=True)), _safe_float_fig7(da.mean(skipna=True)))


def _qc_rows_for_fields_fig7(figure_type: str, init_date: str, group: str, fields: Dict[str, xr.DataArray], members: Optional[List[int]]) -> List[dict]:
    rows = []
    for date in TARGET_DATES:
        tp = fields["tp"].sel(bjt_date=date)
        z500 = fields["z500"].sel(bjt_date=date)
        u850 = fields["u850"].sel(bjt_date=date)
        v850 = fields["v850"].sel(bjt_date=date)
        uvspd = np.sqrt(u850 ** 2 + v850 ** 2)
        tp_min, tp_max, tp_mean = _qc_stats_fig7(tp)
        zmin, zmax, zmean = _qc_stats_fig7(z500)
        uvmin, uvmax, uvmean = _qc_stats_fig7(uvspd)
        rows.append({
            "figure_type": figure_type,
            "init_date": init_date,
            "group": group,
            "date": date.strftime("%Y-%m-%d"),
            "tp_min": tp_min,
            "tp_max": tp_max,
            "tp_mean": tp_mean,
            "z500_min": zmin,
            "z500_max": zmax,
            "z500_mean": zmean,
            "uv850_speed_min": uvmin,
            "uv850_speed_max": uvmax,
            "uv850_speed_mean": uvmean,
            "member_count": len(members) if members is not None else np.nan,
            "members": ",".join(str(m) for m in members) if members is not None else "",
            "grouping_file": GROUP_FILE,
            "grouping_stage": "Total",
            "analysis_stage": "Stage-II diagnostic dates",
            "analysis_dates": "2023-06-18, 2023-06-19, 2023-06-20",
            "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
        })
    return rows


def build_qc_rows_fig7(obs_fields: Dict[str, xr.DataArray], composites: Dict[str, Dict[str, Dict[str, xr.DataArray]]]) -> List[dict]:
    rows = []
    obs_rows = _qc_rows_for_fields_fig7("main", "observed", "MSWEP", obs_fields, None)
    for row in obs_rows:
        row["obs_precip_source"] = "MSWEP"
        row["obs_circulation_source"] = "ERA5"
    rows.extend(obs_rows)
    for group in ["Hot17", "Cold17", DIFF_GROUP]:
        rows.extend(_qc_rows_for_fields_fig7("main", MAIN_INIT_DATE, group, composites[MAIN_INIT_DATE][group], composites[MAIN_INIT_DATE][group].get("members")))
    for group in ["Hot17", "Cold17", DIFF_GROUP]:
        rows.extend(_qc_rows_for_fields_fig7("appendix", APPENDIX_INIT_DATE, group, composites[APPENDIX_INIT_DATE][group], composites[APPENDIX_INIT_DATE][group].get("members")))
    return rows


# ====================================================
# Runner
# ====================================================

def run_fig7_stageII_daily_synoptic_paper(make_plots: bool = True):
    group_df = load_total_hotcold_groups_fig7(GROUP_FILE)
    audit = build_group_audit_table_fig7(group_df)
    audit_path = os.path.join(OUT_DIR, "figure7_stageII_daily_synoptic_group_member_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(f"[Audit] saved: {audit_path}")

    obs_fields = load_observed_daily_fields_fig7()
    composites: Dict[str, Dict[str, Dict[str, xr.DataArray]]] = {}
    for init_date in INIT_DATES:
        fields = build_daily_realtime_member_fields_fig7(init_date)
        composites[init_date] = build_group_composites_fig7(init_date, group_df, fields)

    abs_levels = _tp_abs_levels_fig7([
        obs_fields["tp"],
        composites[MAIN_INIT_DATE]["Hot17"]["tp"],
        composites[MAIN_INIT_DATE]["Cold17"]["tp"],
        composites[APPENDIX_INIT_DATE]["Hot17"]["tp"],
        composites[APPENDIX_INIT_DATE]["Cold17"]["tp"],
    ])
    diff_levels, diff_norm = _tp_diff_levels_fig7([
        composites[MAIN_INIT_DATE][DIFF_GROUP]["tp"],
        composites[APPENDIX_INIT_DATE][DIFF_GROUP]["tp"],
    ])

    qc = pd.DataFrame(build_qc_rows_fig7(obs_fields, composites))
    qc_path = os.path.join(OUT_DIR, "figure7_stageII_daily_synoptic_qc.csv")
    qc.to_csv(qc_path, index=False)
    print(f"[QC] saved: {qc_path}")

    if make_plots:
        plot_figure7_main_fig7(obs_fields, composites[MAIN_INIT_DATE], abs_levels, diff_levels, diff_norm, show_vectors=False, suffix="novectors")
        plot_figure7_main_fig7(obs_fields, composites[MAIN_INIT_DATE], abs_levels, diff_levels, diff_norm, show_vectors=True, suffix="vectors")
        plot_figureS_appendix_fig7(composites[APPENDIX_INIT_DATE], abs_levels, diff_levels, diff_norm, show_vectors=False, suffix="novectors")
        plot_figureS_appendix_fig7(composites[APPENDIX_INIT_DATE], abs_levels, diff_levels, diff_norm, show_vectors=True, suffix="vectors")


# ====================================================
# Member-level spatial significance export (Fig.7)
# ====================================================

def _fig7_significance_domain(da: xr.DataArray) -> xr.DataArray:
    """Select the fixed 90–135 E, 25–55 N FDR family without regridding."""
    lon = da["longitude"]
    lat = da["latitude"]
    lon_mask = (lon >= SIGNIFICANCE_DOMAIN[0]) & (lon <= SIGNIFICANCE_DOMAIN[1])
    lat_mask = (lat >= SIGNIFICANCE_DOMAIN[2]) & (lat <= SIGNIFICANCE_DOMAIN[3])
    return da.sel(longitude=lon[lon_mask], latitude=lat[lat_mask])


def _fig7_validate_groups(hot_members: Sequence[int], cold_members: Sequence[int]) -> None:
    if len(hot_members) != 17 or len(cold_members) != 17:
        raise ValueError(f"Fig.7 significance requires 17 vs 17 members; hot={len(hot_members)}, cold={len(cold_members)}")
    if list(hot_members) == list(cold_members):
        raise ValueError("Fig.7 Hot17 and Cold17 member lists are identical")
    overlap = sorted(set(hot_members).intersection(cold_members))
    if overlap:
        raise ValueError(f"Fig.7 Hot17/Cold17 member overlap: {overlap}")


def _fig7_fdr_bh(
    p_value: np.ndarray, latitude: np.ndarray, longitude: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    domain = (
        (latitude[:, None] >= SIGNIFICANCE_DOMAIN[2])
        & (latitude[:, None] <= SIGNIFICANCE_DOMAIN[3])
        & (longitude[None, :] >= SIGNIFICANCE_DOMAIN[0])
        & (longitude[None, :] <= SIGNIFICANCE_DOMAIN[1])
    )
    finite = np.isfinite(p_value) & domain
    adjusted = np.full(p_value.shape, np.nan, dtype=float)
    rejected = np.zeros(p_value.shape, dtype=bool)
    if finite.any():
        reject_flat, adjusted_flat, _, _ = multipletests(
            p_value[finite], alpha=0.05, method="fdr_bh"
        )
        adjusted[finite] = adjusted_flat
        rejected[finite] = reject_flat
    return adjusted, rejected, int(finite.sum())


def _fig7_welch_panel(
    member_field: xr.DataArray,
    hot_members: Sequence[int],
    cold_members: Sequence[int],
    original_difference: xr.DataArray,
    units: str,
    variable: str,
    date: pd.Timestamp,
) -> Tuple[xr.Dataset, dict]:
    """Run one date × variable Welch test with member as the sole sample axis."""
    field = member_field.sel(bjt_date=date).transpose(
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
    t_statistic[insufficient] = np.nan
    p_value[insufficient] = np.nan
    p_fdr_bh, significant_fdr, n_valid_tests = _fig7_fdr_bh(
        p_value, field.latitude.values, field.longitude.values
    )
    correction_domain = (
        (field.latitude.values[:, None] >= SIGNIFICANCE_DOMAIN[2])
        & (field.latitude.values[:, None] <= SIGNIFICANCE_DOMAIN[3])
        & (field.longitude.values[None, :] >= SIGNIFICANCE_DOMAIN[0])
        & (field.longitude.values[None, :] <= SIGNIFICANCE_DOMAIN[1])
    )
    if not correction_domain.any():
        raise ValueError("Fig.7 significance native grid has no points in the fixed 90–135E, 25–55N domain")
    local_significant = correction_domain & np.isfinite(p_value) & (p_value < 0.05)

    if np.any((p_fdr_bh[np.isfinite(p_fdr_bh)] < 0) | (p_fdr_bh[np.isfinite(p_fdr_bh)] > 1)):
        raise ValueError(f"Fig.7 {variable} {date:%Y-%m-%d}: BH-adjusted p-values outside [0, 1]")
    finite_p = p_value[np.isfinite(p_fdr_bh)]
    if finite_p.size and np.all(finite_p == finite_p[0]):
        warnings.warn(f"Fig.7 {variable} {date:%Y-%m-%d}: finite p-value field is constant")
    fdr_family_mask = np.isfinite(p_fdr_bh)
    if fdr_family_mask.any() and significant_fdr[fdr_family_mask].all():
        warnings.warn(f"Fig.7 {variable} {date:%Y-%m-%d}: entire FDR mask is significant")

    hot_mean_da = hot.mean("member", skipna=True).transpose(
        "latitude", "longitude"
    )
    cold_mean_da = cold.mean("member", skipna=True).transpose(
        "latitude", "longitude"
    )
    hot_mean = hot_mean_da.values
    cold_mean = cold_mean_da.values
    difference = hot_mean - cold_mean
    expected = original_difference.sel(bjt_date=date).transpose(
        "latitude", "longitude"
    )
    mismatch = float(np.nanmax(np.abs(difference - expected.values)))
    if not np.allclose(difference, expected.values, rtol=1e-7, atol=1e-6, equal_nan=True):
        raise ValueError(
            f"Fig.7 {variable} {date:%Y-%m-%d}: member-mean difference does not match original composite; "
            f"max_abs_mismatch={mismatch:.6g}"
        )

    coords = {"s2s_lat": field.latitude.values, "s2s_lon": field.longitude.values}
    dims = ("s2s_lat", "s2s_lon")
    data_vars = {
        "hot_mean": (dims, hot_mean),
        "cold_mean": (dims, cold_mean),
        "hot_minus_cold": (dims, difference),
        "t_statistic": (dims, t_statistic),
        "p_value": (dims, p_value),
        "p_fdr_bh": (dims, p_fdr_bh),
        "significant_local_p005": (dims, local_significant.astype(np.int8)),
        "significant_fdr_005": (dims, significant_fdr.astype(np.int8)),
        "fdr_domain_mask": (dims, correction_domain.astype(np.int8)),
        "n_hot": (dims, n_hot),
        "n_cold": (dims, n_cold),
    }
    if hasattr(result, "df"):
        degrees_of_freedom = np.array(result.df, dtype=float, copy=True)
        degrees_of_freedom[insufficient] = np.nan
        data_vars["degrees_of_freedom"] = (dims, degrees_of_freedom)
    ds = xr.Dataset(data_vars, coords=coords).expand_dims(date=[np.datetime64(date, "ns")])
    ds["p_fdr_bh"].attrs["description"] = "Benjamini-Hochberg adjusted p-value"
    ds["significant_local_p005"].attrs["description"] = "1 where local two-sided Welch p < 0.05 within the fixed analysis domain"
    ds["significant_fdr_005"].attrs["description"] = "1 where Benjamini-Hochberg adjusted p < 0.05"
    ds["fdr_domain_mask"].attrs["description"] = (
        "Grid points included in the fixed 90–135E, 25–55N "
        "Benjamini-Hochberg FDR family"
    )
    ds.attrs["alternative_explicit"] = str(alternative_explicit)
    local_count = int(np.count_nonzero(local_significant))
    fdr_count = int(np.count_nonzero(significant_fdr))
    non_17 = int(np.count_nonzero(correction_domain & ((n_hot != 17) | (n_cold != 17))))
    constant_finite_p = bool(finite_p.size and np.all(finite_p == finite_p[0]))
    all_identical_members = bool(
        np.all(np.nanstd(np.concatenate([hot_values, cold_values], axis=0), axis=0) == 0)
    )
    qc = {
        "figure": "Fig.7", "panel": date.strftime("%Y-%m-%d"), "variable": variable,
        "n_hot_min": int(n_hot[correction_domain].min()), "n_hot_max": int(n_hot[correction_domain].max()),
        "n_cold_min": int(n_cold[correction_domain].min()), "n_cold_max": int(n_cold[correction_domain].max()),
        "non_17_vs_17_gridpoints": non_17, "n_valid_tests": n_valid_tests,
        "invalid_ttest_gridpoints_with_n_ge_2": int(np.count_nonzero(correction_domain & (~insufficient) & ~np.isfinite(p_value))),
        "local_p005_count": local_count,
        "local_p005_fraction": local_count / n_valid_tests if n_valid_tests else np.nan,
        "fdr_005_count": fdr_count,
        "fdr_005_fraction": fdr_count / n_valid_tests if n_valid_tests else np.nan,
        "non_17_likely_reason": "finite-value missingness in member fields" if non_17 else "",
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
        "fdr_domain_mask_output": True,
        "significance_loader_fields": "TP and Z500 only",
        "latitude_order": "ascending" if field.latitude.values[0] < field.latitude.values[-1] else "descending",
        "longitude_min": float(field.longitude.min()), "longitude_max": float(field.longitude.max()),
        "fdr_longitude_range": "90 to 135 E", "fdr_latitude_range": "25 to 55 N",
        "alternative_explicit": alternative_explicit,
    }
    return ds, qc


def _fig7_significance_attrs(units: str, hot_members: Sequence[int], cold_members: Sequence[int]) -> dict:
    return {
        "test": "two-sided Welch independent-samples t test",
        "null_hypothesis": "mean(Hot17) - mean(Cold17) = 0",
        "alternative_hypothesis": "mean(Hot17) - mean(Cold17) != 0",
        "equal_var": "False", "fdr_method": "Benjamini-Hochberg", "fdr_alpha": 0.05,
        "difference_direction": "Hot17 minus Cold17", "grouping_stage": "Total",
        "grouping_file": GROUP_FILE, "analysis_domain": "90-135E, 25-55N",
        "units": units, "member_count_expected": 17,
        "hot_members": ",".join(map(str, hot_members)),
        "cold_members": ",".join(map(str, cold_members)),
        "member_sample_dimension": "member",
        "fdr_family": "one date and one variable",
        "composite_consistency_rtol": 1e-7,
        "composite_consistency_atol": 1e-6,
        "fdr_domain_mask": "included",
        "significance_loader": "TP and Z500 only",
        "interpretation": "conditional ensemble-group contrast",
        "scipy_version": scipy.__version__, "native_grid": "S2S; no interpolation or regridding",
        "statsmodels_version": statsmodels.__version__,
    }


def _update_spatial_significance_audit(section_name: str, section_text: str) -> None:
    os.makedirs(os.path.dirname(SPATIAL_SIGNIFICANCE_AUDIT_PATH), exist_ok=True)
    begin = f"<!-- BEGIN {section_name} -->"
    end = f"<!-- END {section_name} -->"
    header = "# Fig.7/Fig.9 spatial-significance runtime audit\n\n"
    if os.path.exists(SPATIAL_SIGNIFICANCE_AUDIT_PATH):
        with open(SPATIAL_SIGNIFICANCE_AUDIT_PATH, "r", encoding="utf-8") as handle:
            content = handle.read()
    else:
        content = header
    block = f"{begin}\n{section_text.rstrip()}\n{end}"
    if begin in content and end in content:
        content = content[:content.index(begin)] + block + content[content.index(end) + len(end):]
    else:
        content = content.rstrip() + "\n\n" + block + "\n"
    with open(SPATIAL_SIGNIFICANCE_AUDIT_PATH, "w", encoding="utf-8") as handle:
        handle.write(content)


def export_fig7_spatial_significance(
    tp_path: str = FIG7_SIGNIFICANCE_TP_PATH,
    z500_path: str = FIG7_SIGNIFICANCE_Z500_PATH,
    qc_path: str = FIG7_SIGNIFICANCE_QC_PATH,
    init_date: str = MAIN_INIT_DATE,
) -> Tuple[str, str]:
    """Export six independent Fig.7 date × variable FDR families from member fields."""
    os.makedirs(os.path.dirname(tp_path), exist_ok=True)
    group_df = load_total_hotcold_groups_fig7(GROUP_FILE)
    hot_members = get_hotcold_members_fig7(group_df, init_date, "Hot17")
    cold_members = get_hotcold_members_fig7(group_df, init_date, "Cold17")
    _fig7_validate_groups(hot_members, cold_members)
    fields = build_daily_realtime_significance_fields_fig7(init_date)
    original_differences = {
        variable: group_mean_fig7(fields[variable], hot_members)
        - group_mean_fig7(fields[variable], cold_members)
        for variable in ("tp", "z500")
    }

    outputs = {}
    qc_rows = []
    for variable, units in (("tp", "mm day^-1"), ("z500", "gpm")):
        panels = []
        for date in TARGET_DATES:
            panel, qc = _fig7_welch_panel(
                fields[variable], hot_members, cold_members,
                original_differences[variable], units, variable, pd.Timestamp(date),
            )
            panels.append(panel)
            qc.update({"initialization": init_date, "hot_members": str(hot_members), "cold_members": str(cold_members)})
            qc_rows.append(qc)
        result = xr.concat(panels, dim="date")
        result.attrs.update(_fig7_significance_attrs(units, hot_members, cold_members))
        result.attrs.update({"initialization": init_date, "analysis_window": "2023-06-18 to 2023-06-20", "variable": variable})
        outputs[variable] = result

    tp_p = outputs["tp"].p_value.values
    z500_p = outputs["z500"].p_value.values
    if np.array_equal(np.isnan(tp_p), np.isnan(z500_p)) and np.array_equal(
        np.where(np.isnan(tp_p), -1.0, tp_p), np.where(np.isnan(z500_p), -1.0, z500_p)
    ):
        raise ValueError("Fig.7 TP and Z500 p-value fields are identical; possible mask/p-value reuse")
    outputs["tp"].to_netcdf(tp_path)
    outputs["z500"].to_netcdf(z500_path)
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    audit_table = "```csv\n" + pd.DataFrame(qc_rows).to_csv(index=False) + "```"
    section = f"""## Fig.6/Fig.S6 runtime audit

- Source script: `cell_fig7_stageII_daily_synoptic_paper.py`; plotting script: `paper_fig7_from_cache.py`.
- Grouping file: `{GROUP_FILE}`; grouping stage: `Total`; init: `{init_date}`.
- Hot17 ({len(hot_members)}): `{hot_members}`
- Cold17 ({len(cold_members)}): `{cold_members}`
- Hot/Cold overlap: `{sorted(set(hot_members).intersection(cold_members))}`
- Sample axis: `member`; dates were tested separately and were not pooled.
- Welch parameters: two-sided, `equal_var=False`, `nan_policy=omit`; SciPy `{scipy.__version__}`.
- FDR: Benjamini-Hochberg, alpha=0.05; six families (three dates × TP/Z500), each restricted to 90–135 E, 25–55 N.
- Native-grid alignment: S2S latitude/longitude retained; no interpolation or regridding.
- Significance loader: TP/Z500 only; U850/V850 are not read.
- Composite consistency tolerance: `rtol=1e-7`, `atol=1e-6`; each family reports `max_absolute_mismatch` in the QC table.
- FDR-domain mask output: `fdr_domain_mask`; correction-domain grid count is reported for every family.
- Default run flags: `RUN_ORIGINAL_FIG7=False`, `RUN_EXPORT_FIG7_CACHE=False`, `RUN_EXPORT_FIG7_SIGNIFICANCE=True`.
- TP and Z500 masks were calculated independently: `True`.
- Output TP: `{tp_path}`
- Output Z500: `{z500_path}`
- QC CSV: `{qc_path}`
- Runtime audit: `{SPATIAL_SIGNIFICANCE_AUDIT_PATH}`
- Final figure stems: `final_from_cache/jgr_6p5in/fig6_daily_synoptic_init2023-06-12` or `figS6_daily_synoptic_init2023-06-08` (PNG/PDF).
- Display criterion: TP local two-sided Welch p < 0.05; Z500 significance is archived only.

{audit_table}

### Interpretation boundary

These p-values describe conditional spatial differences between the predefined Hot17 and Cold17 ensemble-member groups. The groups were defined by the established Total-period Tmax ranking. The results are statistical support for an ensemble-member contrast within one forecast system, not causal significance, independent confirmation of the grouping, significance of the 2023 event itself, or population inference from independent observations. BH-FDR controls the false-discovery rate across tested grid points within each specified family; it is not a stronger claim of global field significance.

### Fig.6/Fig.S6 caption supplement

Hatching denotes grid points where the Hot17−Cold17 precipitation difference is significant based on a two-sided Welch’s t test using local p < 0.05. Z500 difference contours do not display significance. Precipitation and Z500 were tested separately using the 17 members in each predefined group. Tests were conducted separately for each day.
"""
    _update_spatial_significance_audit("FIG6_" + init_date, section)
    print(f"[Fig7 significance TP] saved: {tp_path}")
    print(f"[Fig7 significance Z500] saved: {z500_path}")
    print(f"[Fig7 significance QC] saved: {qc_path}")
    return tp_path, z500_path


# ====================================================
# Plot-ready cache export (Fig.7)
# ====================================================

CACHE_ROOT_FIG7 = "/data1/huangy/fig6/NC/v12/paper_figures/cache"
FIG7_CACHE_PATH = os.path.join(CACHE_ROOT_FIG7, "fig7_daily_synoptic_plot_cache.nc")
FIG7_CACHE_QC_PATH = os.path.join(CACHE_ROOT_FIG7, "fig7_daily_synoptic_cache_qc.csv")


def _cache_da_fig7(da: xr.DataArray, source: str, units: str, description: str) -> xr.DataArray:
    lat_dim = f"{source}_lat"
    lon_dim = f"{source}_lon"
    rename_map = {
        old: new
        for old, new in {
            "bjt_date": "date",
            "latitude": lat_dim,
            "longitude": lon_dim,
        }.items()
        if (old in da.dims or old in da.coords) and new not in da.dims and new not in da.coords
    }
    out = da.rename(rename_map)
    required_dims = ("date", lat_dim, lon_dim)
    missing_dims = [dim for dim in required_dims if dim not in out.dims]
    if missing_dims:
        raise ValueError(
            f"Fig.7 cache field is missing required dims {missing_dims}; "
            f"got dims={out.dims}, sizes={dict(out.sizes)}"
        )
    out = out.transpose(*required_dims)
    out.attrs.update({"units": units, "description": description})
    return out


def _cache_qc_rows_fig7(ds: xr.Dataset, cache_file: str) -> List[dict]:
    rows = []
    for name, da in ds.data_vars.items():
        vals = da.values
        finite_fraction = float(np.isfinite(vals).sum() / vals.size) if vals.size else np.nan
        rows.append({
            "figure": "Fig.7",
            "cache_file": cache_file,
            "var_name": name,
            "dims": ",".join(da.dims),
            "shape": "x".join(str(da.sizes[d]) for d in da.dims),
            "units": da.attrs.get("units", ""),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "mean": float(np.nanmean(vals)),
            "nan_count": int(np.isnan(vals).sum()),
            "finite_fraction": finite_fraction,
            "description": da.attrs.get("description", ""),
        })
    return rows


def _validate_fig7_s2s_cache_density(ds: xr.Dataset) -> None:
    for name in ["tp_hot", "z500_hot", "tp_cold", "z500_cold", "tp_diff", "z500_diff"]:
        vals = ds[name].values
        finite_fraction = float(np.isfinite(vals).sum() / vals.size) if vals.size else 0.0
        if finite_fraction < 0.8:
            raise ValueError(
                f"Fig.7 S2S cache field {name} finite_fraction={finite_fraction:.3f} < 0.8; "
                "possible coordinate-union alignment occurred"
            )


def export_fig7_plot_cache(cache_path: str = FIG7_CACHE_PATH, qc_path: str = FIG7_CACHE_QC_PATH, init_date: str = MAIN_INIT_DATE) -> str:
    """Export plot-ready Fig.7 fields without changing science-processing logic."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    group_df = load_total_hotcold_groups_fig7(GROUP_FILE)
    obs_fields = load_observed_daily_fields_fig7()
    fields = build_daily_realtime_member_fields_fig7(MAIN_INIT_DATE)
    comp = build_group_composites_fig7(MAIN_INIT_DATE, group_df, fields)
    appendix_fields = build_daily_realtime_member_fields_fig7(APPENDIX_INIT_DATE)
    appendix_comp = build_group_composites_fig7(APPENDIX_INIT_DATE, group_df, appendix_fields)

    tp_abs_levels = _tp_abs_levels_fig7([
        obs_fields["tp"],
        comp["Hot17"]["tp"],
        comp["Cold17"]["tp"],
        appendix_comp["Hot17"]["tp"],
        appendix_comp["Cold17"]["tp"],
    ])
    tp_diff_levels, _ = _tp_diff_levels_fig7([
        comp[DIFF_GROUP]["tp"],
        appendix_comp[DIFF_GROUP]["tp"],
    ])

    if init_date not in (MAIN_INIT_DATE, APPENDIX_INIT_DATE):
        raise ValueError("Daily cache supports June 8 and June 12 only")
    if init_date == APPENDIX_INIT_DATE:
        comp = appendix_comp
    hot_members = get_hotcold_members_fig7(group_df, init_date, "Hot17")
    cold_members = get_hotcold_members_fig7(group_df, init_date, "Cold17")
    _fig7_validate_groups(hot_members, cold_members)
    ds = xr.Dataset({
        "tp_obs": _cache_da_fig7(obs_fields["tp"], "mswep", "mm day^-1", "MSWEP daily precipitation used for Fig.7 observed row"),
        "z500_obs": _cache_da_fig7(obs_fields["z500"], "era5", "gpm", "ERA5 absolute 500 hPa geopotential height used for Fig.7 observed row"),
        "tp_hot": _cache_da_fig7(comp["Hot17"]["tp"], "s2s", "mm day^-1", "S2S Total Hot17 daily precipitation composite"),
        "z500_hot": _cache_da_fig7(comp["Hot17"]["z500"], "s2s", "gpm", "S2S Total Hot17 absolute Z500 composite"),
        "tp_cold": _cache_da_fig7(comp["Cold17"]["tp"], "s2s", "mm day^-1", "S2S Total Cold17 daily precipitation composite"),
        "z500_cold": _cache_da_fig7(comp["Cold17"]["z500"], "s2s", "gpm", "S2S Total Cold17 absolute Z500 composite"),
        "tp_diff": _cache_da_fig7(comp[DIFF_GROUP]["tp"], "s2s", "mm day^-1", "Hot17 minus Cold17 daily precipitation difference"),
        "z500_diff": _cache_da_fig7(comp[DIFF_GROUP]["z500"], "s2s", "gpm", "Hot17 minus Cold17 daily Z500 difference"),
        "tp_abs_levels": xr.DataArray(
            tp_abs_levels,
            dims=("tp_abs_level",),
            attrs={
                "units": "mm day^-1",
                "description": "Fig.7 absolute TP colorbar levels computed from observed, main-init Hot/Cold, and appendix-init Hot/Cold fields",
            },
        ),
        "tp_diff_levels": xr.DataArray(
            tp_diff_levels,
            dims=("tp_diff_level",),
            attrs={
                "units": "mm day^-1",
                "description": "Fig.7 Hot17-minus-Cold17 TP difference colorbar levels computed from main-init and appendix-init differences",
            },
        ),
    })
    ds.attrs.update({
        "source_figure": "Fig.6 daily synoptic evolution" if init_date == MAIN_INIT_DATE else "Fig.S6 daily synoptic evolution",
        "source_script": "cell_fig7_stageII_daily_synoptic_paper.py",
        "analysis_window": "2023-06-18 to 2023-06-20",
        "grouping_file": GROUP_FILE,
        "grouping_stage": "Total",
        "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
        "z500_type": "absolute",
        "z500_diff_type": "Hot17_minus_Cold17 difference of absolute Z500",
        "z500_units": "gpm",
        "tp_units": "mm day^-1",
        "map_extent": str(MAP_EXTENT),
        "nchn_box": str(NCHN_BOX),
        "main_init_date": init_date,
        "initialization": init_date,
        "hot_members": ",".join(map(str, hot_members)),
        "cold_members": ",".join(map(str, cold_members)),
    })
    _validate_fig7_s2s_cache_density(ds)
    ds.to_netcdf(cache_path)
    pd.DataFrame(_cache_qc_rows_fig7(ds, cache_path)).assign(initialization=init_date, grouping_stage="Total", analysis_window="2023-06-18 to 2023-06-20").to_csv(qc_path, index=False)
    print(f"[Fig7 cache] saved: {cache_path}")
    print(f"[Fig7 cache QC] saved: {qc_path}")
    return cache_path

# June 8 exports remain separate from historical June 12 paths.
FIG7_0608_CACHE_PATH = os.path.join(CACHE_ROOT_FIG7, "fig7_daily_synoptic_init2023-06-08_plot_cache.nc")
FIG7_0608_CACHE_QC_PATH = os.path.join(CACHE_ROOT_FIG7, "fig7_daily_synoptic_init2023-06-08_cache_qc.csv")
FIG7_0608_TP_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG7, "fig7_init2023-06-08_spatial_significance_tp.nc")
FIG7_0608_Z500_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG7, "fig7_init2023-06-08_spatial_significance_z500.nc")
FIG7_0608_SIG_QC_PATH = os.path.join(SIGNIFICANCE_ROOT_FIG7, "fig7_init2023-06-08_spatial_significance_qc.csv")

def export_fig7_20230608_plot_cache():
    return export_fig7_plot_cache(FIG7_0608_CACHE_PATH, FIG7_0608_CACHE_QC_PATH, APPENDIX_INIT_DATE)

def export_fig7_20230608_spatial_significance():
    return export_fig7_spatial_significance(FIG7_0608_TP_PATH, FIG7_0608_Z500_PATH, FIG7_0608_SIG_QC_PATH, APPENDIX_INIT_DATE)

# Loading this complete cell defines functions; exports are explicit.
