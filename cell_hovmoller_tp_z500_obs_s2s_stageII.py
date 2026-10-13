import glob
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import scipy
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore", category=FutureWarning)

# ====================================================
# Configuration
# ====================================================

WINDOWS = {
    "0617_0620": {
        "dates": pd.to_datetime(["2023-06-17", "2023-06-18", "2023-06-19", "2023-06-20"]),
        "label": "2023-06-17 to 2023-06-20",
        "stage_split_labels": False,
    },
    "0614_0624": {
        "dates": pd.date_range("2023-06-14", "2023-06-24", freq="D"),
        "label": "2023-06-14 to 2023-06-24",
        "stage_split_labels": True,
    },
}
INIT_DATES = ["2023-06-08", "2023-06-12"]
BJT_OFFSET = pd.Timedelta(hours=8)
G0 = 9.80665

LAT_BAND = (35.0, 44.0)
LON_RANGE = (90.0, 135.0)
DATA_EXTENT = (88.0, 137.0, 33.0, 46.0)

BASE_DATA_DIR = "/data1/huangy/fig6/NC"
CF_DIR = f"{BASE_DATA_DIR}/cf_2023-6"
ERA5_Z500_FILE = f"{BASE_DATA_DIR}/MSE/ERA5/Z500/era5_z500_2023_06.nc"
ERA5_TP_FILE = f"{BASE_DATA_DIR}/MSE/ERA5/tp/era5_tp_2023_06.nc"
MSWEP_2023_DIR = f"{BASE_DATA_DIR}/MSE/MSWEP/3hourly-2023"
GROUP_FILE = "/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv"

OUT_DIR = "/data1/huangy/fig6/NC/v10/hovmoller_tp_z500"
os.makedirs(OUT_DIR, exist_ok=True)

GROUP_ORDER = ["Hot17", "Middle17", "Cold17", "Hot17_minus_Cold17"]
GROUP_LABELS = {
    "Hot17": "Total Hot17",
    "Middle17": "Total Middle17",
    "Cold17": "Total Cold17",
    "Hot17_minus_Cold17": "Total Hot17 − Cold17",
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
    }
}

GRIB_SHORTNAME_ORDER = {
    "pl_500": ("gh", "z"),
    "sfc_tp": ("tp",),
}

# ====================================================
# Generic helpers
# ====================================================

def _guess_data_var(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    lower = {v.lower(): v for v in ds.data_vars}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for var in ds.data_vars:
        short_name = str(ds[var].attrs.get("GRIB_shortName", "")).lower()
        standard_name = str(ds[var].attrs.get("standard_name", "")).lower()
        long_name = str(ds[var].attrs.get("long_name", "")).lower()
        if any(c.lower() == short_name for c in candidates):
            return var
        if any(c.lower() in standard_name or c.lower() in long_name for c in candidates):
            return var
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(f"Could not identify variable from candidates {candidates}; available={list(ds.data_vars)}")


def _standardize_lat_lon(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for name in list(da.dims) + list(da.coords):
        low = name.lower()
        if low in ("lat", "latitude") and name != "latitude":
            rename[name] = "latitude"
        if low in ("lon", "longitude") and name != "longitude":
            rename[name] = "longitude"
    da = da.rename(rename) if rename else da
    if "longitude" not in da.coords or "latitude" not in da.coords:
        raise KeyError(f"Expected latitude/longitude coords after standardization; dims={da.dims}, coords={list(da.coords)}")
    if float(da.longitude.max()) > 180.0:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")
    if float(da.latitude.values[0]) > float(da.latitude.values[-1]):
        da = da.sortby("latitude")
    return da


def _sel_region(da: xr.DataArray, extent: Tuple[float, float, float, float]) -> xr.DataArray:
    da = _standardize_lat_lon(da)
    lon_min, lon_max, lat_min, lat_max = extent
    return da.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))


def _select_level(da: xr.DataArray, level_hpa: int) -> xr.DataArray:
    da = _standardize_lat_lon(da)
    for lev in ("isobaricInhPa", "isobaricInPa", "level", "pressure_level", "plev"):
        if lev in da.dims:
            vals = np.asarray(da[lev].values)
            target = level_hpa * 100 if np.nanmax(vals) > 2000 else level_hpa
            return da.sel({lev: target}, method="nearest")
        if lev in da.coords:
            vals = np.asarray(da[lev].values)
            target = level_hpa * 100 if np.nanmax(vals) > 2000 else level_hpa
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


def _normalize_time_dim(da: xr.DataArray) -> xr.DataArray:
    if "time" in da.dims:
        return da
    if "valid_time" in da.dims:
        return da.rename({"valid_time": "time"})
    if "time" in da.coords and "time" not in da.dims:
        return da.expand_dims(time=np.atleast_1d(da.time.values))
    if "valid_time" in da.coords and "valid_time" not in da.dims:
        vt = np.atleast_1d(da.valid_time.values)
        return da.drop_vars("valid_time", errors="ignore").expand_dims(time=vt)
    raise KeyError(f"No time/valid_time found. dims={da.dims}, coords={list(da.coords)}")


def _to_bjt_daily(da: xr.DataArray, how: str) -> xr.DataArray:
    da = _normalize_time_dim(da)
    shifted = pd.to_datetime(da.time.values) + BJT_OFFSET
    da = da.assign_coords(time=shifted)
    if how == "sum":
        out = da.resample(time="1D").sum(skipna=True)
    elif how == "mean":
        out = da.resample(time="1D").mean(skipna=True)
    else:
        raise ValueError(how)
    return out.rename({"time": "bjt_date"})


def _to_gpm_if_needed(da: xr.DataArray, var_label: str) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    vmax = float(np.nanpercentile(np.abs(da.values), 95)) if da.size else np.nan
    if "m**2" in units or "m2 s-2" in units or "m^2" in units or vmax > 20000:
        da = da / G0
        da.attrs["converted_to_gpm"] = "yes: divided by 9.80665"
        print(f"[{var_label}] converted geopotential to gpm")
    else:
        da.attrs["converted_to_gpm"] = "not needed or already height"
    da.attrs["units"] = "gpm"
    return da


def _tp_to_mm(da: xr.DataArray) -> xr.DataArray:
    units_raw = str(da.attrs.get("units", "")).strip()
    units = units_raw.lower().replace(" ", "")
    if units in ("m", "meter", "metre", "meters", "metres"):
        da = da * 1000.0
    elif units in ("kgm**-2", "kgm-2", "kg/m^2", "kgm^-2", "mm", "millimeter", "millimetre"):
        pass
    else:
        warnings.warn(f"[tp] Unknown precipitation units '{units_raw}'; not applying *1000 automatically.")
    da.attrs["units"] = "mm day-1"
    return da


def _latband_hov(da: xr.DataArray) -> xr.DataArray:
    sub = _sel_region(da, (LON_RANGE[0], LON_RANGE[1], LAT_BAND[0], LAT_BAND[1]))
    weights = np.cos(np.deg2rad(sub.latitude))
    out = sub.weighted(weights).mean("latitude", skipna=True)
    if "bjt_date" in out.dims:
        out = out.transpose("bjt_date", "longitude")
    return out


def _check_target_dates(da: xr.DataArray, name: str, dates: Sequence[pd.Timestamp]) -> None:
    dates = pd.to_datetime(dates)
    have = set(pd.to_datetime(da.bjt_date.values).normalize())
    missing = [d.strftime("%Y-%m-%d") for d in dates if d.normalize() not in have]
    print(f"[{name}] BJT dates={list(pd.to_datetime(da.bjt_date.values).strftime('%Y-%m-%d'))}")
    if missing:
        raise RuntimeError(f"{name} missing target BJT dates: {missing}")

# ====================================================
# Observation / reanalysis loaders for Figure A
# ====================================================

def parse_mswep_filename(path: str) -> pd.Timestamp:
    stem = os.path.basename(path).replace(".nc", "")
    yyyyddd, hh = stem.split(".")
    year = int(yyyyddd[:4])
    doy = int(yyyyddd[4:])
    hour = int(hh)
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1) + pd.Timedelta(hours=hour)


def load_era5_z500_hov(dates: Sequence[pd.Timestamp]) -> xr.DataArray:
    if not os.path.exists(ERA5_Z500_FILE):
        raise FileNotFoundError(f"ERA5_Z500_FILE not found: {ERA5_Z500_FILE}")
    ds = xr.open_dataset(ERA5_Z500_FILE)
    var = _guess_data_var(ds, ["z", "gh", "geopotential", "geopotential_height", "z500"])
    z = _select_level(ds[var], 500)
    z = _to_gpm_if_needed(z, "ERA5 Z500")
    z = _sel_region(z, DATA_EXTENT)
    z_daily = _to_bjt_daily(z, "mean").sel(bjt_date=pd.to_datetime(dates))
    return _latband_hov(z_daily)


def load_era5_tp_hov(dates: Sequence[pd.Timestamp]) -> xr.DataArray:
    if not os.path.exists(ERA5_TP_FILE):
        raise FileNotFoundError(f"ERA5_TP_FILE not found: {ERA5_TP_FILE}")
    ds = xr.open_dataset(ERA5_TP_FILE)
    var = _guess_data_var(ds, ["tp", "total_precipitation", "precip", "precipitation"])
    tp = _sel_region(ds[var], DATA_EXTENT)
    tp_daily = _tp_to_mm(_to_bjt_daily(tp, "sum")).sel(bjt_date=pd.to_datetime(dates))
    return _latband_hov(tp_daily)


def load_mswep_hov(dates: Sequence[pd.Timestamp]) -> xr.DataArray:
    dates = pd.to_datetime(dates)
    if not os.path.isdir(MSWEP_2023_DIR):
        raise FileNotFoundError(f"MSWEP_2023_DIR not found: {MSWEP_2023_DIR}")
    utc0 = dates.min() - BJT_OFFSET - pd.Timedelta(hours=6)
    utc1 = dates.max() + pd.Timedelta(days=1) - BJT_OFFSET + pd.Timedelta(hours=6)
    files = []
    for path in sorted(glob.glob(os.path.join(MSWEP_2023_DIR, "*.nc"))):
        try:
            t = parse_mswep_filename(path)
        except Exception:
            continue
        if utc0 <= t <= utc1:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No MSWEP files found in {MSWEP_2023_DIR} for UTC window {utc0} to {utc1}")

    parts = []
    for path in files:
        ds = xr.open_dataset(path)
        var = _guess_data_var(ds, ["precipitation", "precip", "pr", "rain", "tp"])
        da = _standardize_lat_lon(ds[var])
        if "time" in da.dims or "valid_time" in da.dims or "time" in da.coords or "valid_time" in da.coords:
            da = _normalize_time_dim(da)
        else:
            da = da.expand_dims(time=[np.datetime64(parse_mswep_filename(path).to_datetime64())])
        parts.append(_sel_region(da, DATA_EXTENT))
    pr = xr.concat(parts, dim="time").sortby("time")
    pr_daily = _tp_to_mm(_to_bjt_daily(pr, "sum")).sel(bjt_date=dates)
    return _latband_hov(pr_daily)

# ====================================================
# S2S Total-stage grouping helpers for Figure B
# ====================================================

def _norm_group_text(value) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    norm = {_norm_group_text(c): c for c in columns}
    for cand in candidates:
        key = _norm_group_text(cand)
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
    return series.astype(str).map(_norm_group_text).isin({"total", "totalstage"})


def load_total_hotcold_groups(group_file: str) -> pd.DataFrame:
    if not os.path.exists(group_file):
        raise FileNotFoundError(f"GROUP_FILE not found: {group_file}")
    df = pd.read_csv(group_file)
    print(f"[Group file] columns={list(df.columns)}")

    init_col = _find_column(df.columns, ["init_date", "init", "initial_date", "forecast_init", "date"])
    stage_col = _find_column(df.columns, ["stage", "grouping_stage", "tmax_stage"])
    if init_col is None or stage_col is None:
        raise KeyError(f"Could not identify init/stage columns; columns={list(df.columns)}")

    rows = []
    group_col = _find_column(df.columns, ["hotcold_group", "group", "hotcold", "tmax_group"])
    member_col = _find_column(df.columns, ["member", "members", "member_id", "number"])
    parse_groups = ["Hot17", "Middle17", "Cold17"]

    for init in INIT_DATES:
        df_init = df[_init_match(df[init_col], init)].copy()
        df_total = df_init[_total_stage_mask(df_init[stage_col])].copy()
        if df_total.empty:
            raise ValueError(
                f"No Total-stage rows found for init={init}. "
                f"columns={list(df.columns)}; init rows={len(df_init)}"
            )

        if group_col is not None and member_col is not None:
            for group in parse_groups:
                sub = df_total[df_total[group_col].astype(str).map(_norm_group_text) == _norm_group_text(group)]
                if sub.empty and group == "Middle17":
                    continue
                members = sorted(dict.fromkeys(_parse_member_list(sub[member_col])))
                for member in members:
                    rows.append({
                        "init_date": init,
                        "grouping_stage": "Total",
                        "hotcold_group": group,
                        "member": int(member),
                        "source_file": group_file,
                    })
        else:
            for group in parse_groups:
                col = _find_column(df_total.columns, [group, f"{group}_members", f"{group.lower()}_members", f"{group.lower()}"])
                if col is None:
                    if group == "Middle17":
                        continue
                    raise KeyError(
                        f"Could not identify members for {group}; columns={list(df_total.columns)}; "
                        f"head={df_total.head().to_dict(orient='records')}"
                    )
                members = sorted(dict.fromkeys(_parse_member_list(df_total.iloc[0][col])))
                for member in members:
                    rows.append({
                        "init_date": init,
                        "grouping_stage": "Total",
                        "hotcold_group": group,
                        "member": int(member),
                        "source_file": group_file,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No Total-stage members parsed from {group_file}")
    for init in INIT_DATES:
        for group in ["Hot17", "Cold17"]:
            members = get_hotcold_members(out, init, group)
            if len(members) != 17:
                raise ValueError(f"Expected 17 Total-stage members for init={init}, group={group}, got {len(members)}")
    return out


def get_hotcold_members(group_df: pd.DataFrame, init: str, group: str) -> List[int]:
    sub = group_df[
        (group_df["init_date"].astype(str) == str(init))
        & (group_df["grouping_stage"].astype(str) == "Total")
        & (group_df["hotcold_group"].astype(str) == str(group))
    ].copy()
    members = sub["member"].astype(int).tolist()
    expected = 17 if group in ("Hot17", "Cold17", "Middle17") else len(members)
    if len(members) != expected:
        raise ValueError(f"Expected {expected} members for init={init}, grouping_stage=Total, group={group}, got {len(members)}; members={members}")
    if len(set(members)) != len(members):
        raise ValueError(f"Duplicated members for init={init}, group={group}: {members}")
    if min(members) < 0 or max(members) > 50:
        raise ValueError(f"Member IDs should be 0..50, got init={init}, group={group}, members={members}")
    return members


def build_group_audit_table(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    audit_groups = [g for g in ["Hot17", "Middle17", "Cold17"] if not group_df[group_df["hotcold_group"] == g].empty]
    for init in INIT_DATES:
        for group in audit_groups:
            members = get_hotcold_members(group_df, init, group)
            rows.append({
                "init_date": init,
                "grouping_stage": "Total",
                "group": group,
                "member_count": len(members),
                "members": ",".join(str(m) for m in members),
                "source_file": GROUP_FILE,
                "analysis_stage": "Stage-II diagnostic dates",
                "analysis_window": "2023-06-14 to 2023-06-24",
                "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
            })
    return pd.DataFrame(rows)


def _format_candidate_paths(patterns: Sequence[str], init_date: str) -> List[str]:
    init = pd.Timestamp(init_date)
    fields = {
        "init_date": init.strftime("%Y-%m-%d"),
        "init_yyyymmdd": init.strftime("%Y%m%d"),
        "init_mmdd": init.strftime("%m%d"),
        "init_yyyy_mm": init.strftime("%Y-%m"),
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


def _init_coord_name(da: xr.DataArray) -> Optional[str]:
    for name in ("time", "forecast_reference_time", "date"):
        if name in da.coords or name in da.dims:
            return name
    return None


def get_s2s_init_slice(da: xr.DataArray, init_date: str) -> xr.DataArray:
    coord = _init_coord_name(da)
    if coord is None:
        raise ValueError(
            "Realtime S2S field has no initialization coordinate: "
            f"requested_init={init_date}, coords={list(da.coords)}, "
            f"dims={da.dims}, sizes={dict(da.sizes)}"
        )

    requested = pd.Timestamp(init_date)
    raw_values = np.asarray(da[coord].values).reshape(-1)
    try:
        parsed_values = pd.to_datetime(raw_values)
    except Exception as exc:
        raise ValueError(
            "Could not parse realtime S2S initialization coordinate: "
            f"requested_init={init_date}, coordinate={coord}, "
            f"available_values={raw_values.tolist()}, dims={da.dims}, "
            f"sizes={dict(da.sizes)}"
        ) from exc

    match_indices = [
        index
        for index, value in enumerate(parsed_values)
        if pd.Timestamp(value).date() == requested.date()
    ]
    available = [str(pd.Timestamp(value)) for value in parsed_values]
    if not match_indices:
        raise ValueError(
            "Requested realtime S2S initialization was not found: "
            f"requested_init={init_date}, coordinate={coord}, "
            f"available_initializations={available}, dims={da.dims}, "
            f"sizes={dict(da.sizes)}"
        )
    if len(match_indices) != 1:
        matched = [available[index] for index in match_indices]
        raise ValueError(
            "Requested realtime S2S initialization has multiple same-date matches: "
            f"requested_init={init_date}, coordinate={coord}, matched_values={matched}, "
            f"available_initializations={available}, dims={da.dims}, "
            f"sizes={dict(da.sizes)}"
        )

    matched_raw = raw_values[match_indices[0]]
    matched_text = available[match_indices[0]]
    if coord in da.dims:
        da = da.sel({coord: matched_raw})
    if coord in da.dims and da.sizes.get(coord) == 1:
        da = da.squeeze(coord, drop=True)
    print(
        f"[Init select] requested={init_date} coord={coord} "
        f"matched={matched_text} remaining_dims={da.dims}"
    )
    return da


def _concat_cf_pf(cf: Optional[xr.DataArray], pf: Optional[xr.DataArray]) -> xr.DataArray:
    parts = [_normalize_member_dim(da, stream) for stream, da in (("cf", cf), ("pf", pf)) if da is not None]
    if not parts:
        raise FileNotFoundError("Neither CF nor PF field was loaded")
    out = xr.concat(parts, dim="member", coords="minimal", compat="override")
    return out.assign_coords(member=np.arange(out.sizes["member"]))


def _load_one_stream(
    stream: str,
    product: str,
    init_date: str,
    candidates: Sequence[str],
) -> xr.DataArray:
    if product not in GRIB_SHORTNAME_ORDER:
        raise KeyError(
            f"No product-specific GRIB shortName rule for product={product!r}; "
            f"known_products={list(GRIB_SHORTNAME_ORDER)}"
        )
    patterns = S2S_FILE_CATALOG["realtime"][stream].get(product, [])
    paths = _resolve_patterns(patterns, init_date)
    if not paths:
        tried = _format_candidate_paths(patterns, init_date)
        raise FileNotFoundError(
            f"No realtime GRIB matched: stream={stream}, product={product}, "
            f"requested_init={init_date}, attempted_paths={tried}, "
            f"attempted_shortNames={list(GRIB_SHORTNAME_ORDER[product])}"
        )

    short_name_attempts = GRIB_SHORTNAME_ORDER[product]
    errors = []
    for path in paths:
        for sn in short_name_attempts:
            try:
                ds = _open_cfgrib(path, short_name=sn)
                print(f"[Open] realtime/{stream}/{product}: {os.path.basename(path)} shortName={sn} data_vars={list(ds.data_vars)}")
                var = _guess_data_var(ds, candidates)
            except Exception as exc:
                print(f"[Skip] {path} shortName={sn}: {exc}")
                errors.append(f"path={path}, shortName={sn}: {type(exc).__name__}: {exc}")
                continue

            da = get_s2s_init_slice(ds[var], init_date)
            remaining_init_dims = {
                name: int(da.sizes[name])
                for name in ("time", "forecast_reference_time", "date")
                if name in da.dims and da.sizes[name] > 1
            }
            if remaining_init_dims:
                raise ValueError(
                    "Realtime field retains multiple initialization values after selection: "
                    f"stream={stream}, product={product}, path={path}, "
                    f"requested_init={init_date}, remaining_init_dims={remaining_init_dims}, "
                    f"dims={da.dims}, sizes={dict(da.sizes)}"
                )
            print(f"[Loaded] realtime/{stream}/{product}: {os.path.basename(path)} -> {var}")
            return da
    raise RuntimeError(
        f"Failed to load realtime GRIB: stream={stream}, product={product}, "
        f"paths={paths}, attempted_shortNames={list(short_name_attempts)}, "
        f"collected_errors={errors}"
    )


def load_s2s_ensemble_with_cf(
    product: str,
    init_date: str,
    candidates: Sequence[str],
) -> xr.DataArray:
    cf = _load_one_stream("cf", product, init_date, candidates)
    pf = _load_one_stream("pf", product, init_date, candidates)
    da = _concat_cf_pf(cf, pf)
    da = _sel_region(da, DATA_EXTENT)
    member_count = int(da.sizes.get("member", 0))
    print(f"[Members] realtime/{product}/{init_date}: member_count={member_count}")
    if member_count != 51:
        raise ValueError(f"Expected 51 S2S members after CF+PF merge, got {member_count}")
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
        next_midnight = cursor.normalize() + pd.Timedelta(days=1)
        stop = min(end, next_midnight)
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
        start = bjt_valid[i - 1]
        end = bjt_valid[i]
        inc_i = inc.isel(step=i - 1).drop_vars("step", errors="ignore")
        for day, frac in _interval_day_fractions(start, end):
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


def build_daily_realtime_member_fields(init_date: str, dates: Sequence[pd.Timestamp]) -> Dict[str, xr.DataArray]:
    print(f"\n=== Loading S2S realtime fields for init={init_date} ===")
    z500_raw = load_s2s_ensemble_with_cf(
        "pl_500", init_date, ["gh", "z", "geopotential"]
    )
    tp_raw = load_s2s_ensemble_with_cf(
        "sfc_tp", init_date, ["tp", "total_precipitation"]
    )

    z500 = _to_gpm_if_needed(_select_level(z500_raw, 500), "S2S Z500 realtime")
    z500_daily = process_s2s_bjt_daily_mean(z500, init_date).sel(bjt_date=pd.to_datetime(dates))
    tp_daily = process_s2s_bjt_daily_accum(tp_raw, init_date, output="tp", force_non_negative=True).sel(bjt_date=pd.to_datetime(dates))
    return {"tp": tp_daily, "z500": z500_daily}


def group_mean(da: xr.DataArray, members: List[int]) -> xr.DataArray:
    return da.sel(member=members).mean("member", skipna=True)


def build_group_hovmollers_for_init(
    init_date: str,
    group_df: pd.DataFrame,
    fields: Dict[str, xr.DataArray],
    member_hovs: Optional[Dict[str, xr.DataArray]] = None,
) -> Dict[str, Dict[str, xr.DataArray]]:
    if member_hovs is None:
        member_hovs = {
            "tp": _fig8_member_hov(fields["tp"]),
            "z500": _fig8_member_hov(fields["z500"]),
        }
    tp_member_hov = member_hovs["tp"]
    z500_member_hov = member_hovs["z500"]
    out = {}
    for group in ["Hot17", "Middle17", "Cold17"]:
        if group == "Middle17" and group_df[group_df["hotcold_group"] == "Middle17"].empty:
            continue
        members = get_hotcold_members(group_df, init_date, group)
        tp = tp_member_hov.sel(member=members).mean("member", skipna=True)
        z500 = z500_member_hov.sel(member=members).mean("member", skipna=True)
        out[group] = {"members": members, "tp": tp, "z500": z500}
    out["Hot17_minus_Cold17"] = {
        "members": None,
        "tp": out["Hot17"]["tp"] - out["Cold17"]["tp"],
        "z500": out["Hot17"]["z500"] - out["Cold17"]["z500"],
    }
    return out

# ====================================================
# Plotting helpers
# ====================================================

def _hov_y_coords(dates: Sequence[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dates = pd.to_datetime(dates)
    y_centers = np.arange(len(dates), dtype=float)
    y_edges = np.arange(len(dates) + 1, dtype=float) - 0.5
    y_labels = [d.strftime("%Y-%m-%d") for d in dates]
    return y_centers, y_edges, y_labels


def _pad_hov_for_contourf(values, dates: Sequence[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad the first and last daily rows so contourf fills from
    first-day upper edge to last-day lower edge while keeping
    smooth contourf shading. Values shape should be (time, lon).
    """
    dates = pd.to_datetime(dates)
    y_centers = np.arange(len(dates), dtype=float)
    y_pad = np.concatenate(([y_centers[0] - 0.5], y_centers, [y_centers[-1] + 0.5]))
    values = np.asarray(values)
    v_pad = np.vstack([values[0:1, :], values, values[-1:, :]])
    return y_centers, y_pad, v_pad


def _color_stage_yticklabels(ax, dates: Sequence[pd.Timestamp], enable: bool = True) -> None:
    if not enable:
        return
    for tick_label, d in zip(ax.get_yticklabels(), pd.to_datetime(dates)):
        if d <= pd.Timestamp("2023-06-17"):
            tick_label.set_color("#D55E00")
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color("#0072B2")
            tick_label.set_fontweight("bold")


def _plot_nchn_lon_lines(ax):
    # Reference longitude bounds for the NCHN box, not Z500/5880-gpm contours.
    ax.axvline(114.0, color="0.25", lw=0.8, ls="--")
    ax.axvline(119.0, color="0.25", lw=0.8, ls="--")


def _z500_abs_levels(z500: xr.DataArray) -> np.ndarray:
    zmin = float(np.nanmin(z500.values))
    zmax = float(np.nanmax(z500.values))
    lo = np.floor(zmin / 20.0) * 20.0
    hi = np.ceil(zmax / 20.0) * 20.0
    return np.arange(lo, hi + 1e-6, 20.0)


def _apply_label_path_effects(texts) -> None:
    for txt in texts:
        txt.set_color("0.10")
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def _contour_z500_abs(ax, z500: xr.DataArray, dates: Sequence[pd.Timestamp]):
    y_centers, _, _ = _hov_y_coords(dates)
    data = z500.values
    zmin = float(np.nanmin(data))
    zmax = float(np.nanmax(data))
    levels = _z500_abs_levels(z500)
    valid = levels[(levels >= zmin) & (levels <= zmax)]
    if valid.size:
        cs = ax.contour(z500.longitude, y_centers, data, levels=valid, colors="0.20", linewidths=1.0)
        if len(cs.levels):
            _apply_label_path_effects(ax.clabel(cs, levels=cs.levels[::2], fmt="%.0f", fontsize=7, inline=True))
    if zmin <= 5880 <= zmax:
        ax.contour(z500.longitude, y_centers, data, levels=[5880], colors="purple", linewidths=2.0)
    else:
        print(f"[Info] 5880 outside z500 range for Hovmoller panel: zmin={zmin:.1f}, zmax={zmax:.1f}")


def _contour_z500_diff(ax, z500_diff: xr.DataArray, dates: Sequence[pd.Timestamp]):
    y_centers, _, _ = _hov_y_coords(dates)
    levels = np.array([-80, -60, -40, -20, 20, 40, 60, 80])
    data = z500_diff.values
    levels = levels[(levels >= np.nanmin(data)) & (levels <= np.nanmax(data))]
    neg = levels[levels < 0]
    pos = levels[levels > 0]
    if neg.size:
        cs_neg = ax.contour(z500_diff.longitude, y_centers, data, levels=neg, colors="royalblue", linestyles="--", linewidths=1.0)
        _apply_label_path_effects(ax.clabel(cs_neg, levels=cs_neg.levels[::2], fmt="%.0f", fontsize=7, inline=True))
    if pos.size:
        cs_pos = ax.contour(z500_diff.longitude, y_centers, data, levels=pos, colors="firebrick", linestyles="-", linewidths=1.0)
        _apply_label_path_effects(ax.clabel(cs_pos, levels=cs_pos.levels[::2], fmt="%.0f", fontsize=7, inline=True))


def _format_hov_axis(
    ax,
    ylabel: str,
    show_xlabel: bool,
    dates: Sequence[pd.Timestamp],
    stage_split_labels: bool = False,
):
    dates = pd.to_datetime(dates)
    y_centers, y_edges, y_labels = _hov_y_coords(dates)
    ax.set_xlim(*LON_RANGE)
    ax.set_ylim(y_centers[-1] + 0.5, y_centers[0] - 0.5)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(y_labels)
    _color_stage_yticklabels(ax, dates, enable=stage_split_labels)
    ax.tick_params(axis="y", pad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    if show_xlabel:
        ax.set_xlabel("Longitude (°E)")
    _plot_nchn_lon_lines(ax)
    ax.grid(True, axis="x", ls="--", lw=0.4, alpha=0.5)


def _save_fig(fig, stem: str):
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=250, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _abs_tp_levels(arrays: Sequence[xr.DataArray]) -> np.ndarray:
    pool = np.concatenate([a.values.ravel() for a in arrays])
    if np.all(np.isnan(pool)):
        raise ValueError("All absolute TP values are NaN; cannot build Hovmoller TP levels")
    pmax = max(float(np.nanpercentile(pool, 95)), 1.0)
    pmax = min(pmax, 25.0)
    return np.linspace(0, pmax, 13)


def _diff_tp_levels(tp_diff: xr.DataArray) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    vals = tp_diff.values
    if np.all(np.isnan(vals)):
        raise ValueError("All TP difference values are NaN; cannot build Hovmoller difference levels")
    vmax = max(float(np.nanpercentile(np.abs(vals), 98)), 0.5)
    levels = np.linspace(-vmax, vmax, 13)
    return levels, mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _diff_tp_levels_from_arrays(arrays: Sequence[xr.DataArray]) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    pool = np.concatenate([np.abs(a.values).ravel() for a in arrays])
    if np.all(np.isnan(pool)):
        raise ValueError("All TP difference values are NaN; cannot build Hovmoller difference levels")
    vmax = max(float(np.nanpercentile(pool, 98)), 0.5)
    levels = np.linspace(-vmax, vmax, 13)
    return levels, mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _window_figsize(base: Tuple[float, float], dates: Sequence[pd.Timestamp], full_height: float) -> Tuple[float, float]:
    dates = pd.to_datetime(dates)
    return (base[0], full_height) if len(dates) > 4 else base


def _add_panel_label(ax, label: str) -> None:
    ax.text(
        0.012,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        zorder=30,
    )


def _plot_hov_abs_panel(
    ax,
    tp: xr.DataArray,
    z500: xr.DataArray,
    dates: Sequence[pd.Timestamp],
    levels: np.ndarray,
    panel_label: str,
    title: str,
    show_xlabel: bool = False,
):
    _, y_pad, tp_pad = _pad_hov_for_contourf(tp.values, dates)
    cf = ax.contourf(tp.longitude.values, y_pad, tp_pad, levels=levels, cmap="Blues", extend="max")
    _contour_z500_abs(ax, z500, dates)
    _format_hov_axis(ax, "BJT date", show_xlabel=show_xlabel, dates=dates, stage_split_labels=True)
    ax.set_title(title, fontsize=10)
    _add_panel_label(ax, panel_label)
    return cf


def _plot_hov_diff_panel(
    ax,
    tp_diff: xr.DataArray,
    z500_diff: xr.DataArray,
    dates: Sequence[pd.Timestamp],
    diff_levels: np.ndarray,
    diff_norm: mcolors.TwoSlopeNorm,
    panel_label: str,
    title: str,
    show_xlabel: bool = False,
):
    _, y_pad, tp_pad = _pad_hov_for_contourf(tp_diff.values, dates)
    cf = ax.contourf(
        tp_diff.longitude.values,
        y_pad,
        tp_pad,
        levels=diff_levels,
        cmap="RdBu_r",
        norm=diff_norm,
        extend="both",
    )
    _contour_z500_diff(ax, z500_diff, dates)
    _format_hov_axis(ax, "BJT date", show_xlabel=show_xlabel, dates=dates, stage_split_labels=True)
    ax.set_title(title, fontsize=10)
    _add_panel_label(ax, panel_label)
    return cf


def plot_figure8_main_obs_s2s_0614_0624(
    era5_z500: xr.DataArray,
    era5_tp: xr.DataArray,
    mswep_tp: xr.DataArray,
    hov_0612: Dict[str, Dict[str, xr.DataArray]],
    abs_levels: np.ndarray,
    diff_levels: np.ndarray,
    diff_norm: mcolors.TwoSlopeNorm,
):
    """Paper Fig. 8 main layout: MSWEP obs + S2S Hot/Cold/Hot-Cold.

    The era5_tp argument is retained for backward-compatible call sites, but the
    main paper panel intentionally uses MSWEP precipitation with ERA5 Z500.
    """
    dates = pd.to_datetime(era5_z500.bjt_date.values)
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.8), sharex=True, sharey=True)
    ax_a, ax_b = axs[0, 0], axs[0, 1]
    ax_c, ax_d = axs[1, 0], axs[1, 1]

    cf_abs = _plot_hov_abs_panel(
        ax_a,
        mswep_tp,
        era5_z500,
        dates,
        abs_levels,
        "(a)",
        "MSWEP TP + ERA5 Z500",
    )
    _plot_hov_abs_panel(ax_b, hov_0612["Hot17"]["tp"], hov_0612["Hot17"]["z500"], dates, abs_levels, "(b)", "Total Hot17")
    _plot_hov_abs_panel(ax_c, hov_0612["Cold17"]["tp"], hov_0612["Cold17"]["z500"], dates, abs_levels, "(c)", "Total Cold17", show_xlabel=True)
    cf_diff = _plot_hov_diff_panel(
        ax_d,
        hov_0612["Hot17_minus_Cold17"]["tp"],
        hov_0612["Hot17_minus_Cold17"]["z500"],
        dates,
        diff_levels,
        diff_norm,
        "(d)",
        "Total Hot17 − Cold17",
        show_xlabel=True,
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.18, wspace=0.15, hspace=0.24)
    cax_abs = fig.add_axes([0.18, 0.085, 0.30, 0.025])
    cbar_abs = fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_label("Precipitation (mm day$^{-1}$)")
    cax_diff = fig.add_axes([0.60, 0.085, 0.27, 0.025])
    cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_label("Hot17 − Cold17 TP (mm day$^{-1}$)")
    fig.text(
        0.08,
        0.015,
        "Vertical dashed lines: 114E and 119E\n"
        "Y-label colors: orange = 14–17 Jun; blue = 18–24 Jun",
        fontsize=8.5,
        va="bottom",
    )
    fig.suptitle(
        "Hovmöller propagation diagnosis for the 12 June initialization (Total grouping)",
        fontsize=13,
        y=0.965,
    )
    _save_fig(fig, "figure8_hovmoller_main_mswep_s2s_totalgroup_init2023-06-12_0614_0624")


def plot_figureS_appendix_s2s_0608_0614_0624(
    hov_0608: Dict[str, Dict[str, xr.DataArray]],
    abs_levels: np.ndarray,
    diff_levels: np.ndarray,
    diff_norm: mcolors.TwoSlopeNorm,
):
    dates = pd.to_datetime(hov_0608["Hot17"]["tp"].bjt_date.values)
    fig, axs = plt.subplots(3, 1, figsize=(11.0, 9.2), sharex=True)
    cf_abs = _plot_hov_abs_panel(axs[0], hov_0608["Hot17"]["tp"], hov_0608["Hot17"]["z500"], dates, abs_levels, "(a) Total Hot17", "Total Hot17")
    _plot_hov_abs_panel(axs[1], hov_0608["Cold17"]["tp"], hov_0608["Cold17"]["z500"], dates, abs_levels, "(b) Total Cold17", "Total Cold17")
    cf_diff = _plot_hov_diff_panel(
        axs[2],
        hov_0608["Hot17_minus_Cold17"]["tp"],
        hov_0608["Hot17_minus_Cold17"]["z500"],
        dates,
        diff_levels,
        diff_norm,
        "(c) Total Hot17 − Cold17",
        "Total Hot17 − Cold17",
        show_xlabel=True,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.20, hspace=0.35)
    cax_abs = fig.add_axes([0.18, 0.105, 0.34, 0.025])
    cbar_abs = fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal")
    cbar_abs.set_label("TP (mm day$^{-1}$)")
    cax_diff = fig.add_axes([0.58, 0.105, 0.28, 0.025])
    cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal")
    cbar_diff.set_label("Hot17 − Cold17 TP (mm day$^{-1}$)")
    fig.text(0.12, 0.045, "Vertical dashed lines: 114E and 119E", fontsize=8.5)
    fig.text(0.12, 0.025, "Y-label colors: orange = 14–17 Jun; blue = 18–24 Jun", fontsize=8.5)
    fig.suptitle("Hovmöller propagation diagnosis for the 8 June initialization (Total grouping)", fontsize=13, y=0.965)
    _save_fig(fig, "figureS_hovmoller_appendix_s2s_totalgroup_init2023-06-08_0614_0624")


def plot_figureA_obs(
    era5_z500: xr.DataArray,
    era5_tp: xr.DataArray,
    mswep_tp: xr.DataArray,
    window_name: str,
    window_label: str,
    stage_split_labels: bool = False,
):
    dates = pd.to_datetime(era5_z500.bjt_date.values)
    levels = _abs_tp_levels([era5_tp, mswep_tp])
    fig, axs = plt.subplots(2, 1, figsize=_window_figsize((11, 7.0), dates, 8.2), sharex=True)
    panels = [("ERA5 TP", era5_tp), ("MSWEP TP", mswep_tp)]
    cf = None
    for i, (label, tp) in enumerate(panels):
        ax = axs[i]
        _, y_pad, tp_pad = _pad_hov_for_contourf(tp.values, dates)
        cf = ax.contourf(tp.longitude.values, y_pad, tp_pad, levels=levels, cmap="Blues", extend="max")
        _contour_z500_abs(ax, era5_z500, dates)
        _format_hov_axis(
            ax,
            f"{label}\nBJT date",
            show_xlabel=(i == 1),
            dates=dates,
            stage_split_labels=stage_split_labels,
        )
        ax.set_title(f"{label} shading + ERA5 Z500 contours")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.20, hspace=0.34)
    cax = fig.add_axes([0.26, 0.095, 0.52, 0.028])
    cbar = fig.colorbar(cf, cax=cax, orientation="horizontal")
    cbar.set_label("Precipitation (mm day$^{-1}$)")
    fig.text(0.13, 0.050, "Vertical dashed lines: 114E and 119E", fontsize=9)
    if stage_split_labels:
        fig.text(0.13, 0.030, "Y-label colors: orange = 14–17 Jun; blue = 18–24 Jun", fontsize=9)
    fig.suptitle(f"Observed/reanalysis Hovmöller: TP shading and ERA5 Z500 contours | {window_label}", fontsize=13)
    _save_fig(fig, f"figureA_hovmoller_obs_era5z500_precip_{window_name}")


def plot_figureB_s2s(
    init_date: str,
    hov: Dict[str, Dict[str, xr.DataArray]],
    window_name: str,
    window_label: str,
    stage_split_labels: bool = False,
):
    dates = pd.to_datetime(hov["Hot17"]["tp"].bjt_date.values)
    abs_levels = _abs_tp_levels([hov[g]["tp"] for g in ["Hot17", "Middle17", "Cold17"]])
    diff_levels, diff_norm = _diff_tp_levels(hov["Hot17_minus_Cold17"]["tp"])
    fig, axs = plt.subplots(4, 1, figsize=_window_figsize((11, 10.0), dates, 12.5), sharex=True)
    cf_abs = None
    cf_diff = None
    for i, group in enumerate(GROUP_ORDER):
        ax = axs[i]
        tp = hov[group]["tp"]
        z500 = hov[group]["z500"]
        if group == "Hot17_minus_Cold17":
            _, y_pad, tp_pad = _pad_hov_for_contourf(tp.values, dates)
            cf_diff = ax.contourf(
                tp.longitude.values,
                y_pad,
                tp_pad,
                levels=diff_levels,
                cmap="RdBu_r",
                norm=diff_norm,
                extend="both",
            )
            _contour_z500_diff(ax, z500, dates)
        else:
            _, y_pad, tp_pad = _pad_hov_for_contourf(tp.values, dates)
            cf_abs = ax.contourf(tp.longitude.values, y_pad, tp_pad, levels=abs_levels, cmap="Blues", extend="max")
            _contour_z500_abs(ax, z500, dates)
        _format_hov_axis(
            ax,
            f"{GROUP_LABELS[group]}\nBJT date",
            show_xlabel=(i == 3),
            dates=dates,
            stage_split_labels=stage_split_labels,
        )
        ax.set_title(GROUP_LABELS[group])
    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.17, hspace=0.52)
    if cf_abs is not None:
        cax_abs = fig.add_axes([0.18, 0.095, 0.34, 0.025])
        cbar_abs = fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal")
        cbar_abs.set_label("TP (mm day$^{-1}$)")
    if cf_diff is not None:
        cax_diff = fig.add_axes([0.58, 0.095, 0.28, 0.025])
        cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal")
        cbar_diff.set_label("Hot17 − Cold17 TP (mm day$^{-1}$)")
    fig.text(0.14, 0.052, "Vertical dashed lines: 114E and 119E", fontsize=9)
    if stage_split_labels:
        fig.text(0.14, 0.032, "Y-label colors: orange = 14–17 Jun; blue = 18–24 Jun", fontsize=9)
    fig.suptitle(f"S2S Stage-II Hovmöller: TP shading and Z500 contours | init {init_date} | {window_label}", fontsize=13)
    _save_fig(fig, f"figureB_hovmoller_s2s_stageII_hotcold_{init_date}_{window_name}")

# ====================================================
# QC
# ====================================================

def build_obs_qc(
    era5_z500: xr.DataArray,
    era5_tp: xr.DataArray,
    mswep_tp: xr.DataArray,
    window_name: str,
    dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    rows = []
    for dataset, tp in [("ERA5_TP", era5_tp), ("MSWEP_TP", mswep_tp)]:
        for d in pd.to_datetime(dates):
            tp_day = tp.sel(bjt_date=d)
            z_day = era5_z500.sel(bjt_date=d)
            rows.append({
                "window": window_name,
                "dataset": dataset,
                "date_bjt": d.strftime("%Y-%m-%d"),
                "tp_latband_mean": float(tp_day.mean("longitude", skipna=True)),
                "tp_latband_max": float(tp_day.max(skipna=True)),
                "z500_latband_mean": float(z_day.mean("longitude", skipna=True)),
                "z500_latband_min": float(z_day.min(skipna=True)),
                "z500_latband_max": float(z_day.max(skipna=True)),
                "lon_min": LON_RANGE[0],
                "lon_max": LON_RANGE[1],
                "lat_min": LAT_BAND[0],
                "lat_max": LAT_BAND[1],
            })
    return pd.DataFrame(rows)


def build_s2s_qc(
    init_date: str,
    hov: Dict[str, Dict[str, xr.DataArray]],
    window_name: str,
    dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        members = hov[group].get("members")
        for d in pd.to_datetime(dates):
            tp_day = hov[group]["tp"].sel(bjt_date=d)
            z_day = hov[group]["z500"].sel(bjt_date=d)
            rows.append({
                "window": window_name,
                "init_date": init_date,
                "date_bjt": d.strftime("%Y-%m-%d"),
                "group": group,
                "n_members": len(members) if members is not None else np.nan,
                "members": ",".join(str(m) for m in members) if members is not None else "",
                "grouping_file": GROUP_FILE,
                "grouping_stage": "Total",
                "analysis_stage": "Stage-II diagnostic dates",
                "analysis_window": "2023-06-14 to 2023-06-24",
                "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
                "tp_latband_mean": float(tp_day.mean("longitude", skipna=True)),
                "tp_latband_max": float(tp_day.max(skipna=True)),
                "z500_latband_mean": float(z_day.mean("longitude", skipna=True)),
                "z500_latband_min": float(z_day.min(skipna=True)),
                "z500_latband_max": float(z_day.max(skipna=True)),
            })
    return pd.DataFrame(rows)

# ====================================================
# Main
# ====================================================

def run_hovmoller_all_windows(make_plots: bool = True):
    print("=== Loading Total-stage S2S group data ===")
    group_df = load_total_hotcold_groups(GROUP_FILE)
    audit = build_group_audit_table(group_df)
    audit_path = os.path.join(OUT_DIR, "hovmoller_stageII_hotcold_group_member_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(f"[Audit] saved: {audit_path}")

    all_obs_qc = []
    all_s2s_qc = []
    paper_windows = {}
    paper_hovs = {}
    for window_name, cfg in WINDOWS.items():
        dates = pd.to_datetime(cfg["dates"])
        window_label = str(cfg["label"])
        stage_split_labels = bool(cfg["stage_split_labels"])

        print(f"=== Loading Figure A observation/reanalysis data for window={window_name} ({window_label}) ===")
        era5_z500 = load_era5_z500_hov(dates)
        era5_tp = load_era5_tp_hov(dates)
        mswep_tp = load_mswep_hov(dates)
        for name, da in [("ERA5 Z500", era5_z500), ("ERA5 TP", era5_tp), ("MSWEP TP", mswep_tp)]:
            _check_target_dates(da, f"{name} {window_name}", dates)

        all_obs_qc.append(build_obs_qc(era5_z500, era5_tp, mswep_tp, window_name, dates))
        if window_name == "0614_0624":
            paper_windows[window_name] = {
                "era5_z500": era5_z500,
                "era5_tp": era5_tp,
                "mswep_tp": mswep_tp,
            }
        if make_plots:
            plot_figureA_obs(era5_z500, era5_tp, mswep_tp, window_name, window_label, stage_split_labels)

        print(f"=== Loading Total-stage S2S fields for window={window_name} ({window_label}) ===")
        if window_name == "0614_0624":
            paper_hovs[window_name] = {}
        for init_date in INIT_DATES:
            fields = build_daily_realtime_member_fields(init_date, dates)
            hov = build_group_hovmollers_for_init(init_date, group_df, fields)
            if window_name == "0614_0624":
                paper_hovs[window_name][init_date] = hov
            all_s2s_qc.append(build_s2s_qc(init_date, hov, window_name, dates))
            if make_plots:
                plot_figureB_s2s(init_date, hov, window_name, window_label, stage_split_labels)

    if make_plots and "0614_0624" in paper_windows:
        full_obs = paper_windows["0614_0624"]
        full_hovs = paper_hovs.get("0614_0624", {})
        if "2023-06-08" in full_hovs and "2023-06-12" in full_hovs:
            abs_arrays = [full_obs["mswep_tp"]]
            for init_date in ("2023-06-08", "2023-06-12"):
                abs_arrays.extend(full_hovs[init_date][group]["tp"] for group in ["Hot17", "Cold17"])
            abs_levels = _abs_tp_levels(abs_arrays)
            diff_levels, diff_norm = _diff_tp_levels_from_arrays([
                full_hovs["2023-06-08"]["Hot17_minus_Cold17"]["tp"],
                full_hovs["2023-06-12"]["Hot17_minus_Cold17"]["tp"],
            ])
            plot_figure8_main_obs_s2s_0614_0624(
                full_obs["era5_z500"],
                full_obs["era5_tp"],
                full_obs["mswep_tp"],
                full_hovs["2023-06-12"],
                abs_levels,
                diff_levels,
                diff_norm,
            )
            plot_figureS_appendix_s2s_0608_0614_0624(
                full_hovs["2023-06-08"],
                abs_levels,
                diff_levels,
                diff_norm,
            )

    obs_qc_path = os.path.join(OUT_DIR, "hovmoller_obs_qc.csv")
    pd.concat(all_obs_qc, ignore_index=True).to_csv(obs_qc_path, index=False)
    print(f"[QC] saved: {obs_qc_path}")

    s2s_qc_path = os.path.join(OUT_DIR, "hovmoller_s2s_stageII_hotcold_qc.csv")
    pd.concat(all_s2s_qc, ignore_index=True).to_csv(s2s_qc_path, index=False)
    print(f"[QC] saved: {s2s_qc_path}")


def run_hovmoller_tp_z500_obs_s2s_stageII(make_plots: bool = True):
    run_hovmoller_all_windows(make_plots=make_plots)


# ====================================================
# Plot-ready cache export (Fig.8)
# ====================================================

CACHE_ROOT_FIG8 = "/data1/huangy/fig6/NC/v12/paper_figures/cache"
FIG8_CACHE_PATH = os.path.join(CACHE_ROOT_FIG8, "fig8_hovmoller_plot_cache.nc")
FIG8_CACHE_QC_PATH = os.path.join(CACHE_ROOT_FIG8, "fig8_hovmoller_cache_qc.csv")
FIG8_SIGNIFICANCE_ROOT = "/data1/huangy/fig6/NC/v12/paper_figures/significance"
FIG8_TP_SIGNIFICANCE_PATH = os.path.join(FIG8_SIGNIFICANCE_ROOT, "fig8_hovmoller_significance_tp.nc")
FIG8_Z500_SIGNIFICANCE_PATH = os.path.join(FIG8_SIGNIFICANCE_ROOT, "fig8_hovmoller_significance_z500.nc")
FIG8_SIGNIFICANCE_QC_PATH = os.path.join(FIG8_SIGNIFICANCE_ROOT, "fig8_hovmoller_significance_qc.csv")
FIG8_SIGNIFICANCE_AUDIT_PATH = os.path.join(FIG8_SIGNIFICANCE_ROOT, "fig8_hovmoller_significance_audit.md")


def _cache_hov_da(da: xr.DataArray, source: str, units: str, description: str) -> xr.DataArray:
    lon_dim = f"{source}_lon"
    out = da
    if "longitude" in out.dims or "longitude" in out.coords:
        if lon_dim in out.dims or lon_dim in out.coords:
            raise ValueError(
                f"Fig.8 cache Hovmöller field contains both 'longitude' and {lon_dim!r}."
            )
        out = out.rename({"longitude": lon_dim})

    input_time_dim = "time" if "time" in out.dims else None
    if "bjt_date" in out.dims:
        input_time_dim = "bjt_date"
        if "time" in out.dims:
            raise ValueError(
                "Fig.8 cache Hovmöller field contains both 'bjt_date' and "
                "'time' dimensions."
            )
        before = pd.to_datetime(out["bjt_date"].values)
        if "time" in out.coords:
            # A scalar GRIB initialization coordinate must not block the cache's
            # daily-time dimension name at this serialization boundary.
            out = out.drop_vars("time")
        out = out.rename({"bjt_date": "time"})
        after = pd.to_datetime(out["time"].values)
        if not np.array_equal(before.values, after.values):
            raise ValueError(
                "Fig.8 cache bjt_date-to-time normalization changed coordinate values."
            )

    missing_dims = [dim for dim in ("time", lon_dim) if dim not in out.dims]
    if missing_dims:
        raise ValueError(
            f"Fig.8 cache Hovmöller field is missing required dims {missing_dims}; "
            f"got dims={out.dims}, sizes={dict(out.sizes)}"
        )
    out = out.transpose("time", lon_dim)
    print(
        f"[Fig8 cache dims] source={source} input_time_dim={input_time_dim} "
        f"output_dims={out.dims} time_count={out.sizes['time']}"
    )
    out.attrs.update({"units": units, "description": description})
    return out


def _cache_qc_rows_hov(ds: xr.Dataset, cache_file: str) -> List[dict]:
    rows = []
    for name, da in ds.data_vars.items():
        vals = da.values
        finite_fraction = float(np.isfinite(vals).sum() / vals.size) if vals.size else np.nan
        rows.append({
            "figure": "Fig.8",
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


def _validate_fig8_s2s_cache_density(ds: xr.Dataset) -> None:
    for name in ["tp_hot_hov", "z500_hot_hov", "tp_cold_hov", "z500_cold_hov", "tp_diff_hov", "z500_diff_hov"]:
        vals = ds[name].values
        finite_fraction = float(np.isfinite(vals).sum() / vals.size) if vals.size else 0.0
        if finite_fraction < 0.8:
            raise ValueError(
                f"Fig.8 S2S cache field {name} finite_fraction={finite_fraction:.3f} < 0.8; "
                "possible coordinate-union alignment occurred"
            )


def export_fig8_plot_cache(
    cache_path: str = FIG8_CACHE_PATH,
    qc_path: str = FIG8_CACHE_QC_PATH,
    group_df: Optional[pd.DataFrame] = None,
    fields_0612: Optional[Dict[str, xr.DataArray]] = None,
    fields_0608: Optional[Dict[str, xr.DataArray]] = None,
    member_hovs_0612: Optional[Dict[str, xr.DataArray]] = None,
    init_date: str = "2023-06-12",
) -> str:
    """Export plot-ready Fig.8 Hovmöller fields without changing science logic."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    dates = pd.to_datetime(WINDOWS["0614_0624"]["dates"])
    if group_df is None:
        group_df = load_total_hotcold_groups(GROUP_FILE)
    era5_z500 = load_era5_z500_hov(dates)
    mswep_tp = load_mswep_hov(dates)
    if fields_0612 is None:
        fields_0612 = build_daily_realtime_member_fields("2023-06-12", dates)
    hov = build_group_hovmollers_for_init(
        "2023-06-12", group_df, fields_0612, member_hovs=member_hovs_0612
    )
    if fields_0608 is None:
        fields_0608 = build_daily_realtime_member_fields("2023-06-08", dates)
    hov_0608 = build_group_hovmollers_for_init("2023-06-08", group_df, fields_0608)

    tp_abs_levels = _abs_tp_levels([
        mswep_tp,
        hov_0608["Hot17"]["tp"],
        hov_0608["Cold17"]["tp"],
        hov["Hot17"]["tp"],
        hov["Cold17"]["tp"],
    ])
    tp_diff_levels, _ = _diff_tp_levels_from_arrays([
        hov_0608["Hot17_minus_Cold17"]["tp"],
        hov["Hot17_minus_Cold17"]["tp"],
    ])

    if init_date not in INIT_DATES:
        raise ValueError("Hovmoller cache supports June 8 and June 12 only")
    if init_date == "2023-06-08":
        hov = hov_0608
    hot_members = get_hotcold_members(group_df, init_date, "Hot17")
    cold_members = get_hotcold_members(group_df, init_date, "Cold17")
    ds = xr.Dataset({
        "tp_obs_hov": _cache_hov_da(mswep_tp, "mswep", "mm day^-1", "MSWEP precipitation Hovmöller latitude-band mean"),
        "z500_obs_hov": _cache_hov_da(era5_z500, "era5", "gpm", "ERA5 absolute Z500 Hovmöller latitude-band mean"),
        "tp_hot_hov": _cache_hov_da(hov["Hot17"]["tp"], "s2s", "mm day^-1", "S2S Total Hot17 precipitation Hovmöller"),
        "z500_hot_hov": _cache_hov_da(hov["Hot17"]["z500"], "s2s", "gpm", "S2S Total Hot17 absolute Z500 Hovmöller"),
        "tp_cold_hov": _cache_hov_da(hov["Cold17"]["tp"], "s2s", "mm day^-1", "S2S Total Cold17 precipitation Hovmöller"),
        "z500_cold_hov": _cache_hov_da(hov["Cold17"]["z500"], "s2s", "gpm", "S2S Total Cold17 absolute Z500 Hovmöller"),
        "tp_diff_hov": _cache_hov_da(hov["Hot17_minus_Cold17"]["tp"], "s2s", "mm day^-1", "Hot17 minus Cold17 precipitation Hovmöller difference"),
        "z500_diff_hov": _cache_hov_da(hov["Hot17_minus_Cold17"]["z500"], "s2s", "gpm", "Hot17 minus Cold17 Z500 Hovmöller difference"),
        "tp_abs_levels": xr.DataArray(
            tp_abs_levels,
            dims=("tp_abs_level",),
            attrs={
                "units": "mm day^-1",
                "description": "Fig.8 absolute TP colorbar levels computed from MSWEP obs plus 2023-06-08/2023-06-12 Hot17/Cold17 Hovmöller fields",
            },
        ),
        "tp_diff_levels": xr.DataArray(
            tp_diff_levels,
            dims=("tp_diff_level",),
            attrs={
                "units": "mm day^-1",
                "description": "Fig.8 Hot17-minus-Cold17 TP difference colorbar levels computed from 2023-06-08 and 2023-06-12 differences",
            },
        ),
    })
    ds.attrs.update({
        "source_figure": "Fig.7 Hovmöller propagation diagnosis",
        "source_script": "cell_hovmoller_tp_z500_obs_s2s_stageII.py",
        "analysis_window": "2023-06-14 to 2023-06-24",
        "grouping_file": GROUP_FILE,
        "grouping_stage": "Total",
        "sign_convention": "Hot17_minus_Cold17 = Hot17 - Cold17",
        "z500_type": "absolute",
        "z500_diff_type": "Hot17_minus_Cold17 difference of absolute Z500",
        "z500_units": "gpm",
        "tp_units": "mm day^-1",
        "map_extent": str(LON_RANGE),
        "nchn_box": "114E-119E vertical reference lines; lat band 35N-44N",
        "hovmoller_lat_band": str(LAT_BAND),
        "vertical_reference_lines": "114E, 119E",
        "main_init_date": init_date,
        "initialization": init_date,
        "hot_members": ",".join(map(str, hot_members)),
        "cold_members": ",".join(map(str, cold_members)),
        "s2s_hovmoller_calculation_order": "member spatial field -> member-level cosine-latitude-weighted Hovmoller -> group mean and Hot17-minus-Cold17",
    })
    _validate_fig8_s2s_cache_density(ds)
    ds.to_netcdf(cache_path)
    pd.DataFrame(_cache_qc_rows_hov(ds, cache_path)).assign(initialization=init_date, grouping_stage="Total", analysis_window="2023-06-14 to 2023-06-24").to_csv(qc_path, index=False)
    print(f"[Fig8 cache] saved: {cache_path}")
    print(f"[Fig8 cache QC] saved: {qc_path}")
    return cache_path


# ====================================================
# Member-level Hovmöller significance export (Fig.8)
# ====================================================

def _fig8_member_hov(da: xr.DataArray) -> xr.DataArray:
    """Apply the existing cosine-latitude mean separately to each member."""
    sub = _sel_region(da, (LON_RANGE[0], LON_RANGE[1], LAT_BAND[0], LAT_BAND[1]))
    weights = np.cos(np.deg2rad(sub.latitude))
    return sub.weighted(weights).mean("latitude", skipna=True).transpose(
        "member", "bjt_date", "longitude"
    )


def _fig8_welch_hov(
    member_hov: xr.DataArray,
    hot_members: Sequence[int],
    cold_members: Sequence[int],
    expected_difference: xr.DataArray,
    variable: str,
):
    hot = member_hov.sel(member=list(hot_members))
    cold = member_hov.sel(member=list(cold_members))
    hot_values = hot.values
    cold_values = cold.values
    n_hot = np.isfinite(hot_values).sum(axis=0)
    n_cold = np.isfinite(cold_values).sum(axis=0)

    alternative_explicit = True
    try:
        result = ttest_ind(
            hot_values, cold_values, axis=0, equal_var=False,
            nan_policy="omit", alternative="two-sided",
        )
    except TypeError:
        alternative_explicit = False
        result = ttest_ind(
            hot_values, cold_values, axis=0, equal_var=False,
            nan_policy="omit",
        )

    t_statistic = np.array(result.statistic, dtype=float, copy=True)
    p_value = np.array(result.pvalue, dtype=float, copy=True)
    if getattr(result, "df", None) is None:
        degrees_of_freedom = np.full(p_value.shape, np.nan, dtype=float)
    else:
        degrees_of_freedom = np.array(result.df, dtype=float, copy=True)

    insufficient = (n_hot < 2) | (n_cold < 2)
    invalid_test = insufficient | ~np.isfinite(t_statistic) | ~np.isfinite(p_value)
    t_statistic[invalid_test] = np.nan
    p_value[invalid_test] = np.nan
    degrees_of_freedom[invalid_test] = np.nan

    hot_mean_da = hot.mean("member", skipna=True).transpose("bjt_date", "longitude")
    cold_mean_da = cold.mean("member", skipna=True).transpose("bjt_date", "longitude")
    difference_da = hot_mean_da - cold_mean_da
    expected = expected_difference.transpose("bjt_date", "longitude")
    delta = np.abs(difference_da.values - expected.values)
    max_mismatch = float(np.nanmax(delta)) if np.isfinite(delta).any() else np.nan
    if not np.allclose(
        difference_da.values, expected.values,
        rtol=1e-7, atol=1e-6, equal_nan=True,
    ):
        raise ValueError(
            f"Fig.8 {variable} member-Hov Hot17-minus-Cold17 mismatch: "
            f"max_absolute_mismatch={max_mismatch}"
        )

    dates = pd.to_datetime(member_hov.bjt_date.values)
    lon = np.asarray(member_hov.longitude.values)
    valid_test_mask = np.isfinite(p_value)
    local_significant = valid_test_mask & (p_value < 0.05)

    coords = {"time": dates.values, "s2s_lon": lon}
    dims = ("time", "s2s_lon")
    dataset = xr.Dataset(
        {
            "hot_mean": xr.DataArray(hot_mean_da.values, dims=dims, coords=coords),
            "cold_mean": xr.DataArray(cold_mean_da.values, dims=dims, coords=coords),
            "hot_minus_cold": xr.DataArray(difference_da.values, dims=dims, coords=coords),
            "t_statistic": xr.DataArray(t_statistic, dims=dims, coords=coords),
            "p_value": xr.DataArray(p_value, dims=dims, coords=coords),
            "significant_local_p005": xr.DataArray(local_significant.astype(np.int8), dims=dims, coords=coords),
            "valid_test_mask": xr.DataArray(valid_test_mask.astype(np.int8), dims=dims, coords=coords),
            "n_hot": xr.DataArray(n_hot, dims=dims, coords=coords),
            "n_cold": xr.DataArray(n_cold, dims=dims, coords=coords),
            "degrees_of_freedom": xr.DataArray(degrees_of_freedom, dims=dims, coords=coords),
        }
    )
    units = "mm day^-1" if variable == "tp" else "gpm"
    dataset.attrs.update(
        {
            "test": "two-sided Welch independent-samples t test",
            "sample_dimension": "member",
            "equal_var": "False",
            "nan_policy": "omit",
            "difference_direction": "Hot17 minus Cold17",
            "main_figure_significance": "uncorrected local p < 0.05",
            "expected_hot_members": 17,
            "expected_cold_members": 17,
            "grouping_file": GROUP_FILE,
            "grouping_stage": "Total",
            "initialization": "2023-06-12",
            "latitude_range": "35-44N",
            "latitude_average_method": "cosine-latitude-weighted mean",
            "time_range": "2023-06-14 through 2023-06-24 BJT",
            "longitude_range": "90-135E",
            "units": units,
            "interpretation": "conditional difference between predefined ensemble-member groups",
        }
    )

    finite_p = p_value[np.isfinite(p_value)]
    valid_count = int(valid_test_mask.sum())
    qc = {
        "variable": variable,
        "initialization": "2023-06-12",
        "hot_member_ids": ",".join(map(str, hot_members)),
        "cold_member_ids": ",".join(map(str, cold_members)),
        "n_hot_expected": 17,
        "n_cold_expected": 17,
        "hot_cold_overlap": ",".join(map(str, sorted(set(hot_members) & set(cold_members)))),
        "time_range": "2023-06-14 to 2023-06-24 BJT",
        "longitude_range": "90-135E",
        "latitude_range": "35-44N",
        "latitude_average_method": "cosine-latitude-weighted mean",
        "total_hov_cells": int(p_value.size),
        "valid_test_cells": valid_count,
        "non_17_vs_17_cells": int(((n_hot != 17) | (n_cold != 17)).sum()),
        "invalid_ttest_cells_with_n_ge_2": int((invalid_test & ~insufficient).sum()),
        "local_p005_count": int(local_significant.sum()),
        "local_p005_fraction": float(local_significant.sum() / valid_count) if valid_count else np.nan,
        "p_min": float(finite_p.min()) if finite_p.size else np.nan,
        "p_max": float(finite_p.max()) if finite_p.size else np.nan,
        "max_absolute_mismatch": max_mismatch,
        "rtol": 1e-7,
        "atol": 1e-6,
        "all_nan_p": not bool(finite_p.size),
        "constant_finite_p": bool(finite_p.size and np.all(finite_p == finite_p[0])),
        "all_local_significant": bool(valid_count and local_significant.sum() == valid_count),
        "alternative_two_sided_explicit": alternative_explicit,
        "scipy_version": scipy.__version__,
        "tp_z500_masks_independently_generated": True,
    }
    return dataset, qc


def _fig8_write_significance_audit(qc_rows: Sequence[dict]) -> None:
    rows = {row["variable"]: row for row in qc_rows}
    text = f"""# Fig.7 Hovmöller spatial-significance runtime audit

- Source script: `cell_hovmoller_tp_z500_obs_s2s_stageII.py`
- Initialization: 2023-06-12
- Grouping source: `{GROUP_FILE}`
- Grouping stage: Total; Hot17/Cold17 are fixed predefined groups; Middle17 is excluded.
- Hot17 IDs: {rows['tp']['hot_member_ids']}
- Cold17 IDs: {rows['tp']['cold_member_ids']}
- Group counts: 17 Hot and 17 Cold; overlap: {rows['tp']['hot_cold_overlap'] or 'none'}
- Canonical calculation order: member daily latitude-longitude field -> member-level 35-44N cosine-latitude-weighted Hovmöller -> group means / Hot17-Cold17 -> Welch test.
- Welch sample dimension: member only; time and longitude remain diagnostic coordinates.
- Time range: 2023-06-14 through 2023-06-24 BJT; longitude range: 90-135E.
- Test: two-sided Welch independent-samples t test (`axis=0`, `equal_var=False`, `nan_policy=omit`).
- Main-figure criterion: uncorrected local Welch p < 0.05.
- TP valid/local cells: {rows['tp']['valid_test_cells']} / {rows['tp']['local_p005_count']}.
- Z500 valid/local cells: {rows['z500']['valid_test_cells']} / {rows['z500']['local_p005_count']}.
- Composite max mismatch (TP/Z500): {rows['tp']['max_absolute_mismatch']} / {rows['z500']['max_absolute_mismatch']} (`rtol=1e-7`, `atol=1e-6`).
- Outputs: `{FIG8_TP_SIGNIFICANCE_PATH}`, `{FIG8_Z500_SIGNIFICANCE_PATH}`, `{FIG8_SIGNIFICANCE_QC_PATH}`.

The tests quantify conditional differences between the predefined Hot17 and Cold17 ensemble-member groups at each time-longitude cell. They do not establish causality or constitute independent observational population inference.

## Caption supplement

Stippling indicates time-longitude cells where the Hot17-Cold17 precipitation difference is locally significant at p < 0.05 based on a two-sided Welch's t test. Ordinary labeled Z500 difference contours are retained without displaying Z500 significance. Independent Z500 test results remain archived for traceability. Tests were conducted separately for precipitation and Z500 using the 17 members in each predefined group at each time-longitude cell.
"""
    with open(FIG8_SIGNIFICANCE_AUDIT_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)


def export_fig8_hovmoller_significance(
    group_df: Optional[pd.DataFrame] = None,
    fields: Optional[Dict[str, xr.DataArray]] = None,
    member_hovs: Optional[Dict[str, xr.DataArray]] = None,
    init_date: str = "2023-06-12",
    tp_path: str = FIG8_TP_SIGNIFICANCE_PATH,
    z500_path: str = FIG8_Z500_SIGNIFICANCE_PATH,
    qc_path: str = FIG8_SIGNIFICANCE_QC_PATH,
) -> Tuple[str, str]:
    """Export member-level Fig.8 local-Welch Hovmöller diagnostics."""
    os.makedirs(FIG8_SIGNIFICANCE_ROOT, exist_ok=True)
    dates = pd.to_datetime(WINDOWS["0614_0624"]["dates"])
    if group_df is None:
        group_df = load_total_hotcold_groups(GROUP_FILE)
    hot_members = get_hotcold_members(group_df, init_date, "Hot17")
    cold_members = get_hotcold_members(group_df, init_date, "Cold17")
    overlap = set(hot_members) & set(cold_members)
    if len(hot_members) != 17 or len(cold_members) != 17 or overlap:
        raise ValueError(
            f"Invalid Fig.8 groups: Hot={len(hot_members)}, Cold={len(cold_members)}, "
            f"overlap={sorted(overlap)}"
        )

    if fields is None:
        fields = build_daily_realtime_member_fields(init_date, dates)
    if member_hovs is None:
        member_hovs = {
            "tp": _fig8_member_hov(fields["tp"]),
            "z500": _fig8_member_hov(fields["z500"]),
        }
    existing = build_group_hovmollers_for_init(
        init_date, group_df, fields, member_hovs=member_hovs
    )
    outputs = {}
    qc_rows = []
    for variable, path in (
        ("tp", tp_path),
        ("z500", z500_path),
    ):
        member_hov = member_hovs[variable]
        ds, qc = _fig8_welch_hov(
            member_hov,
            hot_members,
            cold_members,
            existing["Hot17_minus_Cold17"][variable],
            variable,
        )
        ds.attrs.update({"initialization": init_date, "hot_members": ",".join(map(str, hot_members)), "cold_members": ",".join(map(str, cold_members)), "variable": variable})
        qc["initialization"] = init_date
        ds.to_netcdf(path)
        outputs[variable] = path
        qc_rows.append(qc)
        print(f"[Fig8 significance] saved {variable}: {path}")

    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    # June 8 must not overwrite the historical June 12 audit.
    if init_date == "2023-06-12":
        _fig8_write_significance_audit(qc_rows)
    print(f"[Fig8 significance QC] saved: {qc_path}")
    if init_date == "2023-06-12":
        print(f"[Fig8 significance audit] saved: {FIG8_SIGNIFICANCE_AUDIT_PATH}")
    return outputs["tp"], outputs["z500"]

FIG8_0608_CACHE_PATH = os.path.join(CACHE_ROOT_FIG8, "fig8_hovmoller_init2023-06-08_plot_cache.nc")
FIG8_0608_CACHE_QC_PATH = os.path.join(CACHE_ROOT_FIG8, "fig8_hovmoller_init2023-06-08_cache_qc.csv")
FIG8_0608_TP_PATH = os.path.join(FIG8_SIGNIFICANCE_ROOT, "fig8_init2023-06-08_hovmoller_significance_tp.nc")
FIG8_0608_Z500_PATH = os.path.join(FIG8_SIGNIFICANCE_ROOT, "fig8_init2023-06-08_hovmoller_significance_z500.nc")
FIG8_0608_SIG_QC_PATH = os.path.join(FIG8_SIGNIFICANCE_ROOT, "fig8_init2023-06-08_hovmoller_significance_qc.csv")

def export_fig8_20230608_plot_cache():
    return export_fig8_plot_cache(FIG8_0608_CACHE_PATH, FIG8_0608_CACHE_QC_PATH, init_date="2023-06-08")

def export_fig8_20230608_hovmoller_significance():
    return export_fig8_hovmoller_significance(init_date="2023-06-08", tp_path=FIG8_0608_TP_PATH, z500_path=FIG8_0608_Z500_PATH, qc_path=FIG8_0608_SIG_QC_PATH)

# Loading this complete cell defines functions; exports are explicit.
