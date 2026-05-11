# %% [markdown]
# # Anomaly evolution module (two notebook cells)
#
# Cell 1 computes S2S anomaly fields and summary diagnostics.  Cell 2, run only
# after the summary has been checked, draws Figures A-I.  Both cells use one
# consistent processing pathway: CF+PF merge -> valid time from init+step -> BJT
# valid date -> daily aggregation -> stage mean -> hindcast climatology -> anomaly.

# %%
# =============================================================================
# Cell 1: Compute S2S ensemble-mean anomaly fields for two init dates and stages
# =============================================================================
import glob
import os
import warnings
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# 0. User configuration
# -----------------------------------------------------------------------------
INIT_DATES = ["2023-06-08", "2023-06-12"]
STAGES = {
    "Stage-I": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17")),
    "Stage-II": (pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24")),
    "Total": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-24")),
}

# All stage windows are BJT valid dates, not raw UTC step dates.
BJT_OFFSET = pd.Timedelta(hours=8)
MAP_EXTENT = (60.0, 150.0, 0.0, 60.0)  # lon_min, lon_max, lat_min, lat_max
NCHN_BOX = (114.0, 119.0, 35.0, 44.0)  # lon_min, lon_max, lat_min, lat_max
WNPSH_BOX = (115.0, 150.0, 15.0, 25.0)
G0 = 9.80665
SECONDS_PER_DAY = 86400.0
EF_DENOM_THRESHOLD = 5.0  # W m^-2; mask EF where LHF_up + SHF_up is too small.

BASE_DATA_DIR = "/data1/huangy/fig6/NC"
CF_DIR = f"{BASE_DATA_DIR}/cf_2023-6"
HINDCAST_DIRS = {
    "2023-06-08": "/data1/huangy/fig6/NC/pl_08.6.23",
    "2023-06-12": "/data1/huangy/fig6/NC/pl_12.6.23",
}


def _hindcast_dir_for_init(init_date: str) -> str:
    key = pd.Timestamp(init_date).strftime("%Y-%m-%d")
    if key not in HINDCAST_DIRS:
        raise KeyError(f"No hindcast directory configured for init_date={key}")
    return HINDCAST_DIRS[key]


print("[CHECK HINDCAST_DIRS]")
for _init in INIT_DATES:
    _hdir = _hindcast_dir_for_init(_init)
    print(f"  init={_init} | dir={_hdir} | exists={os.path.exists(_hdir)}")
    if not os.path.exists(_hdir):
        raise FileNotFoundError(f"Hindcast directory does not exist for {_init}: {_hdir}")

    print(f"  [CHECK HINDCAST files for {_init}]")
    for p in sorted(Path(_hdir).glob(f"*{_init}*"))[:30]:
        print("    ", p.name)

OUT_DIR = "/data1/huangy/fig6/NC/v5/factor_verification/anomaly_evolution"
os.makedirs(OUT_DIR, exist_ok=True)

HINDCAST_CATALOG = {
    "cf": {
        "pl_500": ["{hindcast_dir}/ecmf_hindcast_cf_pl_500_{init_date}.grib"],
        "pl_850": ["{hindcast_dir}/ecmf_hindcast_cf_pl_850_{init_date}.grib"],
        "pl_zqt": [
            "{hindcast_dir}/ecmf_hindcast_cf_pl_850_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_cf_pl_500_{init_date}.grib",
        ],
        "pl_uv": ["{hindcast_dir}/ecmf_hindcast_cf_pl_850_{init_date}.grib"],
        "sfc_tp": [
            "{hindcast_dir}/ecmf_hindcast_cf_sfc_mx2t_and_tp_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_cf_sfc_tp_{init_date}.grib",
        ],
        "sfc_mx2t": [
            "{hindcast_dir}/ecmf_hindcast_cf_sfc_mx2t_and_tp_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_cf_sfc_mx2t_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_cf_sfc_mx2t6_{init_date}.grib",
        ],
        "sfc_sm20": ["{hindcast_dir}/ecmf_hindcast_cf_sfc_sm20_{init_date}.grib"],
        "sfc_ssr": ["{hindcast_dir}/ecmf_hindcast_cf_sfc_fluxes_{init_date}.grib"],
        "sfc_lhf": ["{hindcast_dir}/ecmf_hindcast_cf_sfc_fluxes_{init_date}.grib"],
        "sfc_shf": ["{hindcast_dir}/ecmf_hindcast_cf_sfc_fluxes_{init_date}.grib"],
    },
    "pf": {
        "pl_500": ["{hindcast_dir}/ecmf_hindcast_pf_pl_500_{init_date}.grib"],
        "pl_850": ["{hindcast_dir}/ecmf_hindcast_pf_pl_850_{init_date}.grib"],
        "pl_zqt": [
            "{hindcast_dir}/ecmf_hindcast_pf_pl_850_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_pf_pl_500_{init_date}.grib",
        ],
        "pl_uv": ["{hindcast_dir}/ecmf_hindcast_pf_pl_850_{init_date}.grib"],
        "sfc_tp": [
            "{hindcast_dir}/ecmf_hindcast_pf_sfc_tp_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_pf_sfc_mx2t_and_tp_{init_date}.grib",
        ],
        "sfc_mx2t": [
            "{hindcast_dir}/ecmf_hindcast_pf_sfc_mx2t_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_pf_sfc_mx2t6_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_pf_sfc_mx2t_and_tp_{init_date}.grib",
        ],
        "sfc_sm20": ["{hindcast_dir}/ecmf_hindcast_pf_sfc_sm20_{init_date}.grib"],
        "sfc_ssr": ["{hindcast_dir}/ecmf_hindcast_pf_sfc_fluxes_{init_date}.grib"],
        "sfc_lhf": ["{hindcast_dir}/ecmf_hindcast_pf_sfc_fluxes_{init_date}.grib"],
        "sfc_shf": ["{hindcast_dir}/ecmf_hindcast_pf_sfc_fluxes_{init_date}.grib"],
    },
}

# Keep the catalog centralized.  Exact, verified realtime paths and exact
# hindcast filenames in per-init HINDCAST_DIRS are first-priority.  Broad fallback globs are
# retained only after these exact paths.
S2S_FILE_CATALOG = {
    "realtime": {
        "pf": {
            "pl_zqt": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"],
            "pl_500": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"],
            "pl_850": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"],
            "pl_uv": [
                f"{BASE_DATA_DIR}/MSE/ecmf_pf_131_132_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*pf*131*132*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*pf*uv*2023-06*.grb*",
            ],
            "sfc_tp": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_tp2023-06.grb"],
            "sfc_mx2t": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_mx2t6_2023-06.grb"],
            "sfc_sm20": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_sm2023-06.grb"],
            "sfc_ssr": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_radiation2023-06.grb"],
            "sfc_lhf": [f"{BASE_DATA_DIR}/MSE/ecmf_pf_slhf_2023-06.grib"],
            "sfc_shf": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_hflux2023-06.grb"],
        },
        "cf": {
            "pl_zqt": [
                f"{CF_DIR}/ecmf_cf_pl_850_2023-06.grib",
                f"{CF_DIR}/ecmf_cf_pl_500_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*rel*cf*pl*zqt*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*pl*zqt*2023-06*.grb*",
            ],
            "pl_500": [
                f"{CF_DIR}/ecmf_cf_pl_500_2023-06.grib",
                f"{CF_DIR}/ecmf_cf_pl500_156_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*rel*cf*pl*zqt*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*pl*zqt*2023-06*.grb*",
            ],
            "pl_850": [
                f"{CF_DIR}/ecmf_cf_pl_850_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*rel*cf*pl*zqt*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*pl*zqt*2023-06*.grb*",
            ],
            "pl_uv": [
                f"{CF_DIR}/ecmf_cf_pl_850_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*131*132*2023-06*.grb*",
                f"{BASE_DATA_DIR}/MSE/*cf*131*132*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*uv*2023-06*.grb*",
            ],
            "sfc_tp": [
                f"{CF_DIR}/ecmf_cf_sfc_tp_2023-6.grib",
                f"{CF_DIR}/ecmf_cf_sfc_tp_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*rel*cf*sfc*tp*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*tp*2023-06*.grb*",
            ],
            "sfc_mx2t": [
                f"{CF_DIR}/ecmf_cf_sfc_mx2t6_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*rel*cf*sfc*mx2t*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*mx2t*2023-06*.grb*",
            ],
            "sfc_sm20": [
                f"{CF_DIR}/ecmf_cf_sfc_sm20_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*rel*cf*sfc*sm*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*sm*2023-06*.grb*",
            ],
            "sfc_ssr": [
                f"{CF_DIR}/ecmf_cf_sfc_sshf+slhf+snsr_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*radiation*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*ssr*2023-06*.grb*",
            ],
            "sfc_lhf": [
                f"{CF_DIR}/ecmf_cf_sfc_sshf+slhf+snsr_2023-06.grib",
                f"{BASE_DATA_DIR}/MSE/*cf*slhf*2023-06*.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*slhf*2023-06*.grb*",
            ],
            "sfc_shf": [
                f"{CF_DIR}/ecmf_cf_sfc_sshf+slhf+snsr_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*hflux*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*cf*sfc*sshf*2023-06*.grb*",
            ],
        },
    },
    "hindcast": HINDCAST_CATALOG,
}

VAR_META = {
    "Z500": {"units": "gpm", "long_name": "500 hPa geopotential height anomaly"},
    "z850": {"units": "gpm", "long_name": "850 hPa geopotential height anomaly"},
    "u850": {"units": "m s-1", "long_name": "850 hPa zonal wind anomaly"},
    "v850": {"units": "m s-1", "long_name": "850 hPa meridional wind anomaly"},
    "qu850": {"units": "kg kg-1 m s-1", "long_name": "850 hPa zonal moisture flux anomaly"},
    "qv850": {"units": "kg kg-1 m s-1", "long_name": "850 hPa meridional moisture flux anomaly"},
    "tp": {"units": "mm day-1", "long_name": "stage-mean daily precipitation anomaly"},
    "mx2t": {"units": "degC", "long_name": "stage-mean daily maximum 2m temperature anomaly"},
    "sm20": {"units": "m3 m-3", "long_name": "soil moisture anomaly"},
    "SSR": {"units": "W m-2", "long_name": "surface solar radiation anomaly"},
    "LHF": {"units": "W m-2", "long_name": "latent heat flux anomaly, upward positive"},
    "SHF": {"units": "W m-2", "long_name": "sensible heat flux anomaly, upward positive"},
    "EF": {"units": "1", "long_name": "evaporative fraction anomaly"},
}


def _format_candidate_paths(patterns: Sequence[str], init_date: str) -> List[str]:
    init = pd.Timestamp(init_date)
    init_date_key = init.strftime("%Y-%m-%d")
    fields = {
        "init_date": init_date_key,
        "init_yyyymmdd": init.strftime("%Y%m%d"),
        "init_mmdd": init.strftime("%m%d"),
        "init_yyyy_mm": init.strftime("%Y-%m"),
        "hindcast_dir": _hindcast_dir_for_init(init_date_key),
    }
    return [pattern.format(**fields) for pattern in patterns]


def _resolve_patterns(patterns: Sequence[str], init_date: str) -> List[str]:
    paths: List[str] = []
    for candidate in _format_candidate_paths(patterns, init_date):
        # Exact templates and glob templates both pass through here; exact files
        # are first-priority because they are listed first in S2S_FILE_CATALOG.
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
    return da.rename(rename) if rename else da


def _sel_region(da: xr.DataArray, extent: Tuple[float, float, float, float] = MAP_EXTENT) -> xr.DataArray:
    da = _standardize_lat_lon(da)
    lon_min, lon_max, lat_min, lat_max = extent
    lat_slice = slice(lat_max, lat_min) if float(da.latitude.values[0]) > float(da.latitude.values[-1]) else slice(lat_min, lat_max)
    out = da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    if out.sizes.get("longitude", 0) == 0 and float(da.longitude.max()) > 180 and lon_min < 180:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")
        out = da.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))
    return out


def _member_dim(da: xr.DataArray) -> Optional[str]:
    for dim in ("member", "number", "realization", "ensemble", "perturbationNumber"):
        if dim in da.dims:
            return dim
    return None


def _sample_dims(da: xr.DataArray) -> List[str]:
    return [d for d in ("member", "number", "realization", "ensemble", "hc_init", "hindcast_time", "file_sample") if d in da.dims]


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


def get_s2s_init_slice(da: xr.DataArray, init_date: str, realtime: bool = True) -> xr.DataArray:
    """Select realtime init or same-MMDD hindcast inits, then normalize time dims."""
    coord = _init_coord_name(da)
    if coord is None:
        return da
    init = pd.Timestamp(init_date)
    times = pd.to_datetime(da[coord].values.ravel())
    if realtime:
        matches = [t for t in times if pd.Timestamp(t).date() == init.date()]
    else:
        matches = [t for t in times if pd.Timestamp(t).month == init.month and pd.Timestamp(t).day == init.day]
    if matches:
        da = da.sel({coord: matches})
    else:
        print(f"  [Info] no explicit init coordinate match for {init_date}; keeping all {coord} values")

    if realtime and coord in da.dims and da.sizes.get(coord) == 1:
        da = da.squeeze(coord, drop=True)
    elif (not realtime) and coord in da.dims and coord != "hc_init":
        da = da.rename({coord: "hc_init"})
    elif (not realtime) and coord in da.coords and coord not in da.dims:
        da = da.rename({coord: "hindcast_time"})
    return da


def _select_level(da: xr.DataArray, level_hpa: int) -> xr.DataArray:
    """
    Robustly select pressure level.

    Handles two cases:
    1. level coordinate is a real dimension, e.g. isobaricInhPa: [500, 850]
       -> select nearest level by isel/sel.
    2. level coordinate is scalar, e.g. isobaricInhPa=850 for an already single-level file
       -> verify it is close to requested level, then return da without calling .sel().
    """
    level_names = ("isobaricInhPa", "isobaricInPa", "pressure_level", "level", "plev")

    def _target_from_values(values, level_hpa):
        vals = np.asarray(values, dtype=float)
        # Values larger than 2000 are likely Pa rather than hPa.
        if vals.size and np.nanmax(np.abs(vals)) > 2000:
            return float(level_hpa) * 100.0
        return float(level_hpa)

    for lev in level_names:
        # Case 1: pressure level is a real dimension.
        if lev in da.dims:
            vals = np.asarray(da[lev].values, dtype=float)
            target = _target_from_values(vals, level_hpa)

            if vals.size == 1:
                nearest = float(vals.ravel()[0])
                selected = da.isel({lev: 0}, drop=True)
            else:
                idx = int(np.nanargmin(np.abs(vals - target)))
                nearest = float(vals[idx])
                selected = da.isel({lev: idx}, drop=True)

            tol = 100.0 if target > 2000 else 1.0
            if abs(nearest - target) > tol:
                warnings.warn(
                    f"[level select] Requested {level_hpa} hPa but nearest {lev}={nearest}; "
                    f"target={target}. Please check the pressure-level file.",
                    UserWarning,
                    stacklevel=2,
                )

            return selected.drop_vars(lev, errors="ignore")

        # Case 2: pressure level is only a scalar/non-index coordinate.
        if lev in da.coords:
            vals = np.asarray(da[lev].values, dtype=float)
            if vals.size == 1:
                actual = float(vals.ravel()[0])
                target = _target_from_values(vals, level_hpa)
                tol = 100.0 if target > 2000 else 1.0

                if abs(actual - target) > tol:
                    warnings.warn(
                        f"[level check] Field has scalar {lev}={actual}, "
                        f"but requested {level_hpa} hPa (target={target}). "
                        f"Returning the field unchanged; please verify file level.",
                        UserWarning,
                        stacklevel=2,
                    )

                # Drop scalar level coord to avoid later xr.align conflicts.
                return da.drop_vars(lev, errors="ignore")

            # Non-dimension but non-scalar coordinate: do not call .sel(), just warn.
            warnings.warn(
                f"[level select] {lev} is a coordinate but not a dimension, with shape {vals.shape}. "
                f"Cannot select by .sel(); returning unchanged.",
                UserWarning,
                stacklevel=2,
            )
            return da

    # No pressure-level coordinate found; assume the file is already the requested level.
    return da


def _build_bjt_valid_coord(da: xr.DataArray, init_date: str) -> xr.DataArray:
    # Explicitly use init time + step to construct UTC valid time, then shift to BJT.
    if "step" not in da.coords:
        if "valid_time" in da.coords:
            vt = pd.to_datetime(da.valid_time.values) + BJT_OFFSET
            time_dim = "valid_time" if "valid_time" in da.dims else da.valid_time.dims[-1]
            return da.assign_coords(bjt_valid_time=(time_dim, vt)).swap_dims({time_dim: "bjt_valid_time"})
        raise ValueError("S2S field has no step or valid_time coordinate")
    init = pd.Timestamp(init_date)
    bjt = pd.to_datetime([init + pd.Timedelta(s) + BJT_OFFSET for s in da.step.values])
    return da.assign_coords(bjt_valid_time=("step", bjt)).swap_dims({"step": "bjt_valid_time"})


def _to_gpm_if_needed(da: xr.DataArray, var_label: str) -> xr.DataArray:
    """Only Z500/z850 use this gpm conversion; flux variables never call it."""
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


def _to_celsius_if_needed(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    if units in ("k", "kelvin") or float(da.mean(skipna=True)) > 100:
        da = da - 273.15
    da.attrs["units"] = "degC"
    return da


def _tp_to_mm(da: xr.DataArray) -> xr.DataArray:
    units_raw = str(da.attrs.get("units", "")).strip()
    units = units_raw.lower().replace(" ", "")
    if units in ("m", "meter", "metre", "meters", "metres"):
        da = da * 1000.0
    elif units in ("kgm**-2", "kgm-2", "kg/m^2", "kgm^-2", "mm", "millimeter", "millimetre", "millimeters", "millimetres"):
        pass
    else:
        warnings.warn(f"[tp] Unknown precipitation units '{units_raw}'; not applying *1000 automatically.", UserWarning, stacklevel=2)
    da.attrs["units"] = "mm day-1"
    return da


def _sm20_to_volumetric(da: xr.DataArray) -> xr.DataArray:
    """Convert sm20 to volumetric soil moisture (m3 m-3) using units only."""
    units_raw = str(da.attrs.get("units", "")).strip()
    units = units_raw.lower().replace(" ", "").replace("^", "")
    da = da.copy()
    if units in ("m3m-3", "m3/m3", "m3m**-3", "dimensionless", "1"):
        conversion = f"kept original units '{units_raw}' as volumetric"
    elif units in ("kgm-3", "kg/m3", "kgm**-3"):
        da = da / 1000.0
        conversion = f"converted from '{units_raw}' to m3 m-3 by dividing by 1000"
    else:
        warnings.warn(
            f"[sm20] Unknown soil-moisture units '{units_raw}'; not scaling automatically.",
            UserWarning,
            stacklevel=2,
        )
        conversion = f"unknown units '{units_raw}'; no scaling applied"
    da.attrs["original_units"] = units_raw
    da.attrs["conversion_note"] = conversion
    da.attrs["units"] = "m3 m-3"
    print(f"  [sm20] {conversion}")
    return da


def _energy_to_wm2(da: xr.DataArray, upward_positive: bool = False) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    if "j" in units or "joule" in units:
        da = da / SECONDS_PER_DAY
    elif "w" not in units:
        warnings.warn(f"[flux] Unknown accumulated-flux units '{units}'; not dividing by 86400 automatically.", UserWarning, stacklevel=2)
    if upward_positive:
        # ECMWF accumulated surface latent/sensible heat fluxes are positive downward;
        # factor verification uses upward-positive LHF/SHF, so multiply by -1 here.
        da = -da
    da.attrs["units"] = "W m-2"
    return da


def process_s2s_bjt_daily_mean(da: xr.DataArray, init_date: str) -> xr.DataArray:
    """BJT daily mean, using valid date = init UTC + step + 8 h."""
    da = _build_bjt_valid_coord(da, init_date)
    return da.resample(bjt_valid_time="1D").mean().rename({"bjt_valid_time": "bjt_date"})


def process_s2s_bjt_daily_max(da: xr.DataArray, init_date: str) -> xr.DataArray:
    da = _build_bjt_valid_coord(da, init_date)
    return da.resample(bjt_valid_time="1D").max().rename({"bjt_valid_time": "bjt_date"})


def _interval_day_fractions(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, float]]:
    if end <= start:
        return []
    total = (end - start).total_seconds()
    pieces: List[Tuple[pd.Timestamp, float]] = []
    cursor = start
    while cursor < end:
        next_midnight = cursor.normalize() + pd.Timedelta(days=1)
        stop = min(end, next_midnight)
        pieces.append((cursor.normalize(), (stop - cursor).total_seconds() / total))
        cursor = stop
    return pieces


def _apportion_accum_to_calendar_days(da: xr.DataArray, init_date: str, force_non_negative: bool = False) -> xr.DataArray:
    """Split step-differenced accumulated increments across BJT natural days.

    This is the verified BJT day-boundary logic: each increment is assigned to
    the BJT calendar days it overlaps in proportion to overlap duration, rather
    than simply summing by the ending valid time.
    """
    if "step" not in da.coords:
        raise ValueError("Accumulated S2S field requires a step coordinate")
    init = pd.Timestamp(init_date)
    steps = [pd.Timedelta(s) for s in da.step.values]
    bjt_valid = [init + s + BJT_OFFSET for s in steps]
    inc = da.diff("step")
    if force_non_negative:
        # Precipitation accumulations can reset or carry tiny floating errors;
        # clip negative increments immediately after step differencing.
        inc = inc.where(inc >= 0.0, 0.0)
    inc = inc.assign_coords(step=da.step.values[1:])
    apportioned: List[xr.DataArray] = []
    for i in range(1, len(steps)):
        start = bjt_valid[i - 1]
        end = bjt_valid[i]
        inc_i = inc.isel(step=i - 1).drop_vars("step", errors="ignore")
        for day, frac in _interval_day_fractions(start, end):
            apportioned.append((inc_i * frac).expand_dims(bjt_date=[day]))
    if not apportioned:
        raise ValueError("No accumulated increments available for BJT day apportionment")
    return xr.concat(apportioned, dim="bjt_date").groupby("bjt_date").sum(skipna=True)


def process_s2s_bjt_daily_accum(da: xr.DataArray, init_date: str, output: str, upward_positive: bool = False, force_non_negative: bool = False) -> xr.DataArray:
    """Step-difference accumulated S2S fields and apportion to BJT days.

    Used for tp, SSR, LHF, and SHF.  Accumulated step values are never averaged
    directly.  Cross-day increments are split by duration proportion.
    """
    daily_amount = _apportion_accum_to_calendar_days(da, init_date, force_non_negative=force_non_negative)
    daily_amount.attrs.update(da.attrs)
    if output == "tp":
        daily = _tp_to_mm(daily_amount)
    elif output == "flux":
        daily = _energy_to_wm2(daily_amount, upward_positive=upward_positive)
    else:
        raise ValueError(output)
    return daily


def extract_window_only(daily: xr.DataArray, stage_name: str) -> xr.DataArray:
    start, end = STAGES[stage_name]
    return daily.sel(bjt_date=slice(start, end))


def _stage_mean(daily: xr.DataArray, stage_name: str) -> xr.DataArray:
    return extract_window_only(daily, stage_name).mean("bjt_date", skipna=True)


def _area_mean(da: xr.DataArray, box: Tuple[float, float, float, float]) -> float:
    sub = _sel_region(da, box)
    lat_weights = np.cos(np.deg2rad(sub.latitude))
    dims = [d for d in ("latitude", "longitude") if d in sub.dims]
    return float(sub.weighted(lat_weights).mean(dims, skipna=True)) if "latitude" in dims else float(sub.mean(dims, skipna=True))


def _concat_cf_pf(cf: Optional[xr.DataArray], pf: Optional[xr.DataArray]) -> xr.DataArray:
    parts_with_stream = [("cf", cf), ("pf", pf)]
    parts = [_normalize_member_dim(da, stream) for stream, da in parts_with_stream if da is not None]
    if not parts:
        raise FileNotFoundError("Neither CF nor PF field was loaded")
    out = xr.concat(parts, dim="member", coords="minimal", compat="override")
    return out.assign_coords(member=np.arange(out.sizes["member"]))


def _load_one_stream(kind: str, stream: str, product: str, init_date: str, short_name: Optional[str], candidates: Sequence[str], realtime: bool) -> Optional[xr.DataArray]:
    patterns = S2S_FILE_CATALOG[kind][stream].get(product, [])
    paths = _resolve_patterns(patterns, init_date)
    if not paths:
        tried = _format_candidate_paths(patterns, init_date)
        print(f"  [Missing] {kind}/{stream}/{product}: no files matched. Tried:")
        for candidate in tried:
            print("   ", candidate, os.path.exists(candidate))
        return None
    for path in paths:
        # Try requested shortName first, then fall back to unfiltered opening so
        # files using gh instead of z, or mixed flux bundles, are still reported
        # with their data_vars and can be selected by candidates.
        short_name_attempts = [short_name] if short_name is None else [short_name, None]
        for sn in short_name_attempts:
            try:
                ds = _open_cfgrib(path, short_name=sn)
                print(f"  [Open] {kind}/{stream}/{product}: {os.path.basename(path)} shortName={sn} data_vars={list(ds.data_vars)}")
                var = _guess_data_var(ds, candidates)
                da = get_s2s_init_slice(ds[var], init_date, realtime=realtime)
                print(f"  [Loaded] {kind}/{stream}/{product}: {os.path.basename(path)} -> {var}")
                return da
            except Exception as exc:
                print(f"  [Skip] {path} shortName={sn}: {exc}")
    return None


def load_s2s_ensemble_with_cf(kind: str, product: str, init_date: str, short_name: Optional[str], candidates: Sequence[str], realtime: bool) -> xr.DataArray:
    cf = _load_one_stream(kind, "cf", product, init_date, short_name, candidates, realtime)
    pf = _load_one_stream(kind, "pf", product, init_date, short_name, candidates, realtime)
    da = _concat_cf_pf(cf, pf)
    da = _sel_region(da, MAP_EXTENT)
    print(f"  [Members] {kind}/{product}/{init_date}: member_count={da.sizes.get('member', 0)}")
    return da


def _sample_mean(da: xr.DataArray) -> xr.DataArray:
    dims = [d for d in _sample_dims(da) if d in da.dims]
    return da.mean(dims, skipna=True) if dims else da


def _member_mean(da: xr.DataArray) -> xr.DataArray:
    return da.mean("member", skipna=True) if "member" in da.dims else da


def _clean_final_dims(da: xr.DataArray, keep_member: bool = False) -> xr.DataArray:
    keep = {"init_date", "stage", "latitude", "longitude"}
    if keep_member:
        keep.add("member")
    drop_dims = [d for d in da.dims if d not in keep]
    if drop_dims:
        da = da.mean(drop_dims, skipna=True)
    order = [d for d in ("init_date", "stage", "member", "latitude", "longitude") if d in da.dims]
    return da.transpose(*order)


def load_z500_reforecast_climatology(init_date: str, stage_name: str, level_hpa: int) -> xr.DataArray:
    product = "pl_500" if level_hpa == 500 else "pl_850"
    z_hc = load_s2s_ensemble_with_cf("hindcast", product, init_date, "z", ["gh", "z", "geopotential"], realtime=False)
    z_hc = _to_gpm_if_needed(_select_level(z_hc, level_hpa), f"z{level_hpa} hindcast")
    return _sample_mean(_stage_mean(process_s2s_bjt_daily_mean(z_hc, init_date), stage_name))


def apply_z500_anomaly(realtime_abs: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    return realtime_abs - climatology


def _stage_bundle_from_daily(realtime_daily: xr.DataArray, hindcast_daily: xr.DataArray, stage_name: str) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, Optional[xr.DataArray], int, int]:
    rt_stage_members = _stage_mean(realtime_daily, stage_name)
    hc_stage_samples = _stage_mean(hindcast_daily, stage_name)
    member_count = int(rt_stage_members.sizes.get("member", 1))
    hindcast_sample_count = int(np.prod([hc_stage_samples.sizes[d] for d in _sample_dims(hc_stage_samples) if d in hc_stage_samples.dims]) or 1)
    rt_abs = _member_mean(rt_stage_members)
    hc_clim = _sample_mean(hc_stage_samples)
    anom = rt_abs - hc_clim
    member_anom = rt_stage_members - hc_clim if "member" in rt_stage_members.dims else None
    return rt_abs, hc_clim, anom, member_anom, member_count, hindcast_sample_count


def _summary_record(init_date: str, stage_name: str, varname: str, anom: xr.DataArray, member_count: int, hindcast_sample_count: int) -> dict:
    vals = anom.values
    has_lat_lon = "latitude" in anom.dims and "longitude" in anom.dims
    rec = {
        "init_date": init_date,
        "stage": stage_name,
        "variable": varname,
        "units": VAR_META[varname]["units"],
        "dims": str(dict(anom.sizes)),
        "min": float(np.nanmin(vals)) if vals.size else np.nan,
        "max": float(np.nanmax(vals)) if vals.size else np.nan,
        "mean": float(np.nanmean(vals)) if vals.size else np.nan,
        "nchn_mean": _area_mean(anom, NCHN_BOX) if has_lat_lon else np.nan,
        "nan_ratio": float(np.isnan(vals).sum() / vals.size) if vals.size else np.nan,
        "has_lat_lon": has_lat_lon,
        "member_count": member_count,
        "hindcast_sample_count": hindcast_sample_count,
    }
    if varname == "z850" and has_lat_lon:
        rec["wnpsh_like_index"] = _area_mean(anom, WNPSH_BOX)
    return rec


def _print_check(rec: dict, extra: str = "") -> None:
    print(
        f"[SUMMARY] init={rec['init_date']} stage={rec['stage']} var={rec['variable']} units={rec['units']} "
        f"dims={rec['dims']} min={rec['min']:.3g} max={rec['max']:.3g} mean={rec['mean']:.3g} "
        f"nchn={rec['nchn_mean']:.3g} nan={rec['nan_ratio']:.2%} members={rec['member_count']} "
        f"hc_samples={rec['hindcast_sample_count']} has_lat_lon={rec['has_lat_lon']} {extra}"
    )


def _daily_pl_fields(kind: str, init_date: str, realtime: bool) -> Dict[str, xr.DataArray]:
    # PF pl_500/pl_850 may both point to the validated zqt bundle, but realtime
    # CF must use the exact cf_2023-6 pressure-level files prioritized in the catalog.
    if realtime:
        z500_product = "pl_500"
        z850_product = "pl_850"
    else:
        z500_product = "pl_500"
        z850_product = "pl_850"
    z500_raw = load_s2s_ensemble_with_cf(kind, z500_product, init_date, "z", ["gh", "z", "geopotential"], realtime=realtime)
    z850_raw = load_s2s_ensemble_with_cf(kind, z850_product, init_date, "z", ["gh", "z", "geopotential"], realtime=realtime)
    q_raw = load_s2s_ensemble_with_cf(kind, z850_product, init_date, "q", ["q", "specific_humidity"], realtime=realtime)
    u_raw = load_s2s_ensemble_with_cf(kind, "pl_uv", init_date, "u", ["u", "u_component_of_wind"], realtime=realtime)
    v_raw = load_s2s_ensemble_with_cf(kind, "pl_uv", init_date, "v", ["v", "v_component_of_wind"], realtime=realtime)

    z500 = _to_gpm_if_needed(_select_level(z500_raw, 500), "Z500 realtime" if realtime else "Z500 hindcast")
    z850 = _to_gpm_if_needed(_select_level(z850_raw, 850), "z850 realtime" if realtime else "z850 hindcast")
    q850, u850, v850 = xr.align(_select_level(q_raw, 850), _select_level(u_raw, 850), _select_level(v_raw, 850), join="inner")

    # Moisture flux anomaly is computed from real flux fields q*u and q*v first,
    # never from q_anom * u_anom.  q/u/v are aligned on the same 850 hPa grid,
    # member/init/step dimensions before flux construction.
    qu850 = q850 * u850
    qv850 = q850 * v850
    qu850.attrs["units"] = VAR_META["qu850"]["units"]
    qv850.attrs["units"] = VAR_META["qv850"]["units"]

    return {
        "Z500": process_s2s_bjt_daily_mean(z500, init_date),
        "z850": process_s2s_bjt_daily_mean(z850, init_date),
        "u850": process_s2s_bjt_daily_mean(u850, init_date),
        "v850": process_s2s_bjt_daily_mean(v850, init_date),
        "qu850": process_s2s_bjt_daily_mean(qu850, init_date),
        "qv850": process_s2s_bjt_daily_mean(qv850, init_date),
    }


def _daily_surface_fields(kind: str, init_date: str, realtime: bool) -> Dict[str, xr.DataArray]:
    tp = load_s2s_ensemble_with_cf(kind, "sfc_tp", init_date, "tp", ["tp", "total_precipitation"], realtime=realtime)
    mx2t = load_s2s_ensemble_with_cf(kind, "sfc_mx2t", init_date, "mx2t6", ["mx2t6", "mx2t", "maximum_2m_temperature"], realtime=realtime)
    sm20 = load_s2s_ensemble_with_cf(kind, "sfc_sm20", init_date, None, ["sm20", "swvl1", "soil_moisture"], realtime=realtime)
    ssr = load_s2s_ensemble_with_cf(kind, "sfc_ssr", init_date, "ssr", ["ssr", "surface_net_solar_radiation", "surface_solar_radiation"], realtime=realtime)
    lhf = load_s2s_ensemble_with_cf(kind, "sfc_lhf", init_date, "slhf", ["slhf", "lflux", "surface_latent_heat_flux"], realtime=realtime)
    shf = load_s2s_ensemble_with_cf(kind, "sfc_shf", init_date, "sshf", ["sshf", "hflux", "surface_sensible_heat_flux"], realtime=realtime)

    tp_daily = process_s2s_bjt_daily_accum(tp, init_date, output="tp", force_non_negative=True)
    mx2t_daily = process_s2s_bjt_daily_max(_to_celsius_if_needed(mx2t), init_date)
    sm20_converted = _sm20_to_volumetric(sm20)
    sm20_daily = process_s2s_bjt_daily_mean(sm20_converted, init_date)
    ssr_daily = process_s2s_bjt_daily_accum(ssr, init_date, output="flux", upward_positive=False)
    lhf_daily = process_s2s_bjt_daily_accum(lhf, init_date, output="flux", upward_positive=True)
    shf_daily = process_s2s_bjt_daily_accum(shf, init_date, output="flux", upward_positive=True)

    # EF is derived from processed daily upward-positive LHF and SHF, not from
    # anomalies.  Mask denominator <= 5 W m^-2 and clip physically to [0, 1].
    denom = lhf_daily + shf_daily
    ef_daily = (lhf_daily / denom).where(denom > EF_DENOM_THRESHOLD).clip(0.0, 1.0)
    ef_daily.attrs["units"] = "1"
    return {
        "tp": tp_daily,
        "mx2t": mx2t_daily,
        "sm20": sm20_daily,
        "SSR": ssr_daily,
        "LHF": lhf_daily,
        "SHF": shf_daily,
        "EF": ef_daily,
    }


def compute_anomaly_fields_for_init(init_date: str) -> Tuple[xr.Dataset, pd.DataFrame]:
    print("\n" + "=" * 80)
    print(f"Computing anomaly fields for init {init_date}; all stages use BJT valid dates")
    print("=" * 80)
    rt_daily = {**_daily_pl_fields("realtime", init_date, realtime=True), **_daily_surface_fields("realtime", init_date, realtime=True)}
    hc_daily = {**_daily_pl_fields("hindcast", init_date, realtime=False), **_daily_surface_fields("hindcast", init_date, realtime=False)}

    out_vars: Dict[str, xr.DataArray] = {}
    summary_rows: List[dict] = []
    member_anom_vars: Dict[str, xr.DataArray] = {}

    for varname in VAR_META:
        for stage_name in STAGES:
            rt_abs, hc_clim, anom, member_anom, member_count, hc_sample_count = _stage_bundle_from_daily(rt_daily[varname], hc_daily[varname], stage_name)
            rec = _summary_record(init_date, stage_name, varname, anom, member_count, hc_sample_count)
            extra = ""
            if varname in ("tp", "SSR", "LHF", "SHF"):
                extra = "accumulation=step-diff + BJT calendar-day proportional apportionment"
            if varname in ("Z500", "z850"):
                extra = f"{extra} gpm_conversion={rt_daily[varname].attrs.get('converted_to_gpm', 'checked')}"
            if varname == "EF":
                extra = f"EF_from_daily_flux=True denom_threshold={EF_DENOM_THRESHOLD}Wm-2 clipped=[0,1]"
            if varname == "sm20":
                extra = (
                    f"sm20_rt_original_units={rt_daily[varname].attrs.get('original_units', 'unknown')} "
                    f"sm20_rt_conversion={rt_daily[varname].attrs.get('conversion_note', 'unknown')} "
                    f"sm20_hc_original_units={hc_daily[varname].attrs.get('original_units', 'unknown')} "
                    f"sm20_hc_conversion={hc_daily[varname].attrs.get('conversion_note', 'unknown')}"
                )
            _print_check(rec, extra=extra)
            if member_count != 51:
                warnings.warn(
                    f"[member_count] init={init_date} stage={stage_name} var={varname}: "
                    f"expected realtime CF+PF member_count=51, got {member_count}. "
                    "Check whether realtime CF files in CF_DIR were opened and merged.",
                    UserWarning,
                    stacklevel=2,
                )
            summary_rows.append(rec)

            for field_name, da in (("abs", rt_abs), ("clim", hc_clim), ("anom", anom)):
                key = f"{varname}_{field_name}"
                da = _clean_final_dims(da.expand_dims(init_date=[init_date], stage=[stage_name]))
                da.attrs.update(VAR_META[varname])
                out_vars[key] = xr.concat([out_vars[key], da], dim="stage") if key in out_vars else da
            if member_anom is not None:
                mkey = f"{varname}_member_anom"
                mda = _clean_final_dims(member_anom.expand_dims(init_date=[init_date], stage=[stage_name]), keep_member=True)
                member_anom_vars[mkey] = xr.concat([member_anom_vars[mkey], mda], dim="stage") if mkey in member_anom_vars else mda

    ds = xr.Dataset({**out_vars, **member_anom_vars}).assign_coords(init_date=[init_date], stage=list(STAGES))
    ds.attrs.update(
        description="S2S realtime ensemble-mean absolute fields, hindcast climatologies, and anomalies",
        anomaly_definition="realtime ensemble mean minus same-init same-BJT-valid-window hindcast climatology",
        bjt_rule="valid_BJT = init_UTC + step + 8 hours; stages are selected by BJT valid date",
        accumulation_rule="tp/SSR/LHF/SHF use step differences and proportional apportionment to BJT calendar days",
        flux_rule="LHF/SHF are upward positive; EF = daily LHF_up/(LHF_up+SHF_up), denom > 5 W m-2, clipped [0,1]",
        moisture_flux_rule="qu850/qv850 anomalies are real flux anomalies, not q_anom times wind_anom",
    )
    nc_path = os.path.join(OUT_DIR, f"anomaly_fields_init_{init_date}.nc")
    ds.to_netcdf(nc_path)
    print(f"[Saved] {nc_path}")
    return ds, pd.DataFrame(summary_rows)


anom_dict: MutableMapping[str, xr.Dataset] = {}
summary_tables: List[pd.DataFrame] = []
for _init in INIT_DATES:
    _ds, _summary = compute_anomaly_fields_for_init(_init)
    anom_dict[_init] = _ds
    summary_tables.append(_summary)

anom_ds = xr.concat([anom_dict[d] for d in INIT_DATES], dim="init_date")
combined_nc = os.path.join(OUT_DIR, "anomaly_fields_two_inits_all_stages.nc")
anom_ds.to_netcdf(combined_nc)
print(f"[Saved] {combined_nc}")

summary_df = pd.concat(summary_tables, ignore_index=True)
summary_csv = os.path.join(OUT_DIR, "anomaly_evolution_summary_long.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"[Saved] {summary_csv}")
print(summary_df.to_string(index=False))

print("=" * 80)
print("FINAL QC SUMMARY")
print("=" * 80)
_qc_cols = ["init_date", "stage", "variable", "member_count", "hindcast_sample_count", "nan_ratio"]

_bad_members = summary_df.loc[summary_df["member_count"] != 51, _qc_cols]
print("[FINAL QC] member_count != 51")
if _bad_members.empty:
    print("  None")
else:
    print(_bad_members.to_string(index=False))

_bad_hindcast = summary_df.loc[summary_df["hindcast_sample_count"] <= 1, _qc_cols]
print("[FINAL QC] hindcast_sample_count <= 1")
if _bad_hindcast.empty:
    print("  None")
else:
    print(_bad_hindcast.to_string(index=False))

_low_hindcast = summary_df.loc[summary_df["hindcast_sample_count"] < 20, _qc_cols]
print("[FINAL QC WARNING] hindcast_sample_count < 20")
if _low_hindcast.empty:
    print("  None")
else:
    print(_low_hindcast.to_string(index=False))

_bad_nan = summary_df.loc[summary_df["nan_ratio"] > 0.05, _qc_cols]
print("[FINAL QC] nan_ratio > 5%")
if _bad_nan.empty:
    print("  None")
else:
    print(_bad_nan.to_string(index=False))
print("=" * 80)
print("Cell 1 complete.  Please inspect the summary and FINAL QC before running Cell 2 plotting.")

# %%
# =============================================================================
# Cell 2: Plot anomaly evolution figures by variable group
# =============================================================================
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Reopen Cell-1 output if this plotting cell is run in a fresh kernel.
if "OUT_DIR" not in globals():
    OUT_DIR = "/data1/huangy/fig6/NC/v5/factor_verification/anomaly_evolution"
if "STAGES" not in globals():
    STAGES = {"Stage-I": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17")), "Stage-II": (pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24")), "Total": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-24"))}
if "MAP_EXTENT" not in globals():
    MAP_EXTENT = (60.0, 150.0, 0.0, 60.0)
if "NCHN_BOX" not in globals():
    NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
if "WNPSH_BOX" not in globals():
    WNPSH_BOX = (115.0, 150.0, 15.0, 25.0)
if "anom_ds" not in globals():
    combined_nc = os.path.join(OUT_DIR, "anomaly_fields_two_inits_all_stages.nc")
    anom_ds = xr.open_dataset(combined_nc)

PLOT_INIT_DATES = [str(v) for v in anom_ds.init_date.values]
PLOT_STAGES = list(STAGES)
PROJ = ccrs.PlateCarree()


def _da2d(ds: xr.Dataset, var: str, init_date: str, stage_name: str) -> xr.DataArray:
    return ds[var].sel(init_date=init_date, stage=stage_name).squeeze(drop=True)


def _plot_box(ax, box, color="gold", lw=2.0, label=None):
    lon_min, lon_max, lat_min, lat_max = box
    ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], color=color, lw=lw, transform=PROJ, label=label)


def _setup_map(ax):
    ax.set_extent(MAP_EXTENT, crs=PROJ)
    ax.coastlines(resolution="50m", color="0.25", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, edgecolor="0.45", linewidth=0.5)
    try:
        ax.add_feature(cfeature.LAKES, edgecolor="0.55", facecolor="none", linewidth=0.4)
        ax.add_feature(cfeature.RIVERS, edgecolor="0.65", linewidth=0.3)
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.55", alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    _plot_box(ax, NCHN_BOX, color="gold", lw=2.0)
    _plot_box(ax, WNPSH_BOX, color="purple", lw=1.2)


LEVEL_FLOORS = {
    "Z500_anom": 10.0,
    "z850_anom": 5.0,
    "tp_anom": 0.5,
    "mx2t_anom": 0.5,
    "sm20_anom": 0.005,
    "SSR_anom": 10.0,
    "LHF_anom": 5.0,
    "SHF_anom": 5.0,
    "EF_anom": 0.05,
}


def _symmetric_levels(ds: xr.Dataset, var: str, n: int = 21, percentile: float = 98.0, floor: Optional[float] = None) -> np.ndarray:
    floor_value = LEVEL_FLOORS.get(var, 1.0) if floor is None else floor
    data = ds[var].values
    vmax = float(np.nanpercentile(np.abs(data), percentile)) if np.isfinite(data).any() else floor_value
    vmax = max(vmax, floor_value)
    return np.linspace(-vmax, vmax, n)


def _save_fig(fig, basename: str):
    png = os.path.join(OUT_DIR, f"{basename}.png")
    pdf = os.path.join(OUT_DIR, f"{basename}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png}")
    print(f"[Saved] {pdf}")


def plot_anomaly_grid(shading_var: str, basename: str, suptitle: str, cbar_label: str, cmap: str = "RdBu_r", contour_z500_5880: bool = False, vector_vars: Optional[Tuple[str, str]] = None, vector_key: Optional[Tuple[float, str]] = None, vector_stride: int = 5, vector_scale: Optional[float] = None):
    levels = _symmetric_levels(anom_ds, shading_var)
    norm = TwoSlopeNorm(vmin=float(levels[0]), vcenter=0.0, vmax=float(levels[-1]))
    fig, axes = plt.subplots(len(PLOT_INIT_DATES), len(PLOT_STAGES), figsize=(15, 8), subplot_kw={"projection": PROJ}, constrained_layout=True)
    mappable = None
    quiver_obj = None

    for i, init_date in enumerate(PLOT_INIT_DATES):
        for j, stage_name in enumerate(PLOT_STAGES):
            ax = axes[i, j]
            _setup_map(ax)
            da = _da2d(anom_ds, shading_var, init_date, stage_name)
            lon = da.longitude.values
            lat = da.latitude.values
            mappable = ax.contourf(lon, lat, da.values, levels=levels, cmap=cmap, norm=norm, extend="both", transform=PROJ)
            if contour_z500_5880:
                abs_da = _da2d(anom_ds, "Z500_abs", init_date, stage_name)
                amin = float(abs_da.min(skipna=True))
                amax = float(abs_da.max(skipna=True))
                if amin <= 5880.0 <= amax:
                    cs = ax.contour(abs_da.longitude.values, abs_da.latitude.values, abs_da.values, levels=[5880.0], colors="black", linewidths=1.6, transform=PROJ)
                    ax.clabel(cs, fmt={5880.0: "5880 gpm"}, fontsize=8)
            if vector_vars is not None:
                u = _da2d(anom_ds, vector_vars[0], init_date, stage_name)
                v = _da2d(anom_ds, vector_vars[1], init_date, stage_name)
                quiver_obj = ax.quiver(
                    u.longitude.values[::vector_stride],
                    u.latitude.values[::vector_stride],
                    u.values[::vector_stride, ::vector_stride],
                    v.values[::vector_stride, ::vector_stride],
                    transform=PROJ,
                    color="darkgreen",
                    scale=vector_scale,
                    width=0.0022,
                    headwidth=3.0,
                )
            ax.set_title(f"Init: {init_date} | {stage_name}", fontsize=10)

    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.045, pad=0.055)
    cbar.set_label(cbar_label, fontsize=11)
    if quiver_obj is not None and vector_key is not None:
        key_value, key_label = vector_key
        axes[0, -1].quiverkey(quiver_obj, X=0.83, Y=1.08, U=key_value, label=key_label, labelpos="E", coordinates="axes", fontproperties={"size": 9})
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    _save_fig(fig, basename)


def plot_all_anomaly_evolution_figures():
    plot_anomaly_grid(
        shading_var="Z500_anom",
        basename="figureA_z500_anomaly_evolution",
        suptitle="Figure A: Z500 anomaly = realtime ensemble mean - same-init hindcast climatology; stages use BJT valid dates; region 0-60N, 60-150E",
        cbar_label="Z500 anomaly (gpm); black contour: realtime absolute Z500 = 5880 gpm",
        contour_z500_5880=True,
    )
    plot_anomaly_grid(
        shading_var="z850_anom",
        basename="figureB_z850_uv850_anomaly_evolution",
        suptitle="Figure B: z850 anomaly with 850-hPa wind anomaly; replaces standalone WNPSH anomaly map; stages use BJT valid dates",
        cbar_label="z850 anomaly (gpm); vectors: u850/v850 anomaly (m s$^{-1}$)",
        vector_vars=("u850_anom", "v850_anom"),
        vector_key=(5.0, "5 m s$^{-1}$"),
        vector_stride=5,
        vector_scale=80.0,
    )
    plot_anomaly_grid(
        shading_var="tp_anom",
        basename="figureC_tp_moistureflux_anomaly_evolution",
        suptitle="Figure C: stage-mean daily precipitation anomaly with true 850-hPa moisture-flux anomaly vectors; stages use BJT valid dates",
        cbar_label="tp anomaly (mm day$^{-1}$); vectors: q850*u850 and q850*v850 anomaly (kg kg$^{-1}$ m s$^{-1}$)",
        vector_vars=("qu850_anom", "qv850_anom"),
        vector_key=(0.02, "0.02 kg kg$^{-1}$ m s$^{-1}$"),
        vector_stride=5,
        vector_scale=0.25,
    )
    plot_anomaly_grid("mx2t_anom", "figureD_mx2t_anomaly_evolution", "Figure D: stage-mean daily Tmax anomaly; BJT valid dates", "mx2t anomaly (°C)")
    plot_anomaly_grid("sm20_anom", "figureE_sm20_anomaly_evolution", "Figure E: soil moisture anomaly; BJT valid dates", "sm20 anomaly (m$^3$ m$^{-3}$)")
    plot_anomaly_grid("SSR_anom", "figureF_SSR_anomaly_evolution", "Figure F: SSR anomaly; accumulated radiation apportioned to BJT days", "SSR anomaly (W m$^{-2}$)")
    plot_anomaly_grid("LHF_anom", "figureG_LHF_anomaly_evolution", "Figure G: LHF anomaly (upward positive); BJT valid dates", "LHF anomaly, upward positive (W m$^{-2}$)")
    plot_anomaly_grid("SHF_anom", "figureH_SHF_anomaly_evolution", "Figure H: SHF anomaly (upward positive); BJT valid dates", "SHF anomaly, upward positive (W m$^{-2}$)")
    plot_anomaly_grid("EF_anom", "figureI_EF_anomaly_evolution", "Figure I: EF anomaly from processed daily LHF/SHF; BJT valid dates", "EF anomaly (unitless)")
    print("=" * 80)
    print(f"Anomaly evolution figures saved to: {OUT_DIR}")
    print("=" * 80)


# In a notebook, run Cell 2 and then call this function after inspecting Cell 1 summary:
# plot_all_anomaly_evolution_figures()
