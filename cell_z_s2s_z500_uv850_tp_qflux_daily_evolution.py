import glob
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=FutureWarning)

INIT_DATES = ["2023-06-08", "2023-06-12"]
TARGET_DATES = pd.to_datetime(["2023-06-17", "2023-06-18", "2023-06-19", "2023-06-20"])

BJT_OFFSET = pd.Timedelta(hours=8)
MAP_EXTENT = (60.0, 150.0, 0.0, 60.0)
NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
G0 = 9.80665

BASE_DATA_DIR = "/data1/huangy/fig6/NC"
CF_DIR = f"{BASE_DATA_DIR}/cf_2023-6"
HINDCAST_DIRS = {
    "2023-06-08": "/data1/huangy/fig6/NC/pl_08.6.23",
    "2023-06-12": "/data1/huangy/fig6/NC/pl_12.6.23",
}

OUT_DIR = "/data1/huangy/fig6/NC/v7/factor_verification/daily_evolution_s2s_circulation_precip"
os.makedirs(OUT_DIR, exist_ok=True)

VECTOR_STRIDE = 7
WIND_MIN_MS = 1.5
QFLUX_VECTOR_STRIDE = 7
QFLUX_MIN = 0.006
VECTOR_COLOR = "#00C853"

HINDCAST_CATALOG = {
    "cf": {
        "pl_500": ["{hindcast_dir}/ecmf_hindcast_cf_pl_500_{init_date}.grib"],
        "pl_850": ["{hindcast_dir}/ecmf_hindcast_cf_pl_850_{init_date}.grib"],
        "pl_uv": ["{hindcast_dir}/ecmf_hindcast_cf_pl_850_{init_date}.grib"],
        "sfc_tp": [
            "{hindcast_dir}/ecmf_hindcast_cf_sfc_mx2t_and_tp_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_cf_sfc_tp_{init_date}.grib",
        ],
    },
    "pf": {
        "pl_500": ["{hindcast_dir}/ecmf_hindcast_pf_pl_500_{init_date}.grib"],
        "pl_850": ["{hindcast_dir}/ecmf_hindcast_pf_pl_850_{init_date}.grib"],
        "pl_uv": ["{hindcast_dir}/ecmf_hindcast_pf_pl_850_{init_date}.grib"],
        "sfc_tp": [
            "{hindcast_dir}/ecmf_hindcast_pf_sfc_tp_{init_date}.grib",
            "{hindcast_dir}/ecmf_hindcast_pf_sfc_mx2t_and_tp_{init_date}.grib",
        ],
    },
}

S2S_FILE_CATALOG = {
    "realtime": {
        "pf": {
            "pl_500": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"],
            "pl_850": [f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"],
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
            "pl_850": [
                f"{CF_DIR}/ecmf_cf_pl_850_2023-06.grib",
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
    },
    "hindcast": HINDCAST_CATALOG,
}


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
                if np.isfinite(val) and abs(val - target) <= max(1.0, 0.01 * abs(target)):
                    return da.drop_vars(lev, errors="ignore")
                raise ValueError(
                    f"Field has scalar coordinate {lev}={val}, but requested level={target}. "
                    f"Please check whether the file is the expected pressure level."
                )

            if len(da[lev].dims) == 1:
                dim = da[lev].dims[0]
                if dim in da.dims:
                    idx = int(np.nanargmin(np.abs(vals - target)))
                    return da.isel({dim: idx}).drop_vars(lev, errors="ignore")

            raise KeyError(
                f"Coordinate {lev} exists but is not an indexable dimension. "
                f"dims={da.dims}, coord_dims={da[lev].dims}, requested={target}"
            )
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


def _to_gpm_if_needed(da: xr.DataArray, var_label: str) -> xr.DataArray:
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


def _concat_cf_pf(cf: Optional[xr.DataArray], pf: Optional[xr.DataArray]) -> xr.DataArray:
    parts = [_normalize_member_dim(da, stream) for stream, da in (("cf", cf), ("pf", pf)) if da is not None]
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


def _area_mean(da: xr.DataArray, box: Tuple[float, float, float, float]) -> float:
    sub = _sel_region(da, box)
    lat_weights = np.cos(np.deg2rad(sub.latitude))
    dims = [d for d in ("latitude", "longitude") if d in sub.dims]
    return float(sub.weighted(lat_weights).mean(dims, skipna=True)) if "latitude" in dims else float(sub.mean(dims, skipna=True))


def _daily_pl_fields(kind: str, init_date: str, realtime: bool) -> Dict[str, xr.DataArray]:
    z500_raw = load_s2s_ensemble_with_cf(kind, "pl_500", init_date, "z", ["gh", "z", "geopotential"], realtime=realtime)
    q_raw = load_s2s_ensemble_with_cf(kind, "pl_850", init_date, "q", ["q", "specific_humidity"], realtime=realtime)
    u_raw = load_s2s_ensemble_with_cf(kind, "pl_uv", init_date, "u", ["u", "u_component_of_wind"], realtime=realtime)
    v_raw = load_s2s_ensemble_with_cf(kind, "pl_uv", init_date, "v", ["v", "v_component_of_wind"], realtime=realtime)

    z500 = _to_gpm_if_needed(_select_level(z500_raw, 500), "Z500 realtime" if realtime else "Z500 hindcast")
    q850, u850, v850 = xr.align(_select_level(q_raw, 850), _select_level(u_raw, 850), _select_level(v_raw, 850), join="inner")

    qu850 = q850 * u850
    qv850 = q850 * v850
    qu850.attrs["units"] = "kg kg-1 m s-1"
    qv850.attrs["units"] = "kg kg-1 m s-1"

    return {
        "Z500": process_s2s_bjt_daily_mean(z500, init_date),
        "u850": process_s2s_bjt_daily_mean(u850, init_date),
        "v850": process_s2s_bjt_daily_mean(v850, init_date),
        "qu850": process_s2s_bjt_daily_mean(qu850, init_date),
        "qv850": process_s2s_bjt_daily_mean(qv850, init_date),
    }


def _daily_z500_field(kind: str, init_date: str, realtime: bool) -> xr.DataArray:
    z500_raw = load_s2s_ensemble_with_cf(kind, "pl_500", init_date, "z", ["gh", "z", "geopotential"], realtime=realtime)
    z500 = _to_gpm_if_needed(_select_level(z500_raw, 500), "Z500 realtime" if realtime else "Z500 hindcast")
    return process_s2s_bjt_daily_mean(z500, init_date)


def _daily_surface_fields(kind: str, init_date: str, realtime: bool) -> Dict[str, xr.DataArray]:
    tp = load_s2s_ensemble_with_cf(kind, "sfc_tp", init_date, "tp", ["tp", "total_precipitation"], realtime=realtime)
    tp_daily = process_s2s_bjt_daily_accum(tp, init_date, output="tp", force_non_negative=True)
    return {"tp": tp_daily}


def _select_target_dates(daily: xr.DataArray) -> xr.DataArray:
    return daily.sel(bjt_date=TARGET_DATES)


def build_daily_s2s_fields_for_init(init_date: str) -> Dict[str, xr.DataArray]:
    rt_daily = {**_daily_pl_fields("realtime", init_date, realtime=True), **_daily_surface_fields("realtime", init_date, realtime=True)}
    hc_z500_daily = _daily_z500_field("hindcast", init_date, realtime=False)

    rt_tp_members = _select_target_dates(rt_daily["tp"])
    rt_z500_members = _select_target_dates(rt_daily["Z500"])
    rt_u850_members = _select_target_dates(rt_daily["u850"])
    rt_v850_members = _select_target_dates(rt_daily["v850"])
    rt_qu850_members = _select_target_dates(rt_daily["qu850"])
    rt_qv850_members = _select_target_dates(rt_daily["qv850"])

    hc_z500_samples = _select_target_dates(hc_z500_daily)
    z500_clim = _sample_mean(hc_z500_samples)

    return {
        "tp_abs": _member_mean(rt_tp_members),
        "z500_abs": _member_mean(rt_z500_members),
        "z500_clim": z500_clim,
        "z500_anom": _member_mean(rt_z500_members) - z500_clim,
        "u850_abs": _member_mean(rt_u850_members),
        "v850_abs": _member_mean(rt_v850_members),
        "qu850_abs": _member_mean(rt_qu850_members),
        "qv850_abs": _member_mean(rt_qv850_members),
        "number_of_members_used": int(rt_tp_members.sizes.get("member", 1)),
        "z500_clim_year_count": int(np.prod([hc_z500_samples.sizes[d] for d in _sample_dims(hc_z500_samples) if d in hc_z500_samples.dims]) or 1),
    }


def _setup_map(ax):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", lw=0.8)
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    ax.add_feature(cfeature.LAKES.with_scale("110m"), linewidth=0.3, edgecolor="gray", facecolor="none")
    ax.add_feature(cfeature.RIVERS.with_scale("110m"), linewidth=0.25, edgecolor="steelblue")
    ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.4, alpha=0.6)


def _plot_box(ax, box=NCHN_BOX):
    x0, x1, y0, y1 = box
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="k", lw=1.2, transform=ccrs.PlateCarree())


def _save_fig(fig, stem):
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _prepare_uv_quiver_arrays(lon, lat, u2d, v2d):
    lon_sub = lon.values[::VECTOR_STRIDE]
    lat_sub = lat.values[::VECTOR_STRIDE]
    u_sub = u2d.values[::VECTOR_STRIDE, ::VECTOR_STRIDE]
    v_sub = v2d.values[::VECTOR_STRIDE, ::VECTOR_STRIDE]
    x2d, y2d = np.meshgrid(lon_sub, lat_sub)
    ws = np.sqrt(u_sub ** 2 + v_sub ** 2)
    mask = ws >= WIND_MIN_MS
    return x2d[mask], y2d[mask], u_sub[mask], v_sub[mask]


def _plot_uv850_quiver(ax, lon, lat, u2d, v2d):
    x_plot, y_plot, u_plot, v_plot = _prepare_uv_quiver_arrays(lon, lat, u2d, v2d)
    print(f"[UV quiver] n_vectors={u_plot.size}, WIND_MIN_MS={WIND_MIN_MS}, stride={VECTOR_STRIDE}")
    if u_plot.size == 0:
        print(f"[Warn] no UV850 vectors after ws>={WIND_MIN_MS}")
        return None
    common_kwargs = dict(alpha=1.0, scale=130, pivot="mid", headwidth=6.2, headlength=7.2, headaxislength=6.2, transform=ccrs.PlateCarree())
    ax.quiver(x_plot, y_plot, u_plot, v_plot, color="white", width=0.0042, zorder=20, **common_kwargs)
    q = ax.quiver(x_plot, y_plot, u_plot, v_plot, color=VECTOR_COLOR, width=0.0030, zorder=21, **common_kwargs)
    return q


def _prepare_qflux_quiver_arrays(lon, lat, qu2d, qv2d):
    lon_sub = lon.values[::QFLUX_VECTOR_STRIDE]
    lat_sub = lat.values[::QFLUX_VECTOR_STRIDE]
    qu_sub = qu2d.values[::QFLUX_VECTOR_STRIDE, ::QFLUX_VECTOR_STRIDE]
    qv_sub = qv2d.values[::QFLUX_VECTOR_STRIDE, ::QFLUX_VECTOR_STRIDE]
    x2d, y2d = np.meshgrid(lon_sub, lat_sub)
    qmag = np.sqrt(qu_sub ** 2 + qv_sub ** 2)
    mask = qmag >= QFLUX_MIN
    return x2d[mask], y2d[mask], qu_sub[mask], qv_sub[mask]


def _plot_qflux850_quiver(ax, lon, lat, qu2d, qv2d):
    x_plot, y_plot, qu_plot, qv_plot = _prepare_qflux_quiver_arrays(lon, lat, qu2d, qv2d)
    print(f"[QFLUX quiver] n_vectors={qu_plot.size}, QFLUX_MIN={QFLUX_MIN}, stride={QFLUX_VECTOR_STRIDE}")
    if qu_plot.size == 0:
        print(f"[Warn] no qflux vectors after |qflux|>={QFLUX_MIN}")
        return None
    common_kwargs = dict(alpha=1.0, scale=1.0, pivot="mid", headwidth=6.2, headlength=7.2, headaxislength=6.2, transform=ccrs.PlateCarree())
    ax.quiver(x_plot, y_plot, qu_plot, qv_plot, color="white", width=0.0040, zorder=20, **common_kwargs)
    q = ax.quiver(x_plot, y_plot, qu_plot, qv_plot, color=VECTOR_COLOR, width=0.0029, zorder=21, **common_kwargs)
    return q


def _plot_tp_z500_uv(field_map: Dict[str, Dict[str, xr.DataArray]], use_anom: bool, stem: str):
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    all_tp = np.concatenate([field_map[i]["tp_abs"].values.ravel() for i in INIT_DATES])
    pmax = max(float(np.nanpercentile(all_tp, 98)), 1.0)
    plev = np.linspace(0, pmax, 13)
    qobj = None
    for r, init_date in enumerate(INIT_DATES):
        f = field_map[init_date]
        for c, d in enumerate(TARGET_DATES):
            ax = axs[r, c]
            _setup_map(ax)
            _plot_box(ax)
            tp = f["tp_abs"].sel(bjt_date=d)
            cf = ax.contourf(tp.longitude, tp.latitude, tp, levels=plev, cmap="Blues", extend="max", transform=ccrs.PlateCarree())
            zabs = f["z500_abs"].sel(bjt_date=d)
            if use_anom:
                zan = f["z500_anom"].sel(bjt_date=d)
                lv = np.array([-80, -60, -40, -20, 20, 40, 60, 80])
                lv = lv[(lv >= np.nanmin(zan.values)) & (lv <= np.nanmax(zan.values))]
                if lv.size:
                    neg = lv[lv < 0]
                    pos = lv[lv > 0]
                    if neg.size:
                        ax.contour(zan.longitude, zan.latitude, zan, levels=neg, colors="royalblue", linestyles="--", linewidths=1.0, transform=ccrs.PlateCarree())
                    if pos.size:
                        ax.contour(zan.longitude, zan.latitude, zan, levels=pos, colors="firebrick", linestyles="-", linewidths=1.0, transform=ccrs.PlateCarree())
            else:
                lv = np.arange(5400, 6021, 40)
                ax.contour(zabs.longitude, zabs.latitude, zabs, levels=lv, colors="0.2", linewidths=0.7, transform=ccrs.PlateCarree())
            c5880 = ax.contour(zabs.longitude, zabs.latitude, zabs, levels=[5880], colors="purple", linewidths=1.8, transform=ccrs.PlateCarree())
            if len(c5880.allsegs[0]) == 0:
                print(f"[Warn] 5880 not present for init={init_date}, date={d.date()}")
            q = _plot_uv850_quiver(ax, f["u850_abs"].longitude, f["u850_abs"].latitude, f["u850_abs"].sel(bjt_date=d), f["v850_abs"].sel(bjt_date=d))
            if qobj is None and q is not None:
                qobj = q
            ax.set_title(f"Init {init_date} | {d.strftime('%Y-%m-%d')} BJT", fontsize=9)
    cax = fig.add_axes([0.2, 0.08, 0.6, 0.025])
    fig.colorbar(cf, cax=cax, orientation="horizontal", label="TP (mm day$^{-1}$)")
    if qobj is not None:
        axs[0, 0].quiverkey(qobj, X=0.08, Y=-0.12, U=5.0, label="5 m s$^{-1}$", labelpos="E", coordinates="axes")
    _save_fig(fig, stem)


def _plot_tp_qflux(field_map: Dict[str, Dict[str, xr.DataArray]], stem: str):
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    all_tp = np.concatenate([field_map[i]["tp_abs"].values.ravel() for i in INIT_DATES])
    pmax = max(float(np.nanpercentile(all_tp, 98)), 1.0)
    plev = np.linspace(0, pmax, 13)
    qobj = None
    for r, init_date in enumerate(INIT_DATES):
        f = field_map[init_date]
        for c, d in enumerate(TARGET_DATES):
            ax = axs[r, c]
            _setup_map(ax)
            _plot_box(ax)
            tp = f["tp_abs"].sel(bjt_date=d)
            cf = ax.contourf(tp.longitude, tp.latitude, tp, levels=plev, cmap="Blues", extend="max", transform=ccrs.PlateCarree())
            q = _plot_qflux850_quiver(ax, f["qu850_abs"].longitude, f["qu850_abs"].latitude, f["qu850_abs"].sel(bjt_date=d), f["qv850_abs"].sel(bjt_date=d))
            if qobj is None and q is not None:
                qobj = q
            ax.set_title(f"Init {init_date} | {d.strftime('%Y-%m-%d')} BJT", fontsize=9)
    cax = fig.add_axes([0.2, 0.08, 0.6, 0.025])
    fig.colorbar(cf, cax=cax, orientation="horizontal", label="TP (mm day$^{-1}$)")
    if qobj is not None:
        axs[0, 0].quiverkey(qobj, X=0.08, Y=-0.12, U=0.02, label="0.02 kg kg$^{-1}$ m s$^{-1}$", labelpos="E", coordinates="axes")
    _save_fig(fig, stem)


def _build_qc_rows(init_date: str, f: Dict[str, xr.DataArray]) -> List[dict]:
    rows = []
    for d in TARGET_DATES:
        tp = f["tp_abs"].sel(bjt_date=d)
        zabs = f["z500_abs"].sel(bjt_date=d)
        zan = f["z500_anom"].sel(bjt_date=d)
        u = f["u850_abs"].sel(bjt_date=d)
        v = f["v850_abs"].sel(bjt_date=d)
        qu = f["qu850_abs"].sel(bjt_date=d)
        qv = f["qv850_abs"].sel(bjt_date=d)
        uvspd = np.sqrt(u ** 2 + v ** 2)
        qmag = np.sqrt(qu ** 2 + qv ** 2)

        rows.append({
            "init_date": init_date,
            "date_bjt": d.strftime("%Y-%m-%d"),
            "number_of_members_used": f.get("number_of_members_used", np.nan),
            "tp_nchn_mean": _area_mean(tp, NCHN_BOX),
            "tp_domain_max": float(tp.max(skipna=True)),
            "z500_abs_nchn_mean": _area_mean(zabs, NCHN_BOX),
            "z500_anom_nchn_mean": _area_mean(zan, NCHN_BOX),
            "u850_nchn_mean": _area_mean(u, NCHN_BOX),
            "v850_nchn_mean": _area_mean(v, NCHN_BOX),
            "uv850_speed_nchn_mean": _area_mean(uvspd, NCHN_BOX),
            "qu850_nchn_mean": _area_mean(qu, NCHN_BOX),
            "qv850_nchn_mean": _area_mean(qv, NCHN_BOX),
            "qflux_mag_nchn_mean": _area_mean(qmag, NCHN_BOX),
            "z500_clim_year_count": f.get("z500_clim_year_count", np.nan),
            "tp_units": str(tp.attrs.get("units", "")),
            "z500_units": str(zabs.attrs.get("units", "")),
            "u850_units": str(u.attrs.get("units", "")),
            "v850_units": str(v.attrs.get("units", "")),
            "qflux_units": str(qu.attrs.get("units", "kg kg-1 m s-1")),
            "bjt_start": f"{d.strftime('%Y-%m-%d')} 00:00",
            "bjt_end": (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00"),
            "utc_start": (d - pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
            "utc_end": (d + pd.Timedelta(hours=16)).strftime("%Y-%m-%d %H:%M"),
        })
    return rows


def run_cell_z_s2s_daily_evolution(make_plots: bool = True):
    field_map: Dict[str, Dict[str, xr.DataArray]] = {}
    all_rows = []
    for init_date in INIT_DATES:
        print(f"\n=== Processing init_date={init_date} ===")
        f = build_daily_s2s_fields_for_init(init_date)
        field_map[init_date] = f
        all_rows.extend(_build_qc_rows(init_date, f))

    qc_path = os.path.join(OUT_DIR, "daily_evolution_s2s_z500_uv850_tp_qflux_qc.csv")
    pd.DataFrame(all_rows).to_csv(qc_path, index=False)
    print(f"[QC] saved: {qc_path}")

    if make_plots:
        _plot_tp_z500_uv(field_map, use_anom=False, stem="figure_s2s_daily_evolution_tp_z500abs_uv850")
        _plot_tp_z500_uv(field_map, use_anom=True, stem="figure_s2s_daily_evolution_tp_z500anom_uv850")
        _plot_tp_qflux(field_map, stem="figure_s2s_daily_evolution_tp_qflux850")


if __name__ == "__main__":
    run_cell_z_s2s_daily_evolution(make_plots=True)
