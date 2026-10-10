import glob
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore", category=FutureWarning)

# ====================================================
# Basic configuration
# ====================================================

INIT_DATES = ["2023-06-08", "2023-06-12"]
TARGET_DATES = pd.to_datetime(["2023-06-17", "2023-06-18", "2023-06-19", "2023-06-20"])

BJT_OFFSET = pd.Timedelta(hours=8)
MAP_EXTENT = (90.0, 135.0, 25.0, 55.0)
DATA_EXTENT = (88.0, 137.0, 23.0, 57.0)
NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
G0 = 9.80665

BASE_DATA_DIR = "/data1/huangy/fig6/NC"
CF_DIR = f"{BASE_DATA_DIR}/cf_2023-6"

GROUP_FILE = "/data1/huangy/fig6/NC/v7/tmax_error_stage_eval/tables/tmax_member_hotcold_group_by_stage.csv"

OUT_DIR = "/data1/huangy/fig6/NC/v8/factor_verification/stageII_hotcold_daily_circulation"
os.makedirs(OUT_DIR, exist_ok=True)

# Final S2S quiver style from current Cell Z.
VECTOR_COLOR = "#00C853"
VECTOR_STRIDE = 7
WIND_MIN_MS = 1.5
WIND_DIFF_MIN_MS = 0.5

GROUP_ORDER = ["Hot17", "Middle17", "Cold17", "Hot17_minus_Cold17"]
GROUP_LABELS = {
    "Hot17": "Hot17",
    "Middle17": "Middle17",
    "Cold17": "Cold17",
    "Hot17_minus_Cold17": "Hot17 − Cold17",
}

# ====================================================
# S2S realtime file candidates
# ====================================================

S2S_FILE_CATALOG = {
    "realtime": {
        "pf": {
            "pl_500": [
                f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb",
            ],
            "pl_uv": [
                f"{BASE_DATA_DIR}/MSE/ecmf_pf_131_132_2023-06.grib",
                f"{BASE_DATA_DIR}/2023-06/*pf*131*132*2023-06*.grb*",
                f"{BASE_DATA_DIR}/2023-06/*pf*uv*2023-06*.grb*",
            ],
            "sfc_tp": [
                f"{BASE_DATA_DIR}/2023-06/ecmf_rel_pf_sfc_tp2023-06.grb",
            ],
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
# Grouping helpers
# ====================================================

def load_stageII_hotcold_groups(group_file: str) -> pd.DataFrame:
    if not os.path.exists(group_file):
        raise FileNotFoundError(f"GROUP_FILE not found: {group_file}")

    df = pd.read_csv(group_file)
    required = {"init", "stage", "hotcold_group", "member"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Group file missing required columns: {missing}; columns={list(df.columns)}")

    df = df.copy()
    df["init"] = df["init"].astype(str)
    df["stage"] = df["stage"].astype(str)
    df["hotcold_group"] = df["hotcold_group"].astype(str)
    df["member"] = df["member"].astype(int)

    # Critical: only use Stage-II grouping.
    df2 = df[df["stage"] == "Stage-II"].copy()
    if df2.empty:
        raise ValueError("No Stage-II rows found in the hot/cold member group file")

    print("[Group check] Stage-II group counts:")
    print(df2.groupby(["init", "stage", "hotcold_group"]).size())

    for init in INIT_DATES:
        for group in ["Hot17", "Middle17", "Cold17"]:
            members = get_hotcold_members(df2, init, group)
            if len(members) != 17:
                raise ValueError(f"Expected 17 members for init={init}, group={group}, got {len(members)}")

    return df2


def get_hotcold_members(group_df: pd.DataFrame, init: str, group: str) -> List[int]:
    sub = group_df[
        (group_df["init"].astype(str) == str(init))
        & (group_df["stage"].astype(str) == "Stage-II")
        & (group_df["hotcold_group"].astype(str) == str(group))
    ].copy()

    members = sub["member"].astype(int).tolist()
    if len(members) != 17:
        raise ValueError(
            f"Expected 17 members for init={init}, stage=Stage-II, group={group}, got {len(members)}. "
            f"members={members}"
        )
    if len(set(members)) != 17:
        raise ValueError(f"Duplicated members for init={init}, group={group}: {members}")
    if min(members) < 0 or max(members) > 50:
        raise ValueError(f"Member IDs should be 0..50, got init={init}, group={group}, members={members}")
    return members


def build_group_audit_table(group_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for init in INIT_DATES:
        for group in ["Hot17", "Middle17", "Cold17"]:
            members = get_hotcold_members(group_df, init, group)
            rows.append({
                "init_date": init,
                "stage": "Stage-II",
                "hotcold_group": group,
                "n_members": len(members),
                "members": ",".join(str(m) for m in members),
            })
    return pd.DataFrame(rows)


# ====================================================
# Generic S2S readers / utilities
# ====================================================

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
                raise ValueError(f"Field has scalar coordinate {lev}={val}, but requested level={target}.")

            if len(da[lev].dims) == 1:
                dim = da[lev].dims[0]
                if dim in da.dims:
                    idx = int(np.nanargmin(np.abs(vals - target)))
                    return da.isel({dim: idx}).drop_vars(lev, errors="ignore")

            raise KeyError(
                f"Coordinate {lev} exists but is not indexable. dims={da.dims}, coord_dims={da[lev].dims}"
            )

    return da


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


def _apportion_accum_to_calendar_days(
    da: xr.DataArray,
    init_date: str,
    force_non_negative: bool = False,
) -> xr.DataArray:
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


def process_s2s_bjt_daily_accum(
    da: xr.DataArray,
    init_date: str,
    output: str,
    force_non_negative: bool = False,
) -> xr.DataArray:
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


def _load_one_stream(
    stream: str,
    product: str,
    init_date: str,
    short_name: Optional[str],
    candidates: Sequence[str],
) -> Optional[xr.DataArray]:
    patterns = S2S_FILE_CATALOG["realtime"][stream].get(product, [])
    paths = _resolve_patterns(patterns, init_date)

    if not paths:
        tried = _format_candidate_paths(patterns, init_date)
        print(f"  [Missing] realtime/{stream}/{product}: no files matched. Tried:")
        for candidate in tried:
            print("   ", candidate, os.path.exists(candidate))
        return None

    for path in paths:
        short_name_attempts = [short_name] if short_name is None else [short_name, None]
        for sn in short_name_attempts:
            try:
                ds = _open_cfgrib(path, short_name=sn)
                print(
                    f"  [Open] realtime/{stream}/{product}: "
                    f"{os.path.basename(path)} shortName={sn} data_vars={list(ds.data_vars)}"
                )
                var = _guess_data_var(ds, candidates)
                da = get_s2s_init_slice(ds[var], init_date)
                print(f"  [Loaded] realtime/{stream}/{product}: {os.path.basename(path)} -> {var}")
                return da
            except Exception as exc:
                print(f"  [Skip] {path} shortName={sn}: {exc}")

    return None


def load_s2s_ensemble_with_cf(
    product: str,
    init_date: str,
    short_name: Optional[str],
    candidates: Sequence[str],
) -> xr.DataArray:
    cf = _load_one_stream("cf", product, init_date, short_name, candidates)
    pf = _load_one_stream("pf", product, init_date, short_name, candidates)
    da = _concat_cf_pf(cf, pf)
    da = _sel_region(da, DATA_EXTENT)

    member_count = int(da.sizes.get("member", 0))
    print(f"  [Members] realtime/{product}/{init_date}: member_count={member_count}")
    if member_count != 51:
        raise ValueError(f"Expected 51 S2S members after CF+PF merge, got {member_count}")

    return da


# ====================================================
# Daily S2S fields
# ====================================================

def build_daily_realtime_member_fields(init_date: str) -> Dict[str, xr.DataArray]:
    print(f"\n=== Loading S2S realtime fields for init={init_date} ===")

    z500_raw = load_s2s_ensemble_with_cf("pl_500", init_date, "z", ["gh", "z", "geopotential"])
    u_raw = load_s2s_ensemble_with_cf("pl_uv", init_date, "u", ["u", "u_component_of_wind"])
    v_raw = load_s2s_ensemble_with_cf("pl_uv", init_date, "v", ["v", "v_component_of_wind"])
    tp_raw = load_s2s_ensemble_with_cf("sfc_tp", init_date, "tp", ["tp", "total_precipitation"])

    z500 = _to_gpm_if_needed(_select_level(z500_raw, 500), "Z500 realtime")
    u850 = _select_level(u_raw, 850)
    v850 = _select_level(v_raw, 850)

    z500_daily = process_s2s_bjt_daily_mean(z500, init_date).sel(bjt_date=TARGET_DATES)
    u850_daily = process_s2s_bjt_daily_mean(u850, init_date).sel(bjt_date=TARGET_DATES)
    v850_daily = process_s2s_bjt_daily_mean(v850, init_date).sel(bjt_date=TARGET_DATES)
    tp_daily = process_s2s_bjt_daily_accum(tp_raw, init_date, output="tp", force_non_negative=True).sel(bjt_date=TARGET_DATES)

    return {
        "tp": tp_daily,
        "z500": z500_daily,
        "u850": u850_daily,
        "v850": v850_daily,
    }


# ====================================================
# Composite logic
# ====================================================

def group_mean(da: xr.DataArray, members: List[int]) -> xr.DataArray:
    return da.sel(member=members).mean("member", skipna=True)


def build_group_composites_for_init(
    init_date: str,
    group_df: pd.DataFrame,
    fields: Dict[str, xr.DataArray],
) -> Dict[str, Dict[str, xr.DataArray]]:
    out = {}

    for group in ["Hot17", "Middle17", "Cold17"]:
        members = get_hotcold_members(group_df, init_date, group)
        out[group] = {
            "members": members,
            "tp": group_mean(fields["tp"], members),
            "z500": group_mean(fields["z500"], members),
            "u850": group_mean(fields["u850"], members),
            "v850": group_mean(fields["v850"], members),
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
# Plotting utilities
# ====================================================

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


def _save_fig(fig, stem: str):
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _prepare_uv_quiver_arrays(lon, lat, u2d, v2d, min_speed: float):
    lon_sub = lon.values[::VECTOR_STRIDE]
    lat_sub = lat.values[::VECTOR_STRIDE]
    u_sub = u2d.values[::VECTOR_STRIDE, ::VECTOR_STRIDE]
    v_sub = v2d.values[::VECTOR_STRIDE, ::VECTOR_STRIDE]
    x2d, y2d = np.meshgrid(lon_sub, lat_sub)
    ws = np.sqrt(u_sub ** 2 + v_sub ** 2)
    mask = ws >= min_speed
    return x2d[mask], y2d[mask], u_sub[mask], v_sub[mask]


def _plot_uv850_quiver(ax, lon, lat, u2d, v2d, is_difference: bool = False):
    min_speed = WIND_DIFF_MIN_MS if is_difference else WIND_MIN_MS
    x_plot, y_plot, u_plot, v_plot = _prepare_uv_quiver_arrays(lon, lat, u2d, v2d, min_speed)
    label = "UV850 diff" if is_difference else "UV850 abs"
    print(f"[{label} quiver] n_vectors={u_plot.size}, min_speed={min_speed}, stride={VECTOR_STRIDE}")
    if u_plot.size == 0:
        print(f"[Warn] no {label} vectors after speed>={min_speed}")
        return None

    scale = 42 if is_difference else 85
    halo_width = 0.0045 if is_difference else 0.0048
    main_width = 0.0032 if is_difference else 0.0035
    common_kwargs = dict(
        alpha=1.0,
        scale=scale,
        pivot="mid",
        headwidth=6.2,
        headlength=7.2,
        headaxislength=6.2,
        transform=ccrs.PlateCarree(),
    )
    ax.quiver(x_plot, y_plot, u_plot, v_plot, color="white", width=halo_width, zorder=20, **common_kwargs)
    q = ax.quiver(x_plot, y_plot, u_plot, v_plot, color=VECTOR_COLOR, width=main_width, zorder=21, **common_kwargs)
    return q


def _tp_abs_levels(composites: Dict[str, Dict[str, xr.DataArray]]) -> np.ndarray:
    pool = np.concatenate([composites[g]["tp"].values.ravel() for g in ["Hot17", "Middle17", "Cold17"]])
    if np.all(np.isnan(pool)):
        raise ValueError("All absolute TP values are NaN; cannot plot")
    pmax = max(float(np.nanpercentile(pool, 95)), 1.0)
    pmax = min(pmax, 25.0)
    return np.linspace(0, pmax, 13)


def _tp_diff_levels(composites: Dict[str, Dict[str, xr.DataArray]]) -> Tuple[np.ndarray, mcolors.TwoSlopeNorm]:
    vals = composites["Hot17_minus_Cold17"]["tp"].values
    if np.all(np.isnan(vals)):
        raise ValueError("All Hot17-Cold17 TP difference values are NaN; cannot plot")
    vmax = max(float(np.nanpercentile(np.abs(vals), 98)), 0.5)
    levels = np.linspace(-vmax, vmax, 13)
    return levels, mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_stageII_daily_circulation(init_date: str, composites: Dict[str, Dict[str, xr.DataArray]]):
    fig, axs = plt.subplots(4, 4, figsize=(20, 14), subplot_kw={"projection": ccrs.PlateCarree()})
    abs_levels = _tp_abs_levels(composites)
    diff_levels, diff_norm = _tp_diff_levels(composites)
    cf_abs = None
    cf_diff = None
    q_abs_key = None
    q_diff_key = None

    for row, group in enumerate(GROUP_ORDER):
        comp = composites[group]
        for col, d in enumerate(TARGET_DATES):
            ax = axs[row, col]
            _setup_map(ax)
            _plot_box(ax)

            tp = comp["tp"].sel(bjt_date=d)
            z500 = comp["z500"].sel(bjt_date=d)
            u850 = comp["u850"].sel(bjt_date=d)
            v850 = comp["v850"].sel(bjt_date=d)

            if group == "Hot17_minus_Cold17":
                cf_diff = ax.contourf(
                    tp.longitude,
                    tp.latitude,
                    tp,
                    levels=diff_levels,
                    cmap="RdBu_r",
                    norm=diff_norm,
                    extend="both",
                    transform=ccrs.PlateCarree(),
                )
                zlevels = np.array([-80, -60, -40, -20, 20, 40, 60, 80])
                zmin = float(np.nanmin(z500.values))
                zmax = float(np.nanmax(z500.values))
                zlevels = zlevels[(zlevels >= zmin) & (zlevels <= zmax)]
                neg = zlevels[zlevels < 0]
                pos = zlevels[zlevels > 0]
                if neg.size:
                    ax.contour(z500.longitude, z500.latitude, z500, levels=neg, colors="royalblue", linestyles="--", linewidths=1.0, transform=ccrs.PlateCarree())
                if pos.size:
                    ax.contour(z500.longitude, z500.latitude, z500, levels=pos, colors="firebrick", linestyles="-", linewidths=1.0, transform=ccrs.PlateCarree())
                q = _plot_uv850_quiver(ax, u850.longitude, u850.latitude, u850, v850, is_difference=True)
                if q_diff_key is None and q is not None:
                    q_diff_key = q
            else:
                cf_abs = ax.contourf(
                    tp.longitude,
                    tp.latitude,
                    tp,
                    levels=abs_levels,
                    cmap="Blues",
                    extend="max",
                    transform=ccrs.PlateCarree(),
                )
                ax.contour(z500.longitude, z500.latitude, z500, levels=np.arange(5400, 6021, 40), colors="0.2", linewidths=0.7, transform=ccrs.PlateCarree())
                c5880 = ax.contour(z500.longitude, z500.latitude, z500, levels=[5880], colors="purple", linewidths=1.8, transform=ccrs.PlateCarree())
                if len(c5880.allsegs[0]) == 0:
                    print(f"[Warn] 5880 contour absent: init={init_date}, group={group}, date={d.strftime('%Y-%m-%d')}")
                q = _plot_uv850_quiver(ax, u850.longitude, u850.latitude, u850, v850, is_difference=False)
                if q_abs_key is None and q is not None:
                    q_abs_key = q

            if row == 0:
                ax.set_title(f"{d.strftime('%Y-%m-%d')} BJT", fontsize=10)
            if col == 0:
                ax.text(0.02, 0.5, GROUP_LABELS[group], transform=ax.transAxes, va="center", ha="left", rotation=90, fontsize=10, fontweight="bold", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    if cf_abs is not None:
        cax_abs = fig.add_axes([0.12, 0.075, 0.36, 0.018])
        fig.colorbar(cf_abs, cax=cax_abs, orientation="horizontal", label="TP (mm day$^{-1}$)")
    if cf_diff is not None:
        cax_diff = fig.add_axes([0.56, 0.075, 0.32, 0.018])
        fig.colorbar(cf_diff, cax=cax_diff, orientation="horizontal", label="Hot17 − Cold17 TP (mm day$^{-1}$)")

    if q_abs_key is not None:
        axs[0, 0].quiverkey(q_abs_key, X=0.08, Y=-0.11, U=5.0, label="5 m s$^{-1}$", labelpos="E", coordinates="axes")
    if q_diff_key is not None:
        axs[3, 0].quiverkey(q_diff_key, X=0.08, Y=-0.11, U=2.0, label="2 m s$^{-1}$", labelpos="E", coordinates="axes")

    fig.suptitle(f"S2S Stage-II hot/cold daily circulation | init {init_date}", fontsize=15, y=0.96)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.92, bottom=0.12, wspace=0.06, hspace=0.12)
    _save_fig(fig, f"figure_s2s_stageII_hotcold_daily_circulation_{init_date}")


# ====================================================
# QC
# ====================================================

def _area_mean(da: xr.DataArray, box: Tuple[float, float, float, float] = NCHN_BOX) -> float:
    sub = _sel_region(da, box)
    lat_weights = np.cos(np.deg2rad(sub.latitude))
    dims = [d for d in ("latitude", "longitude") if d in sub.dims]
    return float(sub.weighted(lat_weights).mean(dims, skipna=True)) if "latitude" in dims else float(sub.mean(dims, skipna=True))


def build_qc_rows(init_date: str, composites: Dict[str, Dict[str, xr.DataArray]]) -> List[dict]:
    rows = []
    for group in GROUP_ORDER:
        comp = composites[group]
        members = comp.get("members")
        for d in TARGET_DATES:
            tp = comp["tp"].sel(bjt_date=d)
            z500 = comp["z500"].sel(bjt_date=d)
            u850 = comp["u850"].sel(bjt_date=d)
            v850 = comp["v850"].sel(bjt_date=d)
            uvspd = np.sqrt(u850 ** 2 + v850 ** 2)
            rows.append({
                "init_date": init_date,
                "date_bjt": d.strftime("%Y-%m-%d"),
                "group": group,
                "n_members": len(members) if members is not None else np.nan,
                "members": ",".join(str(m) for m in members) if members is not None else "",
                "tp_nchn_mean": _area_mean(tp),
                "tp_domain_max": float(tp.max(skipna=True)),
                "z500_nchn_mean": _area_mean(z500),
                "u850_nchn_mean": _area_mean(u850),
                "v850_nchn_mean": _area_mean(v850),
                "uv850_speed_nchn_mean": _area_mean(uvspd),
            })
    return rows


def run_stageII_hotcold_daily_circulation(make_plots: bool = True):
    group_df = load_stageII_hotcold_groups(GROUP_FILE)
    audit = build_group_audit_table(group_df)
    audit_path = os.path.join(OUT_DIR, "stageII_hotcold_group_member_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(f"[Audit] saved: {audit_path}")

    all_qc_rows = []
    for init_date in INIT_DATES:
        fields = build_daily_realtime_member_fields(init_date)
        composites = build_group_composites_for_init(init_date, group_df, fields)
        all_qc_rows.extend(build_qc_rows(init_date, composites))
        if make_plots:
            plot_stageII_daily_circulation(init_date, composites)

    qc_path = os.path.join(OUT_DIR, "stageII_hotcold_daily_circulation_qc.csv")
    pd.DataFrame(all_qc_rows).to_csv(qc_path, index=False)
    print(f"[QC] saved: {qc_path}")


if __name__ == "__main__":
    run_stageII_hotcold_daily_circulation(make_plots=True)
