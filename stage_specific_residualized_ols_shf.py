# %%
# =============================================================================
# Cell 1: Read S2S/ERA5 data and write table_stage_member_raw_factors_SHF.csv
# =============================================================================
import os
import glob
import warnings
import builtins
import gc
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Four-initialization test-pipeline configuration. All generated outputs are isolated under v12.
FULL_INIT_LIST = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]
MAIN_INIT = "2023-06-12"
CONTRAST_INIT = "2023-06-08"
AUX_INITS = ["2023-06-05", "2023-06-01"]
PAPER_MLR_STAGES = ["Stage-I_dry", "Total"]
EXPECTED_PAPER_COMBINATIONS = len(FULL_INIT_LIST) * len(PAPER_MLR_STAGES)
EXP_OUT_DIR = "/data1/huangy/fig6/NC/v12"
FOUR_INIT_ROOT = f"{EXP_OUT_DIR}/four_init_full_pipeline_NCVIconcurrent"

def assert_four_init_output(path):
    root = os.path.abspath(FOUR_INIT_ROOT)
    target = os.path.abspath(path)
    if os.path.commonpath([root, target]) != root:
        raise RuntimeError(f"Four-init output is outside FOUR_INIT_ROOT: {path}")

_OUTPUT_PATH_AUDIT_ROWS = []
CURRENT_CELL_NAME = "global"

try:
    _PIPELINE_SOURCE = os.path.abspath(__file__)
except NameError:
    _PIPELINE_SOURCE = None

_PIPELINE_SOURCE_LINES = []

if not hasattr(pd.DataFrame, "_four_init_original_to_csv"):
    pd.DataFrame._four_init_original_to_csv = pd.DataFrame.to_csv
if not hasattr(pd, "_four_init_original_read_csv"):
    pd._four_init_original_read_csv = pd.read_csv
if not hasattr(plt.Figure, "_four_init_original_savefig"):
    plt.Figure._four_init_original_savefig = plt.Figure.savefig
if not hasattr(builtins, "_four_init_original_open"):
    builtins._four_init_original_open = builtins.open
if not hasattr(os, "_four_init_original_makedirs"):
    os._four_init_original_makedirs = os.makedirs
if not hasattr(os, "_four_init_original_remove"):
    os._four_init_original_remove = os.remove
if not hasattr(glob, "_four_init_original_glob"):
    glob._four_init_original_glob = glob.glob

_ORIGINAL_TO_CSV = pd.DataFrame._four_init_original_to_csv
_ORIGINAL_READ_CSV = pd._four_init_original_read_csv
_ORIGINAL_SAVEFIG = plt.Figure._four_init_original_savefig
_ORIGINAL_OPEN = builtins._four_init_original_open
_ORIGINAL_MAKEDIRS = os._four_init_original_makedirs
_ORIGINAL_REMOVE = os._four_init_original_remove
_ORIGINAL_GLOB = glob._four_init_original_glob

if _PIPELINE_SOURCE is not None and os.path.exists(_PIPELINE_SOURCE):
    with _ORIGINAL_OPEN(_PIPELINE_SOURCE, "r", encoding="utf-8") as _source_file:
        _PIPELINE_SOURCE_LINES = _source_file.readlines()


def set_audit_cell_name(name):
    global CURRENT_CELL_NAME
    CURRENT_CELL_NAME = str(name)


def _output_audit_cell_name():
    return CURRENT_CELL_NAME


def _is_allowed_old_read(path):
    path_abs = os.path.abspath(os.fspath(path))
    old_roots = [f"/data1/huangy/fig6/NC/v{version}" for version in [7, 8, 9, 10, 11]]
    return any(path_abs == root or path_abs.startswith(root + os.sep) for root in old_roots)


def _record_output_path(operation, path, is_write_operation):
    path_str = os.fspath(path)
    root_abs = os.path.abspath(FOUR_INIT_ROOT)
    path_abs = os.path.abspath(path_str)
    under_root = os.path.commonpath([root_abs, path_abs]) == root_abs
    allowed_old_read = (not is_write_operation) and _is_allowed_old_read(path_abs)
    status = "ok" if (under_root or allowed_old_read or not is_write_operation) else "outside_four_init_root"
    _OUTPUT_PATH_AUDIT_ROWS.append({
        "cell_name": _output_audit_cell_name(),
        "operation": operation,
        "path": path_str,
        "is_write_operation": bool(is_write_operation),
        "under_four_init_root": bool(under_root),
        "allowed_old_read": bool(allowed_old_read),
        "status": status,
    })
    if is_write_operation:
        assert_four_init_output(path_str)
    return path


def _audited_to_csv(self, path_or_buf=None, *args, **kwargs):
    if isinstance(path_or_buf, (str, os.PathLike)):
        _record_output_path("to_csv", path_or_buf, True)
    return _ORIGINAL_TO_CSV(self, path_or_buf, *args, **kwargs)


def _audited_read_csv(filepath_or_buffer, *args, **kwargs):
    if isinstance(filepath_or_buffer, (str, os.PathLike)):
        _record_output_path("read_csv", filepath_or_buffer, False)
    return _ORIGINAL_READ_CSV(filepath_or_buffer, *args, **kwargs)


def _audited_savefig(self, fname, *args, **kwargs):
    if isinstance(fname, (str, os.PathLike)):
        _record_output_path("savefig", fname, True)
    return _ORIGINAL_SAVEFIG(self, fname, *args, **kwargs)


def _audited_open(file, mode="r", *args, **kwargs):
    if isinstance(file, (str, os.PathLike)) and any(flag in mode for flag in ("w", "a", "x", "+")):
        _record_output_path("open_write", file, True)
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)


def _audited_makedirs(name, *args, **kwargs):
    if isinstance(name, (str, os.PathLike)):
        _record_output_path("os.makedirs", name, True)
    return _ORIGINAL_MAKEDIRS(name, *args, **kwargs)


def _audited_remove(path, *args, **kwargs):
    if isinstance(path, (str, os.PathLike)):
        _record_output_path("os.remove", path, True)
    return _ORIGINAL_REMOVE(path, *args, **kwargs)


def _audited_glob(pathname, *args, **kwargs):
    if isinstance(pathname, (str, os.PathLike)):
        _record_output_path("glob_output_check_or_input_read", pathname, False)
    return _ORIGINAL_GLOB(pathname, *args, **kwargs)


pd.DataFrame.to_csv = _audited_to_csv
pd.read_csv = _audited_read_csv
plt.Figure.savefig = _audited_savefig
builtins.open = _audited_open
os.makedirs = _audited_makedirs
os.remove = _audited_remove
glob.glob = _audited_glob

set_audit_cell_name("Cell 1: Read S2S/ERA5 data and write raw SHF factor tables")

print("\n" + "=" * 80)
print("Cell 1: build raw SHF factor table from S2S/ERA5 inputs...")
print("=" * 80)

EXP_OUT_DIR = "/data1/huangy/fig6/NC/v12"
FOUR_INIT_ROOT = f"{EXP_OUT_DIR}/four_init_full_pipeline_NCVIconcurrent"
RESID_OLS_DIR = FOUR_INIT_ROOT
SUB_DIRS = {
    "tables": f"{RESID_OLS_DIR}/tables",
    "corr_raw": f"{RESID_OLS_DIR}/figures/corr_raw",
    "corr_resid": f"{RESID_OLS_DIR}/figures/corr_resid",
    "lmg_main": f"{RESID_OLS_DIR}/figures/lmg_main",
    "lmg_sensitivity": f"{RESID_OLS_DIR}/figures/lmg_sensitivity",
    "ols_lmg_combo": f"{RESID_OLS_DIR}/figures/ols_lmg_combo",
    "tmax_timeseries": f"{RESID_OLS_DIR}/figures/tmax_timeseries",
    "metrics_heatmap": f"{RESID_OLS_DIR}/figures/metrics_heatmap",
}
def assert_output_under_v7(path):
    assert_four_init_output(path)

assert_output_under_v7(RESID_OLS_DIR)
for d in SUB_DIRS.values():
    assert_output_under_v7(d)
for d in SUB_DIRS.values():
    os.makedirs(d, exist_ok=True)

start_date_list_run = FULL_INIT_LIST
study_window_dict = {
    "Stage-I_dry": ("2023-06-14", "2023-06-17"),
    "Stage-II_wet": ("2023-06-18", "2023-06-24"),
    "Total": ("2023-06-14", "2023-06-24"),
}
study_window_dict_run = study_window_dict

DEBUG_MODE = False
factor_cols_raw = [
    "WNPSH", "Initial_MSEstar_max_NCHN", "NCVI", "NCHN_tp",
    "sm_avg", "SSR_Avg", "SHF_Avg", "z500_anom_NCHN", "Initial_Barrier_NCHN",
]

ERA5_TMAX_FILE = "/data1/huangy/fig6/NC/ai_test/pangu/era5_T/era5_2m_temperature_6_2023.nc"
CF_DIR = "/data1/huangy/fig6/NC/cf_2023-6"
PF_DIR = "/data1/huangy/fig6/NC/2023-06"
MSE_DIR = "/data1/huangy/fig6/NC/MSE"
OROG_FILE = f"{MSE_DIR}/ecmf_cf_orog_2023-06.grib"
NCHN_BOX = (35, 44, 114, 119)
WNPSH_BOX = (15, 25, 115, 150)
Z500_REFORECAST_CANDIDATES = [
    "/data1/huangy/fig6/NC/pl_z03_22/merged_{exp_start_date}.nc",
    "/data6/bjlu/s2s/ecmwf/reforecast/pf/2023-06/pl_z03_22/merged_{exp_start_date}.nc",
    f"{MSE_DIR}/Antecedent/merged_{{exp_start_date}}.nc",
    f"{MSE_DIR}/Antecedent/z500/merged_{{exp_start_date}}.nc",
]
EXPECTED_RAW_ROWS = len(start_date_list_run) * len(study_window_dict_run) * 51
EXPECTED_OLS_COMBINATIONS = len(start_date_list_run) * len(study_window_dict_run)

FILE_CACHE = {}

def clear_file_cache():
    for cached in FILE_CACHE.values():
        try:
            if cached is not None:
                cached.close()
        except Exception:
            pass
    FILE_CACHE.clear()

def get_ds(path):
    if path in FILE_CACHE:
        return FILE_CACHE[path]
    ext = str(path).lower()
    if DEBUG_MODE and ext.endswith((".grb", ".grib")):
        print(f"  [DEBUG get_ds] opening GRIB: {path}")
    if not os.path.exists(path):
        print(f"  [Error] 文件不存在: {path}")
        return None
    try:
        if ext.endswith((".grb", ".grib")):
            FILE_CACHE[path] = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        else:
            FILE_CACHE[path] = xr.open_dataset(path)
    except Exception as e:
        print(f"  [Critical Error] 文件物理损坏或读取失败: {path}\n    具体原因: {e}")
        FILE_CACHE[path] = None
    return FILE_CACHE[path]

def _detect_lat_dim(obj):
    for dim in ["latitude", "lat"]:
        if dim in obj.dims or dim in obj.coords:
            return dim
    return "latitude"

def _detect_lon_dim(obj):
    for dim in ["longitude", "lon"]:
        if dim in obj.dims or dim in obj.coords:
            return dim
    return "longitude"

def _detect_level_dim(obj):
    for dim in ["pressure_level", "level", "isobaricInhPa"]:
        if dim in obj.dims:
            return dim
    return None

def _has_level_dim(obj):
    return _detect_level_dim(obj) is not None

def _get_var(obj, candidates):
    if isinstance(obj, xr.DataArray):
        return obj
    if obj is None:
        raise ValueError("试图从空(None)数据中提取变量。")
    for var in candidates:
        if var in obj.data_vars:
            return obj[var]
    return obj[list(obj.data_vars)[0]]

def load_s2s_ensemble_with_cf(pf_path, cf_path, var_candidates, level=None):
    pf_ds = get_ds(pf_path)
    cf_ds = get_ds(cf_path)
    if pf_ds is None or cf_ds is None:
        return None
    pf_da = _get_var(pf_ds, var_candidates)
    cf_da = _get_var(cf_ds, var_candidates)
    if level is not None:
        if _has_level_dim(pf_da):
            pf_da = pf_da.sel({_detect_level_dim(pf_da): level}, method="nearest")
        if _has_level_dim(cf_da):
            cf_da = cf_da.sel({_detect_level_dim(cf_da): level}, method="nearest")
    if "number" not in cf_da.dims:
        cf_da = cf_da.expand_dims(number=[0])
    else:
        cf_da = cf_da.assign_coords(number=[0])
    try:
        return xr.concat([cf_da, pf_da], dim="number")
    except Exception:
        return pf_da

def _cell1_memory_debug(message):
    try:
        import psutil
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        print(f"{message} RSS={rss_gb:.2f} GB")
    except Exception:
        print(message)

def _open_current_init_variable(path, var_candidates, exp_start_date, level=None, region=None):
    if not os.path.exists(path):
        print(f"  [Cell1] missing S2S file: {path}")
        return None
    ext = str(path).lower()
    filter_candidates = []
    if ext.endswith((".grb", ".grib")):
        for short_name in var_candidates:
            filt = {"shortName": short_name}
            if level is not None:
                filt.update({"typeOfLevel": "isobaricInhPa", "level": int(level)})
            filter_candidates.append(filt)
        if level is not None:
            filter_candidates.append({"typeOfLevel": "isobaricInhPa", "level": int(level)})
        filter_candidates.append(None)
    else:
        filter_candidates = [None]
    last_error = None
    for filter_by_keys in filter_candidates:
        ds = None
        try:
            if ext.endswith((".grb", ".grib")):
                backend_kwargs = {"indexpath": ""}
                if filter_by_keys:
                    backend_kwargs["filter_by_keys"] = filter_by_keys
                ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)
            else:
                ds = xr.open_dataset(path)
            da = _get_var(ds, var_candidates)
            if level is not None and _has_level_dim(da):
                da = da.sel({_detect_level_dim(da): level}, method="nearest")
            da = get_s2s_init_slice(da.to_dataset(name="_cell1_var"), "_cell1_var", exp_start_date)
            if da is None:
                return None
            if region is not None:
                da = safe_slice_region(da, *region)
            return da.load()
        except Exception as exc:
            last_error = exc
        finally:
            if ds is not None:
                ds.close()
    print(f"  [Cell1] failed to open current-init variable: path={path}; init={exp_start_date}; error={last_error}")
    return None

def load_s2s_current_init_with_cf(pf_path, cf_path, var_candidates, exp_start_date, level=None, region=None):
    pf_da = _open_current_init_variable(pf_path, var_candidates, exp_start_date, level=level, region=region)
    cf_da = _open_current_init_variable(cf_path, var_candidates, exp_start_date, level=level, region=region)
    if pf_da is None or cf_da is None:
        del pf_da, cf_da
        gc.collect()
        return None
    if "number" not in cf_da.dims:
        cf_da = cf_da.expand_dims(number=[0])
    else:
        cf_da = cf_da.assign_coords(number=[0])
    try:
        combined = xr.concat([cf_da, pf_da], dim="number").load()
    except Exception:
        combined = pf_da.load()
    del pf_da, cf_da
    gc.collect()
    return combined

def safe_slice_region(da, lat_min, lat_max, lon_min, lon_max):
    lat_dim = _detect_lat_dim(da)
    lon_dim = _detect_lon_dim(da)
    lat_slice = slice(lat_max, lat_min) if float(da[lat_dim][0]) > float(da[lat_dim][-1]) else slice(lat_min, lat_max)
    return da.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})

def load_era5_tmax_daily(filepath):
    print(f"  [ERA5 Tmax] file path: {filepath}")
    if not os.path.exists(filepath):
        print("  [ERA5 Tmax] file missing; daily Tmax unavailable.")
        return None
    try:
        ds = xr.open_dataset(filepath)
        da = _get_var(ds, ["t2m", "2t", "2m_temperature"])
        rn = {d: ("latitude" if "lat" in str(d).lower() else "longitude") for d in da.dims if "lat" in str(d).lower() or "lon" in str(d).lower()}
        if rn:
            da = da.rename(rn)
        da = safe_slice_region(da, *NCHN_BOX)
        raw_units = str(da.attrs.get("units", "")).lower().strip()
        finite_vals = da.values[np.isfinite(da.values)]
        raw_median = float(np.nanmedian(finite_vals)) if finite_vals.size > 0 else np.nan
        if raw_units in ["k", "kelvin"] or (not np.isnan(raw_median) and raw_median > 150):
            da = da - 273.15
            da.attrs["units"] = "degC"
        tc = next((c for c in ["time", "valid_time", "date"] if c in da.coords), None)
        if tc is None:
            return None
        if tc not in da.dims:
            da = da.swap_dims({da[tc].dims[0]: tc})
        if tc != "time":
            da = da.rename({tc: "time"})
        da = da.assign_coords(time=pd.to_datetime(da["time"].values) + pd.Timedelta(hours=8))
        print("  [ERA5 Tmax] UTC+8h conversion completed: True")
        da_daily = da.resample(time="1D").max()
        da_daily_mean = da_daily.mean(dim=[_detect_lat_dim(da_daily), _detect_lon_dim(da_daily)])
        daily_mean = float(da_daily_mean.mean().values)
        daily_min = float(da_daily_mean.min().values)
        daily_max = float(da_daily_mean.max().values)
        date_min = pd.to_datetime(da_daily_mean.time.values).min().strftime("%Y-%m-%d")
        date_max = pd.to_datetime(da_daily_mean.time.values).max().strftime("%Y-%m-%d")
        print(f"  [ERA5 Tmax] daily Tmax date range: {date_min} to {date_max}")
        print(f"  [ERA5 Tmax] daily Tmax min/max/mean: {daily_min:.2f}/{daily_max:.2f}/{daily_mean:.2f} °C")
        if daily_mean > 100:
            raise ValueError(f"ERA5 Tmax daily mean > 100 ({daily_mean:.2f}); unit conversion likely failed.")
        return da_daily_mean
    except Exception as e:
        print(f"  [Warning] ERA5 Tmax load failed: {e}")
        return None

def get_s2s_init_slice(ds, var_name, exp_start_date):
    tc = next((c for c in ["time", "valid_time", "forecast_reference_time", "date"] if c in ds.coords), None)
    if not tc:
        return None
    dates_in_ds = pd.to_datetime(ds[tc].values).date
    target_date = pd.Timestamp(exp_start_date).date()
    if target_date not in dates_in_ds:
        return None
    target_idx = np.where(dates_in_ds == target_date)[0][0]
    actual_dim = ds[tc].dims[0]
    da_init = ds[var_name].isel({actual_dim: target_idx})
    rn = {d: ("latitude" if "lat" in str(d).lower() else "longitude") for d in da_init.dims if "lat" in str(d).lower() or "lon" in str(d).lower()}
    if rn:
        da_init = da_init.rename(rn)
    return da_init

def align_to_lead_day(da):
    if da is None:
        return None
    step_hours = pd.to_timedelta(da.step.values).total_seconds() / 3600.0
    lead_float = step_hours / 24.0
    if not np.allclose(lead_float, np.round(lead_float), atol=1e-6):
        print(f"  [Warning] align_to_lead_day: non-24h-multiple steps detected: {step_hours[:10]}")
    lead_days = np.round(lead_float).astype(int)
    da_ld = da.assign_coords(lead_day=("step", lead_days)).swap_dims({"step": "lead_day"})
    if len(da_ld.lead_day) != len(np.unique(da_ld.lead_day)):
        da_ld = da_ld.groupby("lead_day").mean(dim="lead_day")
    valid_lds = [ld for ld in [1, 2, 3] if ld in da_ld.lead_day.values]
    if not valid_lds:
        return None
    return da_ld.sel(lead_day=valid_lds)

def align_common_lead_day(arrays):
    valid_arrays = [a for a in arrays if a is not None]
    if not valid_arrays:
        return [None] * len(arrays)
    common_lds = set(valid_arrays[0].lead_day.values)
    for a in valid_arrays[1:]:
        common_lds = common_lds.intersection(set(a.lead_day.values))
    common_lds = sorted(list(common_lds))
    if not common_lds:
        return [None] * len(arrays)
    return [a.sel(lead_day=common_lds) if a is not None else None for a in arrays]

def process_s2s_tmax_bjt_daily_max(da_init, region=None):
    if da_init is None:
        return None
    tc = next((c for c in ["time", "valid_time", "forecast_reference_time", "date"] if c in da_init.coords), None)
    bjt_time = pd.to_datetime(da_init[tc].values) + da_init.step.values + pd.Timedelta(hours=8)
    da = da_init.assign_coords(valid_bjt=("step", bjt_time))
    if tc in da.coords and tc not in da.dims:
        da = da.drop_vars(tc)
    da = da.swap_dims({"step": "valid_bjt"}).rename({"valid_bjt": "time"}).sortby("time")
    raw_units = str(da.attrs.get("units", "")).lower().strip()
    finite_vals = da.values[np.isfinite(da.values)]
    raw_median = float(np.nanmedian(finite_vals)) if finite_vals.size > 0 else np.nan
    if raw_units in ["k", "kelvin"] or (not np.isnan(raw_median) and raw_median > 150):
        da = da - 273.15
        da.attrs["units"] = "degC"
    da_daily = da.resample(time="1D").max()
    if region:
        da_daily = safe_slice_region(da_daily, *region)
    lat_dim = _detect_lat_dim(da_daily)
    lon_dim = _detect_lon_dim(da_daily)
    da_daily_mean = da_daily.mean(dim=[lat_dim, lon_dim])
    if float(np.nanmean(da_daily_mean.values)) > 100:
        raise ValueError("S2S Tmax still appears to be Kelvin.")
    return da_daily_mean

def _apportion_accum_to_calendar_days(da_init, force_non_negative=False):
    other_dims = [d for d in da_init.dims if d != "time"]
    da = da_init.transpose("time", *other_dims).sortby("time")
    times = pd.to_datetime(da.time.values)
    vals = da.values
    daily_accum = {}
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        inc_val = vals[i] - vals[i - 1]
        if force_non_negative:
            inc_val = np.maximum(inc_val, 0.0)
        total_duration = (t1 - t0).total_seconds()
        if total_duration <= 0:
            continue
        d0, d1 = t0.date(), t1.date()
        if d0 == d1 or (t1.hour == 0 and t1.minute == 0 and t1.second == 0 and t1.microsecond == 0 and d1 == d0 + pd.Timedelta(days=1)):
            date_str = d0.strftime("%Y-%m-%d")
            daily_accum.setdefault(date_str, np.zeros_like(inc_val))
            daily_accum[date_str] += inc_val
        else:
            current_t = t0
            while current_t < t1:
                next_midnight = pd.Timestamp(current_t.date() + pd.Timedelta(days=1))
                segment_end = min(next_midnight, t1)
                segment_duration = (segment_end - current_t).total_seconds()
                ratio = segment_duration / total_duration
                date_str = current_t.date().strftime("%Y-%m-%d")
                daily_accum.setdefault(date_str, np.zeros_like(inc_val))
                daily_accum[date_str] += inc_val * ratio
                current_t = segment_end
    dates = sorted(list(daily_accum.keys()))
    if not dates:
        return xr.DataArray(np.nan, dims=["time"], coords={"time": [pd.Timestamp(times[0].date())]})
    time_coords = pd.to_datetime(dates)
    stacked_vals = np.stack([daily_accum[d] for d in dates], axis=0)
    new_coords = {"time": time_coords}
    for dim in other_dims:
        new_coords[dim] = da.coords[dim]
    return xr.DataArray(stacked_vals, dims=["time"] + other_dims, coords=new_coords)

def process_accum_variable(da_init, var_type, region=None):
    if da_init is None:
        return None
    tc = next((c for c in ["time", "valid_time", "forecast_reference_time", "date"] if c in da_init.coords), None)
    bjt_time = pd.to_datetime(da_init[tc].values) + da_init.step.values + pd.Timedelta(hours=8)
    da_accum = da_init.assign_coords(valid_bjt=("step", bjt_time))
    if tc in da_accum.coords and tc not in da_accum.dims:
        da_accum = da_accum.drop_vars(tc)
    da_accum = da_accum.swap_dims({"step": "valid_bjt"}).rename({"valid_bjt": "time"}).sortby("time")
    if var_type == "precip":
        da_daily = _apportion_accum_to_calendar_days(da_accum, force_non_negative=True)
        if str(da_init.attrs.get("units", "")).lower().strip() in ["m", "meter", "meters"]:
            da_daily = da_daily * 1000.0
    elif var_type == "flux_ssr":
        da_daily = _apportion_accum_to_calendar_days(da_accum, force_non_negative=False) / 86400.0
    elif var_type == "flux_shf":
        da_daily = _apportion_accum_to_calendar_days(da_accum, force_non_negative=False) / 86400.0
        # S2S SHF is downward-positive by default; SHF_Avg is stored as upward-positive.
        da_daily = -1.0 * da_daily
    else:
        raise ValueError(f"Unsupported accumulated variable type: {var_type}")
    if region:
        da_daily = safe_slice_region(da_daily, *region)
    lat_dim = _detect_lat_dim(da_daily)
    lon_dim = _detect_lon_dim(da_daily)
    return da_daily.mean(dim=[lat_dim, lon_dim])

def process_s2s_bjt_daily_mean(da_init, var_type, region=None):
    if da_init is None:
        return None
    tc = next((c for c in ["time", "valid_time", "forecast_reference_time", "date"] if c in da_init.coords), None)
    bjt_time = pd.to_datetime(da_init[tc].values) + da_init.step.values + pd.Timedelta(hours=8)
    da = da_init.assign_coords(valid_bjt=("step", bjt_time))
    if tc in da.coords and tc not in da.dims:
        da = da.drop_vars(tc)
    da = da.swap_dims({"step": "valid_bjt"}).rename({"valid_bjt": "time"}).sortby("time")
    if var_type == "soil_moisture":
        raw_units = str(da.attrs.get("units", "")).lower().replace("**", "^").replace(" ", "")
        if raw_units in ["kgm-3", "kgm^-3", "kg/m3", "kg/m^3"]:
            da = da / 1000.0
    elif var_type in ["z500", "wnpsh"]:
        if float(da.mean()) > 10000:
            da = da / 9.80665
    da_daily = da.resample(time="1D").mean()
    if region:
        da_daily = safe_slice_region(da_daily, *region)
    lat_dim = _detect_lat_dim(da_daily)
    lon_dim = _detect_lon_dim(da_daily)
    return da_daily.mean(dim=[lat_dim, lon_dim])

def extract_window_only(da_daily, w_start, w_end, agg_func="mean"):
    if da_daily is None:
        return None
    ws, we = pd.Timestamp(w_start).date(), pd.Timestamp(w_end).date()
    vd = pd.to_datetime(da_daily.time.values).date
    idx = [i for i, d in enumerate(vd) if ws <= d <= we]
    if not idx:
        return None
    da_win = da_daily.isel(time=idx)
    return da_win.sum(dim="time").values if agg_func == "sum" else da_win.mean(dim="time").values

def load_z500_reforecast_climatology(exp_start_date, region=NCHN_BOX):
    z500_hc_path = None
    for template in Z500_REFORECAST_CANDIDATES:
        path = template.format(exp_start_date=exp_start_date)
        if os.path.exists(path) and z500_hc_path is None:
            z500_hc_path = path
    if z500_hc_path is None:
        raise FileNotFoundError(f"Z500 reforecast missing for {exp_start_date}")
    ds = xr.open_dataset(z500_hc_path)
    da = _get_var(ds, ["gh", "z", "geopotential"])
    lev_dim = _detect_level_dim(da)
    if lev_dim:
        da = da.sel({lev_dim: 500}, method="nearest")
    rn = {}
    for d in da.dims:
        dl = str(d).lower()
        if "lat" in dl and dl != "latitude":
            rn[d] = "latitude"
        elif "lon" in dl and dl != "longitude":
            rn[d] = "longitude"
    if rn:
        da = da.rename(rn)
    if "time" in da.dims:
        da = da.rename({"time": "hc_time"})
    if float(da.mean()) > 10000:
        da = da / 9.80665
        da.attrs["units"] = "gpm"
    bjt_time = pd.Timestamp(exp_start_date) + da.step.values + pd.Timedelta(hours=8)
    da = da.assign_coords(valid_bjt=("step", bjt_time))
    da = da.swap_dims({"step": "valid_bjt"}).rename({"valid_bjt": "time"}).sortby("time")
    da_daily = da.resample(time="1D").mean()
    da_daily = safe_slice_region(da_daily, region[0], region[1], region[2], region[3])
    da_daily = da_daily.mean(dim=[_detect_lat_dim(da_daily), _detect_lon_dim(da_daily)], skipna=True)
    drop_dims = [d for d in da_daily.dims if d != "time"]
    result = da_daily.mean(dim=drop_dims, skipna=True) if drop_dims else da_daily
    result = result.load()
    ds.close()
    return result

def apply_z500_anomaly(da_z500_rt_daily, z500_clim):
    if da_z500_rt_daily is None or z500_clim is None:
        return None
    rt_aligned, clim_aligned = xr.align(da_z500_rt_daily, z500_clim, join="inner")
    da_z500_anom = rt_aligned - clim_aligned
    if "number" not in da_z500_anom.dims or da_z500_anom.sizes["number"] != 51:
        raise ValueError("Z500 anomaly missing 'number' dim or size != 51.")
    return da_z500_anom

cp, Lv, g, Rd = 1005.0, 2.5e6, 9.81, 287.05

def _qsat(T_K, p_hPa):
    t_c = T_K - 273.15
    es_Pa = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5)) * 100.0
    return (0.622 * es_Pa) / (p_hPa * 100.0 - 0.378 * es_Pa)

def _lcl_pressure(T2m_K, Td2m_K, sp_Pa):
    lcl_T = 56.0 + 1.0 / (1.0 / (Td2m_K - 56.0) + np.log(T2m_K / Td2m_K) / 800.0)
    return (sp_Pa * (lcl_T / T2m_K) ** (cp / Rd)) / 100.0

def compute_real_ncvi(exp_start_date):
    da_rt = load_s2s_current_init_with_cf(
        f"{MSE_DIR}/ecmf_pf_60_2023-06.grib",
        f"{CF_DIR}/ecmf_cf_60_2023-06.grib",
        ["pv"],
        exp_start_date,
        region=(35, 50, 115, 130),
    )
    if da_rt is None:
        return None
    da_rt_mean = da_rt.mean(dim=[_detect_lat_dim(da_rt), _detect_lon_dim(da_rt)])
    da_rt_ld = align_to_lead_day(da_rt_mean)
    date_dash = exp_start_date
    date_nodash = exp_start_date.replace("-", "")
    hc_pf_candidates = [
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_pt320_60_{date_dash}.grib",
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_pt320_{date_dash}.grib",
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_60_{date_dash}.grib",
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_{date_dash}.grib",
    ]
    hc_cf_candidates = [
        f"{CF_DIR}/ecmf_cf_NCVI_pt320_60_{date_dash}.grib",
        f"{CF_DIR}/ecmf_cf_NCVI_pt320_{date_dash}.grib",
        f"{CF_DIR}/ecmf_cf_NCVI_60_{date_dash}.grib",
        f"{CF_DIR}/ecmf_cf_NCVI_{date_dash}.grib",
    ]
    hc_pf_candidates += sorted(set(glob.glob(f"{MSE_DIR}/Antecedent/**/*NCVI*{date_dash}*.grib", recursive=True) + glob.glob(f"{MSE_DIR}/Antecedent/**/*NCVI*{date_nodash}*.grib", recursive=True)))
    hc_cf_candidates += sorted(set(glob.glob(f"{CF_DIR}/**/*NCVI*{date_dash}*.grib", recursive=True) + glob.glob(f"{CF_DIR}/**/*NCVI*{date_nodash}*.grib", recursive=True) + glob.glob(f"{CF_DIR}/**/*ncvi*{date_dash}*.grib", recursive=True) + glob.glob(f"{CF_DIR}/**/*ncvi*{date_nodash}*.grib", recursive=True)))
    hc_pf = next((p for p in hc_pf_candidates if os.path.exists(p)), None)
    hc_cf = next((p for p in hc_cf_candidates if os.path.exists(p)), None)
    da_anomaly = da_rt_ld
    if hc_pf and hc_cf:
        da_hc = load_s2s_current_init_with_cf(
            hc_pf,
            hc_cf,
            ["pv"],
            exp_start_date,
            region=(35, 50, 115, 130),
        )
        if da_hc is not None:
            da_hc_mean = da_hc.mean(dim=["latitude", "longitude"])
            da_hc_ld = align_to_lead_day(da_hc_mean)
            if da_hc_ld is not None:
                drop_dims = [d for d in da_hc_ld.dims if d != "lead_day"]
                da_hc_clim = da_hc_ld.mean(dim=drop_dims, skipna=True) if drop_dims else da_hc_ld
                aligned = align_common_lead_day([da_rt_ld, da_hc_clim])
                if aligned[0] is not None:
                    da_anomaly = aligned[0] - aligned[1]
    if da_anomaly is not None:
        ncvi_val = np.asarray(da_anomaly.mean(dim="lead_day").values).squeeze() * 1e6
        if np.shape(ncvi_val) != (51,):
            return None
        return ncvi_val
    return None

def compute_real_mse(exp_start_date):
    t2m = load_s2s_current_init_with_cf(
        f"{MSE_DIR}/ecmf_pf_sfc_2t_2023-06.grib", f"{CF_DIR}/ecmf_cf_sfc_2t_2023-06.grib",
        ["2t", "t2m"], exp_start_date, region=NCHN_BOX,
    )
    td2m = load_s2s_current_init_with_cf(
        f"{MSE_DIR}/ecmf_pf_sfc_2td_2023-06.grib", f"{CF_DIR}/ecmf_cf_sfc_2td_2023-06.grib",
        ["2d", "d2m"], exp_start_date, region=NCHN_BOX,
    )
    sp = load_s2s_current_init_with_cf(
        f"{MSE_DIR}/ecmf_pf_sfc_sp_2023-06.grib", f"{CF_DIR}/ecmf_cf_sfc_sp_2023-06.grib",
        ["sp"], exp_start_date, region=NCHN_BOX,
    )
    if any(da is None for da in [t2m, td2m, sp]):
        return None, None
    t2m_ld = align_to_lead_day(t2m)
    td2m_ld = align_to_lead_day(td2m)
    sp_ld = align_to_lead_day(sp)
    t2m_ld, td2m_ld, sp_ld = align_common_lead_day([t2m_ld, td2m_ld, sp_ld])
    if any(da is None for da in [t2m_ld, td2m_ld, sp_ld]):
        return None, None
    orog_ds = get_ds(OROG_FILE)
    if orog_ds is None:
        return None, None
    orog_raw = _get_var(orog_ds, ["z", "orog"])
    rn = {}
    for d in orog_raw.dims:
        dl = str(d).lower()
        if "lat" in dl:
            rn[d] = "latitude"
        elif "lon" in dl:
            rn[d] = "longitude"
    if rn:
        orog_raw = orog_raw.rename(rn)
    orog = safe_slice_region(orog_raw, 35, 44, 114, 119)
    target_grid = t2m_ld.isel(number=0, lead_day=0, drop=True)
    try:
        orog = orog.interp(latitude=target_grid.latitude, longitude=target_grid.longitude, method="nearest")
    except Exception:
        orog = orog.reindex_like(target_grid, method="nearest")
    drop_dims = [d for d in orog.dims if d not in ("latitude", "longitude")]
    if drop_dims:
        orog = orog.mean(dim=drop_dims)
    gz_s_grid = xr.where(orog > 9000, orog, g * orog)
    mse_s_grid = cp * t2m_ld + Lv * _qsat(td2m_ld, sp_ld / 100.0) + gz_s_grid
    lcl_p_grid = _lcl_pressure(t2m_ld, td2m_ld, sp_ld)
    mse_star_levs = []
    for lev in [1000, 925, 850, 700, 500, 300]:
        t_lev = load_s2s_current_init_with_cf(
            f"{MSE_DIR}/ecmf_pl{lev}_130_2023-06.grib", f"{CF_DIR}/ecmf_cf_pl{lev}_130_2023-06.grib",
            ["t"], exp_start_date, level=lev, region=NCHN_BOX,
        )
        z_lev = load_s2s_current_init_with_cf(
            f"{MSE_DIR}/ecmf_pl{lev}_156_2023-06.grib", f"{CF_DIR}/ecmf_cf_pl{lev}_156_2023-06.grib",
            ["z", "gh"], exp_start_date, level=lev, region=NCHN_BOX,
        )
        if t_lev is None or z_lev is None:
            continue
        t_lev_ld = align_to_lead_day(t_lev)
        z_lev_ld = align_to_lead_day(z_lev)
        t_lev_ld, z_lev_ld, _ = align_common_lead_day([t_lev_ld, z_lev_ld, t2m_ld])
        if t_lev_ld is not None and z_lev_ld is not None:
            phi_lev = xr.where(z_lev_ld > 1e4, z_lev_ld, g * z_lev_ld)
            mse_star_levs.append((cp * t_lev_ld + Lv * _qsat(t_lev_ld, float(lev)) + phi_lev).assign_coords(level=lev))
        del t_lev, z_lev, t_lev_ld, z_lev_ld
        gc.collect()
    if not mse_star_levs:
        return None, None
    mse_star_all = xr.concat(mse_star_levs, dim="level")
    mse_star_masked = mse_star_all.where((mse_star_all["level"] > 300) & (mse_star_all["level"] <= lcl_p_grid))
    mse_star_max_grid = mse_star_masked.max(dim="level")
    barrier_grid = mse_star_max_grid - mse_s_grid
    spatial_dims_mse = [d for d in ["latitude", "longitude", "lat", "lon"] if d in mse_star_max_grid.dims]
    spatial_dims_bar = [d for d in ["latitude", "longitude", "lat", "lon"] if d in barrier_grid.dims]
    mse_star_max_mean = mse_star_max_grid.mean(dim=spatial_dims_mse)
    barrier_mean = barrier_grid.mean(dim=spatial_dims_bar)
    mse_val = np.asarray(mse_star_max_mean.mean(dim="lead_day").values).squeeze() / 1000.0
    bar_val = np.asarray(barrier_mean.mean(dim="lead_day").values).squeeze() / 1000.0
    if np.shape(mse_val) != (51,) or np.shape(bar_val) != (51,):
        return None, None
    return mse_val, bar_val

raw_records = []
daily_tmax_records = []
daily_member_raw_records = []

def append_daily_tmax_records(exp_start_date, da_tmax_daily, era5_tmax_daily):
    if da_tmax_daily is None:
        return
    target_dates = pd.date_range("2023-06-14", "2023-06-24", freq="D")
    for day in target_dates:
        date_str = day.strftime("%Y-%m-%d")
        if day not in pd.to_datetime(da_tmax_daily.time.values):
            continue
        s2s_day = da_tmax_daily.sel(time=day)
        era5_val = np.nan
        if era5_tmax_daily is not None and day in pd.to_datetime(era5_tmax_daily.time.values):
            era5_val = float(era5_tmax_daily.sel(time=day).values)
        vals = np.asarray(s2s_day.values).squeeze()
        if vals.shape != (51,):
            continue
        for member, value in enumerate(vals):
            daily_tmax_records.append({
                "init_date": exp_start_date,
                "date": date_str,
                "member": member,
                "s2s_tmax": value,
                "era5_tmax": era5_val,
            })

def get_daily_stage(day):
    day_ts = pd.Timestamp(day)
    for stage in ["Stage-I_dry", "Stage-II_wet"]:
        w_start, w_end = study_window_dict_run[stage]
        if pd.Timestamp(w_start) <= day_ts <= pd.Timestamp(w_end):
            return stage
    return None

def extract_daily_member_values(da_daily, day):
    if da_daily is None:
        return None
    if day not in pd.to_datetime(da_daily.time.values):
        return None
    vals = np.asarray(da_daily.sel(time=day).values).squeeze()
    return vals if vals.shape == (51,) else None

def append_daily_member_raw_factor_records(exp_start_date, init_factors, da_tmax_daily, da_tp_daily, da_sm_daily, da_ssr_daily, da_shf_daily, da_z500_daily, da_wnpsh_daily):
    target_dates = pd.date_range("2023-06-14", "2023-06-24", freq="D")
    for day in target_dates:
        stage = get_daily_stage(day)
        if stage is None:
            continue
        daily_arrays = {
            "y_tmax": extract_daily_member_values(da_tmax_daily, day),
            "NCHN_tp": extract_daily_member_values(da_tp_daily, day),
            "sm_avg": extract_daily_member_values(da_sm_daily, day),
            "SSR_Avg": extract_daily_member_values(da_ssr_daily, day),
            # SHF_Avg is stored as upward-positive; do not flip it again in Cell 2/3.
            "SHF_Avg": extract_daily_member_values(da_shf_daily, day),
            "z500_anom_NCHN": extract_daily_member_values(da_z500_daily, day),
            "WNPSH": extract_daily_member_values(da_wnpsh_daily, day),
        }
        shape_inputs = dict(daily_arrays)
        shape_inputs.update(init_factors)
        bad_shapes = {k: np.shape(v) for k, v in shape_inputs.items() if np.shape(v) != (51,)}
        if bad_shapes:
            print(f"  [Warning] daily member factor shape check failed for {exp_start_date} {day:%Y-%m-%d}: {bad_shapes}; skipped")
            continue
        for member in range(51):
            row = {"init_date": exp_start_date, "date": day.strftime("%Y-%m-%d"), "member": member, "stage": stage}
            for name, vals in daily_arrays.items():
                row[name] = vals[member]
            for name, vals in init_factors.items():
                row[name] = vals[member]
            daily_member_raw_records.append(row)

print("-> Cell 1 uses per-init, per-variable CF+PF loading to limit peak memory.")
era5_tmax_daily = load_era5_tmax_daily(ERA5_TMAX_FILE)

for exp_start_date in start_date_list_run:
    print(f"\n[Processing Start Date: {exp_start_date}]")
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] init start.")
    init_factors = {}
    try:
        _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading MSE* / barrier...")
        max_val, bar_val = compute_real_mse(exp_start_date)
        _cell1_memory_debug(f"[Cell1][{exp_start_date}] MSE* / barrier done.")
        _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading NCVI...")
        ncvi_val = compute_real_ncvi(exp_start_date)
        _cell1_memory_debug(f"[Cell1][{exp_start_date}] NCVI done.")
        init_factors["Initial_MSEstar_max_NCHN"] = max_val
        init_factors["Initial_Barrier_NCHN"] = bar_val
        init_factors["NCVI"] = ncvi_val
    except Exception as e:
        print(f"  [Warning] Initial 因子推算异常: {e}")
    bad_shapes = {k: np.shape(v) for k, v in init_factors.items() if np.shape(v) != (51,)}
    if bad_shapes:
        print(f"  [Warning] 初始因子 shape 不符合要求 (51,)，跳过该起报日: {bad_shapes}")
        clear_file_cache()
        gc.collect()
        continue

    _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading Tmax...")
    tmp_tmax = load_s2s_current_init_with_cf(
        f"{PF_DIR}/ecmf_rel_pf_sfc_mx2t6_2023-06.grb",
        f"{CF_DIR}/ecmf_cf_sfc_mx2t6_2023-06.grib",
        ["mx2t6", "mx2t", "tmax"], exp_start_date, region=NCHN_BOX,
    )
    da_tmax_m = process_s2s_tmax_bjt_daily_max(tmp_tmax, region=NCHN_BOX)
    del tmp_tmax
    gc.collect()
    append_daily_tmax_records(exp_start_date, da_tmax_m, era5_tmax_daily)
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] Tmax done.")

    _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading TP...")
    tmp_tp = load_s2s_current_init_with_cf(
        f"{PF_DIR}/ecmf_rel_pf_sfc_tp2023-06.grb",
        f"{CF_DIR}/ecmf_cf_sfc_tp_2023-06.grib",
        ["tp", "total_precipitation"], exp_start_date, region=NCHN_BOX,
    )
    da_tp_m = process_accum_variable(tmp_tp, "precip", region=NCHN_BOX)
    del tmp_tp
    gc.collect()
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] TP done.")

    _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading SM...")
    tmp_sm = load_s2s_current_init_with_cf(
        f"{PF_DIR}/ecmf_rel_pf_sfc_sm2023-06.grb",
        f"{CF_DIR}/ecmf_cf_sfc_sm20_2023-06.grib",
        ["sm20", "sm", "swvl1", "swvl2", "vwc1"], exp_start_date, region=NCHN_BOX,
    )
    da_sm_m = process_s2s_bjt_daily_mean(tmp_sm, "soil_moisture", region=NCHN_BOX)
    del tmp_sm
    gc.collect()
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] SM done.")

    _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading SSR...")
    tmp_ssr = load_s2s_current_init_with_cf(
        f"{PF_DIR}/ecmf_rel_pf_sfc_radiation2023-06.grb",
        f"{CF_DIR}/ecmf_cf_sfc_sshf+slhf+snsr_2023-06.grib",
        ["snsr", "ssr", "ssrd", "surface_net_solar_radiation"], exp_start_date, region=NCHN_BOX,
    )
    da_ssr_m = process_accum_variable(tmp_ssr, "flux_ssr", region=NCHN_BOX)
    del tmp_ssr
    gc.collect()
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] SSR done.")

    _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading SHF...")
    tmp_sshf = load_s2s_current_init_with_cf(
        f"{PF_DIR}/ecmf_rel_pf_sfc_hflux2023-06.grb",
        f"{CF_DIR}/ecmf_cf_sfc_sshf+slhf+snsr_2023-06.grib",
        ["sshf", "ishf", "surface_sensible_heat_flux"], exp_start_date, region=NCHN_BOX,
    )
    da_sshf_m = process_accum_variable(tmp_sshf, "flux_shf", region=NCHN_BOX)
    del tmp_sshf
    gc.collect()
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] SHF done.")

    _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading Z500...")
    tmp_z500 = load_s2s_current_init_with_cf(
        f"{PF_DIR}/ecmf_rel_pf_pl_zqt2023-06.grb",
        f"{CF_DIR}/ecmf_cf_pl500_156_2023-06.grib",
        ["gh", "z", "geopotential"], exp_start_date, level=500, region=NCHN_BOX,
    )
    da_z500_raw_m = process_s2s_bjt_daily_mean(tmp_z500, "z500", region=NCHN_BOX)
    del tmp_z500
    gc.collect()
    try:
        z500_clim = load_z500_reforecast_climatology(exp_start_date, region=NCHN_BOX)
        da_z500_m = apply_z500_anomaly(da_z500_raw_m, z500_clim)
    except Exception:
        da_z500_m = None
    del da_z500_raw_m
    if "z500_clim" in locals():
        del z500_clim
    gc.collect()
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] Z500 done.")

    _cell1_memory_debug(f"[Cell1][{exp_start_date}] loading WNPSH...")
    tmp_wnpsh = load_s2s_current_init_with_cf(
        f"{PF_DIR}/ecmf_rel_pf_pl_zqt2023-06.grb",
        f"{CF_DIR}/ecmf_cf_pl850_156_2023-06.grib",
        ["gh", "z", "geopotential"], exp_start_date, level=850, region=WNPSH_BOX,
    )
    da_wnpsh_m = process_s2s_bjt_daily_mean(tmp_wnpsh, "wnpsh", region=WNPSH_BOX)
    del tmp_wnpsh
    gc.collect()
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] WNPSH done.")

    append_daily_member_raw_factor_records(exp_start_date, init_factors, da_tmax_m, da_tp_m, da_sm_m, da_ssr_m, da_sshf_m, da_z500_m, da_wnpsh_m)

    for stage, (win_start, win_end) in study_window_dict_run.items():
        try:
            y_tmax = extract_window_only(da_tmax_m, win_start, win_end, agg_func="mean")
            dyn_factors = {
                "NCHN_tp": extract_window_only(da_tp_m, win_start, win_end, agg_func="sum"),
                "sm_avg": extract_window_only(da_sm_m, win_start, win_end, agg_func="mean"),
                "SSR_Avg": extract_window_only(da_ssr_m, win_start, win_end, agg_func="mean"),
                "SHF_Avg": extract_window_only(da_sshf_m, win_start, win_end, agg_func="mean"),
                "z500_anom_NCHN": extract_window_only(da_z500_m, win_start, win_end, agg_func="mean"),
                "WNPSH": extract_window_only(da_wnpsh_m, win_start, win_end, agg_func="mean"),
            }
        except Exception:
            continue
        shapes_to_check = {"y_tmax": y_tmax}
        shapes_to_check.update(dyn_factors)
        shapes_to_check.update(init_factors)
        bad_shapes = {k: np.shape(v) for k, v in shapes_to_check.items() if np.shape(v) != (51,)}
        if bad_shapes:
            continue
        for i in range(51):
            row = {"init_date": exp_start_date, "stage": stage, "member": i, "y_tmax": y_tmax[i]}
            for f in factor_cols_raw:
                if f in init_factors:
                    row[f] = init_factors[f][i]
                elif f in dyn_factors:
                    row[f] = dyn_factors[f][i]
            raw_records.append(row)
    del da_tmax_m, da_tp_m, da_sm_m, da_ssr_m, da_sshf_m, da_z500_m, da_wnpsh_m
    del init_factors
    clear_file_cache()
    gc.collect()
    _cell1_memory_debug(f"[Cell1][{exp_start_date}] init done.")

df_raw_all = pd.DataFrame(raw_records).dropna().reset_index(drop=True)
raw_csv = f"{SUB_DIRS['tables']}/table_stage_member_raw_factors_SHF.csv"
df_raw_all.to_csv(raw_csv, index=False)
# SHF_Avg is stored as upward-positive in the raw factor table.
df_raw_all.groupby(["init_date", "stage"])["SHF_Avg"].agg(["count", "mean", "std", "min", "max"]).reset_index().to_csv(
    f"{SUB_DIRS['tables']}/table_stage_member_raw_factors_SHF_check.csv", index=False
)
pd.DataFrame(daily_tmax_records).to_csv(f"{SUB_DIRS['tables']}/table_daily_tmax_SHF.csv", index=False)
pd.DataFrame(daily_member_raw_records).to_csv(f"{SUB_DIRS['tables']}/table_daily_member_raw_factors_SHF.csv", index=False)
raw_coverage_rows = []
for _init in FULL_INIT_LIST:
    for _stage in study_window_dict_run:
        _g = df_raw_all[(df_raw_all.get("init_date", pd.Series(dtype=str)) == _init) & (df_raw_all.get("stage", pd.Series(dtype=str)) == _stage)] if not df_raw_all.empty else pd.DataFrame()
        _members = sorted(_g["member"].astype(int).unique().tolist()) if "member" in _g.columns else []
        _missing_factors = [c for c in ["y_tmax", *factor_cols_raw] if c not in _g.columns or _g[c].isna().all()]
        raw_coverage_rows.append({"init_date": _init, "stage": _stage, "n_members": len(_members), "member_ids_0_50": _members == list(range(51)), "required_factors_not_all_nan": not _missing_factors, "missing_factors": ";".join(_missing_factors), "available": len(_members) == 51 and not _missing_factors})
pd.DataFrame(raw_coverage_rows).to_csv(f"{SUB_DIRS['tables']}/table_raw_factor_coverage_qc.csv", index=False)
if len(df_raw_all) != EXPECTED_RAW_ROWS:
    raise RuntimeError(f"raw table row check failed: expected {EXPECTED_RAW_ROWS}, got {len(df_raw_all)}; file={raw_csv}")
print(f"Cell 1 complete: wrote {raw_csv} ({len(df_raw_all)} rows).")

# %%
# =============================================================================
# Cell 2: Read raw CSV; run stage-specific residualization + OLS + LMG; write tables
# =============================================================================

set_audit_cell_name("Cell 2: residualization + OLS + LMG")
import ast
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)

print("\n" + "=" * 80)
print("Cell 2: residualization + OLS + LMG from raw CSV only...")
print("=" * 80)

EXP_OUT_DIR = "/data1/huangy/fig6/NC/v12"
FOUR_INIT_ROOT = f"{EXP_OUT_DIR}/four_init_full_pipeline_NCVIconcurrent"
RESID_OLS_DIR = FOUR_INIT_ROOT
SUB_DIRS = {
    "tables": f"{RESID_OLS_DIR}/tables",
    "corr_raw": f"{RESID_OLS_DIR}/figures/corr_raw",
    "corr_resid": f"{RESID_OLS_DIR}/figures/corr_resid",
    "lmg_main": f"{RESID_OLS_DIR}/figures/lmg_main",
    "lmg_sensitivity": f"{RESID_OLS_DIR}/figures/lmg_sensitivity",
    "ols_lmg_combo": f"{RESID_OLS_DIR}/figures/ols_lmg_combo",
    "tmax_timeseries": f"{RESID_OLS_DIR}/figures/tmax_timeseries",
    "metrics_heatmap": f"{RESID_OLS_DIR}/figures/metrics_heatmap",
}
def assert_output_under_v7(path):
    assert_four_init_output(path)

assert_output_under_v7(RESID_OLS_DIR)
for d in SUB_DIRS.values():
    assert_output_under_v7(d)
for d in SUB_DIRS.values():
    os.makedirs(d, exist_ok=True)

start_date_list_run = FULL_INIT_LIST
study_window_dict = {
    "Stage-I_dry": ("2023-06-14", "2023-06-17"),
    "Stage-II_wet": ("2023-06-18", "2023-06-24"),
    "Total": ("2023-06-14", "2023-06-24"),
}
study_window_dict_run = study_window_dict
resid_strategies = {
    "Stage-I_dry": "SM-led residualization",
    "Stage-II_wet": "TP-led residualization",
    # Mechanistic attribution focuses on Stage-I and Stage-II; Total is retained only as an overall diagnostic.
    "Total": "TP-led residualization (Overall Diagnostic)",
}
EXPECTED_RAW_ROWS = len(start_date_list_run) * len(study_window_dict_run) * 51
EXPECTED_OLS_COMBINATIONS = len(start_date_list_run) * len(study_window_dict_run)
RAW_MODEL_FACTORS = [
    "WNPSH", "z500_anom_NCHN", "NCHN_tp", "sm_avg", "SSR_Avg", "SHF_Avg",
    "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI",
]

def calc_resid(y_col, x_cols, df):
    y_std = df[y_col].std()
    x_std_sum = df[x_cols].std().sum()
    if y_std == 0:
        return np.zeros(len(df))
    if x_std_sum == 0:
        return df[y_col] - df[y_col].mean()
    X = sm.add_constant(df[x_cols])
    return sm.OLS(df[y_col], X).fit().resid

def compute_lmg_mc(X, y, feature_names, mc_samples=2000):
    np.random.seed(42)
    n_samples, n_features = X.shape
    if n_samples < n_features + 2:
        return {name: np.nan for name in feature_names}
    X_s = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    y_s = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)
    lmg_raw = np.zeros(n_features)
    model = LinearRegression()
    for _ in range(mc_samples):
        order = np.random.permutation(n_features)
        prev_r2 = 0.0
        for i in range(n_features):
            idx = order[:i + 1]
            r2 = model.fit(X_s[:, idx], y_s).score(X_s[:, idx], y_s)
            lmg_raw[order[i]] += max(0, (r2 - prev_r2))
            prev_r2 = r2
    tot = lmg_raw.sum()
    if tot == 0:
        return {name: 0.0 for name in feature_names}
    return {name: val for name, val in zip(feature_names, (lmg_raw / tot) * 100.0)}

def _stringify_list(values):
    return ";".join(values) if values else ""

def safe_corr(a, b):
    if len(a) < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]

def fit_ols_lmg(df, factor_names, model_type, residualization_strategy):
    valid_factors = [f for f in factor_names if f in df.columns and df[f].std() >= 1e-12]
    dropped_factors = [f for f in factor_names if f not in valid_factors]
    X = df[valid_factors].values
    y = df["y_tmax"].values
    model = sm.OLS(y, sm.add_constant(X)).fit()
    y_pred = model.fittedvalues
    lmg_dict = compute_lmg_mc(X, y, valid_factors)
    sorted_lmg = sorted(lmg_dict.items(), key=lambda item: item[1], reverse=True)
    summary = {
        "init_date": df["init_date"].iloc[0],
        "stage": df["stage"].iloc[0],
        "model_type": model_type,
        "residualization_strategy": residualization_strategy,
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "Pred_vs_ECMWF_Corr": safe_corr(y_pred, y),
        "Pred_vs_ECMWF_RMSE": np.sqrt(np.mean((y_pred - y) ** 2)),
        "Pred_vs_ECMWF_Bias": np.mean(y_pred - y),
        "Top1_factor_name": sorted_lmg[0][0] if sorted_lmg else np.nan,
        "Top1_factor_importance_pct": sorted_lmg[0][1] if sorted_lmg else np.nan,
        "dropped_factors": dropped_factors,
        "lmg_dict": lmg_dict,
    }
    fitted = df[["init_date", "stage", "member", "y_tmax"]].copy()
    fitted["model_type"] = model_type
    fitted["fitted_value"] = y_pred
    fitted["residual"] = y - y_pred
    return summary, lmg_dict, fitted

def build_factor_direction_rows(df, factor_names, model_kind):
    n_members = len(df)
    if n_members != 51:
        print(f"  [Warning] factor direction diagnostics for {df['init_date'].iloc[0]} | {df['stage'].iloc[0]} | {model_kind}: n_members={n_members}, expected 51")
    valid_factors = [f for f in factor_names if f in df.columns and df[f].std() >= 1e-12]
    if not valid_factors:
        return []
    y = df["y_tmax"].astype(float).values
    y_std = np.nanstd(y)
    y_s = (y - np.nanmean(y)) / (y_std + 1e-8)
    X = df[valid_factors].astype(float).values
    X_s = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    model = sm.OLS(y_s, sm.add_constant(X_s)).fit()
    coef_map = {factor: model.params[i + 1] for i, factor in enumerate(valid_factors)}
    rows = []
    for factor in valid_factors:
        coef = float(coef_map[factor])
        rows.append({
            "init_date": df["init_date"].iloc[0],
            "stage": df["stage"].iloc[0],
            "model_kind": model_kind,
            "factor": factor,
            "pearson_r_with_y": safe_corr(df[factor].astype(float).values, y),
            "ols_coef_standardized": coef,
            "coef_sign": "positive" if coef > 0 else ("negative" if coef < 0 else "zero"),
            "n_members": n_members,
        })
    return rows

def build_lmg_long_rows(summary_row, lmg_dict):
    dropped = summary_row.get("dropped_factors", [])
    dropped_str = dropped if isinstance(dropped, str) else _stringify_list(dropped)
    return [
        {
            "init_date": summary_row["init_date"],
            "stage": summary_row["stage"],
            "model_type": summary_row["model_type"],
            "residualization_strategy": summary_row["residualization_strategy"],
            "factor_name": factor_name,
            "relative_importance_pct": importance,
            "R2": summary_row["R2"],
            "Adj_R2": summary_row["Adj_R2"],
            "dropped_factors": dropped_str,
        }
        for factor_name, importance in lmg_dict.items()
    ]

def stringify_summary_df(df):
    df_csv = df.copy()
    if not df_csv.empty:
        df_csv["dropped_factors"] = df_csv["dropped_factors"].apply(_stringify_list)
        df_csv["lmg_dict"] = df_csv["lmg_dict"].apply(repr)
    return df_csv

raw_csv = f"{SUB_DIRS['tables']}/table_stage_member_raw_factors_SHF.csv"
df_raw_all = pd.read_csv(raw_csv)
if len(df_raw_all) != EXPECTED_RAW_ROWS:
    raise RuntimeError(f"raw table row check failed: expected {EXPECTED_RAW_ROWS}, got {len(df_raw_all)}; file={raw_csv}")

resid_records = []
tp_param_resid_records = []
fitted_records = []
raw_ols_results = []
main_ols_results = []
sens_ols_results = []
raw_lmg_long_records = []
main_lmg_long_records = []
sens_lmg_long_records = []
factor_direction_records = []
residual_audit_records = []
residual_keycase_records = []
residualized_model_predictor_records = []

def _manual_residual_and_model(df, y_col, control_cols):
    X = sm.add_constant(df[control_cols], has_constant="add")
    model = sm.OLS(df[y_col], X).fit()
    fitted = model.predict(X)
    resid = df[y_col] - fitted
    return model, fitted, resid

def _append_residual_audit(df, init_date, stage, model_type, resid_name, y_col, control_cols, resid_series):
    model, fitted, resid_manual = _manual_residual_and_model(df, y_col, control_cols)
    resid_code = pd.Series(resid_series, index=df.index, dtype=float)
    manual_max_abs_diff = float(np.nanmax(np.abs(resid_code.values - resid_manual.values)))
    for control_col in control_cols:
        corr_before = safe_corr(df[y_col].astype(float).values, df[control_col].astype(float).values)
        corr_after = safe_corr(resid_code.values.astype(float), df[control_col].astype(float).values)
        residual_audit_records.append({
            "init_date": init_date,
            "stage": stage,
            "model_type": model_type,
            "resid_name": resid_name,
            "y_col": y_col,
            "control_cols": ";".join(control_cols),
            "control_col": control_col,
            "n_members": int(len(df)),
            "corr_y_control_before": corr_before,
            "corr_resid_control_after": corr_after,
            "mean_resid": float(np.nanmean(resid_code.values)),
            "std_resid": float(np.nanstd(resid_code.values)),
            "manual_max_abs_diff": manual_max_abs_diff,
            "corr_Tmax_SHF_raw": safe_corr(df["y_tmax"].astype(float).values, df["SHF_Avg"].astype(float).values) if "SHF_Avg" in df.columns else np.nan,
            "corr_Tmax_SHF_resid_semipartial": safe_corr(df["y_tmax"].astype(float).values, df["SHF_resid"].astype(float).values) if "SHF_resid" in df.columns else np.nan,
            "corr_Tmaxresid_SHFresid_partial": np.nan,
            "resid_formula_ok": bool((manual_max_abs_diff < 1e-8) and (np.isnan(corr_after) or abs(corr_after) < 1e-8)),
        })
    return model

for (exp_start_date, stage), df_raw in df_raw_all.groupby(["init_date", "stage"], sort=False):
    df_raw = df_raw.dropna().copy()
    if len(df_raw) != 51:
        raise RuntimeError(f"raw group size check failed for {exp_start_date}/{stage}: expected 51, got {len(df_raw)}")

    raw_summary, raw_lmg, raw_fitted = fit_ols_lmg(df_raw, RAW_MODEL_FACTORS, "raw", "No residualization")
    raw_ols_results.append(raw_summary)
    raw_lmg_long_records.extend(build_lmg_long_rows(raw_summary, raw_lmg))
    fitted_records.extend(raw_fitted.to_dict("records"))
    factor_direction_records.extend(build_factor_direction_rows(df_raw, RAW_MODEL_FACTORS, "raw"))

    df_resid = df_raw.copy()
    if stage == "Stage-I_dry":
        df_resid["SSR_resid"] = calc_resid("SSR_Avg", ["sm_avg"], df_resid)
        df_resid["SHF_resid"] = calc_resid("SHF_Avg", ["sm_avg"], df_resid)
        _append_residual_audit(df_resid, exp_start_date, stage, "residualized_main", "SSR_resid", "SSR_Avg", ["sm_avg"], df_resid["SSR_resid"])
        shf_model = _append_residual_audit(df_resid, exp_start_date, stage, "residualized_main", "SHF_resid", "SHF_Avg", ["sm_avg"], df_resid["SHF_resid"])
        target_factors = ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "sm_avg", "SSR_resid", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        residualized_model_predictor_records.append({"init_date": exp_start_date, "stage": stage, "model_type": "residualized_main", "predictors_used": ";".join(target_factors), "target_col": "y_tmax", "n_predictors": len(target_factors), "n_members": int(len(df_resid))})
        if exp_start_date == "2023-06-12":
            tmax_resid = calc_resid("y_tmax", ["sm_avg"], df_resid)
            corr_partial = safe_corr(tmax_resid, df_resid["SHF_resid"].astype(float).values)
            for rec in residual_audit_records:
                if rec["init_date"] == exp_start_date and rec["stage"] == stage and rec["resid_name"] == "SHF_resid":
                    rec["corr_Tmaxresid_SHFresid_partial"] = corr_partial
            X_key = sm.add_constant(df_resid[["sm_avg"]], has_constant="add")
            fitted_key = shf_model.predict(X_key)
            resid_manual = df_resid["SHF_Avg"] - fitted_key
            for i, (_, row) in enumerate(df_resid.reset_index(drop=True).iterrows()):
                residual_keycase_records.append({
                    "init_date": exp_start_date, "stage": stage, "member": int(row["member"]),
                    "SHF_Avg": float(row["SHF_Avg"]), "sm_avg": float(row["sm_avg"]),
                    "fitted_SHF_from_sm": float(fitted_key.iloc[i]), "SHF_resid_code": float(row["SHF_resid"]),
                    "SHF_resid_manual": float(resid_manual.iloc[i]), "diff": float(row["SHF_resid"] - resid_manual.iloc[i]),
                })
            print(f"[Keycase 2023-06-12 Stage-I] corr(Tmax,SHF_Avg)={safe_corr(df_resid['y_tmax'].values, df_resid['SHF_Avg'].values):.3f}, corr(SHF_Avg,sm_avg)={safe_corr(df_resid['SHF_Avg'].values, df_resid['sm_avg'].values):.3f}, corr(Tmax,sm_avg)={safe_corr(df_resid['y_tmax'].values, df_resid['sm_avg'].values):.3f}, corr(Tmax,SHF_resid)={safe_corr(df_resid['y_tmax'].values, df_resid['SHF_resid'].values):.3f}, corr(SHF_resid,sm_avg)={safe_corr(df_resid['SHF_resid'].values, df_resid['sm_avg'].values):.3e}, beta={float(shf_model.params['sm_avg']):.6g}, intercept={float(shf_model.params['const']):.6g}")
    else:
        # Stage-II uses TP-led residualization; Total uses the same TP-led form only as an overall diagnostic.
        df_resid["SM_resid"] = calc_resid("sm_avg", ["NCHN_tp"], df_resid)
        df_resid["SSR_resid"] = calc_resid("SSR_Avg", ["NCHN_tp"], df_resid)
        df_resid["SHF_resid"] = calc_resid("SHF_Avg", ["NCHN_tp"], df_resid)
        _append_residual_audit(df_resid, exp_start_date, stage, "residualized_main", "SM_resid", "sm_avg", ["NCHN_tp"], df_resid["SM_resid"])
        _append_residual_audit(df_resid, exp_start_date, stage, "residualized_main", "SSR_resid", "SSR_Avg", ["NCHN_tp"], df_resid["SSR_resid"])
        _append_residual_audit(df_resid, exp_start_date, stage, "residualized_main", "SHF_resid", "SHF_Avg", ["NCHN_tp"], df_resid["SHF_resid"])
        target_factors = ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "SM_resid", "SSR_resid", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        residualized_model_predictor_records.append({"init_date": exp_start_date, "stage": stage, "model_type": "residualized_main", "predictors_used": ";".join(target_factors), "target_col": "y_tmax", "n_predictors": len(target_factors), "n_members": int(len(df_resid))})

    resid_records.extend(df_resid.to_dict("records"))
    main_summary, main_lmg, main_fitted = fit_ols_lmg(df_resid, target_factors, "residualized_main", resid_strategies.get(stage, ""))
    main_ols_results.append(main_summary)
    main_lmg_long_records.extend(build_lmg_long_rows(main_summary, main_lmg))
    fitted_records.extend(main_fitted.to_dict("records"))
    factor_direction_records.extend(build_factor_direction_rows(df_resid, target_factors, "residualized_main"))

    if stage == "Stage-II_wet":
        df_sens = df_raw.copy()
        df_sens["TP_param_resid"] = calc_resid("NCHN_tp", ["z500_anom_NCHN", "WNPSH"], df_sens)
        df_sens["SM_param_resid"] = calc_resid("sm_avg", ["TP_param_resid"], df_sens)
        df_sens["SSR_param_resid"] = calc_resid("SSR_Avg", ["TP_param_resid"], df_sens)
        df_sens["SHF_param_resid"] = calc_resid("SHF_Avg", ["TP_param_resid"], df_sens)
        _append_residual_audit(df_sens, exp_start_date, stage, "sensitivity_TPparam", "TP_param_resid", "NCHN_tp", ["z500_anom_NCHN", "WNPSH"], df_sens["TP_param_resid"])
        _append_residual_audit(df_sens, exp_start_date, stage, "sensitivity_TPparam", "SM_param_resid", "sm_avg", ["TP_param_resid"], df_sens["SM_param_resid"])
        _append_residual_audit(df_sens, exp_start_date, stage, "sensitivity_TPparam", "SSR_param_resid", "SSR_Avg", ["TP_param_resid"], df_sens["SSR_param_resid"])
        _append_residual_audit(df_sens, exp_start_date, stage, "sensitivity_TPparam", "SHF_param_resid", "SHF_Avg", ["TP_param_resid"], df_sens["SHF_param_resid"])
        sens_factors = ["WNPSH", "z500_anom_NCHN", "TP_param_resid", "SM_param_resid", "SSR_param_resid", "SHF_param_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        residualized_model_predictor_records.append({"init_date": exp_start_date, "stage": stage, "model_type": "sensitivity_TPparam", "predictors_used": ";".join(sens_factors), "target_col": "y_tmax", "n_predictors": len(sens_factors), "n_members": int(len(df_sens))})
        tp_param_resid_records.extend(df_sens.to_dict("records"))
        sens_summary, sens_lmg, sens_fitted = fit_ols_lmg(df_sens, sens_factors, "sensitivity_TPparam", "TP_param_resid-led sensitivity")
        sens_ols_results.append(sens_summary)
        sens_lmg_long_records.extend(build_lmg_long_rows(sens_summary, sens_lmg))
        fitted_records.extend(sens_fitted.to_dict("records"))
        factor_direction_records.extend(build_factor_direction_rows(df_sens, sens_factors, "sensitivity_TPparam"))

df_resid_all = pd.DataFrame(resid_records)
df_tp_param_resid_all = pd.DataFrame(tp_param_resid_records)
df_fitted_all = pd.DataFrame(fitted_records)
df_raw_ols = pd.DataFrame(raw_ols_results)
df_main_ols = pd.DataFrame(main_ols_results)
df_sens_ols = pd.DataFrame(sens_ols_results)
df_raw_lmg_long = pd.DataFrame(raw_lmg_long_records)
df_lmg_long = pd.DataFrame(main_lmg_long_records)
df_sens_lmg_long = pd.DataFrame(sens_lmg_long_records)
df_factor_direction = pd.DataFrame(factor_direction_records)

actual_n = len(df_main_ols)
print(f"\n[RESULT CHECK] successful combinations = {actual_n}/{EXPECTED_OLS_COMBINATIONS}")
if actual_n != EXPECTED_OLS_COMBINATIONS:
    raise RuntimeError(f"OLS combination check failed: expected {EXPECTED_OLS_COMBINATIONS}/{EXPECTED_OLS_COMBINATIONS}, got {actual_n}/{EXPECTED_OLS_COMBINATIONS}")

df_resid_all.to_csv(f"{SUB_DIRS['tables']}/table_stage_member_residualized_factors_SHF.csv", index=False)
df_fitted_all.to_csv(f"{SUB_DIRS['tables']}/table_stage_member_ols_fitted_SHF.csv", index=False)
stringify_summary_df(df_raw_ols).to_csv(f"{SUB_DIRS['tables']}/table_stage_raw_ols_lmg_SHF.csv", index=False)
df_raw_lmg_long.to_csv(f"{SUB_DIRS['tables']}/table_stage_raw_lmg_long_SHF.csv", index=False)
df_factor_direction.to_csv(f"{SUB_DIRS['tables']}/table_factor_direction_diagnostics_SHF.csv", index=False)
stringify_summary_df(df_main_ols).to_csv(f"{SUB_DIRS['tables']}/table_stage_ols_lmg_SHF.csv", index=False)
df_lmg_long.to_csv(f"{SUB_DIRS['tables']}/table_stage_lmg_long_SHF.csv", index=False)
if not df_sens_ols.empty:
    df_tp_param_resid_all.to_csv(f"{SUB_DIRS['tables']}/table_stage_member_TPparam_residualized_factors_SHF.csv", index=False)
    stringify_summary_df(df_sens_ols).to_csv(f"{SUB_DIRS['tables']}/table_stageII_TPparam_sensitivity_SHF.csv", index=False)
    df_sens_lmg_long.to_csv(f"{SUB_DIRS['tables']}/table_stageII_TPparam_sensitivity_lmg_long_SHF.csv", index=False)
if residual_audit_records:
    pd.DataFrame(residual_audit_records).to_csv(f"{SUB_DIRS['tables']}/residualization_audit.csv", index=False)
if residual_keycase_records:
    pd.DataFrame(residual_keycase_records).to_csv(f"{SUB_DIRS['tables']}/residualization_keycase_0612_stageI_SHF_vs_SM.csv", index=False)
if residualized_model_predictor_records:
    pd.DataFrame(residualized_model_predictor_records).to_csv(f"{SUB_DIRS['tables']}/residualized_model_predictor_list.csv", index=False)
print("Cell 2 complete: wrote raw/main/sensitivity OLS, fitted values, and long-format LMG tables.")

# %%
# =============================================================================
# Cell 3: Read CSVs and draw correlation matrices, scatter+LMG, Tmax series, heatmaps
# =============================================================================

set_audit_cell_name("Cell 3: plot matrices, Tmax series, and heatmaps")
import ast
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

print("\n" + "=" * 80)
print("Cell 3: plot matrices, scatter+LMG composites, Tmax series, and heatmaps from CSV files only...")
print("=" * 80)

EXP_OUT_DIR = "/data1/huangy/fig6/NC/v12"
FOUR_INIT_ROOT = f"{EXP_OUT_DIR}/four_init_full_pipeline_NCVIconcurrent"
RESID_OLS_DIR = FOUR_INIT_ROOT
ERA5_TMAX_FILE = "/data1/huangy/fig6/NC/ai_test/pangu/era5_T/era5_2m_temperature_6_2023.nc"
SUB_DIRS = {
    "tables": f"{RESID_OLS_DIR}/tables",
    "corr_raw": f"{RESID_OLS_DIR}/figures/corr_raw",
    "corr_resid": f"{RESID_OLS_DIR}/figures/corr_resid",
    "lmg_main": f"{RESID_OLS_DIR}/figures/lmg_main",
    "lmg_sensitivity": f"{RESID_OLS_DIR}/figures/lmg_sensitivity",
    "ols_lmg_combo": f"{RESID_OLS_DIR}/figures/ols_lmg_combo",
    "tmax_timeseries": f"{RESID_OLS_DIR}/figures/tmax_timeseries",
    "metrics_heatmap": f"{RESID_OLS_DIR}/figures/metrics_heatmap",
}
def assert_output_under_v7(path):
    assert_four_init_output(path)

for d in SUB_DIRS.values():
    assert_output_under_v7(d)
    os.makedirs(d, exist_ok=True)
assert_output_under_v7(RESID_OLS_DIR)

start_date_list_run = FULL_INIT_LIST
study_window_dict = {
    "Stage-I_dry": ("2023-06-14", "2023-06-17"),
    "Stage-II_wet": ("2023-06-18", "2023-06-24"),
    "Total": ("2023-06-14", "2023-06-24"),
}
study_window_dict_run = study_window_dict
resid_strategies = {
    "Stage-I_dry": "SM-led residualization",
    "Stage-II_wet": "TP-led residualization",
    "Total": "TP-led residualization (Overall Diagnostic)",
}
EXPECTED_RAW_ROWS = len(start_date_list_run) * len(study_window_dict_run) * 51
EXPECTED_OLS_COMBINATIONS = len(start_date_list_run) * len(study_window_dict_run)
DAILY_PROJECTION_STAGES = ["Stage-I_dry", "Stage-II_wet"]
EXPECTED_DAILY_PROJECTION_ROWS = len(start_date_list_run) * (4 + 7)
EXPECTED_DAILY_PROJECTION_STAGE_DAYS = {"Stage-I_dry": 4, "Stage-II_wet": 7}

raw_csv = f"{SUB_DIRS['tables']}/table_stage_member_raw_factors_SHF.csv"
resid_csv = f"{SUB_DIRS['tables']}/table_stage_member_residualized_factors_SHF.csv"
fitted_csv = f"{SUB_DIRS['tables']}/table_stage_member_ols_fitted_SHF.csv"
daily_csv = f"{SUB_DIRS['tables']}/table_daily_tmax_SHF.csv"
daily_member_raw_csv = f"{SUB_DIRS['tables']}/table_daily_member_raw_factors_SHF.csv"
daily_ols_projection_csv = f"{SUB_DIRS['tables']}/table_daily_tmax_with_ols_projection_SHF.csv"
daily_projection_consistency_csv = f"{SUB_DIRS['tables']}/table_daily_ols_projection_consistency_SHF.csv"
daily_projection_setup_csv = f"{SUB_DIRS['tables']}/table_daily_ols_projection_model_setup_SHF.csv"
daily_stage_predictor_compare_csv = f"{SUB_DIRS['tables']}/table_daily_to_stage_predictor_comparison_SHF.csv"
raw_ols_csv = f"{SUB_DIRS['tables']}/table_stage_raw_ols_lmg_SHF.csv"
main_ols_csv = f"{SUB_DIRS['tables']}/table_stage_ols_lmg_SHF.csv"
sens_ols_csv = f"{SUB_DIRS['tables']}/table_stageII_TPparam_sensitivity_SHF.csv"
factor_direction_csv = f"{SUB_DIRS['tables']}/table_factor_direction_diagnostics_SHF.csv"
raw_shf_check_csv = f"{SUB_DIRS['tables']}/table_stage_member_raw_factors_SHF_check.csv"
residual_audit_csv = f"{SUB_DIRS['tables']}/residualization_audit.csv"
residual_keycase_csv = f"{SUB_DIRS['tables']}/residualization_keycase_0612_stageI_SHF_vs_SM.csv"
residualized_predictor_list_csv = f"{SUB_DIRS['tables']}/residualized_model_predictor_list.csv"
for p in [
    raw_csv, resid_csv, fitted_csv, daily_csv, daily_member_raw_csv, daily_ols_projection_csv,
    daily_projection_consistency_csv, daily_projection_setup_csv, daily_stage_predictor_compare_csv,
    raw_ols_csv, main_ols_csv, sens_ols_csv, factor_direction_csv, raw_shf_check_csv,
    residual_audit_csv, residual_keycase_csv, residualized_predictor_list_csv,
]:
    assert_output_under_v7(p)

for stale_daily_projection_csv in [
    daily_ols_projection_csv,
    daily_projection_consistency_csv,
    daily_projection_setup_csv,
    daily_stage_predictor_compare_csv,
]:
    if os.path.exists(stale_daily_projection_csv):
        os.remove(stale_daily_projection_csv)
        print(f"[Cell 3] Removed stale daily projection output: {stale_daily_projection_csv}")
for init in start_date_list_run:
    for stale_tmax_fig in [
        f"{SUB_DIRS['tmax_timeseries']}/tmax_timeseries_{init}_SHF.png",
        f"{SUB_DIRS['tmax_timeseries']}/tmax_timeseries_stage_mean_diagnostic_{init}_SHF.png",
    ]:
        if os.path.exists(stale_tmax_fig):
            os.remove(stale_tmax_fig)
            print(f"[Cell 3] Removed stale Tmax time-series figure: {stale_tmax_fig}")

df_raw_all = pd.read_csv(raw_csv)
df_resid_all = pd.read_csv(resid_csv)
df_fitted_all = pd.read_csv(fitted_csv)
df_daily_tmax = pd.read_csv(daily_csv) if os.path.exists(daily_csv) else pd.DataFrame()
df_daily_member_raw = pd.read_csv(daily_member_raw_csv) if os.path.exists(daily_member_raw_csv) else pd.DataFrame()
if not df_daily_member_raw.empty:
    df_daily_member_raw["date"] = pd.to_datetime(df_daily_member_raw["date"])
df_raw_ols = pd.read_csv(raw_ols_csv)
df_main_ols = pd.read_csv(main_ols_csv)
df_sens_ols = pd.read_csv(sens_ols_csv) if os.path.exists(sens_ols_csv) else pd.DataFrame()
df_factor_direction = pd.read_csv(factor_direction_csv) if os.path.exists(factor_direction_csv) else pd.DataFrame()
df_raw_shf_check = pd.read_csv(raw_shf_check_csv) if os.path.exists(raw_shf_check_csv) else pd.DataFrame()
if not os.path.exists(residual_audit_csv):
    raise RuntimeError(f"Missing required residual audit CSV: {residual_audit_csv}")
if not os.path.exists(residual_keycase_csv):
    raise RuntimeError(f"Missing required residual keycase CSV: {residual_keycase_csv}")
if not os.path.exists(residualized_predictor_list_csv):
    raise RuntimeError(f"Missing required residualized predictor list CSV: {residualized_predictor_list_csv}")
df_residual_audit = pd.read_csv(residual_audit_csv)
df_residual_keycase = pd.read_csv(residual_keycase_csv)
df_residualized_predictor_list = pd.read_csv(residualized_predictor_list_csv)
print(f"[Cell 3] ERA5 Tmax file path used for daily metrics: {ERA5_TMAX_FILE}")
if len(df_raw_all) != EXPECTED_RAW_ROWS:
    raise RuntimeError(f"raw table row check failed: expected {EXPECTED_RAW_ROWS}, got {len(df_raw_all)}; file={raw_csv}")
if len(df_main_ols) != EXPECTED_OLS_COMBINATIONS:
    raise RuntimeError(f"OLS combination check failed: expected {EXPECTED_OLS_COMBINATIONS}/{EXPECTED_OLS_COMBINATIONS}, got {len(df_main_ols)}/{EXPECTED_OLS_COMBINATIONS}")

print("\n[SHF_Avg(up) range check] SHF_Avg is upward-positive; Cell 2/3 do not flip the sign again.")
if not df_raw_shf_check.empty:
    for _, row in df_raw_shf_check.iterrows():
        print(f"  {row['init_date']} | {row['stage']}: SHF_Avg min={row['min']:.2f}, max={row['max']:.2f}, mean={row['mean']:.2f}, std={row['std']:.2f}")
        if row['mean'] < 0:
            print("    [Warning] SHF_Avg(up) mean is negative; verify upward-positive convention and input sign.")
        if abs(row['mean']) > 1000 or abs(row['min']) > 3000 or abs(row['max']) > 3000:
            print("    [Warning] SHF_Avg(up) range appears unusually large; please inspect source flux units.")
else:
    shf_stats = df_raw_all.groupby(["init_date", "stage"])["SHF_Avg"].agg(["min", "max", "mean", "std"]).reset_index()
    for _, row in shf_stats.iterrows():
        print(f"  {row['init_date']} | {row['stage']}: SHF_Avg min={row['min']:.2f}, max={row['max']:.2f}, mean={row['mean']:.2f}, std={row['std']:.2f}")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.linewidth": 1.2})
label_map = {
    "y_tmax": "Tmax",
    "SHF_Avg": "SHF_Avg(up)", "SHF_resid": "SHF_resid", "SHF_param_resid": "SHF_param_resid",
    "SM_resid": "SM_resid", "SM_param_resid": "SM_param_resid", "SSR_resid": "SSR_resid",
    "SSR_param_resid": "SSR_param_resid", "NCHN_tp": "NCHN_tp", "z500_anom_NCHN": "z500_anom",
    "Initial_MSEstar_max_NCHN": "Initial_MSEstar_max", "Initial_Barrier_NCHN": "Initial_Barrier",
    "NCVI": "NCVI", "WNPSH": "WNPSH", "sm_avg": "sm_avg", "TP_param_resid": "TP_param_resid",
}

def parse_lmg_dict(value):
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    parsed = ast.literal_eval(str(value))
    return parsed if isinstance(parsed, dict) else {}

def safe_corr_r2(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1] ** 2)

def clean_stage_name(stage):
    return stage.replace("Stage-I_dry", "Stage-I").replace("Stage-II_wet", "Stage-II")

def stage_day_count(stage):
    w_start, w_end = study_window_dict_run[stage]
    return len(pd.date_range(w_start, w_end, freq="D"))

def _fit_linear_coefficients(y, X):
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if y_arr.ndim != 1:
        y_arr = y_arr.squeeze()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if np.nanstd(y_arr) == 0:
        return np.array([float(np.nanmean(y_arr))] + [0.0] * X_arr.shape[1])
    if np.nansum(np.nanstd(X_arr, axis=0)) == 0:
        return np.array([float(np.nanmean(y_arr))] + [0.0] * X_arr.shape[1])
    design = np.column_stack([np.ones(len(y_arr)), X_arr])
    return np.linalg.lstsq(design, y_arr, rcond=None)[0]

def _apply_linear_coefficients(beta, X):
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    design = np.column_stack([np.ones(len(X_arr)), X_arr])
    return design @ beta

def residualized_main_factors_for_stage(stage):
    if stage == "Stage-I_dry":
        return ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "sm_avg", "SSR_resid", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
    return ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "SM_resid", "SSR_resid", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]

def raw_needed_cols_for_projection(stage):
    if stage == "Stage-I_dry":
        return ["y_tmax", "SSR_Avg", "SHF_Avg", "sm_avg"]
    return ["y_tmax", "sm_avg", "SSR_Avg", "SHF_Avg", "NCHN_tp"]

def daily_needed_cols_for_projection(stage):
    return [
        "y_tmax", "NCHN_tp", "sm_avg", "SSR_Avg", "SHF_Avg",
        "z500_anom_NCHN", "WNPSH", "Initial_MSEstar_max_NCHN",
        "Initial_Barrier_NCHN", "NCVI",
    ]

def fit_stage_projection_components(init, stage):
    raw_stage = df_raw_all[(df_raw_all["init_date"] == init) & (df_raw_all["stage"] == stage)].copy()
    resid_stage = df_resid_all[(df_resid_all["init_date"] == init) & (df_resid_all["stage"] == stage)].copy()
    raw_before = len(raw_stage)
    resid_before = len(resid_stage)
    factors = residualized_main_factors_for_stage(stage)

    needed_raw_cols = raw_needed_cols_for_projection(stage)
    missing_raw_cols = [c for c in needed_raw_cols if c not in raw_stage.columns]
    if missing_raw_cols:
        print(f"  [Daily projection debug] {init} | {stage}: raw_stage rows before subset dropna={raw_before}, after subset dropna=0, missing_raw_cols={missing_raw_cols}")
        return None, f"missing raw columns: {missing_raw_cols}"
    raw_stage = raw_stage.dropna(subset=needed_raw_cols).copy()

    needed_resid_cols = ["y_tmax"] + factors
    needed_resid_cols = [c for c in needed_resid_cols if c in resid_stage.columns]
    resid_stage = resid_stage.dropna(subset=needed_resid_cols).copy()
    print(f"  [Daily projection debug] {init} | {stage}: raw_stage rows before subset dropna={raw_before}, after subset dropna={len(raw_stage)}; resid_stage rows before subset dropna={resid_before}, after subset dropna={len(resid_stage)}")

    valid_factors = [f for f in factors if f in resid_stage.columns and resid_stage[f].std() >= 1e-12]
    if raw_stage.empty:
        return None, f"raw_stage empty after subset dropna on {needed_raw_cols} (before={raw_before})"
    if resid_stage.empty:
        return None, f"resid_stage empty after subset dropna on {needed_resid_cols} (before={resid_before})"
    if not valid_factors:
        return None, f"valid_factors empty after subset dropna; candidate factors={factors}"
    ols_beta = _fit_linear_coefficients(resid_stage["y_tmax"].values, resid_stage[valid_factors].values)
    resid_models = {}
    if stage == "Stage-I_dry":
        resid_models["SSR_resid"] = ("SSR_Avg", ["sm_avg"], _fit_linear_coefficients(raw_stage["SSR_Avg"].values, raw_stage[["sm_avg"]].values))
        resid_models["SHF_resid"] = ("SHF_Avg", ["sm_avg"], _fit_linear_coefficients(raw_stage["SHF_Avg"].values, raw_stage[["sm_avg"]].values))
    else:
        resid_models["SM_resid"] = ("sm_avg", ["NCHN_tp"], _fit_linear_coefficients(raw_stage["sm_avg"].values, raw_stage[["NCHN_tp"]].values))
        resid_models["SSR_resid"] = ("SSR_Avg", ["NCHN_tp"], _fit_linear_coefficients(raw_stage["SSR_Avg"].values, raw_stage[["NCHN_tp"]].values))
        resid_models["SHF_resid"] = ("SHF_Avg", ["NCHN_tp"], _fit_linear_coefficients(raw_stage["SHF_Avg"].values, raw_stage[["NCHN_tp"]].values))
    return {
        "ols_beta": ols_beta,
        "valid_factors": valid_factors,
        "resid_models": resid_models,
        "stage_fitted_mean": float(_apply_linear_coefficients(ols_beta, resid_stage[valid_factors].values).mean()),
        "ols_intercept": float(ols_beta[0]),
    }, "ok"

def prepare_daily_predictors_for_stage(daily_group, stage, components):
    df_proj = daily_group.copy()
    # Stage OLS uses window-total precipitation. For daily plotting, scale daily precipitation by
    # the number of days so the stage-mean of daily predictors equals the stage-level predictor.
    if "NCHN_tp" in df_proj.columns:
        df_proj["NCHN_tp"] = df_proj["NCHN_tp"].astype(float) * stage_day_count(stage)
    for resid_name, (y_col, x_cols, beta) in components["resid_models"].items():
        fitted_common = _apply_linear_coefficients(beta, df_proj[x_cols].astype(float).values)
        df_proj[resid_name] = df_proj[y_col].astype(float).values - fitted_common
    return df_proj

def build_daily_ols_projection(init, daily_stats, stage_mean_rows):
    if df_daily_member_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    projection_rows = []
    setup_rows = []
    stage_mean_lookup = {(row["stage"]): row for row in stage_mean_rows if row["init_date"] == init}
    components_by_stage = {}
    for stage in DAILY_PROJECTION_STAGES:
        components, reason = fit_stage_projection_components(init, stage)
        if components is None:
            message = f"Daily projection setup failed for {init} | {stage}: {reason}"
            print(f"  [Warning] {message}")
            if stage == "Stage-I_dry":
                raise RuntimeError(message)
            continue
        components_by_stage[stage] = components
        setup_rows.append({
                "init_date": init,
                "stage": stage,
                "same_ols_coefficients_as_stage_model": True,
                "same_intercept_as_stage_model": True,
                "same_scaler_as_stage_model": "No scaler used in stage OLS",
                "same_residualization_equations_as_stage_model": True,
                "same_factor_sign_convention": "SHF_Avg upward-positive; no Cell 3 sign flip",
                "ols_intercept": components["ols_intercept"],
                "valid_factors": ";".join(components["valid_factors"]),
                "residualization_equations": ";".join([f"{name}={y_col}~{'+'.join(x_cols)}" for name, (y_col, x_cols, _) in components["resid_models"].items()]),
                "baseline_double_added": False,
                "prediction_formula": "X_daily_stage_scaled @ stage_OLS_beta; no extra stage mean/baseline added",
            })
    for day in sorted(daily_stats["date"].unique()):
        day_ts = pd.Timestamp(day)
        daily_group = df_daily_member_raw[(df_daily_member_raw["init_date"] == init) & (df_daily_member_raw["date"] == day_ts)].copy()
        if daily_group.empty:
            expected_stage = None
            for candidate_stage in DAILY_PROJECTION_STAGES:
                w_start, w_end = study_window_dict_run[candidate_stage]
                if pd.Timestamp(w_start) <= day_ts <= pd.Timestamp(w_end):
                    expected_stage = candidate_stage
                    break
            message = f"daily_group empty for {init} | {day_ts:%Y-%m-%d} | expected_stage={expected_stage}"
            print(f"  [Warning] {message}")
            if expected_stage == "Stage-I_dry":
                raise RuntimeError(message)
            continue
        stage = daily_group["stage"].iloc[0]
        daily_needed_cols = daily_needed_cols_for_projection(stage)
        missing_daily_cols = [c for c in daily_needed_cols if c not in daily_group.columns]
        if missing_daily_cols:
            message = f"daily_group missing required columns for {init} | {day_ts:%Y-%m-%d} | stage={stage}: {missing_daily_cols}"
            print(f"  [Warning] {message}")
            if stage == "Stage-I_dry":
                raise RuntimeError(message)
            continue
        daily_before = len(daily_group)
        daily_group = daily_group.dropna(subset=daily_needed_cols).copy()
        if daily_group.empty:
            message = f"daily_group empty after subset dropna for {init} | {day_ts:%Y-%m-%d} | stage={stage}: before={daily_before}, subset={daily_needed_cols}"
            print(f"  [Warning] {message}")
            if stage == "Stage-I_dry":
                raise RuntimeError(message)
            continue
        components = components_by_stage.get(stage)
        if components is None:
            message = f"projection components missing for {init} | {day_ts:%Y-%m-%d} | stage={stage}"
            print(f"  [Warning] {message}")
            if stage == "Stage-I_dry":
                raise RuntimeError(message)
            continue
        daily_resid = prepare_daily_predictors_for_stage(daily_group, stage, components)
        fitted_daily = _apply_linear_coefficients(components["ols_beta"], daily_resid[components["valid_factors"]].astype(float).values)
        stats_row = daily_stats[daily_stats["date"] == day_ts]
        if stats_row.empty:
            message = f"daily_stats empty for {init} | {day_ts:%Y-%m-%d} | stage={stage}"
            print(f"  [Warning] {message}")
            if stage == "Stage-I_dry":
                raise RuntimeError(message)
            continue
        stage_mean = stage_mean_lookup.get(stage, {})
        projection_rows.append({
            "init_date": init,
            "date": day_ts,
            "stage": stage,
            "ERA5_absT": float(stats_row.iloc[0]["ERA5_absT"]),
            "S2S_mean_absT": float(stats_row.iloc[0]["S2S_mean_absT"]),
            "S2S_spread": float(stats_row.iloc[0]["Spread_S2S"]),
            "OLS_daily_projection": float(np.nanmean(fitted_daily)),
            "OLS_daily_member_spread": float(np.nanstd(fitted_daily, ddof=1)),
            "OLS_stage_mean_absT": float(stage_mean.get("OLS_stage_mean_absT", np.nan)),
        })
    # Debug: reaggregate daily predictors back to stage scale and compare with Cell 2 stage predictors.
    comparison_rows = []
    for stage, components in components_by_stage.items():
        raw_stage = df_raw_all[(df_raw_all["init_date"] == init) & (df_raw_all["stage"] == stage)].copy()
        daily_stage = df_daily_member_raw[(df_daily_member_raw["init_date"] == init) & (df_daily_member_raw["stage"] == stage)].copy()
        if raw_stage.empty or daily_stage.empty:
            continue
        agg_map = {"NCHN_tp": "sum"}
        compare_cols = ["y_tmax", "sm_avg", "SSR_Avg", "SHF_Avg", "z500_anom_NCHN", "WNPSH", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        for col in compare_cols:
            if col in daily_stage.columns:
                agg_map[col] = "mean"
        daily_agg = daily_stage.groupby("member").agg(agg_map).reset_index()
        merged_raw = raw_stage[["member"] + [c for c in agg_map if c in raw_stage.columns]].merge(daily_agg, on="member", suffixes=("_stage", "_daily_reagg"))
        for col in agg_map:
            diff = merged_raw[f"{col}_daily_reagg"] - merged_raw[f"{col}_stage"]
            comparison_rows.append({"init_date": init, "stage": stage, "predictor_space": "raw", "factor": col, "max_abs_diff": float(np.nanmax(np.abs(diff))), "mean_abs_diff": float(np.nanmean(np.abs(diff)))})
        daily_resid_parts = []
        for _, day_group in daily_stage.groupby("date"):
            daily_resid_parts.append(prepare_daily_predictors_for_stage(day_group, stage, components))
        daily_resid_all = pd.concat(daily_resid_parts, ignore_index=True) if daily_resid_parts else pd.DataFrame()
        if not daily_resid_all.empty:
            resid_agg_map = {"NCHN_tp": "mean"}
            for col in [c for c in components["valid_factors"] if c != "NCHN_tp"]:
                if col in daily_resid_all.columns:
                    resid_agg_map[col] = "mean"
            daily_resid_agg = daily_resid_all.groupby("member").agg(resid_agg_map).reset_index()
            resid_stage = df_resid_all[(df_resid_all["init_date"] == init) & (df_resid_all["stage"] == stage)].copy()
            merged_resid = resid_stage[["member"] + [c for c in resid_agg_map if c in resid_stage.columns]].merge(daily_resid_agg, on="member", suffixes=("_stage", "_daily_reagg"))
            for col in resid_agg_map:
                diff = merged_resid[f"{col}_daily_reagg"] - merged_resid[f"{col}_stage"]
                comparison_rows.append({"init_date": init, "stage": stage, "predictor_space": "residualized_main", "factor": col, "max_abs_diff": float(np.nanmax(np.abs(diff))), "mean_abs_diff": float(np.nanmean(np.abs(diff)))})
    return pd.DataFrame(projection_rows), pd.DataFrame(setup_rows), pd.DataFrame(comparison_rows)

def display_corr_matrix(df, cols):
    corr = df[cols].corr()
    display_names = {col: label_map.get(col, col) for col in cols}
    return corr.rename(index=display_names, columns=display_names)

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    base = plt.get_cmap(cmap_name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc_{minval:.2f}_{maxval:.2f}",
        base(np.linspace(minval, maxval, n)),
    )

def lighten_color(color, amount=0.6):
    # amount 越大越接近白色；只改变绘图颜色，不改变任何表格数值。
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)

def readable_text_color(facecolor):
    rgba = mcolors.to_rgba(facecolor)
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.35 else "black"

# 1. Correlation matrices from raw/residualized CSVs.
for (init, stg), group in df_raw_all.groupby(["init_date", "stage"]):
    fig, ax = plt.subplots(figsize=(10, 8))
    raw_cols = ["y_tmax", "NCHN_tp", "sm_avg", "SSR_Avg", "SHF_Avg", "z500_anom_NCHN", "WNPSH", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
    raw_cols = [c for c in raw_cols if c in group.columns]
    sns.heatmap(display_corr_matrix(group, raw_cols), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title(f"Raw Member-axis Correlation | Init {init} | {stg}", fontsize=14)
    fig.savefig(f"{SUB_DIRS['corr_raw']}/corr_raw_{init}_{stg}_SHF.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    resid_group = df_resid_all[(df_resid_all["init_date"] == init) & (df_resid_all["stage"] == stg)]
    if not resid_group.empty:
        if stg == "Stage-I_dry":
            res_cols = ["y_tmax", "NCHN_tp", "sm_avg", "SSR_resid", "SHF_resid", "z500_anom_NCHN", "WNPSH", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        else:
            res_cols = ["y_tmax", "NCHN_tp", "SM_resid", "SSR_resid", "SHF_resid", "z500_anom_NCHN", "WNPSH", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        res_cols = [c for c in res_cols if c in resid_group.columns]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(display_corr_matrix(resid_group, res_cols), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, square=True)
        ax.set_title(f"Residualized Correlation | Init {init} | {stg} | {resid_strategies.get(stg, '')}", fontsize=14)
        fig.savefig(f"{SUB_DIRS['corr_resid']}/corr_resid_{init}_{stg}_SHF.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# 2. Combined OLS scatter + LMG bar figures.
def plot_ols_lmg_combo(summary_row, color, filename):
    init, stg, model_type = summary_row["init_date"], summary_row["stage"], summary_row["model_type"]
    lmg_dict = parse_lmg_dict(summary_row["lmg_dict"])
    fit_group = df_fitted_all[(df_fitted_all["init_date"] == init) & (df_fitted_all["stage"] == stg) & (df_fitted_all["model_type"] == model_type)]
    if fit_group.empty or not lmg_dict:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.05, 1.25]})
    axes[0].scatter(fit_group["y_tmax"], fit_group["fitted_value"], c=color, edgecolor="black", alpha=0.82)
    lim_min = float(np.nanmin([fit_group["y_tmax"].min(), fit_group["fitted_value"].min()]))
    lim_max = float(np.nanmax([fit_group["y_tmax"].max(), fit_group["fitted_value"].max()]))
    pad = max((lim_max - lim_min) * 0.08, 0.5)
    axes[0].plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], "k--", lw=1)
    axes[0].set_xlim(lim_min - pad, lim_max + pad)
    axes[0].set_ylim(lim_min - pad, lim_max + pad)
    axes[0].set_xlabel("ECMWF member Tmax (°C)")
    axes[0].set_ylabel("OLS fitted Tmax (°C)")
    axes[0].set_title(f"Predicted vs ECMWF | R²={summary_row['R2']:.2f}, RMSE={summary_row['Pred_vs_ECMWF_RMSE']:.2f}")

    facs, imps = zip(*sorted(lmg_dict.items(), key=lambda item: item[1]))
    clean_facs = [label_map.get(f, f) for f in facs]
    bars = axes[1].barh(clean_facs, imps, color=color, edgecolor="black")
    axes[1].set_xlabel("LMG relative importance (%)")
    axes[1].set_title("Relative Importance (LMG)")
    for bar in bars:
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}%", va="center")
    axes[1].set_xlim(0, max(imps) * 1.22 if max(imps) > 0 else 1)
    fig.suptitle(f"OLS + LMG | {model_type} | Init {init} | {stg}", fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close(fig)

for _, r in df_raw_ols.iterrows():
    plot_ols_lmg_combo(r, "#5B8FF9", f"{SUB_DIRS['ols_lmg_combo']}/ols_lmg_raw_{r['init_date']}_{r['stage']}_SHF.png")
for _, r in df_main_ols.iterrows():
    plot_ols_lmg_combo(r, "#2E8B57", f"{SUB_DIRS['ols_lmg_combo']}/ols_lmg_residualized_{r['init_date']}_{r['stage']}_SHF.png")
for _, r in df_sens_ols.iterrows():
    plot_ols_lmg_combo(r, "#E68600", f"{SUB_DIRS['ols_lmg_combo']}/ols_lmg_sensitivity_TPparam_{r['init_date']}_{r['stage']}_SHF.png")

# Preserve standalone LMG bar outputs for backward-compatible inspection.
for _, r in df_main_ols.iterrows():
    lmg_dict = parse_lmg_dict(r["lmg_dict"])
    if not lmg_dict:
        continue
    facs, imps = zip(*sorted(lmg_dict.items(), key=lambda item: item[1]))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh([label_map.get(f, f) for f in facs], imps, color="steelblue", edgecolor="black")
    ax.set_title(f"LMG Relative Importance | Init {r['init_date']} | {r['stage']} | {resid_strategies.get(r['stage'], '')}", fontsize=13)
    ax.set_xlabel("Importance (%)")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}%", va="center")
    ax.set_xlim(0, max(imps) * 1.22 if max(imps) > 0 else 1)
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['lmg_main']}/lmg_{r['init_date']}_{r['stage']}_SHF.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

for _, r in df_sens_ols.iterrows():
    lmg_dict = parse_lmg_dict(r["lmg_dict"])
    if not lmg_dict:
        continue
    facs, imps = zip(*sorted(lmg_dict.items(), key=lambda item: item[1]))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh([label_map.get(f, f) for f in facs], imps, color="darkorange", edgecolor="black")
    ax.set_title(f"Sensitivity LMG | Init {r['init_date']} | {r['stage']} | TP_param_resid", fontsize=13)
    ax.set_xlabel("Importance (%)")
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.1f}%", va="center")
    ax.set_xlim(0, max(imps) * 1.22 if max(imps) > 0 else 1)
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['lmg_sensitivity']}/sens_lmg_{r['init_date']}_{r['stage']}_SHF.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# 3. Tmax stage metrics, time-series plots, and heatmaps.
metrics_records = []
daily_ols_projection_records = []
daily_projection_consistency_records = []
daily_projection_setup_records = []
daily_stage_predictor_compare_records = []
daily_projection_inconsistent = False
if not df_daily_tmax.empty:
    df_daily_tmax["date"] = pd.to_datetime(df_daily_tmax["date"])
    for init in start_date_list_run:
        init_daily = df_daily_tmax[df_daily_tmax["init_date"] == init]
        daily_stats = init_daily.groupby("date").agg(
            S2S_mean_absT=("s2s_tmax", "mean"),
            Spread_S2S=("s2s_tmax", "std"),
            ERA5_absT=("era5_tmax", "mean"),
        ).reset_index()
        stage_mean_rows_for_init = []
        for stage, (w_start, w_end) in study_window_dict_run.items():
            fit_group = df_fitted_all[(df_fitted_all["init_date"] == init) & (df_fitted_all["stage"] == stage) & (df_fitted_all["model_type"] == "residualized_main")]
            if fit_group.empty:
                continue
            mask = (daily_stats["date"] >= pd.Timestamp(w_start)) & (daily_stats["date"] <= pd.Timestamp(w_end))
            era5_vals = daily_stats.loc[mask, "ERA5_absT"].values
            s2s_vals = daily_stats.loc[mask, "S2S_mean_absT"].values
            ols_mean = float(fit_group["fitted_value"].mean())
            ols_spread = float(fit_group["fitted_value"].std())
            r2_match = df_main_ols[(df_main_ols["init_date"] == init) & (df_main_ols["stage"] == stage)]
            r2_member_axis = float(r2_match.iloc[0]["R2"]) if not r2_match.empty else np.nan
            metric_record = {
                "init_date": init,
                "stage": stage,
                "ERA5_absT": float(np.nanmean(era5_vals)),
                "S2S_mean_absT": float(np.nanmean(s2s_vals)),
                "OLS_stage_mean_absT": ols_mean,
                "Bias_S2S_vs_ERA5": float(np.nanmean(s2s_vals - era5_vals)),
                "RMSE_S2S_vs_ERA5": float(np.sqrt(np.nanmean((s2s_vals - era5_vals) ** 2))),
                # R2_S2S_vs_ERA5 is computed along the daily time axis.
                "R2_S2S_vs_ERA5": safe_corr_r2(s2s_vals, era5_vals),
                # OLS stage-mean RMSE is a stage-level external consistency diagnostic, not a daily forecast skill score.
                "Bias_OLS_stage_mean_vs_ERA5": float(np.nanmean(ols_mean - era5_vals)),
                "RMSE_OLS_stage_mean_vs_ERA5": float(np.sqrt(np.nanmean((ols_mean - era5_vals) ** 2))),
                "Spread_S2S": float(np.nanmean(daily_stats.loc[mask, "Spread_S2S"].values)),
                "Spread_OLS_member_fitted": ols_spread,
                # R2_OLS_member_axis is computed along the ensemble-member axis.
                # It should not be interpreted as the same type of skill score as R2_S2S_vs_ERA5.
                "R2_OLS_member_axis": r2_member_axis,
            }
            metrics_records.append(metric_record)
            stage_mean_rows_for_init.append(metric_record)
        ols_daily_projection, setup_debug, predictor_compare = build_daily_ols_projection(init, daily_stats, stage_mean_rows_for_init)
        if not setup_debug.empty:
            daily_projection_setup_records.extend(setup_debug.to_dict("records"))
        if not predictor_compare.empty:
            daily_stage_predictor_compare_records.extend(predictor_compare.to_dict("records"))
        if not ols_daily_projection.empty:
            daily_ols_projection_records.extend(ols_daily_projection.to_dict("records"))

def plot_stage_mean_diagnostic_only(init, reason_text):
    init_daily = df_daily_tmax[df_daily_tmax["init_date"] == init]
    daily_stats = init_daily.groupby("date").agg(
        S2S_mean_absT=("s2s_tmax", "mean"),
        Spread_S2S=("s2s_tmax", "std"),
        ERA5_absT=("era5_tmax", "mean"),
    ).reset_index()
    stage_mean_rows = []
    for rec in metrics_records:
        if rec["init_date"] != init or rec["stage"] not in DAILY_PROJECTION_STAGES:
            continue
        w_start, w_end = study_window_dict_run[rec["stage"]]
        for d in daily_stats.loc[(daily_stats["date"] >= pd.Timestamp(w_start)) & (daily_stats["date"] <= pd.Timestamp(w_end)), "date"]:
            stage_mean_rows.append({"date": d, "OLS_stage_mean_absT": rec["OLS_stage_mean_absT"], "stage": rec["stage"]})
    stage_mean_df = pd.DataFrame(stage_mean_rows)
    if stage_mean_df.empty:
        print(f"  [Warning] skipped stage-mean diagnostic plot for {init}: no stage mean rows")
        return
    plot_df = daily_stats.merge(stage_mean_df, on="date", how="left")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), color="#F6D365", alpha=0.22, label="Stage-I_dry")
    ax.axvspan(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), pd.Timestamp("2023-06-18") + pd.Timedelta(hours=12), color="#CCCCCC", alpha=0.22, label="transition / break")
    ax.axvspan(pd.Timestamp("2023-06-18") + pd.Timedelta(hours=12), pd.Timestamp("2023-06-24") + pd.Timedelta(hours=18), color="#A1C4FD", alpha=0.22, label="Stage-II_wet")
    ax.plot(plot_df["date"], plot_df["ERA5_absT"], color="black", marker="o", lw=2.0, label="ERA5 Obs")
    ax.plot(plot_df["date"], plot_df["S2S_mean_absT"], color="#5B8FF9", marker="o", lw=1.8, label="S2S ensemble mean")
    ax.fill_between(plot_df["date"], plot_df["S2S_mean_absT"] - plot_df["Spread_S2S"], plot_df["S2S_mean_absT"] + plot_df["Spread_S2S"], color="#5B8FF9", alpha=0.18, label="S2S ±1 std")
    ax.step(plot_df["date"], plot_df["OLS_stage_mean_absT"], where="mid", color="#2E8B57", lw=1.8, linestyle="--", label="OLS stage-mean diagnostic only")
    ax.text(0.01, 0.98, reason_text, transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.set_title(f"Tmax Time Series | Stage-mean Diagnostic Only | Init {init}")
    ax.set_ylabel("Tmax (°C)")
    ax.set_xlabel("Date")
    ax.legend(loc="best", ncol=2, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['tmax_timeseries']}/tmax_timeseries_stage_mean_diagnostic_{init}_SHF.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_validated_daily_projection_timeseries(df_daily_projection_checked):
    for init in start_date_list_run:
        init_daily = df_daily_tmax[df_daily_tmax["init_date"] == init]
        daily_stats = init_daily.groupby("date").agg(
            S2S_mean_absT=("s2s_tmax", "mean"),
            Spread_S2S=("s2s_tmax", "std"),
            ERA5_absT=("era5_tmax", "mean"),
        ).reset_index()
        init_projection = df_daily_projection_checked[df_daily_projection_checked["init_date"] == init]
        plot_df = daily_stats.merge(init_projection[["date", "stage", "OLS_daily_projection", "OLS_daily_member_spread", "OLS_stage_mean_absT"]], on="date", how="left")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.axvspan(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), color="#F6D365", alpha=0.22, label="Stage-I_dry")
        ax.axvspan(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), pd.Timestamp("2023-06-18") + pd.Timedelta(hours=12), color="#CCCCCC", alpha=0.22, label="transition / break")
        ax.axvspan(pd.Timestamp("2023-06-18") + pd.Timedelta(hours=12), pd.Timestamp("2023-06-24") + pd.Timedelta(hours=18), color="#A1C4FD", alpha=0.22, label="Stage-II_wet")
        ax.plot(plot_df["date"], plot_df["ERA5_absT"], color="black", marker="o", lw=2.0, label="ERA5 Obs")
        ax.plot(plot_df["date"], plot_df["S2S_mean_absT"], color="#5B8FF9", marker="o", lw=1.8, label="S2S ensemble mean")
        ax.fill_between(plot_df["date"], plot_df["S2S_mean_absT"] - plot_df["Spread_S2S"], plot_df["S2S_mean_absT"] + plot_df["Spread_S2S"], color="#5B8FF9", alpha=0.18, label="S2S ±1 std")
        ax.plot(plot_df["date"], plot_df["OLS_daily_projection"], color="#2E8B57", marker="s", lw=1.8, label="Residualized OLS daily projection")
        ax.fill_between(plot_df["date"], plot_df["OLS_daily_projection"] - plot_df["OLS_daily_member_spread"], plot_df["OLS_daily_projection"] + plot_df["OLS_daily_member_spread"], color="#2E8B57", alpha=0.12, label="OLS daily member spread")
        ax.step(plot_df["date"], plot_df["OLS_stage_mean_absT"], where="mid", color="#2E8B57", lw=1.1, alpha=0.45, linestyle="--", label="OLS stage-mean diagnostic")
        total_metrics = next((m for m in metrics_records if m["init_date"] == init and m["stage"] == "Total"), None)
        if total_metrics:
            ax.text(0.01, 0.98, f"OLS stage mean vs ERA5 Total: Bias={total_metrics['Bias_OLS_stage_mean_vs_ERA5']:.2f}°C | RMSE={total_metrics['RMSE_OLS_stage_mean_vs_ERA5']:.2f}°C | member-axis R²={total_metrics['R2_OLS_member_axis']:.2f}\nOLS stage-mean RMSE is a stage-level external diagnostic, not daily forecast skill.", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        ax.set_title(f"Tmax Time Series | Init {init}")
        ax.set_ylabel("Tmax (°C)")
        ax.set_xlabel("Date")
        ax.legend(loc="best", ncol=2, fontsize=9)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(f"{SUB_DIRS['tmax_timeseries']}/tmax_timeseries_{init}_SHF.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

expected_daily_projection_pairs = {(init, stage) for init in start_date_list_run for stage in DAILY_PROJECTION_STAGES}

if daily_projection_setup_records:
    df_projection_setup_out = pd.DataFrame(daily_projection_setup_records)
    setup_pairs = set(zip(df_projection_setup_out["init_date"], df_projection_setup_out["stage"]))
    missing_setup_pairs = expected_daily_projection_pairs - setup_pairs
    if missing_setup_pairs:
        raise RuntimeError(f"daily projection model setup missing expected init/stage pairs: {sorted(missing_setup_pairs)}")
    df_projection_setup_out.to_csv(daily_projection_setup_csv, index=False)
else:
    raise RuntimeError("daily projection model setup table is empty; cannot validate Stage-I/Stage-II projection setup")

if daily_stage_predictor_compare_records:
    df_predictor_compare_out = pd.DataFrame(daily_stage_predictor_compare_records)
    compare_pairs = set(zip(df_predictor_compare_out["init_date"], df_predictor_compare_out["stage"]))
    missing_compare_pairs = expected_daily_projection_pairs - compare_pairs
    if missing_compare_pairs:
        raise RuntimeError(f"daily-to-stage predictor comparison missing expected init/stage pairs: {sorted(missing_compare_pairs)}")
    df_predictor_compare_out.to_csv(daily_stage_predictor_compare_csv, index=False)
    predictor_diff_threshold = 1e-3
    predictor_warning_threshold = 1e-5
    warning_predictor_diffs = df_predictor_compare_out[(df_predictor_compare_out["max_abs_diff"].abs() > predictor_warning_threshold) & (df_predictor_compare_out["max_abs_diff"].abs() <= predictor_diff_threshold)]
    if not warning_predictor_diffs.empty:
        warning_details = warning_predictor_diffs[["init_date", "stage", "predictor_space", "factor", "max_abs_diff"]].to_dict("records")
        for rec in warning_details:
            print(f"  [Warning] daily-to-stage predictor mismatch within tolerance: init={rec['init_date']}, stage={rec['stage']}, predictor_space={rec['predictor_space']}, factor={rec['factor']}, max_abs_diff={rec['max_abs_diff']}")
    bad_predictor_diffs = df_predictor_compare_out[df_predictor_compare_out["max_abs_diff"].abs() > predictor_diff_threshold]
    if not bad_predictor_diffs.empty:
        details = bad_predictor_diffs[["init_date", "stage", "predictor_space", "factor", "max_abs_diff"]].to_dict("records")
        for rec in details:
            print(f"  [Error] daily-to-stage predictor mismatch: init={rec['init_date']}, stage={rec['stage']}, predictor_space={rec['predictor_space']}, factor={rec['factor']}, max_abs_diff={rec['max_abs_diff']}")
        raise RuntimeError(f"daily-to-stage predictor comparison exceeded threshold {predictor_diff_threshold}: {details}")
else:
    raise RuntimeError("daily-to-stage predictor comparison table is empty; cannot validate daily predictors against stage predictors")

if daily_ols_projection_records:
    df_daily_projection_out = pd.DataFrame(daily_ols_projection_records).sort_values(["init_date", "date", "stage"]).reset_index(drop=True)
    df_daily_projection_out.to_csv(daily_ols_projection_csv, index=False)
else:
    raise RuntimeError("daily OLS projection table was not generated; check table_daily_member_raw_factors_SHF.csv availability")

# Re-read the final saved projection table before all final row-count and consistency checks.
df_daily_projection_check = pd.read_csv(daily_ols_projection_csv)
df_daily_projection_check["date"] = pd.to_datetime(df_daily_projection_check["date"])
if len(df_daily_projection_check) != EXPECTED_DAILY_PROJECTION_ROWS:
    raise RuntimeError(f"daily OLS projection row check failed: expected {EXPECTED_DAILY_PROJECTION_ROWS}, got {len(df_daily_projection_check)}; file={daily_ols_projection_csv}")
projection_pairs = set(zip(df_daily_projection_check["init_date"], df_daily_projection_check["stage"]))
missing_projection_pairs = expected_daily_projection_pairs - projection_pairs
if missing_projection_pairs:
    raise RuntimeError(f"daily OLS projection missing expected init/stage pairs: {sorted(missing_projection_pairs)}")
for (init, stage), group in df_daily_projection_check.groupby(["init_date", "stage"]):
    expected_days = EXPECTED_DAILY_PROJECTION_STAGE_DAYS[stage]
    actual_days = group["date"].nunique()
    if actual_days != expected_days:
        raise RuntimeError(f"daily OLS projection day-count check failed for {init}/{stage}: expected {expected_days}, got {actual_days}")

# Consistency is intentionally computed from the final saved CSV, not from an in-memory pre-write object.
daily_projection_consistency_records = []
daily_projection_inconsistent = False
for (init, stage), group in df_daily_projection_check.groupby(["init_date", "stage"], sort=False):
    stage_mean_absT = float(group["OLS_stage_mean_absT"].iloc[0])
    daily_mean = float(group["OLS_daily_projection"].mean())
    if not np.isfinite(daily_mean) or not np.isfinite(stage_mean_absT):
        raise RuntimeError(f"daily OLS projection finite check failed for {init}/{stage}: mean_daily={daily_mean}, OLS_stage_mean_absT={stage_mean_absT}")
    diff = daily_mean - stage_mean_absT
    warn_flag = abs(diff) > 0.05
    daily_projection_inconsistent = daily_projection_inconsistent or warn_flag
    daily_projection_consistency_records.append({
        "init_date": init,
        "stage": stage,
        "n_days": int(group["date"].nunique()),
        "mean_over_days_OLS_daily_projection": daily_mean,
        "OLS_stage_mean_absT": stage_mean_absT,
        "difference": diff,
        "abs_difference": abs(diff),
        "warning_abs_diff_gt_0p05C": warn_flag,
        "checked_from_final_saved_csv": True,
    })
pd.DataFrame(daily_projection_consistency_records).to_csv(daily_projection_consistency_csv, index=False)
bad_consistency = [rec for rec in daily_projection_consistency_records if rec["warning_abs_diff_gt_0p05C"]]
if bad_consistency:
    reason = "Daily OLS projection failed the ±0.05°C consistency check; this robust figure shows only the stage-mean diagnostic."
    for init in start_date_list_run:
        plot_stage_mean_diagnostic_only(init, reason)
    for rec in bad_consistency:
        print(f"  [Warning] Daily OLS projection consistency check failed for {rec['init_date']} | {rec['stage']}: mean_daily={rec['mean_over_days_OLS_daily_projection']:.3f}, stage_mean={rec['OLS_stage_mean_absT']:.3f}, diff={rec['difference']:.3f}°C")
    raise RuntimeError(f"daily OLS projection consistency failed after re-reading final CSV: {bad_consistency}")

# All final CSV checks passed; only now write the formal Tmax time-series figures.
plot_validated_daily_projection_timeseries(df_daily_projection_check)

df_metrics = pd.DataFrame(metrics_records)
if not df_metrics.empty:
    for bad_col in ["R2_OLS_vs_ERA5", "Bias_OLS_vs_ERA5", "RMSE_OLS_vs_ERA5", "OLS_mean_absT", "Spread_OLS"]:
        assert bad_col not in df_metrics.columns, f"Deprecated metric column still exists: {bad_col}"
    for good_col in [
        "OLS_stage_mean_absT",
        "Bias_OLS_stage_mean_vs_ERA5",
        "RMSE_OLS_stage_mean_vs_ERA5",
        "Spread_OLS_member_fitted",
        "R2_OLS_member_axis",
    ]:
        assert good_col in df_metrics.columns, f"Expected metric column missing: {good_col}"

    df_metrics.to_csv(f"{SUB_DIRS['tables']}/table_tmax_stage_metrics_SHF.csv", index=False)
    df_metrics["init_stage"] = df_metrics["init_date"].str.replace("2023-06-", "06", regex=False) + "_" + df_metrics["stage"].map(clean_stage_name)
    heatmap_specs = [
        ("absolute_temperature_heatmap_SHF.png", ["ERA5_absT", "S2S_mean_absT", "OLS_stage_mean_absT"], "Absolute Temperature Heatmap (°C)", "YlOrRd"),
        ("error_metrics_heatmap_SHF.png", ["Bias_S2S_vs_ERA5", "RMSE_S2S_vs_ERA5", "Bias_OLS_stage_mean_vs_ERA5", "RMSE_OLS_stage_mean_vs_ERA5"], "Error Metrics Heatmap (external bias/RMSE only; OLS stage-mean RMSE is not daily forecast skill)", "coolwarm"),
        ("spread_heatmap_SHF.png", ["Spread_S2S", "Spread_OLS_member_fitted"], "Spread Heatmap (member std)", "viridis"),
        ("ols_member_axis_r2_heatmap_SHF.png", ["R2_OLS_member_axis"], "OLS Fitting Quality Heatmap (member-axis R²)", "YlGnBu"),
        ("s2s_daily_tracking_r2_heatmap_SHF.png", ["R2_S2S_vs_ERA5"], "S2S Daily Tracking Heatmap (daily time-axis R²)", "PuBuGn"),
    ]
    for fname, cols, title, cmap in heatmap_specs:
        plot_mat = df_metrics.set_index("init_stage")[cols]
        fig_h = max(4, 0.55 * len(plot_mat) + 1.5)
        fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(cols)), fig_h))
        sns.heatmap(plot_mat, annot=True, fmt=".2f", cmap=cmap, ax=ax, linewidths=0.5, linecolor="white")
        ax.set_title(f"{title}")
        ax.set_xlabel("")
        ax.set_ylabel("init_stage")
        fig.tight_layout()
        fig.savefig(f"{SUB_DIRS['metrics_heatmap']}/{fname}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    summary_rows = []
    stage_order = ["Stage-I_dry", "Stage-II_wet", "Total"]
    era5_stage = df_metrics.drop_duplicates("stage").set_index("stage")
    for stage in stage_order:
        if stage in era5_stage.index:
            summary_rows.append({"block": "ERA5 observation", "row_metric": "ERA5_absT", "stage": stage, "value": float(era5_stage.loc[stage, "ERA5_absT"]), "unit": "degC"})
    for init in start_date_list_run:
        init_metrics = df_metrics[df_metrics["init_date"] == init].set_index("stage")
        for metric in ["S2S_mean_absT", "Bias_S2S_vs_ERA5", "RMSE_S2S_vs_ERA5", "Spread_S2S", "OLS_stage_mean_absT", "Bias_OLS_stage_mean_vs_ERA5", "RMSE_OLS_stage_mean_vs_ERA5", "Spread_OLS_member_fitted"]:
            for stage in stage_order:
                if stage in init_metrics.index:
                    summary_rows.append({"block": f"Init {init}", "row_metric": metric, "stage": stage, "value": float(init_metrics.loc[stage, metric]), "unit": "degC"})
    df_abs_error_summary = pd.DataFrame(summary_rows)
    df_abs_error_summary.to_csv(f"{SUB_DIRS['tables']}/table_tmax_abs_error_summary.csv", index=False)

    row_keys = df_abs_error_summary[["block", "row_metric"]].drop_duplicates().itertuples(index=False, name=None)
    row_keys = list(row_keys)
    pivot = df_abs_error_summary.pivot_table(index=["block", "row_metric"], columns="stage", values="value", aggfunc="first").reindex(row_keys)
    pivot = pivot.reindex(columns=stage_order)
    all_bias = df_abs_error_summary[df_abs_error_summary["row_metric"].str.contains("Bias")]["value"].abs().values
    all_rmse = df_abs_error_summary[df_abs_error_summary["row_metric"].str.contains("RMSE")]["value"].values
    all_spread = df_abs_error_summary[df_abs_error_summary["row_metric"].str.contains("Spread")]["value"].values
    bias_lim = max(float(np.nanpercentile(all_bias[np.isfinite(all_bias)], 95)) if np.isfinite(all_bias).any() else 1.0, 0.1)
    rmse_max = max(float(np.nanpercentile(all_rmse[np.isfinite(all_rmse)], 95)) if np.isfinite(all_rmse).any() else 1.0, 0.1)
    spread_max = max(float(np.nanpercentile(all_spread[np.isfinite(all_spread)], 95)) if np.isfinite(all_spread).any() else 1.0, 0.1)
    bias_norm = mcolors.TwoSlopeNorm(vmin=-bias_lim, vcenter=0.0, vmax=bias_lim)
    rmse_norm = mcolors.Normalize(vmin=0.0, vmax=rmse_max)
    spread_norm = mcolors.Normalize(vmin=0.0, vmax=spread_max)
    # Truncated + whitened colormaps keep the table paper-like: color supports interpretation without overpowering numbers.
    bias_cmap = truncate_colormap("RdBu_r", 0.20, 0.80)
    rmse_cmap = truncate_colormap("Oranges", 0.05, 0.45)
    spread_cmap = truncate_colormap("BuGn", 0.05, 0.45)
    abs_temp_facecolor = "#fbfbfb"
    bias_lighten = 0.62
    rmse_lighten = 0.58
    spread_lighten = 0.58

    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.42 * len(pivot) + 2.0)))
    ax.set_xlim(-2.7, len(stage_order))
    ax.set_ylim(len(pivot), -1.2)
    ax.axis("off")
    for j, stage in enumerate(stage_order):
        ax.text(j + 0.5, -0.35, stage, ha="center", va="center", fontweight="bold")
    last_block = None
    for i, ((block, metric), row) in enumerate(pivot.iterrows()):
        if block != last_block:
            ax.text(-2.62, i + 0.5, block, ha="left", va="center", fontweight="bold")
            last_block = block
        ax.text(-0.2, i + 0.5, metric, ha="right", va="center")
        for j, stage in enumerate(stage_order):
            val = row.get(stage, np.nan)
            if not np.isfinite(val):
                facecolor = "white"
                label = ""
            elif "Bias" in metric:
                facecolor = lighten_color(bias_cmap(bias_norm(np.clip(val, -bias_lim, bias_lim))), amount=bias_lighten)
                label = f"{val:.2f}°C"
            elif "RMSE" in metric:
                facecolor = lighten_color(rmse_cmap(rmse_norm(np.clip(max(val, 0.0), 0.0, rmse_max))), amount=rmse_lighten)
                label = f"{val:.2f}°C"
            elif "Spread" in metric:
                facecolor = lighten_color(spread_cmap(spread_norm(np.clip(max(val, 0.0), 0.0, spread_max))), amount=spread_lighten)
                label = f"{val:.2f}°C"
            else:
                facecolor = abs_temp_facecolor
                label = f"{val:.2f}°C"
            ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor=facecolor, edgecolor="#e0e0e0", linewidth=1.0))
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=10, color=readable_text_color(facecolor))
    ax.set_title("Tmax Absolute Temperature and Error Summary", fontsize=14, pad=20)
    ax.text(0.5, 1.01, "Absolute temperature rows are unshaded; Bias, RMSE, and Spread use separate visual encodings.", transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{SUB_DIRS['metrics_heatmap']}/fig_tmax_abs_error_summary_table.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
else:
    print("  [Warning] table_daily_tmax_SHF.csv is missing/empty; skipped Tmax time-series and metric heatmaps.")

print("\n" + "=" * 80)
print("【终端 Summary】")
print(f"ERA5 Tmax file path: {ERA5_TMAX_FILE}")
if daily_projection_setup_records:
    print("[Daily OLS projection setup check]")
    for rec in daily_projection_setup_records:
        print(f"  {rec['init_date']} | {rec['stage']}: coefficients={rec['same_ols_coefficients_as_stage_model']}, intercept={rec['same_intercept_as_stage_model']}, scaler={rec['same_scaler_as_stage_model']}, residualization={rec['same_residualization_equations_as_stage_model']}, SHF/sign={rec['same_factor_sign_convention']}, double_baseline={rec['baseline_double_added']}")
if daily_projection_consistency_records:
    print("[Daily OLS projection consistency]")
    for rec in daily_projection_consistency_records:
        status = "WARNING" if rec["warning_abs_diff_gt_0p05C"] else "OK"
        print(f"  {rec['init_date']} | {rec['stage']}: mean_daily={rec['mean_over_days_OLS_daily_projection']:.3f}, OLS_stage_mean={rec['OLS_stage_mean_absT']:.3f}, diff={rec['difference']:.3f}°C [{status}]")
if daily_stage_predictor_compare_records:
    max_diff = max(abs(rec["max_abs_diff"]) for rec in daily_stage_predictor_compare_records)
    print(f"[Daily→stage predictor comparison] max_abs_diff={max_diff:.6g}; tolerance=1e-3; details: {daily_stage_predictor_compare_csv}")
formula_pass = bool((df_residual_audit["manual_max_abs_diff"] < 1e-8).all())
ortho_pass = bool((df_residual_audit["corr_resid_control_after"].isna() | (df_residual_audit["corr_resid_control_after"].abs() < 1e-8)).all())
df_keycase_0612 = df_residual_keycase[(df_residual_keycase["init_date"] == "2023-06-12") & (df_residual_keycase["stage"] == "Stage-I_dry")]
keycase_pass = (not df_keycase_0612.empty) and bool((df_keycase_0612["diff"].abs() < 1e-8).all())
print(f"[Residual formula check] {'PASS' if formula_pass else 'FAIL'}")
print(f"[Residual orthogonality check] {'PASS' if ortho_pass else 'FAIL'}")
print(f"[Keycase 0612 Stage-I SHF_resid check] {'PASS' if keycase_pass else 'FAIL'}")
print(f"[Residual direction reversed?] {'YES' if not formula_pass else 'NO'}")
if formula_pass:
    print("[Interpretation note] No residual formula reversal detected. A stronger |corr(Tmax, SHF_resid)| than |corr(Tmax, SHF_Avg)| can occur as a semi-partial (suppressor-like) effect.")
for _, r in df_main_ols.iterrows():
    init = r["init_date"]
    stg = r["stage"]
    lmg = parse_lmg_dict(r["lmg_dict"])
    print(f"[{init} | {stg}]")
    print(f"  OLS main model: R²={r['R2']:.2f}, Adj-R²={r['Adj_R2']:.2f}, Top1={r['Top1_factor_name']} ({r['Top1_factor_importance_pct']:.1f}%)")
    if stg == "Stage-I_dry":
        sum_imp = lmg.get("sm_avg", 0) + lmg.get("SHF_resid", 0) + lmg.get("SSR_resid", 0)
        print(f"  [Coupling Impt] sm_avg + SHF_resid + SSR_resid = {sum_imp:.1f}%")
    else:
        sum_imp = lmg.get("NCHN_tp", 0) + lmg.get("SM_resid", 0) + lmg.get("SSR_resid", 0) + lmg.get("SHF_resid", 0)
        print(f"  [Coupling Impt] NCHN_tp + SM_resid + SSR_resid + SHF_resid = {sum_imp:.1f}%")
    if not df_metrics.empty:
        mm = df_metrics[(df_metrics["init_date"] == init) & (df_metrics["stage"] == stg)]
        if not mm.empty:
            m = mm.iloc[0]
            print(f"  S2S vs ERA5: Bias={m['Bias_S2S_vs_ERA5']:.2f}, RMSE={m['RMSE_S2S_vs_ERA5']:.2f}, daily time-axis R²={m['R2_S2S_vs_ERA5']:.2f}, Spread={m['Spread_S2S']:.2f}")
            print(f"  OLS stage mean vs ERA5: Bias={m['Bias_OLS_stage_mean_vs_ERA5']:.2f}, RMSE={m['RMSE_OLS_stage_mean_vs_ERA5']:.2f}, Spread={m['Spread_OLS_member_fitted']:.2f}")
            print(f"  OLS member-axis fitting: R²={m['R2_OLS_member_axis']:.2f}")
    if not df_factor_direction.empty:
        diag = df_factor_direction[(df_factor_direction["init_date"] == init) & (df_factor_direction["stage"] == stg)]
        diag = diag[diag["factor"].isin(["SHF_Avg", "SHF_resid", "sm_avg", "SSR_resid", "z500_anom_NCHN"])]
        if not diag.empty:
            print("  Factor direction diagnostics (member-axis r / standardized coef):")
            for _, drow in diag.iterrows():
                factor_label = "SHF_Avg(up)" if drow["factor"] == "SHF_Avg" else drow["factor"]
                print(f"    [{drow['model_kind']}] {factor_label}: r={drow['pearson_r_with_y']:.2f}, beta_std={drow['ols_coef_standardized']:.2f}")
        neg_shf = diag[(diag["model_kind"] == "residualized_main") & (diag["factor"] == "SHF_resid") & (diag["pearson_r_with_y"] < 0) & (diag["ols_coef_standardized"] < 0)]
        if stg == "Stage-I_dry" and not neg_shf.empty:
            print("  SHF_resid shows negative member-axis association with Tmax; this is a physical/statistical sign, not a plotting error.")
    if stg == "Stage-II_wet" and not df_sens_ols.empty:
        sens_match = df_sens_ols[(df_sens_ols["init_date"] == init) & (df_sens_ols["stage"] == stg)]
        if not sens_match.empty:
            sens_lmg = parse_lmg_dict(sens_match.iloc[0]["lmg_dict"])
            print(f"  [Sensitivity] TP_param_resid Importance = {sens_lmg.get('TP_param_resid', 0):.1f}%")
print("=" * 80)
print("SHF sign convention: SHF_Avg was converted in Cell 1 and stored as upward-positive; Cell 2/3 do not flip it again.")
print(f"Final output root: {RESID_OLS_DIR}")
print(f"Stage-Specific Residualized OLS (SHF Version) 执行完毕！结果位于: {RESID_OLS_DIR}")
print("=" * 80)

# %%
# =============================================================================
# Cell 4: SSR collinearity sensitivity / SSR_sensitivity (independent outputs)
# =============================================================================

set_audit_cell_name("Cell 4: SSR collinearity sensitivity")
import os
import ast
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)

print("\n" + "=" * 80)
print("Cell 4: SSR collinearity sensitivity / SSR_sensitivity ...")
print("=" * 80)

def assert_output_under_v7(path):
    assert_four_init_output(path)

def safe_corr(a, b):
    if len(a) < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]

def compute_lmg_mc(X, y, feature_names, mc_samples=2000):
    np.random.seed(42)
    n_samples, n_features = X.shape
    if n_samples < n_features + 2:
        return {name: np.nan for name in feature_names}
    X_s = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    y_s = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)
    lmg_raw = np.zeros(n_features)
    model = LinearRegression()
    for _ in range(mc_samples):
        order = np.random.permutation(n_features)
        prev_r2 = 0.0
        for i in range(n_features):
            idx = order[:i + 1]
            r2 = model.fit(X_s[:, idx], y_s).score(X_s[:, idx], y_s)
            lmg_raw[order[i]] += max(0, (r2 - prev_r2))
            prev_r2 = r2
    tot = lmg_raw.sum()
    if tot == 0:
        return {name: 0.0 for name in feature_names}
    return {name: val for name, val in zip(feature_names, (lmg_raw / tot) * 100.0)}

def fit_ols_lmg(df, factor_names, model_type, residualization_strategy):
    valid_factors = [f for f in factor_names if f in df.columns and df[f].std() >= 1e-12]
    dropped_factors = [f for f in factor_names if f not in valid_factors]
    X = df[valid_factors].values
    y = df["y_tmax"].values
    model = sm.OLS(y, sm.add_constant(X)).fit()
    y_pred = model.fittedvalues
    lmg_dict = compute_lmg_mc(X, y, valid_factors)
    sorted_lmg = sorted(lmg_dict.items(), key=lambda item: item[1], reverse=True)
    summary = {
        "init_date": df["init_date"].iloc[0],
        "stage": df["stage"].iloc[0],
        "model_type": model_type,
        "residualization_strategy": residualization_strategy,
        "R2": model.rsquared,
        "Adj_R2": model.rsquared_adj,
        "Pred_vs_ECMWF_Corr": safe_corr(y_pred, y),
        "Pred_vs_ECMWF_RMSE": np.sqrt(np.mean((y_pred - y) ** 2)),
        "Pred_vs_ECMWF_Bias": np.mean(y_pred - y),
        "Top1_factor_name": sorted_lmg[0][0] if sorted_lmg else np.nan,
        "Top1_factor_importance_pct": sorted_lmg[0][1] if sorted_lmg else np.nan,
        "dropped_factors": dropped_factors,
        "lmg_dict": lmg_dict,
    }
    fitted = df[["init_date", "stage", "member", "y_tmax"]].copy()
    fitted["model_type"] = model_type
    fitted["fitted_value"] = y_pred
    fitted["residual"] = y - y_pred
    return summary, lmg_dict, fitted

def build_lmg_long_rows(summary_row, lmg_dict):
    return [
        {
            "init_date": summary_row["init_date"],
            "stage": summary_row["stage"],
            "model_type": summary_row["model_type"],
            "residualization_strategy": summary_row["residualization_strategy"],
            "factor_name": factor_name,
            "relative_importance_pct": importance,
            "R2": summary_row["R2"],
            "Adj_R2": summary_row["Adj_R2"],
            "dropped_factors": ";".join(summary_row.get("dropped_factors", [])),
        }
        for factor_name, importance in lmg_dict.items()
    ]

def residualized_main_factors_for_stage(stage):
    if stage == "Stage-I_dry":
        return ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "sm_avg", "SSR_resid", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
    return ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "SM_resid", "SSR_resid", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]

BASE_RESID_DIR = FOUR_INIT_ROOT
base_tables = f"{BASE_RESID_DIR}/tables"
SSR_SENS_DIR = f"{FOUR_INIT_ROOT}/optional_ssr_sensitivity"
SSR_SUB_DIRS = {
    "tables": f"{SSR_SENS_DIR}/tables",
    "figures": f"{SSR_SENS_DIR}/figures",
}
for p in [BASE_RESID_DIR, base_tables, SSR_SENS_DIR] + list(SSR_SUB_DIRS.values()):
    assert_output_under_v7(p)
for d in SSR_SUB_DIRS.values():
    os.makedirs(d, exist_ok=True)

req_files = {
    "raw": f"{base_tables}/table_stage_member_raw_factors_SHF.csv",
    "resid": f"{base_tables}/table_stage_member_residualized_factors_SHF.csv",
    "full_ols": f"{base_tables}/table_stage_ols_lmg_SHF.csv",
    "full_lmg_long": f"{base_tables}/table_stage_lmg_long_SHF.csv",
    "resid_audit": f"{base_tables}/residualization_audit.csv",
    "predictor_list": f"{base_tables}/residualized_model_predictor_list.csv",
}
for k, fp in req_files.items():
    if not os.path.exists(fp):
        raise RuntimeError(f"Missing required base CSV [{k}]: {fp}")

df_raw_base = pd.read_csv(req_files["raw"])
df_resid_base = pd.read_csv(req_files["resid"])
df_full_ols = pd.read_csv(req_files["full_ols"])
df_full_lmg_long = pd.read_csv(req_files["full_lmg_long"])
pd.read_csv(req_files["resid_audit"])
pd.read_csv(req_files["predictor_list"])

def _resid_series(y, Xdf):
    Xc = sm.add_constant(Xdf, has_constant="add")
    model = sm.OLS(y, Xc).fit()
    pred = model.predict(Xc)
    return (y - pred), pred

def _pairwise_max_abs_corr(df, cols):
    if len(cols) < 2:
        return np.nan
    c = df[cols].astype(float).corr().abs()
    np.fill_diagonal(c.values, np.nan)
    return float(np.nanmax(c.values))

def _condition_number(df, cols):
    X0 = df[cols].astype(float).values
    Xz = (X0 - np.nanmean(X0, axis=0)) / (np.nanstd(X0, axis=0) + 1e-8)
    X = sm.add_constant(Xz, has_constant="add")
    return float(np.linalg.cond(X))

ssr_audit_rows, sens_summary_rows, sens_lmg_rows, sens_fitted_rows, compare_rows, vif_rows = [], [], [], [], [], []
stage_order = ["Stage-I_dry", "Stage-II_wet", "Total"]

for (init, stage), g_resid in df_resid_base.groupby(["init_date", "stage"], sort=False):
    g_resid0 = g_resid.copy()
    g_raw0 = df_raw_base[(df_raw_base["init_date"] == init) & (df_raw_base["stage"] == stage)].copy()

    if stage == "Stage-I_dry":
        stage_control = "sm_avg"
        no_ssr_predictors = ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "sm_avg", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        ssrrel_predictors = ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "sm_avg", "SHF_resid", "SSR_relSHF", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
    else:
        stage_control = "NCHN_tp"
        no_ssr_predictors = ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "SM_resid", "SHF_resid", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
        ssrrel_predictors = ["WNPSH", "z500_anom_NCHN", "NCHN_tp", "SM_resid", "SHF_resid", "SSR_relSHF", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", "NCVI"]
    resid_required = ["member", "y_tmax"] + no_ssr_predictors
    raw_required = ["member", "SSR_Avg", stage_control]
    missing_resid_cols = [c for c in resid_required if c not in g_resid0.columns]
    missing_raw_cols = [c for c in raw_required if c not in g_raw0.columns]
    if missing_resid_cols or missing_raw_cols:
        raise RuntimeError(
            f"SSR sensitivity required columns missing for {init}/{stage}: "
            f"missing_resid_cols={missing_resid_cols}, missing_raw_cols={missing_raw_cols}"
        )
    g_resid = g_resid0.dropna(subset=resid_required).sort_values("member").reset_index(drop=True)
    g_raw = g_raw0.dropna(subset=raw_required).sort_values("member").reset_index(drop=True)
    if len(g_resid) != 51:
        nan_counts = g_resid0[resid_required].isna().sum().to_dict()
        raise RuntimeError(
            f"SSR sensitivity residualized group size check failed for {init}/{stage}: "
            f"expected 51 after subset dropna, got {len(g_resid)}; "
            f"required_cols={resid_required}; nan_counts={nan_counts}"
        )
    if len(g_raw) != 51:
        nan_counts = g_raw0[raw_required].isna().sum().to_dict()
        raise RuntimeError(
            f"SSR sensitivity raw group size check failed for {init}/{stage}: "
            f"expected 51 after subset dropna, got {len(g_raw)}; "
            f"required_cols={raw_required}; nan_counts={nan_counts}"
        )
    if not np.array_equal(g_raw["member"].values, g_resid["member"].values):
        raise RuntimeError(
            f"member order mismatch between raw and residualized tables for {init}/{stage}; "
            "please merge by member before SSR sensitivity calculation."
        )
    ssr_tmp, _ = _resid_series(g_raw["SSR_Avg"], g_raw[[stage_control]])

    ssr_tmp_series = pd.Series(np.asarray(ssr_tmp, dtype=float), index=g_resid.index)
    ssr_relshf, ssr_tmp_fit = _resid_series(ssr_tmp_series, g_resid[["SHF_resid"]])
    g_sens = g_resid.copy()
    g_sens["SSR_relSHF"] = ssr_relshf.values
    manual_max_abs_diff = float(np.nanmax(np.abs(g_sens["SSR_relSHF"].values - ssr_relshf.values)))
    corr_before = safe_corr(np.asarray(ssr_tmp), g_resid["SHF_resid"].values)
    corr_after_shf = safe_corr(g_sens["SSR_relSHF"].values, g_resid["SHF_resid"].values)
    corr_after_stage_control = safe_corr(g_sens["SSR_relSHF"].values, g_raw[stage_control].values)
    resid_formula_ok = (manual_max_abs_diff < 1e-8) and (np.isnan(corr_after_shf) or abs(corr_after_shf) < 1e-8) and (np.isnan(corr_after_stage_control) or abs(corr_after_stage_control) < 1e-8)
    ssr_audit_rows.append({
        "init_date": init, "stage": stage, "model_type": "residualized_SSRrelSHF", "resid_name": "SSR_relSHF",
        "first_control": stage_control, "second_control": "SHF_resid", "n_members": len(g_sens),
        "corr_SSRtmp_SHFresid_before": corr_before, "corr_SSRrelSHF_SHFresid_after": corr_after_shf,
        "corr_SSRrelSHF_stage_control_after": corr_after_stage_control, "manual_max_abs_diff": manual_max_abs_diff,
        "resid_formula_ok": resid_formula_ok,
    })
    if not resid_formula_ok:
        raise RuntimeError(f"SSR_relSHF audit failed for {init}/{stage}")

    for mt, preds in [("residualized_noSSR", no_ssr_predictors), ("residualized_SSRrelSHF", ssrrel_predictors)]:
        summary, lmg_dict, fitted = fit_ols_lmg(g_sens, preds, mt, "SSR sensitivity")
        summary["predictors_used"] = ";".join(preds)
        summary["n_predictors"] = len(preds)
        summary["n_members"] = len(g_sens)
        sens_summary_rows.append(summary)
        sens_lmg_rows.extend(build_lmg_long_rows(summary, lmg_dict))
        sens_fitted_rows.extend(fitted.to_dict("records"))
        max_pair = _pairwise_max_abs_corr(g_sens, preds)
        cond_num = _condition_number(g_sens, preds)
        Xv = sm.add_constant(g_sens[preds], has_constant="add")
        for i, fac in enumerate(preds, start=1):
            try:
                v = variance_inflation_factor(Xv.values, i)
            except Exception:
                v = np.nan
            vif_rows.append({"init_date": init, "stage": stage, "model_type": mt, "factor": fac, "VIF": float(v) if np.isfinite(v) else np.nan, "max_abs_pairwise_corr": max_pair, "condition_number": cond_num})

for (init, stage) in sorted({(r["init_date"], r["stage"]) for r in sens_summary_rows}):
    b = df_full_ols[(df_full_ols["init_date"] == init) & (df_full_ols["stage"] == stage) & (df_full_ols["model_type"] == "residualized_main")].iloc[0]
    n = [r for r in sens_summary_rows if r["init_date"] == init and r["stage"] == stage and r["model_type"] == "residualized_noSSR"][0]
    s = [r for r in sens_summary_rows if r["init_date"] == init and r["stage"] == stage and r["model_type"] == "residualized_SSRrelSHF"][0]
    compare_rows.append({
        "init_date": init, "stage": stage,
        "baseline_R2": b["R2"], "baseline_Adj_R2": b["Adj_R2"],
        "noSSR_R2": n["R2"], "noSSR_Adj_R2": n["Adj_R2"],
        "SSRrelSHF_R2": s["R2"], "SSRrelSHF_Adj_R2": s["Adj_R2"],
        "delta_Adj_R2_noSSR_minus_full": n["Adj_R2"] - b["Adj_R2"],
        "delta_Adj_R2_SSRrelSHF_minus_full": s["Adj_R2"] - b["Adj_R2"],
        "top1_full": b["Top1_factor_name"], "top1_noSSR": n["Top1_factor_name"], "top1_SSRrelSHF": s["Top1_factor_name"],
        "SSR_removed_R2_retention": (n["R2"] / b["R2"]) if b["R2"] != 0 else np.nan,
    })
    full_preds = residualized_main_factors_for_stage(stage)
    full_required = ["member", "y_tmax"] + full_preds
    df_grp0 = df_resid_base[(df_resid_base["init_date"] == init) & (df_resid_base["stage"] == stage)].copy()
    df_grp = df_grp0.dropna(subset=full_required).sort_values("member").reset_index(drop=True)
    if len(df_grp) != 51:
        nan_counts = df_grp0[full_required].isna().sum().to_dict()
        raise RuntimeError(
            f"full baseline VIF group size check failed for {init}/{stage}: "
            f"expected 51 after subset dropna, got {len(df_grp)}; "
            f"required_cols={full_required}; nan_counts={nan_counts}"
        )
    max_pair = _pairwise_max_abs_corr(df_grp, full_preds)
    cond_num = _condition_number(df_grp, full_preds)
    Xv = sm.add_constant(df_grp[full_preds], has_constant="add")
    for i, fac in enumerate(full_preds, start=1):
        try:
            v = variance_inflation_factor(Xv.values, i)
        except Exception:
            v = np.nan
        vif_rows.append({"init_date": init, "stage": stage, "model_type": "residualized_full_baseline", "factor": fac, "VIF": float(v) if np.isfinite(v) else np.nan, "max_abs_pairwise_corr": max_pair, "condition_number": cond_num})

df_ssr_audit = pd.DataFrame(ssr_audit_rows)
df_ssr_summary = pd.DataFrame(sens_summary_rows)
df_ssr_lmg_long = pd.DataFrame(sens_lmg_rows)
df_ssr_fitted = pd.DataFrame(sens_fitted_rows)
df_ssr_cmp = pd.DataFrame(compare_rows)
df_ssr_vif = pd.DataFrame(vif_rows)

df_ssr_audit.to_csv(f"{SSR_SUB_DIRS['tables']}/table_ssr_relative_residual_audit.csv", index=False)
df_ssr_summary.to_csv(f"{SSR_SUB_DIRS['tables']}/table_ssr_sensitivity_ols_lmg_SHF.csv", index=False)
df_ssr_lmg_long.to_csv(f"{SSR_SUB_DIRS['tables']}/table_ssr_sensitivity_lmg_long_SHF.csv", index=False)
df_ssr_fitted.to_csv(f"{SSR_SUB_DIRS['tables']}/table_ssr_sensitivity_fitted_SHF.csv", index=False)
df_ssr_cmp.to_csv(f"{SSR_SUB_DIRS['tables']}/table_ssr_sensitivity_model_comparison_SHF.csv", index=False)
df_ssr_vif.to_csv(f"{SSR_SUB_DIRS['tables']}/table_ssr_sensitivity_vif_SHF.csv", index=False)

hm_adj = df_ssr_cmp.melt(id_vars=["init_date", "stage"], value_vars=["baseline_Adj_R2", "noSSR_Adj_R2", "SSRrelSHF_Adj_R2"], var_name="model", value_name="Adj_R2")
hm_adj["row"] = hm_adj["init_date"] + "_" + hm_adj["stage"]
pvt_adj = hm_adj.pivot(index="row", columns="model", values="Adj_R2")
fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(pvt_adj))))
sns.heatmap(pvt_adj, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
ax.set_title("SSR sensitivity Adj_R2")
fig.tight_layout()
fig.savefig(f"{SSR_SUB_DIRS['figures']}/heatmap_ssr_sensitivity_Adj_R2.png", dpi=220, bbox_inches="tight")
plt.close(fig)

hm_delta = df_ssr_cmp.copy()
hm_delta["row"] = hm_delta["init_date"] + "_" + hm_delta["stage"]
pvt_delta = hm_delta.set_index("row")[["delta_Adj_R2_noSSR_minus_full", "delta_Adj_R2_SSRrelSHF_minus_full"]]
fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(pvt_delta))))
sns.heatmap(pvt_delta, annot=True, fmt=".3f", cmap="coolwarm", center=0.0, ax=ax)
ax.set_title("SSR sensitivity ΔAdj_R2")
fig.tight_layout()
fig.savefig(f"{SSR_SUB_DIRS['figures']}/heatmap_ssr_sensitivity_delta_Adj_R2.png", dpi=220, bbox_inches="tight")
plt.close(fig)

for init in sorted(df_ssr_cmp["init_date"].unique()):
    for stage in stage_order:
        if not ((df_ssr_cmp["init_date"] == init) & (df_ssr_cmp["stage"] == stage)).any():
            continue
        rows = []
        full_lmg = df_full_lmg_long[(df_full_lmg_long["init_date"] == init) & (df_full_lmg_long["stage"] == stage)]
        for _, r in full_lmg.iterrows():
            rows.append({"model_type": "residualized_full_baseline", "factor_name": r["factor_name"], "relative_importance_pct": r["relative_importance_pct"]})
        for mt in ["residualized_noSSR", "residualized_SSRrelSHF"]:
            part = df_ssr_lmg_long[(df_ssr_lmg_long["init_date"] == init) & (df_ssr_lmg_long["stage"] == stage) & (df_ssr_lmg_long["model_type"] == mt)]
            for _, r in part.iterrows():
                rows.append({"model_type": mt, "factor_name": r["factor_name"], "relative_importance_pct": r["relative_importance_pct"]})
        p = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=p, x="factor_name", y="relative_importance_pct", hue="model_type", ax=ax)
        ax.set_title(f"LMG compare SSR sensitivity | {init} | {stage}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(f"{SSR_SUB_DIRS['figures']}/lmg_compare_ssr_sensitivity_{init}_{stage}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

print("[SSR sensitivity summary]")
for _, r in df_ssr_cmp.sort_values(["init_date", "stage"]).iterrows():
    sub_vif = df_ssr_vif[(df_ssr_vif["init_date"] == r["init_date"]) & (df_ssr_vif["stage"] == r["stage"])]
    mv = sub_vif.groupby("model_type")["VIF"].max().to_dict()
    audit_ok = bool(df_ssr_audit[(df_ssr_audit["init_date"] == r["init_date"]) & (df_ssr_audit["stage"] == r["stage"])]["resid_formula_ok"].all())
    print(f"{r['init_date']} | {r['stage']} | full Adj_R2={r['baseline_Adj_R2']:.3f}, noSSR Adj_R2={r['noSSR_Adj_R2']:.3f}, SSRrelSHF Adj_R2={r['SSRrelSHF_Adj_R2']:.3f}, delta noSSR-full={r['delta_Adj_R2_noSSR_minus_full']:.3f}, delta SSRrelSHF-full={r['delta_Adj_R2_SSRrelSHF_minus_full']:.3f}, top1(full/noSSR/SSRrelSHF)=({r['top1_full']}/{r['top1_noSSR']}/{r['top1_SSRrelSHF']}), max VIF(full/noSSR/SSRrelSHF)=({mv.get('residualized_full_baseline', np.nan):.2f}/{mv.get('residualized_noSSR', np.nan):.2f}/{mv.get('residualized_SSRrelSHF', np.nan):.2f}), SSR_relSHF audit={'PASS' if audit_ok else 'FAIL'}")
    if (r["noSSR_Adj_R2"] >= 0.75) or (r["delta_Adj_R2_noSSR_minus_full"] > -0.10):
        print("  No-SSR model retains high explanatory power; SSR is not essential for member-axis explainability.")
    elif (r["delta_Adj_R2_noSSR_minus_full"] <= -0.10) and (r["delta_Adj_R2_SSRrelSHF_minus_full"] > r["delta_Adj_R2_noSSR_minus_full"]):
        print("  SSR contains additional independent radiative information after removing SHF-overlapping component.")
    else:
        print("  SSR contribution is mainly shared with SHF/cloud/precipitation-related variability.")
print(f"SSR sensitivity outputs: {SSR_SENS_DIR}")

# %%
# =============================================================================
# Cell 5: TCC cloud mechanism diagnostics / TCC_cloud_mechanism
# =============================================================================

set_audit_cell_name("Cell 5: TCC cloud mechanism diagnostics")
import os, ast, warnings
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)
print("\n" + "=" * 80)
print("Cell 5: TCC cloud mechanism diagnostics / TCC_cloud_mechanism")
print("=" * 80)

def assert_output_under_v8_tcc(path):
    assert_four_init_output(path)

def safe_corr_p(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan, np.nan
    r, p = pearsonr(a[m], b[m]); return float(r), float(p)

def compute_lmg_mc(X, y, feature_names, mc_samples=2000):
    np.random.seed(42)
    n_samples, n_features = X.shape
    if n_samples < n_features + 2:
        return {name: np.nan for name in feature_names}
    X_s = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    y_s = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)
    lmg_raw = np.zeros(n_features); model = LinearRegression()
    for _ in range(mc_samples):
        order = np.random.permutation(n_features); prev_r2 = 0.0
        for i in range(n_features):
            idx = order[:i + 1]; r2 = model.fit(X_s[:, idx], y_s).score(X_s[:, idx], y_s)
            lmg_raw[order[i]] += max(0, r2 - prev_r2); prev_r2 = r2
    tot = lmg_raw.sum()
    return {name: (0.0 if tot == 0 else val / tot * 100.0) for name, val in zip(feature_names, lmg_raw)}

def fit_ols_lmg(df, factors, model_type):
    X = df[factors].values; y = df["y_tmax"].values
    model = sm.OLS(y, sm.add_constant(X)).fit(); yp = model.fittedvalues
    lmg = compute_lmg_mc(X, y, factors, mc_samples=2000)
    top = sorted(lmg.items(), key=lambda kv: kv[1], reverse=True)[0]
    s = {"init_date": df["init_date"].iloc[0], "stage": df["stage"].iloc[0], "model_type": model_type, "R2": model.rsquared, "Adj_R2": model.rsquared_adj,
         "Pred_vs_ECMWF_Corr": safe_corr_p(yp, y)[0], "Pred_vs_ECMWF_RMSE": float(np.sqrt(np.mean((yp - y) ** 2))),
         "Top1_factor_name": top[0], "Top1_factor_importance_pct": top[1], "lmg_dict": lmg}
    fit = df[["init_date", "stage", "member", "y_tmax"]].copy(); fit["model_type"] = model_type; fit["fitted_value"] = yp
    return s, lmg, fit, model

def calc_resid(y, Xdf):
    Xc = sm.add_constant(Xdf, has_constant="add"); m = sm.OLS(y, Xc).fit(); p = m.predict(Xc)
    return y - p, p

def _cond_num(df, cols):
    X0 = df[cols].astype(float).values
    Xz = (X0 - np.nanmean(X0, axis=0)) / (np.nanstd(X0, axis=0) + 1e-8)
    return float(np.linalg.cond(sm.add_constant(Xz, has_constant="add")))
def safe_slice_region_tcc(da, lat_min, lat_max, lon_min, lon_max):
    lat_dim = "latitude" if "latitude" in da.dims else "lat"
    lon_dim = "longitude" if "longitude" in da.dims else "lon"
    lat_vals = da[lat_dim].values
    lat_slice = slice(lat_max, lat_min) if float(lat_vals[0]) > float(lat_vals[-1]) else slice(lat_min, lat_max)
    return da.sel({lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})
def normalize_time_dim_tcc(da):
    time_candidates = ["time", "valid_time", "validTime", "date"]
    tc = None
    for c in time_candidates:
        if c in da.dims or c in da.coords:
            tc = c
            break
    if tc is None:
        raise RuntimeError(f"Cannot detect ERA5 TCC time coordinate. dims={list(da.dims)}, coords={list(da.coords)}")
    if tc not in da.dims:
        if tc not in da.coords:
            raise RuntimeError(f"Detected time coordinate {tc} not in coords.")
        if len(da[tc].dims) != 1:
            raise RuntimeError(f"Time coordinate {tc} is not 1D: dims={da[tc].dims}")
        src_dim = da[tc].dims[0]
        if src_dim not in da.dims:
            raise RuntimeError(f"Time coordinate {tc} uses source dim {src_dim}, but it is not in da.dims={da.dims}")
        da = da.swap_dims({src_dim: tc})
    if tc != "time":
        da = da.rename({tc: "time"})
    da = da.assign_coords(time=pd.to_datetime(da["time"].values)).sortby("time")
    return da, "time"

OUT5 = FOUR_INIT_ROOT
SUB5 = {"tables": f"{OUT5}/tables", "qc": f"{OUT5}/qc", "corr_matrix": f"{OUT5}/figures/corr_matrix", "ols_lmg_combo": f"{OUT5}/figures/ols_lmg_combo",
        "model_compare_heatmap": f"{OUT5}/figures/model_compare_heatmap", "tmax_timeseries": f"{OUT5}/figures/tmax_timeseries",
        "cloud_chain_scatter": f"{OUT5}/figures/cloud_chain_scatter", "cloud_radiation_timeseries": f"{OUT5}/figures/cloud_radiation_timeseries"}
for p in [OUT5] + list(SUB5.values()):
    assert_output_under_v8_tcc(p); os.makedirs(p, exist_ok=True)

base_tables = f"{FOUR_INIT_ROOT}/tables"
ssr_sens_tables = f"{FOUR_INIT_ROOT}/optional_ssr_sensitivity/tables"
for fp in ["table_daily_member_raw_factors_SHF.csv", "table_daily_tmax_SHF.csv", "table_stage_member_raw_factors_SHF.csv", "table_stage_member_residualized_factors_SHF.csv", "table_stage_ols_lmg_SHF.csv", "table_stage_lmg_long_SHF.csv"]:
    if not os.path.exists(f"{base_tables}/{fp}"): raise RuntimeError(f"Missing required input: {base_tables}/{fp}")
df_daily_member = pd.read_csv(f"{base_tables}/table_daily_member_raw_factors_SHF.csv"); df_daily_member["date"] = pd.to_datetime(df_daily_member["date"])
df_daily_tmax = pd.read_csv(f"{base_tables}/table_daily_tmax_SHF.csv"); df_daily_tmax["date"] = pd.to_datetime(df_daily_tmax["date"])

study_window_dict_tcc = {"Stage-I_dry": ("2023-06-14", "2023-06-17"), "Transition_break": ("2023-06-18", "2023-06-20"), "Stage-II_wet": ("2023-06-18", "2023-06-24"), "Total": ("2023-06-14", "2023-06-24")}
window_days = {"Stage-I_dry": 4, "Transition_break": 3, "Stage-II_wet": 7, "Total": 11}

# Build real daily TCC member table from S2S TCC steps
cf_tcc = "/data1/huangy/fig6/NC/cf_2023-6/ecmf_cf_sfc_tcc_2023-06.grib"; pf_tcc = "/data1/huangy/fig6/NC/cf_2023-6/ecmf_pf_sfc_tcc_2023-06.grib"; era5_tcc = "/data1/huangy/fig6/NC/MSE/era5_tcc_2023_06.nc"
for p in [cf_tcc, pf_tcc, era5_tcc]:
    if not os.path.exists(p): raise RuntimeError(f"Missing TCC file: {p}")
ds_cf = xr.open_dataset(cf_tcc, engine="cfgrib", backend_kwargs={"indexpath": ""})
ds_pf = xr.open_dataset(pf_tcc, engine="cfgrib", backend_kwargs={"indexpath": ""})
v_cf = next((k for k in ["tcc", "total_cloud_cover", "cloud_cover"] if k in ds_cf.data_vars), None)
v_pf = next((k for k in ["tcc", "total_cloud_cover", "cloud_cover"] if k in ds_pf.data_vars), None)
if v_cf is None or v_pf is None: raise RuntimeError("Cannot find TCC variable in cf/pf.")
da_cf, da_pf = ds_cf[v_cf], ds_pf[v_pf]
if "number" not in da_cf.dims: da_cf = da_cf.expand_dims(number=[0])
else: da_cf = da_cf.assign_coords(number=[0])
if "number" not in da_pf.dims: raise RuntimeError("pf tcc missing number dimension.")
da_all = xr.concat([da_cf, da_pf], dim="number")
if da_all.sizes.get("number", 0) != 51: raise RuntimeError(f"S2S TCC member size is {da_all.sizes.get('number')}, expected 51.")
da_all = da_all.assign_coords(number=np.arange(da_all.sizes["number"]))
da_all = safe_slice_region_tcc(da_all, 35, 44, 114, 119)
latn = "latitude" if "latitude" in da_all.dims else "lat"; lonn = "longitude" if "longitude" in da_all.dims else "lon"
if float(da_all.max()) > 1.5: da_all = da_all / 100.0
if float(da_all.min()) < -0.01 or float(da_all.max()) > 1.01: raise RuntimeError("S2S TCC range QC failed.")
time_dim = "time" if "time" in da_all.dims else ("forecast_reference_time" if "forecast_reference_time" in da_all.dims else None)
if time_dim is None or "step" not in da_all.dims: raise RuntimeError("S2S TCC missing time/step dims.")
daily_tcc_rows = []
for init in FULL_INIT_LIST:
    init_ts = pd.Timestamp(init)
    try:
        sub = da_all.sel({time_dim: init_ts})
    except Exception as e:
        print(f"[TCC QC] missing init in S2S TCC file: init={init}; error={e}")
        continue
    selected_time = pd.Timestamp(sub[time_dim].values)
    if selected_time.date() != init_ts.date():
        raise RuntimeError(f"S2S TCC selected init time mismatch: target={init_ts}, selected={selected_time}")
    for step in sub["step"].values:
        step_td = pd.to_timedelta(step)
        valid_utc = init_ts + step_td
        ws = valid_utc - pd.Timedelta(hours=24); we = valid_utc
        wc_bjt = ws + pd.Timedelta(hours=20); d = pd.Timestamp(wc_bjt.date())
        if not (pd.Timestamp("2023-06-14") <= d <= pd.Timestamp("2023-06-24")): continue
        vals = sub.sel(step=step).mean(dim=[latn, lonn]).values
        for m, v in enumerate(vals):
            daily_tcc_rows.append({"init_date": init, "date": d, "member": int(m), "TCC_Avg": float(v), "step_hours": float(step_td / pd.Timedelta(hours=1)),
                                   "valid_utc": valid_utc, "window_start_utc": ws, "window_end_utc": we, "window_center_bjt": wc_bjt})
df_daily_member_tcc = pd.DataFrame(daily_tcc_rows)
if df_daily_member_tcc.empty: raise RuntimeError("No S2S TCC daily member rows extracted.")
qc_nm = df_daily_member_tcc.groupby(["init_date", "date"])["member"].nunique().reset_index(name="n_members")
dup = df_daily_member_tcc.duplicated(["init_date", "date", "member"])
if dup.any():
    raise RuntimeError(f"S2S TCC duplicate init/date/member rows detected: {df_daily_member_tcc.loc[dup, ['init_date','date','member']].head(10).to_dict('records')}")
row_counts = df_daily_member_tcc.groupby(["init_date", "date"]).size().reset_index(name="n_rows")
bad_rows = row_counts[row_counts["n_rows"] != 51]
if not bad_rows.empty:
    raise RuntimeError(f"S2S TCC row-count QC failed: {bad_rows.head(10).to_dict('records')}")
bad_nm = qc_nm[qc_nm["n_members"] != 51]
if not bad_nm.empty: raise RuntimeError(f"S2S TCC member-count QC failed: {bad_nm.to_dict('records')[:10]}")
if df_daily_member_tcc["TCC_Avg"].min() < -0.01 or df_daily_member_tcc["TCC_Avg"].max() > 1.01: raise RuntimeError("TCC_Avg out-of-range QC failed.")

# ERA5 TCC: hourly instant, averaged over same UTC 24h windows
ds_e = xr.open_dataset(era5_tcc)
v_e = next((k for k in ["tcc", "total_cloud_cover", "cloud_cover"] if k in ds_e.data_vars), None)
if v_e is None: raise RuntimeError("Cannot find ERA5 TCC variable.")
dae = ds_e[v_e]
dae, t_e = normalize_time_dim_tcc(dae)
dae = safe_slice_region_tcc(dae, 35, 44, 114, 119)
lat_e = "latitude" if "latitude" in dae.dims else "lat"; lon_e = "longitude" if "longitude" in dae.dims else "lon"
if float(dae.max()) > 1.5: dae = dae / 100.0
if float(dae.min()) < -0.01 or float(dae.max()) > 1.01: raise RuntimeError("ERA5 TCC range QC failed.")
print(f"[ERA5 TCC] selected variable={v_e}, dims={dae.dims}, time coord={t_e}, time range={pd.to_datetime(dae[t_e].values).min()} to {pd.to_datetime(dae[t_e].values).max()}")
era5_cmp_rows = []
for (init, d), gg in df_daily_member_tcc.groupby(["init_date", "date"], sort=False):
    ws = pd.to_datetime(gg["window_start_utc"].iloc[0]); we = pd.to_datetime(gg["window_end_utc"].iloc[0]); wc = pd.to_datetime(gg["window_center_bjt"].iloc[0]); sh = gg["step_hours"].iloc[0]
    mask_t = (dae[t_e] >= np.datetime64(ws)) & (dae[t_e] < np.datetime64(we))
    ea_window = dae.sel({t_e: mask_t})
    if ea_window.sizes[t_e] < 20:
        raise RuntimeError(f"ERA5 TCC window too short for {init}/{d}: n_hours={ea_window.sizes[t_e]}, ws={ws}, we={we}")
    ea = ea_window.mean(dim=[t_e, lat_e, lon_e]).item()
    era5_cmp_rows.append({"init_date": init, "date": d, "step_hours": sh, "window_start_utc": ws, "window_end_utc": we, "window_center_bjt": wc,
                     "S2S_TCC_mean": gg["TCC_Avg"].mean(), "S2S_TCC_spread": gg["TCC_Avg"].std(), "ERA5_TCC": float(ea)})
df_daily_tcc_cmp = pd.DataFrame(era5_cmp_rows)
df_tcc_qc = df_daily_tcc_cmp.merge(qc_nm, on=["init_date", "date"], how="left")
df_tcc_qc["TCC_min"] = df_daily_member_tcc.groupby(["init_date", "date"])["TCC_Avg"].min().values
df_tcc_qc["TCC_max"] = df_daily_member_tcc.groupby(["init_date", "date"])["TCC_Avg"].max().values
df_tcc_qc["TCC_std"] = df_daily_member_tcc.groupby(["init_date", "date"])["TCC_Avg"].std().values
df_tcc_qc["TCC_mean"] = df_daily_member_tcc.groupby(["init_date", "date"])["TCC_Avg"].mean().values
df_tcc_qc["S2S_minus_ERA5_TCC"] = df_tcc_qc["S2S_TCC_mean"] - df_tcc_qc["ERA5_TCC"]
_expected_tcc_pairs = pd.MultiIndex.from_product([FULL_INIT_LIST, pd.date_range("2023-06-14", "2023-06-24", freq="D")], names=["init_date", "date"]).to_frame(index=False)
df_tcc_qc = _expected_tcc_pairs.merge(df_tcc_qc, on=["init_date", "date"], how="left")
df_tcc_qc["n_members"] = df_tcc_qc["n_members"].fillna(0).astype(int)
df_tcc_qc["coverage_status"] = np.where(df_tcc_qc["n_members"] == 51, "ok", "missing init in S2S TCC file or incomplete date coverage")
print("[TCC QC] init/date/member range...")
print(df_tcc_qc[["init_date","date","n_members","TCC_min","TCC_max","TCC_mean","TCC_std"]].to_string(index=False))

stage_rows = []
for (init, member), g in df_daily_member.groupby(["init_date", "member"]):
    for stg, (s, e) in study_window_dict_tcc.items():
        mask = (g["date"] >= pd.to_datetime(s)) & (g["date"] <= pd.to_datetime(e)); gg = g[mask]
        gt = df_daily_member_tcc[(df_daily_member_tcc["init_date"] == init) & (df_daily_member_tcc["member"] == member) & (df_daily_member_tcc["date"] >= pd.to_datetime(s)) & (df_daily_member_tcc["date"] <= pd.to_datetime(e))]
        if gg["date"].nunique() != window_days[stg] or gt["date"].nunique() != window_days[stg]:
            print(f"[TCC QC] coverage mismatch; skipped init={init}, stage={stg}, member={member}")
            continue
        row = {"init_date": init, "stage": stg, "member": member, "y_tmax": gg["y_tmax"].mean(), "NCHN_tp": gg["NCHN_tp"].sum(),
               "sm_avg": gg["sm_avg"].mean(), "SSR_Avg": gg["SSR_Avg"].mean(), "SHF_Avg": gg["SHF_Avg"].mean(), "z500_anom_NCHN": gg["z500_anom_NCHN"].mean(), "WNPSH": gg["WNPSH"].mean(),
               "Initial_MSEstar_max_NCHN": gg["Initial_MSEstar_max_NCHN"].mean(), "Initial_Barrier_NCHN": gg["Initial_Barrier_NCHN"].mean(), "NCVI": gg["NCVI"].mean(),
               "TCC_Avg": gt["TCC_Avg"].mean()}
        stage_rows.append(row)
df_stage_tcc = pd.DataFrame(stage_rows)
for (init, stg), gg in df_stage_tcc.groupby(["init_date", "stage"]):
    if len(gg) != 51: raise RuntimeError(f"Stage-member size failed for {init}/{stg}")

aud_rows=[]; tccresid_aud_rows=[]; sum_rows=[]; lmg_rows=[]; fit_rows=[]; cmp_rows=[]; vif_rows=[]; chain_rows=[]; proj_rows=[]; proj_cons_rows=[]; pred_cmp_rows=[]
for (init, stg), g in df_stage_tcc.groupby(["init_date", "stage"], sort=False):
    g = g.sort_values("member").reset_index(drop=True)
    if g["TCC_Avg"].std() < 1e-6:
        raise RuntimeError("TCC_Avg has near-zero member variance; likely failed to extract real TCC values.")
    if stg == "Stage-I_dry":
        shf_resid, _ = calc_resid(g["SHF_Avg"], g[["sm_avg"]]); ssr_tmp, _ = calc_resid(g["SSR_Avg"], g[["sm_avg"]]); g["SHF_resid"] = shf_resid
        tcc_resid, tcc_pred = calc_resid(g["TCC_Avg"], g[["sm_avg"]]); g["TCC_resid"] = tcc_resid
        noSSR = ["WNPSH","z500_anom_NCHN","NCHN_tp","sm_avg","SHF_resid","Initial_MSEstar_max_NCHN","Initial_Barrier_NCHN","NCVI"]; stage_ctrl = "sm_avg"; tcc_resid_base = "SM"
    else:
        sm_resid, _ = calc_resid(g["sm_avg"], g[["NCHN_tp"]]); g["SM_resid"] = sm_resid
        shf_resid, _ = calc_resid(g["SHF_Avg"], g[["NCHN_tp"]]); ssr_tmp, _ = calc_resid(g["SSR_Avg"], g[["NCHN_tp"]]); g["SHF_resid"] = shf_resid
        tcc_resid, tcc_pred = calc_resid(g["TCC_Avg"], g[["NCHN_tp"]]); g["TCC_resid"] = tcc_resid
        noSSR = ["WNPSH","z500_anom_NCHN","NCHN_tp","SM_resid","SHF_resid","Initial_MSEstar_max_NCHN","Initial_Barrier_NCHN","NCVI"]; stage_ctrl = "NCHN_tp"; tcc_resid_base = "TP"
    g["SSR_relSHF"], pred = calc_resid(pd.Series(ssr_tmp, index=g.index), g[["SHF_resid"]])
    manual_diff = float(np.nanmax(np.abs(g["SSR_relSHF"].values - (np.asarray(ssr_tmp)-pred.values))))
    c1 = safe_corr_p(g["SSR_relSHF"], g[stage_ctrl])[0]; c2 = safe_corr_p(g["SSR_relSHF"], g["SHF_resid"])[0]
    ok = (manual_diff < 1e-8) and (np.isnan(c1) or abs(c1)<1e-8) and (np.isnan(c2) or abs(c2)<1e-8)
    aud_rows.append({"init_date":init,"stage":stg,"resid_name":"SSR_relSHF","first_control":stage_ctrl,"second_control":"SHF_resid","n_members":len(g),
                     "manual_max_abs_diff":manual_diff,"corr_resid_first_control_after":c1,"corr_SSRrelSHF_SHFresid_after":c2,"resid_formula_ok":ok})
    if not ok: raise RuntimeError(f"TCC residual audit failed for {init}/{stg}")
    tcc_manual_diff = float(np.nanmax(np.abs(g["TCC_resid"].values - (g["TCC_Avg"].values - tcc_pred.values))))
    tcc_corr = safe_corr_p(g["TCC_resid"], g[stage_ctrl])[0]
    tcc_ok = (len(g) == 51) and (tcc_manual_diff < 1e-8) and (np.isnan(tcc_corr) or abs(tcc_corr) < 1e-8)
    tccresid_aud_rows.append({"init_date": init, "stage": stg, "resid_name": "TCC_resid", "tcc_resid_base": tcc_resid_base, "control_col": stage_ctrl,
                              "n_members": len(g), "corr_TCC_resid_control": tcc_corr, "manual_max_abs_diff": tcc_manual_diff, "resid_formula_ok": tcc_ok})
    print(f"[TCC_resid audit] {init} | {stg}: base={tcc_resid_base}, n_members={len(g)}, corr(TCC_resid,{stage_ctrl})={tcc_corr:.3e}, manual_max_abs_diff={tcc_manual_diff:.3e}")
    if not tcc_ok:
        raise RuntimeError(f"TCC_resid audit failed for {init}/{stg}")
    models = {"noSSR": noSSR, "noSSR_TCCresid": noSSR + ["TCC_resid"], "noSSR_SSRrelSHF": noSSR + ["SSR_relSHF"]}
    daily_tmax_stats = df_daily_tmax.groupby(["init_date","date"]).agg(S2S_mean_absT=("s2s_tmax","mean"), Spread_S2S=("s2s_tmax","std"), ERA5_absT=("era5_tmax","mean")).reset_index()
    stage_mean_era5 = daily_tmax_stats[(daily_tmax_stats["init_date"]==init)&(daily_tmax_stats["date"]>=pd.to_datetime(study_window_dict_tcc[stg][0]))&(daily_tmax_stats["date"]<=pd.to_datetime(study_window_dict_tcc[stg][1]))]["ERA5_absT"].mean()
    g["Tmax_bias_member"] = g["y_tmax"] - stage_mean_era5
    rr={}
    for mt, preds in models.items():
        s,lm,fit,m = fit_ols_lmg(g,preds,mt); s["predictors_used"]=";".join(preds); s["n_predictors"]=len(preds); s["n_members"]=len(g)
        sum_rows.append(s); fit_rows.extend(fit.to_dict("records")); lmg_rows.extend([{"init_date":init,"stage":stg,"model_type":mt,"factor_name":k,"relative_importance_pct":v} for k,v in lm.items()]); rr[mt]=s
        max_pair=float(g[preds].corr().abs().where(~np.eye(len(preds),dtype=bool)).max().max()) if len(preds)>1 else np.nan
        cond=_cond_num(g,preds); Xv=sm.add_constant(g[preds],has_constant="add")
        for i,f in enumerate(preds,1):
            try:v=float(variance_inflation_factor(Xv.values,i))
            except Exception:v=np.nan
            vif_rows.append({"init_date":init,"stage":stg,"model_type":mt,"factor":f,"VIF":v,"max_abs_pairwise_corr":max_pair,"condition_number":cond})
    cmp_rows.append({"init_date":init,"stage":stg,"noSSR_R2":rr["noSSR"]["R2"],"noSSR_Adj_R2":rr["noSSR"]["Adj_R2"],
                     "noSSR_TCCresid_R2":rr["noSSR_TCCresid"]["R2"],"noSSR_TCCresid_Adj_R2":rr["noSSR_TCCresid"]["Adj_R2"],
                     "noSSR_SSRrelSHF_R2":rr["noSSR_SSRrelSHF"]["R2"],"noSSR_SSRrelSHF_Adj_R2":rr["noSSR_SSRrelSHF"]["Adj_R2"],
                     "delta_Adj_R2_TCCresid_minus_noSSR":rr["noSSR_TCCresid"]["Adj_R2"]-rr["noSSR"]["Adj_R2"],
                     "delta_Adj_R2_SSRrelSHF_minus_noSSR":rr["noSSR_SSRrelSHF"]["Adj_R2"]-rr["noSSR"]["Adj_R2"],
                     "top1_noSSR":rr["noSSR"]["Top1_factor_name"],"top1_TCCresid":rr["noSSR_TCCresid"]["Top1_factor_name"],"top1_SSRrelSHF":rr["noSSR_SSRrelSHF"]["Top1_factor_name"],
                     "TCCresid_importance_pct":rr["noSSR_TCCresid"]["lmg_dict"].get("TCC_resid",np.nan),"SSRrelSHF_importance_pct":rr["noSSR_SSRrelSHF"]["lmg_dict"].get("SSR_relSHF",np.nan)})
    rz,pz=safe_corr_p(g["z500_anom_NCHN"],g["TCC_Avg"]); rt,pt=safe_corr_p(g["TCC_Avg"],g["SSR_Avg"]); rtr,ptr=safe_corr_p(g["TCC_Avg"],g["SSR_relSHF"])
    rtt,ptt=safe_corr_p(g["TCC_Avg"],g["y_tmax"]); rs,ps=safe_corr_p(g["SSR_Avg"],g["y_tmax"]); rsr,psr=safe_corr_p(g["SSR_relSHF"],g["y_tmax"]); rzt,pzt=safe_corr_p(g["z500_anom_NCHN"],g["y_tmax"]); rtp,ptp=safe_corr_p(g["NCHN_tp"],g["TCC_Avg"]); rty,pty=safe_corr_p(g["NCHN_tp"],g["y_tmax"])
    chain_rows.append({"init_date":init,"stage":stg,"corr(z500_anom_NCHN,TCC_Avg)":rz,"p_z500_TCC":pz,"corr(TCC_Avg,SSR_Avg)":rt,"p_TCC_SSR":pt,"corr(TCC_Avg,SSR_relSHF)":rtr,"p_TCC_SSRrelSHF":ptr,
                       "corr(TCC_Avg,y_tmax)":rtt,"p_TCC_Tmax":ptt,"corr(SSR_Avg,y_tmax)":rs,"p_SSR_Tmax":ps,"corr(SSR_relSHF,y_tmax)":rsr,"p_SSRrelSHF_Tmax":psr,
                       "corr(z500_anom_NCHN,y_tmax)":rzt,"p_z500_Tmax":pzt,"corr(NCHN_tp,TCC_Avg)":rtp,"p_TP_TCC":ptp,"corr(NCHN_tp,y_tmax)":rty,"p_TP_Tmax":pty,
                       "corr(TCC_Avg,Tmax_bias_member)":safe_corr_p(g["TCC_Avg"],g["Tmax_bias_member"])[0],"corr(SSR_Avg,Tmax_bias_member)":safe_corr_p(g["SSR_Avg"],g["Tmax_bias_member"])[0],"corr(SSR_relSHF,Tmax_bias_member)":safe_corr_p(g["SSR_relSHF"],g["Tmax_bias_member"])[0]})

    # correlation matrices
    smcol = "sm_avg" if stg == "Stage-I_dry" else "SM_resid"
    cm1 = g[["y_tmax","z500_anom_NCHN","TCC_Avg","SSR_Avg","SSR_relSHF","SHF_resid","NCHN_tp",smcol,"WNPSH"]].corr()
    fig,ax=plt.subplots(figsize=(9,7)); sns.heatmap(cm1,annot=True,fmt=".2f",cmap="coolwarm",vmin=-1,vmax=1,ax=ax); ax.set_title(f"cloud_chain {init} {stg}"); fig.tight_layout(); fig.savefig(f"{SUB5['corr_matrix']}/corr_cloud_chain_{init}_{stg}.png",dpi=220); plt.close(fig)
    target_predictor_cols = ["y_tmax"] + noSSR + ["TCC_resid", "SSR_relSHF"]
    target_predictor_cols = list(dict.fromkeys(target_predictor_cols))
    cm2 = g[target_predictor_cols].corr()
    fig,ax=plt.subplots(figsize=(10,8)); sns.heatmap(cm2,annot=True,fmt=".2f",cmap="coolwarm",vmin=-1,vmax=1,ax=ax); ax.set_title(f"Target-predictor correlations | noSSR + TCC_resid | {init} {stg}"); fig.tight_layout(); fig.savefig(f"{SUB5['corr_matrix']}/corr_target_predictors_tccresid_{init}_{stg}.png",dpi=220); plt.close(fig)

df_aud=pd.DataFrame(aud_rows); df_tccresid_aud=pd.DataFrame(tccresid_aud_rows); df_sum=pd.DataFrame(sum_rows); df_lmg=pd.DataFrame(lmg_rows); df_fit=pd.DataFrame(fit_rows); df_cmp=pd.DataFrame(cmp_rows); df_vif=pd.DataFrame(vif_rows); df_chain=pd.DataFrame(chain_rows)
df_stage_tcc.to_csv(f"{SUB5['tables']}/table_stage_member_raw_factors_TCC.csv",index=False)
df_daily_member_tcc.to_csv(f"{SUB5['tables']}/table_daily_member_tcc_SHF.csv",index=False)
df_daily_tcc_cmp.to_csv(f"{SUB5['tables']}/table_daily_tcc_era5_s2s_compare.csv",index=False)
df_tcc_qc.to_csv(f"{SUB5['tables']}/table_tcc_qc_summary.csv", index=False)
df_aud.to_csv(f"{SUB5['tables']}/table_tcc_residual_audit.csv",index=False)
df_tccresid_aud.to_csv(f"{SUB5['tables']}/table_tccresid_residual_audit.csv",index=False)
df_sum.to_csv(f"{SUB5['tables']}/table_tccresid_model_ols_lmg_SHF.csv",index=False)
df_lmg.to_csv(f"{SUB5['tables']}/table_tccresid_model_lmg_long_SHF.csv",index=False)
df_fit.to_csv(f"{SUB5['tables']}/table_tccresid_model_fitted_SHF.csv",index=False)
df_cmp.to_csv(f"{SUB5['tables']}/table_tccresid_model_comparison_SHF.csv",index=False)
df_vif.to_csv(f"{SUB5['tables']}/table_tccresid_model_vif_SHF.csv",index=False)
df_chain.to_csv(f"{SUB5['tables']}/table_tcc_cloud_chain_diagnostics.csv",index=False)

# Cell 5 summary-table figures in Cell 2/3 Tmax summary style.
import matplotlib.colors as mcolors
import matplotlib.patches as patches


def _tcc_table_lighten_color(color, amount=0.6):
    import colorsys
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, 1 - amount * (1 - l), s)


def _tcc_table_readable_text_color(facecolor):
    r, g, b = mcolors.to_rgb(facecolor)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def _tcc_table_truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap_name}", cmap(np.linspace(minval, maxval, n)))


stage_order_summary = ["Stage-I_dry", "Stage-II_wet", "Total"]
daily_tmax_summary_stats = df_daily_tmax.groupby(["init_date", "date"], as_index=False).agg(
    S2S_mean_absT=("s2s_tmax", "mean"),
    Spread_S2S=("s2s_tmax", "std"),
    ERA5_absT=("era5_tmax", "mean"),
)
era5_stage_summary = {}
for _stage in stage_order_summary:
    _s, _e = study_window_dict_tcc[_stage]
    _mask = (daily_tmax_summary_stats["date"] >= pd.to_datetime(_s)) & (daily_tmax_summary_stats["date"] <= pd.to_datetime(_e))
    era5_stage_summary[_stage] = float(daily_tmax_summary_stats.loc[_mask].groupby("date")["ERA5_absT"].mean().mean())


def _stage_daily_tmax_stats(init_date, stage):
    _s, _e = study_window_dict_tcc[stage]
    _daily = daily_tmax_summary_stats[
        (daily_tmax_summary_stats["init_date"] == init_date)
        & (daily_tmax_summary_stats["date"] >= pd.to_datetime(_s))
        & (daily_tmax_summary_stats["date"] <= pd.to_datetime(_e))
    ].copy()
    if _daily.empty:
        raise RuntimeError(f"Cell 5 summary table missing daily Tmax stats for init={init_date}, stage={stage}")
    return _daily


def _build_tcc_tmax_summary_rows(model_type):
    rows = []
    for _stage in stage_order_summary:
        rows.append({"block": "ERA5 observation", "row_metric": "ERA5_absT", "stage": _stage, "value": era5_stage_summary[_stage], "unit": "degC"})
    for _init in sorted(df_stage_tcc["init_date"].unique()):
        for _stage in stage_order_summary:
            _daily = _stage_daily_tmax_stats(_init, _stage)
            _era5 = float(_daily["ERA5_absT"].mean())
            _s2s_mean = float(_daily["S2S_mean_absT"].mean())
            _fit = df_fit[(df_fit["init_date"] == _init) & (df_fit["stage"] == _stage) & (df_fit["model_type"] == model_type)].copy()
            if _fit.empty:
                raise RuntimeError(f"Cell 5 summary table missing fitted values for init={_init}, stage={_stage}, model={model_type}")
            _fit_vals = _fit["fitted_value"].astype(float).values
            _ols_mean = float(np.nanmean(_fit_vals))
            vals = {
                "S2S_mean_absT": _s2s_mean,
                "Bias_S2S_vs_ERA5": float(_s2s_mean - _era5),
                "RMSE_S2S_vs_ERA5": float(np.sqrt(np.nanmean((_daily["S2S_mean_absT"].values - _daily["ERA5_absT"].values) ** 2))),
                "Spread_S2S": float(_daily["Spread_S2S"].mean()),
                "OLS_stage_mean_absT": _ols_mean,
                "Bias_OLS_stage_mean_vs_ERA5": float(_ols_mean - _era5),
                "RMSE_OLS_stage_mean_vs_ERA5": float(np.sqrt(np.nanmean((_ols_mean - _daily["ERA5_absT"].values) ** 2))),
                "Spread_OLS_member_fitted": float(np.nanstd(_fit_vals, ddof=1)),
            }
            for _metric in ["S2S_mean_absT", "Bias_S2S_vs_ERA5", "RMSE_S2S_vs_ERA5", "Spread_S2S", "OLS_stage_mean_absT", "Bias_OLS_stage_mean_vs_ERA5", "RMSE_OLS_stage_mean_vs_ERA5", "Spread_OLS_member_fitted"]:
                rows.append({"block": f"Init {_init}", "row_metric": _metric, "stage": _stage, "value": vals[_metric], "unit": "degC"})
    return pd.DataFrame(rows)


def _plot_tcc_tmax_summary_table(model_type, filename, title):
    summary = _build_tcc_tmax_summary_rows(model_type)
    print(f"[Cell 5 Tmax summary table QC] {model_type}")
    print(summary.pivot_table(index=["block", "row_metric"], columns="stage", values="value", aggfunc="first").reindex(columns=stage_order_summary).round(3).to_string())
    row_keys = list(summary[["block", "row_metric"]].drop_duplicates().itertuples(index=False, name=None))
    pivot = summary.pivot_table(index=["block", "row_metric"], columns="stage", values="value", aggfunc="first").reindex(row_keys)
    pivot = pivot.reindex(columns=stage_order_summary)
    all_bias = summary[summary["row_metric"].str.contains("Bias")]["value"].abs().values
    all_rmse = summary[summary["row_metric"].str.contains("RMSE")]["value"].values
    all_spread = summary[summary["row_metric"].str.contains("Spread")]["value"].values
    bias_lim = max(float(np.nanpercentile(all_bias[np.isfinite(all_bias)], 95)) if np.isfinite(all_bias).any() else 1.0, 0.1)
    rmse_max = max(float(np.nanpercentile(all_rmse[np.isfinite(all_rmse)], 95)) if np.isfinite(all_rmse).any() else 1.0, 0.1)
    spread_max = max(float(np.nanpercentile(all_spread[np.isfinite(all_spread)], 95)) if np.isfinite(all_spread).any() else 1.0, 0.1)
    bias_norm = mcolors.TwoSlopeNorm(vmin=-bias_lim, vcenter=0.0, vmax=bias_lim)
    rmse_norm = mcolors.Normalize(vmin=0.0, vmax=rmse_max)
    spread_norm = mcolors.Normalize(vmin=0.0, vmax=spread_max)
    bias_cmap = _tcc_table_truncate_colormap("RdBu_r", 0.20, 0.80)
    rmse_cmap = _tcc_table_truncate_colormap("Oranges", 0.05, 0.45)
    spread_cmap = _tcc_table_truncate_colormap("BuGn", 0.05, 0.45)
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.42 * len(pivot) + 2.0)))
    ax.set_xlim(-2.7, len(stage_order_summary))
    ax.set_ylim(len(pivot), -1.2)
    ax.axis("off")
    for j, stage in enumerate(stage_order_summary):
        ax.text(j + 0.5, -0.35, stage, ha="center", va="center", fontweight="bold")
    last_block = None
    for i, ((block, metric), row) in enumerate(pivot.iterrows()):
        if block != last_block:
            ax.text(-2.62, i + 0.5, block, ha="left", va="center", fontweight="bold")
            last_block = block
        ax.text(-0.2, i + 0.5, metric, ha="right", va="center")
        for j, stage in enumerate(stage_order_summary):
            val = row.get(stage, np.nan)
            if not np.isfinite(val):
                facecolor, label = "white", ""
            elif "Bias" in metric:
                facecolor = _tcc_table_lighten_color(bias_cmap(bias_norm(np.clip(val, -bias_lim, bias_lim))), amount=0.62)
                label = f"{val:.2f}°C"
            elif "RMSE" in metric:
                facecolor = _tcc_table_lighten_color(rmse_cmap(rmse_norm(np.clip(max(val, 0.0), 0.0, rmse_max))), amount=0.58)
                label = f"{val:.2f}°C"
            elif "Spread" in metric:
                facecolor = _tcc_table_lighten_color(spread_cmap(spread_norm(np.clip(max(val, 0.0), 0.0, spread_max))), amount=0.58)
                label = f"{val:.2f}°C"
            else:
                facecolor, label = "#fbfbfb", f"{val:.2f}°C"
            ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor=facecolor, edgecolor="#e0e0e0", linewidth=1.0))
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=10, color=_tcc_table_readable_text_color(facecolor))
    ax.set_title(title, fontsize=14, pad=20)
    ax.text(0.5, 1.01, "Absolute temperature rows are unshaded; Bias, RMSE, and Spread use separate visual encodings.", transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{SUB5['model_compare_heatmap']}/{filename}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    return summary


tcc_tmax_summary_by_model = {}
for _model, _fname, _title in [
    ("noSSR", "tmax_summary_table_noSSR.png", "Tmax Absolute Temperature and Error Summary | noSSR"),
    ("noSSR_SSRrelSHF", "tmax_summary_table_noSSR_SSRrelSHF.png", "Tmax Absolute Temperature and Error Summary | noSSR + SSR⊥SHF"),
    ("noSSR_TCCresid", "tmax_summary_table_noSSR_TCCresid.png", "Tmax Absolute Temperature and Error Summary | noSSR + TCC_resid"),
]:
    tcc_tmax_summary_by_model[_model] = _plot_tcc_tmax_summary_table(_model, _fname, _title)

_s2s_metrics_to_compare = ["S2S_mean_absT", "Bias_S2S_vs_ERA5", "RMSE_S2S_vs_ERA5", "Spread_S2S"]
_ref_s2s = tcc_tmax_summary_by_model["noSSR"]
_ref_s2s = _ref_s2s[_ref_s2s["row_metric"].isin(_s2s_metrics_to_compare)].sort_values(["block", "row_metric", "stage"]).reset_index(drop=True)
for _model, _summary in tcc_tmax_summary_by_model.items():
    _cur_s2s = _summary[_summary["row_metric"].isin(_s2s_metrics_to_compare)].sort_values(["block", "row_metric", "stage"]).reset_index(drop=True)
    if not np.allclose(_cur_s2s["value"].values, _ref_s2s["value"].values, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"Cell 5 summary S2S rows differ between noSSR and {_model}")
print("[Cell 5 Tmax summary table QC] S2S rows are identical across noSSR/noSSR_SSRrelSHF/noSSR_TCCresid tables.")

_cell3_summary_csv = f"{base_tables}/table_tmax_abs_error_summary.csv"
if os.path.exists(_cell3_summary_csv):
    _cell3_summary = pd.read_csv(_cell3_summary_csv)
    _check_metrics = ["ERA5_absT", "S2S_mean_absT", "Bias_S2S_vs_ERA5", "RMSE_S2S_vs_ERA5", "Spread_S2S"]
    _cell5_ref = tcc_tmax_summary_by_model["noSSR"]
    for _metric in _check_metrics:
        _c5_rows = _cell5_ref[_cell5_ref["row_metric"] == _metric]
        for _, _row in _c5_rows.iterrows():
            _c3_match = _cell3_summary[
                (_cell3_summary["block"] == _row["block"])
                & (_cell3_summary["row_metric"] == _metric)
                & (_cell3_summary["stage"] == _row["stage"])
            ]
            if _c3_match.empty:
                raise RuntimeError(f"Cell 3 summary comparison missing row: block={_row['block']}, metric={_metric}, stage={_row['stage']}")
            _diff = abs(float(_c3_match["value"].iloc[0]) - float(_row["value"]))
            if _diff > 1e-6:
                raise RuntimeError(
                    f"Cell 5 S2S/ERA5 summary differs from Cell 3 for block={_row['block']}, "
                    f"metric={_metric}, stage={_row['stage']}: diff={_diff}"
                )
    print(f"[Cell 5 Tmax summary table QC] ERA5/S2S rows match Cell 3 summary within 1e-6: {_cell3_summary_csv}")
else:
    print(f"[Cell 5 Tmax summary table QC] Cell 3 summary CSV not found; skipped cross-check: {_cell3_summary_csv}")

for init in sorted(df_stage_tcc["init_date"].unique()):
    g = df_daily_tmax[df_daily_tmax["init_date"] == init].copy()
    if g.empty:
        continue
    fig,ax=plt.subplots(figsize=(11,4))
    dts = df_daily_tmax.groupby(["init_date","date"]).agg(S2S_mean_absT=("s2s_tmax","mean"), Spread_S2S=("s2s_tmax","std"), ERA5_absT=("era5_tmax","mean")).reset_index()
    g2 = dts[dts["init_date"]==init]
    # Build OLS daily diagnostic fitted series from stage-trained coefficients + daily predictors.
    TMAX_DIAG_MODEL = "noSSR_TCCresid"
    model_pref = TMAX_DIAG_MODEL if ((df_sum["init_date"] == init) & (df_sum["model_type"] == TMAX_DIAG_MODEL)).any() else "noSSR"
    stage_map = {d: "Stage-I_dry" for d in pd.date_range("2023-06-14", "2023-06-17")}
    stage_map.update({d: "Stage-II_wet" for d in pd.date_range("2023-06-18", "2023-06-24")})
    daily_base = df_daily_member[df_daily_member["init_date"] == init].copy()
    daily_base = daily_base.merge(
        df_daily_member_tcc[df_daily_member_tcc["init_date"] == init][["date", "member", "TCC_Avg"]],
        on=["date", "member"],
        how="inner",
    )
    daily_base["stage"] = daily_base["date"].dt.normalize().map(stage_map)
    daily_base = daily_base[daily_base["stage"].isin(["Stage-I_dry", "Stage-II_wet"])].copy()
    if daily_base.empty:
        raise RuntimeError(f"Cell 5 daily diagnostic base table is empty for init={init}")

    daily_fit_records = []
    for st in ["Stage-I_dry", "Stage-II_wet"]:
        sg = df_stage_tcc[(df_stage_tcc["init_date"] == init) & (df_stage_tcc["stage"] == st)].sort_values("member").reset_index(drop=True)
        if sg.empty:
            continue
        # Recreate stage residual equations and model predictors.
        if st == "Stage-I_dry":
            shf_model = sm.OLS(sg["SHF_Avg"], sm.add_constant(sg[["sm_avg"]], has_constant="add")).fit()
            ssr_tmp_model = sm.OLS(sg["SSR_Avg"], sm.add_constant(sg[["sm_avg"]], has_constant="add")).fit()
            tcc_model = sm.OLS(sg["TCC_Avg"], sm.add_constant(sg[["sm_avg"]], has_constant="add")).fit()
            sg["SHF_resid"] = sg["SHF_Avg"] - shf_model.predict(sm.add_constant(sg[["sm_avg"]], has_constant="add"))
            sg["TCC_resid"] = sg["TCC_Avg"] - tcc_model.predict(sm.add_constant(sg[["sm_avg"]], has_constant="add"))
            ssr_tmp_stage = sg["SSR_Avg"] - ssr_tmp_model.predict(sm.add_constant(sg[["sm_avg"]], has_constant="add"))
            shf_ctrl = "sm_avg"
            tcc_ctrl = "sm_avg"
            noSSR = ["WNPSH","z500_anom_NCHN","NCHN_tp","sm_avg","SHF_resid","Initial_MSEstar_max_NCHN","Initial_Barrier_NCHN","NCVI"]
        else:
            sm_model = sm.OLS(sg["sm_avg"], sm.add_constant(sg[["NCHN_tp"]], has_constant="add")).fit()
            shf_model = sm.OLS(sg["SHF_Avg"], sm.add_constant(sg[["NCHN_tp"]], has_constant="add")).fit()
            ssr_tmp_model = sm.OLS(sg["SSR_Avg"], sm.add_constant(sg[["NCHN_tp"]], has_constant="add")).fit()
            tcc_model = sm.OLS(sg["TCC_Avg"], sm.add_constant(sg[["NCHN_tp"]], has_constant="add")).fit()
            sg["SM_resid"] = sg["sm_avg"] - sm_model.predict(sm.add_constant(sg[["NCHN_tp"]], has_constant="add"))
            sg["SHF_resid"] = sg["SHF_Avg"] - shf_model.predict(sm.add_constant(sg[["NCHN_tp"]], has_constant="add"))
            sg["TCC_resid"] = sg["TCC_Avg"] - tcc_model.predict(sm.add_constant(sg[["NCHN_tp"]], has_constant="add"))
            ssr_tmp_stage = sg["SSR_Avg"] - ssr_tmp_model.predict(sm.add_constant(sg[["NCHN_tp"]], has_constant="add"))
            shf_ctrl = "NCHN_tp"
            tcc_ctrl = "NCHN_tp"
            noSSR = ["WNPSH","z500_anom_NCHN","NCHN_tp","SM_resid","SHF_resid","Initial_MSEstar_max_NCHN","Initial_Barrier_NCHN","NCVI"]
        ssr_rel_model = sm.OLS(ssr_tmp_stage, sm.add_constant(sg[["SHF_resid"]], has_constant="add")).fit()
        sg["SSR_relSHF"] = ssr_tmp_stage - ssr_rel_model.predict(sm.add_constant(sg[["SHF_resid"]], has_constant="add"))
        preds = noSSR + (["TCC_resid"] if model_pref == "noSSR_TCCresid" else (["SSR_relSHF"] if model_pref == "noSSR_SSRrelSHF" else []))
        missing_stage_cols = [c for c in ["y_tmax"] + preds if c not in sg.columns]
        if missing_stage_cols:
            raise RuntimeError(
                f"Daily diagnostic stage model missing columns for {init}/{st}: {missing_stage_cols}; "
                f"available columns={list(sg.columns)}"
            )
        stage_model = sm.OLS(
            sg["y_tmax"],
            sm.add_constant(sg[preds], has_constant="add"),
        ).fit()
        dg = daily_base[daily_base["stage"] == st].copy()
        stage_days = 4 if st == "Stage-I_dry" else 7
        dg["NCHN_tp"] = dg["NCHN_tp"] * stage_days
        # QC: re-aggregated daily NCHN_tp should match stage-level NCHN_tp after scale adjustment.
        reagg_tp = dg.groupby("member", as_index=False)["NCHN_tp"].mean().rename(columns={"NCHN_tp": "NCHN_tp_reagg"})
        tp_cmp = sg[["member", "NCHN_tp"]].merge(reagg_tp, on="member", how="inner")
        if len(tp_cmp) != 51:
            raise RuntimeError(f"Daily diagnostic NCHN_tp QC merge size mismatch for {init}/{st}: got {len(tp_cmp)}")
        tp_max_abs_diff = float(np.nanmax(np.abs(tp_cmp["NCHN_tp_reagg"].values - tp_cmp["NCHN_tp"].values)))
        print(f"[Cell 5 NCHN_tp QC] {init} | {st}: max_abs_diff(reagg_daily_vs_stage)={tp_max_abs_diff:.6g}")
        dg["SHF_resid"] = dg["SHF_Avg"].values - shf_model.predict(sm.add_constant(dg[[shf_ctrl]], has_constant="add")).values
        dg["TCC_resid"] = dg["TCC_Avg"].values - tcc_model.predict(sm.add_constant(dg[[tcc_ctrl]], has_constant="add")).values
        ssr_tmp_daily = dg["SSR_Avg"].values - ssr_tmp_model.predict(sm.add_constant(dg[[shf_ctrl]], has_constant="add")).values
        dg["SSR_relSHF"] = ssr_tmp_daily - ssr_rel_model.predict(sm.add_constant(dg[["SHF_resid"]], has_constant="add")).values
        if st != "Stage-I_dry":
            dg["SM_resid"] = dg["sm_avg"].values - sm_model.predict(sm.add_constant(dg[["NCHN_tp"]], has_constant="add")).values
        Xd = sm.add_constant(dg[preds], has_constant="add")
        dg["OLS_daily_fitted"] = stage_model.predict(Xd).values
        daily_fit_records.append(dg[["date", "member", "stage", "OLS_daily_fitted"]])

    daily_fit_df = pd.concat(daily_fit_records, ignore_index=True)
    daily_fit_stats = daily_fit_df.groupby("date", as_index=False).agg(
        OLS_daily_fitted_mean=("OLS_daily_fitted", "mean"),
        OLS_daily_fitted_spread=("OLS_daily_fitted", "std"),
    )
    g2 = g2.merge(daily_fit_stats, on="date", how="left")

    ax.axvspan(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), color="#F6D365", alpha=0.20, label="Stage-I_dry")
    ax.axvspan(pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24") + pd.Timedelta(hours=18), color="#A1C4FD", alpha=0.20, label="Stage-II_wet")
    ax.plot(g2["date"], g2["ERA5_absT"], color="black", lw=2.0, label="ERA5 Tmax")
    ax.plot(g2["date"], g2["S2S_mean_absT"], color="#1f77b4", lw=1.8, label="S2S mean Tmax")
    ax.fill_between(g2["date"], g2["S2S_mean_absT"]-g2["Spread_S2S"], g2["S2S_mean_absT"]+g2["Spread_S2S"], color="#1f77b4", alpha=0.12, label="S2S spread")
    model_label = "noSSR+TCC_resid" if model_pref == "noSSR_TCCresid" else model_pref
    ax.plot(g2["date"], g2["OLS_daily_fitted_mean"], color="#2E8B57", lw=1.8, marker="s", ms=4, label=f"OLS stage-fitted mean (stage-trained {model_label})")
    ax.fill_between(g2["date"], g2["OLS_daily_fitted_mean"]-g2["OLS_daily_fitted_spread"], g2["OLS_daily_fitted_mean"]+g2["OLS_daily_fitted_spread"], color="#2E8B57", alpha=0.10, label="OLS daily fitted member spread")
    ax.set_title(f"Tmax daily diagnostic series | Init {init} | stage-trained OLS")
    ax.set_ylabel("Tmax (°C)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{SUB5['tmax_timeseries']}/tmax_timeseries_tccresid_stage_mean_diagnostic_{init}.png",dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    dcmp=df_daily_tcc_cmp[df_daily_tcc_cmp["init_date"]==init]
    axs[0].plot(dcmp["date"],dcmp["S2S_TCC_mean"],label="S2S TCC")
    axs[0].fill_between(dcmp["date"], dcmp["S2S_TCC_mean"]-dcmp["S2S_TCC_spread"], dcmp["S2S_TCC_mean"]+dcmp["S2S_TCC_spread"], alpha=0.15)
    axs[0].plot(dcmp["date"],dcmp["ERA5_TCC"],label="ERA5 TCC"); axs[0].legend()
    dssr=df_daily_member[df_daily_member["init_date"]==init].groupby("date").agg(SSR_mean=("SSR_Avg","mean"), SSR_spread=("SSR_Avg","std")).reset_index()
    axs[1].plot(dssr["date"],dssr["SSR_mean"],label="S2S SSR mean")
    axs[1].fill_between(dssr["date"], dssr["SSR_mean"]-dssr["SSR_spread"], dssr["SSR_mean"]+dssr["SSR_spread"], alpha=0.15)
    axs[1].legend()
    axs[2].plot(g2["date"], g2["ERA5_absT"], label="ERA5 Tmax")
    axs[2].plot(g2["date"], g2["S2S_mean_absT"], label="S2S Tmax")
    axs[2].fill_between(g2["date"], g2["S2S_mean_absT"]-g2["Spread_S2S"], g2["S2S_mean_absT"]+g2["Spread_S2S"], alpha=0.15)
    for a in axs:
        a.axvspan(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), color="#F6D365", alpha=0.16)
        a.axvspan(pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24") + pd.Timedelta(hours=18), color="#A1C4FD", alpha=0.16)
        a.axvspan(pd.Timestamp("2023-06-18"),pd.Timestamp("2023-06-20"),color="gray",alpha=.15)
    axs[2].legend(); fig.tight_layout(); fig.savefig(f"{SUB5['cloud_radiation_timeseries']}/cloud_radiation_timeseries_{init}.png",dpi=220); plt.close(fig)

# OLS scatter + LMG combo figures
label_map = {"z500_anom_NCHN":"Z500 anom","TCC_Avg":"TCC","TCC_resid":"TCC resid","SSR_relSHF":"SSR⊥SHF","SHF_resid":"SHF resid","SM_resid":"SM resid","sm_avg":"SM","NCHN_tp":"TP","Initial_MSEstar_max_NCHN":"MSE* max","Initial_Barrier_NCHN":"MSE barrier","WNPSH":"WNPSH","NCVI":"NCVI"}
for _, srow in df_sum.iterrows():
    init, stg, mt = srow["init_date"], srow["stage"], srow["model_type"]
    ff = df_fit[(df_fit["init_date"]==init)&(df_fit["stage"]==stg)&(df_fit["model_type"]==mt)]
    ll = df_lmg[(df_lmg["init_date"]==init)&(df_lmg["stage"]==stg)&(df_lmg["model_type"]==mt)].sort_values("relative_importance_pct", ascending=True)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.5))
    ax1.scatter(ff["y_tmax"],ff["fitted_value"],s=30,alpha=.8); mn=min(ff["y_tmax"].min(),ff["fitted_value"].min()); mx=max(ff["y_tmax"].max(),ff["fitted_value"].max()); ax1.plot([mn,mx],[mn,mx],"k--",lw=1)
    rmse=np.sqrt(np.mean((ff["fitted_value"]-ff["y_tmax"])**2)); ax1.set_title(f"{init} {stg} {mt}\nR²={srow['R2']:.2f} Adj-R²={srow['Adj_R2']:.2f} RMSE={rmse:.2f}")
    ax1.set_xlabel("ECMWF member Tmax"); ax1.set_ylabel("OLS fitted Tmax")
    bar_labels = [label_map.get(x, x) for x in ll["factor_name"]]
    bar_vals = ll["relative_importance_pct"].astype(float).values
    # keep original sorting, add percentage labels with right-side padding
    finite_vals = bar_vals[np.isfinite(bar_vals)]
    vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    if vmax <= 1.0:
        plot_vals = bar_vals * 100.0
        disp_vals = plot_vals
        xvals_for_lim = plot_vals
    else:
        plot_vals = bar_vals
        disp_vals = bar_vals
        xvals_for_lim = bar_vals
    bars = ax2.barh(bar_labels, plot_vals)
    xmax = float(np.nanmax(xvals_for_lim[np.isfinite(xvals_for_lim)])) if np.isfinite(xvals_for_lim).any() else 1.0
    pad = max(0.6, xmax * 0.12)
    ax2.set_xlim(0, xmax + pad)
    for rect, val in zip(bars, disp_vals):
        if not np.isfinite(val):
            continue
        x = rect.get_width()
        y = rect.get_y() + rect.get_height() / 2
        ax2.text(x + pad * 0.15, y, f"{val:.1f}%", va="center", ha="left", fontsize=9)
    ax2.set_title("LMG relative importance")
    fig.tight_layout(); fig.savefig(f"{SUB5['ols_lmg_combo']}/ols_lmg_tccresid_{mt}_{init}_{stg}.png",dpi=220); plt.close(fig)

for _,r in df_chain.iterrows():
    ig = df_stage_tcc[(df_stage_tcc["init_date"]==r["init_date"])&(df_stage_tcc["stage"]==r["stage"])].copy()
    fig,ax=plt.subplots(figsize=(5,4)); sc=ax.scatter(ig["z500_anom_NCHN"],ig["TCC_Avg"],c=(ig["y_tmax"]-ig["y_tmax"].mean()),cmap="coolwarm"); ax.set_title(f"z500 vs TCC {r['init_date']} {r['stage']}"); fig.colorbar(sc,ax=ax); fig.tight_layout(); fig.savefig(f"{SUB5['cloud_chain_scatter']}/scatter_z500_vs_TCC_{r['init_date']}_{r['stage']}.png",dpi=220); plt.close(fig)
    cb=fig.colorbar(sc,ax=ax); cb.set_label("Tmax anomaly or Tmax bias")
    fig.tight_layout(); fig.savefig(f"{SUB5['cloud_chain_scatter']}/scatter_z500_vs_TCC_{r['init_date']}_{r['stage']}.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); sc=ax.scatter(ig["TCC_Avg"],ig["SSR_Avg"],c=ig["NCHN_tp"],cmap="viridis"); ax.set_title(f"TCC vs SSR {r['init_date']} {r['stage']}"); cb=fig.colorbar(sc,ax=ax); cb.set_label("NCHN_tp"); fig.tight_layout(); fig.savefig(f"{SUB5['cloud_chain_scatter']}/scatter_TCC_vs_SSR_{r['init_date']}_{r['stage']}.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); sc=ax.scatter(ig["TCC_Avg"],ig["y_tmax"]-ig["y_tmax"].mean(),c=ig["SSR_Avg"],cmap="plasma"); ax.set_title(f"TCC vs TmaxBias {r['init_date']} {r['stage']}"); cb=fig.colorbar(sc,ax=ax); cb.set_label("SSR_Avg"); fig.tight_layout(); fig.savefig(f"{SUB5['cloud_chain_scatter']}/scatter_TCC_vs_TmaxBias_{r['init_date']}_{r['stage']}.png",dpi=220); plt.close(fig)

def _heat(df, cols, fn, title):
    t=df.copy()
    stage_short = {"Stage-I_dry":"Stage-I","Transition_break":"Trans","Stage-II_wet":"Stage-II","Total":"Total"}
    t["row"]=t["init_date"].str.replace("2023-","",regex=False).str.replace("-","",regex=False).str[2:]+" "+t["stage"].map(stage_short)
    p=t.set_index("row")[cols]
    fig_h = max(6 if fn=="heatmap_tccresid_model_delta_Adj_R2.png" else 4, 0.65*len(p))
    fig,ax=plt.subplots(figsize=(8,fig_h))
    p_plot = p.copy()
    if fn == "heatmap_tccresid_model_delta_Adj_R2.png":
        p_plot.columns = ["TCC_resid - noSSR", "SSR⊥SHF - noSSR"]
    sns.heatmap(p_plot,annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax,annot_kws={"fontsize":9.5})
    ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
    if fn == "heatmap_tcc_chain_correlations.png":
        ax.set_title("TCC–SSR–Tmax chain correlations including auxiliary transition window")
    else:
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(f"{SUB5['model_compare_heatmap']}/{fn}",dpi=220); plt.close(fig)
_heat(df_cmp,["noSSR_Adj_R2","noSSR_TCCresid_Adj_R2","noSSR_SSRrelSHF_Adj_R2"],"heatmap_tccresid_model_Adj_R2.png","Adj_R2")
_heat(df_cmp,["delta_Adj_R2_TCCresid_minus_noSSR","delta_Adj_R2_SSRrelSHF_minus_noSSR"],"heatmap_tccresid_model_delta_Adj_R2.png","Delta Adj_R2")
df_cmp_imp = df_cmp.rename(columns={"TCCresid_importance_pct":"TCC_resid","SSRrelSHF_importance_pct":"SSR⊥SHF"})
_heat(df_cmp_imp,["TCC_resid","SSR⊥SHF"],"heatmap_tccresid_importance.png","LMG importance")
_heat(df_chain,[c for c in ["corr(z500_anom_NCHN,TCC_Avg)","corr(TCC_Avg,SSR_Avg)","corr(TCC_Avg,y_tmax)","corr(SSR_Avg,y_tmax)","corr(NCHN_tp,TCC_Avg)"] if c in df_chain.columns],"heatmap_tcc_chain_correlations.png","Chain correlations")

print("Transition_break is an auxiliary cloud-radiation diagnostic window, not a primary attribution stage.")
print("[TCC cloud mechanism summary]")
for _,r in df_cmp.sort_values(["init_date","stage"]).iterrows():
    ch = df_chain[(df_chain["init_date"]==r["init_date"])&(df_chain["stage"]==r["stage"])].iloc[0]
    sv = df_vif[(df_vif["init_date"]==r["init_date"])&(df_vif["stage"]==r["stage"])].groupby("model_type")["VIF"].max().to_dict()
    print(f"{r['init_date']} | {r['stage']}: noSSR Adj_R2={r['noSSR_Adj_R2']:.3f}, noSSR+TCC_resid Adj_R2={r['noSSR_TCCresid_Adj_R2']:.3f}, noSSR+SSRrelSHF Adj_R2={r['noSSR_SSRrelSHF_Adj_R2']:.3f}, delta TCC_resid-noSSR={r['delta_Adj_R2_TCCresid_minus_noSSR']:.3f}, delta SSRrelSHF-noSSR={r['delta_Adj_R2_SSRrelSHF_minus_noSSR']:.3f}, TCC_resid imp={r['TCCresid_importance_pct']:.2f}%, SSRrelSHF imp={r['SSRrelSHF_importance_pct']:.2f}%, corr(z500,TCC)={ch['corr(z500_anom_NCHN,TCC_Avg)']:.2f}, corr(TCC,SSR)={ch['corr(TCC_Avg,SSR_Avg)']:.2f}, corr(TCC,Tmax)={ch['corr(TCC_Avg,y_tmax)']:.2f}, corr(TP,TCC)={ch['corr(NCHN_tp,TCC_Avg)']:.2f}, maxVIF(noSSR/TCCresid/SSRrelSHF)=({sv.get('noSSR',np.nan):.2f}/{sv.get('noSSR_TCCresid',np.nan):.2f}/{sv.get('noSSR_SSRrelSHF',np.nan):.2f})")
    if (r["delta_Adj_R2_TCCresid_minus_noSSR"] >= 0.05) and (r["TCCresid_importance_pct"] >= 5):
        print("  TCC_resid adds non-negligible cloud-state information beyond the no-SSR model.")
    if (ch["corr(TCC_Avg,SSR_Avg)"] < -0.4) and (ch["corr(TCC_Avg,y_tmax)"] < -0.3):
        print("  Cloud-radiation cooling pathway is supported in member-axis diagnostics.")
    if (
        (r["init_date"] == "2023-06-08")
        and (ch["stage"] == "Transition_break")
        and (abs(ch["corr(z500_anom_NCHN,TCC_Avg)"]) > 0.2)
        and (ch["corr(TCC_Avg,SSR_Avg)"] < 0)
        and (ch["corr(TCC_Avg,Tmax_bias_member)"] < 0)
    ):
        print("  2023-06-08 Transition_break shows a possible Z500-cloud-SSR-Tmax cold-bias pathway.")
    else:
        print("  No robust evidence that z500 directly controls TCC in this window; z500 may affect Tmax through other pathways.")

print("For noSSR + TCC_resid, Cell 5 LMG uses the same TCC_resid definition as Cell 6 LOFO validation.")
print("TCC_resid is residualized against SM for Stage-I and against TP for Stage-II/Total/Transition_break if retained.")
print("TCC_Avg is retained only for cloud-radiation chain diagnostics, not for model importance attribution.")
print("Target-predictor correlation matrices include y_tmax for diagnostic interpretation only; y_tmax is not included as a predictor in OLS/LMG/VIF.")

# %%
# Cell 6: LOFO validation for noSSR + TCC_resid
# Standalone validation block: reads the Cell-5 stage-member TCC table, rebuilds the

set_audit_cell_name("Cell 6: LOFO validation")
# stage-specific residual predictors needed for noSSR + TCC_resid, then compares
# LOFO single-factor removal diagnostics against newly computed LMG importances.
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

LOFO_OUT_DIR = f"{FOUR_INIT_ROOT}/lofo_validation"
LOFO_SUBDIRS = {
    "figures": f"{LOFO_OUT_DIR}/figures",
    "tables": f"{LOFO_OUT_DIR}/tables",
    "qc": f"{LOFO_OUT_DIR}/qc",
}


def assert_lofo_output_under_v8(path):
    if not str(path).startswith(LOFO_OUT_DIR):
        raise RuntimeError(f"LOFO output path is outside FOUR_INIT_ROOT: {path}")


for _lofo_path in [LOFO_OUT_DIR, *LOFO_SUBDIRS.values()]:
    assert_lofo_output_under_v8(_lofo_path)
    os.makedirs(_lofo_path, exist_ok=True)

LOFO_BASE_TCC_TABLE = f"{FOUR_INIT_ROOT}/tables/table_stage_member_raw_factors_TCC.csv"
if not os.path.exists(LOFO_BASE_TCC_TABLE):
    raise RuntimeError(
        f"Required v8 Cell-5 stage TCC table not found for LOFO validation: {LOFO_BASE_TCC_TABLE}. "
        "Please run Cell 5 first to generate the v8 TCC stage-member table."
    )
print("Cell 6 LOFO reads the v8 Cell 5 TCC table to ensure LMG–LOFO input consistency.")
df_lofo_stage_raw = pd.read_csv(LOFO_BASE_TCC_TABLE)

LOFO_INIT_ORDER = FULL_INIT_LIST
LOFO_STAGE_ORDER = ["Stage-I_dry", "Stage-II_wet", "Total"]
LOFO_ROW_LABELS = {
    (init, stage): f"{pd.Timestamp(init):%m-%d} {'Stage-I' if stage == 'Stage-I_dry' else ('Stage-II' if stage == 'Stage-II_wet' else 'Total')}"
    for init in LOFO_INIT_ORDER for stage in LOFO_STAGE_ORDER
}
LOFO_FACTOR_LABELS = {
    "WNPSH": "WNPSH",
    "z500_anom_NCHN": "Z500 anom",
    "NCHN_tp": "TP",
    "sm_avg": "SM",
    "SM_resid": "SM resid",
    "SHF_resid": "SHF resid",
    "Initial_MSEstar_max_NCHN": "MSE* max",
    "Initial_Barrier_NCHN": "MSE barrier",
    "NCVI": "NCVI",
    "TCC_resid": "TCC resid",
}
LOFO_HEATMAP_FACTOR_ORDER = [
    "WNPSH",
    "z500_anom_NCHN",
    "NCHN_tp",
    "sm_avg",
    "SM_resid",
    "SHF_resid",
    "Initial_MSEstar_max_NCHN",
    "Initial_Barrier_NCHN",
    "NCVI",
    "TCC_resid",
]
LOFO_MODEL_NAME = "noSSR_TCCresid"
LOFO_STAGE_RANK = {stage: idx for idx, stage in enumerate(LOFO_STAGE_ORDER)}


def _lofo_predictors_for_stage(stage):
    if stage == "Stage-I_dry":
        return [
            "WNPSH",
            "z500_anom_NCHN",
            "NCHN_tp",
            "sm_avg",
            "SHF_resid",
            "Initial_MSEstar_max_NCHN",
            "Initial_Barrier_NCHN",
            "NCVI",
            "TCC_resid",
        ]
    return [
        "WNPSH",
        "z500_anom_NCHN",
        "NCHN_tp",
        "SM_resid",
        "SHF_resid",
        "Initial_MSEstar_max_NCHN",
        "Initial_Barrier_NCHN",
        "NCVI",
        "TCC_resid",
    ]


def _lofo_residual(y, controls):
    y_series = pd.Series(y, dtype=float)
    x_df = pd.DataFrame(controls).astype(float)
    model = sm.OLS(y_series, sm.add_constant(x_df, has_constant="add")).fit()
    fitted = model.predict(sm.add_constant(x_df, has_constant="add"))
    return y_series - fitted, model


def _lofo_safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _lofo_safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.nanstd(a[mask]) < 1e-12 or np.nanstd(b[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _lofo_compute_lmg_mc(X, y, feature_names, mc_samples=2000):
    """Monte-Carlo LMG with the same seed/style used by the earlier cells."""
    np.random.seed(42)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    feature_names = list(feature_names)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    ys = (y - y.mean()) / (y.std() + 1e-8)
    p = len(feature_names)
    contrib = {name: 0.0 for name in feature_names}

    for _ in range(mc_samples):
        order = np.random.permutation(p)
        prev_r2 = 0.0
        used = []
        for idx in order:
            used.append(idx)
            model = LinearRegression().fit(Xs[:, used], ys)
            r2 = model.score(Xs[:, used], ys)
            contrib[feature_names[idx]] += r2 - prev_r2
            prev_r2 = r2

    for name in feature_names:
        contrib[name] /= mc_samples
    total = sum(contrib.values())
    if np.isfinite(total) and abs(total) > 1e-12:
        return {name: 100.0 * val / total for name, val in contrib.items()}
    return {name: np.nan for name in feature_names}


def _lofo_prepare_stage_group(df_stage, init, stage):
    g = df_stage[(df_stage["init_date"] == init) & (df_stage["stage"] == stage)].copy()
    if "member" in g.columns:
        g = g.sort_values("member").reset_index(drop=True)
    else:
        g = g.reset_index(drop=True)

    base_required = [
        "y_tmax",
        "WNPSH",
        "z500_anom_NCHN",
        "NCHN_tp",
        "sm_avg",
        "SHF_Avg",
        "Initial_MSEstar_max_NCHN",
        "Initial_Barrier_NCHN",
        "NCVI",
        "TCC_Avg",
    ]
    missing_base = [c for c in base_required if c not in g.columns]
    if missing_base:
        raise RuntimeError(f"LOFO missing base columns for {init}/{stage}: {missing_base}")

    if stage == "Stage-I_dry":
        g["SHF_resid"], _ = _lofo_residual(g["SHF_Avg"], g[["sm_avg"]])
        g["TCC_resid"], _ = _lofo_residual(g["TCC_Avg"], g[["sm_avg"]])
        tcc_resid_base = "SM"
    else:
        g["SM_resid"], _ = _lofo_residual(g["sm_avg"], g[["NCHN_tp"]])
        g["SHF_resid"], _ = _lofo_residual(g["SHF_Avg"], g[["NCHN_tp"]])
        g["TCC_resid"], _ = _lofo_residual(g["TCC_Avg"], g[["NCHN_tp"]])
        tcc_resid_base = "TP"

    predictors = _lofo_predictors_for_stage(stage)
    required = ["y_tmax"] + predictors
    if "member" in g.columns:
        required = ["member"] + required
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise RuntimeError(f"LOFO missing model columns for {init}/{stage}: {missing}")

    n_before = len(g)
    g_model = g.dropna(subset=required).copy()
    if "member" in g_model.columns:
        g_model = g_model.sort_values("member").reset_index(drop=True)
    else:
        g_model = g_model.reset_index(drop=True)
    if n_before != 51:
        raise RuntimeError(f"LOFO expected 51 raw members for {init}/{stage}, got {n_before}")
    if len(g_model) != 51:
        raise RuntimeError(f"LOFO expected 51 modeling members after dropna for {init}/{stage}, got {len(g_model)}")

    tcc_control_col = "sm_avg" if stage == "Stage-I_dry" else "NCHN_tp"
    tcc_resid_control_corr = _lofo_safe_corr(g_model["TCC_resid"], g_model[tcc_control_col])
    if not np.isfinite(tcc_resid_control_corr) or abs(tcc_resid_control_corr) > 1e-8:
        raise RuntimeError(
            f"LOFO TCC_resid orthogonality check failed for {init}/{stage}: "
            f"corr(TCC_resid,{tcc_control_col})={tcc_resid_control_corr}"
        )
    return g_model, predictors, tcc_resid_base, tcc_control_col, tcc_resid_control_corr, n_before


lofo_comparison_records = []
lofo_summary_records = []
lofo_fitted_records = []
lofo_qc_lines = [
    "LOFO validation for noSSR + TCC_resid",
    "LOFO provides an independent sensitivity check for LMG relative importance.",
    "A large LOFO_delta_R2 indicates that removing this predictor substantially reduces full-model explanatory power.",
    "LOFO_delta_adjR2 is retained in output tables and QC as a secondary reference metric.",
    "Consistency between LMG rank and LOFO rank supports the robustness of the inferred dominant predictors.",
    "LOFO and LMG are diagnostic attribution tools and do not prove causality.",
    "Negative Delta_AdjR2 can occur because adjusted R2 penalizes model complexity and is sensitive to redundant predictors.",
    "LOFO bar length shows Delta R2; values in parentheses show the normalized percentage contribution among positive LOFO Delta R2 values within the same init-stage.",
    "",
]

for init in LOFO_INIT_ORDER:
    for stage in LOFO_STAGE_ORDER:
        g_model, predictors, tcc_resid_base, tcc_control_col, tcc_resid_control_corr, n_raw = _lofo_prepare_stage_group(df_lofo_stage_raw, init, stage)
        y = g_model["y_tmax"].astype(float).to_numpy()
        X_df = g_model[predictors].astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)

        full_model = sm.OLS(y, sm.add_constant(X_scaled, has_constant="add")).fit()
        full_fitted = np.asarray(full_model.fittedvalues, dtype=float)
        r2_full = float(full_model.rsquared)
        adjr2_full = float(full_model.rsquared_adj)
        rmse_full = _lofo_safe_rmse(y, full_fitted)

        lmg_pct = _lofo_compute_lmg_mc(X_df.to_numpy(), y, predictors, mc_samples=2000)
        lmg_rank_map = {
            factor: rank
            for rank, factor in enumerate(
                sorted(predictors, key=lambda f: (np.nan_to_num(lmg_pct.get(f, np.nan), nan=-np.inf)), reverse=True),
                start=1,
            )
        }

        reduced_rows = []
        for factor_idx, factor in enumerate(predictors):
            X_reduced = np.delete(X_scaled, factor_idx, axis=1)
            reduced_model = sm.OLS(y, sm.add_constant(X_reduced, has_constant="add")).fit()
            r2_without = float(reduced_model.rsquared)
            adjr2_without = float(reduced_model.rsquared_adj)
            reduced_rows.append({
                "init": init,
                "stage": stage,
                "factor": factor,
                "R2_full": r2_full,
                "AdjR2_full": adjr2_full,
                "R2_without_factor": r2_without,
                "AdjR2_without_factor": adjr2_without,
                "LOFO_delta_R2": r2_full - r2_without,
                "LOFO_delta_adjR2": adjr2_full - adjr2_without,
                "LMG_importance": lmg_pct.get(factor, np.nan),
                "LMG_importance_percent": lmg_pct.get(factor, np.nan),
                "LMG_rank": lmg_rank_map[factor],
            })

        group_lofo = pd.DataFrame(reduced_rows)
        group_lofo["LOFO_rank_R2"] = group_lofo["LOFO_delta_R2"].rank(ascending=False, method="min").astype(int)
        group_lofo["LOFO_rank_adjR2"] = group_lofo["LOFO_delta_adjR2"].rank(ascending=False, method="min").astype(int)
        lofo_comparison_records.extend(group_lofo.to_dict("records"))

        lofo_summary_records.append({
            "init": init,
            "stage": stage,
            "n_members": int(len(g_model)),
            "n_raw_members": int(n_raw),
            "n_predictors": int(len(predictors)),
            "R2_full": r2_full,
            "AdjR2_full": adjr2_full,
            "RMSE_full": rmse_full,
            "predictor_list": ";".join(predictors),
            "tcc_resid_base": tcc_resid_base,
            "tcc_resid_control_col": tcc_control_col,
            "corr_TCC_resid_control": tcc_resid_control_corr,
        })

        member_values = g_model["member"].to_numpy() if "member" in g_model.columns else np.arange(len(g_model))
        for member, y_val, fit_val in zip(member_values, y, full_fitted):
            lofo_fitted_records.append({
                "init": init,
                "stage": stage,
                "member": member,
                "y_tmax": float(y_val),
                "full_fitted_tmax": float(fit_val),
                "model_type": LOFO_MODEL_NAME,
            })

        top3_lmg = group_lofo.sort_values("LMG_rank").head(3)
        top3_lofo = group_lofo.sort_values("LOFO_rank_R2").head(3)
        top3_lmg_txt = ", ".join(f"{LOFO_FACTOR_LABELS.get(r.factor, r.factor)} ({r.LMG_importance_percent:.2f}%)" for r in top3_lmg.itertuples())
        top3_lofo_txt = ", ".join(f"{LOFO_FACTOR_LABELS.get(r.factor, r.factor)} ({r.LOFO_delta_R2:.4f})" for r in top3_lofo.itertuples())
        print(f"[LOFO noSSR_TCCresid] {init} | {stage}: R2_full={r2_full:.3f}, AdjR2_full={adjr2_full:.3f}, Top-3 LMG={top3_lmg_txt}, Top-3 LOFO factors (ranked by LOFO_delta_R2)={top3_lofo_txt}")

        lofo_qc_lines.extend([
            f"[{init} | {stage}]",
            f"member count raw/model = {n_raw}/{len(g_model)}",
            f"predictor list = {predictors}",
            f"TCC_resid residualization base = {tcc_resid_base}",
            f"TCC_resid orthogonality corr(TCC_resid, {tcc_control_col}) = {tcc_resid_control_corr:.12e}",
            f"Full model R2 / AdjR2 / RMSE = {r2_full:.6f} / {adjr2_full:.6f} / {rmse_full:.6f}",
            "LOFO deltas (sorted by LOFO_delta_R2):",
        ])
        for r in group_lofo.sort_values("LOFO_rank_R2").itertuples():
            lofo_qc_lines.append(
                f"  {r.factor}: LOFO_delta_R2={r.LOFO_delta_R2:.6f}, LOFO_delta_adjR2={r.LOFO_delta_adjR2:.6f}"
            )
        lofo_qc_lines.extend([
            f"Top-3 LMG factors = {top3_lmg_txt}",
            f"Top-3 LOFO factors ranked by LOFO_delta_R2 = {top3_lofo_txt}",
            "LMG rank and LOFO rank based on LOFO_delta_R2 are expected to be similar when both diagnostics identify the same dominant predictors, but exact agreement is not required because the two metrics answer different sensitivity questions.",
            "",
        ])


df_lofo_cmp = pd.DataFrame(lofo_comparison_records)
df_lofo_sum = pd.DataFrame(lofo_summary_records)
df_lofo_fit = pd.DataFrame(lofo_fitted_records)

df_lofo_cmp["_stage_order"] = df_lofo_cmp["stage"].map(LOFO_STAGE_RANK)
df_lofo_cmp = df_lofo_cmp.sort_values(["init", "_stage_order", "LOFO_rank_R2"]).drop(columns="_stage_order").reset_index(drop=True)
df_lofo_sum["_stage_order"] = df_lofo_sum["stage"].map(LOFO_STAGE_RANK)
df_lofo_sum = df_lofo_sum.sort_values(["init", "_stage_order"]).drop(columns="_stage_order").reset_index(drop=True)
df_lofo_fit["_stage_order"] = df_lofo_fit["stage"].map(LOFO_STAGE_RANK)
df_lofo_fit = df_lofo_fit.sort_values(["init", "_stage_order", "member"]).drop(columns="_stage_order").reset_index(drop=True)
lofo_cmp_csv = f"{LOFO_SUBDIRS['tables']}/lofo_lmg_comparison_noSSR_TCCresid.csv"
lofo_sum_csv = f"{LOFO_SUBDIRS['tables']}/lofo_full_model_summary_noSSR_TCCresid.csv"
lofo_fit_csv = f"{LOFO_SUBDIRS['tables']}/lofo_full_model_fitted_noSSR_TCCresid.csv"
for _out_path in [lofo_cmp_csv, lofo_sum_csv, lofo_fit_csv]:
    assert_lofo_output_under_v8(_out_path)
df_lofo_cmp.to_csv(lofo_cmp_csv, index=False)
df_lofo_sum.to_csv(lofo_sum_csv, index=False)
df_lofo_fit.to_csv(lofo_fit_csv, index=False)

lofo_qc_txt = f"{LOFO_SUBDIRS['qc']}/qc_lofo_noSSR_TCCresid_summary.txt"
assert_lofo_output_under_v8(lofo_qc_txt)
with open(lofo_qc_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lofo_qc_lines))

# Figure 1: LMG vs LOFO heatmap comparison.
row_order = [LOFO_ROW_LABELS[(init, stage)] for init in LOFO_INIT_ORDER for stage in LOFO_STAGE_ORDER]
col_order = LOFO_HEATMAP_FACTOR_ORDER
col_labels = [LOFO_FACTOR_LABELS.get(c, c) for c in col_order]
plot_df = df_lofo_cmp.copy()
plot_df["row_label"] = plot_df.apply(lambda r: LOFO_ROW_LABELS[(r["init"], r["stage"])], axis=1)
lmg_heat = plot_df.pivot(index="row_label", columns="factor", values="LMG_importance_percent").reindex(index=row_order, columns=col_order)
lofo_heat = plot_df.pivot(index="row_label", columns="factor", values="LOFO_delta_R2").reindex(index=row_order, columns=col_order)

fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), constrained_layout=True)
sns.heatmap(lmg_heat, ax=axes[0], cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "LMG importance (%)"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[0].set_title("LMG importance (%)")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_xticklabels(col_labels, rotation=45, ha="right")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
sns.heatmap(lofo_heat, ax=axes[1], cmap="coolwarm", center=0, annot=True, fmt=".3f", cbar_kws={"label": "LOFO Delta R2"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[1].set_title("LOFO Delta R2")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_xticklabels(col_labels, rotation=45, ha="right")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
fig.suptitle("LMG vs LOFO validation | noSSR + TCC_resid", fontsize=15)
lofo_heat_png = f"{LOFO_SUBDIRS['figures']}/fig_lofo_lmg_heatmap_comparison_noSSR_TCCresid.png"
lofo_heat_pdf = f"{LOFO_SUBDIRS['figures']}/fig_lofo_lmg_heatmap_comparison_noSSR_TCCresid.pdf"
for _out_path in [lofo_heat_png, lofo_heat_pdf]:
    assert_lofo_output_under_v8(_out_path)
fig.savefig(lofo_heat_png, dpi=240)
fig.savefig(lofo_heat_pdf)
plt.close(fig)

# Figure 2: full-model scatter plus LOFO Delta R2 bars, split by init date.
for init in LOFO_INIT_ORDER:
    fig, axes = plt.subplots(len(LOFO_STAGE_ORDER), 2, figsize=(12.5, 3.1 * len(LOFO_STAGE_ORDER)), constrained_layout=True)
    for row_idx, stage in enumerate(LOFO_STAGE_ORDER):
        ax_scatter = axes[row_idx, 0]
        ax_bar = axes[row_idx, 1]
        ff = df_lofo_fit[(df_lofo_fit["init"] == init) & (df_lofo_fit["stage"] == stage)]
        ss = df_lofo_sum[(df_lofo_sum["init"] == init) & (df_lofo_sum["stage"] == stage)].iloc[0]
        ax_scatter.scatter(ff["y_tmax"], ff["full_fitted_tmax"], s=28, alpha=0.8, color="#2f6db3")
        mn = float(min(ff["y_tmax"].min(), ff["full_fitted_tmax"].min()))
        mx = float(max(ff["y_tmax"].max(), ff["full_fitted_tmax"].max()))
        pad = max(0.2, 0.05 * (mx - mn))
        ax_scatter.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", lw=1)
        ax_scatter.set_xlim(mn - pad, mx + pad)
        ax_scatter.set_ylim(mn - pad, mx + pad)
        ax_scatter.set_title(f"{LOFO_ROW_LABELS[(init, stage)]}\nR²={ss['R2_full']:.2f}, Adj-R²={ss['AdjR2_full']:.2f}, RMSE={ss['RMSE_full']:.2f}")
        ax_scatter.set_xlabel("ECMWF member Tmax")
        ax_scatter.set_ylabel("Full-model fitted Tmax")

        bb = df_lofo_cmp[(df_lofo_cmp["init"] == init) & (df_lofo_cmp["stage"] == stage)].sort_values("LOFO_delta_R2", ascending=True)
        labels = [LOFO_FACTOR_LABELS.get(x, x) for x in bb["factor"]]
        vals = bb["LOFO_delta_R2"].astype(float).to_numpy()
        positive_sum = float(np.nansum(np.maximum(vals, 0.0)))
        if positive_sum > 0:
            vals_pct = np.maximum(vals, 0.0) / positive_sum * 100.0
        else:
            vals_pct = np.zeros_like(vals, dtype=float)
        colors = ["#d95f02" if v >= 0 else "#7570b3" for v in vals]
        bars = ax_bar.barh(labels, vals, color=colors, alpha=0.85)
        xmin = float(np.nanmin([0.0, *vals]))
        xmax = float(np.nanmax([0.0, *vals]))
        span = max(1e-6, xmax - xmin)
        ax_bar.set_xlim(xmin - 0.12 * span, xmax + 0.18 * span)
        ax_bar.axvline(0, color="0.3", lw=0.8)
        for rect, val, pct in zip(bars, vals, vals_pct):
            x = rect.get_width()
            y_mid = rect.get_y() + rect.get_height() / 2
            ha = "left" if val >= 0 else "right"
            dx = 0.015 * span if val >= 0 else -0.015 * span
            pct_display = pct if val > 0 else 0.0
            ax_bar.text(x + dx, y_mid, f"{val:.3f} ({pct_display:.1f}%)", va="center", ha=ha, fontsize=8.5)
        ax_bar.set_title("LOFO Delta R2")
        ax_bar.set_xlabel("Delta R2")
    fig.suptitle(f"LOFO validation combo | noSSR + TCC_resid | Init {init}", fontsize=15)
    lofo_combo_png = f"{LOFO_SUBDIRS['figures']}/fig_lofo_combo_noSSR_TCCresid_init{init}.png"
    lofo_combo_pdf = f"{LOFO_SUBDIRS['figures']}/fig_lofo_combo_noSSR_TCCresid_init{init}.pdf"
    for _out_path in [lofo_combo_png, lofo_combo_pdf]:
        assert_lofo_output_under_v8(_out_path)
    fig.savefig(lofo_combo_png, dpi=240)
    fig.savefig(lofo_combo_pdf)
    plt.close(fig)

# Figure 3: rank comparison between LMG and LOFO Delta R2 ranks.
lmg_rank_heat = plot_df.pivot(index="row_label", columns="factor", values="LMG_rank").reindex(index=row_order, columns=col_order)
lofo_rank_heat = plot_df.pivot(index="row_label", columns="factor", values="LOFO_rank_R2").reindex(index=row_order, columns=col_order)
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), constrained_layout=True)
sns.heatmap(lmg_rank_heat, ax=axes[0], cmap="viridis_r", annot=True, fmt=".0f", cbar_kws={"label": "LMG rank"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[0].set_title("LMG rank")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_xticklabels(col_labels, rotation=45, ha="right")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
sns.heatmap(lofo_rank_heat, ax=axes[1], cmap="viridis_r", annot=True, fmt=".0f", cbar_kws={"label": "LOFO rank (Delta R2)"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[1].set_title("LOFO rank (Delta R2)")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_xticklabels(col_labels, rotation=45, ha="right")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
fig.suptitle("LMG vs LOFO rank comparison | noSSR + TCC_resid", fontsize=15)
lofo_rank_png = f"{LOFO_SUBDIRS['figures']}/fig_lofo_lmg_rank_comparison_noSSR_TCCresid.png"
lofo_rank_pdf = f"{LOFO_SUBDIRS['figures']}/fig_lofo_lmg_rank_comparison_noSSR_TCCresid.pdf"
for _out_path in [lofo_rank_png, lofo_rank_pdf]:
    assert_lofo_output_under_v8(_out_path)
fig.savefig(lofo_rank_png, dpi=240)
fig.savefig(lofo_rank_pdf)
plt.close(fig)

print(f"[LOFO noSSR_TCCresid] comparison table: {lofo_cmp_csv}")
print(f"[LOFO noSSR_TCCresid] full model summary: {lofo_sum_csv}")
print(f"[LOFO noSSR_TCCresid] QC summary: {lofo_qc_txt}")
print(f"[LOFO noSSR_TCCresid] figures: {LOFO_SUBDIRS['figures']}")

# Cell 7: NCVI_concurrent sensitivity / NCVI_concurrent_sensitivity
# Independent sensitivity module: compares the original NCVI factor from the Cell-5

set_audit_cell_name("Cell 7: NCVI-concurrent calculation and sensitivity")
# stage-member table against a stage-concurrent NCVI diagnostic, then reruns the
# noSSR + TCC_resid OLS/LMG/LOFO diagnostics with either NCVI_old or NCVI_concurrent.
import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print("Cell 7: NCVI_concurrent sensitivity / NCVI_concurrent_sensitivity")

NCVI_CONCURRENT_OUT_DIR = FOUR_INIT_ROOT
NCVI_CONCURRENT_SUBDIRS = {
    "tables": f"{NCVI_CONCURRENT_OUT_DIR}/tables",
    "figures": f"{NCVI_CONCURRENT_OUT_DIR}/figures",
    "timeseries": f"{NCVI_CONCURRENT_OUT_DIR}/figures/timeseries",
    "qc": f"{NCVI_CONCURRENT_OUT_DIR}/qc",
}


def assert_ncvi_concurrent_output_path(path):
    if not str(path).startswith(FOUR_INIT_ROOT):
        raise RuntimeError(f"NCVI_concurrent sensitivity output path is outside FOUR_INIT_ROOT: {path}")


for _ncvi_path in [NCVI_CONCURRENT_OUT_DIR, *NCVI_CONCURRENT_SUBDIRS.values()]:
    assert_ncvi_concurrent_output_path(_ncvi_path)
    os.makedirs(_ncvi_path, exist_ok=True)

NCVI_CONCURRENT_BASE_TCC_TABLE = f"{FOUR_INIT_ROOT}/tables/table_stage_member_raw_factors_TCC.csv"
if not os.path.exists(NCVI_CONCURRENT_BASE_TCC_TABLE):
    raise RuntimeError(
        f"Required v8 Cell-5 stage TCC table not found for NCVI_concurrent sensitivity: {NCVI_CONCURRENT_BASE_TCC_TABLE}. "
        "Please run Cell 5 first to generate the v8 TCC stage-member table."
    )

df_ncvi_stage_raw = pd.read_csv(NCVI_CONCURRENT_BASE_TCC_TABLE)

NCVI_CONCURRENT_BASE_TABLES = f"{FOUR_INIT_ROOT}/tables"
NCVI_CONCURRENT_DAILY_MEMBER_RAW_TABLE = f"{NCVI_CONCURRENT_BASE_TABLES}/table_daily_member_raw_factors_SHF.csv"
NCVI_CONCURRENT_DAILY_TMAX_TABLE = f"{NCVI_CONCURRENT_BASE_TABLES}/table_daily_tmax_SHF.csv"
NCVI_CONCURRENT_DAILY_TCC_TABLE = f"{FOUR_INIT_ROOT}/tables/table_daily_member_tcc_SHF.csv"
for _required_input in [NCVI_CONCURRENT_DAILY_MEMBER_RAW_TABLE, NCVI_CONCURRENT_DAILY_TMAX_TABLE, NCVI_CONCURRENT_DAILY_TCC_TABLE]:
    if not os.path.exists(_required_input):
        raise RuntimeError(f"Required input for NCVI_concurrent daily time series is missing: {_required_input}")
df_ncvi_daily_member_raw = pd.read_csv(NCVI_CONCURRENT_DAILY_MEMBER_RAW_TABLE)
df_ncvi_daily_member_raw["date"] = pd.to_datetime(df_ncvi_daily_member_raw["date"])
df_ncvi_daily_tmax = pd.read_csv(NCVI_CONCURRENT_DAILY_TMAX_TABLE)
df_ncvi_daily_tmax["date"] = pd.to_datetime(df_ncvi_daily_tmax["date"])
df_ncvi_daily_tcc = pd.read_csv(NCVI_CONCURRENT_DAILY_TCC_TABLE)
df_ncvi_daily_tcc["date"] = pd.to_datetime(df_ncvi_daily_tcc["date"])

NCVI_CONCURRENT_INIT_ORDER = FULL_INIT_LIST
NCVI_CONCURRENT_STAGE_WINDOWS = {
    "Stage-I_dry": ("2023-06-14", "2023-06-17"),
    "Stage-II_wet": ("2023-06-18", "2023-06-24"),
    "Total": ("2023-06-14", "2023-06-24"),
}
NCVI_CONCURRENT_STAGE_ORDER = list(NCVI_CONCURRENT_STAGE_WINDOWS.keys())
NCVI_CONCURRENT_STAGE_RANK = {stage: idx for idx, stage in enumerate(NCVI_CONCURRENT_STAGE_ORDER)}
NCVI_CONCURRENT_ROW_LABELS = {
    (init, stage): f"{pd.Timestamp(init):%m-%d} {'Stage-I' if stage == 'Stage-I_dry' else ('Stage-II' if stage == 'Stage-II_wet' else 'Total')}"
    for init in NCVI_CONCURRENT_INIT_ORDER for stage in NCVI_CONCURRENT_STAGE_ORDER
}
NCVI_CONCURRENT_FACTOR_LABELS = {
    "WNPSH": "WNPSH",
    "z500_anom_NCHN": "Z500 anom",
    "NCHN_tp": "TP",
    "sm_avg": "SM",
    "SM_resid": "SM resid",
    "SHF_resid": "SHF resid",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
    "Initial_Barrier_NCHN": "Init barrier",
    "NCVI_old": "NCVI old",
    "NCVI_concurrent": "NCVI conc.",
    "TCC_resid": "TCC resid",
}
NCVI_CONCURRENT_HEATMAP_ORDER_OLD = [
    "WNPSH",
    "z500_anom_NCHN",
    "NCHN_tp",
    "sm_avg",
    "SM_resid",
    "SHF_resid",
    "Initial_MSEstar_max_NCHN",
    "Initial_Barrier_NCHN",
    "NCVI_old",
    "TCC_resid",
]
NCVI_CONCURRENT_HEATMAP_ORDER_CONCURRENT = [
    "WNPSH",
    "z500_anom_NCHN",
    "NCHN_tp",
    "sm_avg",
    "SM_resid",
    "SHF_resid",
    "Initial_MSEstar_max_NCHN",
    "Initial_Barrier_NCHN",
    "NCVI_concurrent",
    "TCC_resid",
]


def _ncvi_concurrent_predictors_for_stage(stage, ncvi_col):
    if stage == "Stage-I_dry":
        return [
            "WNPSH",
            "z500_anom_NCHN",
            "NCHN_tp",
            "sm_avg",
            "SHF_resid",
            "Initial_MSEstar_max_NCHN",
            "Initial_Barrier_NCHN",
            ncvi_col,
            "TCC_resid",
        ]
    return [
        "WNPSH",
        "z500_anom_NCHN",
        "NCHN_tp",
        "SM_resid",
        "SHF_resid",
        "Initial_MSEstar_max_NCHN",
        "Initial_Barrier_NCHN",
        ncvi_col,
        "TCC_resid",
    ]


def _ncvi_concurrent_residual(y, controls):
    y_series = pd.Series(y, dtype=float).reset_index(drop=True)
    x_df = pd.DataFrame(controls).astype(float).reset_index(drop=True)
    model = sm.OLS(y_series, sm.add_constant(x_df, has_constant="add")).fit()
    fitted = model.predict(sm.add_constant(x_df, has_constant="add"))
    return y_series - fitted, model


def _ncvi_concurrent_safe_corr_p(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) < 1e-12 or np.nanstd(b[mask]) < 1e-12:
        return np.nan, np.nan
    r, p = pearsonr(a[mask], b[mask])
    return float(r), float(p)


def _ncvi_concurrent_safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _ncvi_concurrent_compute_lmg_mc(X, y, feature_names, mc_samples=2000):
    np.random.seed(42)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    feature_names = list(feature_names)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    ys = (y - y.mean()) / (y.std() + 1e-8)
    p = len(feature_names)
    contrib = {name: 0.0 for name in feature_names}
    for _ in range(mc_samples):
        order = np.random.permutation(p)
        prev_r2 = 0.0
        used = []
        for idx in order:
            used.append(idx)
            model = LinearRegression().fit(Xs[:, used], ys)
            r2 = model.score(Xs[:, used], ys)
            contrib[feature_names[idx]] += r2 - prev_r2
            prev_r2 = r2
    for name in feature_names:
        contrib[name] /= mc_samples
    total = sum(contrib.values())
    if np.isfinite(total) and abs(total) > 1e-12:
        return {name: 100.0 * val / total for name, val in contrib.items()}
    return {name: np.nan for name in feature_names}


def _ncvi_concurrent_select_bjt_window(da, init_date, window_start, window_end):
    if "step" not in da.dims:
        raise RuntimeError(f"NCVI_concurrent requires a step dimension; dims={da.dims}")
    step_hours = pd.to_timedelta(da["step"].values).total_seconds() / 3600.0
    valid_bjt = pd.Timestamp(init_date) + pd.to_timedelta(step_hours, unit="h") + pd.Timedelta(hours=8)
    start_date = pd.Timestamp(window_start).date()
    end_date = pd.Timestamp(window_end).date()
    mask = np.array([(ts.date() >= start_date) and (ts.date() <= end_date) for ts in valid_bjt], dtype=bool)
    if not mask.any():
        raise RuntimeError(
            f"NCVI_concurrent found no S2S steps in BJT window for init={init_date}, "
            f"window={window_start} to {window_end}. Available BJT dates: "
            f"{sorted({ts.strftime('%Y-%m-%d') for ts in valid_bjt})}"
        )
    return da.isel(step=np.where(mask)[0]), valid_bjt[mask]


def _ncvi_concurrent_candidate_files(init_date):
    date_dash = init_date
    date_nodash = init_date.replace("-", "")
    hc_pf_candidates = [
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_pt320_60_{date_dash}.grib",
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_pt320_{date_dash}.grib",
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_60_{date_dash}.grib",
        f"{MSE_DIR}/Antecedent/ecmf_pf_NCVI_{date_dash}.grib",
    ]
    hc_cf_candidates = [
        f"{CF_DIR}/ecmf_cf_NCVI_pt320_60_{date_dash}.grib",
        f"{CF_DIR}/ecmf_cf_NCVI_pt320_{date_dash}.grib",
        f"{CF_DIR}/ecmf_cf_NCVI_60_{date_dash}.grib",
        f"{CF_DIR}/ecmf_cf_NCVI_{date_dash}.grib",
    ]
    hc_pf_candidates += sorted(set(glob.glob(f"{MSE_DIR}/Antecedent/**/*NCVI*{date_dash}*.grib", recursive=True) + glob.glob(f"{MSE_DIR}/Antecedent/**/*NCVI*{date_nodash}*.grib", recursive=True)))
    hc_cf_candidates += sorted(set(glob.glob(f"{CF_DIR}/**/*NCVI*{date_dash}*.grib", recursive=True) + glob.glob(f"{CF_DIR}/**/*NCVI*{date_nodash}*.grib", recursive=True) + glob.glob(f"{CF_DIR}/**/*ncvi*{date_dash}*.grib", recursive=True) + glob.glob(f"{CF_DIR}/**/*ncvi*{date_nodash}*.grib", recursive=True)))
    hc_pf = next((p for p in hc_pf_candidates if os.path.exists(p)), None)
    hc_cf = next((p for p in hc_cf_candidates if os.path.exists(p)), None)
    return hc_pf, hc_cf


def compute_ncvi_concurrent(exp_start_date, stage, window_start, window_end):
    """Compute stage-concurrent NCVI with the same PV region/unit convention as compute_real_ncvi."""
    rt_da_all = load_s2s_ensemble_with_cf(f"{MSE_DIR}/ecmf_pf_60_2023-06.grib", f"{CF_DIR}/ecmf_cf_60_2023-06.grib", ["pv"])
    da_rt = get_s2s_init_slice(rt_da_all.to_dataset(name="pv"), "pv", exp_start_date) if rt_da_all is not None else None
    if da_rt is None:
        raise RuntimeError(f"Cannot load realtime PV data for NCVI_concurrent init={exp_start_date}")
    if "number" not in da_rt.dims:
        raise RuntimeError(f"NCVI_concurrent realtime PV lacks number dimension for init={exp_start_date}; dims={da_rt.dims}")
    da_rt = da_rt.assign_coords(number=np.arange(da_rt.sizes["number"]))
    da_rt_mean = safe_slice_region(da_rt, 35, 50, 115, 130).mean(dim=[_detect_lat_dim(da_rt), _detect_lon_dim(da_rt)])
    rt_window, rt_valid_bjt = _ncvi_concurrent_select_bjt_window(da_rt_mean, exp_start_date, window_start, window_end)
    da_for_members = rt_window
    anomaly_used = False
    climatology_file_used = ""
    fallback_reason = "climatology_not_used"

    hc_pf, hc_cf = _ncvi_concurrent_candidate_files(exp_start_date)
    if hc_pf and hc_cf:
        climatology_file_used = f"pf={hc_pf}; cf={hc_cf}"
        hc_da_all = load_s2s_ensemble_with_cf(hc_pf, hc_cf, ["pv"])
        if hc_da_all is not None:
            rn = {d: ("latitude" if "lat" in str(d).lower() else "longitude") for d in hc_da_all.dims if "lat" in str(d).lower() or "lon" in str(d).lower()}
            if rn:
                hc_da_all = hc_da_all.rename(rn)
            da_hc = get_s2s_init_slice(hc_da_all.to_dataset(name="pv"), "pv", exp_start_date)
            if da_hc is None:
                da_hc = hc_da_all
            try:
                da_hc_mean = safe_slice_region(da_hc, 35, 50, 115, 130).mean(dim=[_detect_lat_dim(da_hc), _detect_lon_dim(da_hc)])
                hc_window, _ = _ncvi_concurrent_select_bjt_window(da_hc_mean, exp_start_date, window_start, window_end)
                hc_drop_dims = list(hc_window.dims)
                hc_clim_scalar = hc_window.mean(dim=hc_drop_dims, skipna=True)
                da_for_members = rt_window - hc_clim_scalar
                anomaly_used = True
                fallback_reason = ""
            except Exception as exc:
                fallback_reason = f"climatology_window_failed: {exc}"
        else:
            fallback_reason = "climatology_files_found_but_unreadable"
    else:
        fallback_reason = "climatology_file_missing"

    mean_dims = [d for d in da_for_members.dims if d != "number"]
    ncvi_values = np.asarray(da_for_members.mean(dim=mean_dims, skipna=True).sel(number=np.arange(da_rt.sizes["number"])).values, dtype=float).squeeze() * 1e6
    if ncvi_values.shape != (51,):
        raise RuntimeError(f"NCVI_concurrent shape check failed for {exp_start_date}/{stage}: expected (51,), got {ncvi_values.shape}")
    return {
        "values": ncvi_values,
        "anomaly_used": bool(anomaly_used),
        "climatology_file_used": climatology_file_used,
        "fallback_reason": fallback_reason,
        "valid_bjt_dates": sorted({pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in rt_valid_bjt}),
        "window_start": window_start,
        "window_end": window_end,
    }


def _ncvi_concurrent_prepare_stage_group(df_stage_values, init, stage):
    g = df_stage_values[(df_stage_values["init_date"] == init) & (df_stage_values["stage"] == stage)].copy()
    if "member" not in g.columns:
        raise RuntimeError(f"NCVI_concurrent stage table lacks member column for {init}/{stage}")
    g = g.sort_values("member").reset_index(drop=True)
    base_required = [
        "member",
        "y_tmax",
        "WNPSH",
        "z500_anom_NCHN",
        "NCHN_tp",
        "sm_avg",
        "SHF_Avg",
        "Initial_MSEstar_max_NCHN",
        "Initial_Barrier_NCHN",
        "NCVI_old",
        "NCVI_concurrent",
        "TCC_Avg",
    ]
    missing_base = [c for c in base_required if c not in g.columns]
    if missing_base:
        raise RuntimeError(f"NCVI_concurrent missing base columns for {init}/{stage}: {missing_base}")
    if len(g) != 51 or g["member"].nunique() != 51:
        raise RuntimeError(f"NCVI_concurrent expected 51 members for {init}/{stage}, got rows={len(g)}, unique_members={g['member'].nunique()}")

    if stage == "Stage-I_dry":
        g["SHF_resid"], _ = _ncvi_concurrent_residual(g["SHF_Avg"], g[["sm_avg"]])
        g["TCC_resid"], _ = _ncvi_concurrent_residual(g["TCC_Avg"], g[["sm_avg"]])
    else:
        g["SM_resid"], _ = _ncvi_concurrent_residual(g["sm_avg"], g[["NCHN_tp"]])
        g["SHF_resid"], _ = _ncvi_concurrent_residual(g["SHF_Avg"], g[["NCHN_tp"]])
        g["TCC_resid"], _ = _ncvi_concurrent_residual(g["TCC_Avg"], g[["NCHN_tp"]])
    return g


def _ncvi_concurrent_fit_lmg_lofo(g, init, stage, model_version, ncvi_col):
    predictors = _ncvi_concurrent_predictors_for_stage(stage, ncvi_col)
    required = ["member", "y_tmax"] + predictors
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise RuntimeError(f"NCVI_concurrent missing model columns for {init}/{stage}/{model_version}: {missing}")
    g_model = g.dropna(subset=required).sort_values("member").reset_index(drop=True)
    if len(g_model) != 51 or g_model["member"].nunique() != 51:
        raise RuntimeError(f"NCVI_concurrent expected 51 modeling members for {init}/{stage}/{model_version}, got rows={len(g_model)}, unique_members={g_model['member'].nunique()}")

    y = g_model["y_tmax"].astype(float).to_numpy()
    X_df = g_model[predictors].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    full_model = sm.OLS(y, sm.add_constant(X_scaled, has_constant="add")).fit()
    fitted = np.asarray(full_model.fittedvalues, dtype=float)
    r2_full = float(full_model.rsquared)
    adjr2_full = float(full_model.rsquared_adj)
    rmse_full = _ncvi_concurrent_safe_rmse(y, fitted)

    lmg_pct = _ncvi_concurrent_compute_lmg_mc(X_df.to_numpy(), y, predictors, mc_samples=2000)
    lmg_rank_map = {
        factor: rank
        for rank, factor in enumerate(
            sorted(predictors, key=lambda f: np.nan_to_num(lmg_pct.get(f, np.nan), nan=-np.inf), reverse=True),
            start=1,
        )
    }
    lofo_rows = []
    for factor_idx, factor in enumerate(predictors):
        X_reduced = np.delete(X_scaled, factor_idx, axis=1)
        reduced_model = sm.OLS(y, sm.add_constant(X_reduced, has_constant="add")).fit()
        r2_without = float(reduced_model.rsquared)
        adjr2_without = float(reduced_model.rsquared_adj)
        lofo_rows.append({
            "init_date": init,
            "stage": stage,
            "model_version": model_version,
            "factor": factor,
            "R2_full": r2_full,
            "AdjR2_full": adjr2_full,
            "R2_without_factor": r2_without,
            "AdjR2_without_factor": adjr2_without,
            "LOFO_delta_R2": r2_full - r2_without,
            "LOFO_delta_adjR2": adjr2_full - adjr2_without,
        })
    lofo_df = pd.DataFrame(lofo_rows)
    lofo_df["LOFO_rank_R2"] = lofo_df["LOFO_delta_R2"].rank(ascending=False, method="min").astype(int)
    lofo_df["LOFO_rank_adjR2"] = lofo_df["LOFO_delta_adjR2"].rank(ascending=False, method="min").astype(int)

    lmg_rows = []
    for factor in predictors:
        lmg_rows.append({
            "init_date": init,
            "stage": stage,
            "model_version": model_version,
            "factor": factor,
            "LMG_importance_percent": lmg_pct.get(factor, np.nan),
            "LMG_rank": lmg_rank_map[factor],
            "R2_full": r2_full,
            "AdjR2_full": adjr2_full,
        })
    lmg_df = pd.DataFrame(lmg_rows)
    ncvi_lmg = lmg_df[lmg_df["factor"] == ncvi_col].iloc[0]
    ncvi_lofo = lofo_df[lofo_df["factor"] == ncvi_col].iloc[0]
    top1_lmg = lmg_df.sort_values("LMG_rank").iloc[0]
    model_summary = {
        "init_date": init,
        "stage": stage,
        "model_version": model_version,
        "n_members": int(len(g_model)),
        "n_predictors": int(len(predictors)),
        "R2_full": r2_full,
        "AdjR2_full": adjr2_full,
        "RMSE_full": rmse_full,
        "predictor_list": ";".join(predictors),
        "NCVI_factor_used": ncvi_col,
        "Top1_LMG_factor": top1_lmg["factor"],
        "Top1_LMG_importance_percent": float(top1_lmg["LMG_importance_percent"]),
        "NCVI_LMG_importance_percent": float(ncvi_lmg["LMG_importance_percent"]),
        "NCVI_LMG_rank": int(ncvi_lmg["LMG_rank"]),
        "NCVI_LOFO_delta_R2": float(ncvi_lofo["LOFO_delta_R2"]),
        "NCVI_LOFO_delta_adjR2": float(ncvi_lofo["LOFO_delta_adjR2"]),
        "NCVI_LOFO_rank_R2": int(ncvi_lofo["LOFO_rank_R2"]),
        "figure_suite_generated": bool(ncvi_col == "NCVI_concurrent"),
        "NCVI_version": "concurrent" if ncvi_col == "NCVI_concurrent" else "old",
    }
    fit_df = pd.DataFrame({
        "init_date": init,
        "stage": stage,
        "model_version": model_version,
        "member": g_model["member"].to_numpy(),
        "y_tmax": y,
        "full_fitted_tmax": fitted,
    })
    return model_summary, lmg_df, lofo_df, fit_df


ncvi_stage_value_records = []
ncvi_metadata_by_group = {}
for init in NCVI_CONCURRENT_INIT_ORDER:
    for stage, (window_start, window_end) in NCVI_CONCURRENT_STAGE_WINDOWS.items():
        result = compute_ncvi_concurrent(init, stage, window_start, window_end)
        ncvi_metadata_by_group[(init, stage)] = result
        g_old = df_ncvi_stage_raw[(df_ncvi_stage_raw["init_date"] == init) & (df_ncvi_stage_raw["stage"] == stage)].copy()
        if "member" not in g_old.columns:
            raise RuntimeError(f"NCVI_concurrent base table lacks member column for {init}/{stage}")
        g_old = g_old.sort_values("member").reset_index(drop=True)
        if len(g_old) != 51 or g_old["member"].nunique() != 51:
            raise RuntimeError(f"NCVI_concurrent base table expected 51 members for {init}/{stage}, got rows={len(g_old)}, unique_members={g_old['member'].nunique()}")
        if "NCVI" not in g_old.columns:
            raise RuntimeError(f"NCVI_concurrent base table lacks NCVI column for {init}/{stage}")
        for row, ncvi_concurrent in zip(g_old.itertuples(index=False), result["values"]):
            ncvi_old = float(getattr(row, "NCVI"))
            ncvi_stage_value_records.append({
                "init_date": init,
                "stage": stage,
                "member": getattr(row, "member"),
                "NCVI_old": ncvi_old,
                "NCVI_concurrent": float(ncvi_concurrent),
                "NCVI_delta_concurrent_minus_old": float(ncvi_concurrent - ncvi_old),
                "ncvi_old_source": "table_stage_member_raw_factors_TCC.NCVI",
                "ncvi_concurrent_window_start": window_start,
                "ncvi_concurrent_window_end": window_end,
                "anomaly_used": result["anomaly_used"],
                "climatology_file_used": result["climatology_file_used"],
                "fallback_reason": result["fallback_reason"],
            })
        print(
            f"[NCVI_concurrent] {init} | {stage}: n=51, anomaly_used={result['anomaly_used']}, "
            f"window={window_start} to {window_end}, valid_bjt_dates={result['valid_bjt_dates']}"
        )

df_ncvi_values = pd.DataFrame(ncvi_stage_value_records)

df_ncvi_model_base = df_ncvi_stage_raw.merge(
    df_ncvi_values[["init_date", "stage", "member", "NCVI_old", "NCVI_concurrent"]],
    on=["init_date", "stage", "member"],
    how="inner",
)

ncvi_model_summary_records = []
ncvi_lmg_records = []
ncvi_lofo_records = []
ncvi_fit_records = []
ncvi_qc_lines = [
    "NCVI_concurrent sensitivity for noSSR + TCC_resid",
    "This module compares the original NCVI factor with a stage-concurrent NCVI diagnostic.",
    "If NCVI_concurrent ranks substantially higher than NCVI_old, the stage-concurrent vortex activity may contain more direct member-axis information for Tmax or precipitation differences.",
    "If NCVI_concurrent remains low-ranked, the cold-vortex index is unlikely to be a dominant source of ensemble spread in this diagnostic framework.",
    "LMG and LOFO are diagnostic attribution tools and do not prove causality.",
    "NCVI_concurrent is not residualized; only NCVI_old versus NCVI_concurrent replacement is tested in the primary models.",
    "TCC_resid follows Cell 6: residual(TCC_Avg ~ sm_avg) for Stage-I and residual(TCC_Avg ~ NCHN_tp) for Stage-II/Total.",
    "This module outputs both old-vs-concurrent comparison figures and a concurrent NCVI standalone figure suite.",
    "Concurrent standalone figures are intended for one-to-one comparison with existing Cell 5 / Cell 6 figures.",
    "y_tmax appears in target-predictor correlation matrices for diagnostics only and is not added as an extra predictor in OLS/LMG/LOFO.",
    "",
]

for init in NCVI_CONCURRENT_INIT_ORDER:
    for stage in NCVI_CONCURRENT_STAGE_ORDER:
        g = _ncvi_concurrent_prepare_stage_group(df_ncvi_model_base, init, stage)
        r_old_conc, p_old_conc = _ncvi_concurrent_safe_corr_p(g["NCVI_old"], g["NCVI_concurrent"])
        r_old_tmax, p_old_tmax = _ncvi_concurrent_safe_corr_p(g["NCVI_old"], g["y_tmax"])
        r_conc_tmax, p_conc_tmax = _ncvi_concurrent_safe_corr_p(g["NCVI_concurrent"], g["y_tmax"])
        r_old_tp, p_old_tp = _ncvi_concurrent_safe_corr_p(g["NCVI_old"], g["NCHN_tp"])
        r_conc_tp, p_conc_tp = _ncvi_concurrent_safe_corr_p(g["NCVI_concurrent"], g["NCHN_tp"])
        ncvi_qc_lines.extend([
            f"[{init} | {stage}]",
            f"NCVI_concurrent member count = {len(g)}",
            f"NCVI_old mean/std/min/max = {g['NCVI_old'].mean():.6f}/{g['NCVI_old'].std():.6f}/{g['NCVI_old'].min():.6f}/{g['NCVI_old'].max():.6f}",
            f"NCVI_concurrent mean/std/min/max = {g['NCVI_concurrent'].mean():.6f}/{g['NCVI_concurrent'].std():.6f}/{g['NCVI_concurrent'].min():.6f}/{g['NCVI_concurrent'].max():.6f}",
            f"corr(NCVI_old, NCVI_concurrent) = {r_old_conc:.6f}, p={p_old_conc:.6g}",
            f"corr(NCVI_old, y_tmax) = {r_old_tmax:.6f}, p={p_old_tmax:.6g}",
            f"corr(NCVI_concurrent, y_tmax) = {r_conc_tmax:.6f}, p={p_conc_tmax:.6g}",
            f"corr(NCVI_old, NCHN_tp) = {r_old_tp:.6f}, p={p_old_tp:.6g}",
            f"corr(NCVI_concurrent, NCHN_tp) = {r_conc_tp:.6f}, p={p_conc_tp:.6g}",
            f"anomaly_used = {ncvi_metadata_by_group[(init, stage)]['anomaly_used']}",
            f"climatology_file_used = {ncvi_metadata_by_group[(init, stage)]['climatology_file_used']}",
            f"fallback_reason = {ncvi_metadata_by_group[(init, stage)]['fallback_reason']}",
        ])

        for model_version, ncvi_col in [
            ("noSSR_TCCresid_NCVIold", "NCVI_old"),
            ("noSSR_TCCresid_NCVIconcurrent", "NCVI_concurrent"),
        ]:
            model_summary, lmg_df, lofo_df, fit_df = _ncvi_concurrent_fit_lmg_lofo(g, init, stage, model_version, ncvi_col)
            ncvi_model_summary_records.append(model_summary)
            ncvi_lmg_records.extend(lmg_df.to_dict("records"))
            ncvi_lofo_records.extend(lofo_df.to_dict("records"))
            ncvi_fit_records.extend(fit_df.to_dict("records"))
            ncvi_qc_lines.extend([
                f"{model_version} R2/AdjR2/RMSE = {model_summary['R2_full']:.6f}/{model_summary['AdjR2_full']:.6f}/{model_summary['RMSE_full']:.6f}",
                f"{model_version} NCVI LMG rank / importance = {model_summary['NCVI_LMG_rank']} / {model_summary['NCVI_LMG_importance_percent']:.6f}%",
                f"{model_version} NCVI LOFO rank_R2 / Delta_R2 = {model_summary['NCVI_LOFO_rank_R2']} / {model_summary['NCVI_LOFO_delta_R2']:.6f}",
            ])
            print(
                f"[NCVI_concurrent sensitivity] {init} | {stage} | {model_version}: "
                f"R2={model_summary['R2_full']:.3f}, AdjR2={model_summary['AdjR2_full']:.3f}, "
                f"NCVI LMG rank={model_summary['NCVI_LMG_rank']}, NCVI LOFO rank_R2={model_summary['NCVI_LOFO_rank_R2']}"
            )
        ncvi_qc_lines.append("")


df_ncvi_model_cmp = pd.DataFrame(ncvi_model_summary_records)
df_ncvi_lmg = pd.DataFrame(ncvi_lmg_records)
df_ncvi_lofo = pd.DataFrame(ncvi_lofo_records)
df_ncvi_fit = pd.DataFrame(ncvi_fit_records)
for _df in [df_ncvi_values, df_ncvi_model_cmp, df_ncvi_lmg, df_ncvi_lofo, df_ncvi_fit]:
    _df["_stage_order"] = _df["stage"].map(NCVI_CONCURRENT_STAGE_RANK)

df_ncvi_values.sort_values(["init_date", "_stage_order", "member"], inplace=True)
df_ncvi_model_cmp.sort_values(["init_date", "_stage_order", "model_version"], inplace=True)
df_ncvi_lmg.sort_values(["init_date", "_stage_order", "model_version", "LMG_rank"], inplace=True)
df_ncvi_lofo.sort_values(["init_date", "_stage_order", "model_version", "LOFO_rank_R2"], inplace=True)
df_ncvi_fit.sort_values(["init_date", "_stage_order", "model_version", "member"], inplace=True)
for _df in [df_ncvi_values, df_ncvi_model_cmp, df_ncvi_lmg, df_ncvi_lofo, df_ncvi_fit]:
    _df.drop(columns="_stage_order", inplace=True)

table_ncvi_values_csv = f"{NCVI_CONCURRENT_SUBDIRS['tables']}/table_ncvi_concurrent_stage_values.csv"
table_ncvi_cmp_csv = f"{NCVI_CONCURRENT_SUBDIRS['tables']}/ncvi_concurrent_model_comparison.csv"
table_ncvi_lmg_csv = f"{NCVI_CONCURRENT_SUBDIRS['tables']}/ncvi_concurrent_lmg_long.csv"
table_ncvi_lofo_csv = f"{NCVI_CONCURRENT_SUBDIRS['tables']}/ncvi_concurrent_lofo_long.csv"
table_ncvi_fit_csv = f"{NCVI_CONCURRENT_SUBDIRS['tables']}/ncvi_concurrent_full_model_fitted.csv"
qc_ncvi_txt = f"{NCVI_CONCURRENT_SUBDIRS['qc']}/qc_ncvi_concurrent_sensitivity.txt"
for _out_path in [table_ncvi_values_csv, table_ncvi_cmp_csv, table_ncvi_lmg_csv, table_ncvi_lofo_csv, table_ncvi_fit_csv, qc_ncvi_txt]:
    assert_ncvi_concurrent_output_path(_out_path)

df_ncvi_values.to_csv(table_ncvi_values_csv, index=False)
df_ncvi_model_cmp.to_csv(table_ncvi_cmp_csv, index=False)
df_ncvi_lmg.to_csv(table_ncvi_lmg_csv, index=False)
df_ncvi_lofo.to_csv(table_ncvi_lofo_csv, index=False)
df_ncvi_fit.to_csv(table_ncvi_fit_csv, index=False)
with open(qc_ncvi_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(ncvi_qc_lines))

# Figure 1: NCVI_old vs NCVI_concurrent scatter diagnostics.
plot_inits = NCVI_CONCURRENT_INIT_ORDER
plot_stages = NCVI_CONCURRENT_STAGE_ORDER
nrows = len(plot_inits)
ncols = len(plot_stages)

x_all = df_ncvi_values["NCVI_old"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
y_all = df_ncvi_values["NCVI_concurrent"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()


def _ncvi_scatter_pad_limits(vals, frac=0.08):
    if vals.empty:
        return -1.0, 1.0
    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 0.1)
    else:
        pad = (vmax - vmin) * frac
    return vmin - pad, vmax + pad


xlim = _ncvi_scatter_pad_limits(x_all)
ylim = _ncvi_scatter_pad_limits(y_all)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(4.2 * ncols, 3.1 * nrows),
    constrained_layout=True,
    squeeze=False,
)
for i, init in enumerate(plot_inits):
    for j, stage in enumerate(plot_stages):
        ax = axes[i, j]
        gg = df_ncvi_values[
            (df_ncvi_values["init_date"] == init)
            & (df_ncvi_values["stage"] == stage)
        ].copy()
        if gg.empty:
            ax.text(
                0.5,
                0.5,
                "missing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="0.4",
            )
            ax.set_axis_off()
            continue
        r, p = _ncvi_concurrent_safe_corr_p(gg["NCVI_old"], gg["NCVI_concurrent"])
        ax.scatter(
            gg["NCVI_old"],
            gg["NCVI_concurrent"],
            s=18,
            alpha=0.75,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_title(
            f"{init} | {stage}\n"
            f"r={r:.2f}, p={p:.2g}, n={len(gg)}",
            fontsize=9,
        )
        if i == nrows - 1:
            ax.set_xlabel("NCVI old", fontsize=9)
        else:
            ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("NCVI concurrent", fontsize=9)
        else:
            ax.set_ylabel("")
fig.suptitle(
    "NCVI old vs stage-concurrent NCVI anomaly diagnostics",
    fontsize=14,
    fontweight="bold",
)
fig_ncvi_scatter_png = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_old_vs_concurrent_scatter.png"
fig_ncvi_scatter_pdf = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_old_vs_concurrent_scatter.pdf"
for _out_path in [fig_ncvi_scatter_png, fig_ncvi_scatter_pdf]:
    assert_ncvi_concurrent_output_path(_out_path)
fig.savefig(fig_ncvi_scatter_png, dpi=240)
fig.savefig(fig_ncvi_scatter_pdf)
plt.close(fig)

# Figure 2: LMG heatmap comparison.
row_order = [NCVI_CONCURRENT_ROW_LABELS[(init, stage)] for init in NCVI_CONCURRENT_INIT_ORDER for stage in NCVI_CONCURRENT_STAGE_ORDER]
plot_lmg = df_ncvi_lmg.copy()
plot_lmg["row_label"] = plot_lmg.apply(lambda r: NCVI_CONCURRENT_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
lmg_old = plot_lmg[plot_lmg["model_version"] == "noSSR_TCCresid_NCVIold"].pivot(index="row_label", columns="factor", values="LMG_importance_percent").reindex(index=row_order, columns=NCVI_CONCURRENT_HEATMAP_ORDER_OLD)
lmg_conc = plot_lmg[plot_lmg["model_version"] == "noSSR_TCCresid_NCVIconcurrent"].pivot(index="row_label", columns="factor", values="LMG_importance_percent").reindex(index=row_order, columns=NCVI_CONCURRENT_HEATMAP_ORDER_CONCURRENT)
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), constrained_layout=True)
sns.heatmap(lmg_old, ax=axes[0], cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "LMG importance (%)"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[0].set_title("NCVI_old model")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_xticklabels([NCVI_CONCURRENT_FACTOR_LABELS.get(c, c) for c in NCVI_CONCURRENT_HEATMAP_ORDER_OLD], rotation=45, ha="right")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
sns.heatmap(lmg_conc, ax=axes[1], cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={"label": "LMG importance (%)"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[1].set_title("NCVI conc. model")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_xticklabels([NCVI_CONCURRENT_FACTOR_LABELS.get(c, c) for c in NCVI_CONCURRENT_HEATMAP_ORDER_CONCURRENT], rotation=45, ha="right")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
fig.suptitle("LMG comparison | NCVI_old vs NCVI conc. | noSSR + TCC_resid", fontsize=15)
fig_ncvi_lmg_png = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_old_vs_concurrent_lmg_heatmap.png"
fig_ncvi_lmg_pdf = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_old_vs_concurrent_lmg_heatmap.pdf"
for _out_path in [fig_ncvi_lmg_png, fig_ncvi_lmg_pdf]:
    assert_ncvi_concurrent_output_path(_out_path)
fig.savefig(fig_ncvi_lmg_png, dpi=240)
fig.savefig(fig_ncvi_lmg_pdf)
plt.close(fig)

# Figure 3: LOFO Delta R2 heatmap comparison.
plot_lofo = df_ncvi_lofo.copy()
plot_lofo["row_label"] = plot_lofo.apply(lambda r: NCVI_CONCURRENT_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
lofo_old = plot_lofo[plot_lofo["model_version"] == "noSSR_TCCresid_NCVIold"].pivot(index="row_label", columns="factor", values="LOFO_delta_R2").reindex(index=row_order, columns=NCVI_CONCURRENT_HEATMAP_ORDER_OLD)
lofo_conc = plot_lofo[plot_lofo["model_version"] == "noSSR_TCCresid_NCVIconcurrent"].pivot(index="row_label", columns="factor", values="LOFO_delta_R2").reindex(index=row_order, columns=NCVI_CONCURRENT_HEATMAP_ORDER_CONCURRENT)
fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), constrained_layout=True)
sns.heatmap(lofo_old, ax=axes[0], cmap="coolwarm", center=0, annot=True, fmt=".3f", cbar_kws={"label": "LOFO Delta R2"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[0].set_title("NCVI_old model")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_xticklabels([NCVI_CONCURRENT_FACTOR_LABELS.get(c, c) for c in NCVI_CONCURRENT_HEATMAP_ORDER_OLD], rotation=45, ha="right")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
sns.heatmap(lofo_conc, ax=axes[1], cmap="coolwarm", center=0, annot=True, fmt=".3f", cbar_kws={"label": "LOFO Delta R2"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 8})
axes[1].set_title("NCVI conc. model")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_xticklabels([NCVI_CONCURRENT_FACTOR_LABELS.get(c, c) for c in NCVI_CONCURRENT_HEATMAP_ORDER_CONCURRENT], rotation=45, ha="right")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
fig.suptitle("LOFO Delta R2 comparison | NCVI_old vs NCVI conc. | noSSR + TCC_resid", fontsize=15)
fig_ncvi_lofo_png = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_old_vs_concurrent_lofo_heatmap.png"
fig_ncvi_lofo_pdf = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_old_vs_concurrent_lofo_heatmap.pdf"
for _out_path in [fig_ncvi_lofo_png, fig_ncvi_lofo_pdf]:
    assert_ncvi_concurrent_output_path(_out_path)
fig.savefig(fig_ncvi_lofo_png, dpi=240)
fig.savefig(fig_ncvi_lofo_pdf)
plt.close(fig)

# Figure 4: NCVI rank summary.
rank_df = df_ncvi_model_cmp.copy()
rank_df["row_label"] = rank_df.apply(lambda r: NCVI_CONCURRENT_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
rank_df["model_label"] = rank_df["model_version"].map({
    "noSSR_TCCresid_NCVIold": "NCVI_old",
    "noSSR_TCCresid_NCVIconcurrent": "NCVI conc.",
})
rank_lmg = rank_df.pivot(index="row_label", columns="model_label", values="NCVI_LMG_rank").reindex(index=row_order, columns=["NCVI_old", "NCVI conc."])
rank_lofo = rank_df.pivot(index="row_label", columns="model_label", values="NCVI_LOFO_rank_R2").reindex(index=row_order, columns=["NCVI_old", "NCVI conc."])
fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), constrained_layout=True)
sns.heatmap(rank_lmg, ax=axes[0], cmap="viridis_r", annot=True, fmt=".0f", cbar_kws={"label": "LMG rank"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 9})
axes[0].set_title("NCVI LMG rank")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
sns.heatmap(rank_lofo, ax=axes[1], cmap="viridis_r", annot=True, fmt=".0f", cbar_kws={"label": "LOFO rank (Delta R2)"}, linewidths=0.4, linecolor="white", annot_kws={"fontsize": 9})
axes[1].set_title("NCVI LOFO rank_R2")
axes[1].set_xlabel("")
axes[1].set_ylabel("")
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
fig.suptitle("NCVI rank summary | old vs concurrent | noSSR + TCC_resid", fontsize=15)
fig_ncvi_rank_png = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_rank_summary.png"
fig_ncvi_rank_pdf = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_ncvi_rank_summary.pdf"
for _out_path in [fig_ncvi_rank_png, fig_ncvi_rank_pdf]:
    assert_ncvi_concurrent_output_path(_out_path)
fig.savefig(fig_ncvi_rank_png, dpi=240)
fig.savefig(fig_ncvi_rank_pdf)
plt.close(fig)


# Concurrent NCVI standalone figure suite A: Cell-5-style single-stage LMG combo figures.
for init in NCVI_CONCURRENT_INIT_ORDER:
    for stage in NCVI_CONCURRENT_STAGE_ORDER:
        model_version = "noSSR_TCCresid_NCVIconcurrent"
        ff = df_ncvi_fit[(df_ncvi_fit["init_date"] == init) & (df_ncvi_fit["stage"] == stage) & (df_ncvi_fit["model_version"] == model_version)]
        ss = df_ncvi_model_cmp[(df_ncvi_model_cmp["init_date"] == init) & (df_ncvi_model_cmp["stage"] == stage) & (df_ncvi_model_cmp["model_version"] == model_version)].iloc[0]
        bb = df_ncvi_lmg[(df_ncvi_lmg["init_date"] == init) & (df_ncvi_lmg["stage"] == stage) & (df_ncvi_lmg["model_version"] == model_version)].sort_values("LMG_importance_percent", ascending=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1.25]}, constrained_layout=True)
        ax_scatter, ax_bar = axes
        ax_scatter.scatter(ff["y_tmax"], ff["full_fitted_tmax"], s=32, alpha=0.82, color="#2f6db3")
        mn = float(min(ff["y_tmax"].min(), ff["full_fitted_tmax"].min()))
        mx = float(max(ff["y_tmax"].max(), ff["full_fitted_tmax"].max()))
        pad = max(0.2, 0.05 * (mx - mn))
        ax_scatter.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", lw=1)
        ax_scatter.set_xlim(mn - pad, mx + pad)
        ax_scatter.set_ylim(mn - pad, mx + pad)
        ax_scatter.set_xlabel("ECMWF member Tmax")
        ax_scatter.set_ylabel("OLS fitted Tmax")
        ax_scatter.set_title(f"OLS fit\nR²={ss['R2_full']:.2f}, Adj-R²={ss['AdjR2_full']:.2f}, RMSE={ss['RMSE_full']:.2f}")
        labels = [NCVI_CONCURRENT_FACTOR_LABELS.get(x, x) for x in bb["factor"]]
        vals = bb["LMG_importance_percent"].astype(float).to_numpy()
        ax_bar.barh(labels, vals, color="#4daf4a", alpha=0.85)
        xmax = float(np.nanmax([1.0, *vals]))
        ax_bar.set_xlim(0, xmax * 1.16)
        for y_idx, val in enumerate(vals):
            ax_bar.text(val + xmax * 0.015, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.5)
        ax_bar.set_title("LMG relative importance")
        ax_bar.set_xlabel("Importance (%)")
        fig.suptitle(f"noSSR_TCCresid_NCVIconcurrent | {init} | {stage}", fontsize=14)
        fig_lmg_combo_png = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_lmg_combo_noSSR_TCCresid_NCVIconcurrent_{init}_{stage}.png"
        fig_lmg_combo_pdf = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_lmg_combo_noSSR_TCCresid_NCVIconcurrent_{init}_{stage}.pdf"
        for _out_path in [fig_lmg_combo_png, fig_lmg_combo_pdf]:
            assert_ncvi_concurrent_output_path(_out_path)
        fig.savefig(fig_lmg_combo_png, dpi=240)
        fig.savefig(fig_lmg_combo_pdf)
        plt.close(fig)

# Concurrent NCVI standalone figure suite B: Cell-6-style per-init LOFO combo figures.
for init in NCVI_CONCURRENT_INIT_ORDER:
    model_version = "noSSR_TCCresid_NCVIconcurrent"
    fig, axes = plt.subplots(len(NCVI_CONCURRENT_STAGE_ORDER), 2, figsize=(12.5, 3.1 * len(NCVI_CONCURRENT_STAGE_ORDER)), constrained_layout=True)
    for row_idx, stage in enumerate(NCVI_CONCURRENT_STAGE_ORDER):
        ax_scatter = axes[row_idx, 0]
        ax_bar = axes[row_idx, 1]
        ff = df_ncvi_fit[(df_ncvi_fit["init_date"] == init) & (df_ncvi_fit["stage"] == stage) & (df_ncvi_fit["model_version"] == model_version)]
        ss = df_ncvi_model_cmp[(df_ncvi_model_cmp["init_date"] == init) & (df_ncvi_model_cmp["stage"] == stage) & (df_ncvi_model_cmp["model_version"] == model_version)].iloc[0]
        ax_scatter.scatter(ff["y_tmax"], ff["full_fitted_tmax"], s=28, alpha=0.8, color="#2f6db3")
        mn = float(min(ff["y_tmax"].min(), ff["full_fitted_tmax"].min()))
        mx = float(max(ff["y_tmax"].max(), ff["full_fitted_tmax"].max()))
        pad = max(0.2, 0.05 * (mx - mn))
        ax_scatter.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", lw=1)
        ax_scatter.set_xlim(mn - pad, mx + pad)
        ax_scatter.set_ylim(mn - pad, mx + pad)
        ax_scatter.set_title(f"{NCVI_CONCURRENT_ROW_LABELS[(init, stage)]}\nR²={ss['R2_full']:.2f}, Adj-R²={ss['AdjR2_full']:.2f}, RMSE={ss['RMSE_full']:.2f}")
        ax_scatter.set_xlabel("ECMWF member Tmax")
        ax_scatter.set_ylabel("Full-model fitted Tmax")
        bb = df_ncvi_lofo[(df_ncvi_lofo["init_date"] == init) & (df_ncvi_lofo["stage"] == stage) & (df_ncvi_lofo["model_version"] == model_version)].sort_values("LOFO_delta_R2", ascending=True)
        labels = [NCVI_CONCURRENT_FACTOR_LABELS.get(x, x) for x in bb["factor"]]
        vals = bb["LOFO_delta_R2"].astype(float).to_numpy()
        positive_sum = float(np.nansum(np.maximum(vals, 0.0)))
        vals_pct = np.maximum(vals, 0.0) / positive_sum * 100.0 if positive_sum > 0 else np.zeros_like(vals, dtype=float)
        colors = ["#d95f02" if v >= 0 else "#7570b3" for v in vals]
        bars = ax_bar.barh(labels, vals, color=colors, alpha=0.85)
        xmin = float(np.nanmin([0.0, *vals]))
        xmax = float(np.nanmax([0.0, *vals]))
        span = max(1e-6, xmax - xmin)
        ax_bar.set_xlim(xmin - 0.12 * span, xmax + 0.18 * span)
        ax_bar.axvline(0, color="0.3", lw=0.8)
        for rect, val, pct in zip(bars, vals, vals_pct):
            x = rect.get_width()
            y_mid = rect.get_y() + rect.get_height() / 2
            ha = "left" if val >= 0 else "right"
            dx = 0.015 * span if val >= 0 else -0.015 * span
            pct_display = pct if val > 0 else 0.0
            ax_bar.text(x + dx, y_mid, f"{val:.3f} ({pct_display:.1f}%)", va="center", ha=ha, fontsize=8.5)
        ax_bar.set_title("LOFO Delta R2")
        ax_bar.set_xlabel("Delta R2")
    fig.suptitle(f"LOFO validation combo | noSSR + TCC_resid + NCVI conc. | Init {init}", fontsize=15)
    fig_lofo_combo_png = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_lofo_combo_noSSR_TCCresid_NCVIconcurrent_init_{init}.png"
    fig_lofo_combo_pdf = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/fig_lofo_combo_noSSR_TCCresid_NCVIconcurrent_init_{init}.pdf"
    for _out_path in [fig_lofo_combo_png, fig_lofo_combo_pdf]:
        assert_ncvi_concurrent_output_path(_out_path)
    fig.savefig(fig_lofo_combo_png, dpi=240)
    fig.savefig(fig_lofo_combo_pdf)
    plt.close(fig)

# Concurrent NCVI standalone figure suite C: target-predictor correlation matrices.
for init in NCVI_CONCURRENT_INIT_ORDER:
    for stage in NCVI_CONCURRENT_STAGE_ORDER:
        g_corr = _ncvi_concurrent_prepare_stage_group(df_ncvi_model_base, init, stage)
        predictors = _ncvi_concurrent_predictors_for_stage(stage, "NCVI_concurrent")
        corr_cols = ["y_tmax"] + predictors
        missing_corr_cols = [c for c in corr_cols if c not in g_corr.columns]
        if missing_corr_cols:
            raise RuntimeError(f"NCVI_concurrent target-predictor matrix missing columns for {init}/{stage}: {missing_corr_cols}")
        corr_df = g_corr[corr_cols].astype(float).corr()
        corr_labels = ["Tmax"] + [NCVI_CONCURRENT_FACTOR_LABELS.get(c, c) for c in predictors]
        fig, ax = plt.subplots(figsize=(10.5, 9.0))
        sns.heatmap(corr_df, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, linewidths=0.4, linecolor="white", cbar_kws={"label": "Pearson r"}, annot_kws={"fontsize": 8})
        ax.set_xticklabels(corr_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_labels, rotation=0, fontsize=8)
        fig.subplots_adjust(left=0.22, bottom=0.22, right=0.96, top=0.90)
        ax.set_title(f"Target-predictor correlations | noSSR + TCC_resid + NCVI conc. | {init} {stage}")
        fig_corr_png = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/corr_target_predictors_tccresid_NCVIconcurrent_{init}_{stage}.png"
        fig_corr_pdf = f"{NCVI_CONCURRENT_SUBDIRS['figures']}/corr_target_predictors_tccresid_NCVIconcurrent_{init}_{stage}.pdf"
        for _out_path in [fig_corr_png, fig_corr_pdf]:
            assert_ncvi_concurrent_output_path(_out_path)
        fig.savefig(fig_corr_png, dpi=240)
        fig.savefig(fig_corr_pdf)
        plt.close(fig)


# Concurrent NCVI standalone figure suite D: daily Tmax diagnostic time series.
def _ncvi_concurrent_stage_day_count(stage):
    s, e = NCVI_CONCURRENT_STAGE_WINDOWS[stage]
    return int((pd.Timestamp(e) - pd.Timestamp(s)).days + 1)


def _ncvi_concurrent_daily_fit_for_stage(init, stage, ncvi_col):
    if stage not in ["Stage-I_dry", "Stage-II_wet"]:
        raise RuntimeError(f"NCVI_concurrent daily time series only supports Stage-I/Stage-II, got {stage}")
    sg = _ncvi_concurrent_prepare_stage_group(df_ncvi_model_base, init, stage)
    predictors = _ncvi_concurrent_predictors_for_stage(stage, ncvi_col)
    g_model = sg.dropna(subset=["member", "y_tmax"] + predictors).sort_values("member").reset_index(drop=True)
    if len(g_model) != 51 or g_model["member"].nunique() != 51:
        raise RuntimeError(f"NCVI_concurrent daily fit expected 51 stage members for {init}/{stage}/{ncvi_col}, got {len(g_model)}")
    scaler = StandardScaler()
    X_stage = scaler.fit_transform(g_model[predictors].astype(float))
    stage_model = sm.OLS(g_model["y_tmax"].astype(float).to_numpy(), sm.add_constant(X_stage, has_constant="add")).fit()

    if stage == "Stage-I_dry":
        shf_model = sm.OLS(sg["SHF_Avg"], sm.add_constant(sg[["sm_avg"]], has_constant="add")).fit()
        tcc_model = sm.OLS(sg["TCC_Avg"], sm.add_constant(sg[["sm_avg"]], has_constant="add")).fit()
        shf_ctrl = "sm_avg"
        tcc_ctrl = "sm_avg"
        sm_model = None
    else:
        sm_model = sm.OLS(sg["sm_avg"], sm.add_constant(sg[["NCHN_tp"]], has_constant="add")).fit()
        shf_model = sm.OLS(sg["SHF_Avg"], sm.add_constant(sg[["NCHN_tp"]], has_constant="add")).fit()
        tcc_model = sm.OLS(sg["TCC_Avg"], sm.add_constant(sg[["NCHN_tp"]], has_constant="add")).fit()
        shf_ctrl = "NCHN_tp"
        tcc_ctrl = "NCHN_tp"

    win_start, win_end = NCVI_CONCURRENT_STAGE_WINDOWS[stage]
    dg = df_ncvi_daily_member_raw[
        (df_ncvi_daily_member_raw["init_date"] == init)
        & (df_ncvi_daily_member_raw["date"] >= pd.Timestamp(win_start))
        & (df_ncvi_daily_member_raw["date"] <= pd.Timestamp(win_end))
    ].copy()
    if dg.empty:
        raise RuntimeError(f"NCVI_concurrent daily time series base rows are empty for {init}/{stage}")
    dg = dg.merge(
        df_ncvi_daily_tcc[df_ncvi_daily_tcc["init_date"] == init][["date", "member", "TCC_Avg"]],
        on=["date", "member"],
        how="left",
        suffixes=("", "_tcc_daily"),
    )
    if "TCC_Avg_tcc_daily" in dg.columns:
        dg["TCC_Avg"] = dg["TCC_Avg_tcc_daily"]
        dg = dg.drop(columns=["TCC_Avg_tcc_daily"])
    dg = dg.merge(
        sg[["member", "NCVI_old", "NCVI_concurrent"]],
        on="member",
        how="left",
    )
    daily_required = ["date", "member", "NCHN_tp", "sm_avg", "SHF_Avg", "TCC_Avg", "z500_anom_NCHN", "WNPSH", "Initial_MSEstar_max_NCHN", "Initial_Barrier_NCHN", ncvi_col]
    missing_daily = [c for c in daily_required if c not in dg.columns]
    if missing_daily:
        raise RuntimeError(f"NCVI_concurrent daily time series missing columns for {init}/{stage}/{ncvi_col}: {missing_daily}")
    before_drop = len(dg)
    dg = dg.dropna(subset=daily_required).copy()
    if dg.empty:
        raise RuntimeError(f"NCVI_concurrent daily time series rows empty after dropna for {init}/{stage}/{ncvi_col}; before={before_drop}")
    expected_days = _ncvi_concurrent_stage_day_count(stage)
    day_member_counts = dg.groupby("date")["member"].nunique()
    if len(day_member_counts) != expected_days or not (day_member_counts == 51).all():
        raise RuntimeError(
            f"NCVI_concurrent daily time series member count check failed for {init}/{stage}/{ncvi_col}: "
            f"expected {expected_days} days with 51 members, got {day_member_counts.to_dict()}"
        )

    dg["NCHN_tp"] = dg["NCHN_tp"].astype(float) * expected_days
    pred_shf = np.asarray(shf_model.predict(sm.add_constant(dg[[shf_ctrl]].astype(float), has_constant="add")), dtype=float).squeeze()
    pred_tcc = np.asarray(tcc_model.predict(sm.add_constant(dg[[tcc_ctrl]].astype(float), has_constant="add")), dtype=float).squeeze()
    if pred_shf.shape[0] != len(dg) or pred_tcc.shape[0] != len(dg):
        raise RuntimeError(
            f"NCVI_concurrent daily residual prediction shape mismatch for {init}/{stage}/{ncvi_col}: "
            f"expected {len(dg)}, got SHF={pred_shf.shape}, TCC={pred_tcc.shape}"
        )
    dg["SHF_resid"] = dg["SHF_Avg"].astype(float).values - pred_shf
    dg["TCC_resid"] = dg["TCC_Avg"].astype(float).values - pred_tcc
    if stage != "Stage-I_dry":
        pred_sm = np.asarray(sm_model.predict(sm.add_constant(dg[["NCHN_tp"]].astype(float), has_constant="add")), dtype=float).squeeze()
        if pred_sm.shape[0] != len(dg):
            raise RuntimeError(
                f"NCVI_concurrent daily SM residual prediction shape mismatch for {init}/{stage}/{ncvi_col}: "
                f"expected {len(dg)}, got {pred_sm.shape}"
            )
        dg["SM_resid"] = dg["sm_avg"].astype(float).values - pred_sm

    missing_predictors = [c for c in predictors if c not in dg.columns]
    if missing_predictors:
        raise RuntimeError(f"NCVI_concurrent daily predictors missing for {init}/{stage}/{ncvi_col}: {missing_predictors}")
    X_daily = scaler.transform(dg[predictors].astype(float))
    pred_daily = stage_model.predict(sm.add_constant(X_daily, has_constant="add"))
    pred_daily = np.asarray(pred_daily, dtype=float).squeeze()
    if pred_daily.shape[0] != len(dg):
        raise RuntimeError(
            f"NCVI_concurrent daily fitted shape mismatch for {init}/{stage}/{ncvi_col}: "
            f"expected {len(dg)}, got {pred_daily.shape}"
        )
    dg["OLS_daily_fitted"] = pred_daily
    return dg[["date", "member", "OLS_daily_fitted"]].assign(init_date=init, stage=stage, model_version=f"noSSR_TCCresid_{'NCVIconcurrent' if ncvi_col == 'NCVI_concurrent' else 'NCVIold'}")


def _ncvi_concurrent_plot_timeseries(init, stats_df, fit_stats, output_png, output_pdf, title):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axvspan(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), color="#F6D365", alpha=0.20, label="Stage-I")
    ax.axvspan(pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24") + pd.Timedelta(hours=18), color="#A1C4FD", alpha=0.20, label="Stage-II")
    ax.plot(stats_df["date"], stats_df["ERA5_absT"], color="black", lw=2.0, label="ERA5 Tmax")
    ax.plot(stats_df["date"], stats_df["S2S_mean_absT"], color="#1f77b4", lw=1.8, label="S2S mean Tmax")
    ax.fill_between(stats_df["date"], stats_df["S2S_mean_absT"] - stats_df["Spread_S2S"], stats_df["S2S_mean_absT"] + stats_df["Spread_S2S"], color="#1f77b4", alpha=0.12, label="S2S spread")
    ax.plot(fit_stats["date"], fit_stats["OLS_daily_fitted_mean"], color="#2E8B57", lw=1.8, marker="s", ms=4, label="OLS fitted mean")
    ax.fill_between(fit_stats["date"], fit_stats["OLS_daily_fitted_mean"] - fit_stats["OLS_daily_fitted_spread"], fit_stats["OLS_daily_fitted_mean"] + fit_stats["OLS_daily_fitted_spread"], color="#2E8B57", alpha=0.10, label="OLS fitted spread")
    ax.set_title(title)
    ax.set_ylabel("Tmax (°C)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower right", ncol=2, fontsize=8.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    for _out_path in [output_png, output_pdf]:
        assert_ncvi_concurrent_output_path(_out_path)
    fig.savefig(output_png, dpi=240)
    fig.savefig(output_pdf)
    plt.close(fig)

ncvi_timeseries_records = []
ncvi_timeseries_files = []
for init in NCVI_CONCURRENT_INIT_ORDER:
    fit_parts_concurrent = []
    for stage in ["Stage-I_dry", "Stage-II_wet"]:
        fit_parts_concurrent.append(_ncvi_concurrent_daily_fit_for_stage(init, stage, "NCVI_concurrent"))
    daily_fit_concurrent = pd.concat(fit_parts_concurrent, ignore_index=True)
    fit_stats_concurrent = daily_fit_concurrent.groupby("date", as_index=False).agg(
        OLS_daily_fitted_mean=("OLS_daily_fitted", "mean"),
        OLS_daily_fitted_spread=("OLS_daily_fitted", "std"),
    )
    stats_df = df_ncvi_daily_tmax[df_ncvi_daily_tmax["init_date"] == init].groupby(["init_date", "date"], as_index=False).agg(
        S2S_mean_absT=("s2s_tmax", "mean"),
        Spread_S2S=("s2s_tmax", "std"),
        ERA5_absT=("era5_tmax", "mean"),
    )
    stats_df = stats_df[(stats_df["date"] >= pd.Timestamp("2023-06-14")) & (stats_df["date"] <= pd.Timestamp("2023-06-24"))].copy()
    merged_check = stats_df[["date"]].merge(fit_stats_concurrent[["date"]], on="date", how="inner")
    if len(stats_df) != 11 or len(fit_stats_concurrent) != 11 or len(merged_check) != 11:
        raise RuntimeError(
            f"NCVI_concurrent time series date alignment failed for {init}: "
            f"daily_stats={len(stats_df)}, fitted={len(fit_stats_concurrent)}, merged={len(merged_check)}"
        )
    main_png = f"{NCVI_CONCURRENT_SUBDIRS['timeseries']}/fig_timeseries_noSSR_TCCresid_NCVIconcurrent_init_{init}.png"
    main_pdf = f"{NCVI_CONCURRENT_SUBDIRS['timeseries']}/fig_timeseries_noSSR_TCCresid_NCVIconcurrent_init_{init}.pdf"
    _ncvi_concurrent_plot_timeseries(
        init,
        stats_df,
        fit_stats_concurrent,
        main_png,
        main_pdf,
        f"Tmax daily diagnostic series | Init {init} | NCVI conc. stage-trained OLS",
    )
    ncvi_timeseries_files.extend([main_png, main_pdf])
    ncvi_timeseries_records.append(
        f"[{init}] time series dates={stats_df['date'].min():%Y-%m-%d} to {stats_df['date'].max():%Y-%m-%d}; "
        f"ERA5/S2S/fitted aligned days={len(merged_check)}; Stage-I and Stage-II stage-trained models applied."
    )

ncvi_qc_lines.extend([
    "Time series figure suite generated successfully.",
    "Cell 7 time-series only plots NCVI_concurrent stage-trained OLS; old-vs-concurrent time-series comparison is no longer generated.",
    "Time series files:",
    *[f"  {fp}" for fp in ncvi_timeseries_files],
    *ncvi_timeseries_records,
    "",
])
with open(qc_ncvi_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(ncvi_qc_lines))

# Final Cell-7 output and metadata checks for the primary v9 figure suite.
_expected_ncvi_version = {
    "noSSR_TCCresid_NCVIold": ("old", False),
    "noSSR_TCCresid_NCVIconcurrent": ("concurrent", True),
}
for _model_version, (_expected_version, _expected_suite_flag) in _expected_ncvi_version.items():
    _rows = df_ncvi_model_cmp[df_ncvi_model_cmp["model_version"] == _model_version]
    if _rows.empty:
        raise RuntimeError(f"NCVI_concurrent model comparison missing model_version={_model_version}")
    if set(_rows["NCVI_version"].astype(str)) != {_expected_version}:
        raise RuntimeError(
            f"NCVI_concurrent model comparison has incorrect NCVI_version for {_model_version}: "
            f"{sorted(set(_rows['NCVI_version'].astype(str)))}"
        )
    if set(_rows["figure_suite_generated"].astype(bool)) != {_expected_suite_flag}:
        raise RuntimeError(
            f"NCVI_concurrent model comparison has incorrect figure_suite_generated for {_model_version}: "
            f"{sorted(set(_rows['figure_suite_generated'].astype(bool)))}"
        )

_fig_dir = NCVI_CONCURRENT_SUBDIRS["figures"]
_ncvi_output_checks = {
    "single-stage concurrent LMG combo PNGs": (
        f"{_fig_dir}/fig_lmg_combo_noSSR_TCCresid_NCVIconcurrent_*.png",
        len(NCVI_CONCURRENT_INIT_ORDER) * len(NCVI_CONCURRENT_STAGE_ORDER),
    ),
    "per-init concurrent LOFO combo PNGs": (
        f"{_fig_dir}/fig_lofo_combo_noSSR_TCCresid_NCVIconcurrent_init_*.png",
        len(NCVI_CONCURRENT_INIT_ORDER),
    ),
    "concurrent target-predictor correlation PNGs": (
        f"{_fig_dir}/corr_target_predictors_tccresid_NCVIconcurrent_*.png",
        len(NCVI_CONCURRENT_INIT_ORDER) * len(NCVI_CONCURRENT_STAGE_ORDER),
    ),
    "concurrent main time series PNGs": (
        f"{NCVI_CONCURRENT_SUBDIRS['timeseries']}/fig_timeseries_noSSR_TCCresid_NCVIconcurrent_init_*.png",
        len(NCVI_CONCURRENT_INIT_ORDER),
    ),
}
for _desc, (_pattern, _expected_count) in _ncvi_output_checks.items():
    _matches = glob.glob(_pattern)
    if len(_matches) != _expected_count:
        raise RuntimeError(
            f"NCVI_concurrent output count check failed for {_desc}: "
            f"expected {_expected_count}, got {len(_matches)}, pattern={_pattern}, matches={_matches}"
        )

for _required_fig in [fig_ncvi_scatter_png, fig_ncvi_lmg_png, fig_ncvi_lofo_png, fig_ncvi_rank_png, *ncvi_timeseries_files]:
    assert_ncvi_concurrent_output_path(_required_fig)
    if not os.path.exists(_required_fig):
        raise RuntimeError(f"NCVI_concurrent required figure missing: {_required_fig}")

print("[NCVI_concurrent sensitivity] output count checks passed for primary v9 figure suite.")

print(f"[NCVI_concurrent sensitivity] stage values: {table_ncvi_values_csv}")
print(f"[NCVI_concurrent sensitivity] model comparison: {table_ncvi_cmp_csv}")
print(f"[NCVI_concurrent sensitivity] LMG long: {table_ncvi_lmg_csv}")
print(f"[NCVI_concurrent sensitivity] LOFO long: {table_ncvi_lofo_csv}")
print(f"[NCVI_concurrent sensitivity] QC summary: {qc_ncvi_txt}")
print(f"[NCVI_concurrent sensitivity] figures: {NCVI_CONCURRENT_SUBDIRS['figures']}")

# %%
# =============================================================================
# Cell 8: Final slim MLR with NCVI_concurrent
# =============================================================================

set_audit_cell_name("Cell 8: final slim MLR")
import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("\n" + "=" * 80)
print("Cell 8: Final slim MLR with NCVI_concurrent")
print("=" * 80)

SLIM_MLR_ROOT = FOUR_INIT_ROOT
SLIM_MLR_SUBDIRS = {
    "tables": f"{SLIM_MLR_ROOT}/tables",
    "figures": f"{SLIM_MLR_ROOT}/figures",
    "lmg_combo": f"{SLIM_MLR_ROOT}/figures/lmg_combo",
    "lmg_combo_init": f"{SLIM_MLR_ROOT}/figures/lmg_combo_init",
    "lofo_combo": f"{SLIM_MLR_ROOT}/figures/lofo_combo",
    "corr_matrix": f"{SLIM_MLR_ROOT}/figures/corr_matrix",
    "heatmap": f"{SLIM_MLR_ROOT}/figures/heatmap",
    "timeseries": f"{SLIM_MLR_ROOT}/figures/timeseries",
    "qc": f"{SLIM_MLR_ROOT}/qc",
}


def assert_slim_mlr_output_path(path):
    if not str(path).startswith(f"{FOUR_INIT_ROOT}/") and str(path) != FOUR_INIT_ROOT:
        raise RuntimeError(f"Slim MLR output path is outside FOUR_INIT_ROOT: {path}")


assert_slim_mlr_output_path(SLIM_MLR_ROOT + "/")
for _slim_dir in SLIM_MLR_SUBDIRS.values():
    assert_slim_mlr_output_path(_slim_dir + "/")
    os.makedirs(_slim_dir, exist_ok=True)

SLIM_STAGE_TCC_TABLE = f"{FOUR_INIT_ROOT}/tables/table_stage_member_raw_factors_TCC.csv"
SLIM_NCVI_STAGE_TABLE = f"{FOUR_INIT_ROOT}/tables/table_ncvi_concurrent_stage_values.csv"
SLIM_FULL_MODEL_CMP_TABLE = f"{FOUR_INIT_ROOT}/tables/ncvi_concurrent_model_comparison.csv"
SLIM_BASE_TABLES = f"{FOUR_INIT_ROOT}/tables"
SLIM_DAILY_MEMBER_RAW_TABLE = f"{SLIM_BASE_TABLES}/table_daily_member_raw_factors_SHF.csv"
SLIM_DAILY_TMAX_TABLE = f"{SLIM_BASE_TABLES}/table_daily_tmax_SHF.csv"
SLIM_DAILY_TCC_TABLE = f"{FOUR_INIT_ROOT}/tables/table_daily_member_tcc_SHF.csv"
for _required_slim_input in [SLIM_STAGE_TCC_TABLE, SLIM_NCVI_STAGE_TABLE, SLIM_FULL_MODEL_CMP_TABLE, SLIM_DAILY_MEMBER_RAW_TABLE, SLIM_DAILY_TMAX_TABLE, SLIM_DAILY_TCC_TABLE]:
    if not os.path.exists(_required_slim_input):
        raise RuntimeError(
            f"Required input for final slim MLR is missing: {_required_slim_input}. "
            "Please run Cell 5 and Cell 7 first before running the final slim MLR cell."
        )

SLIM_INIT_ORDER = FULL_INIT_LIST
SLIM_STAGE_ORDER = ["Stage-I_dry", "Stage-II_wet", "Total"]
SLIM_ROW_LABELS = {
    (init, stage): f"{pd.Timestamp(init):%m-%d} {'Stage-I' if stage == 'Stage-I_dry' else ('Stage-II' if stage == 'Stage-II_wet' else 'Total')}"
    for init in SLIM_INIT_ORDER for stage in SLIM_STAGE_ORDER
}
SLIM_DISPLAY_LABELS = {
    "sm_avg": "SM",
    "SM_resid": "SM resid",
    "SHF_resid": "SHF resid",
    "z500_anom_NCHN": "Z500 anom",
    "NCHN_tp": "TP",
    "TCC_resid": "TCC resid",
    "NCVI_concurrent": "NCVI conc.",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
}
SLIM_PREDICTORS_BY_STAGE = {
    "Stage-I_dry": [
        "sm_avg",
        "SHF_resid",
        "z500_anom_NCHN",
        "Initial_MSEstar_max_NCHN",
        "TCC_resid",
        "NCVI_concurrent",
    ],
    "Stage-II_wet": [
        "NCHN_tp",
        "z500_anom_NCHN",
        "NCVI_concurrent",
        "TCC_resid",
        "SM_resid",
        "Initial_MSEstar_max_NCHN",
    ],
    "Total": [
        "NCHN_tp",
        "z500_anom_NCHN",
        "NCVI_concurrent",
        "Initial_MSEstar_max_NCHN",
        "TCC_resid",
        "SM_resid",
    ],
}
SLIM_HEATMAP_FACTOR_ORDER = [
    "sm_avg",
    "SHF_resid",
    "NCHN_tp",
    "SM_resid",
    "z500_anom_NCHN",
    "Initial_MSEstar_max_NCHN",
    "TCC_resid",
    "NCVI_concurrent",
]


def _slim_resid_series(df, y_col, control_cols):
    missing = [c for c in [y_col] + control_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Slim MLR residualization missing columns: {missing}")
    X = sm.add_constant(df[control_cols].astype(float), has_constant="add")
    model = sm.OLS(df[y_col].astype(float), X).fit()
    fitted = model.predict(X)
    return df[y_col].astype(float) - fitted


def _slim_safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(pearsonr(a, b)[0])


def _slim_compute_lmg_mc(X, y, feature_names, mc_samples=2000):
    np.random.seed(42)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape
    if n_samples < n_features + 2:
        return {name: np.nan for name in feature_names}
    X_s = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    y_s = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)
    lmg_raw = np.zeros(n_features)
    model = LinearRegression()
    for _ in range(mc_samples):
        order = np.random.permutation(n_features)
        prev_r2 = 0.0
        for i in range(n_features):
            idx = order[: i + 1]
            r2 = model.fit(X_s[:, idx], y_s).score(X_s[:, idx], y_s)
            lmg_raw[order[i]] += max(0, r2 - prev_r2)
            prev_r2 = r2
    tot = lmg_raw.sum()
    if tot == 0:
        return {name: 0.0 for name in feature_names}
    return {name: val for name, val in zip(feature_names, (lmg_raw / tot) * 100.0)}


def _slim_fit_ols_from_standardized(X_scaled, y, predictor_names, context):
    X_const = sm.add_constant(X_scaled, has_constant="add")
    rank = np.linalg.matrix_rank(X_const)
    expected_rank = len(predictor_names) + 1
    if rank < expected_rank:
        raise RuntimeError(f"Slim MLR rank deficient for {context}: rank={rank}, expected={expected_rank}")
    model = sm.OLS(y, X_const).fit()
    fitted = np.asarray(model.predict(X_const), dtype=float).squeeze()
    rmse = float(np.sqrt(np.mean((np.asarray(y, dtype=float) - fitted) ** 2)))
    return model, fitted, rmse


def _slim_vif_rows(df_model, predictors, init, stage, condition_number):
    X0 = df_model[predictors].astype(float).values
    Xz = (X0 - np.nanmean(X0, axis=0)) / (np.nanstd(X0, axis=0) + 1e-8)
    X_const = sm.add_constant(Xz, has_constant="add")
    rows = []
    for idx, factor in enumerate(predictors):
        rows.append({
            "init_date": init,
            "stage": stage,
            "predictor": factor,
            "display_label": SLIM_DISPLAY_LABELS.get(factor, factor),
            "VIF": float(variance_inflation_factor(X_const, idx + 1)),
            "condition_number": float(condition_number),
        })
    return rows


def _slim_predictor_role(factor):
    if factor in {"SHF_resid", "SM_resid", "TCC_resid"}:
        return "residualized predictor"
    if factor == "NCVI_concurrent":
        return "stage-concurrent circulation diagnostic"
    if factor == "Initial_MSEstar_max_NCHN":
        return "antecedent thermodynamic preconditioning"
    if factor in {"sm_avg", "NCHN_tp"}:
        return "hydrothermal stage predictor"
    return "circulation predictor"


def _slim_residualization_base(stage, factor):
    if factor in {"SHF_resid", "TCC_resid"} and stage == "Stage-I_dry":
        return "SM"
    if factor in {"SM_resid", "TCC_resid"} and stage in {"Stage-II_wet", "Total"}:
        return "TP"
    return "none"


def _slim_prepare_stage_group(df_stage, df_ncvi, init, stage):
    g = df_stage[(df_stage["init_date"].astype(str) == init) & (df_stage["stage"] == stage)].copy()
    if len(g) != 51 or g["member"].nunique() != 51:
        raise RuntimeError(f"Slim MLR expected 51 raw members for {init}/{stage}, got rows={len(g)}, unique_members={g['member'].nunique() if 'member' in g.columns else 'NA'}")
    nc = df_ncvi[(df_ncvi["init_date"].astype(str) == init) & (df_ncvi["stage"] == stage)][["init_date", "stage", "member", "NCVI_concurrent"]].copy()
    if len(nc) != 51 or nc["member"].nunique() != 51:
        raise RuntimeError(f"Slim MLR expected 51 NCVI_concurrent members for {init}/{stage}, got rows={len(nc)}, unique_members={nc['member'].nunique() if 'member' in nc.columns else 'NA'}")
    if "NCVI_concurrent" in g.columns:
        g = g.drop(columns=["NCVI_concurrent"])
    g = g.merge(nc, on=["init_date", "stage", "member"], how="left", validate="one_to_one")
    if g["NCVI_concurrent"].isna().any():
        missing_members = g.loc[g["NCVI_concurrent"].isna(), "member"].tolist()
        raise RuntimeError(f"Slim MLR NCVI_concurrent merge produced missing values for {init}/{stage}; members={missing_members}")
    if stage == "Stage-I_dry":
        g["SHF_resid"] = _slim_resid_series(g, "SHF_Avg", ["sm_avg"])
        g["TCC_resid"] = _slim_resid_series(g, "TCC_Avg", ["sm_avg"])
    elif stage in {"Stage-II_wet", "Total"}:
        g["SM_resid"] = _slim_resid_series(g, "sm_avg", ["NCHN_tp"])
        g["TCC_resid"] = _slim_resid_series(g, "TCC_Avg", ["NCHN_tp"])
        g["SHF_resid_QC"] = _slim_resid_series(g, "SHF_Avg", ["NCHN_tp"])
    else:
        raise RuntimeError(f"Slim MLR unsupported stage: {stage}")
    predictors = SLIM_PREDICTORS_BY_STAGE[stage]
    required_cols = ["member", "y_tmax"] + predictors
    missing = [c for c in required_cols if c not in g.columns]
    if missing:
        raise RuntimeError(f"Slim MLR missing required columns for {init}/{stage}: {missing}")
    g_model = g.dropna(subset=required_cols).sort_values("member").reset_index(drop=True)
    if len(g_model) != 51 or g_model["member"].nunique() != 51:
        nan_counts = g[required_cols].isna().sum().to_dict()
        raise RuntimeError(f"Slim MLR modeling sample must be 51 for {init}/{stage}; got rows={len(g_model)}, unique_members={g_model['member'].nunique()}; nan_counts={nan_counts}")
    return g_model


def _slim_make_lmg_combo(init, stage, g_model, predictors, model, fitted, lmg_df):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax_scatter, ax_bar = axes
    y = g_model["y_tmax"].astype(float).values
    ax_scatter.scatter(y, fitted, s=32, alpha=0.82, color="#2f6db3", edgecolor="white", linewidth=0.4)
    lim_min = float(min(np.nanmin(y), np.nanmin(fitted)) - 0.3)
    lim_max = float(max(np.nanmax(y), np.nanmax(fitted)) + 0.3)
    ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max], ls="--", color="0.35", lw=1.0)
    ax_scatter.set_xlim(lim_min, lim_max)
    ax_scatter.set_ylim(lim_min, lim_max)
    ax_scatter.set_xlabel("ECMWF member Tmax (°C)")
    ax_scatter.set_ylabel("Slim OLS fitted Tmax (°C)")
    rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
    ax_scatter.set_title(f"R²={model.rsquared:.2f}, Adj.R²={model.rsquared_adj:.2f}, RMSE={rmse:.2f}")

    bars_df = lmg_df.sort_values("LMG_percent", ascending=True)
    vals = bars_df["LMG_percent"].astype(float).values
    labels = bars_df["display_label"].tolist()
    ax_bar.barh(labels, vals, color="#6AA84F", alpha=0.85)
    ax_bar.set_xlabel("LMG relative importance (%)")
    ax_bar.set_title("Slim LMG importance")
    xmax = max(float(np.nanmax(vals)) if len(vals) else 1.0, 1.0)
    ax_bar.set_xlim(0, xmax * 1.18)
    for y_idx, val in enumerate(vals):
        ax_bar.text(val + xmax * 0.025, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.5)
    fig.suptitle(f"Slim MLR | NCVI conc. | {init} | {stage}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png = f"{SLIM_MLR_SUBDIRS['lmg_combo']}/fig_slim_lmg_combo_NCVIconcurrent_{init}_{stage}.png"
    pdf = f"{SLIM_MLR_SUBDIRS['lmg_combo']}/fig_slim_lmg_combo_NCVIconcurrent_{init}_{stage}.pdf"
    for out_path in [png, pdf]:
        assert_slim_mlr_output_path(out_path)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)


def _slim_make_corr_matrix(init, stage, g_model, predictors):
    cols = ["y_tmax"] + predictors
    labels = ["Tmax"] + [SLIM_DISPLAY_LABELS.get(c, c) for c in predictors]
    corr_df = g_model[cols].astype(float).corr()
    fig, ax = plt.subplots(figsize=(8.6, 7.8))
    sns.heatmap(corr_df, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, linewidths=0.4, linecolor="white", cbar_kws={"label": "Pearson r"}, annot_kws={"fontsize": 8})
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, rotation=0, fontsize=8)
    ax.set_title(f"Slim target-predictor correlations | NCVI conc. | {init} {stage}")
    fig.subplots_adjust(left=0.20, bottom=0.22, right=0.96, top=0.90)
    png = f"{SLIM_MLR_SUBDIRS['corr_matrix']}/corr_slim_target_predictors_NCVIconcurrent_{init}_{stage}.png"
    pdf = f"{SLIM_MLR_SUBDIRS['corr_matrix']}/corr_slim_target_predictors_NCVIconcurrent_{init}_{stage}.pdf"
    for out_path in [png, pdf]:
        assert_slim_mlr_output_path(out_path)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)


def _slim_plot_heatmap(df_plot, value_col, title, cbar_label, cmap, center, output_stem, fmt=".2f"):
    row_order = [SLIM_ROW_LABELS[(init, stage)] for init in SLIM_INIT_ORDER for stage in SLIM_STAGE_ORDER]
    col_order = [SLIM_DISPLAY_LABELS[f] for f in SLIM_HEATMAP_FACTOR_ORDER]
    plot_df = df_plot.copy()
    plot_df["row_label"] = plot_df.apply(lambda r: SLIM_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
    mat = plot_df.pivot(index="row_label", columns="display_label", values=value_col).reindex(index=row_order, columns=col_order)
    fig_h = max(5.5, 0.55 * len(row_order) + 2.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    sns.heatmap(mat, ax=ax, cmap=cmap, center=center, annot=True, fmt=fmt, linewidths=0.4, linecolor="white", cbar_kws={"label": cbar_label}, annot_kws={"fontsize": 9})
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    png = f"{SLIM_MLR_SUBDIRS['heatmap']}/{output_stem}.png"
    pdf = f"{SLIM_MLR_SUBDIRS['heatmap']}/{output_stem}.pdf"
    for out_path in [png, pdf]:
        assert_slim_mlr_output_path(out_path)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)


slim_qc_lines = [
    "Final slim MLR with NCVI_concurrent QC summary",
    f"Input Cell-5 TCC stage table: {SLIM_STAGE_TCC_TABLE}; exists={os.path.exists(SLIM_STAGE_TCC_TABLE)}",
    f"Input Cell-7 NCVI_concurrent stage values: {SLIM_NCVI_STAGE_TABLE}; exists={os.path.exists(SLIM_NCVI_STAGE_TABLE)}",
    f"Input Cell-7 full model comparison: {SLIM_FULL_MODEL_CMP_TABLE}; exists={os.path.exists(SLIM_FULL_MODEL_CMP_TABLE)}",
    "This slim MLR uses NCVI_concurrent as a stage-concurrent circulation/cold-vortex anomaly diagnostic, not as an initial-condition predictor.",
    "Init MSE* is retained as an antecedent thermodynamic preconditioning factor.",
    "Tmax/y_tmax appears in correlation matrices for diagnostics only and is not included in the slim predictor list.",
    "",
]

df_slim_stage = pd.read_csv(SLIM_STAGE_TCC_TABLE)
df_slim_ncvi = pd.read_csv(SLIM_NCVI_STAGE_TABLE)
df_slim_full_cmp = pd.read_csv(SLIM_FULL_MODEL_CMP_TABLE)
df_slim_daily_member_raw = pd.read_csv(SLIM_DAILY_MEMBER_RAW_TABLE)
df_slim_daily_member_raw["date"] = pd.to_datetime(df_slim_daily_member_raw["date"])
df_slim_daily_tmax = pd.read_csv(SLIM_DAILY_TMAX_TABLE)
df_slim_daily_tmax["date"] = pd.to_datetime(df_slim_daily_tmax["date"])
df_slim_daily_tcc = pd.read_csv(SLIM_DAILY_TCC_TABLE)
df_slim_daily_tcc["date"] = pd.to_datetime(df_slim_daily_tcc["date"])
for _df in [df_slim_stage, df_slim_ncvi, df_slim_full_cmp, df_slim_daily_member_raw, df_slim_daily_tmax, df_slim_daily_tcc]:
    if "init_date" in _df.columns:
        _df["init_date"] = _df["init_date"].astype(str)

slim_predictor_rows = []
for stage, predictors in SLIM_PREDICTORS_BY_STAGE.items():
    for predictor in predictors:
        slim_predictor_rows.append({
            "stage": stage,
            "predictors_raw_name": predictor,
            "predictors_display_label": SLIM_DISPLAY_LABELS.get(predictor, predictor),
            "role/category": _slim_predictor_role(predictor),
            "residualization_base": _slim_residualization_base(stage, predictor),
        })

table_slim_predictor_sets = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_predictor_sets.csv"
assert_slim_mlr_output_path(table_slim_predictor_sets)
pd.DataFrame(slim_predictor_rows).to_csv(table_slim_predictor_sets, index=False)

slim_summary_rows = []
slim_coef_rows = []
slim_lmg_rows = []
slim_lofo_rows = []
slim_vif_rows = []
slim_fitted_rows = []
slim_stage_artifacts = {}

for init in SLIM_INIT_ORDER:
    for stage in SLIM_STAGE_ORDER:
        g_model = _slim_prepare_stage_group(df_slim_stage, df_slim_ncvi, init, stage)
        predictors = SLIM_PREDICTORS_BY_STAGE[stage]
        y = g_model["y_tmax"].astype(float).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(g_model[predictors].astype(float))
        model, fitted, rmse = _slim_fit_ols_from_standardized(X_scaled, y, predictors, f"{init}/{stage}")
        condition_number = float(np.linalg.cond(sm.add_constant(X_scaled, has_constant="add")))
        lmg_dict = _slim_compute_lmg_mc(g_model[predictors].astype(float).values, y, predictors, mc_samples=2000)
        lmg_sorted = sorted(lmg_dict.items(), key=lambda kv: kv[1], reverse=True)
        lmg_rank = {factor: rank + 1 for rank, (factor, _) in enumerate(lmg_sorted)}
        full_rmse = rmse

        slim_summary_rows.append({
            "init_date": init,
            "stage": stage,
            "n": int(len(g_model)),
            "R2": float(model.rsquared),
            "Adj_R2": float(model.rsquared_adj),
            "RMSE": full_rmse,
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "predictor_count": len(predictors),
            "predictor_list": ";".join(predictors),
        })
        slim_qc_lines.extend([
            f"[{init} | {stage}] n={len(g_model)}; predictors={';'.join(predictors)}",
            f"[{init} | {stage}] residualization bases: " + "; ".join(f"{p}={_slim_residualization_base(stage, p)}" for p in predictors if _slim_residualization_base(stage, p) != "none"),
            f"[{init} | {stage}] R2={model.rsquared:.6f}, Adj_R2={model.rsquared_adj:.6f}, RMSE={full_rmse:.6f}",
        ])

        params = np.asarray(model.params, dtype=float)
        pvalues = np.asarray(model.pvalues, dtype=float)
        tvalues = np.asarray(model.tvalues, dtype=float)
        raw_intercept = float(params[0] - np.sum(params[1:] * scaler.mean_ / (scaler.scale_ + 1e-12)))
        for idx, predictor in enumerate(predictors, start=1):
            slim_coef_rows.append({
                "init_date": init,
                "stage": stage,
                "predictor": predictor,
                "display_label": SLIM_DISPLAY_LABELS.get(predictor, predictor),
                "coef": float(params[idx] / (scaler.scale_[idx - 1] + 1e-12)),
                "standardized_coef": float(params[idx]),
                "p_value": float(pvalues[idx]),
                "t_value": float(tvalues[idx]),
            })
        slim_coef_rows.append({
            "init_date": init,
            "stage": stage,
            "predictor": "const",
            "display_label": "Intercept",
            "coef": raw_intercept,
            "standardized_coef": np.nan,
            "p_value": float(pvalues[0]),
            "t_value": float(tvalues[0]),
        })

        for predictor in predictors:
            slim_lmg_rows.append({
                "init_date": init,
                "stage": stage,
                "predictor": predictor,
                "display_label": SLIM_DISPLAY_LABELS.get(predictor, predictor),
                "LMG": float(lmg_dict[predictor]),
                "LMG_percent": float(lmg_dict[predictor]),
                "LMG_rank": int(lmg_rank[predictor]),
            })

        lofo_tmp = []
        for predictor in predictors:
            reduced_predictors = [p for p in predictors if p != predictor]
            reduced_idx = [predictors.index(p) for p in reduced_predictors]
            reduced_model, reduced_fitted, reduced_rmse = _slim_fit_ols_from_standardized(
                X_scaled[:, reduced_idx], y, reduced_predictors, f"{init}/{stage}/without_{predictor}"
            )
            lofo_tmp.append({
                "init_date": init,
                "stage": stage,
                "predictor": predictor,
                "display_label": SLIM_DISPLAY_LABELS.get(predictor, predictor),
                "Delta_R2": float(model.rsquared - reduced_model.rsquared),
                "Delta_Adj_R2": float(model.rsquared_adj - reduced_model.rsquared_adj),
                "Delta_RMSE": float(reduced_rmse - full_rmse),
                "R2_without_factor": float(reduced_model.rsquared),
                "Adj_R2_without_factor": float(reduced_model.rsquared_adj),
            })
        positive_sum = sum(max(row["Delta_R2"], 0.0) for row in lofo_tmp)
        lofo_rank_r2 = {row["predictor"]: rank + 1 for rank, row in enumerate(sorted(lofo_tmp, key=lambda r: r["Delta_R2"], reverse=True))}
        lofo_rank_adj = {row["predictor"]: rank + 1 for rank, row in enumerate(sorted(lofo_tmp, key=lambda r: r["Delta_Adj_R2"], reverse=True))}
        for row in lofo_tmp:
            row["LOFO_rank_R2"] = int(lofo_rank_r2[row["predictor"]])
            row["LOFO_rank_Adj_R2"] = int(lofo_rank_adj[row["predictor"]])
            row["positive_delta_R2_share_percent"] = float(max(row["Delta_R2"], 0.0) / positive_sum * 100.0) if positive_sum > 0 else 0.0
            slim_lofo_rows.append(row)

        slim_vif_rows.extend(_slim_vif_rows(g_model, predictors, init, stage, condition_number))
        max_vif = max(row["VIF"] for row in slim_vif_rows if row["init_date"] == init and row["stage"] == stage)
        slim_qc_lines.append(f"[{init} | {stage}] max VIF={max_vif:.6f}; condition_number={condition_number:.6f}")

        for member, yy, ff in zip(g_model["member"].values, y, fitted):
            slim_fitted_rows.append({
                "init_date": init,
                "stage": stage,
                "member": member,
                "y_tmax": float(yy),
                "y_fitted": float(ff),
                "residual": float(yy - ff),
            })

        lmg_df_stage = pd.DataFrame([r for r in slim_lmg_rows if r["init_date"] == init and r["stage"] == stage])
        _slim_make_lmg_combo(init, stage, g_model, predictors, model, fitted, lmg_df_stage)
        _slim_make_corr_matrix(init, stage, g_model, predictors)
        slim_stage_artifacts[(init, stage)] = {
            "g_model": g_model,
            "predictors": predictors,
            "model": model,
            "fitted": fitted,
            "rmse": full_rmse,
            "scaler": scaler,
        }

# Write model tables.
table_slim_model_summary = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_model_summary.csv"
table_slim_model_coefficients = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_model_coefficients.csv"
table_slim_lmg_long = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_lmg_long.csv"
table_slim_lofo_long = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_lofo_long.csv"
table_slim_vif = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_vif.csv"
table_slim_fitted_members = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_fitted_members.csv"
for _out_csv in [table_slim_model_summary, table_slim_model_coefficients, table_slim_lmg_long, table_slim_lofo_long, table_slim_vif, table_slim_fitted_members]:
    assert_slim_mlr_output_path(_out_csv)

df_slim_summary = pd.DataFrame(slim_summary_rows)
df_slim_coef = pd.DataFrame(slim_coef_rows)
df_slim_lmg = pd.DataFrame(slim_lmg_rows)
df_slim_lofo = pd.DataFrame(slim_lofo_rows)
df_slim_vif = pd.DataFrame(slim_vif_rows)
df_slim_fitted = pd.DataFrame(slim_fitted_rows)

df_slim_summary.to_csv(table_slim_model_summary, index=False)
df_slim_coef.to_csv(table_slim_model_coefficients, index=False)
df_slim_lmg.to_csv(table_slim_lmg_long, index=False)
df_slim_lofo.to_csv(table_slim_lofo_long, index=False)
df_slim_vif.to_csv(table_slim_vif, index=False)
df_slim_fitted.to_csv(table_slim_fitted_members, index=False)

# Slim versus Cell-7 full NCVI_concurrent model comparison.
full_subset = df_slim_full_cmp[df_slim_full_cmp["model_version"] == "noSSR_TCCresid_NCVIconcurrent"].copy()
if full_subset.empty:
    raise RuntimeError("Slim MLR cannot find Cell-7 full model rows for model_version=noSSR_TCCresid_NCVIconcurrent")
slim_vs_full_rows = []
for _, slim_row in df_slim_summary.iterrows():
    init = slim_row["init_date"]
    stage = slim_row["stage"]
    full_rows = full_subset[(full_subset["init_date"].astype(str) == init) & (full_subset["stage"] == stage)]
    if len(full_rows) != 1:
        raise RuntimeError(f"Slim MLR expected one full-model comparison row for {init}/{stage}, got {len(full_rows)}")
    full_row = full_rows.iloc[0]
    full_r2 = float(full_row["R2_full"])
    full_adj = float(full_row["AdjR2_full"])
    full_rmse = float(full_row["RMSE_full"])
    slim_vs_full_rows.append({
        "init_date": init,
        "stage": stage,
        "full_R2": full_r2,
        "slim_R2": float(slim_row["R2"]),
        "full_Adj_R2": full_adj,
        "slim_Adj_R2": float(slim_row["Adj_R2"]),
        "full_RMSE": full_rmse,
        "slim_RMSE": float(slim_row["RMSE"]),
        "Delta_R2": float(slim_row["R2"] - full_r2),
        "Delta_Adj_R2": float(slim_row["Adj_R2"] - full_adj),
        "Delta_RMSE": float(slim_row["RMSE"] - full_rmse),
    })
    slim_qc_lines.append(
        f"[{init} | {stage}] full-vs-slim Delta_R2={slim_row['R2'] - full_r2:.6f}, "
        f"Delta_Adj_R2={slim_row['Adj_R2'] - full_adj:.6f}, Delta_RMSE={slim_row['RMSE'] - full_rmse:.6f}"
    )

table_slim_vs_full = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_vs_full_model_metrics.csv"
assert_slim_mlr_output_path(table_slim_vs_full)
df_slim_vs_full = pd.DataFrame(slim_vs_full_rows)
df_slim_vs_full.to_csv(table_slim_vs_full, index=False)

# Publication-style init-level slim LMG combo figures.
SLIM_STAGE_SHORT_LABELS = {
    "Stage-I_dry": "Stage-I",
    "Stage-II_wet": "Stage-II",
    "Total": "Total",
}
slim_init_lmg_combo_files = []
for init in SLIM_INIT_ORDER:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), gridspec_kw={"width_ratios": [1.0, 1.18]}, constrained_layout=False)
    for row_idx, stage in enumerate(SLIM_STAGE_ORDER):
        artifact = slim_stage_artifacts[(init, stage)]
        g_model = artifact["g_model"]
        fitted = artifact["fitted"]
        model = artifact["model"]
        rmse = artifact["rmse"]
        y = g_model["y_tmax"].astype(float).values
        ax_scatter, ax_bar = axes[row_idx, 0], axes[row_idx, 1]
        ax_scatter.scatter(y, fitted, s=30, alpha=0.75, color="#2f6db3", edgecolor="white", linewidth=0.4)
        lim_min = float(min(np.nanmin(y), np.nanmin(fitted)))
        lim_max = float(max(np.nanmax(y), np.nanmax(fitted)))
        pad = max((lim_max - lim_min) * 0.05, 0.25)
        lim_min -= pad
        lim_max += pad
        ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max], ls="--", color="0.35", lw=1.0)
        ax_scatter.set_xlim(lim_min, lim_max)
        ax_scatter.set_ylim(lim_min, lim_max)
        ax_scatter.set_xlabel("ECMWF member Tmax (°C)")
        ax_scatter.set_ylabel("Slim OLS fitted Tmax (°C)")
        ax_scatter.set_title(f"{SLIM_STAGE_SHORT_LABELS[stage]}: R²={model.rsquared:.2f}, Adj.R²={model.rsquared_adj:.2f}, RMSE={rmse:.2f}")

        bars_df = df_slim_lmg[(df_slim_lmg["init_date"] == init) & (df_slim_lmg["stage"] == stage)].sort_values("LMG_percent", ascending=False)
        expected_labels = [SLIM_DISPLAY_LABELS.get(p, p) for p in SLIM_PREDICTORS_BY_STAGE[stage]]
        if set(bars_df["display_label"].tolist()) != set(expected_labels):
            raise RuntimeError(
                f"Slim init-level LMG combo label check failed for {init}/{stage}: "
                f"got={bars_df['display_label'].tolist()}, expected={expected_labels}"
            )
        plot_df = bars_df.sort_values("LMG_percent", ascending=True)
        vals = plot_df["LMG_percent"].astype(float).values
        labels = plot_df["display_label"].tolist()
        ax_bar.barh(labels, vals, color="#6AA84F", alpha=0.86)
        ax_bar.set_xlabel("LMG relative importance (%)")
        ax_bar.set_title(f"{SLIM_STAGE_SHORT_LABELS[stage]} LMG")
        xmax = max(float(np.nanmax(vals)) if len(vals) else 1.0, 1.0)
        ax_bar.set_xlim(0, xmax * 1.15)
        for y_idx, val in enumerate(vals):
            ax_bar.text(val + xmax * 0.025, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.5)
    fig.suptitle(f"Slim MLR LMG combo | NCVI conc. | Init {init}", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.07, wspace=0.36, hspace=0.55)
    png = f"{SLIM_MLR_SUBDIRS['lmg_combo_init']}/fig_slim_lmg_combo_NCVIconcurrent_init_{init}.png"
    pdf = f"{SLIM_MLR_SUBDIRS['lmg_combo_init']}/fig_slim_lmg_combo_NCVIconcurrent_init_{init}.pdf"
    for out_path in [png, pdf]:
        assert_slim_mlr_output_path(out_path)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    slim_init_lmg_combo_files.extend([png, pdf])
    print(f"[Cell 8 slim MLR] init-level LMG combo figures saved: {png}; {pdf}")

# Per-init slim LOFO combo figures.
for init in SLIM_INIT_ORDER:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11.5), gridspec_kw={"width_ratios": [1.0, 1.15]}, constrained_layout=True)
    for row_idx, stage in enumerate(SLIM_STAGE_ORDER):
        artifact = slim_stage_artifacts[(init, stage)]
        g_model = artifact["g_model"]
        fitted = artifact["fitted"]
        model = artifact["model"]
        rmse = artifact["rmse"]
        y = g_model["y_tmax"].astype(float).values
        ax_scatter, ax_bar = axes[row_idx, 0], axes[row_idx, 1]
        ax_scatter.scatter(y, fitted, s=28, alpha=0.82, color="#2f6db3", edgecolor="white", linewidth=0.4)
        lim_min = float(min(np.nanmin(y), np.nanmin(fitted)) - 0.3)
        lim_max = float(max(np.nanmax(y), np.nanmax(fitted)) + 0.3)
        ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max], ls="--", color="0.35", lw=1.0)
        ax_scatter.set_xlim(lim_min, lim_max)
        ax_scatter.set_ylim(lim_min, lim_max)
        ax_scatter.set_xlabel("ECMWF member Tmax (°C)")
        ax_scatter.set_ylabel("Slim OLS fitted Tmax (°C)")
        ax_scatter.set_title(f"{stage}: R²={model.rsquared:.2f}, Adj.R²={model.rsquared_adj:.2f}, RMSE={rmse:.2f}")

        bars_df = df_slim_lofo[(df_slim_lofo["init_date"] == init) & (df_slim_lofo["stage"] == stage)].sort_values("Delta_R2", ascending=True)
        vals = bars_df["Delta_R2"].astype(float).values
        labels = bars_df["display_label"].tolist()
        colors = ["#6AA84F" if v >= 0 else "#C0504D" for v in vals]
        ax_bar.barh(labels, vals, color=colors, alpha=0.85)
        ax_bar.axvline(0, color="0.35", lw=0.8)
        ax_bar.set_title("Slim LOFO Delta R²")
        ax_bar.set_xlabel("Delta R²")
        span = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals))), 0.01)
        ax_bar.set_xlim(float(np.nanmin(vals)) - 0.18 * span, float(np.nanmax(vals)) + 0.35 * span)
        for y_idx, (_, bar_row) in enumerate(bars_df.reset_index(drop=True).iterrows()):
            val = float(bar_row["Delta_R2"])
            pct = float(bar_row["positive_delta_R2_share_percent"])
            ha = "left" if val >= 0 else "right"
            dx = 0.015 * span if val >= 0 else -0.015 * span
            ax_bar.text(val + dx, y_idx, f"{val:.3f} ({pct:.1f}%)", va="center", ha=ha, fontsize=8.5)
    fig.suptitle(f"Slim LOFO combo | NCVI conc. | Init {init}", fontsize=15)
    png = f"{SLIM_MLR_SUBDIRS['lofo_combo']}/fig_slim_lofo_combo_NCVIconcurrent_init_{init}.png"
    pdf = f"{SLIM_MLR_SUBDIRS['lofo_combo']}/fig_slim_lofo_combo_NCVIconcurrent_init_{init}.pdf"
    for out_path in [png, pdf]:
        assert_slim_mlr_output_path(out_path)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

# Heatmaps.
_slim_plot_heatmap(
    df_slim_lmg,
    "LMG_percent",
    "Slim MLR LMG importance | NCVI conc.",
    "LMG importance (%)",
    "YlOrRd",
    None,
    "heatmap_slim_lmg_importance_NCVIconcurrent",
    fmt=".1f",
)
_slim_plot_heatmap(
    df_slim_lofo,
    "Delta_R2",
    "Slim MLR LOFO Delta R² | NCVI conc.",
    "LOFO Delta R²",
    "coolwarm",
    0,
    "heatmap_slim_lofo_deltaR2_NCVIconcurrent",
    fmt=".3f",
)
metrics_plot = df_slim_vs_full.copy()
metrics_plot["row_label"] = metrics_plot.apply(lambda r: SLIM_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
metrics_mat = metrics_plot.pivot(index="row_label", columns="stage", values="Delta_R2")
metric_long = metrics_plot.melt(id_vars=["row_label"], value_vars=["Delta_R2", "Delta_Adj_R2", "Delta_RMSE"], var_name="metric", value_name="value")
metric_mat = metric_long.pivot(index="row_label", columns="metric", values="value").reindex(
    index=[SLIM_ROW_LABELS[(init, stage)] for init in SLIM_INIT_ORDER for stage in SLIM_STAGE_ORDER],
    columns=["Delta_R2", "Delta_Adj_R2", "Delta_RMSE"],
)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
sns.heatmap(metric_mat, ax=ax, cmap="coolwarm", center=0, annot=True, fmt=".3f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Slim - full metric delta"}, annot_kws={"fontsize": 9})
ax.set_title("Slim vs full NCVI_concurrent model metrics")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
heatmap_slim_vs_full_png = f"{SLIM_MLR_SUBDIRS['heatmap']}/heatmap_slim_vs_full_metrics_NCVIconcurrent.png"
heatmap_slim_vs_full_pdf = f"{SLIM_MLR_SUBDIRS['heatmap']}/heatmap_slim_vs_full_metrics_NCVIconcurrent.pdf"
for _out_path in [heatmap_slim_vs_full_png, heatmap_slim_vs_full_pdf]:
    assert_slim_mlr_output_path(_out_path)
fig.savefig(heatmap_slim_vs_full_png, dpi=240)
fig.savefig(heatmap_slim_vs_full_pdf)
plt.close(fig)


# Figure-only daily stage-trained OLS projection time-series for the final slim MLR.
SLIM_DAILY_STAGE_WINDOWS = {
    "Stage-I_dry": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17")),
    "Stage-II_wet": (pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24")),
}


def _slim_stage_day_count(stage):
    start, end = SLIM_DAILY_STAGE_WINDOWS[stage]
    return int((end - start).days + 1)


def _slim_daily_fit_for_stage(init, stage):
    if stage not in ["Stage-I_dry", "Stage-II_wet"]:
        raise RuntimeError(f"Slim daily time-series only supports Stage-I/Stage-II, got {stage}")
    artifact = slim_stage_artifacts[(init, stage)]
    predictors = artifact["predictors"]
    expected_predictors = SLIM_PREDICTORS_BY_STAGE[stage]
    if predictors != expected_predictors:
        raise RuntimeError(f"Slim daily predictor mismatch for {init}/{stage}: predictors={predictors}, expected={expected_predictors}")
    g_stage = artifact["g_model"].sort_values("member").reset_index(drop=True)
    stage_model = artifact["model"]
    scaler = artifact["scaler"]

    if stage == "Stage-I_dry":
        shf_model = sm.OLS(g_stage["SHF_Avg"].astype(float), sm.add_constant(g_stage[["sm_avg"]].astype(float), has_constant="add")).fit()
        tcc_model = sm.OLS(g_stage["TCC_Avg"].astype(float), sm.add_constant(g_stage[["sm_avg"]].astype(float), has_constant="add")).fit()
        shf_ctrl = "sm_avg"
        tcc_ctrl = "sm_avg"
        sm_model = None
    else:
        sm_model = sm.OLS(g_stage["sm_avg"].astype(float), sm.add_constant(g_stage[["NCHN_tp"]].astype(float), has_constant="add")).fit()
        tcc_model = sm.OLS(g_stage["TCC_Avg"].astype(float), sm.add_constant(g_stage[["NCHN_tp"]].astype(float), has_constant="add")).fit()
        shf_model = None
        shf_ctrl = None
        tcc_ctrl = "NCHN_tp"

    win_start, win_end = SLIM_DAILY_STAGE_WINDOWS[stage]
    day_count = _slim_stage_day_count(stage)
    dg = df_slim_daily_member_raw[
        (df_slim_daily_member_raw["init_date"] == init)
        & (df_slim_daily_member_raw["date"] >= win_start)
        & (df_slim_daily_member_raw["date"] <= win_end)
    ].copy()
    if dg.empty:
        raise RuntimeError(f"Slim daily projection base rows are empty for {init}/{stage}")
    dg = dg.merge(
        df_slim_daily_tcc[df_slim_daily_tcc["init_date"] == init][["date", "member", "TCC_Avg"]],
        on=["date", "member"],
        how="left",
        suffixes=("", "_tcc_daily"),
    )
    if "TCC_Avg_tcc_daily" in dg.columns:
        dg["TCC_Avg"] = dg["TCC_Avg_tcc_daily"]
        dg = dg.drop(columns=["TCC_Avg_tcc_daily"])
    dg = dg.merge(g_stage[["member", "NCVI_concurrent"]], on="member", how="left", validate="many_to_one")
    tmax_daily = df_slim_daily_tmax[(df_slim_daily_tmax["init_date"] == init)][["date", "member", "s2s_tmax", "era5_tmax"]].copy()
    dg = dg.merge(tmax_daily, on=["date", "member"], how="left", validate="one_to_one")

    daily_required = ["date", "member", "NCHN_tp", "sm_avg", "SHF_Avg", "TCC_Avg", "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "NCVI_concurrent", "s2s_tmax", "era5_tmax"]
    missing_daily = [c for c in daily_required if c not in dg.columns]
    if missing_daily:
        raise RuntimeError(f"Slim daily projection missing columns for {init}/{stage}: {missing_daily}")
    dg = dg.dropna(subset=daily_required).copy()
    day_member_counts = dg.groupby("date")["member"].nunique()
    if len(day_member_counts) != day_count or not (day_member_counts == 51).all():
        raise RuntimeError(f"Slim daily projection member-count check failed for {init}/{stage}: {day_member_counts.to_dict()}")

    if stage == "Stage-II_wet":
        dg["NCHN_tp"] = dg["NCHN_tp"].astype(float) * day_count
        pred_sm = np.asarray(sm_model.predict(sm.add_constant(dg[["NCHN_tp"]].astype(float), has_constant="add")), dtype=float).squeeze()
        if pred_sm.shape[0] != len(dg):
            raise RuntimeError(f"Slim daily SM residual shape mismatch for {init}/{stage}: expected={len(dg)}, got={pred_sm.shape}")
        dg["SM_resid"] = dg["sm_avg"].astype(float).values - pred_sm
    else:
        pred_shf = np.asarray(shf_model.predict(sm.add_constant(dg[[shf_ctrl]].astype(float), has_constant="add")), dtype=float).squeeze()
        if pred_shf.shape[0] != len(dg):
            raise RuntimeError(f"Slim daily SHF residual shape mismatch for {init}/{stage}: expected={len(dg)}, got={pred_shf.shape}")
        dg["SHF_resid"] = dg["SHF_Avg"].astype(float).values - pred_shf

    pred_tcc = np.asarray(tcc_model.predict(sm.add_constant(dg[[tcc_ctrl]].astype(float), has_constant="add")), dtype=float).squeeze()
    if pred_tcc.shape[0] != len(dg):
        raise RuntimeError(f"Slim daily TCC residual shape mismatch for {init}/{stage}: expected={len(dg)}, got={pred_tcc.shape}")
    dg["TCC_resid"] = dg["TCC_Avg"].astype(float).values - pred_tcc

    missing_predictors = [c for c in predictors if c not in dg.columns]
    if missing_predictors:
        raise RuntimeError(f"Slim daily projection predictors missing for {init}/{stage}: {missing_predictors}")
    if predictors != expected_predictors:
        raise RuntimeError(f"Slim daily predictor-set check failed for {init}/{stage}: {predictors} != {expected_predictors}")
    X_daily = scaler.transform(dg[predictors].astype(float))
    pred_daily = stage_model.predict(sm.add_constant(X_daily, has_constant="add"))
    pred_daily = np.asarray(pred_daily, dtype=float).squeeze()
    if pred_daily.shape[0] != len(dg):
        raise RuntimeError(f"Slim daily fitted shape mismatch for {init}/{stage}: expected={len(dg)}, got={pred_daily.shape}")
    dg["Slim_OLS_daily_fitted"] = pred_daily
    return dg[["init_date", "date", "member", "s2s_tmax", "era5_tmax", "Slim_OLS_daily_fitted", *predictors]].rename(columns={"s2s_tmax": "S2S_Tmax"}).assign(stage=stage)


def _slim_plot_daily_timeseries(init, ts_df, output_png, output_pdf):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axvspan(pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17") + pd.Timedelta(hours=18), color="#F6D365", alpha=0.20, label="Stage-I")
    ax.axvspan(pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24") + pd.Timedelta(hours=18), color="#A1C4FD", alpha=0.20, label="Stage-II")
    ax.plot(ts_df["date"], ts_df["ERA5_Tmax"], color="black", lw=2.0, label="ERA5 Tmax")
    ax.plot(ts_df["date"], ts_df["S2S_mean_Tmax"], color="#1f77b4", lw=1.8, label="S2S mean Tmax")
    ax.fill_between(ts_df["date"], ts_df["S2S_q25"], ts_df["S2S_q75"], color="#1f77b4", alpha=0.14, label="S2S spread")
    ax.plot(ts_df["date"], ts_df["Slim_OLS_mean"], color="#2E8B57", lw=1.9, marker="s", ms=4, label="Slim OLS fitted mean")
    ax.fill_between(ts_df["date"], ts_df["Slim_OLS_q25"], ts_df["Slim_OLS_q75"], color="#2E8B57", alpha=0.12, label="Slim OLS fitted spread")
    ax.set_title(f"Tmax daily diagnostic series | Init {init} | Slim MLR NCVI conc.")
    ax.set_ylabel("Tmax (°C)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower right", ncol=2, fontsize=8.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    for out_path in [output_png, output_pdf]:
        assert_slim_mlr_output_path(out_path)
    fig.savefig(output_png, dpi=240)
    fig.savefig(output_pdf)
    plt.close(fig)


slim_daily_projection_parts = []
slim_daily_ts_files = []
slim_daily_qc_lines = [
    "Slim daily time-series uses stage-trained slim OLS models for projection only; no daily model is retrained.",
    "Stage-I daily rows use the Stage-I_dry slim model; Stage-II daily rows use the Stage-II_wet slim model; Total is not used for daily projection.",
]
for init in SLIM_INIT_ORDER:
    init_parts = []
    for stage in ["Stage-I_dry", "Stage-II_wet"]:
        stage_daily = _slim_daily_fit_for_stage(init, stage)
        init_parts.append(stage_daily)
        slim_daily_projection_parts.append(stage_daily)
        slim_daily_qc_lines.append(
            f"[{init} | {stage}] daily projection rows={len(stage_daily)}, dates={stage_daily['date'].min():%Y-%m-%d} to {stage_daily['date'].max():%Y-%m-%d}, predictors={';'.join(SLIM_PREDICTORS_BY_STAGE[stage])}"
        )
    init_daily = pd.concat(init_parts, ignore_index=True)
    if set(init_daily["stage"].unique()) != {"Stage-I_dry", "Stage-II_wet"}:
        raise RuntimeError(f"Slim daily projection stage check failed for {init}: stages={sorted(init_daily['stage'].unique())}")
    ts_summary = init_daily.groupby(["init_date", "date", "stage"], as_index=False).agg(
        ERA5_Tmax=("era5_tmax", "mean"),
        S2S_mean_Tmax=("S2S_Tmax", "mean"),
        S2S_q25=("S2S_Tmax", lambda x: float(np.nanquantile(x, 0.25))),
        S2S_q75=("S2S_Tmax", lambda x: float(np.nanquantile(x, 0.75))),
        Slim_OLS_mean=("Slim_OLS_daily_fitted", "mean"),
        Slim_OLS_q25=("Slim_OLS_daily_fitted", lambda x: float(np.nanquantile(x, 0.25))),
        Slim_OLS_q75=("Slim_OLS_daily_fitted", lambda x: float(np.nanquantile(x, 0.75))),
    )
    if len(ts_summary) != 11 or ts_summary["date"].min() != pd.Timestamp("2023-06-14") or ts_summary["date"].max() != pd.Timestamp("2023-06-24"):
        raise RuntimeError(f"Slim daily time-series date range check failed for {init}: n_dates={len(ts_summary)}, min={ts_summary['date'].min()}, max={ts_summary['date'].max()}")
    png = f"{SLIM_MLR_SUBDIRS['timeseries']}/fig_slim_daily_timeseries_NCVIconcurrent_{init}.png"
    pdf = f"{SLIM_MLR_SUBDIRS['timeseries']}/fig_slim_daily_timeseries_NCVIconcurrent_{init}.pdf"
    _slim_plot_daily_timeseries(init, ts_summary, png, pdf)
    slim_daily_ts_files.extend([png, pdf])


df_slim_daily_projection = pd.concat(slim_daily_projection_parts, ignore_index=True)
table_slim_daily_projection = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_daily_ols_projection_NCVIconcurrent.csv"
table_slim_daily_summary = f"{SLIM_MLR_SUBDIRS['tables']}/table_slim_daily_timeseries_summary_NCVIconcurrent.csv"
assert_slim_mlr_output_path(table_slim_daily_projection)
assert_slim_mlr_output_path(table_slim_daily_summary)
df_slim_daily_projection.to_csv(table_slim_daily_projection, index=False)
df_slim_daily_summary = df_slim_daily_projection.groupby(["init_date", "date", "stage"], as_index=False).agg(
    ERA5_Tmax=("era5_tmax", "mean"),
    S2S_mean_Tmax=("S2S_Tmax", "mean"),
    S2S_q25=("S2S_Tmax", lambda x: float(np.nanquantile(x, 0.25))),
    S2S_q75=("S2S_Tmax", lambda x: float(np.nanquantile(x, 0.75))),
    Slim_OLS_mean=("Slim_OLS_daily_fitted", "mean"),
    Slim_OLS_q25=("Slim_OLS_daily_fitted", lambda x: float(np.nanquantile(x, 0.25))),
    Slim_OLS_q75=("Slim_OLS_daily_fitted", lambda x: float(np.nanquantile(x, 0.75))),
)
df_slim_daily_summary.to_csv(table_slim_daily_summary, index=False)
for (init, stage, date), gg in df_slim_daily_projection.groupby(["init_date", "stage", "date"]):
    if gg["member"].nunique() != 51:
        raise RuntimeError(f"Slim daily projection final member-count check failed for {init}/{stage}/{date}: n={gg['member'].nunique()}")
if df_slim_daily_projection["date"].min() != pd.Timestamp("2023-06-14") or df_slim_daily_projection["date"].max() != pd.Timestamp("2023-06-24"):
    raise RuntimeError("Slim daily projection final date range check failed")
for fig_path in slim_daily_ts_files:
    if not os.path.exists(fig_path):
        raise RuntimeError(f"Slim daily time-series figure missing: {fig_path}")
slim_qc_lines.extend(["", "Slim daily time-series diagnostics:", *slim_daily_qc_lines, *[f"  {fp}" for fp in slim_daily_ts_files], ""])
print("Cell 8 slim daily time-series uses stage-trained slim OLS models for projection only; no daily model is retrained.")

# QC summary and output checks.
required_slim_csvs = [
    table_slim_predictor_sets,
    table_slim_model_summary,
    table_slim_model_coefficients,
    table_slim_lmg_long,
    table_slim_lofo_long,
    table_slim_vif,
    table_slim_fitted_members,
    table_slim_vs_full,
    table_slim_daily_projection,
    table_slim_daily_summary,
]
slim_lmg_combo_png_pattern = os.path.join(SLIM_MLR_SUBDIRS["lmg_combo"], "fig_slim_lmg_combo_NCVIconcurrent_*.png")
slim_init_lmg_combo_png_pattern = os.path.join(SLIM_MLR_SUBDIRS["lmg_combo_init"], "fig_slim_lmg_combo_NCVIconcurrent_init_*.png")
slim_init_lmg_combo_pdf_pattern = os.path.join(SLIM_MLR_SUBDIRS["lmg_combo_init"], "fig_slim_lmg_combo_NCVIconcurrent_init_*.pdf")
slim_lofo_combo_png_pattern = os.path.join(SLIM_MLR_SUBDIRS["lofo_combo"], "fig_slim_lofo_combo_NCVIconcurrent_init_*.png")
slim_corr_png_pattern = os.path.join(SLIM_MLR_SUBDIRS["corr_matrix"], "corr_slim_target_predictors_NCVIconcurrent_*.png")
slim_heatmap_png_pattern = os.path.join(SLIM_MLR_SUBDIRS["heatmap"], "*.png")
slim_timeseries_png_pattern = os.path.join(SLIM_MLR_SUBDIRS["timeseries"], "fig_slim_daily_timeseries_NCVIconcurrent_*.png")
slim_lmg_combo_png_count = len(glob.glob(slim_lmg_combo_png_pattern))
slim_init_lmg_combo_png_count = len(glob.glob(slim_init_lmg_combo_png_pattern))
slim_init_lmg_combo_pdf_count = len(glob.glob(slim_init_lmg_combo_pdf_pattern))
slim_lofo_combo_png_count = len(glob.glob(slim_lofo_combo_png_pattern))
slim_corr_png_count = len(glob.glob(slim_corr_png_pattern))
slim_heatmap_png_count = len(glob.glob(slim_heatmap_png_pattern))
slim_timeseries_png_count = len(glob.glob(slim_timeseries_png_pattern))
slim_qc_lines.extend([
    "",
    "Output file count checks:",
    f"  slim LMG combo PNG count = {slim_lmg_combo_png_count}",
    f"  slim init-level LMG combo PNG count = {slim_init_lmg_combo_png_count}",
    f"  slim init-level LMG combo PDF count = {slim_init_lmg_combo_pdf_count}",
    f"  slim LOFO combo PNG count = {slim_lofo_combo_png_count}",
    f"  slim target-predictor correlation PNG count = {slim_corr_png_count}",
    f"  slim heatmap PNG count = {slim_heatmap_png_count}",
    f"  slim daily time-series PNG count = {slim_timeseries_png_count}",
    "",
])
qc_slim_summary_txt = f"{SLIM_MLR_SUBDIRS['qc']}/qc_slim_mlr_NCVIconcurrent_summary.txt"
assert_slim_mlr_output_path(qc_slim_summary_txt)
with open(qc_slim_summary_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(slim_qc_lines))

_slim_output_checks = {
    "slim LMG combo PNG": (slim_lmg_combo_png_pattern, len(SLIM_INIT_ORDER) * len(SLIM_STAGE_ORDER)),
    "slim init-level LMG combo PNG": (slim_init_lmg_combo_png_pattern, len(SLIM_INIT_ORDER)),
    "slim init-level LMG combo PDF": (slim_init_lmg_combo_pdf_pattern, len(SLIM_INIT_ORDER)),
    "slim LOFO combo PNG": (slim_lofo_combo_png_pattern, len(SLIM_INIT_ORDER)),
    "slim correlation matrix PNG": (slim_corr_png_pattern, len(SLIM_INIT_ORDER) * len(SLIM_STAGE_ORDER)),
    "slim heatmap PNG": (slim_heatmap_png_pattern, 3),
    "slim daily time-series PNG": (slim_timeseries_png_pattern, len(SLIM_INIT_ORDER)),
}
for _desc, (_pattern, _expected_count) in _slim_output_checks.items():
    _matches = glob.glob(_pattern)
    _count_ok = len(_matches) >= _expected_count if _desc == "slim heatmap PNG" else len(_matches) == _expected_count
    if not _count_ok:
        raise RuntimeError(f"Slim MLR output count check failed for {_desc}: expected {_expected_count}, got {len(_matches)}, pattern={_pattern}, matches={_matches}")
for _csv in required_slim_csvs:
    assert_slim_mlr_output_path(_csv)
    if not os.path.exists(_csv):
        raise RuntimeError(f"Slim MLR required CSV missing: {_csv}")
if not os.path.exists(qc_slim_summary_txt):
    raise RuntimeError(f"Slim MLR QC summary missing: {qc_slim_summary_txt}")

print("[Slim MLR NCVI_concurrent] output count checks passed.")
print(f"[Slim MLR NCVI_concurrent] tables: {SLIM_MLR_SUBDIRS['tables']}")
print(f"[Slim MLR NCVI_concurrent] figures: {SLIM_MLR_SUBDIRS['figures']}")
print(f"[Slim MLR NCVI_concurrent] QC summary: {qc_slim_summary_txt}")

# %%
# ============================================================
# Cell 9: Johnson validation for final slim MLR with NCVI_concurrent
# ============================================================

set_audit_cell_name("Cell 9: Johnson / LOFO / method importance validation")
import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

print("\n" + "=" * 80)
print("Cell 9: Johnson validation for final slim MLR with NCVI_concurrent")
print("=" * 80)

JOHNSON_ROOT = FOUR_INIT_ROOT
JOHNSON_SUBDIRS = {
    "tables": f"{JOHNSON_ROOT}/tables",
    "figures": f"{JOHNSON_ROOT}/figures",
    "heatmap": f"{JOHNSON_ROOT}/figures/heatmap",
    "method_combo": f"{JOHNSON_ROOT}/figures/method_combo",
    "group_summary": f"{JOHNSON_ROOT}/figures/group_summary",
    "qc": f"{JOHNSON_ROOT}/qc",
}


def assert_johnson_output_path(path):
    if not str(path).startswith(f"{FOUR_INIT_ROOT}/") and str(path) != FOUR_INIT_ROOT:
        raise RuntimeError(f"Johnson validation output path is outside FOUR_INIT_ROOT: {path}")


assert_johnson_output_path(JOHNSON_ROOT + "/")
for _johnson_dir in JOHNSON_SUBDIRS.values():
    assert_johnson_output_path(_johnson_dir + "/")
    os.makedirs(_johnson_dir, exist_ok=True)

JOHNSON_SLIM_ROOT = FOUR_INIT_ROOT
JOHNSON_SLIM_TABLES = f"{JOHNSON_SLIM_ROOT}/tables"
JOHNSON_SLIM_INPUTS = {
    "summary": f"{JOHNSON_SLIM_TABLES}/table_slim_model_summary.csv",
    "lmg": f"{JOHNSON_SLIM_TABLES}/table_slim_lmg_long.csv",
    "lofo": f"{JOHNSON_SLIM_TABLES}/table_slim_lofo_long.csv",
    "fitted": f"{JOHNSON_SLIM_TABLES}/table_slim_fitted_members.csv",
    "predictor_sets": f"{JOHNSON_SLIM_TABLES}/table_slim_predictor_sets.csv",
    "vif": f"{JOHNSON_SLIM_TABLES}/table_slim_vif.csv",
}
JOHNSON_STAGE_TCC_TABLE = f"{FOUR_INIT_ROOT}/tables/table_stage_member_raw_factors_TCC.csv"
JOHNSON_NCVI_STAGE_TABLE = f"{FOUR_INIT_ROOT}/tables/table_ncvi_concurrent_stage_values.csv"
for _input_name, _input_path in {**JOHNSON_SLIM_INPUTS, "stage_tcc": JOHNSON_STAGE_TCC_TABLE, "ncvi_stage": JOHNSON_NCVI_STAGE_TABLE}.items():
    if not os.path.exists(_input_path):
        raise RuntimeError(
            f"Required input for Cell 9 Johnson validation is missing ({_input_name}): {_input_path}. "
            "Please run Cell 5, Cell 7, and Cell 8 before running Cell 9."
        )

JOHNSON_INIT_ORDER = FULL_INIT_LIST
JOHNSON_STAGE_ORDER = ["Stage-I_dry", "Stage-II_wet", "Total"]
JOHNSON_ROW_LABELS = {
    (init, stage): f"{pd.Timestamp(init):%m-%d} {'Stage-I' if stage == 'Stage-I_dry' else ('Stage-II' if stage == 'Stage-II_wet' else 'Total')}"
    for init in JOHNSON_INIT_ORDER for stage in JOHNSON_STAGE_ORDER
}
JOHNSON_DISPLAY_LABELS = {
    "sm_avg": "SM",
    "SM_resid": "SM resid",
    "SHF_resid": "SHF resid",
    "NCHN_tp": "TP",
    "z500_anom_NCHN": "Z500 anom",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
    "TCC_resid": "TCC resid",
    "NCVI_concurrent": "NCVI conc.",
}
JOHNSON_EXPECTED_PREDICTORS = {
    "Stage-I_dry": ["sm_avg", "SHF_resid", "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "TCC_resid", "NCVI_concurrent"],
    "Stage-II_wet": ["NCHN_tp", "z500_anom_NCHN", "NCVI_concurrent", "TCC_resid", "SM_resid", "Initial_MSEstar_max_NCHN"],
    "Total": ["NCHN_tp", "z500_anom_NCHN", "NCVI_concurrent", "Initial_MSEstar_max_NCHN", "TCC_resid", "SM_resid"],
}
JOHNSON_HEATMAP_FACTOR_ORDER = [
    "sm_avg", "SHF_resid", "NCHN_tp", "SM_resid", "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "TCC_resid", "NCVI_concurrent",
]
JOHNSON_MECHANISM_GROUPS = {
    "Stage-I_dry": {
        "sm_avg": "land_soil",
        "SHF_resid": "land_heat_flux",
        "z500_anom_NCHN": "circulation",
        "NCVI_concurrent": "circulation_cold_vortex",
        "Initial_MSEstar_max_NCHN": "antecedent_thermodynamic",
        "TCC_resid": "cloud_radiation_residual",
    },
    "Stage-II_wet": {
        "NCHN_tp": "precipitation_hydrological",
        "z500_anom_NCHN": "circulation",
        "NCVI_concurrent": "circulation_cold_vortex",
        "TCC_resid": "cloud_radiation_residual",
        "SM_resid": "land_memory_residual",
        "Initial_MSEstar_max_NCHN": "antecedent_thermodynamic",
    },
    "Total": {
        "NCHN_tp": "precipitation_hydrological",
        "z500_anom_NCHN": "circulation",
        "NCVI_concurrent": "circulation_cold_vortex",
        "TCC_resid": "cloud_radiation_residual",
        "SM_resid": "land_memory_residual",
        "Initial_MSEstar_max_NCHN": "antecedent_thermodynamic",
    },
}


def _johnson_resid_series(df, y_col, control_cols):
    X = sm.add_constant(df[control_cols].astype(float), has_constant="add")
    model = sm.OLS(df[y_col].astype(float), X).fit()
    return df[y_col].astype(float) - model.predict(X)


def _johnson_prepare_stage_group(df_stage, df_ncvi, init, stage):
    g = df_stage[(df_stage["init_date"].astype(str) == init) & (df_stage["stage"] == stage)].copy()
    if len(g) != 51 or g["member"].nunique() != 51:
        raise RuntimeError(f"Johnson validation expected 51 raw stage members for {init}/{stage}, got rows={len(g)}, unique_members={g['member'].nunique() if 'member' in g.columns else 'NA'}")
    nc = df_ncvi[(df_ncvi["init_date"].astype(str) == init) & (df_ncvi["stage"] == stage)][["init_date", "stage", "member", "NCVI_concurrent"]].copy()
    if len(nc) != 51 or nc["member"].nunique() != 51:
        raise RuntimeError(f"Johnson validation expected 51 NCVI_concurrent rows for {init}/{stage}, got rows={len(nc)}, unique_members={nc['member'].nunique() if 'member' in nc.columns else 'NA'}")
    if "NCVI_concurrent" in g.columns:
        g = g.drop(columns=["NCVI_concurrent"])
    g = g.merge(nc, on=["init_date", "stage", "member"], how="left", validate="one_to_one")
    if stage == "Stage-I_dry":
        g["SHF_resid"] = _johnson_resid_series(g, "SHF_Avg", ["sm_avg"])
        g["TCC_resid"] = _johnson_resid_series(g, "TCC_Avg", ["sm_avg"])
    else:
        g["SM_resid"] = _johnson_resid_series(g, "sm_avg", ["NCHN_tp"])
        g["TCC_resid"] = _johnson_resid_series(g, "TCC_Avg", ["NCHN_tp"])
    predictors = JOHNSON_EXPECTED_PREDICTORS[stage]
    required = ["member", "y_tmax"] + predictors
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise RuntimeError(f"Johnson validation missing columns for {init}/{stage}: {missing}")
    g_model = g.dropna(subset=required).sort_values("member").reset_index(drop=True)
    if len(g_model) != 51 or g_model["member"].nunique() != 51:
        nan_counts = g[required].isna().sum().to_dict()
        raise RuntimeError(f"Johnson validation modeling sample must be 51 for {init}/{stage}; rows={len(g_model)}, unique_members={g_model['member'].nunique()}; nan_counts={nan_counts}")
    return g_model


def _johnson_relative_weights(X_df, y_series, factor_names):
    if len(X_df) != 51:
        raise RuntimeError(f"Johnson relative weights require n=51, got n={len(X_df)}")
    X = X_df[factor_names].astype(float).copy()
    y = pd.Series(y_series, index=X.index).astype(float)
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    yz = (y - y.mean()) / (y.std(ddof=1) + 1e-12)
    Rxx = np.corrcoef(Xz.values, rowvar=False)
    rxy = np.array([np.corrcoef(Xz[col].values, yz.values)[0, 1] for col in factor_names], dtype=float)
    eigval, eigvec = np.linalg.eigh(Rxx)
    min_eig = float(np.nanmin(eigval))
    eigval_clip = np.clip(eigval, 1e-12, None)
    delta = eigvec @ np.diag(np.sqrt(eigval_clip)) @ eigvec.T
    beta = np.linalg.pinv(delta) @ rxy
    raw_weights = (delta ** 2) @ (beta ** 2)
    raw_weights = np.where(raw_weights < 1e-12, 0.0, raw_weights)
    total = float(np.sum(raw_weights))
    pct = raw_weights / total * 100.0 if total > 0 else np.zeros_like(raw_weights)
    rows = []
    ranks = {factor: rank + 1 for rank, factor in enumerate([factor_names[i] for i in np.argsort(-pct)])}
    for i, factor in enumerate(factor_names):
        rows.append({
            "factor": factor,
            "display_label": JOHNSON_DISPLAY_LABELS.get(factor, factor),
            "johnson_raw": float(raw_weights[i]),
            "johnson_pct": float(pct[i]),
            "johnson_rank": int(ranks[factor]),
        })
    return pd.DataFrame(rows), total, min_eig


def _johnson_spearman(a, b):
    r, _ = spearmanr(a, b)
    return float(r) if np.isfinite(r) else np.nan


def _johnson_top_set(df, rank_col, k):
    return set(df.sort_values(rank_col).head(k)["factor"].tolist())


def _johnson_factor_list(df, rank_col, k):
    return ";".join(df.sort_values(rank_col).head(k)["factor"].tolist())


def _johnson_group_list(df, rank_col, k, stage):
    groups = [JOHNSON_MECHANISM_GROUPS[stage][f] for f in df.sort_values(rank_col).head(k)["factor"].tolist()]
    return ";".join(dict.fromkeys(groups))


def _johnson_interpretation(stage, group_overlap_lmg_johnson, group_overlap_lmg_lofo, k):
    if group_overlap_lmg_johnson >= min(k, 3) - 1 and group_overlap_lmg_lofo >= min(k, 3) - 1:
        if stage == "Stage-I_dry":
            return "Stage-I robust: dominant groups remain land heat flux / soil moisture / circulation when supported by overlap diagnostics."
        return "Stage-II/Total robust: dominant groups remain precipitation / circulation / cold-vortex activity when supported by overlap diagnostics."
    return "Method-sensitive: top factors differ, interpret individual percentages cautiously."


def _johnson_plot_heatmap(df, value_col, title, cbar_label, output_stem, cmap="YlOrRd", center=None, fmt=".1f"):
    row_order = [JOHNSON_ROW_LABELS[(init, stage)] for init in JOHNSON_INIT_ORDER for stage in JOHNSON_STAGE_ORDER]
    col_order = [JOHNSON_DISPLAY_LABELS[f] for f in JOHNSON_HEATMAP_FACTOR_ORDER]
    plot_df = df.copy()
    plot_df["row_label"] = plot_df.apply(lambda r: JOHNSON_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
    mat = plot_df.pivot(index="row_label", columns="display_label", values=value_col).reindex(index=row_order, columns=col_order)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    sns.heatmap(mat, ax=ax, cmap=cmap, center=center, annot=True, fmt=fmt, linewidths=0.4, linecolor="white", cbar_kws={"label": cbar_label}, annot_kws={"fontsize": 9})
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    png = f"{JOHNSON_SUBDIRS['heatmap']}/{output_stem}.png"
    pdf = f"{JOHNSON_SUBDIRS['heatmap']}/{output_stem}.pdf"
    for out_path in [png, pdf]:
        assert_johnson_output_path(out_path)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)


johnson_qc_lines = [
    "Johnson validation for final slim MLR with NCVI_concurrent",
    "Top3 is the primary comparison.",
    "Top4 is supplementary.",
    "Top5 is retained only for QC.",
    "Johnson validation is used to test robustness of LMG, not to redefine the final slim model.",
    "LOFO group percentages use positive Delta R2 values only; negative Delta R2 rows are retained in factor-level tables.",
    "Input files used:",
    *[f"  {name}: {path}" for name, path in JOHNSON_SLIM_INPUTS.items()],
    f"  stage_tcc: {JOHNSON_STAGE_TCC_TABLE}",
    f"  ncvi_stage: {JOHNSON_NCVI_STAGE_TABLE}",
    "",
]

df_johnson_summary = pd.read_csv(JOHNSON_SLIM_INPUTS["summary"])
df_johnson_lmg = pd.read_csv(JOHNSON_SLIM_INPUTS["lmg"])
df_johnson_lofo = pd.read_csv(JOHNSON_SLIM_INPUTS["lofo"])
df_johnson_predictors = pd.read_csv(JOHNSON_SLIM_INPUTS["predictor_sets"])
df_johnson_vif = pd.read_csv(JOHNSON_SLIM_INPUTS["vif"])
df_johnson_stage = pd.read_csv(JOHNSON_STAGE_TCC_TABLE)
df_johnson_ncvi = pd.read_csv(JOHNSON_NCVI_STAGE_TABLE)
for _df in [df_johnson_summary, df_johnson_lmg, df_johnson_lofo, df_johnson_predictors, df_johnson_vif, df_johnson_stage, df_johnson_ncvi]:
    if "init_date" in _df.columns:
        _df["init_date"] = _df["init_date"].astype(str)

for stage in JOHNSON_STAGE_ORDER:
    predictor_rows = df_johnson_predictors[df_johnson_predictors["stage"] == stage].sort_values("predictors_raw_name")
    expected_sorted = sorted(JOHNSON_EXPECTED_PREDICTORS[stage])
    observed_sorted = sorted(predictor_rows["predictors_raw_name"].tolist())
    if observed_sorted != expected_sorted:
        raise RuntimeError(f"Cell 9 predictor-set assertion failed for {stage}: observed={observed_sorted}, expected={expected_sorted}")

method_rows = []
rank_corr_rows = []
group_rows = []
topk_rows = []
johnson_sum_checks = []
combo_pngs = []
combo_pdfs = []

for init in JOHNSON_INIT_ORDER:
    for stage in JOHNSON_STAGE_ORDER:
        predictors = JOHNSON_EXPECTED_PREDICTORS[stage]
        g_model = _johnson_prepare_stage_group(df_johnson_stage, df_johnson_ncvi, init, stage)
        X_df = g_model[predictors].astype(float)
        y = g_model["y_tmax"].astype(float)
        slim_summary = df_johnson_summary[(df_johnson_summary["init_date"].astype(str) == init) & (df_johnson_summary["stage"] == stage)]
        if len(slim_summary) != 1:
            raise RuntimeError(f"Cell 9 expected one slim summary row for {init}/{stage}, got {len(slim_summary)}")
        r2_slim = float(slim_summary.iloc[0]["R2"])
        johnson_df, johnson_sum, min_eig = _johnson_relative_weights(X_df, y, predictors)
        sum_diff = float(johnson_sum - r2_slim)
        johnson_sum_checks.append({"init_date": init, "stage": stage, "sum_diff": sum_diff, "johnson_sum": johnson_sum, "R2": r2_slim})
        if abs(sum_diff) > 1e-5:
            johnson_qc_lines.append(f"WARNING [{init} | {stage}] sum(Johnson raw) - R2 = {sum_diff:.8e}")
        lmg_sub = df_johnson_lmg[(df_johnson_lmg["init_date"].astype(str) == init) & (df_johnson_lmg["stage"] == stage)][["predictor", "LMG_percent", "LMG_rank"]]
        lofo_sub = df_johnson_lofo[(df_johnson_lofo["init_date"].astype(str) == init) & (df_johnson_lofo["stage"] == stage)][["predictor", "Delta_R2", "LOFO_rank_R2"]]
        comp = pd.DataFrame({"factor": predictors})
        comp = comp.merge(lmg_sub.rename(columns={"predictor": "factor", "LMG_percent": "lmg_pct", "LMG_rank": "lmg_rank"}), on="factor", how="left")
        comp = comp.merge(johnson_df, on="factor", how="left")
        comp = comp.merge(lofo_sub.rename(columns={"predictor": "factor", "Delta_R2": "lofo_delta_r2", "LOFO_rank_R2": "lofo_rank"}), on="factor", how="left")
        if comp[["lmg_pct", "lmg_rank", "johnson_pct", "johnson_rank", "lofo_delta_r2", "lofo_rank"]].isna().any().any():
            raise RuntimeError(f"Cell 9 method comparison has missing importance values for {init}/{stage}")
        for k in [3, 4, 5]:
            comp[f"in_lmg_top{k}"] = comp["lmg_rank"] <= k
            comp[f"in_johnson_top{k}"] = comp["johnson_rank"] <= k
            comp[f"in_lofo_top{k}"] = comp["lofo_rank"] <= k
        comp["init_date"] = init
        comp["stage"] = stage
        comp["display_label"] = comp["factor"].map(JOHNSON_DISPLAY_LABELS)
        comp["mechanism_group"] = comp["factor"].map(JOHNSON_MECHANISM_GROUPS[stage])
        method_rows.extend(comp[[
            "init_date", "stage", "factor", "display_label", "lmg_pct", "lmg_rank", "johnson_raw", "johnson_pct", "johnson_rank", "lofo_delta_r2", "lofo_rank",
            "in_lmg_top3", "in_johnson_top3", "in_lofo_top3", "in_lmg_top4", "in_johnson_top4", "in_lofo_top4", "in_lmg_top5", "in_johnson_top5", "in_lofo_top5",
        ]].to_dict("records"))

        spearman_lmg_johnson = _johnson_spearman(comp["lmg_rank"], comp["johnson_rank"])
        spearman_lmg_lofo = _johnson_spearman(comp["lmg_rank"], comp["lofo_rank"])
        spearman_johnson_lofo = _johnson_spearman(comp["johnson_rank"], comp["lofo_rank"])
        stability_flag = "stable" if spearman_lmg_johnson >= 0.7 else ("moderately_stable" if spearman_lmg_johnson >= 0.5 else "method_sensitive")
        rank_row = {
            "init_date": init,
            "stage": stage,
            "spearman_lmg_johnson": spearman_lmg_johnson,
            "spearman_lmg_lofo": spearman_lmg_lofo,
            "spearman_johnson_lofo": spearman_johnson_lofo,
            "stability_flag": stability_flag,
        }
        for k in [3, 4, 5]:
            top_lmg = _johnson_top_set(comp, "lmg_rank", k)
            top_johnson = _johnson_top_set(comp, "johnson_rank", k)
            top_lofo = _johnson_top_set(comp, "lofo_rank", k)
            rank_row[f"top{k}_overlap_lmg_johnson"] = len(top_lmg & top_johnson)
            rank_row[f"top{k}_overlap_lmg_lofo"] = len(top_lmg & top_lofo)
            rank_row[f"top{k}_overlap_johnson_lofo"] = len(top_johnson & top_lofo)
            lmg_groups = set(_johnson_group_list(comp, "lmg_rank", k, stage).split(";"))
            johnson_groups = set(_johnson_group_list(comp, "johnson_rank", k, stage).split(";"))
            lofo_groups = set(_johnson_group_list(comp, "lofo_rank", k, stage).split(";"))
            topk_rows.append({
                "init_date": init,
                "stage": stage,
                "k": k,
                "lmg_top_factors": _johnson_factor_list(comp, "lmg_rank", k),
                "johnson_top_factors": _johnson_factor_list(comp, "johnson_rank", k),
                "lofo_top_factors": _johnson_factor_list(comp, "lofo_rank", k),
                "lmg_top_groups": ";".join(lmg_groups),
                "johnson_top_groups": ";".join(johnson_groups),
                "lofo_top_groups": ";".join(lofo_groups),
                "factor_overlap_lmg_johnson": len(top_lmg & top_johnson),
                "group_overlap_lmg_johnson": len(lmg_groups & johnson_groups),
                "factor_overlap_lmg_lofo": len(top_lmg & top_lofo),
                "group_overlap_lmg_lofo": len(lmg_groups & lofo_groups),
                "interpretation_note": _johnson_interpretation(stage, len(lmg_groups & johnson_groups), len(lmg_groups & lofo_groups), k),
            })
        rank_corr_rows.append(rank_row)

        lofo_positive_sum = float(np.sum(np.maximum(comp["lofo_delta_r2"].astype(float).values, 0.0)))
        comp["lofo_positive_pct"] = np.where(lofo_positive_sum > 0, np.maximum(comp["lofo_delta_r2"].astype(float), 0.0) / lofo_positive_sum * 100.0, 0.0)
        for method, value_col in [("LMG", "lmg_pct"), ("Johnson", "johnson_pct"), ("LOFO", "lofo_positive_pct")]:
            group_df = comp.groupby("mechanism_group", as_index=False)[value_col].sum().rename(columns={value_col: "group_importance_pct"})
            group_df["group_rank"] = group_df["group_importance_pct"].rank(ascending=False, method="first").astype(int)
            for _, gr in group_df.iterrows():
                group_rows.append({
                    "init_date": init,
                    "stage": stage,
                    "method": method,
                    "mechanism_group": gr["mechanism_group"],
                    "group_importance_raw": float(gr["group_importance_pct"]),
                    "group_importance_pct": float(gr["group_importance_pct"]),
                    "group_rank": int(gr["group_rank"]),
                })

        johnson_qc_lines.extend([
            f"[{init} | {stage}] n=51; predictors={';'.join(predictors)}",
            f"[{init} | {stage}] R2={r2_slim:.6f}; sum(Johnson raw)={johnson_sum:.6f}; diff={sum_diff:.8e}; min_eigenvalue={min_eig:.8e}",
            f"[{init} | {stage}] Spearman LMG-Johnson={spearman_lmg_johnson:.3f}, LMG-LOFO={spearman_lmg_lofo:.3f}, Johnson-LOFO={spearman_johnson_lofo:.3f}; stability={stability_flag}",
        ])

        plot_order = comp.sort_values("lmg_rank")["factor"].tolist()
        plot_labels = [JOHNSON_DISPLAY_LABELS[f] for f in plot_order]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
        for ax, title, value_col, color in zip(axes, ["LMG %", "Johnson %", "LOFO +DeltaR² %"], ["lmg_pct", "johnson_pct", "lofo_positive_pct"], ["#6AA84F", "#3C78D8", "#E69138"]):
            vals = comp.set_index("factor").loc[plot_order, value_col].astype(float).values
            ax.barh(plot_labels[::-1], vals[::-1], color=color, alpha=0.85)
            xmax = max(float(np.nanmax(vals)), 1.0)
            ax.set_xlim(0, xmax * 1.18)
            ax.set_title(title)
            for y_idx, val in enumerate(vals[::-1]):
                ax.text(val + xmax * 0.025, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.5)
        fig.suptitle(f"Method importance comparison | {init} | {stage}\nTop3 is the primary comparison; Top4 is supplementary.", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        combo_png = f"{JOHNSON_SUBDIRS['method_combo']}/fig_method_importance_combo_{init}_{stage}.png"
        combo_pdf = f"{JOHNSON_SUBDIRS['method_combo']}/fig_method_importance_combo_{init}_{stage}.pdf"
        for out_path in [combo_png, combo_pdf]:
            assert_johnson_output_path(out_path)
        fig.savefig(combo_png, dpi=240)
        fig.savefig(combo_pdf)
        plt.close(fig)
        combo_pngs.append(combo_png)
        combo_pdfs.append(combo_pdf)

df_method_comp = pd.DataFrame(method_rows)
df_rank_corr = pd.DataFrame(rank_corr_rows)
df_group_importance = pd.DataFrame(group_rows)
df_topk_overlap = pd.DataFrame(topk_rows)

table_method_comp = f"{JOHNSON_SUBDIRS['tables']}/table_method_importance_comparison.csv"
table_rank_corr = f"{JOHNSON_SUBDIRS['tables']}/table_method_rank_correlations.csv"
table_group_importance = f"{JOHNSON_SUBDIRS['tables']}/table_mechanism_group_importance.csv"
table_topk_overlap = f"{JOHNSON_SUBDIRS['tables']}/table_topk_mechanism_overlap.csv"
for out_csv in [table_method_comp, table_rank_corr, table_group_importance, table_topk_overlap]:
    assert_johnson_output_path(out_csv)
df_method_comp.to_csv(table_method_comp, index=False)
df_rank_corr.to_csv(table_rank_corr, index=False)
df_group_importance.to_csv(table_group_importance, index=False)
df_topk_overlap.to_csv(table_topk_overlap, index=False)

_johnson_plot_heatmap(df_method_comp, "johnson_pct", "Johnson relative weights (%) | final slim MLR", "Johnson relative weight (%)", "heatmap_johnson_relative_weights", cmap="YlOrRd", center=None, fmt=".1f")
rank_diff = df_method_comp.copy()
rank_diff["rank_diff"] = rank_diff["johnson_rank"] - rank_diff["lmg_rank"]
_johnson_plot_heatmap(rank_diff, "rank_diff", "Johnson rank - LMG rank | final slim MLR", "Johnson rank - LMG rank", "heatmap_lmg_johnson_rank_difference", cmap="coolwarm", center=0, fmt=".0f")

rank_plot = df_rank_corr.copy()
rank_plot["row_label"] = rank_plot.apply(lambda r: JOHNSON_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
rank_mat = rank_plot.set_index("row_label")[["spearman_lmg_johnson", "spearman_lmg_lofo", "spearman_johnson_lofo"]].reindex([JOHNSON_ROW_LABELS[(init, stage)] for init in JOHNSON_INIT_ORDER for stage in JOHNSON_STAGE_ORDER])
fig, ax = plt.subplots(figsize=(7.8, 5.5))
sns.heatmap(rank_mat, ax=ax, cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Spearman rank correlation"}, annot_kws={"fontsize": 9})
ax.set_title("Method Spearman rank correlations")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
for stem in ["heatmap_method_spearman_correlation"]:
    png = f"{JOHNSON_SUBDIRS['heatmap']}/{stem}.png"
    pdf = f"{JOHNSON_SUBDIRS['heatmap']}/{stem}.pdf"
    for out_path in [png, pdf]:
        assert_johnson_output_path(out_path)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
plt.close(fig)

top3_plot = df_topk_overlap[df_topk_overlap["k"] == 3].copy()
top3_plot["row_label"] = top3_plot.apply(lambda r: JOHNSON_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
top3_mat = top3_plot.set_index("row_label")[["factor_overlap_lmg_johnson", "factor_overlap_lmg_lofo", "group_overlap_lmg_johnson", "group_overlap_lmg_lofo"]].reindex([JOHNSON_ROW_LABELS[(init, stage)] for init in JOHNSON_INIT_ORDER for stage in JOHNSON_STAGE_ORDER])
fig, ax = plt.subplots(figsize=(8.8, 5.5))
sns.heatmap(top3_mat, ax=ax, cmap="YlGnBu", vmin=0, vmax=3, annot=True, fmt=".0f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Top3 overlap count"}, annot_kws={"fontsize": 9})
ax.set_title("Top3 factor and mechanism-group overlap")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
for stem in ["heatmap_top3_overlap"]:
    png = f"{JOHNSON_SUBDIRS['heatmap']}/{stem}.png"
    pdf = f"{JOHNSON_SUBDIRS['heatmap']}/{stem}.pdf"
    for out_path in [png, pdf]:
        assert_johnson_output_path(out_path)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
plt.close(fig)

group_plot = df_group_importance.copy()
group_plot["row_label"] = group_plot.apply(lambda r: JOHNSON_ROW_LABELS[(r["init_date"], r["stage"])], axis=1)
group_plot["panel"] = group_plot["row_label"] + " | " + group_plot["method"]
JOHNSON_MECHANISM_GROUP_DISPLAY_LABELS = {
    "antecedent_thermodynamic": "antecedent_thermodynamic\n(Init MSE*)",
    "circulation": "circulation\n(Z500 anom)",
    "circulation_cold_vortex": "circulation_cold_vortex\n(NCVI conc.)",
    "cloud_radiation_residual": "cloud_radiation_residual\n(TCC resid)",
    "land_heat_flux": "land_heat_flux\n(SHF resid)",
    "land_soil": "land_soil\n(SM)",
    "land_memory_residual": "land_memory_residual\n(SM resid)",
    "precipitation_hydrological": "precipitation_hydrological\n(TP)",
}
group_plot["mechanism_group_display_label"] = group_plot["mechanism_group"].map(JOHNSON_MECHANISM_GROUP_DISPLAY_LABELS).fillna(group_plot["mechanism_group"])
group_mat = group_plot.pivot(index="panel", columns="mechanism_group_display_label", values="group_importance_pct").fillna(0.0)
fig_h = max(8.4, 0.34 * len(group_mat) + 3.2)
fig, ax = plt.subplots(figsize=(14.5, fig_h))
sns.heatmap(group_mat, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Mechanism-group contribution (%)"}, annot_kws={"fontsize": 8})
ax.set_title("Mechanism-group importance comparison | LMG / Johnson / LOFO")
ax.set_xlabel("Mechanism groups are aggregated from the final slim predictors; labels in parentheses indicate the corresponding predictor names.")
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
fig.tight_layout()
group_png = f"{JOHNSON_SUBDIRS['group_summary']}/fig_mechanism_group_importance_comparison.png"
group_pdf = f"{JOHNSON_SUBDIRS['group_summary']}/fig_mechanism_group_importance_comparison.pdf"
for out_path in [group_png, group_pdf]:
    assert_johnson_output_path(out_path)
fig.savefig(group_png, dpi=240)
fig.savefig(group_pdf)
plt.close(fig)

johnson_qc_lines.extend([
    "",
    "Top-k overlap summary:",
    *[f"[{r.init_date} | {r.stage} | Top{int(r.k)}] factor overlap LMG-Johnson={int(r.factor_overlap_lmg_johnson)}, LMG-LOFO={int(r.factor_overlap_lmg_lofo)}; group overlap LMG-Johnson={int(r.group_overlap_lmg_johnson)}, LMG-LOFO={int(r.group_overlap_lmg_lofo)}; {r.interpretation_note}" for r in df_topk_overlap.itertuples(index=False)],
])
qc_johnson_txt = f"{JOHNSON_SUBDIRS['qc']}/qc_johnson_validation_slim_NCVIconcurrent.txt"
assert_johnson_output_path(qc_johnson_txt)
with open(qc_johnson_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(johnson_qc_lines))

required_johnson_csvs = [table_method_comp, table_rank_corr, table_group_importance, table_topk_overlap]
required_johnson_pngs = [
    f"{JOHNSON_SUBDIRS['heatmap']}/heatmap_johnson_relative_weights.png",
    f"{JOHNSON_SUBDIRS['heatmap']}/heatmap_lmg_johnson_rank_difference.png",
    f"{JOHNSON_SUBDIRS['heatmap']}/heatmap_method_spearman_correlation.png",
    f"{JOHNSON_SUBDIRS['heatmap']}/heatmap_top3_overlap.png",
    group_png,
    *combo_pngs,
]
required_johnson_pdfs = [p.replace(".png", ".pdf") for p in required_johnson_pngs]
for csv_path in required_johnson_csvs:
    if (not os.path.exists(csv_path)) or pd.read_csv(csv_path).empty:
        raise RuntimeError(f"Cell 9 required CSV missing or empty: {csv_path}")
if df_method_comp.groupby(["init_date", "stage"]).size().shape[0] != len(JOHNSON_INIT_ORDER) * len(JOHNSON_STAGE_ORDER):
    raise RuntimeError("Cell 9 Johnson method comparison does not contain the expected four-init stage combinations")
if not (df_method_comp.groupby(["init_date", "stage"]).size() == 6).all():
    raise RuntimeError("Cell 9 Johnson method comparison does not contain exactly 6 factors for every init-stage")
if not all(len(_johnson_prepare_stage_group(df_johnson_stage, df_johnson_ncvi, init, stage)) == 51 for init in JOHNSON_INIT_ORDER for stage in JOHNSON_STAGE_ORDER):
    raise RuntimeError("Cell 9 n=51 check failed during final validation")
if any(abs(row["sum_diff"]) > 1e-5 for row in johnson_sum_checks):
    print(f"[Cell 9 Johnson validation] WARNING: Johnson raw-weight sum differs from R2 by >1e-5 for: {johnson_sum_checks}")
for fig_path in [*required_johnson_pngs, *required_johnson_pdfs]:
    assert_johnson_output_path(fig_path)
    if not os.path.exists(fig_path):
        raise RuntimeError(f"Cell 9 required figure missing: {fig_path}")
if not os.path.exists(qc_johnson_txt):
    raise RuntimeError(f"Cell 9 QC file missing: {qc_johnson_txt}")

print(f"[Cell 9 Johnson validation] output root: {JOHNSON_ROOT}")
print(f"[Cell 9 Johnson validation] Johnson validation table: {table_method_comp}")
print(f"[Cell 9 Johnson validation] method comparison table: {table_method_comp}")
print(f"[Cell 9 Johnson validation] rank correlation table: {table_rank_corr}")
print(f"[Cell 9 Johnson validation] mechanism group table: {table_group_importance}")
print(f"[Cell 9 Johnson validation] QC path: {qc_johnson_txt}")
print(f"[Cell 9 Johnson validation] figure count: PNG={len(required_johnson_pngs)}, PDF={len(required_johnson_pdfs)}")
print("Cell 9 validates LMG robustness on the final slim MLR using Johnson relative weights and LOFO Delta R². Top3 is the primary comparison; Top4 is supplementary; Top5 is QC only.")

# %%
# ============================================================
# Cell 10: Paper-style figures for slim MLR main results and robustness
# ============================================================

set_audit_cell_name("Cell 10: paper figures")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n" + "=" * 80)
print("Cell 10: Paper-style figures for slim MLR main results and robustness")
print("=" * 80)

PAPER_FIG_OUT_DIR = FOUR_INIT_ROOT
PAPER_FIG_SUBDIRS = {
    "main_results": f"{PAPER_FIG_OUT_DIR}/figures/main_results",
    "robustness": f"{PAPER_FIG_OUT_DIR}/figures/robustness",
    "qc": f"{PAPER_FIG_OUT_DIR}/qc",
}


def assert_paper_fig_output_path(path):
    if not str(path).startswith(f"{FOUR_INIT_ROOT}/") and str(path) != FOUR_INIT_ROOT:
        raise RuntimeError(f"Paper figure output path is outside FOUR_INIT_ROOT: {path}")


assert_paper_fig_output_path(PAPER_FIG_OUT_DIR + "/")
for _paper_dir in PAPER_FIG_SUBDIRS.values():
    assert_paper_fig_output_path(_paper_dir + "/")
    os.makedirs(_paper_dir, exist_ok=True)

PAPER_SLIM_TABLE_DIR = f"{FOUR_INIT_ROOT}/tables"
PAPER_JOHNSON_TABLE_DIR = f"{FOUR_INIT_ROOT}/tables"
PAPER_INPUT_TABLES = {
    "slim_summary": f"{PAPER_SLIM_TABLE_DIR}/table_slim_model_summary.csv",
    "slim_fitted": f"{PAPER_SLIM_TABLE_DIR}/table_slim_fitted_members.csv",
    "slim_lmg": f"{PAPER_SLIM_TABLE_DIR}/table_slim_lmg_long.csv",
    "johnson_method_comparison": f"{PAPER_JOHNSON_TABLE_DIR}/table_method_importance_comparison.csv",
}
for _name, _path in PAPER_INPUT_TABLES.items():
    if not os.path.exists(_path):
        raise RuntimeError(f"Cell 10 missing upstream {_name} table: {_path}. Please run Cell 8 and Cell 9 first.")

PAPER_INIT_ORDER = [MAIN_INIT, CONTRAST_INIT, *AUX_INITS]  # main, contrast, then auxiliary appendix initializations.
PAPER_STAGE_ORDER = ["Stage-I_dry", "Stage-II_wet", "Total"]
PAPER_STAGE_SHORT = {"Stage-I_dry": "Stage-I", "Stage-II_wet": "Stage-II", "Total": "Total"}
PAPER_DISPLAY_LABELS = {
    "sm_avg": "SM",
    "SM_resid": "SM resid",
    "SHF_resid": "SHF resid",
    "NCHN_tp": "TP",
    "z500_anom_NCHN": "Z500 anom",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
    "TCC_resid": "TCC resid",
    "NCVI_concurrent": "NCVI conc.",
}
PAPER_SLIM_PREDICTORS = {
    "Stage-I_dry": ["sm_avg", "SHF_resid", "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "TCC_resid", "NCVI_concurrent"],
    "Stage-II_wet": ["NCHN_tp", "z500_anom_NCHN", "NCVI_concurrent", "TCC_resid", "SM_resid", "Initial_MSEstar_max_NCHN"],
    "Total": ["NCHN_tp", "z500_anom_NCHN", "NCVI_concurrent", "Initial_MSEstar_max_NCHN", "TCC_resid", "SM_resid"],
}

df_paper_summary = pd.read_csv(PAPER_INPUT_TABLES["slim_summary"])
df_paper_fitted = pd.read_csv(PAPER_INPUT_TABLES["slim_fitted"])
df_paper_lmg = pd.read_csv(PAPER_INPUT_TABLES["slim_lmg"])
df_paper_methods = pd.read_csv(PAPER_INPUT_TABLES["johnson_method_comparison"])
for _df in [df_paper_summary, df_paper_fitted, df_paper_lmg, df_paper_methods]:
    if "init_date" in _df.columns:
        _df["init_date"] = _df["init_date"].astype(str)


def _paper_get_summary_row(init, stage):
    rows = df_paper_summary[(df_paper_summary["init_date"] == init) & (df_paper_summary["stage"] == stage)]
    if len(rows) != 1:
        raise RuntimeError(f"Cell 10 expected one slim summary row for {init}/{stage}, got {len(rows)}")
    return rows.iloc[0]


def _paper_plot_scatter(ax, init, stage):
    fit_rows = df_paper_fitted[(df_paper_fitted["init_date"] == init) & (df_paper_fitted["stage"] == stage)]
    if len(fit_rows) != 51:
        raise RuntimeError(f"Cell 10 expected 51 fitted-member rows for {init}/{stage}, got {len(fit_rows)}")
    y = fit_rows["y_tmax"].astype(float).values
    fitted = fit_rows["y_fitted"].astype(float).values
    summary = _paper_get_summary_row(init, stage)
    ax.scatter(y, fitted, s=30, alpha=0.75, color="#2f6db3", edgecolor="white", linewidth=0.4)
    lim_min = float(min(np.nanmin(y), np.nanmin(fitted)))
    lim_max = float(max(np.nanmax(y), np.nanmax(fitted)))
    pad = max((lim_max - lim_min) * 0.05, 0.25)
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], ls="--", color="0.35", lw=1.0)
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_xlabel("ECMWF member Tmax (°C)")
    ax.set_ylabel("Slim OLS fitted Tmax (°C)")
    ax.set_title(f"{PAPER_STAGE_SHORT[stage]}: R²={summary['R2']:.2f}, Adj.R²={summary['Adj_R2']:.2f}, RMSE={summary['RMSE']:.2f}")


def _paper_ordered_lmg(init, stage):
    rows = df_paper_lmg[(df_paper_lmg["init_date"] == init) & (df_paper_lmg["stage"] == stage)].copy()
    expected = PAPER_SLIM_PREDICTORS[stage]
    if set(rows["predictor"].tolist()) != set(expected):
        raise RuntimeError(f"Cell 10 LMG predictor mismatch for {init}/{stage}: got={rows['predictor'].tolist()}, expected={expected}")
    rows["display_label"] = rows["predictor"].map(PAPER_DISPLAY_LABELS)
    return rows.sort_values("LMG_percent", ascending=False)


def _paper_plot_lmg_bar(ax, init, stage, order_desc=True):
    rows = _paper_ordered_lmg(init, stage)
    plot_rows = rows.sort_values("LMG_percent", ascending=True) if order_desc else rows
    vals = plot_rows["LMG_percent"].astype(float).values
    labels = plot_rows["display_label"].tolist()
    ax.barh(labels, vals, color="#6AA84F", alpha=0.86)
    xmax = max(float(np.nanmax(vals)) if len(vals) else 1.0, 1.0)
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel("LMG relative importance (%)")
    ax.set_title(f"{PAPER_STAGE_SHORT[stage]} LMG")
    for y_idx, val in enumerate(vals):
        ax.text(val + xmax * 0.025, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.5)


def _paper_save_main_results(init):
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), gridspec_kw={"width_ratios": [1.0, 1.18]}, constrained_layout=False)
    for row_idx, stage in enumerate(PAPER_STAGE_ORDER):
        _paper_plot_scatter(axes[row_idx, 0], init, stage)
        _paper_plot_lmg_bar(axes[row_idx, 1], init, stage)
    fig.suptitle(f"Slim MLR | NCVI conc. | {init}", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.07, wspace=0.36, hspace=0.55)
    png = f"{PAPER_FIG_SUBDIRS['main_results']}/fig_paper_slim_main_results_{init}.png"
    pdf = f"{PAPER_FIG_SUBDIRS['main_results']}/fig_paper_slim_main_results_{init}.pdf"
    for out_path in [png, pdf]:
        assert_paper_fig_output_path(out_path)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _paper_method_rows(init, stage):
    rows = df_paper_methods[(df_paper_methods["init_date"] == init) & (df_paper_methods["stage"] == stage)].copy()
    expected = PAPER_SLIM_PREDICTORS[stage]
    if set(rows["factor"].tolist()) != set(expected):
        raise RuntimeError(f"Cell 10 method-comparison predictor mismatch for {init}/{stage}: got={rows['factor'].tolist()}, expected={expected}")
    rows["display_label"] = rows["factor"].map(PAPER_DISPLAY_LABELS)
    pos_sum = float(np.maximum(rows["lofo_delta_r2"].astype(float), 0.0).sum())
    rows["lofo_positive_pct"] = np.where(pos_sum > 0, np.maximum(rows["lofo_delta_r2"].astype(float), 0.0) / pos_sum * 100.0, 0.0)
    return rows


def _paper_save_robustness(init):
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 10.5), constrained_layout=False)
    # LOFO normalized contribution is computed from positive LOFO Delta R² within each init-stage; this is for visual comparison only.
    for row_idx, stage in enumerate(PAPER_STAGE_ORDER):
        rows = _paper_method_rows(init, stage)
        order = _paper_ordered_lmg(init, stage)["predictor"].tolist()
        for col_idx, (method_title, value_col, color, x_label) in enumerate([
            ("LMG", "lmg_pct", "#6AA84F", "LMG relative importance (%)"),
            ("Johnson", "johnson_pct", "#3C78D8", "Johnson relative weights (%)"),
            ("LOFO", "lofo_positive_pct", "#E69138", "LOFO normalized contribution (%)"),
        ]):
            ax = axes[row_idx, col_idx]
            plot_rows = rows.set_index("factor").loc[order].reset_index()
            vals = plot_rows[value_col].astype(float).values[::-1]
            labels = plot_rows["display_label"].tolist()[::-1]
            ax.barh(labels, vals, color=color, alpha=0.86)
            xmax = max(float(np.nanmax(vals)) if len(vals) else 1.0, 1.0)
            ax.set_xlim(0, xmax * 1.15)
            ax.set_xlabel(x_label)
            title_prefix = PAPER_STAGE_SHORT[stage] if col_idx == 0 else ""
            ax.set_title(f"{title_prefix} {method_title}".strip())
            for y_idx, val in enumerate(vals):
                ax.text(val + xmax * 0.025, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.2)
    fig.suptitle(f"Robustness of slim MLR factor importance | {init}\nTop3 is the primary comparison; Top4 is supplementary.", fontsize=15, y=0.99)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.07, wspace=0.45, hspace=0.58)
    png = f"{PAPER_FIG_SUBDIRS['robustness']}/fig_paper_mlr_robustness_{init}.png"
    pdf = f"{PAPER_FIG_SUBDIRS['robustness']}/fig_paper_mlr_robustness_{init}.pdf"
    for out_path in [png, pdf]:
        assert_paper_fig_output_path(out_path)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf

paper_main_files = []
paper_robustness_files = []
for init in PAPER_INIT_ORDER:
    main_png, main_pdf = _paper_save_main_results(init)
    robust_png, robust_pdf = _paper_save_robustness(init)
    paper_main_files.extend([main_png, main_pdf])
    paper_robustness_files.extend([robust_png, robust_pdf])

qc_paper_txt = f"{PAPER_FIG_SUBDIRS['qc']}/qc_paper_figures_slim_NCVIconcurrent.txt"
assert_paper_fig_output_path(qc_paper_txt)
paper_qc_lines = [
    "Paper-style figures for slim MLR main results and robustness",
    "2023-06-12 marked as main text.",
    "2023-06-08 marked as appendix / supplementary.",
    "Upstream tables used:",
    *[f"  {name}: {path}" for name, path in PAPER_INPUT_TABLES.items()],
    f"Figure A PNG count = {len([p for p in paper_main_files if p.endswith('.png')])}",
    f"Figure A PDF count = {len([p for p in paper_main_files if p.endswith('.pdf')])}",
    f"Figure B PNG count = {len([p for p in paper_robustness_files if p.endswith('.png')])}",
    f"Figure B PDF count = {len([p for p in paper_robustness_files if p.endswith('.pdf')])}",
    "Generated files:",
    *[f"  {p}" for p in [*paper_main_files, *paper_robustness_files]],
]
with open(qc_paper_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(paper_qc_lines))

required_paper_files = [*paper_main_files, *paper_robustness_files, qc_paper_txt]
for required_path in required_paper_files:
    assert_paper_fig_output_path(required_path)
    if not os.path.exists(required_path):
        raise RuntimeError(f"Cell 10 required paper-style figure output missing: {required_path}")
if len([p for p in paper_main_files if p.endswith(".png")]) != 2 or len([p for p in paper_main_files if p.endswith(".pdf")]) != 2:
    raise RuntimeError(f"Cell 10 Figure A output count check failed: {paper_main_files}")
if len([p for p in paper_robustness_files if p.endswith(".png")]) != 2 or len([p for p in paper_robustness_files if p.endswith(".pdf")]) != 2:
    raise RuntimeError(f"Cell 10 Figure B output count check failed: {paper_robustness_files}")

print(f"[Cell 10 paper figures] output root: {PAPER_FIG_OUT_DIR}")
print(f"[Cell 10 paper figures] Figure A files: {paper_main_files}")
print(f"[Cell 10 paper figures] Figure B files: {paper_robustness_files}")
print(f"[Cell 10 paper figures] QC: {qc_paper_txt}")

# %%
# ============================================================
# Cell 11: Stage-I raw/residual consistency audit for Fig.5 and slim MLR
# ============================================================

set_audit_cell_name("Cell 11: Stage-I optional diagnostics")
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

print("\n" + "=" * 80)
print("Cell 11: Stage-I raw/residual consistency audit for Fig.5 and slim MLR")
print("=" * 80)

STAGEI_AUDIT_DIR = f"{FOUR_INIT_ROOT}/optional_stageI_raw_resid_consistency_audit"
STAGEI_AUDIT_SUBDIRS = {
    "tables": f"{STAGEI_AUDIT_DIR}/tables",
    "figures": f"{STAGEI_AUDIT_DIR}/figures",
    "qc": f"{STAGEI_AUDIT_DIR}/qc",
}


def assert_stagei_audit_output_path(path):
    if not str(path).startswith(f"{STAGEI_AUDIT_DIR}/"):
        raise RuntimeError(f"Stage-I audit output path is outside FOUR_INIT_ROOT: {path}")


assert_stagei_audit_output_path(STAGEI_AUDIT_DIR + "/")
for _stagei_dir in STAGEI_AUDIT_SUBDIRS.values():
    assert_stagei_audit_output_path(_stagei_dir + "/")
    os.makedirs(_stagei_dir, exist_ok=True)

PROCESS_TS_ROOT = "/data1/huangy/fig6/NC/v11/hotcold_stage_group_timeseries_tables_multiinit_totalfig6"
PROCESS_TS_TABLE_DIR = f"{PROCESS_TS_ROOT}/tables"
PROCESS_TS_FIG_DIR = f"{PROCESS_TS_ROOT}/paper_theme_timeseries"
STAGEI_GROUP_FILE = "/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv"
STAGEI_FIG5_TABLE_DIR = PROCESS_TS_TABLE_DIR
STAGEI_DAILY_TS_FILE = f"{STAGEI_FIG5_TABLE_DIR}/daily_timeseries_all_factors.csv"
STAGEI_STAGE_MEAN_FILE = f"{STAGEI_FIG5_TABLE_DIR}/stage_mean_summary.csv"
STAGEI_GROUP_DIFF_FILE = f"{STAGEI_FIG5_TABLE_DIR}/group_difference_summary.csv"
STAGEI_MEMBER_STAGE_FILE = f"{STAGEI_FIG5_TABLE_DIR}/member_stage_values_for_tests.csv"
STAGEI_GROUP_AUDIT_FILE = f"{STAGEI_FIG5_TABLE_DIR}/group_member_audit.csv"
STAGEI_FIG5_TABLES = {
    "daily_timeseries": STAGEI_DAILY_TS_FILE,
    "stage_mean_summary": STAGEI_STAGE_MEAN_FILE,
    "group_difference_summary": STAGEI_GROUP_DIFF_FILE,
    "member_stage_values": STAGEI_MEMBER_STAGE_FILE,
    "group_member_audit": STAGEI_GROUP_AUDIT_FILE,
}
STAGEI_MLR_RAW_TABLE = f"{FOUR_INIT_ROOT}/tables/table_stage_member_raw_factors_TCC.csv"
STAGEI_NCVI_TABLE = f"{FOUR_INIT_ROOT}/tables/table_ncvi_concurrent_stage_values.csv"
STAGEI_SLIM_FITTED_TABLE = f"{FOUR_INIT_ROOT}/tables/table_slim_fitted_members.csv"
STAGEI_SLIM_SUMMARY_TABLE = f"{FOUR_INIT_ROOT}/tables/table_slim_model_summary.csv"
STAGEI_SLIM_LMG_TABLE = f"{FOUR_INIT_ROOT}/tables/table_slim_lmg_long.csv"
STAGEI_INPUTS = {
    "group_file": STAGEI_GROUP_FILE,
    **STAGEI_FIG5_TABLES,
    "mlr_raw": STAGEI_MLR_RAW_TABLE,
    "ncvi": STAGEI_NCVI_TABLE,
    "slim_fitted": STAGEI_SLIM_FITTED_TABLE,
    "slim_summary": STAGEI_SLIM_SUMMARY_TABLE,
    "slim_lmg": STAGEI_SLIM_LMG_TABLE,
}
for _name, _path in STAGEI_INPUTS.items():
    if not os.path.exists(_path):
        if _name in {"mlr_raw", "ncvi", "slim_fitted", "slim_summary", "slim_lmg"}:
            raise RuntimeError(
                f"Required v12 four-init MLR input for Stage-I raw/residual consistency audit is missing ({_name}): {_path}. "
                "Please run Cell 7 \u2192 Cell 8 before running this audit."
            )
        raise RuntimeError(f"Required input for Stage-I raw/residual consistency audit is missing ({_name}): {_path}")

STAGEI_INIT_ORDER_MAIN = ["2023-06-12", "2023-06-08"]
STAGEI_INIT_ORDER_ALL = ["2023-06-12", "2023-06-08", "2023-06-05", "2023-06-01"]
STAGEI_INIT_ORDER = STAGEI_INIT_ORDER_ALL
STAGEI_WINDOW_START = pd.Timestamp("2023-06-14")
STAGEI_WINDOW_END = pd.Timestamp("2023-06-17")
STAGEI_DISPLAY_LABELS = {
    "y_tmax": "Stage-I Tmax",
    "sm_avg": "SM",
    "SHF_Avg": "Raw SHF",
    "SHF_resid": "SHF resid",
    "TCC_Avg": "Raw TCC",
    "TCC_resid": "TCC resid",
    "z500_anom_NCHN": "Z500 anom",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
    "NCVI_concurrent": "NCVI conc.",
}
STAGEI_GROUP_COLORS = {"Hot17": "#D62728", "Middle17": "#6BAED6", "Cold17": "#2CA02C"}
STAGEI_GROUP_ORDER = ["Hot17", "Middle17", "Cold17"]


def _stagei_find_col(df, candidates, context):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot find {context}; candidates={candidates}; columns={list(df.columns)}")


def _stagei_normalize_stage(value):
    s = str(value)
    if s in {"Stage-I", "Stage-I_dry", "Stage_I", "Stage1", "stage1"}:
        return "Stage-I"
    return s


def _stagei_normalize_group(value):
    s = str(value)
    low = s.lower()
    if "hot" in low:
        return "Hot17"
    if "cold" in low:
        return "Cold17"
    if "middle" in low or "mid" in low:
        return "Middle17"
    return s


def _stagei_resid_series(df, y_col, control_cols):
    X = sm.add_constant(df[control_cols].astype(float), has_constant="add")
    model = sm.OLS(df[y_col].astype(float), X).fit()
    return df[y_col].astype(float) - model.predict(X)


def _stagei_corr_stats(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {"pearson_r": np.nan, "spearman_r": np.nan, "linear_slope": np.nan, "linear_R2": np.nan, "p_value": np.nan, "n_members": int(len(x))}
    pearson_r, p_value = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)
    X = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, X).fit()
    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "linear_slope": float(model.params[1]),
        "linear_R2": float(model.rsquared),
        "p_value": float(p_value),
        "n_members": int(len(x)),
    }


def _stagei_add_regression_line(ax, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0:
        return
    model = sm.OLS(y[mask], sm.add_constant(x[mask], has_constant="add")).fit()
    xs = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 50)
    ys = model.params[0] + model.params[1] * xs
    ax.plot(xs, ys, color="0.25", lw=1.0, ls="--")


def _stagei_scatter_panel(ax, df, x_col, y_col, title):
    for group in STAGEI_GROUP_ORDER:
        gg = df[df["hotcold_group"] == group]
        ax.scatter(gg[x_col], gg[y_col], s=34, alpha=0.78, color=STAGEI_GROUP_COLORS[group], label=group, edgecolor="white", linewidth=0.4)
    _stagei_add_regression_line(ax, df[x_col], df[y_col])
    stats = _stagei_corr_stats(df[x_col], df[y_col])
    ax.set_title(f"{title}\nr={stats['pearson_r']:.2f}, R²={stats['linear_R2']:.2f}, slope={stats['linear_slope']:.2g}")
    ax.set_xlabel(STAGEI_DISPLAY_LABELS.get(x_col, x_col))
    ax.set_ylabel(STAGEI_DISPLAY_LABELS.get(y_col, y_col))


def _stagei_load_groups():
    df_group = pd.read_csv(STAGEI_GROUP_FILE)
    init_col = _stagei_find_col(df_group, ["init_date", "init", "forecast_init"], "group init column")
    stage_col = _stagei_find_col(df_group, ["stage", "stage_name", "window"], "group stage column")
    member_col = _stagei_find_col(df_group, ["member", "number", "ensemble_member"], "group member column")
    group_col = _stagei_find_col(df_group, ["hotcold_group", "group", "member_group", "tmax_group"], "hot/cold group column")
    out = df_group[[init_col, stage_col, member_col, group_col]].copy()
    out.columns = ["init_date", "stage", "member", "hotcold_group"]
    out["init_date"] = out["init_date"].astype(str)
    out["stage"] = out["stage"].map(_stagei_normalize_stage)
    out["hotcold_group"] = out["hotcold_group"].map(_stagei_normalize_group)
    out = out[(out["init_date"].isin(STAGEI_INIT_ORDER)) & (out["stage"] == "Stage-I")].copy()
    for init in STAGEI_INIT_ORDER:
        g = out[out["init_date"] == init]
        counts = g.groupby("hotcold_group")["member"].nunique().to_dict()
        if counts != {"Hot17": 17, "Middle17": 17, "Cold17": 17}:
            raise RuntimeError(f"Stage-I group count check failed for {init}: {counts}")
        if g["member"].nunique() != 51 or len(g) != 51:
            raise RuntimeError(f"Stage-I group membership must be 51 unique rows for {init}, got rows={len(g)}, unique={g['member'].nunique()}")
    return out


stagei_qc_lines = [
    "Stage-I raw/residual consistency audit for Fig.5 and slim MLR",
    "Stage-I model window = 2023-06-14 to 2023-06-17.",
    "Fig.5 may display 2023-06-18 as transition/next-stage context; 2023-06-18 is not used in Stage-I MLR scalar/residualization.",
    "Input files:",
    *[f"  {k}: {v}" for k, v in STAGEI_INPUTS.items()],
    "",
]

df_stagei_groups = _stagei_load_groups()
df_stagei_raw = pd.read_csv(STAGEI_MLR_RAW_TABLE)
df_stagei_raw["init_date"] = df_stagei_raw["init_date"].astype(str)
df_stagei_ncvi = pd.read_csv(STAGEI_NCVI_TABLE)
df_stagei_ncvi["init_date"] = df_stagei_ncvi["init_date"].astype(str)
df_stagei_fit = pd.read_csv(STAGEI_SLIM_FITTED_TABLE)
df_stagei_fit["init_date"] = df_stagei_fit["init_date"].astype(str)
df_fig5_member = pd.read_csv(STAGEI_FIG5_TABLES["member_stage_values"])
if "date" in pd.read_csv(STAGEI_FIG5_TABLES["daily_timeseries"], nrows=1).columns:
    df_fig5_daily_dates = pd.read_csv(STAGEI_FIG5_TABLES["daily_timeseries"], usecols=["date"])
    df_fig5_daily_dates["date"] = pd.to_datetime(df_fig5_daily_dates["date"], errors="coerce")
    fig5_includes_0618 = bool((df_fig5_daily_dates["date"] == pd.Timestamp("2023-06-18")).any())
else:
    fig5_includes_0618 = False
stagei_qc_lines.append(f"Fig.5 display table contains 2023-06-18 = {fig5_includes_0618}")

stagei_member_rows = []
for init in STAGEI_INIT_ORDER:
    raw = df_stagei_raw[(df_stagei_raw["init_date"] == init) & (df_stagei_raw["stage"] == "Stage-I_dry")].copy()
    if len(raw) != 51 or raw["member"].nunique() != 51:
        raise RuntimeError(f"Stage-I MLR raw table expected 51 rows for {init}, got rows={len(raw)}, unique={raw['member'].nunique() if 'member' in raw.columns else 'NA'}")
    nc = df_stagei_ncvi[(df_stagei_ncvi["init_date"] == init) & (df_stagei_ncvi["stage"] == "Stage-I_dry")][["init_date", "stage", "member", "NCVI_concurrent"]].copy()
    fit = df_stagei_fit[(df_stagei_fit["init_date"] == init) & (df_stagei_fit["stage"] == "Stage-I_dry")][["init_date", "stage", "member", "y_fitted"]].copy()
    raw = raw.merge(nc, on=["init_date", "stage", "member"], how="left", validate="one_to_one")
    raw = raw.merge(fit, on=["init_date", "stage", "member"], how="left", validate="one_to_one")
    raw["SHF_resid"] = _stagei_resid_series(raw, "SHF_Avg", ["sm_avg"])
    raw["TCC_resid"] = _stagei_resid_series(raw, "TCC_Avg", ["sm_avg"])
    if "SSR_Avg" in raw.columns:
        raw["SSR_resid"] = _stagei_resid_series(raw, "SSR_Avg", ["sm_avg"])
    raw = raw.merge(df_stagei_groups[df_stagei_groups["init_date"] == init][["init_date", "member", "hotcold_group"]], on=["init_date", "member"], how="left", validate="one_to_one")
    if raw["hotcold_group"].isna().any():
        raise RuntimeError(f"Stage-I group merge produced missing group labels for {init}")
    counts = raw.groupby("hotcold_group")["member"].nunique().to_dict()
    if counts != {"Hot17": 17, "Middle17": 17, "Cold17": 17}:
        raise RuntimeError(f"Stage-I audit merged group count check failed for {init}: {counts}")
    corr_shf_sm = _stagei_corr_stats(raw["sm_avg"], raw["SHF_resid"])["pearson_r"]
    corr_tcc_sm = _stagei_corr_stats(raw["sm_avg"], raw["TCC_resid"])["pearson_r"]
    if abs(corr_shf_sm) > 1e-8 or abs(corr_tcc_sm) > 1e-8:
        raise RuntimeError(f"Stage-I residual orthogonality check failed for {init}: corr(SHF_resid,SM)={corr_shf_sm}, corr(TCC_resid,SM)={corr_tcc_sm}")
    slim_stage = raw[[
        "init_date", "stage", "member", "hotcold_group", "y_tmax", "sm_avg", "SHF_Avg", "SHF_resid", "TCC_Avg", "TCC_resid",
        "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "NCVI_concurrent", "y_fitted",
    ]].rename(columns={"y_fitted": "slim_ols_fitted_value"})
    stagei_member_rows.extend(slim_stage.to_dict("records"))


df_stagei_member_audit = pd.DataFrame(stagei_member_rows)
table_stagei_member = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_raw_resid_member_audit.csv"
assert_stagei_audit_output_path(table_stagei_member)
df_stagei_member_audit.to_csv(table_stagei_member, index=False)

summary_vars = ["y_tmax", "sm_avg", "SHF_Avg", "SHF_resid", "TCC_Avg", "TCC_resid", "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "NCVI_concurrent"]
group_summary_rows = []
for init in STAGEI_INIT_ORDER:
    g = df_stagei_member_audit[df_stagei_member_audit["init_date"] == init]
    for var in summary_vars:
        row = {"init_date": init, "stage": "Stage-I_dry", "variable": var}
        for group in STAGEI_GROUP_ORDER:
            gg = g[g["hotcold_group"] == group][var].astype(float)
            row[f"{group}_mean"] = float(gg.mean())
            row[f"{group}_std"] = float(gg.std())
        row["Hot17_minus_Cold17"] = row["Hot17_mean"] - row["Cold17_mean"]
        group_summary_rows.append(row)

table_stagei_group_summary = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_raw_resid_group_difference_summary.csv"
assert_stagei_audit_output_path(table_stagei_group_summary)
df_stagei_group_summary = pd.DataFrame(group_summary_rows)
df_stagei_group_summary.to_csv(table_stagei_group_summary, index=False)

scatter_predictors = ["sm_avg", "SHF_Avg", "SHF_resid", "TCC_Avg", "TCC_resid", "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "NCVI_concurrent"]
scatter_rows = []
for init in STAGEI_INIT_ORDER:
    g = df_stagei_member_audit[df_stagei_member_audit["init_date"] == init]
    for predictor in scatter_predictors:
        stats = _stagei_corr_stats(g[predictor], g["y_tmax"])
        scatter_rows.append({"init_date": init, "stage": "Stage-I_dry", "predictor": predictor, **stats})

table_stagei_scatter = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_raw_resid_scatter_stats.csv"
assert_stagei_audit_output_path(table_stagei_scatter)
df_stagei_scatter = pd.DataFrame(scatter_rows)
df_stagei_scatter.to_csv(table_stagei_scatter, index=False)

fig5_consistency_rows = []
fig5_consistency_tolerance = 1e-6
fig5_init_col = _stagei_find_col(df_fig5_member, ["init", "init_date"], "Fig.5 member init column")
fig5_grouping_stage_col = _stagei_find_col(df_fig5_member, ["grouping_stage"], "Fig.5 member grouping_stage column")
fig5_eval_stage_col = _stagei_find_col(df_fig5_member, ["eval_stage"], "Fig.5 member eval_stage column")
fig5_factor_col = _stagei_find_col(df_fig5_member, ["factor"], "Fig.5 member factor column")
fig5_member_col = _stagei_find_col(df_fig5_member, ["member"], "Fig.5 member column")
fig5_value_col = _stagei_find_col(df_fig5_member, ["member_stage_mean_value"], "Fig.5 member-stage scalar value column")
fig5_group_col = _stagei_find_col(df_fig5_member, ["hotcold_group"], "Fig.5 member hotcold_group column")
df_fig5_member[fig5_init_col] = df_fig5_member[fig5_init_col].astype(str)
df_fig5_member[fig5_grouping_stage_col] = df_fig5_member[fig5_grouping_stage_col].map(_stagei_normalize_stage)
df_fig5_member[fig5_eval_stage_col] = df_fig5_member[fig5_eval_stage_col].map(_stagei_normalize_stage)
fig5_to_mlr_factor_map = {
    "Tmax": "y_tmax",
    "sm_avg": "sm_avg",
    "SHF_Avg": "SHF_Avg",
    "TCC_Avg": "TCC_Avg",
    "z500_anom_NCHN": "z500_anom_NCHN",
    "NCVI_conc": "NCVI_concurrent",
}
fig5_core_factors = {"Tmax", "sm_avg", "SHF_Avg"}
for init in STAGEI_INIT_ORDER:
    fig5_g = df_fig5_member[
        (df_fig5_member[fig5_init_col] == init)
        & (df_fig5_member[fig5_grouping_stage_col] == "Stage-I")
        & (df_fig5_member[fig5_eval_stage_col] == "Stage-I")
    ].copy()
    mlr_g = df_stagei_member_audit[df_stagei_member_audit["init_date"] == init].copy()
    mlr_g["_stagei_member_key"] = mlr_g["member"].astype(str)
    for factor_fig5, factor_mlr in fig5_to_mlr_factor_map.items():
        status = "ok"
        max_diff = np.nan
        mean_diff = np.nan
        n_compared = 0
        consistent = False
        if factor_mlr not in mlr_g.columns:
            status = "missing_mlr_factor"
            warning_msg = f"Fig.5 vs MLR scalar consistency missing MLR factor for {init}/{factor_fig5}->{factor_mlr}"
            stagei_qc_lines.append(f"WARNING: {warning_msg}")
            if factor_fig5 in fig5_core_factors:
                raise RuntimeError(warning_msg)
        else:
            fig5_factor_g = fig5_g[fig5_g[fig5_factor_col] == factor_fig5].copy()
            if fig5_factor_g.empty:
                status = "missing_fig5_factor"
                warning_msg = f"Fig.5 vs MLR scalar consistency missing Fig.5 factor for {init}/{factor_fig5}"
                stagei_qc_lines.append(f"WARNING: {warning_msg}")
                if factor_fig5 in fig5_core_factors:
                    raise RuntimeError(warning_msg)
            else:
                fig5_factor_g["_stagei_member_key"] = fig5_factor_g[fig5_member_col].astype(str)
                fig5_factor_g = fig5_factor_g[["_stagei_member_key", fig5_value_col]].drop_duplicates("_stagei_member_key")
                merged = fig5_factor_g.merge(
                    mlr_g[["_stagei_member_key", factor_mlr]],
                    on="_stagei_member_key",
                    how="inner",
                    validate="one_to_one",
                )
                n_compared = int(len(merged))
                if n_compared:
                    diffs = np.abs(merged[fig5_value_col].astype(float).values - merged[factor_mlr].astype(float).values)
                    max_diff = float(np.nanmax(diffs))
                    mean_diff = float(np.nanmean(diffs))
                    consistent = bool(n_compared == 51 and max_diff <= fig5_consistency_tolerance)
                    status = "ok" if consistent else "mismatch"
                else:
                    status = "missing_fig5_factor"
                if not consistent:
                    warning_msg = (
                        f"Fig.5 vs MLR scalar mismatch for {init}/{factor_fig5}->{factor_mlr}: "
                        f"n={n_compared}, max diff={max_diff}"
                    )
                    stagei_qc_lines.append(f"WARNING: {warning_msg}")
                    print(f"[Stage-I audit WARNING] {warning_msg}")
        fig5_consistency_rows.append({
            "init_date": init,
            "factor_fig5": factor_fig5,
            "factor_mlr": factor_mlr,
            "n_members_compared": n_compared,
            "max_abs_diff_member_scalar": max_diff,
            "mean_abs_diff_member_scalar": mean_diff,
            "fig5_grouping_stage": "Stage-I",
            "fig5_eval_stage": "Stage-I",
            "mlr_stage": "Stage-I_dry",
            "consistent_within_tolerance": consistent,
            "status": status,
        })

table_stagei_consistency = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_fig5_vs_mlr_scalar_consistency.csv"
assert_stagei_audit_output_path(table_stagei_consistency)
pd.DataFrame(fig5_consistency_rows).to_csv(table_stagei_consistency, index=False)

for init in STAGEI_INIT_ORDER:
    g = df_stagei_member_audit[df_stagei_member_audit["init_date"] == init]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    panels = [
        ("sm_avg", "y_tmax", "Tmax vs SM"),
        ("SHF_Avg", "y_tmax", "Tmax vs Raw SHF"),
        ("SHF_resid", "y_tmax", "Tmax vs SHF resid"),
        ("sm_avg", "SHF_Avg", "Raw SHF vs SM"),
        ("sm_avg", "SHF_resid", "SHF resid vs SM"),
        ("SHF_resid", "SHF_Avg", "Raw SHF vs SHF resid"),
    ]
    for ax, (x_col, y_col, title) in zip(axes.flat, panels):
        _stagei_scatter_panel(ax, g, x_col, y_col, title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Stage-I raw/residual SHF consistency audit | Init {init}", fontsize=15)
    fig_stagei_shf_png = f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_SHF_raw_resid_scatter_audit_{init}.png"
    fig_stagei_shf_pdf = f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_SHF_raw_resid_scatter_audit_{init}.pdf"
    for out_path in [fig_stagei_shf_png, fig_stagei_shf_pdf]:
        assert_stagei_audit_output_path(out_path)
    fig.savefig(fig_stagei_shf_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_stagei_shf_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    slim_panels = [
        ("sm_avg", "Tmax vs SM"),
        ("SHF_resid", "Tmax vs SHF resid"),
        ("z500_anom_NCHN", "Tmax vs Z500 anom"),
        ("Initial_MSEstar_max_NCHN", "Tmax vs Init MSE*"),
        ("TCC_resid", "Tmax vs TCC resid"),
        ("NCVI_concurrent", "Tmax vs NCVI conc."),
    ]
    for ax, (x_col, title) in zip(axes.flat, slim_panels):
        _stagei_scatter_panel(ax, g, x_col, "y_tmax", title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Stage-I slim predictor scatter audit | Init {init}", fontsize=15)
    fig_stagei_slim_png = f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_slim_predictor_scatter_audit_{init}.png"
    fig_stagei_slim_pdf = f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_slim_predictor_scatter_audit_{init}.pdf"
    for out_path in [fig_stagei_slim_png, fig_stagei_slim_pdf]:
        assert_stagei_audit_output_path(out_path)
    fig.savefig(fig_stagei_slim_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_stagei_slim_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), constrained_layout=True)
    contrast_vars = [("SHF_Avg", "Raw SHF"), ("SHF_resid", "SHF resid"), ("sm_avg", "SM")]
    for ax, (var, title) in zip(axes, contrast_vars):
        data = [g[g["hotcold_group"] == group][var].astype(float).values for group in STAGEI_GROUP_ORDER]
        ax.boxplot(data, labels=STAGEI_GROUP_ORDER, patch_artist=True)
        for idx, group in enumerate(STAGEI_GROUP_ORDER, start=1):
            vals = g[g["hotcold_group"] == group][var].astype(float).values
            ax.scatter(np.full_like(vals, idx, dtype=float), vals, color=STAGEI_GROUP_COLORS[group], s=20, alpha=0.72, edgecolor="white", linewidth=0.3)
        diff = float(g[g["hotcold_group"] == "Hot17"][var].mean() - g[g["hotcold_group"] == "Cold17"][var].mean())
        ax.set_title(f"{title}\nHot-Cold mean diff={diff:.3g}")
        ax.set_ylabel(STAGEI_DISPLAY_LABELS.get(var, var))
    fig.suptitle(f"Stage-I raw vs residual group contrast | Init {init}", fontsize=15)
    fig_stagei_contrast_png = f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_raw_vs_resid_group_contrast_{init}.png"
    fig_stagei_contrast_pdf = f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_raw_vs_resid_group_contrast_{init}.pdf"
    for out_path in [fig_stagei_contrast_png, fig_stagei_contrast_pdf]:
        assert_stagei_audit_output_path(out_path)
    fig.savefig(fig_stagei_contrast_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_stagei_contrast_pdf, bbox_inches="tight")
    plt.close(fig)

for init in STAGEI_INIT_ORDER:
    g = df_stagei_member_audit[df_stagei_member_audit["init_date"] == init]
    corr_raw_shf = _stagei_corr_stats(g["SHF_Avg"], g["y_tmax"])["pearson_r"]
    corr_resid_shf = _stagei_corr_stats(g["SHF_resid"], g["y_tmax"])["pearson_r"]
    corr_shf_sm = _stagei_corr_stats(g["sm_avg"], g["SHF_resid"])["pearson_r"]
    corr_tcc_sm = _stagei_corr_stats(g["sm_avg"], g["TCC_resid"])["pearson_r"]
    raw_diff = float(g[g["hotcold_group"] == "Hot17"]["SHF_Avg"].mean() - g[g["hotcold_group"] == "Cold17"]["SHF_Avg"].mean())
    resid_diff = float(g[g["hotcold_group"] == "Hot17"]["SHF_resid"].mean() - g[g["hotcold_group"] == "Cold17"]["SHF_resid"].mean())
    interpretation = (
        "Fig.5 raw SHF evolution should not be interpreted as the residual SHF contribution used in the slim MLR."
        if abs(corr_raw_shf) < 0.25 and abs(corr_resid_shf) >= 0.25
        else "Need to inspect slim MLR input/LMG calculation." if abs(corr_resid_shf) < 0.25 else "Raw and residual SHF both show member-axis signal; compare grouped and residual diagnostics together."
    )
    stagei_qc_lines.extend([
        f"[{init}] Hot/Middle/Cold member counts = {g.groupby('hotcold_group')['member'].nunique().to_dict()}",
        f"[{init}] corr(SHF_resid, sm_avg) = {corr_shf_sm:.8e}",
        f"[{init}] corr(TCC_resid, sm_avg) = {corr_tcc_sm:.8e}",
        f"[{init}] corr(Tmax, SHF_Avg) = {corr_raw_shf:.6f}",
        f"[{init}] corr(Tmax, SHF_resid) = {corr_resid_shf:.6f}",
        f"[{init}] raw SHF Hot-Cold difference = {raw_diff:.6f}",
        f"[{init}] SHF_resid Hot-Cold difference = {resid_diff:.6f}",
        f"[{init}] interpretation = {interpretation}",
    ])
    print(f"[Stage-I audit] init={init}")
    print("  grouping: Stage-I Hot/Middle/Cold = 17/17/17")
    print(f"  corr(Tmax, raw SHF)={corr_raw_shf:.3f}")
    print(f"  corr(Tmax, SHF_resid)={corr_resid_shf:.3f}")
    print(f"  corr(SHF_resid, SM)={corr_shf_sm:.3e}")
    print(f"  raw SHF Hot-Cold={raw_diff:.3g}")
    print(f"  SHF_resid Hot-Cold={resid_diff:.3g}")
    print(f"  interpretation={interpretation}")

qc_stagei_path = f"{STAGEI_AUDIT_SUBDIRS['qc']}/qc_stageI_raw_resid_consistency_audit.txt"
assert_stagei_audit_output_path(qc_stagei_path)
with open(qc_stagei_path, "w", encoding="utf-8") as f:
    f.write("\n".join(stagei_qc_lines))

required_stagei_tables = [table_stagei_member, table_stagei_group_summary, table_stagei_scatter, table_stagei_consistency]
required_stagei_pngs = []
for init in STAGEI_INIT_ORDER:
    required_stagei_pngs.extend([
        f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_SHF_raw_resid_scatter_audit_{init}.png",
        f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_slim_predictor_scatter_audit_{init}.png",
        f"{STAGEI_AUDIT_SUBDIRS['figures']}/fig_stageI_raw_vs_resid_group_contrast_{init}.png",
    ])
for out_path in [*required_stagei_tables, *required_stagei_pngs, qc_stagei_path]:
    assert_stagei_audit_output_path(out_path)
    if not os.path.exists(out_path):
        raise RuntimeError(f"Stage-I raw/residual consistency audit output missing: {out_path}")
print(f"[Stage-I audit] outputs written under {STAGEI_AUDIT_DIR}")

# ============================================================
# Cell 12: Paper-style process figures from exported raw-timeseries and MLR-aligned tables
# ============================================================

# Standalone-safe bootstrap for Cell 12 after notebook/kernel restart.
# These fallbacks let this cell re-read already exported CSVs/figures without
# requiring Cell 1's audit bootstrap to have run in the current kernel.
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

if "FULL_INIT_LIST" not in globals():
    FULL_INIT_LIST = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]

if "MAIN_INIT" not in globals():
    MAIN_INIT = "2023-06-12"

if "CONTRAST_INIT" not in globals():
    CONTRAST_INIT = "2023-06-08"

if "AUX_INITS" not in globals():
    AUX_INITS = ["2023-06-05", "2023-06-01"]

if "EXP_OUT_DIR" not in globals():
    EXP_OUT_DIR = "/data1/huangy/fig6/NC/v12"

if "FOUR_INIT_ROOT" not in globals():
    FOUR_INIT_ROOT = f"{EXP_OUT_DIR}/four_init_full_pipeline_NCVIconcurrent"

if "PROCESS_TS_ROOT" not in globals():
    PROCESS_TS_ROOT = "/data1/huangy/fig6/NC/v11/hotcold_stage_group_timeseries_tables_multiinit_totalfig6"

if "PROCESS_TS_TABLE_DIR" not in globals():
    PROCESS_TS_TABLE_DIR = f"{PROCESS_TS_ROOT}/tables"

if "PROCESS_TS_FIG_DIR" not in globals():
    PROCESS_TS_FIG_DIR = f"{PROCESS_TS_ROOT}/paper_theme_timeseries"

if "STAGEI_AUDIT_DIR" not in globals():
    STAGEI_AUDIT_DIR = f"{FOUR_INIT_ROOT}/optional_stageI_raw_resid_consistency_audit"

if "STAGEI_AUDIT_SUBDIRS" not in globals():
    STAGEI_AUDIT_SUBDIRS = {
        "tables": f"{STAGEI_AUDIT_DIR}/tables",
        "figures": f"{STAGEI_AUDIT_DIR}/figures",
        "qc": f"{STAGEI_AUDIT_DIR}/qc",
    }

if "STAGEI_GROUP_FILE" not in globals():
    STAGEI_GROUP_FILE = "/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv"

if "set_audit_cell_name" not in globals():
    def set_audit_cell_name(name):
        print(f"[audit bootstrap] {name}")

if "assert_four_init_output" not in globals():
    def assert_four_init_output(path):
        root = os.path.abspath(FOUR_INIT_ROOT)
        target = os.path.abspath(path)
        if os.path.commonpath([root, target]) != root:
            raise RuntimeError(f"Output is outside FOUR_INIT_ROOT: {path}")

set_audit_cell_name("Cell 12: paper process figures")
print("\n" + "=" * 80)
print("Cell 12: Paper-style process figures from exported raw-timeseries and MLR-aligned tables")
print("=" * 80)
import matplotlib.dates as mdates

PAPER_PROCESS_ROOT = f"{FOUR_INIT_ROOT}/optional_paper_process_figures"
PAPER_PROCESS_SUBDIRS = {
    "figures": f"{PAPER_PROCESS_ROOT}/figures",
    "tables": f"{PAPER_PROCESS_ROOT}/tables",
    "qc": f"{PAPER_PROCESS_ROOT}/qc",
}


def assert_paper_process_output_path(path):
    if not str(path).startswith(f"{PAPER_PROCESS_ROOT}/"):
        raise RuntimeError(f"Paper process figure output path is outside FOUR_INIT_ROOT: {path}")


assert_paper_process_output_path(PAPER_PROCESS_ROOT + "/")
for _paper_process_dir in PAPER_PROCESS_SUBDIRS.values():
    assert_paper_process_output_path(_paper_process_dir + "/")
    os.makedirs(_paper_process_dir, exist_ok=True)

RAW_TS_TABLE = f"{PROCESS_TS_TABLE_DIR}/daily_timeseries_all_factors.csv"
RAW_MEMBER_STAGE_VALUES = f"{PROCESS_TS_TABLE_DIR}/member_stage_values_for_tests.csv"
RAW_GROUP_DIFF = f"{PROCESS_TS_TABLE_DIR}/group_difference_summary.csv"
PROCESS_STAGE_MEAN = f"{PROCESS_TS_TABLE_DIR}/stage_mean_summary.csv"
PROCESS_MEMBER_LEVEL_TESTS = f"{PROCESS_TS_TABLE_DIR}/member_level_group_difference_tests.csv"
PROCESS_GROUP_AUDIT = f"{PROCESS_TS_TABLE_DIR}/group_member_audit.csv"
PROCESS_MULTIINIT_MEAN_SUMMARY = f"{PROCESS_TS_TABLE_DIR}/multiinit_mean_hot17_minus_cold17_summary.csv"
PROCESS_HOTCOLD_MAIN_SUMMARY = f"{PROCESS_TS_TABLE_DIR}/hot17_minus_cold17_by_init_main_summary.csv"
PROCESS_TS_QC_SUMMARY = f"{PROCESS_TS_TABLE_DIR}/qc_multiinit_total_fig6_summary.txt"
STAGEI_AUDIT_MEMBER = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_raw_resid_member_audit.csv"
STAGEI_AUDIT_GROUP_DIFF = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_raw_resid_group_difference_summary.csv"
STAGEI_AUDIT_SCATTER = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_raw_resid_scatter_stats.csv"
STAGEI_FIG5_CONSISTENCY = f"{STAGEI_AUDIT_SUBDIRS['tables']}/stageI_fig5_vs_mlr_scalar_consistency.csv"
SLIM_MODEL_SUMMARY = f"{FOUR_INIT_ROOT}/tables/table_slim_model_summary.csv"
SLIM_FITTED_MEMBERS = f"{FOUR_INIT_ROOT}/tables/table_slim_fitted_members.csv"
SLIM_LMG_LONG = f"{FOUR_INIT_ROOT}/tables/table_slim_lmg_long.csv"
SLIM_COEF_TABLE = f"{FOUR_INIT_ROOT}/tables/table_slim_model_coefficients.csv"
NCVI_STAGE_VALUES = f"{FOUR_INIT_ROOT}/tables/table_ncvi_concurrent_stage_values.csv"
TCC_V8_ROOT = FOUR_INIT_ROOT
TCC_TABLE_DIR = f"{FOUR_INIT_ROOT}/tables"
TCC_SOURCE_TABLE = f"{TCC_TABLE_DIR}/table_stage_member_raw_factors_TCC.csv"
TCC_DAILY_MEMBER_FILE = f"{TCC_TABLE_DIR}/table_daily_member_tcc_SHF.csv"
TCC_ERA5_S2S_COMPARE_FILE = f"{TCC_TABLE_DIR}/table_daily_tcc_era5_s2s_compare.csv"
HOTCOLD_GROUP_FILE = STAGEI_GROUP_FILE
PAPER_PROCESS_FIG_DIR = f"{FOUR_INIT_ROOT}/figures/paper_process_figures"
PROCESS_FIG_MANIFEST = f"{FOUR_INIT_ROOT}/tables/table_paper_process_figure_manifest.csv"
PROCESS_FIG_QC = f"{FOUR_INIT_ROOT}/qc/qc_paper_process_figures_from_timeseries.txt"

PAPER_PROCESS_INPUTS = {
    "raw_daily_timeseries": RAW_TS_TABLE,
    "raw_member_stage_values": RAW_MEMBER_STAGE_VALUES,
    "raw_group_difference_summary": RAW_GROUP_DIFF,
    "process_stage_mean_summary": PROCESS_STAGE_MEAN,
    "process_member_level_group_difference_tests": PROCESS_MEMBER_LEVEL_TESTS,
    "process_group_member_audit": PROCESS_GROUP_AUDIT,
    "process_multiinit_mean_summary": PROCESS_MULTIINIT_MEAN_SUMMARY,
    "process_hotcold_main_summary": PROCESS_HOTCOLD_MAIN_SUMMARY,
    "process_timeseries_qc_summary": PROCESS_TS_QC_SUMMARY,
    "process_theme_figure_dir": PROCESS_TS_FIG_DIR,
    "stageI_audit_member": STAGEI_AUDIT_MEMBER,
    "stageI_audit_group_difference": STAGEI_AUDIT_GROUP_DIFF,
    "stageI_audit_scatter": STAGEI_AUDIT_SCATTER,
    "stageI_fig5_consistency": STAGEI_FIG5_CONSISTENCY,
    "slim_model_summary": SLIM_MODEL_SUMMARY,
    "slim_fitted_members": SLIM_FITTED_MEMBERS,
    "slim_lmg_long": SLIM_LMG_LONG,
    "slim_model_coefficients": SLIM_COEF_TABLE,
    "ncvi_stage_values": NCVI_STAGE_VALUES,
    "tcc_v8_root": TCC_V8_ROOT,
    "tcc_source_table": TCC_SOURCE_TABLE,
    "tcc_daily_member_file": TCC_DAILY_MEMBER_FILE,
    "tcc_era5_s2s_compare_file": TCC_ERA5_S2S_COMPARE_FILE,
    "hotcold_group_file": HOTCOLD_GROUP_FILE,
}
for _name, _path in PAPER_PROCESS_INPUTS.items():
    if not os.path.exists(_path):
        raise RuntimeError(f"Required input for Cell 12 paper process figures is missing ({_name}): {_path}. Please run the upstream Cell that generates this table first.")

assert_four_init_output(PAPER_PROCESS_FIG_DIR + "/")
os.makedirs(PAPER_PROCESS_FIG_DIR, exist_ok=True)

PROCESS_FIG_INIT_ORDER = ["2023-06-12", "2023-06-08", "2023-06-05", "2023-06-01"]
PROCESS_MAIN_INIT = "2023-06-12"
PROCESS_FIG_CAPTION_NOTE = (
    "Figure 5 is Stage-I-grouped and Stage-I-focused, with 2023-06-18 retained as a transition/reference day. "
    "Figure 6 is Total-grouped and covers 2023-06-14 to 2023-06-24, with the Stage-I/Stage-II boundary shown at 2023-06-17 12:00. "
    "TCC is read from v12 four-init Cell 5 outputs and is available for all four initializations. "
    "NCVI daily panels are process diagnostics and are distinct from the Cell 7 NCVI_concurrent stage scalar used in MLR."
)


def _copy_process_theme_figure(src_path, out_path, required, missing_files, copied_files):
    if not os.path.exists(src_path):
        missing_files.append(src_path)
        if required:
            raise RuntimeError(f"Required main-text process figure source is missing: {src_path}")
        return False
    assert_four_init_output(out_path)
    shutil.copy2(src_path, out_path)
    copied_files.append(out_path)
    return True


process_manifest_rows = []
process_missing_figures = []
process_copied_figures = []
for _process_init in PROCESS_FIG_INIT_ORDER:
    _required_main = _process_init == PROCESS_MAIN_INIT
    _role_suffix = "main_text" if _required_main else "supplementary"
    for _fig_no, _src_stem, _out_stem in [
        (
            "Figure 5",
            f"fig5_stageI_grouped_land_surface_thermal_evolution_init_{_process_init}",
            (
                f"Figure_5_stageI_land_surface_thermal_evolution_init_{_process_init}"
                if _required_main
                else f"FigS5_stageI_land_surface_thermal_evolution_init_{_process_init}"
            ),
        ),
        (
            "Figure 6",
            f"fig6_total_process_chain_precip_to_tmax_init_{_process_init}",
            (
                f"Figure_6_total_process_chain_precip_to_tmax_init_{_process_init}"
                if _required_main
                else f"FigS6_total_process_chain_precip_to_tmax_init_{_process_init}"
            ),
        ),
    ]:
        _source_png = f"{PROCESS_TS_FIG_DIR}/{_src_stem}.png"
        _source_pdf = f"{PROCESS_TS_FIG_DIR}/{_src_stem}.pdf"
        _output_png = f"{PAPER_PROCESS_FIG_DIR}/{_out_stem}.png"
        _output_pdf = f"{PAPER_PROCESS_FIG_DIR}/{_out_stem}.pdf"
        _copied_png = _copy_process_theme_figure(
            _source_png, _output_png, _required_main, process_missing_figures, process_copied_figures
        )
        _copied_pdf = _copy_process_theme_figure(
            _source_pdf, _output_pdf, _required_main, process_missing_figures, process_copied_figures
        )
        process_manifest_rows.append({
            "figure_role": f"{_fig_no} {_role_suffix}",
            "init_date": _process_init,
            "source_png": _source_png,
            "source_pdf": _source_pdf,
            "output_png": _output_png,
            "output_pdf": _output_pdf,
            "copied_png": bool(_copied_png),
            "copied_pdf": bool(_copied_pdf),
            "caption_note": PROCESS_FIG_CAPTION_NOTE,
        })

process_table_row_counts = {}
for _process_table_name, _process_table_path in {
    "daily_timeseries_all_factors.csv": RAW_TS_TABLE,
    "stage_mean_summary.csv": PROCESS_STAGE_MEAN,
    "group_difference_summary.csv": RAW_GROUP_DIFF,
    "member_stage_values_for_tests.csv": RAW_MEMBER_STAGE_VALUES,
    "member_level_group_difference_tests.csv": PROCESS_MEMBER_LEVEL_TESTS,
    "group_member_audit.csv": PROCESS_GROUP_AUDIT,
}.items():
    _tmp_process_df = pd.read_csv(_process_table_path)
    process_table_row_counts[_process_table_name] = int(len(_tmp_process_df))

_process_daily_for_qc = pd.read_csv(RAW_TS_TABLE)
_process_daily_init_col = "init" if "init" in _process_daily_for_qc.columns else "init_date"
_process_daily_factor_col = "factor"
_process_factor_availability = (
    _process_daily_for_qc.groupby(_process_daily_init_col)[_process_daily_factor_col]
    .apply(lambda s: ", ".join(sorted(map(str, pd.unique(s)))))
    .to_dict()
)
_process_tcc_availability = {}
_process_ncvi_availability = {}
for _process_init in PROCESS_FIG_INIT_ORDER:
    _daily_init = _process_daily_for_qc[_process_daily_for_qc[_process_daily_init_col].astype(str) == _process_init]
    _factors = set(_daily_init[_process_daily_factor_col].astype(str)) if not _daily_init.empty else set()
    _process_tcc_availability[_process_init] = "TCC_Avg" in _factors
    _process_ncvi_availability[_process_init] = any("NCVI" in f for f in _factors)

assert_four_init_output(PROCESS_FIG_MANIFEST)
pd.DataFrame(process_manifest_rows).to_csv(PROCESS_FIG_MANIFEST, index=False)

_process_export_qc_lines = [
    "Paper process figures copied/registered from multi-init time-series outputs.",
    f"PROCESS_TS_ROOT = {PROCESS_TS_ROOT}",
    f"PROCESS_TS_TABLE_DIR = {PROCESS_TS_TABLE_DIR}",
    f"PROCESS_TS_FIG_DIR = {PROCESS_TS_FIG_DIR}",
    f"PAPER_PROCESS_FIG_DIR = {PAPER_PROCESS_FIG_DIR}",
    f"TCC source table path = {TCC_SOURCE_TABLE}",
    f"TCC daily member file path = {TCC_DAILY_MEMBER_FILE}",
    f"TCC ERA5/S2S compare file path = {TCC_ERA5_S2S_COMPARE_FILE}",
    f"valid init list = {PROCESS_FIG_INIT_ORDER}",
    f"missing process figure files = {process_missing_figures}",
    f"copied figure files = {process_copied_figures}",
    *[f"row count check for {k} = {v}" for k, v in process_table_row_counts.items()],
    *[f"factor availability by init [{k}] = {v}" for k, v in _process_factor_availability.items()],
    *[f"TCC availability by init [{k}] = {v}" for k, v in _process_tcc_availability.items()],
    *[f"NCVI anomaly availability by init [{k}] = {v}" for k, v in _process_ncvi_availability.items()],
    PROCESS_FIG_CAPTION_NOTE,
]
assert_four_init_output(PROCESS_FIG_QC)
with open(PROCESS_FIG_QC, "w", encoding="utf-8") as f:
    f.write("\n".join(_process_export_qc_lines))


def norm_stage_label(x):
    s = str(x)
    if s in ["Stage-I", "Stage-I_dry", "Stage I", "stageI", "stage1"]:
        return "Stage-I"
    if s in ["Stage-II", "Stage-II_wet", "Stage II", "stageII", "stage2"]:
        return "Stage-II"
    if s in ["Total", "total"]:
        return "Total"
    return s


PAPER_PROCESS_INIT_ORDER = FULL_INIT_LIST
PAPER_PROCESS_INTENDED_USE = {
    MAIN_INIT: "main text",
    CONTRAST_INIT: "appendix/supplementary",
    **{init: "appendix/supplementary" for init in AUX_INITS},
}
PAPER_PROCESS_GROUP_ORDER = ["Hot17", "Middle17", "Cold17"]
PAPER_PROCESS_GROUP_COLORS = {"Hot17": "firebrick", "Middle17": "#6c8ebf", "Cold17": "seagreen"}
PAPER_PROCESS_DISPLAY = {
    "Tmax": "Tmax",
    "y_tmax": "Stage-I Tmax",
    "sm_avg": "SM",
    "SHF_Avg": "SHF",
    "SHF_resid": "SHF resid",
    "LSM_SHF_partial_centered": "SM-SHF contribution to Tmax (°C)",
    "SM_SHF_partial_fitted_Tmax": "SM-SHF partial fitted Tmax (°C)",
    "NCHN_tp": "TP",
    "z500_anom_NCHN": "Z500 anomaly",
    "TCC_Avg": "TCC",
    "TCC_resid": "TCC resid",
    "NCVI_concurrent": "NCVI anomaly",
    "NCVI_conc": "NCVI anomaly",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
    "SM_resid": "SM resid",
}


def _paper_find_col(df, candidates, desc):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find {desc}. Tried {candidates}; available columns={list(df.columns)}")


def _paper_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _paper_corr(x, y):
    xx = pd.to_numeric(pd.Series(x), errors="coerce")
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 3:
        return np.nan
    return float(pearsonr(xx[mask], yy[mask])[0])


def _format_daily_date_axis(ax, start_date, end_date):
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ticks = pd.date_range(start_date, end_date, freq="D")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{d.month}.{d.day}" for d in ticks], rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=8.5, rotation=30)


def _paper_add_regression(ax, x, y):
    xx = pd.to_numeric(pd.Series(x), errors="coerce")
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 3:
        return np.nan, np.nan, np.nan, int(mask.sum())
    slope, intercept = np.polyfit(xx[mask].values.astype(float), yy[mask].values.astype(float), 1)
    grid = np.linspace(float(xx[mask].min()), float(xx[mask].max()), 80)
    ax.plot(grid, intercept + slope * grid, color="black", lw=1.1, alpha=0.75)
    r = float(pearsonr(xx[mask], yy[mask])[0])
    return r, r * r, float(slope), int(mask.sum())


def _paper_linear_stats(x, y):
    xx = pd.to_numeric(pd.Series(x), errors="coerce")
    yy = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 3:
        return np.nan, np.nan, np.nan, int(mask.sum())
    slope, _intercept = np.polyfit(xx[mask].values.astype(float), yy[mask].values.astype(float), 1)
    r = float(pearsonr(xx[mask], yy[mask])[0])
    return r, r * r, float(slope), int(mask.sum())


def _paper_raw_timeseries_panel(ax, raw_df, init, grouping_stage, factor, start_date, end_date, title, ylabel):
    g = raw_df[
        (raw_df["_init"] == init)
        & (raw_df["_grouping_stage"] == grouping_stage)
        & (raw_df["_factor"] == factor)
        & (raw_df["_date"] >= pd.Timestamp(start_date))
        & (raw_df["_date"] <= pd.Timestamp(end_date))
    ].copy().sort_values("_date")
    if g.empty:
        raise RuntimeError(f"Cell 12 raw time-series panel has no data for init={init}, stage={grouping_stage}, factor={factor}")
    era5_vals = pd.to_numeric(g["era5"], errors="coerce")
    era5_plotted = not era5_vals.isna().all()
    if era5_plotted:
        ax.plot(g["_date"], era5_vals, color="black", lw=2.0, ls="-", marker="o", ms=4, label="ERA5 Obs")
    else:
        ax.text(0.02, 0.08, "ERA5 all NaN", transform=ax.transAxes, fontsize=8, color="0.35")
    ax.plot(g["_date"], pd.to_numeric(g["hot17_mean"], errors="coerce"), color="firebrick", lw=1.8, ls="-.", marker="^", ms=4, label="Hot17 Mean")
    ax.plot(g["_date"], pd.to_numeric(g["middle17_mean"], errors="coerce"), color="#1f77b4", lw=1.9, ls="--", marker="s", ms=4, label="Middle17 Mean")
    ax.plot(g["_date"], pd.to_numeric(g["cold17_mean"], errors="coerce"), color="seagreen", lw=1.8, ls="-.", marker="v", ms=4, label="Cold17 Mean")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    _format_daily_date_axis(ax, start_date, end_date)
    ax.grid(alpha=0.25)
    return era5_plotted


def _paper_stagei_scatter(ax, g, x_col, y_col, title):
    for group in PAPER_PROCESS_GROUP_ORDER:
        gg = g[g["hotcold_group"] == group]
        ax.scatter(gg[x_col], gg[y_col], s=24, alpha=0.70, color=PAPER_PROCESS_GROUP_COLORS[group], edgecolor="white", linewidth=0.35, label=group)
    r, r2, slope, n = _paper_add_regression(ax, g[x_col], g[y_col])
    ax.set_title(title, fontsize=10)
    ax.text(
        0.03,
        0.97,
        f"r={r:.2f}\nR²={r2:.2f}\nslope={slope:.2f}\nn={n}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
    )
    ax.set_xlabel(PAPER_PROCESS_DISPLAY.get(x_col, x_col))
    ax.set_ylabel(PAPER_PROCESS_DISPLAY.get(y_col, y_col))
    ax.grid(alpha=0.25)
    return r, r2, slope, n



def _paper_stagei_coefficients(coef_df, init, stagei_df):
    stage_norm = coef_df["stage"].map(norm_stage_label) if "stage" in coef_df.columns else pd.Series([], dtype=str)
    init_col = _paper_find_col(coef_df, ["init_date", "init"], "slim coefficient init column")
    predictor_col = _paper_find_col(coef_df, ["predictor", "factor", "variable"], "slim coefficient predictor column")
    coef_g = coef_df[(coef_df[init_col].astype(str) == init) & (stage_norm == "Stage-I")].copy()
    coef_g[predictor_col] = coef_g[predictor_col].astype(str)
    coef_col = "coef" if "coef" in coef_g.columns else None
    std_coef_col = "standardized_coef" if "standardized_coef" in coef_g.columns else None
    raw_coef_map = dict(zip(coef_g[predictor_col], pd.to_numeric(coef_g[coef_col], errors="coerce"))) if coef_col is not None else {}
    std_coef_map = dict(zip(coef_g[predictor_col], pd.to_numeric(coef_g[std_coef_col], errors="coerce"))) if std_coef_col is not None else {}
    raw_ok = all(k in raw_coef_map and pd.notna(raw_coef_map[k]) for k in ["sm_avg", "SHF_resid", "const"])
    std_ok = all(k in std_coef_map and pd.notna(std_coef_map[k]) for k in ["sm_avg", "SHF_resid"])
    common = {
        "target_space": "absolute Stage-I member mean Tmax in °C",
        "target_variable": "Stage-I member mean Tmax",
        "target_unit": "°C",
        "target_is_standardized": False,
        "target_space_basis": 'Slim MLR uses y = g_model["y_tmax"]; y_tmax is Stage-I member mean Tmax in °C; y_tmax was not standardized; Fig.5 contribution uses coefficients from table_slim_model_coefficients.csv.',
        "predictors_standardized_in_fit": bool(std_coef_col is not None),
        "original_fit_predictor_space": "standardized predictors" if std_coef_col is not None else "raw predictors",
        "intercept_included": "const" in set(coef_g[predictor_col]),
        "sm_mean": float(stagei_df["sm_avg"].mean()),
        "sm_std": float(stagei_df["sm_avg"].std(ddof=0)),
        "shf_resid_mean": float(stagei_df["SHF_resid"].mean()),
        "shf_resid_std": float(stagei_df["SHF_resid"].std(ddof=0)),
    }
    if raw_ok:
        return {
            **common,
            "beta_SM": float(raw_coef_map["sm_avg"]),
            "beta_SHF": float(raw_coef_map["SHF_resid"]),
            "intercept": float(raw_coef_map["const"]),
            "predictor_scaling": "raw",
            "coefficient_space_used_for_Fig5": "raw-equivalent coefficients",
            "coefficient_type_used": "raw-equivalent coefficients from table_slim_model_coefficients.csv",
        }
    if std_ok:
        intercept = float(raw_coef_map.get("const", 0.0)) if pd.notna(raw_coef_map.get("const", np.nan)) else 0.0
        return {
            **common,
            "beta_SM": float(std_coef_map["sm_avg"]),
            "beta_SHF": float(std_coef_map["SHF_resid"]),
            "intercept": intercept,
            "predictor_scaling": "standardized",
            "coefficient_space_used_for_Fig5": "standardized coefficients",
            "coefficient_type_used": "standardized coefficients from table_slim_model_coefficients.csv",
        }
    raise RuntimeError(f"Could not find Stage-I sm_avg and SHF_resid coefficients for init={init} in {SLIM_COEF_TABLE}")


def _paper_add_lsm_shf_contribution(stagei_df, coef_info):
    out = stagei_df.copy()
    if coef_info["predictor_scaling"] == "raw":
        out["X_SM_used"] = out["sm_avg"].astype(float)
        out["X_SHF_resid_used"] = out["SHF_resid"].astype(float)
        out["LSM_SHF_partial_raw"] = coef_info["beta_SM"] * out["X_SM_used"] + coef_info["beta_SHF"] * out["X_SHF_resid_used"]
        out["LSM_SHF_partial_centered"] = (
            coef_info["beta_SM"] * (out["X_SM_used"] - coef_info["sm_mean"])
            + coef_info["beta_SHF"] * (out["X_SHF_resid_used"] - coef_info["shf_resid_mean"])
        )
    else:
        out["X_SM_used"] = (out["sm_avg"].astype(float) - coef_info["sm_mean"]) / (coef_info["sm_std"] + 1e-12)
        out["X_SHF_resid_used"] = (out["SHF_resid"].astype(float) - coef_info["shf_resid_mean"]) / (coef_info["shf_resid_std"] + 1e-12)
        out["LSM_SHF_partial_raw"] = coef_info["beta_SM"] * out["X_SM_used"] + coef_info["beta_SHF"] * out["X_SHF_resid_used"]
        out["LSM_SHF_partial_centered"] = out["LSM_SHF_partial_raw"]
    out["LSM_SHF_with_intercept"] = coef_info["intercept"] + out["LSM_SHF_partial_raw"]
    return out


def _paper_stagei_mean_state_baseline(coef_df, init, stagei_df):
    stage_norm = coef_df["stage"].map(norm_stage_label) if "stage" in coef_df.columns else pd.Series([], dtype=str)
    init_col = _paper_find_col(coef_df, ["init_date", "init"], "slim coefficient init column")
    predictor_col = _paper_find_col(coef_df, ["predictor", "factor", "variable"], "slim coefficient predictor column")
    coef_col = _paper_find_col(coef_df, ["coef"], "raw-equivalent slim coefficient column")
    coef_g = coef_df[(coef_df[init_col].astype(str) == init) & (stage_norm == "Stage-I")].copy()
    coef_g[predictor_col] = coef_g[predictor_col].astype(str)
    coef_g[coef_col] = pd.to_numeric(coef_g[coef_col], errors="coerce")
    intercept_rows = coef_g[coef_g[predictor_col] == "const"]
    intercept = float(intercept_rows[coef_col].iloc[0]) if not intercept_rows.empty and pd.notna(intercept_rows[coef_col].iloc[0]) else 0.0
    predictor_rows = coef_g[(coef_g[predictor_col] != "const") & coef_g[coef_col].notna()].copy()
    all_predictors = predictor_rows[predictor_col].astype(str).tolist()
    used_predictors = []
    missing_predictors = []
    baseline = intercept
    for predictor, beta in zip(predictor_rows[predictor_col].astype(str), predictor_rows[coef_col].astype(float)):
        if predictor in stagei_df.columns:
            mean_x = float(pd.to_numeric(stagei_df[predictor], errors="coerce").mean())
            baseline += float(beta) * mean_x
            used_predictors.append(predictor)
        else:
            missing_predictors.append(predictor)
    fallback_warning = bool(missing_predictors or not used_predictors)
    return {
        "MLR_baseline_at_mean_state": float(baseline),
        "baseline_predictor_count": int(len(used_predictors)),
        "baseline_predictor_list": ";".join(used_predictors),
        "baseline_full_predictor_count": int(len(all_predictors)),
        "baseline_full_predictor_list": ";".join(all_predictors),
        "baseline_missing_predictors": ";".join(missing_predictors),
        "baseline_uses_full_slim_predictor_set": bool((not missing_predictors) and len(used_predictors) == len(all_predictors) and len(used_predictors) > 0),
        "baseline_fallback_warning": fallback_warning,
        "baseline_intercept": float(intercept),
    }


def _paper_stagei_shf_resid_fit(stagei_df):
    x = pd.to_numeric(stagei_df["sm_avg"], errors="coerce").astype(float)
    y = pd.to_numeric(stagei_df["SHF_Avg"], errors="coerce").astype(float)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        raise RuntimeError("Cannot fit Stage-I SHF_Avg ~ sm_avg residual bridge with fewer than 3 valid members")
    slope, intercept = np.polyfit(x[mask].values, y[mask].values, 1)
    return float(intercept), float(slope)


def _paper_lsm_shf_daily_panel(ax, raw_df, init, coef_info, shf_intercept, shf_slope, start_date, end_date, title, ylabel, baseline=0.0):
    pieces = []
    for factor in ["sm_avg", "SHF_Avg"]:
        g = raw_df[
            (raw_df["_init"] == init)
            & (raw_df["_grouping_stage"] == "Stage-I")
            & (raw_df["_factor"] == factor)
            & (raw_df["_date"] >= pd.Timestamp(start_date))
            & (raw_df["_date"] <= pd.Timestamp(end_date))
        ].copy()
        if g.empty:
            raise RuntimeError(f"Cell 12 LSM-SHF daily panel has no data for init={init}, factor={factor}")
        pieces.append(g[["_date", "hot17_mean", "middle17_mean", "cold17_mean"]].rename(columns={
            "hot17_mean": f"{factor}_hot17_mean",
            "middle17_mean": f"{factor}_middle17_mean",
            "cold17_mean": f"{factor}_cold17_mean",
        }))
    daily = pieces[0].merge(pieces[1], on="_date", how="inner", validate="one_to_one").sort_values("_date")
    line_specs = [
        ("Hot17 Mean", "hot17_mean", "firebrick", "-.", "^"),
        ("Middle17 Mean", "middle17_mean", "#1f77b4", "--", "s"),
        ("Cold17 Mean", "cold17_mean", "seagreen", "-.", "v"),
    ]
    for label, suffix, color, ls, marker in line_specs:
        sm_daily = pd.to_numeric(daily[f"sm_avg_{suffix}"], errors="coerce")
        shf_daily = pd.to_numeric(daily[f"SHF_Avg_{suffix}"], errors="coerce")
        shf_resid_daily = shf_daily - (shf_intercept + shf_slope * sm_daily)
        if coef_info["predictor_scaling"] == "raw":
            lsm_daily = (
                coef_info["beta_SM"] * (sm_daily - coef_info["sm_mean"])
                + coef_info["beta_SHF"] * (shf_resid_daily - coef_info["shf_resid_mean"])
            )
        else:
            sm_daily_std = (sm_daily - coef_info["sm_mean"]) / (coef_info["sm_std"] + 1e-12)
            shf_daily_std = (shf_resid_daily - coef_info["shf_resid_mean"]) / (coef_info["shf_resid_std"] + 1e-12)
            lsm_daily = coef_info["beta_SM"] * sm_daily_std + coef_info["beta_SHF"] * shf_daily_std
        lsm_daily = lsm_daily + float(baseline)
        ax.plot(daily["_date"], lsm_daily, color=color, lw=1.8, ls=ls, marker=marker, ms=4, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    _format_daily_date_axis(ax, start_date, end_date)
    ax.grid(alpha=0.25)
    return True

def _paper_add_panel_labels(axes, labels):
    for ax, label in zip(np.ravel(axes), labels):
        ax.text(-0.11, 1.06, label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")


def _paper_lmg_col(df):
    return _paper_find_col(df, ["LMG_percent", "LMG_importance_percent", "LMG_relative_importance_pct", "importance_percent", "LMG"], "LMG percent column")


def _paper_norm_hotcold_group(x):
    s = str(x)
    if s in ["Hot17", "Hot", "hot", "hot17"]:
        return "Hot17"
    if s in ["Middle17", "Middle", "middle", "middle17", "Mid17", "Mid"]:
        return "Middle17"
    if s in ["Cold17", "Cold", "cold", "cold17"]:
        return "Cold17"
    return s


# Load only exported CSV products; Cell 12 does not read GRIB/NC and does not recompute model diagnostics.
df_paper_raw_ts = pd.read_csv(RAW_TS_TABLE)
df_paper_stagei_member = pd.read_csv(STAGEI_AUDIT_MEMBER)
df_paper_stagei_group_diff = pd.read_csv(STAGEI_AUDIT_GROUP_DIFF)
df_paper_stagei_scatter = pd.read_csv(STAGEI_AUDIT_SCATTER)
df_paper_fig5_consistency = pd.read_csv(STAGEI_FIG5_CONSISTENCY)
df_paper_slim_lmg = pd.read_csv(SLIM_LMG_LONG)
df_paper_slim_coef = pd.read_csv(SLIM_COEF_TABLE)
df_paper_ncvi_stage = pd.read_csv(NCVI_STAGE_VALUES)
df_paper_hotcold_groups = pd.read_csv(HOTCOLD_GROUP_FILE)

raw_init_col = _paper_find_col(df_paper_raw_ts, ["init", "init_date"], "raw daily init column")
raw_stage_col = _paper_find_col(df_paper_raw_ts, ["grouping_stage", "stage"], "raw daily grouping stage column")
raw_factor_col = _paper_find_col(df_paper_raw_ts, ["factor"], "raw daily factor column")
raw_date_col = _paper_find_col(df_paper_raw_ts, ["date", "valid_date", "time"], "raw daily date column")

required_raw_ts_cols = ["era5", "hot17_mean", "middle17_mean", "cold17_mean"]
missing_raw_ts_cols = [c for c in required_raw_ts_cols if c not in df_paper_raw_ts.columns]
if missing_raw_ts_cols:
    raise RuntimeError(
        "RAW_TS_TABLE must contain precomputed BJT daily paper-panel columns. "
        f"Missing columns={missing_raw_ts_cols}; available columns={list(df_paper_raw_ts.columns)}"
    )

df_paper_raw_ts = df_paper_raw_ts.copy()
df_paper_raw_ts["_init"] = df_paper_raw_ts[raw_init_col].astype(str)
df_paper_raw_ts["_grouping_stage"] = df_paper_raw_ts[raw_stage_col].map(norm_stage_label)
df_paper_raw_ts["_factor"] = df_paper_raw_ts[raw_factor_col].astype(str)
df_paper_raw_ts["_date"] = pd.to_datetime(df_paper_raw_ts[raw_date_col])
for _raw_line_col in required_raw_ts_cols:
    df_paper_raw_ts[_raw_line_col] = pd.to_numeric(df_paper_raw_ts[_raw_line_col], errors="coerce")

if df_paper_raw_ts[["hot17_mean", "middle17_mean", "cold17_mean"]].isna().all().all():
    raise RuntimeError("RAW_TS_TABLE paper line columns produced all-NaN values.")

if "stage" in df_paper_stagei_member.columns:
    df_paper_stagei_member["_stage"] = df_paper_stagei_member["stage"].map(norm_stage_label)
if "stage" in df_paper_slim_lmg.columns:
    df_paper_slim_lmg["_stage"] = df_paper_slim_lmg["stage"].map(norm_stage_label)
else:
    raise RuntimeError("Cell 12 requires a stage column in SLIM_LMG_LONG")
slim_lmg_init_col = _paper_find_col(df_paper_slim_lmg, ["init_date", "init"], "slim LMG init column")
slim_lmg_factor_col = _paper_find_col(df_paper_slim_lmg, ["predictor", "factor"], "slim LMG factor column")
slim_lmg_pct_col = _paper_lmg_col(df_paper_slim_lmg)
df_paper_slim_lmg["_init"] = df_paper_slim_lmg[slim_lmg_init_col].astype(str)

hotcold_init_col = _paper_find_col(df_paper_hotcold_groups, ["init_date", "init"], "Hot/Cold group init column")
hotcold_stage_col = _paper_find_col(df_paper_hotcold_groups, ["stage", "grouping_stage"], "Hot/Cold group stage column")
hotcold_member_col = _paper_find_col(df_paper_hotcold_groups, ["member", "number", "ensemble_member"], "Hot/Cold group member column")
hotcold_group_col = _paper_find_col(df_paper_hotcold_groups, ["hotcold_group", "group"], "Hot/Cold group label column")
df_paper_hotcold_groups["_init"] = df_paper_hotcold_groups[hotcold_init_col].astype(str)
df_paper_hotcold_groups["_stage"] = df_paper_hotcold_groups[hotcold_stage_col].map(norm_stage_label)
df_paper_hotcold_groups["_member_key"] = df_paper_hotcold_groups[hotcold_member_col].astype(str)
df_paper_hotcold_groups["_group"] = df_paper_hotcold_groups[hotcold_group_col].map(_paper_norm_hotcold_group)
df_paper_hotcold_unique = df_paper_hotcold_groups[["_init", "_stage", "_member_key", "_group"]].drop_duplicates()
for init in PAPER_PROCESS_INIT_ORDER:
    for stage in ["Stage-I", "Stage-II", "Total"]:
        group_check = df_paper_hotcold_unique[(df_paper_hotcold_unique["_init"] == init) & (df_paper_hotcold_unique["_stage"] == stage)]
        counts = group_check.groupby("_group")["_member_key"].nunique().to_dict()
        if counts != {"Hot17": 17, "Middle17": 17, "Cold17": 17}:
            raise RuntimeError(f"Cell 12 HOTCOLD_GROUP_FILE count check failed for {init}/{stage}: {counts}")

fig5_summary_rows = []
fig6_summary_rows = []
fig5_lsm_shf_audit_rows = []
fig5_lsm_shf_summary_rows = []
paper_process_qc = [
    "Cell 12: Paper-style process figures from exported raw-timeseries and MLR-aligned tables",
    f"raw timeseries source path = {RAW_TS_TABLE}",
    f"RAW_TS_TABLE = {RAW_TS_TABLE}",
    "Cell 12 raw panels read precomputed BJT daily values from RAW_TS_TABLE.",
    "ERA5 line is read from RAW_TS_TABLE column: era5.",
    "Hot17 Mean line is read from RAW_TS_TABLE column: hot17_mean.",
    "Cold17 Mean line is read from RAW_TS_TABLE column: cold17_mean.",
    "Middle17 Mean line is read from RAW_TS_TABLE column: middle17_mean.",
    "S2S ensemble mean is not plotted in paper process time-series panels.",
    "Cell 12 does not recompute ERA5 hourly to BJT daily.",
    "Figure polish applied:",
    "- Fig.5 uses Stage-I_dry/Stage-I Hot17 and Cold17 grouping.",
    "- Fig.6 uses Total Hot17 and Cold17 grouping.",
    "- Fig.5 uses 4x2 layout with 6 raw/derived time-series panels and 2 scatter bridge panels.",
    "- Fig.5 LSM-SHF contribution definition audited = True.",
    "- Main Fig.5 LSM-SHF variable = LSM_SHF_partial_centered.",
    "- LSM-SHF partial raw formula = beta_SM * X_SM + beta_SHF * X_SHF_resid.",
    "- LSM-SHF centered formula = beta_SM * (X_SM - mean_X_SM) + beta_SHF * (X_SHF_resid - mean_X_SHF_resid).",
    "- LSM-SHF with-intercept formula = intercept + beta_SM * X_SM + beta_SHF * X_SHF_resid.",
    "- Coefficient source = table_slim_model_coefficients.csv.",
    "- Coefficient stage = Stage-I_dry.",
    "- original_fit_predictor_space = standardized predictors.",
    "- coefficient_space_used_for_Fig5 = raw-equivalent coefficients.",
    "- target_space = absolute Stage-I member mean Tmax in °C.",
    "- y_tmax was not standardized.",
    '- target_space basis = Slim MLR uses y = g_model["y_tmax"]; y_tmax is Stage-I member mean Tmax in °C; Fig.5 contribution uses raw-equivalent coefficients from table_slim_model_coefficients.csv.',
    "- The centered SM–SHF term is a member-level diagnostic linear contribution relative to the per-init Stage-I ensemble-mean predictor state. It should not be interpreted as a strict causal effect.",
    "- Raw uncentered partial is used only for QC, not as main figure variable.",
    "- With-intercept component is used only for QC, not as main figure variable.",
    "- Fig.5 contribution unit label = Linear contribution to Tmax (°C).",
    "- Fig.6 uses a 3x2 vertical layout with 5 raw time-series panels plus one compact note panel.",
    "- Fig.6 note panel contains only grouping/period/boundary/line-style information.",
    "- Fig.6 note panel is unnumbered.",
    "- Fig.6 data panel labels are (a)–(e) only.",
    "- Fig.5 panel (a) = centered SM-SHF partial contribution time series.",
    "- Fig.5 uses process time-series panels: LSM-SHF contribution, SM, Z500 anomaly, SHF, NCVI anomaly, TCC.",
    "- Fig.5 uses 2 Stage-I scatter bridge panels: SM vs Stage-I Tmax, LSM-SHF contribution vs Stage-I Tmax.",
    "- Fig.5 panel (h) = centered SM-SHF partial contribution vs Stage-I Tmax scatter.",
    "- Fig.5 raw SHF scatter removed = True.",
    "- Fig.5 SHF_resid scatter removed = True.",
    "- Fig.5 y-axis for scatter panels = Stage-I Tmax.",
    "- Fig.5 does not include a Tmax time-series panel.",
    "- Fig.6 time range = 2023-06-14 to 2023-06-24.",
    "- Fig.6 uses 5 raw time-series panels: TP, Z500 anomaly, SM, NCVI anomaly, TCC.",
    "- Fig.6 does not include a Tmax time-series panel.",
    "- Fig.6 does not include any MLR/LMG/Johnson/LOFO bar subplot.",
    "- Raw panels have no stage background shading.",
    "- Only a light vertical boundary line is used.",
    "- Daily x-axis labels are shown as M.D, e.g., 6.14, 6.15, without hourly labels.",
    "- Fig.5/Fig.6 time-series group line changed: S2S Ens Mean replaced by Middle17 Mean.",
    "- Middle17 line source column = middle17_mean.",
    "- Fig.5 time-series lines = ERA5 Obs, Hot17 Mean, Middle17 Mean, Cold17 Mean; LSM-SHF contribution panel lines = Hot17 Mean, Middle17 Mean, Cold17 Mean only.",
    "- Fig.6 time-series lines = ERA5 Obs, Hot17 Mean, Middle17 Mean, Cold17 Mean.",
    "- Scatter statistics are shown inside scatter panels, not in long subplot titles.",
    "- No data loading, filtering, statistics, residualization, or MLR/LMG results were changed.",
    "- Additional Fig.5 fitted-component comparison version added = True.",
    "- existing centered Fig.5 retained unchanged.",
    "- new fitted-component Fig.5 is a comparison product.",
    "- MLR_baseline_at_mean_state = intercept + sum(beta_k * mean_X_k) over the full available Stage-I slim-model predictor set.",
    "- SM_SHF_partial_fitted_Tmax = MLR_baseline_at_mean_state + LSM_SHF_partial_centered.",
    "- SM_SHF_partial_fitted_Tmax is a partial fitted component anchored to the full-model mean-state baseline, not the full fitted Tmax itself.",
    "SHF_resid = residual(SHF ~ SM).",
    "Stage-I MLR uses SHF_resid, not raw SHF.",
    "A vertical line denotes the Stage-I/subsequent-stage boundary.",
    f"Stage-I audit source path = {STAGEI_AUDIT_MEMBER}",
    f"Cell 8 LMG source path = {SLIM_LMG_LONG}",
    f"Cell 7 NCVI_concurrent source path = {NCVI_STAGE_VALUES}",
    f"TCC source table path = {TCC_SOURCE_TABLE}",
    f"TCC daily member file path = {TCC_DAILY_MEMBER_FILE}",
    f"TCC ERA5/S2S compare file path = {TCC_ERA5_S2S_COMPARE_FILE}",
    f"Hot/Cold grouping provenance source path = {HOTCOLD_GROUP_FILE}",
    "HOTCOLD_GROUP_FILE is used only for provenance/QC consistency checks; it does not overwrite exported raw time-series or MLR audit group labels.",
    f"valid init list = {FULL_INIT_LIST}",
    "2023-06-12 intended for main text",
    "2023-06-08 intended for appendix/supplementary",
    "2023-06-05 intended for appendix/supplementary",
    "2023-06-01 intended for appendix/supplementary",
    "Fig.5 grouping stage = Stage-I_dry",
    "Fig.5 uses 4x2 layout with 6 raw/derived time-series panels and 2 scatter bridge panels",
    "Fig.5 LSM-SHF contribution definition audited = True",
    "Main Fig.5 LSM-SHF variable = LSM_SHF_partial_centered",
    "LSM-SHF partial raw formula = beta_SM * X_SM + beta_SHF * X_SHF_resid",
    "LSM-SHF centered formula = beta_SM * (X_SM - mean_X_SM) + beta_SHF * (X_SHF_resid - mean_X_SHF_resid)",
    "LSM-SHF with-intercept formula = intercept + beta_SM * X_SM + beta_SHF * X_SHF_resid",
    "Coefficient source = table_slim_model_coefficients.csv",
    "Coefficient stage = Stage-I_dry",
    "original_fit_predictor_space = standardized predictors",
    "coefficient_space_used_for_Fig5 = raw-equivalent coefficients",
    "target_space = absolute Stage-I member mean Tmax in °C",
    "y_tmax was not standardized",
    'target_space basis = Slim MLR uses y = g_model["y_tmax"]; y_tmax is Stage-I member mean Tmax in °C; Fig.5 contribution uses raw-equivalent coefficients from table_slim_model_coefficients.csv',
    "The centered SM–SHF term is a member-level diagnostic linear contribution relative to the per-init Stage-I ensemble-mean predictor state. It should not be interpreted as a strict causal effect",
    "Raw uncentered partial is used only for QC, not as main figure variable",
    "With-intercept component is used only for QC, not as main figure variable",
    "Fig.5 contribution unit label = Linear contribution to Tmax (°C)",
    "Fig.5 panel (a) = centered SM-SHF partial contribution time series",
    "Fig.5 panel (h) = centered SM-SHF partial contribution vs Stage-I Tmax scatter",
    "Fig.5 raw SHF scatter removed = True",
    "Fig.5 SHF_resid scatter removed = True",
    "Fig.5 y-axis for scatter panels = Stage-I Tmax",
    "Fig.5 does not include a Tmax time-series panel",
    "Fig.6 grouping stage = Total",
    "Fig.6 time range = 2023-06-14 to 2023-06-24",
    "Fig.6 uses a 3x2 vertical layout with 5 raw time-series panels plus one compact note panel",
    "Fig.6 note panel contains only grouping/period/boundary/line-style information",
    "Fig.6 note panel is unnumbered",
    "Fig.6 data panel labels are (a)–(e) only",
    "Fig.6 uses 5 raw time-series panels: TP, Z500 anomaly, SM, NCVI anomaly, TCC",
    "Fig.6 does not include a Tmax time-series panel",
    "Fig.6 does not include any MLR/LMG/Johnson/LOFO bar subplot",
    "Raw panels have no stage background shading.",
    "Only a light vertical boundary line is used.",
    "Daily x-axis labels are shown as M.D, e.g., 6.14, 6.15, without hourly labels.",
    "Fig.5/Fig.6 time-series group line changed: S2S Ens Mean replaced by Middle17 Mean",
    "S2S ensemble mean is not plotted in paper process time-series panels",
    "Middle17 line source column = middle17_mean",
    "Fig.5 time-series lines = ERA5 Obs, Hot17 Mean, Middle17 Mean, Cold17 Mean",
    "Fig.6 time-series lines = ERA5 Obs, Hot17 Mean, Middle17 Mean, Cold17 Mean",
    "LSM-SHF contribution panel lines = Hot17 Mean, Middle17 Mean, Cold17 Mean only",
    "Scatter statistics are shown inside scatter panels, not in long subplot titles.",
    "Additional Fig.5 fitted-component comparison version added = True",
    "existing centered Fig.5 retained unchanged",
    "new fitted-component Fig.5 is a comparison product",
    "MLR_baseline_at_mean_state = intercept + sum(beta_k * mean_X_k) over the full available Stage-I slim-model predictor set",
    "SM_SHF_partial_fitted_Tmax = MLR_baseline_at_mean_state + LSM_SHF_partial_centered",
    "SM_SHF_partial_fitted_Tmax is a partial fitted component anchored to the full-model mean-state baseline, not the full fitted Tmax itself",
    "Raw Fig.5/Fig.6 time series are process evolution diagnostics.",
    "MLR uses stage scalar/residual predictors.",
    "Raw SHF and SHF_resid are not interchangeable.",
    "Daily NCVI panels, if shown, are not identical to the MLR NCVI_concurrent stage scalar.",
]

_paper_required_factor_labels = {
    "Tmax": "Tmax",
    "NCHN_tp": "TP",
    "sm_avg": "SM",
    "SHF_Avg": "SHF",
    "z500_anom_NCHN": "Z500 anomaly",
    "NCVI_conc": "NCVI anomaly",
    "TCC_Avg": "TCC",
}
for _init in PAPER_PROCESS_INIT_ORDER:
    _init_factor_set = set(df_paper_raw_ts[df_paper_raw_ts["_init"] == _init]["_factor"].astype(str))
    for _factor, _label in _paper_required_factor_labels.items():
        _available = _factor in _init_factor_set
        paper_process_qc.append(f"[factor availability] init={_init}, factor={_label}, raw_name={_factor}, available={_available}")
        if not _available and _init == MAIN_INIT:
            raise RuntimeError(f"Cell 12 main-process figure factor missing for {MAIN_INIT}: {_factor}")

df_paper_tcc_daily = pd.read_csv(TCC_DAILY_MEMBER_FILE)
if {"init_date", "date", "member"}.issubset(df_paper_tcc_daily.columns):
    df_paper_tcc_daily["init_date"] = df_paper_tcc_daily["init_date"].astype(str)
    df_paper_tcc_daily["date"] = pd.to_datetime(df_paper_tcc_daily["date"], errors="coerce")
    _tcc_window = df_paper_tcc_daily[
        (df_paper_tcc_daily["date"] >= pd.Timestamp("2023-06-14"))
        & (df_paper_tcc_daily["date"] <= pd.Timestamp("2023-06-24"))
    ].copy()
    for _init in PAPER_PROCESS_INIT_ORDER:
        _daily_counts = _tcc_window[_tcc_window["init_date"] == _init].groupby("date")["member"].nunique()
        _all_51 = bool(len(_daily_counts) == 11 and (_daily_counts == 51).all())
        paper_process_qc.append(f"[TCC daily member coverage] init={_init}, dates={len(_daily_counts)}, all_51_members={_all_51}, counts={_daily_counts.to_dict()}")
        if not _all_51 and _init == MAIN_INIT:
            raise RuntimeError(f"Cell 12 main-process figure TCC daily member coverage failed for {MAIN_INIT}: {_daily_counts.to_dict()}")
else:
    raise RuntimeError(f"TCC daily member file missing required columns init_date/date/member: {TCC_DAILY_MEMBER_FILE}")


for init in PAPER_PROCESS_INIT_ORDER:
    for stage in ["Stage-I", "Stage-II", "Total"]:
        group_check = df_paper_hotcold_unique[(df_paper_hotcold_unique["_init"] == init) & (df_paper_hotcold_unique["_stage"] == stage)]
        counts = group_check.groupby("_group")["_member_key"].nunique().to_dict()
        paper_process_qc.append(f"[group count] init={init}, stage={stage}, counts={counts}")
    for fig_stage, fig_label in [("Stage-I", "Fig.5"), ("Total", "Fig.6")]:
        hot_members = df_paper_hotcold_unique[
            (df_paper_hotcold_unique["_init"] == init)
            & (df_paper_hotcold_unique["_stage"] == fig_stage)
            & (df_paper_hotcold_unique["_group"] == "Hot17")
        ]["_member_key"].tolist()
        cold_members = df_paper_hotcold_unique[
            (df_paper_hotcold_unique["_init"] == init)
            & (df_paper_hotcold_unique["_stage"] == fig_stage)
            & (df_paper_hotcold_unique["_group"] == "Cold17")
        ]["_member_key"].tolist()
        paper_process_qc.append(
            f"[{fig_label} member check] init={init}, grouping_stage={fig_stage}, "
            f"Hot17_n={len(hot_members)}, Cold17_n={len(cold_members)}, "
            f"Hot17_members={hot_members}, Cold17_members={cold_members}"
        )

for init in PAPER_PROCESS_INIT_ORDER:
    intended_use = PAPER_PROCESS_INTENDED_USE[init]
    g_stagei = df_paper_stagei_member[df_paper_stagei_member["init_date"].astype(str) == init].copy()
    hotcold_stagei = df_paper_hotcold_unique[(df_paper_hotcold_unique["_init"] == init) & (df_paper_hotcold_unique["_stage"] == "Stage-I")].copy()
    stagei_group_compare = g_stagei.assign(_member_key=g_stagei["member"].astype(str), _audit_group=g_stagei["hotcold_group"].map(_paper_norm_hotcold_group)).merge(
        hotcold_stagei[["_member_key", "_group"]],
        on="_member_key",
        how="left",
        validate="one_to_one",
    )
    if len(stagei_group_compare) != 51 or stagei_group_compare["_group"].isna().any() or not (stagei_group_compare["_audit_group"] == stagei_group_compare["_group"]).all():
        mismatch = stagei_group_compare[stagei_group_compare["_audit_group"] != stagei_group_compare["_group"]][["member", "_audit_group", "_group"]].to_dict("records")
        raise RuntimeError(f"Cell 12 Stage-I audit hotcold_group does not match HOTCOLD_GROUP_FILE for {init}: {mismatch[:10]}")
    total_raw_check = df_paper_raw_ts[(df_paper_raw_ts["_init"] == init) & (df_paper_raw_ts["_grouping_stage"] == "Total")]
    if total_raw_check.empty:
        raise RuntimeError(f"Cell 12 RAW_TS_TABLE has no Total grouped time-series rows for {init}")
    paper_process_qc.append(f"[{init}] Stage-I audit member grouping matches HOTCOLD_GROUP_FILE Stage-I grouping exactly.")
    paper_process_qc.append(f"[{init}] Fig.5 grouping stage = Stage-I_dry; Fig.6 grouping stage = Total; Cell 12 does not regroup members.")
    if len(g_stagei) != 51:
        raise RuntimeError(f"Cell 12 Fig.5 bridge requires 51 Stage-I audit members for {init}, got {len(g_stagei)}")
    required_stagei_cols = ["y_tmax", "sm_avg", "SHF_Avg", "SHF_resid", "hotcold_group"]
    missing_stagei_cols = [c for c in required_stagei_cols if c not in g_stagei.columns]
    if missing_stagei_cols:
        raise RuntimeError(f"Cell 12 Stage-I audit member table missing columns for {init}: {missing_stagei_cols}")
    group_counts = g_stagei.groupby("hotcold_group")["member"].nunique().to_dict()
    if group_counts != {"Hot17": 17, "Middle17": 17, "Cold17": 17}:
        raise RuntimeError(f"Cell 12 Stage-I Hot/Middle/Cold counts failed for {init}: {group_counts}")

    lsm_coef_info = _paper_stagei_coefficients(df_paper_slim_coef, init, g_stagei)
    shf_intercept, shf_slope = _paper_stagei_shf_resid_fit(g_stagei)
    g_stagei = _paper_add_lsm_shf_contribution(g_stagei, lsm_coef_info)
    baseline_info = _paper_stagei_mean_state_baseline(df_paper_slim_coef, init, g_stagei)
    g_stagei["MLR_baseline_at_mean_state"] = baseline_info["MLR_baseline_at_mean_state"]
    g_stagei["SM_SHF_partial_fitted_Tmax"] = g_stagei["MLR_baseline_at_mean_state"] + g_stagei["LSM_SHF_partial_centered"]

    corr_tmax_sm = _paper_corr(g_stagei["sm_avg"], g_stagei["y_tmax"])
    corr_tmax_raw_shf = _paper_corr(g_stagei["SHF_Avg"], g_stagei["y_tmax"])
    corr_tmax_shf_resid = _paper_corr(g_stagei["SHF_resid"], g_stagei["y_tmax"])
    corr_shf_resid_sm = _paper_corr(g_stagei["SHF_resid"], g_stagei["sm_avg"])
    lsm_raw_r, _lsm_raw_r2, _lsm_raw_slope, _lsm_raw_n = _paper_linear_stats(g_stagei["LSM_SHF_partial_raw"], g_stagei["y_tmax"])
    lsm_r, lsm_r2, lsm_slope, lsm_n = _paper_linear_stats(g_stagei["LSM_SHF_partial_centered"], g_stagei["y_tmax"])
    lsm_intercept_r, _lsm_intercept_r2, _lsm_intercept_slope, _lsm_intercept_n = _paper_linear_stats(g_stagei["LSM_SHF_with_intercept"], g_stagei["y_tmax"])
    partial_fitted_r, partial_fitted_r2, partial_fitted_slope, partial_fitted_n = _paper_linear_stats(g_stagei["SM_SHF_partial_fitted_Tmax"], g_stagei["y_tmax"])
    raw_shf_hot_minus_cold = float(g_stagei[g_stagei["hotcold_group"] == "Hot17"]["SHF_Avg"].mean() - g_stagei[g_stagei["hotcold_group"] == "Cold17"]["SHF_Avg"].mean())
    shf_resid_hot_minus_cold = float(g_stagei[g_stagei["hotcold_group"] == "Hot17"]["SHF_resid"].mean() - g_stagei[g_stagei["hotcold_group"] == "Cold17"]["SHF_resid"].mean())
    key_stats = [corr_tmax_sm, corr_tmax_raw_shf, corr_tmax_shf_resid, corr_shf_resid_sm, lsm_raw_r, lsm_intercept_r, lsm_r, lsm_r2, lsm_slope, partial_fitted_r, partial_fitted_r2, partial_fitted_slope, raw_shf_hot_minus_cold, shf_resid_hot_minus_cold]
    if any(pd.isna(v) for v in key_stats):
        raise RuntimeError(f"Cell 12 Fig.5 key stats contain NaN for {init}: {key_stats}")
    consistency_g = df_paper_fig5_consistency[df_paper_fig5_consistency["init_date"].astype(str) == init]
    consistency_status = "ok" if (not consistency_g.empty and consistency_g["consistent_within_tolerance"].fillna(False).all()) else "warning_check_consistency_table"
    fig5_summary_rows.append({
        "init_date": init,
        "corr_tmax_sm": corr_tmax_sm,
        "corr_tmax_raw_shf": corr_tmax_raw_shf,
        "corr_tmax_shf_resid": corr_tmax_shf_resid,
        "corr_shf_resid_sm": corr_shf_resid_sm,
        "raw_shf_hot_minus_cold": raw_shf_hot_minus_cold,
        "shf_resid_hot_minus_cold": shf_resid_hot_minus_cold,
        "corr_lsm_shf_tmax": lsm_r,
        "r2_lsm_shf_tmax": lsm_r2,
        "slope_lsm_shf_tmax": lsm_slope,
        "fig5_scalar_consistency_status": consistency_status,
        "fig5_grouping_stage": "Stage-I_dry",
        "intended_use": intended_use,
    })
    paper_process_qc.extend([
        f"Fig.5 key stats [{init}] corr(Tmax, SM) = {corr_tmax_sm:.6f}",
        f"Fig.5 key stats [{init}] corr(Tmax, raw SHF) = {corr_tmax_raw_shf:.6f}",
        f"Fig.5 key stats [{init}] corr(Tmax, SHF_resid) = {corr_tmax_shf_resid:.6f}",
        f"Fig.5 key stats [{init}] corr(SHF_resid, SM) = {corr_shf_resid_sm:.8e}",
        f"Fig.5 key stats [{init}] raw SHF Hot-Cold difference = {raw_shf_hot_minus_cold:.6f}",
        f"Fig.5 key stats [{init}] SHF_resid Hot-Cold difference = {shf_resid_hot_minus_cold:.6f}",
        f"Fig.5 LSM-SHF [{init}] init_date = {init}",
        f"Fig.5 LSM-SHF [{init}] Stage-I MLR target variable = {lsm_coef_info['target_variable']}",
        f"Fig.5 LSM-SHF [{init}] Stage-I MLR target unit = {lsm_coef_info['target_unit']}",
        f"Fig.5 LSM-SHF [{init}] Stage-I MLR predictors standardized = {lsm_coef_info['predictors_standardized_in_fit']}",
        f"Fig.5 LSM-SHF [{init}] Stage-I MLR intercept included = {lsm_coef_info['intercept_included']}",
        f"Fig.5 LSM-SHF [{init}] Coefficient type used = {lsm_coef_info['coefficient_type_used']}",
        f"Fig.5 LSM-SHF [{init}] original_fit_predictor_space = {lsm_coef_info['original_fit_predictor_space']}",
        f"Fig.5 LSM-SHF [{init}] coefficient_space_used_for_Fig5 = {lsm_coef_info['coefficient_space_used_for_Fig5']}",
        f"Fig.5 LSM-SHF [{init}] target_is_standardized = {lsm_coef_info['target_is_standardized']}",
        f"Fig.5 LSM-SHF [{init}] target_space_basis = {lsm_coef_info['target_space_basis']}",
        f"Fig.5 fitted component [{init}] MLR_baseline_at_mean_state = {baseline_info['MLR_baseline_at_mean_state']:.6f}",
        f"Fig.5 fitted component [{init}] baseline uses full slim-model predictor set = {baseline_info['baseline_uses_full_slim_predictor_set']}",
        f"Fig.5 fitted component [{init}] baseline predictor list = {baseline_info['baseline_predictor_list']}",
        f"Fig.5 fitted component [{init}] baseline missing predictors = {baseline_info['baseline_missing_predictors']}",
        f"Fig.5 fitted component [{init}] corr(SM_SHF_partial_fitted_Tmax, Stage-I Tmax) = {partial_fitted_r:.6f}",
        f"Fig.5 LSM-SHF [{init}] beta_SM = {lsm_coef_info['beta_SM']:.8g}",
        f"Fig.5 LSM-SHF [{init}] beta_SHF = {lsm_coef_info['beta_SHF']:.8g}",
        f"Fig.5 LSM-SHF [{init}] intercept = {lsm_coef_info['intercept']:.8g}",
        f"Fig.5 LSM-SHF [{init}] predictor_scaling = {lsm_coef_info['predictor_scaling']}",
        f"Fig.5 LSM-SHF [{init}] target_space = {lsm_coef_info['target_space']}",
        f"Fig.5 LSM-SHF [{init}] SM mean/std used for scaling = {lsm_coef_info['sm_mean']:.8g}/{lsm_coef_info['sm_std']:.8g}",
        f"Fig.5 LSM-SHF [{init}] SHF_resid mean/std used for scaling = {lsm_coef_info['shf_resid_mean']:.8g}/{lsm_coef_info['shf_resid_std']:.8g}",
        f"Fig.5 LSM-SHF [{init}] corr(LSM_SHF_partial_raw, Stage-I Tmax) = {lsm_raw_r:.6f}",
        f"Fig.5 LSM-SHF [{init}] corr(LSM_SHF_partial_centered, Stage-I Tmax) = {lsm_r:.6f}",
        f"Fig.5 LSM-SHF [{init}] corr(LSM_SHF_with_intercept, Stage-I Tmax) = {lsm_intercept_r:.6f}",
        f"Fig.5 LSM-SHF [{init}] R2 = {lsm_r2:.6f}",
        f"Fig.5 LSM-SHF [{init}] slope = {lsm_slope:.6f}",
        f"Fig.5 LSM-SHF [{init}] n_members = {lsm_n}",
        f"Fig.5 LSM-SHF [{init}] Hot17 mean = {g_stagei[g_stagei['hotcold_group'] == 'Hot17']['LSM_SHF_partial_centered'].mean():.6f}",
        f"Fig.5 LSM-SHF [{init}] Middle17 mean = {g_stagei[g_stagei['hotcold_group'] == 'Middle17']['LSM_SHF_partial_centered'].mean():.6f}",
        f"Fig.5 LSM-SHF [{init}] Cold17 mean = {g_stagei[g_stagei['hotcold_group'] == 'Cold17']['LSM_SHF_partial_centered'].mean():.6f}",
        f"Fig.5 LSM-SHF [{init}] Hot-Cold difference = {g_stagei[g_stagei['hotcold_group'] == 'Hot17']['LSM_SHF_partial_centered'].mean() - g_stagei[g_stagei['hotcold_group'] == 'Cold17']['LSM_SHF_partial_centered'].mean():.6f}",
    ])

    for _row in g_stagei.itertuples(index=False):
        fig5_lsm_shf_audit_rows.append({
            "init_date": init,
            "stage": "Stage-I_dry",
            "member": getattr(_row, "member"),
            "group": getattr(_row, "hotcold_group"),
            "sm_avg": float(getattr(_row, "sm_avg")),
            "SHF_resid": float(getattr(_row, "SHF_resid")),
            "X_SM_used": float(getattr(_row, "X_SM_used")),
            "X_SHF_resid_used": float(getattr(_row, "X_SHF_resid_used")),
            "beta_SM": lsm_coef_info["beta_SM"],
            "beta_SHF": lsm_coef_info["beta_SHF"],
            "intercept": lsm_coef_info["intercept"],
            "predictor_scaling": lsm_coef_info["predictor_scaling"],
            "target_space": lsm_coef_info["target_space"],
            "coefficient_source_file": SLIM_COEF_TABLE,
            "coefficient_space_used_for_Fig5": lsm_coef_info["coefficient_space_used_for_Fig5"],
            "original_fit_predictor_space": lsm_coef_info["original_fit_predictor_space"],
            "fig5_main_variable": "LSM_SHF_partial_centered",
            "target_is_standardized": lsm_coef_info["target_is_standardized"],
            "intercept_used_in_main_figure": False,
            "MLR_baseline_at_mean_state": baseline_info["MLR_baseline_at_mean_state"],
            "SM_SHF_partial_fitted_Tmax": float(getattr(_row, "SM_SHF_partial_fitted_Tmax")),
            "baseline_predictor_count": baseline_info["baseline_predictor_count"],
            "baseline_predictor_list": baseline_info["baseline_predictor_list"],
            "SM_mean_used": lsm_coef_info["sm_mean"],
            "SM_std_used": lsm_coef_info["sm_std"],
            "SHF_resid_mean_used": lsm_coef_info["shf_resid_mean"],
            "SHF_resid_std_used": lsm_coef_info["shf_resid_std"],
            "LSM_SHF_partial_raw": float(getattr(_row, "LSM_SHF_partial_raw")),
            "LSM_SHF_partial_centered": float(getattr(_row, "LSM_SHF_partial_centered")),
            "LSM_SHF_with_intercept": float(getattr(_row, "LSM_SHF_with_intercept")),
            "StageI_Tmax": float(getattr(_row, "y_tmax")),
        })
    fig5_lsm_shf_summary_rows.append({
        "init_date": init,
        "beta_SM": lsm_coef_info["beta_SM"],
        "beta_SHF": lsm_coef_info["beta_SHF"],
        "intercept": lsm_coef_info["intercept"],
        "predictor_scaling": lsm_coef_info["predictor_scaling"],
        "target_space": lsm_coef_info["target_space"],
        "target_variable": lsm_coef_info["target_variable"],
        "target_unit": lsm_coef_info["target_unit"],
        "predictors_standardized_in_fit": lsm_coef_info["predictors_standardized_in_fit"],
        "intercept_included": lsm_coef_info["intercept_included"],
        "coefficient_type_used": lsm_coef_info["coefficient_type_used"],
        "fig5_main_variable": "LSM_SHF_partial_centered",
        "fig5_unit_label": "Linear contribution to Tmax (°C)",
        "coefficient_source_file": SLIM_COEF_TABLE,
        "coefficient_space_used_for_Fig5": lsm_coef_info["coefficient_space_used_for_Fig5"],
        "original_fit_predictor_space": lsm_coef_info["original_fit_predictor_space"],
        "MLR_baseline_at_mean_state": baseline_info["MLR_baseline_at_mean_state"],
        "baseline_predictor_count": baseline_info["baseline_predictor_count"],
        "baseline_predictor_list": baseline_info["baseline_predictor_list"],
        "baseline_full_predictor_count": baseline_info["baseline_full_predictor_count"],
        "baseline_full_predictor_list": baseline_info["baseline_full_predictor_list"],
        "baseline_missing_predictors": baseline_info["baseline_missing_predictors"],
        "baseline_uses_full_slim_predictor_set": baseline_info["baseline_uses_full_slim_predictor_set"],
        "partial_fitted_min": float(g_stagei["SM_SHF_partial_fitted_Tmax"].min()),
        "partial_fitted_max": float(g_stagei["SM_SHF_partial_fitted_Tmax"].max()),
        "partial_fitted_mean": float(g_stagei["SM_SHF_partial_fitted_Tmax"].mean()),
        "Hot17_mean_partial_fitted": float(g_stagei[g_stagei["hotcold_group"] == "Hot17"]["SM_SHF_partial_fitted_Tmax"].mean()),
        "Middle17_mean_partial_fitted": float(g_stagei[g_stagei["hotcold_group"] == "Middle17"]["SM_SHF_partial_fitted_Tmax"].mean()),
        "Cold17_mean_partial_fitted": float(g_stagei[g_stagei["hotcold_group"] == "Cold17"]["SM_SHF_partial_fitted_Tmax"].mean()),
        "Hot_minus_Cold_partial_fitted": float(g_stagei[g_stagei["hotcold_group"] == "Hot17"]["SM_SHF_partial_fitted_Tmax"].mean() - g_stagei[g_stagei["hotcold_group"] == "Cold17"]["SM_SHF_partial_fitted_Tmax"].mean()),
        "corr_partial_fitted_vs_Tmax": partial_fitted_r,
        "r2_partial_fitted_vs_Tmax": partial_fitted_r2,
        "slope_partial_fitted_vs_Tmax": partial_fitted_slope,
        "partial_raw_min": float(g_stagei["LSM_SHF_partial_raw"].min()),
        "partial_raw_max": float(g_stagei["LSM_SHF_partial_raw"].max()),
        "partial_raw_mean": float(g_stagei["LSM_SHF_partial_raw"].mean()),
        "partial_centered_min": float(g_stagei["LSM_SHF_partial_centered"].min()),
        "partial_centered_max": float(g_stagei["LSM_SHF_partial_centered"].max()),
        "partial_centered_mean": float(g_stagei["LSM_SHF_partial_centered"].mean()),
        "with_intercept_min": float(g_stagei["LSM_SHF_with_intercept"].min()),
        "with_intercept_max": float(g_stagei["LSM_SHF_with_intercept"].max()),
        "with_intercept_mean": float(g_stagei["LSM_SHF_with_intercept"].mean()),
        "corr_partial_raw_vs_Tmax": lsm_raw_r,
        "corr_partial_centered_vs_Tmax": lsm_r,
        "corr_with_intercept_vs_Tmax": lsm_intercept_r,
        "r2_partial_centered_vs_Tmax": lsm_r2,
        "slope_partial_centered_vs_Tmax": lsm_slope,
        "n_members": lsm_n,
        "Hot17_mean_partial_centered": float(g_stagei[g_stagei["hotcold_group"] == "Hot17"]["LSM_SHF_partial_centered"].mean()),
        "Middle17_mean_partial_centered": float(g_stagei[g_stagei["hotcold_group"] == "Middle17"]["LSM_SHF_partial_centered"].mean()),
        "Cold17_mean_partial_centered": float(g_stagei[g_stagei["hotcold_group"] == "Cold17"]["LSM_SHF_partial_centered"].mean()),
        "Hot_minus_Cold_partial_centered": float(g_stagei[g_stagei["hotcold_group"] == "Hot17"]["LSM_SHF_partial_centered"].mean() - g_stagei[g_stagei["hotcold_group"] == "Cold17"]["LSM_SHF_partial_centered"].mean()),
    })

    fig, axes = plt.subplots(4, 2, figsize=(10.4, 14.4), constrained_layout=False)
    raw_panels = [
        ("sm_avg", "SM", "Soil moisture (m³ m⁻³)"),
        ("z500_anom_NCHN", "Z500 anomaly", "Z500 anomaly (gpm)"),
        ("SHF_Avg", "SHF", "W m⁻²"),
        ("NCVI_conc", "NCVI anomaly", "NCVI anomaly (PVU)"),
        ("TCC_Avg", "TCC", "TCC (fraction)"),
    ]
    axes_flat = axes.ravel()
    _paper_lsm_shf_daily_panel(
        axes_flat[0], df_paper_raw_ts, init, lsm_coef_info, shf_intercept, shf_slope,
        "2023-06-14", "2023-06-18", "Centered SM–SHF contribution", "Linear contribution to Tmax (°C)"
    )
    axes_flat[0].axvline(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=12), color="0.35", lw=0.8, ls="--", alpha=0.65)
    for ax, (factor, title, ylabel) in zip(axes_flat[1:6], raw_panels):
        era5_ok = _paper_raw_timeseries_panel(ax, df_paper_raw_ts, init, "Stage-I", factor, "2023-06-14", "2023-06-18", title, ylabel)
        if not era5_ok:
            paper_process_qc.append(f"[{init}] Fig.5 factor {factor}: ERA5 line skipped because RAW_TS_TABLE era5 is all NaN.")
        ax.axvline(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=12), color="0.35", lw=0.8, ls="--", alpha=0.65)
    _paper_stagei_scatter(axes_flat[6], g_stagei, "sm_avg", "y_tmax", "SM vs Stage-I Tmax")
    _paper_stagei_scatter(axes_flat[7], g_stagei, "LSM_SHF_partial_centered", "y_tmax", "Centered SM–SHF contribution vs Stage-I Tmax")
    axes_flat[6].legend(loc="best", frameon=False, fontsize=8)
    _paper_add_panel_labels(axes, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"])
    handles, labels = axes_flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Stage-I grouped land-surface thermal evolution | Init {init}", fontsize=15)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.935, bottom=0.085, wspace=0.30, hspace=0.52)
    for ext in ["png", "pdf"]:
        out_path = f"{PAPER_PROCESS_SUBDIRS['figures']}/fig5_stageI_grouped_land_surface_thermal_evolution_init_{init}.{ext}"
        assert_paper_process_output_path(out_path)
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    fig_alt, axes_alt = plt.subplots(4, 2, figsize=(10.8, 15.2), constrained_layout=False)
    axes_alt_flat = axes_alt.ravel()
    _paper_lsm_shf_daily_panel(
        axes_alt_flat[0], df_paper_raw_ts, init, lsm_coef_info, shf_intercept, shf_slope,
        "2023-06-14", "2023-06-18", "SM–SHF partial fitted Tmax", "Partial fitted Tmax from SM–SHF (°C)",
        baseline=baseline_info["MLR_baseline_at_mean_state"],
    )
    axes_alt_flat[0].axvline(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=12), color="0.35", lw=0.8, ls="--", alpha=0.65)
    for ax, (factor, title, ylabel) in zip(axes_alt_flat[1:6], raw_panels):
        era5_ok = _paper_raw_timeseries_panel(ax, df_paper_raw_ts, init, "Stage-I", factor, "2023-06-14", "2023-06-18", title, ylabel)
        if not era5_ok:
            paper_process_qc.append(f"[{init}] alternate Fig.5 factor {factor}: ERA5 line skipped because RAW_TS_TABLE era5 is all NaN.")
        ax.axvline(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=12), color="0.35", lw=0.8, ls="--", alpha=0.65)
    _paper_stagei_scatter(axes_alt_flat[6], g_stagei, "sm_avg", "y_tmax", "SM vs Stage-I Tmax")
    _paper_stagei_scatter(axes_alt_flat[7], g_stagei, "SM_SHF_partial_fitted_Tmax", "y_tmax", "SM–SHF partial fitted Tmax vs Stage-I Tmax")
    axes_alt_flat[6].legend(loc="best", frameon=False, fontsize=8)
    _paper_add_panel_labels(axes_alt, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"])
    handles_alt, labels_alt = axes_alt_flat[1].get_legend_handles_labels()
    fig_alt.legend(handles_alt, labels_alt, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, fontsize=9, frameon=False)
    fig_alt.suptitle(f"Stage-I grouped land-surface thermal evolution | fitted-component comparison | Init {init}", fontsize=15)
    fig_alt.subplots_adjust(left=0.085, right=0.975, top=0.935, bottom=0.085, wspace=0.30, hspace=0.52)
    for ext in ["png", "pdf"]:
        out_path = f"{PAPER_PROCESS_SUBDIRS['figures']}/fig5_stageI_grouped_land_surface_thermal_evolution_fitted_component_init_{init}.{ext}"
        assert_paper_process_output_path(out_path)
        fig_alt.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig_alt)

    fig6_summary_rows.append({
        "init_date": init,
        "stage": "Total",
        "fig6_grouping_stage": "Total",
        "time_range": "2023-06-14 to 2023-06-24",
        "intended_use": intended_use,
    })

    fig, axes = plt.subplots(3, 2, figsize=(10.8, 11.8), constrained_layout=False)
    total_panels = [
        ("NCHN_tp", "TP", "Precipitation (mm day⁻¹)"),
        ("z500_anom_NCHN", "Z500 anomaly", "Z500 anomaly (gpm)"),
        ("sm_avg", "SM", "Soil moisture (m³ m⁻³)"),
        ("NCVI_conc", "NCVI anomaly", "NCVI anomaly (PVU)"),
        ("TCC_Avg", "TCC", "TCC (fraction)"),
    ]
    axes_flat = axes.ravel()
    for ax, (factor, title, ylabel) in zip(axes_flat[:5], total_panels):
        era5_ok = _paper_raw_timeseries_panel(ax, df_paper_raw_ts, init, "Total", factor, "2023-06-14", "2023-06-24", title, ylabel)
        if not era5_ok:
            paper_process_qc.append(f"[{init}] Fig.6 factor {factor}: ERA5 line skipped because RAW_TS_TABLE era5 is all NaN.")
        ax.axvline(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=12), color="0.35", lw=0.8, ls="--", alpha=0.65)
    ax_note = axes_flat[5]
    ax_note.axis("off")
    note_text = (
        "Figure note\n\n"
        "Grouping: Total-period Hot17 / Cold17\n"
        "Period: 2023-06-14–06-24\n"
        "Dashed vertical line: Stage-I boundary\n"
        "Lines: ERA5, Hot17, Middle17, Cold17"
    )
    ax_note.text(
        0.05,
        0.80,
        note_text,
        transform=ax_note.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9, boxstyle="round,pad=0.45"),
    )
    _paper_add_panel_labels(axes_flat[:5], ["(a)", "(b)", "(c)", "(d)", "(e)"])
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"Total-period process chain from precipitation to Tmax | Init {init}", fontsize=15)
    paper_process_qc.append(f"[{init}] Fig.6 panels use Total Hot17/Cold17 grouping and show 2023-06-14 to 2023-06-24; right-bottom note box is a compact note panel only.")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10, wspace=0.32, hspace=0.52)
    for ext in ["png", "pdf"]:
        out_path = f"{PAPER_PROCESS_SUBDIRS['figures']}/fig6_total_process_chain_precip_to_tmax_init_{init}.{ext}"
        assert_paper_process_output_path(out_path)
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

fig5_lsm_shf_audit_path = f"{PAPER_PROCESS_SUBDIRS['tables']}/table_fig5_lsm_shf_contribution_audit.csv"
fig5_lsm_shf_summary_path = f"{PAPER_PROCESS_SUBDIRS['tables']}/table_fig5_lsm_shf_contribution_summary.csv"
fig5_summary_path = f"{PAPER_PROCESS_SUBDIRS['tables']}/paper_fig5_stageI_bridge_summary.csv"
fig6_summary_path = f"{PAPER_PROCESS_SUBDIRS['tables']}/paper_fig6_total_process_chain_summary.csv"
qc_paper_process_path = f"{PAPER_PROCESS_SUBDIRS['qc']}/qc_paper_process_figures_NCVIconcurrent.txt"
for out_path in [fig5_lsm_shf_audit_path, fig5_lsm_shf_summary_path, fig5_summary_path, fig6_summary_path, qc_paper_process_path]:
    assert_paper_process_output_path(out_path)
pd.DataFrame(fig5_lsm_shf_audit_rows).to_csv(fig5_lsm_shf_audit_path, index=False)
pd.DataFrame(fig5_lsm_shf_summary_rows).to_csv(fig5_lsm_shf_summary_path, index=False)
pd.DataFrame(fig5_summary_rows).to_csv(fig5_summary_path, index=False)
pd.DataFrame(fig6_summary_rows).to_csv(fig6_summary_path, index=False)
with open(qc_paper_process_path, "w", encoding="utf-8") as f:
    f.write("\n".join(paper_process_qc))

required_paper_process_outputs = [fig5_lsm_shf_audit_path, fig5_lsm_shf_summary_path, fig5_summary_path, fig6_summary_path, qc_paper_process_path]
for init in PAPER_PROCESS_INIT_ORDER:
    required_paper_process_outputs.extend([
        f"{PAPER_PROCESS_SUBDIRS['figures']}/fig5_stageI_grouped_land_surface_thermal_evolution_init_{init}.png",
        f"{PAPER_PROCESS_SUBDIRS['figures']}/fig5_stageI_grouped_land_surface_thermal_evolution_init_{init}.pdf",
        f"{PAPER_PROCESS_SUBDIRS['figures']}/fig5_stageI_grouped_land_surface_thermal_evolution_fitted_component_init_{init}.png",
        f"{PAPER_PROCESS_SUBDIRS['figures']}/fig5_stageI_grouped_land_surface_thermal_evolution_fitted_component_init_{init}.pdf",
        f"{PAPER_PROCESS_SUBDIRS['figures']}/fig6_total_process_chain_precip_to_tmax_init_{init}.png",
        f"{PAPER_PROCESS_SUBDIRS['figures']}/fig6_total_process_chain_precip_to_tmax_init_{init}.pdf",
    ])
for out_path in required_paper_process_outputs:
    assert_paper_process_output_path(out_path)
    if not os.path.exists(out_path):
        raise RuntimeError(f"Cell 12 paper process figure output is missing: {out_path}")
print(f"[Cell 12 paper process figures] output root = {PAPER_PROCESS_ROOT}")
print(f"[Cell 12 paper process figures] Fig.5 bridge summary = {fig5_summary_path}")
print(f"[Cell 12 paper process figures] Fig.6 Total process-chain summary = {fig6_summary_path}")

# %%
# ============================================================
# Cell 13: Paper MLR Stage-I/Total figures and robustness filtering
# ============================================================

set_audit_cell_name("Cell 13: paper MLR Stage-I/Total figures")
print("\n" + "=" * 80)
print("Cell 13: Paper MLR Stage-I/Total figures and robustness filtering")
print("=" * 80)

PAPER_MLR_OUT = FOUR_INIT_ROOT
PAPER_MLR_SUBDIRS = {
    "figures": f"{PAPER_MLR_OUT}/figures/paper_mlr",
    "tables": f"{PAPER_MLR_OUT}/tables",
    "qc": f"{PAPER_MLR_OUT}/qc",
}


def assert_paper_mlr_output_path(path):
    if not str(path).startswith(f"{FOUR_INIT_ROOT}/") and str(path) != FOUR_INIT_ROOT:
        raise RuntimeError(f"Paper MLR Stage-I/Total output path is outside FOUR_INIT_ROOT: {path}")


assert_paper_mlr_output_path(PAPER_MLR_OUT + "/")
for _paper_mlr_dir in PAPER_MLR_SUBDIRS.values():
    assert_paper_mlr_output_path(_paper_mlr_dir + "/")
    os.makedirs(_paper_mlr_dir, exist_ok=True)

PAPER_MLR_STAGES = ["Stage-I_dry", "Total"]
MAIN_INIT = "2023-06-12"
CONTRAST_INIT = "2023-06-08"
AUX_INITS = ["2023-06-05", "2023-06-01"]
PAPER_ALL_INITS = [MAIN_INIT, CONTRAST_INIT] + AUX_INITS
STAGE_DISPLAY = {"Stage-I_dry": "Stage-I", "Total": "Total"}
FACTOR_DISPLAY = {
    "SHF_resid": "SHF resid",
    "sm_avg": "SM",
    "SM_resid": "SM resid",
    "z500_anom_NCHN": "Z500 anom",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
    "TCC_resid": "TCC resid",
    "NCVI_concurrent": "NCVI anomaly",
    "NCVI": "NCVI anomaly",
    "NCHN_tp": "TP",
}
PAPER_MLR_PREDICTORS = {
    "Stage-I_dry": ["sm_avg", "SHF_resid", "z500_anom_NCHN", "Initial_MSEstar_max_NCHN", "TCC_resid", "NCVI_concurrent"],
    "Total": ["NCHN_tp", "z500_anom_NCHN", "NCVI_concurrent", "Initial_MSEstar_max_NCHN", "TCC_resid", "SM_resid"],
}

PAPER_MLR_SLIM_DIR = f"{FOUR_INIT_ROOT}/tables"
PAPER_MLR_JOHNSON_DIR = f"{FOUR_INIT_ROOT}/tables"
PAPER_MLR_INPUTS = {
    "summary": f"{PAPER_MLR_SLIM_DIR}/table_slim_model_summary.csv",
    "fitted": f"{PAPER_MLR_SLIM_DIR}/table_slim_fitted_members.csv",
    "lmg": f"{PAPER_MLR_SLIM_DIR}/table_slim_lmg_long.csv",
    "method_importance": f"{PAPER_MLR_JOHNSON_DIR}/table_method_importance_comparison.csv",
}
for _name, _path in PAPER_MLR_INPUTS.items():
    if not os.path.exists(_path):
        raise RuntimeError(f"Cell 13 missing upstream {_name} table: {_path}. Please run Cell 8 and Cell 9 first.")

_df_mlr_summary = pd.read_csv(PAPER_MLR_INPUTS["summary"])
_df_mlr_fitted = pd.read_csv(PAPER_MLR_INPUTS["fitted"])
_df_mlr_lmg = pd.read_csv(PAPER_MLR_INPUTS["lmg"])
_df_mlr_methods = pd.read_csv(PAPER_MLR_INPUTS["method_importance"])
_required_paper_tables = {
    "summary": (_df_mlr_summary, {"init_date", "stage"}),
    "fitted": (_df_mlr_fitted, {"init_date", "stage", "member"}),
    "lmg": (_df_mlr_lmg, {"init_date", "stage"}),
    "method_importance": (_df_mlr_methods, {"init_date", "stage"}),
}
for _table_name, (_table_df, _required_cols) in _required_paper_tables.items():
    if _table_df.empty or not _required_cols.issubset(_table_df.columns):
        if _table_name == "fitted":
            _empty_coverage = pd.DataFrame([
                {"init_date": init, "stage": stage, "n_members": 0, "available_for_slim_mlr": False, "available_for_paper_plot": False, "missing_reason": "zero fitted rows"}
                for init in FULL_INIT_LIST for stage in ["Stage-I_dry", "Stage-II_wet", "Total"]
            ])
            _empty_coverage_path = f"{FOUR_INIT_ROOT}/tables/table_init_stage_coverage_qc.csv"
            assert_four_init_output(_empty_coverage_path)
            _empty_coverage.to_csv(_empty_coverage_path, index=False)
            raise RuntimeError("Four-init slim MLR produced zero fitted rows. Check table_init_stage_coverage_qc.csv.")
        raise RuntimeError(f"Paper figures skipped: {_table_name} is empty or missing columns {_required_cols}.")
for _df in [_df_mlr_summary, _df_mlr_fitted, _df_mlr_lmg, _df_mlr_methods]:
    if "init_date" in _df.columns:
        _df["init_date"] = _df["init_date"].astype(str)

def _paper_mlr_role(init):
    if init == MAIN_INIT:
        return "main_text"
    if init == CONTRAST_INIT:
        return "contrast_appendix"
    return "auxiliary_appendix"


def _paper_mlr_n_members(init, stage):
    rows = _df_mlr_fitted[(_df_mlr_fitted["init_date"] == init) & (_df_mlr_fitted["stage"] == stage)]
    return int(rows["member"].nunique()) if "member" in rows.columns else int(len(rows))


def _paper_mlr_coverage(init, stage):
    summary_rows = _df_mlr_summary[(_df_mlr_summary["init_date"] == init) & (_df_mlr_summary["stage"] == stage)]
    fitted_rows = _df_mlr_fitted[(_df_mlr_fitted["init_date"] == init) & (_df_mlr_fitted["stage"] == stage)]
    lmg_rows = _df_mlr_lmg[(_df_mlr_lmg["init_date"] == init) & (_df_mlr_lmg["stage"] == stage)]
    method_rows = _df_mlr_methods[(_df_mlr_methods["init_date"] == init) & (_df_mlr_methods["stage"] == stage)]
    n_members = _paper_mlr_n_members(init, stage)
    reasons = []
    if len(summary_rows) != 1:
        reasons.append(f"summary_rows={len(summary_rows)}")
    if n_members != 51:
        reasons.append(f"n_members={n_members}")
    if lmg_rows.empty:
        reasons.append("missing_lmg")
    if method_rows.empty or "johnson_pct" not in method_rows.columns:
        reasons.append("missing_johnson")
    if method_rows.empty or "lofo_delta_r2" not in method_rows.columns:
        reasons.append("missing_lofo")
    return {
        "init_date": init,
        "stage": stage,
        "data_source": "slim_mlr_and_johnson_tables",
        "fitted_member_n": n_members,
        "model_summary_exists": len(summary_rows) == 1,
        "lmg_exists": not lmg_rows.empty,
        "johnson_exists": (not method_rows.empty and "johnson_pct" in method_rows.columns),
        "lofo_exists": (not method_rows.empty and "lofo_delta_r2" in method_rows.columns),
        "figure_role": _paper_mlr_role(init),
        "available_for_plot": len(reasons) == 0,
        "missing_reason": "; ".join(reasons) if reasons else "",
    }


def _paper_mlr_summary_row(init, stage):
    rows = _df_mlr_summary[(_df_mlr_summary["init_date"] == init) & (_df_mlr_summary["stage"] == stage)]
    if len(rows) != 1:
        raise RuntimeError(f"Cell 13 expected one slim summary row for {init}/{stage}, got {len(rows)}")
    return rows.iloc[0]


def _paper_mlr_lmg_rows(init, stage):
    rows = _df_mlr_lmg[(_df_mlr_lmg["init_date"] == init) & (_df_mlr_lmg["stage"] == stage)].copy()
    if rows.empty:
        raise RuntimeError(f"Cell 13 missing LMG rows for {init}/{stage}")
    pred_col = "predictor" if "predictor" in rows.columns else "factor"
    expected = PAPER_MLR_PREDICTORS[stage]
    if set(rows[pred_col].astype(str)) != set(expected):
        raise RuntimeError(f"Cell 13 LMG predictor mismatch for {init}/{stage}: got={rows[pred_col].tolist()}, expected={expected}")
    pct_col = "LMG_percent" if "LMG_percent" in rows.columns else "LMG_importance_percent"
    rows["factor_name"] = rows[pred_col].astype(str)
    rows["relative_importance_pct"] = rows[pct_col].astype(float)
    if rows["relative_importance_pct"].max(skipna=True) <= 1.0:
        rows["relative_importance_pct"] = rows["relative_importance_pct"] * 100.0
    rows["display_label"] = rows["factor_name"].map(FACTOR_DISPLAY).fillna(rows["factor_name"])
    return rows.sort_values("relative_importance_pct", ascending=False)


def _paper_mlr_method_rows(init, stage):
    rows = _df_mlr_methods[(_df_mlr_methods["init_date"] == init) & (_df_mlr_methods["stage"] == stage)].copy()
    if rows.empty:
        raise RuntimeError(f"Cell 13 missing method-importance rows for {init}/{stage}")
    expected = PAPER_MLR_PREDICTORS[stage]
    if set(rows["factor"].astype(str)) != set(expected):
        raise RuntimeError(f"Cell 13 method predictor mismatch for {init}/{stage}: got={rows['factor'].tolist()}, expected={expected}")
    rows["factor_name"] = rows["factor"].astype(str)
    rows["display_label"] = rows["factor_name"].map(FACTOR_DISPLAY).fillna(rows["factor_name"])
    pos_sum = float(np.maximum(rows["lofo_delta_r2"].astype(float), 0.0).sum())
    rows["lofo_positive_pct"] = np.where(pos_sum > 0, np.maximum(rows["lofo_delta_r2"].astype(float), 0.0) / pos_sum * 100.0, 0.0)
    return rows


def _paper_mlr_plot_scatter(ax, init, stage):
    rows = _df_mlr_fitted[(_df_mlr_fitted["init_date"] == init) & (_df_mlr_fitted["stage"] == stage)].copy()
    if len(rows) != 51:
        raise RuntimeError(f"Cell 13 expected 51 fitted members for {init}/{stage}, got {len(rows)}")
    summary = _paper_mlr_summary_row(init, stage)
    y = rows["y_tmax"].astype(float).values
    yhat = rows["y_fitted"].astype(float).values
    lim_min = float(min(np.nanmin(y), np.nanmin(yhat)))
    lim_max = float(max(np.nanmax(y), np.nanmax(yhat)))
    pad = max((lim_max - lim_min) * 0.05, 0.25)
    ax.scatter(y, yhat, s=32, alpha=0.76, color="#2f6db3", edgecolor="white", linewidth=0.4)
    ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], ls="--", color="0.35", lw=1.0)
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_xlabel("ECMWF member Tmax (°C)")
    ax.set_ylabel("Slim MLR predicted Tmax (°C)")
    ax.set_title(f"{STAGE_DISPLAY[stage]}: R²={summary['R2']:.2f}, Adj.R²={summary['Adj_R2']:.2f}, RMSE={summary['RMSE']:.2f}")


def _paper_mlr_plot_lmg(ax, init, stage):
    rows = _paper_mlr_lmg_rows(init, stage).sort_values("relative_importance_pct", ascending=True)
    vals = rows["relative_importance_pct"].astype(float).values
    labels = rows["display_label"].tolist()
    ax.barh(labels, vals, color="#6AA84F", alpha=0.86)
    xmax = max(float(np.nanmax(vals)) if len(vals) else 1.0, 1.0)
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel("LMG relative importance (%)")
    ax.set_title(f"{STAGE_DISPLAY[stage]} LMG")
    for y_idx, val in enumerate(vals):
        ax.text(val + xmax * 0.025, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.5)


def _paper_mlr_file_prefix(init, kind):
    if init == MAIN_INIT:
        return f"fig_3_slim_main_results_stageI_total_{init}" if kind == "main" else f"fig_4_mlr_robustness_stageI_total_{init}"
    if init == CONTRAST_INIT:
        return f"figS_slim_main_results_stageI_total_{init}" if kind == "main" else f"figS_mlr_robustness_stageI_total_{init}"
    return f"figS_aux_slim_main_results_stageI_total_{init}" if kind == "main" else f"figS_aux_mlr_robustness_stageI_total_{init}"


def _paper_mlr_save_main_figure(init):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.4), gridspec_kw={"width_ratios": [1.0, 1.18]}, constrained_layout=False)
    for row_idx, stage in enumerate(PAPER_MLR_STAGES):
        _paper_mlr_plot_scatter(axes[row_idx, 0], init, stage)
        _paper_mlr_plot_lmg(axes[row_idx, 1], init, stage)
    title = f"MLR attribution | Init {init}" + (" | auxiliary check" if init in AUX_INITS else "")
    fig.suptitle(title, fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.09, wspace=0.36, hspace=0.55)
    prefix = _paper_mlr_file_prefix(init, "main")
    png = f"{PAPER_MLR_SUBDIRS['figures']}/{prefix}.png"
    pdf = f"{PAPER_MLR_SUBDIRS['figures']}/{prefix}.pdf"
    for out_path in [png, pdf]:
        assert_paper_mlr_output_path(out_path)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _paper_mlr_save_robustness_figure(init):
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.2), constrained_layout=False)
    method_specs = [
        ("LMG", "lmg_pct", "#6AA84F", "LMG relative importance (%)"),
        ("Johnson", "johnson_pct", "#3C78D8", "Johnson relative weights (%)"),
        ("LOFO", "lofo_positive_pct", "#E69138", "LOFO normalized contribution (%)"),
    ]
    for col_idx, stage in enumerate(PAPER_MLR_STAGES):
        methods = _paper_mlr_method_rows(init, stage)
        order = _paper_mlr_lmg_rows(init, stage)["factor_name"].tolist()
        plot_base = methods.set_index("factor_name").loc[order].reset_index()
        for row_idx, (title, value_col, color, xlabel) in enumerate(method_specs):
            ax = axes[row_idx, col_idx]
            vals = plot_base[value_col].astype(float).values[::-1]
            labels = plot_base["display_label"].tolist()[::-1]
            ax.barh(labels, vals, color=color, alpha=0.86)
            xmax = max(float(np.nanmax(vals)) if len(vals) else 1.0, 1.0)
            ax.set_xlim(0, xmax * 1.15)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{title} | {STAGE_DISPLAY[stage]}")
            for y_idx, val in enumerate(vals):
                ax.text(val + xmax * 0.025, y_idx, f"{val:.1f}%", va="center", ha="left", fontsize=8.2)
    title = f"Robustness of MLR factor importance | Init {init}" + (" | auxiliary check" if init in AUX_INITS else "")
    fig.suptitle(title, fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.07, wspace=0.38, hspace=0.56)
    prefix = _paper_mlr_file_prefix(init, "robust")
    png = f"{PAPER_MLR_SUBDIRS['figures']}/{prefix}.png"
    pdf = f"{PAPER_MLR_SUBDIRS['figures']}/{prefix}.pdf"
    for out_path in [png, pdf]:
        assert_paper_mlr_output_path(out_path)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf

coverage_rows = []
qc_paper_mlr_lines = [
    "Cell 13: Paper MLR Stage-I/Total figures and robustness filtering",
    "Stage-II_wet was retained in full model outputs but excluded from paper MLR figures.",
    f"Paper MLR stages = {PAPER_MLR_STAGES}.",
    f"Main init = {MAIN_INIT}.",
    f"Contrast init = {CONTRAST_INIT}.",
    f"Auxiliary init checks = {AUX_INITS}.",
    "No data loading, residualization, MLR, LMG, Johnson, or LOFO calculation was changed by this paper-figure filtering cell.",
    "For NCVI: daily time-series and MLR stage-mean predictor represent the same NCVI anomaly factor but different temporal aggregation.",
]
for init in PAPER_ALL_INITS:
    for stage in PAPER_MLR_STAGES:
        coverage_rows.append(_paper_mlr_coverage(init, stage))
df_paper_mlr_coverage = pd.DataFrame(coverage_rows)

selected_summary_rows = []
selected_lmg_rows = []
selected_method_rows = []
generated_main_files = []
generated_robust_files = []
for init in PAPER_ALL_INITS:
    cov_init = df_paper_mlr_coverage[df_paper_mlr_coverage["init_date"] == init]
    available = bool(cov_init["available_for_plot"].all())
    if not available:
        msg = f"Auxiliary init {init} missing from slim MLR outputs; skipped appendix figure." if init in AUX_INITS else f"Required init {init} missing selected Stage-I/Total coverage: {cov_init['missing_reason'].tolist()}"
        qc_paper_mlr_lines.append(msg)
        continue
    for stage in PAPER_MLR_STAGES:
        summary = _paper_mlr_summary_row(init, stage)
        lmg = _paper_mlr_lmg_rows(init, stage)
        top1 = lmg.iloc[0]
        n_members = _paper_mlr_n_members(init, stage)
        selected_summary_rows.append({
            "init_date": init,
            "stage": stage,
            "R2": float(summary["R2"]),
            "Adj_R2": float(summary["Adj_R2"]),
            "RMSE": float(summary["RMSE"]),
            "Top1_factor_name": top1["factor_name"],
            "Top1_factor_importance_pct": float(top1["relative_importance_pct"]),
            "n_members": n_members,
            "available_for_plot": True,
        })
        for _, row in lmg.iterrows():
            selected_lmg_rows.append({
                "init_date": init,
                "stage": stage,
                "factor_name": row["factor_name"],
                "display_label": row["display_label"],
                "relative_importance_pct": float(row["relative_importance_pct"]),
            })
        methods = _paper_mlr_method_rows(init, stage)
        for _, row in methods.iterrows():
            for method, value_col in [("LMG", "lmg_pct"), ("Johnson", "johnson_pct"), ("LOFO", "lofo_positive_pct")]:
                selected_method_rows.append({
                    "init_date": init,
                    "stage": stage,
                    "method": method,
                    "factor_name": row["factor_name"],
                    "display_label": row["display_label"],
                    "importance_pct": float(row[value_col]),
                })
    main_files = _paper_mlr_save_main_figure(init)
    robust_files = _paper_mlr_save_robustness_figure(init)
    generated_main_files.extend(main_files)
    generated_robust_files.extend(robust_files)
    qc_paper_mlr_lines.append(f"Init {init}: Fig.3 files={main_files}; Fig.4 files={robust_files}; n_members={[int(v) for v in cov_init['fitted_member_n'].tolist()]}")

paper_mlr_summary_path = f"{PAPER_MLR_SUBDIRS['tables']}/paper_mlr_selected_stage_model_summary.csv"
paper_mlr_lmg_path = f"{PAPER_MLR_SUBDIRS['tables']}/paper_mlr_selected_stage_lmg_long.csv"
paper_mlr_method_path = f"{PAPER_MLR_SUBDIRS['tables']}/paper_mlr_selected_stage_method_importance.csv"
paper_mlr_coverage_path = f"{PAPER_MLR_SUBDIRS['tables']}/paper_mlr_init_stage_coverage_qc.csv"
paper_mlr_qc_path = f"{PAPER_MLR_SUBDIRS['qc']}/qc_paper_mlr_stageI_total_NCVIconcurrent.txt"
for out_path in [paper_mlr_summary_path, paper_mlr_lmg_path, paper_mlr_method_path, paper_mlr_coverage_path, paper_mlr_qc_path]:
    assert_paper_mlr_output_path(out_path)
pd.DataFrame(selected_summary_rows).to_csv(paper_mlr_summary_path, index=False)
pd.DataFrame(selected_lmg_rows).to_csv(paper_mlr_lmg_path, index=False)
pd.DataFrame(selected_method_rows).to_csv(paper_mlr_method_path, index=False)
df_paper_mlr_coverage.to_csv(paper_mlr_coverage_path, index=False)
qc_paper_mlr_lines.append("Confirmed: Stage-II_wet does not appear in paper MLR figure panels.")
qc_paper_mlr_lines.append("Confirmed: no data calculation, residualization, MLR, LMG, Johnson, or LOFO result was changed.")
with open(paper_mlr_qc_path, "w", encoding="utf-8") as f:
    f.write("\n".join(qc_paper_mlr_lines))

required_main = [
    f"{PAPER_MLR_SUBDIRS['figures']}/fig_3_slim_main_results_stageI_total_{MAIN_INIT}.png",
    f"{PAPER_MLR_SUBDIRS['figures']}/fig_3_slim_main_results_stageI_total_{MAIN_INIT}.pdf",
    f"{PAPER_MLR_SUBDIRS['figures']}/figS_slim_main_results_stageI_total_{CONTRAST_INIT}.png",
    f"{PAPER_MLR_SUBDIRS['figures']}/figS_slim_main_results_stageI_total_{CONTRAST_INIT}.pdf",
]
required_robust = [
    f"{PAPER_MLR_SUBDIRS['figures']}/fig_4_mlr_robustness_stageI_total_{MAIN_INIT}.png",
    f"{PAPER_MLR_SUBDIRS['figures']}/fig_4_mlr_robustness_stageI_total_{MAIN_INIT}.pdf",
    f"{PAPER_MLR_SUBDIRS['figures']}/figS_mlr_robustness_stageI_total_{CONTRAST_INIT}.png",
    f"{PAPER_MLR_SUBDIRS['figures']}/figS_mlr_robustness_stageI_total_{CONTRAST_INIT}.pdf",
]
for out_path in [*required_main, *required_robust, paper_mlr_summary_path, paper_mlr_lmg_path, paper_mlr_method_path, paper_mlr_coverage_path, paper_mlr_qc_path]:
    assert_paper_mlr_output_path(out_path)
    if not os.path.exists(out_path):
        raise RuntimeError(f"Cell 13 required paper MLR Stage-I/Total output is missing: {out_path}")

print(f"[Paper MLR Stage-I/Total] output directory = {PAPER_MLR_OUT}")
print(f"[Paper MLR Stage-I/Total] 06-12 Fig.3/Fig.4 generated = {all(os.path.exists(p) for p in required_main[:2] + required_robust[:2])}")
print(f"[Paper MLR Stage-I/Total] 06-08 supplementary Fig.3/Fig.4 generated = {all(os.path.exists(p) for p in required_main[2:] + required_robust[2:])}")
for aux_init in AUX_INITS:
    aux_cov = df_paper_mlr_coverage[df_paper_mlr_coverage["init_date"] == aux_init]
    if aux_cov["available_for_plot"].all():
        print(f"[Paper MLR Stage-I/Total] {aux_init} auxiliary figures generated")
    else:
        reason = "; ".join([r for r in aux_cov["missing_reason"].tolist() if r])
        print(f"[Paper MLR Stage-I/Total] {aux_init} missing auxiliary init, skipped: {reason}")
print("[Paper MLR Stage-I/Total] n_members by init/stage:")
print(df_paper_mlr_coverage[["init_date", "stage", "fitted_member_n", "available_for_plot", "missing_reason"]].to_string(index=False))
print("[Paper MLR Stage-I/Total] Stage-II_wet is retained in full outputs but excluded from paper MLR figure panels.")
print("[Paper MLR Stage-I/Total] No data calculation, residualization, MLR, LMG, Johnson, or LOFO results were changed.")

# Consolidated four-init coverage and backward-consistency QC for the expanded main pipeline.
_four_stage_raw_path = f"{FOUR_INIT_ROOT}/tables/table_stage_member_raw_factors_TCC.csv"
_four_ncvi_path = f"{FOUR_INIT_ROOT}/tables/table_ncvi_concurrent_stage_values.csv"
_four_stage_raw = pd.read_csv(_four_stage_raw_path) if os.path.exists(_four_stage_raw_path) else pd.DataFrame()
_four_ncvi = pd.read_csv(_four_ncvi_path) if os.path.exists(_four_ncvi_path) else pd.DataFrame()
for _df in [_four_stage_raw, _four_ncvi]:
    if "init_date" in _df.columns:
        _df["init_date"] = _df["init_date"].astype(str)
_four_coverage_rows = []
for _init in FULL_INIT_LIST:
    for _stage in ["Stage-I_dry", "Stage-II_wet", "Total"]:
        _g = _four_stage_raw[(_four_stage_raw.get("init_date", pd.Series(dtype=str)) == _init) & (_four_stage_raw.get("stage", pd.Series(dtype=str)) == _stage)].copy() if not _four_stage_raw.empty else pd.DataFrame()
        _n = int(_g["member"].nunique()) if "member" in _g.columns else 0
        _nc = _four_ncvi[(_four_ncvi.get("init_date", pd.Series(dtype=str)) == _init) & (_four_ncvi.get("stage", pd.Series(dtype=str)) == _stage)].copy() if not _four_ncvi.empty else pd.DataFrame()
        _slim = _df_mlr_summary[(_df_mlr_summary["init_date"] == _init) & (_df_mlr_summary["stage"] == _stage)]
        _required = PAPER_MLR_PREDICTORS.get(_stage, [])
        _missing = [c for c in _required if c not in _g.columns and c not in {"SHF_resid", "SM_resid", "TCC_resid", "NCVI_concurrent"}]
        _four_coverage_rows.append({
            "init_date": _init,
            "stage": _stage,
            "has_tmax": "y_tmax" in _g.columns and _g["y_tmax"].notna().any(),
            "has_tp": "NCHN_tp" in _g.columns and _g["NCHN_tp"].notna().any(),
            "has_sm": "sm_avg" in _g.columns and _g["sm_avg"].notna().any(),
            "has_shf": "SHF_Avg" in _g.columns and _g["SHF_Avg"].notna().any(),
            "has_z500": "z500_anom_NCHN" in _g.columns and _g["z500_anom_NCHN"].notna().any(),
            "has_tcc": "TCC_Avg" in _g.columns and _g["TCC_Avg"].notna().any(),
            "has_ncvi_concurrent": not _nc.empty and "NCVI_concurrent" in _nc.columns and _nc["NCVI_concurrent"].notna().any(),
            "has_mse_star": "Initial_MSEstar_max_NCHN" in _g.columns and _g["Initial_MSEstar_max_NCHN"].notna().any(),
            "has_required_predictors": not _missing,
            "n_members": _n,
            "available_for_slim_mlr": len(_slim) == 1 and _n == 51,
            "available_for_paper_plot": _stage in PAPER_MLR_STAGES and len(_slim) == 1 and _n == 51,
            "missing_reason": "" if len(_slim) == 1 and _n == 51 else f"stage_rows={len(_g)}; n_members={_n}; slim_summary_rows={len(_slim)}; missing_raw={_missing}",
        })
df_four_init_coverage = pd.DataFrame(_four_coverage_rows)
four_init_coverage_path = f"{FOUR_INIT_ROOT}/tables/table_init_stage_coverage_qc.csv"
assert_four_init_output(four_init_coverage_path)
df_four_init_coverage.to_csv(four_init_coverage_path, index=False)

_old_summary_path = "/data1/huangy/fig6/NC/v9/slim_mlr_NCVIconcurrent/tables/table_slim_model_summary.csv"
_old_lmg_path = "/data1/huangy/fig6/NC/v9/slim_mlr_NCVIconcurrent/tables/table_slim_lmg_long.csv"
_old_method_path = "/data1/huangy/fig6/NC/v9/johnson_validation_slim_NCVIconcurrent/tables/table_method_importance_comparison.csv"
_old_summary = pd.read_csv(_old_summary_path) if os.path.exists(_old_summary_path) else pd.DataFrame()
_old_lmg = pd.read_csv(_old_lmg_path) if os.path.exists(_old_lmg_path) else pd.DataFrame()
_old_method = pd.read_csv(_old_method_path) if os.path.exists(_old_method_path) else pd.DataFrame()
for _df in [_old_summary, _old_lmg, _old_method]:
    if "init_date" in _df.columns:
        _df["init_date"] = _df["init_date"].astype(str)


def _standardize_lmg_for_compare(df, pct_alias):
    if df is None or df.empty:
        return pd.DataFrame(columns=["factor", pct_alias])

    factor_col = None
    for c in ["factor", "predictor", "variable", "name", "factor_name"]:
        if c in df.columns:
            factor_col = c
            break

    pct_col = None
    for c in [
        "LMG_percent",
        "LMG_importance_percent",
        "LMG_relative_importance_pct",
        "importance_percent",
        "relative_importance_percent",
        "relative_importance_pct",
    ]:
        if c in df.columns:
            pct_col = c
            break

    if factor_col is None or pct_col is None:
        return pd.DataFrame(columns=["factor", pct_alias])

    out = df[[factor_col, pct_col]].copy()
    out = out.rename(columns={factor_col: "factor", pct_col: pct_alias})
    out["factor"] = out["factor"].astype(str)
    out[pct_alias] = pd.to_numeric(out[pct_alias], errors="coerce")
    out = out.dropna(subset=["factor", pct_alias])
    if not out.empty and out[pct_alias].abs().max(skipna=True) <= 1.0:
        out[pct_alias] = out[pct_alias] * 100.0
    return out


_consistency_rows = []
for _init in [CONTRAST_INIT, MAIN_INIT]:
    for _stage in PAPER_MLR_STAGES:
        _row = {
            "init_date": _init,
            "stage": _stage,
            "R2_diff": np.nan,
            "Adj_R2_diff": np.nan,
            "RMSE_diff": np.nan,
            "LMG_compare_status": "old_comparison_skipped",
            "LMG_max_abs_diff": np.nan,
            "Johnson_max_abs_diff": np.nan,
            "LOFO_max_abs_diff": np.nan,
            "Top1_factor_old": "",
            "Top1_factor_new": "",
            "status": "missing_old_or_new",
        }
        _new_s = _df_mlr_summary[(_df_mlr_summary["init_date"] == _init) & (_df_mlr_summary["stage"] == _stage)]
        _old_s = _old_summary[(_old_summary.get("init_date", pd.Series(dtype=str)) == _init) & (_old_summary.get("stage", pd.Series(dtype=str)) == _stage)] if not _old_summary.empty else pd.DataFrame()
        if len(_new_s) == 1 and len(_old_s) == 1:
            _row["R2_diff"] = float(_new_s.iloc[0]["R2"] - _old_s.iloc[0]["R2"])
            _row["Adj_R2_diff"] = float(_new_s.iloc[0]["Adj_R2"] - _old_s.iloc[0]["Adj_R2"])
            _row["RMSE_diff"] = float(_new_s.iloc[0]["RMSE"] - _old_s.iloc[0]["RMSE"])
            _new_l = _paper_mlr_lmg_rows(_init, _stage)
            _old_l = _old_lmg[(_old_lmg.get("init_date", pd.Series(dtype=str)) == _init) & (_old_lmg.get("stage", pd.Series(dtype=str)) == _stage)].copy() if not _old_lmg.empty else pd.DataFrame()
            if not _new_l.empty and not _old_l.empty:
                _new_l_cmp = _standardize_lmg_for_compare(_new_l, "new_pct")
                _old_l_cmp = _standardize_lmg_for_compare(_old_l, "old_pct")
                _merge_l = _new_l_cmp.merge(_old_l_cmp, on="factor", how="inner")
                if _merge_l.empty:
                    _row["LMG_compare_status"] = "no_common_factors"
                else:
                    _row["LMG_max_abs_diff"] = float((_merge_l["new_pct"] - _merge_l["old_pct"]).abs().max())
                    _row["Top1_factor_new"] = _merge_l.sort_values("new_pct", ascending=False).iloc[0]["factor"]
                    _row["Top1_factor_old"] = _merge_l.sort_values("old_pct", ascending=False).iloc[0]["factor"]
                    _row["LMG_compare_status"] = "ok"
            _new_m = _df_mlr_methods[(_df_mlr_methods["init_date"] == _init) & (_df_mlr_methods["stage"] == _stage)].copy()
            _old_m = _old_method[(_old_method.get("init_date", pd.Series(dtype=str)) == _init) & (_old_method.get("stage", pd.Series(dtype=str)) == _stage)].copy() if not _old_method.empty else pd.DataFrame()
            if not _new_m.empty and not _old_m.empty and "factor" in _old_m.columns:
                _mm = _new_m.merge(_old_m, on="factor", suffixes=("_new", "_old"))
                if {"johnson_pct_new", "johnson_pct_old"}.issubset(_mm.columns):
                    _row["Johnson_max_abs_diff"] = float((_mm["johnson_pct_new"] - _mm["johnson_pct_old"]).abs().max())
                if {"lofo_delta_r2_new", "lofo_delta_r2_old"}.issubset(_mm.columns):
                    _row["LOFO_max_abs_diff"] = float((_mm["lofo_delta_r2_new"] - _mm["lofo_delta_r2_old"]).abs().max())
            _diffs = [abs(_row[c]) for c in ["R2_diff", "Adj_R2_diff", "RMSE_diff", "LMG_max_abs_diff", "Johnson_max_abs_diff", "LOFO_max_abs_diff"] if pd.notna(_row[c])]
            _row["status"] = "PASS" if _diffs and max(_diffs) <= 1e-6 and _row["Top1_factor_old"] == _row["Top1_factor_new"] else "FAIL"
        _consistency_rows.append(_row)
df_four_init_consistency = pd.DataFrame(_consistency_rows)
four_init_consistency_path = f"{FOUR_INIT_ROOT}/tables/table_four_init_vs_old_main_consistency_qc.csv"
assert_four_init_output(four_init_consistency_path)
df_four_init_consistency.to_csv(four_init_consistency_path, index=False)

four_init_full_qc_path = f"{FOUR_INIT_ROOT}/qc/qc_four_init_full_pipeline_NCVIconcurrent.txt"
assert_four_init_output(four_init_full_qc_path)
_tcc_qc_path = f"{FOUR_INIT_ROOT}/tables/table_tcc_qc_summary.csv"
_tcc_audit_path = f"{FOUR_INIT_ROOT}/tables/table_tcc_residual_audit.csv"
_ncvi_qc_path = f"{FOUR_INIT_ROOT}/qc/qc_ncvi_concurrent_sensitivity.txt"
_full_qc_lines = [
    f"Full init list = {FULL_INIT_LIST}",
    f"Output root = {FOUR_INIT_ROOT}",
    "No old output overwritten = True",
    f"TCC coverage table = {_tcc_qc_path}; exists={os.path.exists(_tcc_qc_path)}",
    f"NCVI_concurrent coverage QC = {_ncvi_qc_path}; exists={os.path.exists(_ncvi_qc_path)}",
    f"Raw factor coverage = {FOUR_INIT_ROOT}/tables/table_raw_factor_coverage_qc.csv",
    f"TCC_resid audit = {_tcc_audit_path}; exists={os.path.exists(_tcc_audit_path)}",
    "SHF_resid / SM_resid audit follows the unchanged Cell 2 and Cell 8 residualization logic.",
    f"Slim MLR coverage combinations available = {int(df_four_init_coverage['available_for_slim_mlr'].sum())}/{len(FULL_INIT_LIST) * 3}",
    f"Old 06-08/06-12 consistency = {'PASS' if (df_four_init_consistency['status'] == 'PASS').all() else 'FAIL'}",
    f"Paper figure files generated = {len(generated_main_files) + len(generated_robust_files)}",
    "Initial_MSEstar_max_NCHN follows the existing Cell 1 definition: init-relative lead day 1–3 maximum saturated MSE over NCHN.",
    "For 2023-06-05 and 2023-06-01, Initial_MSEstar_max_NCHN is an early-lead thermodynamic state, not a near-onset condition.",
    "TCC_resid is used in the final slim MLR; raw TCC_Avg is retained only as an upstream diagnostic.",
]
for _, _r in df_four_init_coverage.iterrows():
    _full_qc_lines.append(f"{_r['init_date']} / {_r['stage']}: n_members={_r['n_members']}; available_for_slim_mlr={_r['available_for_slim_mlr']}; available_for_paper_plot={_r['available_for_paper_plot']}; missing_reason={_r['missing_reason']}")
with open(four_init_full_qc_path, "w", encoding="utf-8") as _f:
    _f.write("\n".join(_full_qc_lines))

print(f"[Four-init full pipeline] output root = {FOUR_INIT_ROOT}")
print(df_four_init_coverage[["init_date", "stage", "n_members", "available_for_slim_mlr", "available_for_paper_plot", "missing_reason"]].to_string(index=False))
print(f"[Four-init full pipeline] old 06-08/06-12 consistency = {'PASS' if (df_four_init_consistency['status'] == 'PASS').all() else 'FAIL'}")
print("[Four-init full pipeline] old v7/v8/v9/v11 outputs untouched = True")

# Write the complete runtime output-path manifest after all other output operations.
output_path_audit_csv = f"{FOUR_INIT_ROOT}/tables/table_output_path_audit.csv"
_record_output_path("to_csv", output_path_audit_csv, True)
df_output_path_audit = pd.DataFrame(_OUTPUT_PATH_AUDIT_ROWS, columns=[
    "cell_name",
    "operation",
    "path",
    "is_write_operation",
    "under_four_init_root",
    "allowed_old_read",
    "status",
])
bad_output_paths = df_output_path_audit[
    df_output_path_audit["is_write_operation"] & ~df_output_path_audit["under_four_init_root"]
]
if not bad_output_paths.empty:
    raise RuntimeError(
        "Output-path audit found write operations outside FOUR_INIT_ROOT:\n"
        + bad_output_paths.to_string(index=False)
    )
_ORIGINAL_TO_CSV(df_output_path_audit, output_path_audit_csv, index=False)
print(f"[Four-init full pipeline] output-path audit = {output_path_audit_csv}")

# %%
# ============================================================
# Cell 14: Standalone paper Fig.3-6 renderer from exported tables
# ============================================================

# Standalone paper Fig.3-6 renderer from exported tables.
# This cell intentionally reads only exported CSV tables and does not trigger any
# upstream data-processing, model-fitting, residualization, or regrouping steps.
import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy.stats import pearsonr

FULL_INIT_LIST = ["2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12"]
MAIN_INIT = "2023-06-12"
CONTRAST_INIT = "2023-06-08"
AUX_INITS = ["2023-06-05", "2023-06-01"]
PAPER_MLR_STAGES = ["Stage-I_dry", "Total"]

EXP_OUT_DIR = "/data1/huangy/fig6/NC/v12"
FOUR_INIT_ROOT = f"{EXP_OUT_DIR}/four_init_full_pipeline_NCVIconcurrent"
PROCESS_TS_ROOT = "/data1/huangy/fig6/NC/v11/hotcold_stage_group_timeseries_tables_multiinit_totalfig6"
PROCESS_TS_TABLE_DIR = f"{PROCESS_TS_ROOT}/tables"
STAGEI_GROUP_FILE = "/data1/huangy/fig6/NC/v9/tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv"

PAPER_RENDER_ROOT = f"{FOUR_INIT_ROOT}/paper_figures_standalone"
PAPER_RENDER_FIG_DIR = f"{PAPER_RENDER_ROOT}/figures"
PAPER_RENDER_TABLE_DIR = f"{PAPER_RENDER_ROOT}/tables"
PAPER_RENDER_QC_DIR = f"{PAPER_RENDER_ROOT}/qc"


def _standalone_assert_output(path):
    root = os.path.abspath(PAPER_RENDER_ROOT)
    target = os.path.abspath(path)
    if os.path.commonpath([root, target]) != root:
        raise RuntimeError(f"Standalone paper renderer output is outside PAPER_RENDER_ROOT: {path}")
    return path


for _d in [PAPER_RENDER_ROOT, PAPER_RENDER_FIG_DIR, PAPER_RENDER_TABLE_DIR, PAPER_RENDER_QC_DIR]:
    _standalone_assert_output(_d + "/")
    os.makedirs(_d, exist_ok=True)

FIG34_INPUTS = {
    "summary": f"{FOUR_INIT_ROOT}/tables/table_slim_model_summary.csv",
    "fitted": f"{FOUR_INIT_ROOT}/tables/table_slim_fitted_members.csv",
    "lmg": f"{FOUR_INIT_ROOT}/tables/table_slim_lmg_long.csv",
    "method_importance": f"{FOUR_INIT_ROOT}/tables/table_method_importance_comparison.csv",
}
FIG56_INPUTS = {
    "raw_daily_timeseries": f"{PROCESS_TS_TABLE_DIR}/daily_timeseries_all_factors.csv",
    "raw_member_stage_values": f"{PROCESS_TS_TABLE_DIR}/member_stage_values_for_tests.csv",
    "hotcold_group_file": STAGEI_GROUP_FILE,
    "stageI_audit_member": f"{FOUR_INIT_ROOT}/optional_stageI_raw_resid_consistency_audit/tables/stageI_raw_resid_member_audit.csv",
    "stageI_audit_group_difference": f"{FOUR_INIT_ROOT}/optional_stageI_raw_resid_consistency_audit/tables/stageI_raw_resid_group_difference_summary.csv",
    "stageI_audit_scatter": f"{FOUR_INIT_ROOT}/optional_stageI_raw_resid_consistency_audit/tables/stageI_raw_resid_scatter_stats.csv",
    "stageI_fig5_consistency": f"{FOUR_INIT_ROOT}/optional_stageI_raw_resid_consistency_audit/tables/stageI_fig5_vs_mlr_scalar_consistency.csv",
    "slim_model_summary": f"{FOUR_INIT_ROOT}/tables/table_slim_model_summary.csv",
    "slim_fitted_members": f"{FOUR_INIT_ROOT}/tables/table_slim_fitted_members.csv",
    "slim_lmg_long": f"{FOUR_INIT_ROOT}/tables/table_slim_lmg_long.csv",
    "slim_model_coefficients": f"{FOUR_INIT_ROOT}/tables/table_slim_model_coefficients.csv",
    "ncvi_stage_values": f"{FOUR_INIT_ROOT}/tables/table_ncvi_concurrent_stage_values.csv",
    "tcc_source_table": f"{FOUR_INIT_ROOT}/tables/table_stage_member_raw_factors_TCC.csv",
    "tcc_daily_member_file": f"{FOUR_INIT_ROOT}/tables/table_daily_member_tcc_SHF.csv",
    "tcc_era5_s2s_compare_file": f"{FOUR_INIT_ROOT}/tables/table_daily_tcc_era5_s2s_compare.csv",
}

STANDALONE_OUTPUT_NAME_EXAMPLES = [
    "paper_fig3_mlr_attribution_init_2023-06-12.png",
    "paper_fig4_mlr_robustness_init_2023-06-12.png",
    "paper_fig5_total_process_chain_init_2023-06-12.png",
    "paper_fig6_stageI_land_surface_evolution_init_2023-06-12.png",
]

PAPER_STYLE = {
    "font_family": "Arial",
    "panel_title_size": 9.5,
    "axis_label_size": 9.5,
    "tick_label_size": 8.5,
    "legend_size": 8.5,
    "annotation_size": 8.0,
    "bar_value_size": 7.0,
    "panel_label_size": 10.5,
    "matrix_header_size": 10.5,
    "fig3_lmg_row_header_size": 9.0,
    "fig4_row_header_size": 9.0,
    "line_width": 1.75,
    "marker_size": 4.8,
    "scatter_size": 22,
    "scatter_alpha": 0.62,
    "grid_alpha": 0.16,
    "timeseries_grid_alpha": 0.20,
    "scatter_grid_alpha": 0.18,
    "bar_grid_alpha": 0.12,
    "grid_line_width": 0.45,
    "fig3_wspace": 0.30,
    "fig3_hspace": 0.42,
    "fig4_wspace": 0.36,
    "fig4_hspace": 0.42,
    "fig56_wspace": 0.28,
    "fig56_hspace": 0.42,
    "dpi": 600,
    "method_colors": {"LMG": "#6AA84F", "Johnson": "#3C78D8", "LOFO": "#E69138"},
    "line_colors": {"ERA5 Obs": "black", "Hot17 Mean": "firebrick", "Middle17 Mean": "#1f77b4", "Cold17 Mean": "seagreen"},
}
PAPER_FIG_WIDTH_IN = 6.5
PAPER_FIG_HEIGHTS = {
    "Fig3": 5.6,
    "Fig4": 6.9,
    "Fig5": 7.0,
    "Fig6": 7.3,
}
plt.rcParams.update({
    "font.family": PAPER_STYLE["font_family"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

STANDALONE_STAGE_DISPLAY = {"Stage-I_dry": "Stage-I", "Total": "Total period"}
STANDALONE_FACTOR_DISPLAY = {
    "sm_avg": "SM",
    "SHF_Avg": "SHF",
    "SHF_resid": "SHF resid",
    "LSM_SHF_partial_centered": "SM–SHF contribution to Tmax (°C)",
    "SM_SHF_partial_fitted_Tmax": "SM–SHF (°C)",
    "NCHN_tp": "TP",
    "z500_anom_NCHN": "Z500 anomaly",
    "TCC_Avg": "TCC",
    "NCVI_concurrent": "NCVI anomaly",
    "NCVI_conc": "NCVI anomaly",
    "Initial_MSEstar_max_NCHN": "Init MSE*",
    "TCC_resid": "TCC resid",
    "SM_resid": "SM resid",
    "y_tmax": "Stage-I Tmax",
}
_STANDALONE_GROUP_ORDER = ["Hot17", "Middle17", "Cold17"]
_STANDALONE_GROUP_COLORS = {"Hot17": "firebrick", "Middle17": "#6c8ebf", "Cold17": "seagreen"}
FIG6_SCATTER_STATS_POSITIONS = {
    "g": (0.55, 0.93),
    "h": (0.04, 0.94),
}
FIG6_SCATTER_GROUP_LEGEND = {
    "loc": "upper right",
    "bbox_to_anchor": (0.99, 0.99),
}


def _standalone_role(init):
    if init == MAIN_INIT:
        return "main"
    if init == CONTRAST_INIT:
        return "supplementary"
    return "auxiliary"


def _standalone_norm_stage(x):
    s = str(x)
    if s in {"Stage-I_dry", "Stage-I", "Stage_I", "Stage1"}:
        return "Stage-I"
    if s in {"Stage-II_wet", "Stage-II", "Stage_II", "Stage2"}:
        return "Stage-II"
    return s


def _standalone_find_col(df, candidates, desc):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"Standalone renderer cannot find {desc}; tried={candidates}; columns={list(df.columns)}")


def _apply_axis_base_style(ax):
    ax.set_facecolor("white")
    ax.tick_params(axis="both", labelsize=PAPER_STYLE["tick_label_size"], width=0.8, length=3.2)
    ax.xaxis.label.set_size(PAPER_STYLE["axis_label_size"])
    ax.yaxis.label.set_size(PAPER_STYLE["axis_label_size"])
    ax.title.set_size(PAPER_STYLE["panel_title_size"])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


_STANDALONE_TICK_FORMAT = {
    "NCHN_tp": "%.1f",
    "z500_anom_NCHN": "%.0f",
    "sm_avg": "%.3f",
    "SHF_Avg": "%.1f",
    "NCVI_conc": "%.3f",
    "NCVI_concurrent": "%.3f",
    "TCC_Avg": "%.2f",
    "LSM_SHF_partial_centered": "%.1f",
    "SM_SHF_partial_fitted_Tmax": "%.1f",
}
_STANDALONE_ZERO_REF_FACTORS = {"z500_anom_NCHN", "NCVI_conc", "NCVI_concurrent", "LSM_SHF_partial_centered"}


def _set_y_tick_format(ax, factor):
    fmt = _STANDALONE_TICK_FORMAT.get(factor)
    if fmt:
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))


def _apply_timeseries_axis_style(ax, factor=None):
    _apply_axis_base_style(ax)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=PAPER_STYLE["grid_line_width"], alpha=PAPER_STYLE["timeseries_grid_alpha"])
    _set_y_tick_format(ax, factor)


def _apply_scatter_axis_style(ax):
    _apply_axis_base_style(ax)
    ax.grid(True, which="major", axis="both", linestyle=":", linewidth=PAPER_STYLE["grid_line_width"], alpha=PAPER_STYLE["scatter_grid_alpha"])


def _apply_bar_axis_style(ax):
    _apply_axis_base_style(ax)
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.4, alpha=PAPER_STYLE["bar_grid_alpha"])
    ax.grid(False, axis="y")


_TIMESERIES_MAJOR_TICK_SPACING = {
    ("Stage-I", "sm_avg"): 0.005,
    ("Stage-I", "z500_anom_NCHN"): 10.0,
    ("Stage-I", "SHF_Avg"): 5.0,
    ("Stage-I", "NCVI_conc"): 0.025,
    ("Total", "NCHN_tp"): 1.0,
    ("Total", "z500_anom_NCHN"): 25.0,
    ("Total", "sm_avg"): 0.010,
    ("Total", "NCVI_conc"): 0.05,
}


def _apply_timeseries_major_spacing(ax, stage, factor):
    step = _TIMESERIES_MAJOR_TICK_SPACING.get((stage, factor))
    if step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(step))
    else:
        # Preserve the established renderer behavior only for panels without a
        # physically specified interval; explicit spacings above always win.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _apply_numeric_tick_density(ax, init_date, axis="y", nbins=6):
    """Reduce auxiliary numeric-tick density without changing limits."""
    if init_date == MAIN_INIT:
        return False
    locator = MaxNLocator(nbins=nbins)
    if axis == "y":
        ax.yaxis.set_major_locator(locator)
    elif axis == "x":
        ax.xaxis.set_major_locator(locator)
    else:
        raise ValueError(f"Unsupported axis for numeric tick density: {axis}")
    return True


def _pad_bar_xlim_for_labels(ax, values, pad_fraction=0.28):
    """Reserve right-side room for compact bar-end value labels."""
    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return
    vmax = float(np.nanmax(numeric))
    old_left, old_right = ax.get_xlim()
    required_right = vmax * (1.0 + pad_fraction)
    ax.set_xlim(old_left, max(float(old_right), required_right))


def _add_panel_header(ax, letter, title=None, x=0.0, y=1.025):
    """Add a bold panel letter and an optional normal-weight short title."""
    ax.set_title("")
    letter_artist = ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PAPER_STYLE["panel_label_size"],
        fontweight="bold",
        clip_on=False,
    )
    if title:
        ax.text(
            x + 0.125,
            y,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PAPER_STYLE["panel_title_size"],
            fontweight="normal",
            clip_on=False,
        )
    return letter_artist


def _add_matrix_headers(
    fig,
    axes,
    column_headers,
    row_headers,
    column_y=0.975,
    row_x=0.018,
    row_header_sizes=None,
):
    """Add one stage header per column and one method header per row."""
    for col, header in enumerate(column_headers):
        pos = axes[0, col].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            column_y,
            header,
            ha="center",
            va="top",
            fontsize=PAPER_STYLE["matrix_header_size"],
            fontweight="bold",
        )
    for row, header in enumerate(row_headers):
        pos = axes[row, 0].get_position()
        row_size = PAPER_STYLE["matrix_header_size"] if row_header_sizes is None else row_header_sizes[row]
        fig.text(
            row_x,
            (pos.y0 + pos.y1) / 2,
            header,
            ha="center",
            va="center",
            rotation=90,
            fontsize=row_size,
            fontweight="bold",
        )


def _format_daily_date_axis(ax, start_date, end_date):
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ticks = pd.date_range(start_date, end_date, freq="D")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{d.month}.{d.day}" for d in ticks], rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=PAPER_STYLE["tick_label_size"])


_saved_figure_metadata = {}


def _save_paper_figure(fig, basename):
    png = _standalone_assert_output(f"{PAPER_RENDER_FIG_DIR}/{basename}.png")
    pdf = _standalone_assert_output(f"{PAPER_RENDER_FIG_DIR}/{basename}.pdf")
    width_in, height_in = [float(v) for v in fig.get_size_inches()]
    fig.savefig(png, dpi=PAPER_STYLE["dpi"])
    fig.savefig(pdf)
    try:
        png_arr = plt.imread(png)
        png_height_px, png_width_px = [int(v) for v in png_arr.shape[:2]]
    except Exception:
        png_width_px, png_height_px = np.nan, np.nan
    _saved_figure_metadata[basename] = {
        "figsize_width_in": width_in,
        "figsize_height_in": height_in,
        "png_width_px": png_width_px,
        "png_height_px": png_height_px,
    }
    plt.close(fig)
    return png, pdf


def _standalone_required_source_for(name):
    if name in {"summary", "fitted", "lmg", "slim_model_summary", "slim_fitted_members", "slim_lmg_long", "slim_model_coefficients"}:
        return "Cell 8: slim MLR tables"
    if name == "method_importance":
        return "Cell 9: Johnson/LOFO method-importance tables"
    if name.startswith("stageI_"):
        return "Cell 11: Stage-I audit tables"
    if name in {"raw_daily_timeseries", "raw_member_stage_values", "hotcold_group_file"}:
        return "v11 process time-series tables"
    return "exported v12/v11 upstream tables"


_standalone_qc = [
    "Standalone renderer = True",
    "Standalone Fig.3-6 renderer = True",
    "CSV-only = True",
    "No upstream data calculation performed = True",
    "No upstream calculation = True",
    "No MLR refit = True",
    "No final-model predictor residualization recalculation = True",
    "No OLS/LMG/Johnson/LOFO recalculation performed = True",
    "No Hot17/Middle17/Cold17 regrouping performed = True",
    "No GRIB/NetCDF raw data read = True",
    "Fig.3 source = exported Cell 8 tables",
    "Fig.4 source = exported Cell 8 + Cell 9 tables",
    "Fig.5 source = exported v11 time-series + Total grouping tables",
    "Fig.6 source = exported v11 time-series + Cell 11 audit + Cell 8 raw-equivalent coefficients",
    "Final Fig.5 = Total-period process chain",
    "Final Fig.6 = Stage-I land-surface pathway",
    "Main Fig.6 fitted_component = True",
    "Main Fig.6 SM-SHF = SM_SHF_partial_fitted_Tmax in the original Tmax space.",
    "Main Fig.6 SM-SHF definition = MLR_baseline_at_mean_state + LSM_SHF_partial_centered.",
    "LSM_SHF_partial_centered remains available only as an unrendered auxiliary QC variant.",
    "Fig.5 grouping = Total-period Hot17 / Middle17 / Cold17",
    "Fig.5 period = 2023-06-14 to 2023-06-24 BJT",
    "Fig.6 Stage-I statistical window = 2023-06-14 to 2023-06-17 BJT",
    "Fig.6 plotted time-series extent = 2023-06-14 to 2023-06-18 BJT",
    "Fig.5 legend panel includes ERA5 Obs, Hot17 Mean, Middle17 Mean, Cold17 Mean, and Stage-I boundary.",
    "Figure-type-specific axis styles applied = True",
    "Fig.5/Fig.6 time-series grid style shared = True",
    "Fig.3 scatter grid style applied separately = True",
    "Fig.3/Fig.4 bar panels use x-grid only or no grid = True",
    "Time-series y-limits are panel/init-specific to avoid excessive whitespace = True",
    "Fig.3 scatter x/y limits are equal within each panel = True",
    "Fig.3 scatter axes use rectangular layout, not forced equal aspect = True",
    "Fig.3 scatter ranges are independent for Stage-I and Total = True",
    "Fig.3 adaptive scatter tick interval applied = True",
    "Panel letters and short titles are separate axes-relative text artists = True",
    "All axes-relative panel text uses transform=ax.transAxes = True",
    "Fixed-canvas manuscript export = True",
    'bbox_inches="tight" = False',
    "Final width = 6.5 inch",
    "PNG dpi = 600",
    "Expected PNG width = 3900 px",
    "All paper figures share the same export width = True",
    "Figure-specific heights are allowed = True",
    f"PAPER_FIG_WIDTH_IN = {PAPER_FIG_WIDTH_IN}",
    f"Fig.3 figsize(width_in, height_in) = ({PAPER_FIG_WIDTH_IN}, {PAPER_FIG_HEIGHTS['Fig3']})",
    f"Fig.4 figsize(width_in, height_in) = ({PAPER_FIG_WIDTH_IN}, {PAPER_FIG_HEIGHTS['Fig4']})",
    f"Fig.5 figsize(width_in, height_in) = ({PAPER_FIG_WIDTH_IN}, {PAPER_FIG_HEIGHTS['Fig5']})",
    f"Fig.6 figsize(width_in, height_in) = ({PAPER_FIG_WIDTH_IN}, {PAPER_FIG_HEIGHTS['Fig6']})",
    "No figure-level suptitles = True",
    "Fig.3 layout = columns Stage-I/Total period; rows MLR/LMG",
    "Fig.4 layout = columns Stage-I/Total period; rows LMG/Johnson/LOFO",
    "Fig.3 scatter line = least-squares fitted line, not 1:1 reference line.",
    "12 Jun main-figure post-render y-axis preserved = True",
    "12 Jun main-figure post-render major y ticks preserved = True",
    "Fig.6 g/h statistical values changed = False",
    "Fig.6 g/h statistical annotation positions changed = True",
    "Fig.3 LMG row-header size reduced independently = True",
    "Fig.4 LMG/Johnson/LOFO row-header size reduced independently = True",
    "Bar-value font weight = bold",
    "Bar-value font size = 7.0",
    "Bar x-axis headroom added = True",
    "Bar x-axis headroom fraction = 0.28",
    "Fig.6 g stats are positioned immediately left of the fixed upper-right group legend for all inits = True",
    "Auxiliary y-tick density adjusted where necessary = False; no verified overlap evidence was available to justify panel-specific changes.",
    "Scientific values changed = False",
    "Standalone manifest width check path = table_standalone_paper_figures_manifest.csv",
    f"Output root = {PAPER_RENDER_ROOT}",
]
_manifest_rows = []
_axis_style_rows = []
_main_y_axis_lock_rows = []
_main_y_axis_locks = []
_input_tables = {}
for _name, _path in {**FIG34_INPUTS, **FIG56_INPUTS}.items():
    if not os.path.exists(_path):
        raise RuntimeError(f"Standalone paper renderer missing input '{_name}': {_path}. Please run {_standalone_required_source_for(_name)} first; this renderer will not rerun upstream calculations.")
    _df_probe = pd.read_csv(_path, nrows=5)
    _row_count = sum(1 for _ in open(_path, "r", encoding="utf-8", errors="ignore")) - 1
    _standalone_qc.append(f"input {_name}: exists=True; rows={_row_count}; columns={list(_df_probe.columns)}; path={_path}")
    _input_tables[_name] = pd.read_csv(_path)

_df_slim_summary = _input_tables["summary"].copy()
_df_slim_fitted = _input_tables["fitted"].copy()
_df_slim_lmg = _input_tables["lmg"].copy()
_df_method = _input_tables["method_importance"].copy()
_df_raw_ts = _input_tables["raw_daily_timeseries"].copy()
_df_stagei_member = _input_tables["stageI_audit_member"].copy()
_df_slim_coef = _input_tables["slim_model_coefficients"].copy()

for _df in [_df_slim_summary, _df_slim_fitted, _df_slim_lmg, _df_method, _df_stagei_member, _df_slim_coef]:
    if "init_date" in _df.columns:
        _df["init_date"] = _df["init_date"].astype(str)
if "stage" in _df_slim_summary.columns:
    _df_slim_summary["stage"] = _df_slim_summary["stage"].astype(str)
if "stage" in _df_slim_fitted.columns:
    _df_slim_fitted["stage"] = _df_slim_fitted["stage"].astype(str)
if "stage" in _df_slim_lmg.columns:
    _df_slim_lmg["stage"] = _df_slim_lmg["stage"].astype(str)
if "stage" in _df_method.columns:
    _df_method["stage"] = _df_method["stage"].astype(str)

_raw_init_col = _standalone_find_col(_df_raw_ts, ["init", "init_date"], "raw daily init column")
_raw_stage_col = _standalone_find_col(_df_raw_ts, ["grouping_stage", "stage"], "raw daily stage column")
_raw_factor_col = _standalone_find_col(_df_raw_ts, ["factor"], "raw daily factor column")
_raw_date_col = _standalone_find_col(_df_raw_ts, ["date", "valid_date", "time"], "raw daily date column")
_df_raw_ts["_init"] = _df_raw_ts[_raw_init_col].astype(str)
_df_raw_ts["_stage"] = _df_raw_ts[_raw_stage_col].map(_standalone_norm_stage)
_df_raw_ts["_factor"] = _df_raw_ts[_raw_factor_col].astype(str)
_df_raw_ts["_date"] = pd.to_datetime(_df_raw_ts[_raw_date_col])
for _line_col in ["era5", "hot17_mean", "middle17_mean", "cold17_mean"]:
    if _line_col not in _df_raw_ts.columns:
        raise RuntimeError(f"Standalone paper renderer raw daily table missing required column: {_line_col}")
    _df_raw_ts[_line_col] = pd.to_numeric(_df_raw_ts[_line_col], errors="coerce")


def _pad_limits(vals, frac=0.08, force_zero_min=False, fixed=None):
    if fixed is not None:
        return fixed
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    vmin = float(vals.min())
    vmax = float(vals.max())
    if force_zero_min:
        vmin = 0.0
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 0.1)
    else:
        pad = (vmax - vmin) * frac
    lo = 0.0 if force_zero_min else vmin - pad
    hi = vmax + pad
    return lo, hi


def _panel_timeseries_ylim(g, factor, columns):
    fixed = (0.0, 1.0) if factor == "TCC_Avg" else None
    vals = []
    for col in columns:
        if col in g.columns:
            vals.extend(pd.to_numeric(g[col], errors="coerce").tolist())
    return _pad_limits(vals, force_zero_min=(factor == "NCHN_tp"), fixed=fixed)


def _record_axis_style(figure_id, init, panel, style_type, detail):
    _axis_style_rows.append({
        "figure_id": figure_id,
        "init_date": init,
        "panel": panel,
        "style_type": style_type,
        "detail": detail,
    })


def _capture_main_y_axis(figure_id, panel, ax, init):
    """Snapshot the approved 12 Jun y-axis before layout-only adjustments."""
    if init != MAIN_INIT:
        return
    lock = {
        "figure": figure_id,
        "panel": panel,
        "ax": ax,
        "ylim_before": tuple(float(v) for v in ax.get_ylim()),
        "yticks_before": tuple(float(v) for v in ax.get_yticks()),
    }
    _main_y_axis_locks.append(lock)


def _verify_main_y_axes(figure_id, init):
    """Fail before export if a layout change altered an approved main y-axis."""
    if init != MAIN_INIT:
        return
    for lock in [row for row in _main_y_axis_locks if row["figure"] == figure_id]:
        ax = lock["ax"]
        ylim_after = tuple(float(v) for v in ax.get_ylim())
        yticks_after = tuple(float(v) for v in ax.get_yticks())
        same_ylim = np.allclose(lock["ylim_before"], ylim_after, rtol=0.0, atol=1e-12)
        same_ticks = len(lock["yticks_before"]) == len(yticks_after) and np.allclose(
            lock["yticks_before"], yticks_after, rtol=0.0, atol=1e-12
        )
        unchanged = bool(same_ylim and same_ticks)
        _main_y_axis_lock_rows.append({
            "figure": figure_id,
            "panel": lock["panel"],
            "ylim_before": repr(lock["ylim_before"]),
            "ylim_after": repr(ylim_after),
            "yticks_before": repr(lock["yticks_before"]),
            "yticks_after": repr(yticks_after),
            "unchanged": unchanged,
        })
        if not unchanged:
            ax.set_ylim(*lock["ylim_before"])
            ax.set_yticks(lock["yticks_before"])
            raise RuntimeError(
                f"{figure_id} {lock['panel']} changed the {MAIN_INIT} approved y-axis; "
                "the original limits/ticks were restored and export was stopped."
            )


def _standalone_lmg_rows(init, stage):
    g = _df_slim_lmg[(_df_slim_lmg["init_date"] == init) & (_df_slim_lmg["stage"] == stage)].copy()
    factor_col = _standalone_find_col(g, ["predictor", "factor", "factor_name"], "LMG factor column")
    pct_col = _standalone_find_col(g, ["LMG_percent", "LMG_importance_percent", "LMG_relative_importance_pct", "importance_percent"], "LMG percent column")
    g["factor_name"] = g[factor_col].astype(str)
    g["pct"] = pd.to_numeric(g[pct_col], errors="coerce")
    if g["pct"].max(skipna=True) <= 1.0:
        g["pct"] *= 100.0
    g["display_label"] = g["factor_name"].map(STANDALONE_FACTOR_DISPLAY).fillna(g["factor_name"])
    return g.sort_values("pct", ascending=False)


def _standalone_summary(init, stage):
    g = _df_slim_summary[(_df_slim_summary["init_date"] == init) & (_df_slim_summary["stage"] == stage)]
    if g.empty:
        raise RuntimeError(f"Standalone renderer missing summary row for {init}/{stage}")
    return g.iloc[0]


def _round_limits_to_step(limits, step):
    if limits is None:
        return None
    lo = np.floor(float(limits[0]) / step) * step
    hi = np.ceil(float(limits[1]) / step) * step
    if np.isclose(lo, hi):
        hi = lo + step
    return lo, hi


def _plot_fig3_scatter(ax, init, stage, limits=None):
    g = _df_slim_fitted[(_df_slim_fitted["init_date"] == init) & (_df_slim_fitted["stage"] == stage)].copy()
    if g.empty:
        raise RuntimeError(f"Standalone Fig.3 missing fitted rows for {init}/{stage}")
    y = pd.to_numeric(g["y_tmax"], errors="coerce").to_numpy(float)
    yhat = pd.to_numeric(g["y_fitted"], errors="coerce").to_numpy(float)
    ax.scatter(y, yhat, s=PAPER_STYLE["scatter_size"], color="#2f6db3", alpha=PAPER_STYLE["scatter_alpha"], edgecolor="white", linewidth=0.30)
    if limits is None:
        raw_limits = _pad_limits(np.r_[y, yhat], frac=0.04)
        raw_span = float(raw_limits[1] - raw_limits[0]) if raw_limits is not None else np.inf
        tick_step = 0.5 if (stage == "Stage-I_dry" and raw_span <= 5.0) else 1.0
        limits = _round_limits_to_step(raw_limits, tick_step)
    else:
        tick_step = 0.5 if stage == "Stage-I_dry" else 1.0
    ax.set_xlim(*limits); ax.set_ylim(*limits)
    s = _standalone_summary(init, stage)
    valid = np.isfinite(y) & np.isfinite(yhat)
    r_val = float(np.corrcoef(y[valid], yhat[valid])[0, 1]) if int(valid.sum()) >= 2 else np.nan
    if int(valid.sum()) >= 2 and float(np.nanstd(y[valid])) > 0:
        slope, intercept = [float(v) for v in np.polyfit(y[valid], yhat[valid], 1)]
        line_x = np.array([limits[0], limits[1]], dtype=float)
        ax.plot(line_x, slope * line_x + intercept, ls="-", color="0.38", lw=0.9, alpha=0.78)
    else:
        slope = np.nan
    ax.text(
        0.04,
        0.96,
        f"r = {r_val:.2f}\nR² = {float(s['R2']):.2f}\nslope = {slope:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PAPER_STYLE["annotation_size"],
        bbox=dict(facecolor="white", edgecolor="0.80", alpha=0.86, boxstyle="round,pad=0.25"),
    )
    ax.set_xlabel("ECMWF member Tmax (°C)"); ax.set_ylabel("Slim MLR predicted Tmax (°C)")
    _apply_scatter_axis_style(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))


def _plot_fig3_lmg(ax, init, stage):
    g = _standalone_lmg_rows(init, stage).sort_values("pct", ascending=True)
    ax.barh(g["display_label"], g["pct"], color="#6AA84F", alpha=0.86)
    xmax = max(float(g["pct"].max(skipna=True)), 1.0)
    ax.set_xlim(0, xmax)
    _pad_bar_xlim_for_labels(ax, g["pct"], pad_fraction=0.28)
    ax.set_xlabel("LMG relative importance (%)")
    for i, v in enumerate(g["pct"]):
        ax.text(
            float(v) + xmax * 0.025,
            i,
            f"{float(v):.1f}%",
            va="center",
            fontsize=PAPER_STYLE["bar_value_size"],
            fontweight="bold",
        )
    _apply_bar_axis_style(ax)


def _standalone_fig_prefix(fig_num, init, stem):
    if init == MAIN_INIT:
        return f"paper_fig{fig_num}_{stem}_init_{init}"
    if init == CONTRAST_INIT:
        return f"paper_figS{fig_num}_{stem}_init_{init}"
    return f"paper_figS_aux_{fig_num}_{stem}_init_{init}"


def _render_fig3(init):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(PAPER_FIG_WIDTH_IN, PAPER_FIG_HEIGHTS["Fig3"]),
        constrained_layout=False,
    )
    panel_letters = [["(a)", "(b)"], ["(c)", "(d)"]]
    for col, stage in enumerate(PAPER_MLR_STAGES):
        _plot_fig3_scatter(axes[0, col], init, stage)
        _plot_fig3_lmg(axes[1, col], init, stage)
        _add_panel_header(axes[0, col], panel_letters[0][col])
        _add_panel_header(axes[1, col], panel_letters[1][col])
        _capture_main_y_axis("Fig3", panel_letters[0][col], axes[0, col], init)
        _record_axis_style("Fig3", init, f"{stage} scatter", "scatter", "independent stage range; xlim equals ylim; rectangular axes")
        _record_axis_style("Fig3", init, f"{stage} LMG", "bar", "x-axis grid only")
    fig.subplots_adjust(left=0.155, right=0.985, top=0.885, bottom=0.105, wspace=0.58, hspace=0.48)
    _add_matrix_headers(
        fig,
        axes,
        ["Stage-I", "Total period"],
        ["MLR", "LMG"],
        column_y=0.965,
        row_x=0.018,
        row_header_sizes=[PAPER_STYLE["matrix_header_size"], PAPER_STYLE["fig3_lmg_row_header_size"]],
    )
    _verify_main_y_axes("Fig3", init)
    return _save_paper_figure(fig, _standalone_fig_prefix("3", init, "mlr_attribution"))


def _standalone_method_rows(init, stage):
    g = _df_method[(_df_method["init_date"] == init) & (_df_method["stage"] == stage)].copy()
    if g.empty:
        raise RuntimeError(f"Standalone Fig.4 missing method rows for {init}/{stage}")
    g["display_label"] = g["factor"].astype(str).map(STANDALONE_FACTOR_DISPLAY).fillna(g["factor"].astype(str))
    pos = np.maximum(pd.to_numeric(g["lofo_delta_r2"], errors="coerce"), 0.0)
    g["lofo_positive_pct"] = np.where(float(pos.sum()) > 0, pos / float(pos.sum()) * 100.0, 0.0)
    return g


def _render_fig4(init):
    fig, axes = plt.subplots(3, 2, figsize=(PAPER_FIG_WIDTH_IN, PAPER_FIG_HEIGHTS["Fig4"]), constrained_layout=False)
    method_specs = [
        ("LMG", "lmg_pct", PAPER_STYLE["method_colors"]["LMG"], "LMG relative importance (%)"),
        ("Johnson", "johnson_pct", PAPER_STYLE["method_colors"]["Johnson"], "Johnson relative weights (%)"),
        ("LOFO", "lofo_positive_pct", PAPER_STYLE["method_colors"]["LOFO"], "Normalized positive ΔR² (%)"),
    ]
    for col, stage in enumerate(PAPER_MLR_STAGES):
        methods = _standalone_method_rows(init, stage)
        order = _standalone_lmg_rows(init, stage)["factor_name"].tolist()
        methods = methods.set_index("factor").loc[order].reset_index()
        for row, (title, value_col, color, xlabel) in enumerate(method_specs):
            ax = axes[row, col]
            vals = pd.to_numeric(methods[value_col], errors="coerce").to_numpy(float)[::-1]
            labels = methods["display_label"].tolist()[::-1]
            xmax = max(float(np.nanmax(vals)) if len(vals) else 1.0, 1.0)
            ax.barh(labels, vals, color=color, alpha=0.88)
            ax.set_xlim(0, xmax)
            _pad_bar_xlim_for_labels(ax, vals, pad_fraction=0.28)
            ax.set_xlabel(xlabel)
            for i, v in enumerate(vals):
                ax.text(
                    float(v) + xmax * 0.025,
                    i,
                    f"{float(v):.1f}%",
                    va="center",
                    fontsize=PAPER_STYLE["bar_value_size"],
                    fontweight="bold",
                )
            _apply_bar_axis_style(ax)
            _add_panel_header(ax, f"({chr(ord('a') + row * 2 + col)})")
            _record_axis_style("Fig4", init, f"{title} {stage}", "bar", "x-axis grid only; per-panel x-range")
    fig.subplots_adjust(left=0.185, right=0.985, top=0.900, bottom=0.075, wspace=0.70, hspace=0.68)
    _add_matrix_headers(
        fig,
        axes,
        ["Stage-I", "Total period"],
        ["LMG", "Johnson", "LOFO"],
        column_y=0.975,
        row_x=0.018,
        row_header_sizes=[PAPER_STYLE["fig4_row_header_size"]] * 3,
    )
    return _save_paper_figure(fig, _standalone_fig_prefix("4", init, "mlr_robustness"))

def _stagei_coef_info(init, stagei_df):
    coef = _df_slim_coef.copy()
    coef["_stage"] = coef["stage"].map(_standalone_norm_stage)
    g = coef[(coef["init_date"].astype(str) == init) & (coef["_stage"] == "Stage-I")].copy()
    raw = dict(zip(g["predictor"].astype(str), pd.to_numeric(g["coef"], errors="coerce")))
    return {
        "beta_SM": float(raw["sm_avg"]),
        "beta_SHF": float(raw["SHF_resid"]),
        "intercept": float(raw.get("const", 0.0)),
        "sm_mean": float(stagei_df["sm_avg"].mean()),
        "shf_resid_mean": float(stagei_df["SHF_resid"].mean()),
        "raw_map": raw,
    }


def _add_lsm_variables(init, g):
    info = _stagei_coef_info(init, g)
    out = g.copy()
    out["LSM_SHF_partial_centered"] = info["beta_SM"] * (out["sm_avg"] - info["sm_mean"]) + info["beta_SHF"] * (out["SHF_resid"] - info["shf_resid_mean"])
    baseline = info["intercept"]
    used = []
    missing = []
    for pred, beta in info["raw_map"].items():
        if pred == "const":
            continue
        if pred in out.columns:
            baseline += float(beta) * float(pd.to_numeric(out[pred], errors="coerce").mean())
            used.append(pred)
        else:
            missing.append(pred)
    out["MLR_baseline_at_mean_state"] = baseline
    out["SM_SHF_partial_fitted_Tmax"] = baseline + out["LSM_SHF_partial_centered"]
    return out, info, {"baseline": baseline, "used": used, "missing": missing}


def _raw_panel(ax, init, stage, factor, start, end, title, ylabel):
    g = _df_raw_ts[(_df_raw_ts["_init"] == init) & (_df_raw_ts["_stage"] == stage) & (_df_raw_ts["_factor"] == factor) & (_df_raw_ts["_date"].between(pd.Timestamp(start), pd.Timestamp(end)))].sort_values("_date")
    if g.empty:
        raise RuntimeError(f"Standalone raw panel has no rows for {init}/{stage}/{factor}")
    if not g["era5"].isna().all():
        ax.plot(g["_date"], g["era5"], color=PAPER_STYLE["line_colors"]["ERA5 Obs"], lw=1.95, marker="o", ms=PAPER_STYLE["marker_size"], label="ERA5 Obs")
    ax.plot(g["_date"], g["hot17_mean"], color=PAPER_STYLE["line_colors"]["Hot17 Mean"], lw=PAPER_STYLE["line_width"], ls="-.", marker="^", ms=PAPER_STYLE["marker_size"], label="Hot17 Mean")
    ax.plot(g["_date"], g["middle17_mean"], color=PAPER_STYLE["line_colors"]["Middle17 Mean"], lw=PAPER_STYLE["line_width"], ls="--", marker="s", ms=PAPER_STYLE["marker_size"], label="Middle17 Mean")
    ax.plot(g["_date"], g["cold17_mean"], color=PAPER_STYLE["line_colors"]["Cold17 Mean"], lw=PAPER_STYLE["line_width"], ls="-.", marker="v", ms=PAPER_STYLE["marker_size"], label="Cold17 Mean")
    ax.axvline(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=12), color="0.35", lw=0.8, ls="--", alpha=0.65)
    if factor in _STANDALONE_ZERO_REF_FACTORS:
        ax.axhline(0, color="0.45", lw=0.7, alpha=0.45)
    ylim = _panel_timeseries_ylim(g, factor, ["era5", "hot17_mean", "middle17_mean", "cold17_mean"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title); ax.set_ylabel(ylabel); _format_daily_date_axis(ax, start, end); _apply_timeseries_axis_style(ax, factor); _apply_timeseries_major_spacing(ax, stage, factor)


def _lsm_daily_values(init, info, baseline, start, end):
    pieces = []
    for factor in ["sm_avg", "SHF_Avg"]:
        g = _df_raw_ts[(_df_raw_ts["_init"] == init) & (_df_raw_ts["_stage"] == "Stage-I") & (_df_raw_ts["_factor"] == factor) & (_df_raw_ts["_date"].between(pd.Timestamp(start), pd.Timestamp(end)))].copy()
        pieces.append(g[["_date", "hot17_mean", "middle17_mean", "cold17_mean"]].rename(columns={"hot17_mean": f"{factor}_hot17_mean", "middle17_mean": f"{factor}_middle17_mean", "cold17_mean": f"{factor}_cold17_mean"}))
    daily = pieces[0].merge(pieces[1], on="_date", how="inner").sort_values("_date")
    shf_x = pd.to_numeric(_df_stagei_member[_df_stagei_member["init_date"].astype(str) == init]["sm_avg"], errors="coerce")
    shf_y = pd.to_numeric(_df_stagei_member[_df_stagei_member["init_date"].astype(str) == init]["SHF_Avg"], errors="coerce")
    slope, intercept = np.polyfit(shf_x, shf_y, 1)
    values = {}
    for label, suffix, color, ls, marker in [
        ("Hot17 Mean", "hot17_mean", PAPER_STYLE["line_colors"]["Hot17 Mean"], "-.", "^"),
        ("Middle17 Mean", "middle17_mean", PAPER_STYLE["line_colors"]["Middle17 Mean"], "--", "s"),
        ("Cold17 Mean", "cold17_mean", PAPER_STYLE["line_colors"]["Cold17 Mean"], "-.", "v"),
    ]:
        sm_daily = pd.to_numeric(daily[f"sm_avg_{suffix}"], errors="coerce")
        shf_daily = pd.to_numeric(daily[f"SHF_Avg_{suffix}"], errors="coerce")
        shf_resid_daily = shf_daily - (intercept + slope * sm_daily)
        values[label] = baseline + info["beta_SM"] * (sm_daily - info["sm_mean"]) + info["beta_SHF"] * (shf_resid_daily - info["shf_resid_mean"])
    return daily["_date"], values


def _lsm_daily_panel(ax, init, info, baseline, start, end, title, ylabel, factor_key):
    dates, values = _lsm_daily_values(init, info, baseline, start, end)
    for label, color, ls, marker in [
        ("Hot17 Mean", PAPER_STYLE["line_colors"]["Hot17 Mean"], "-.", "^"),
        ("Middle17 Mean", PAPER_STYLE["line_colors"]["Middle17 Mean"], "--", "s"),
        ("Cold17 Mean", PAPER_STYLE["line_colors"]["Cold17 Mean"], "-.", "v"),
    ]:
        vals = values[label]
        ax.plot(dates, vals, color=color, lw=PAPER_STYLE["line_width"], ls=ls, marker=marker, ms=PAPER_STYLE["marker_size"], label=label)
    ax.axvline(pd.Timestamp("2023-06-17") + pd.Timedelta(hours=12), color="0.35", lw=0.8, ls="--", alpha=0.65)
    if factor_key in _STANDALONE_ZERO_REF_FACTORS:
        ax.axhline(0, color="0.45", lw=0.7, alpha=0.45)
    panel_vals = []
    for series in values.values():
        panel_vals.extend(pd.to_numeric(series, errors="coerce").tolist())
    ylim = _pad_limits(panel_vals)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title); ax.set_ylabel(ylabel); _format_daily_date_axis(ax, start, end); _apply_timeseries_axis_style(ax, factor_key); _apply_timeseries_major_spacing(ax, "Stage-I", factor_key)


def _add_scatter_stats(ax, r_value, r2_value, position):
    x_frac, y_frac = [float(v) for v in position]
    ax.text(
        x_frac,
        y_frac,
        f"r={r_value:.2f}\nR²={r2_value:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PAPER_STYLE["annotation_size"],
        fontweight="normal",
        zorder=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, boxstyle="round,pad=0.15"),
    )


def _fig6_scatter_stats_position(init, panel):
    # A shared axes-relative information-block layout is used for all four inits.
    return FIG6_SCATTER_STATS_POSITIONS[panel]


def _scatter_panel(ax, g, x_col, y_col, title, stats_position):
    for group in _STANDALONE_GROUP_ORDER:
        gg = g[g["hotcold_group"] == group]
        ax.scatter(gg[x_col], gg[y_col], s=PAPER_STYLE["scatter_size"], alpha=0.66, color=_STANDALONE_GROUP_COLORS[group], edgecolor="white", linewidth=0.35, label=group)
    x = pd.to_numeric(g[x_col], errors="coerce"); y = pd.to_numeric(g[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(float(x[mask].min()), float(x[mask].max()), 80)
    ax.plot(xs, intercept + slope * xs, color="black", lw=1.1, alpha=0.75)
    r = float(pearsonr(x[mask], y[mask])[0]); r2 = r * r
    _add_scatter_stats(ax, r, r2, stats_position)
    ax.set_title(title); ax.set_xlabel(STANDALONE_FACTOR_DISPLAY.get(x_col, x_col)); ax.set_ylabel(STANDALONE_FACTOR_DISPLAY.get(y_col, y_col)); _apply_scatter_axis_style(ax)
    if x_col == "sm_avg":
        ax.xaxis.set_major_locator(MultipleLocator(0.005))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    if x_col in {"LSM_SHF_partial_centered", "SM_SHF_partial_fitted_Tmax"}:
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if y_col == "y_tmax":
        ax.yaxis.set_major_locator(MultipleLocator(0.5))


def _render_fig6_stageI_land_surface(init, fitted_component=True):
    g = _df_stagei_member[_df_stagei_member["init_date"].astype(str) == init].copy()
    g, info, base = _add_lsm_variables(init, g)
    fig, axes = plt.subplots(4, 2, figsize=(PAPER_FIG_WIDTH_IN, PAPER_FIG_HEIGHTS["Fig6"]), constrained_layout=False)
    flat = axes.ravel()
    if fitted_component:
        _lsm_daily_panel(flat[0], init, info, base["baseline"], "2023-06-14", "2023-06-18", "SM–SHF", "SM–SHF (°C)", "SM_SHF_partial_fitted_Tmax")
        scatter_x, scatter_title = "SM_SHF_partial_fitted_Tmax", "SM–SHF vs Stage-I Tmax"
        suffix = ""
    else:
        _lsm_daily_panel(flat[0], init, info, 0.0, "2023-06-14", "2023-06-18", "SM–SHF", "SM–SHF contribution (°C)", "LSM_SHF_partial_centered")
        scatter_x, scatter_title = "LSM_SHF_partial_centered", "SM–SHF vs Stage-I Tmax"
        suffix = "_centered_contribution_qc"
    figure_id = "Fig6" if fitted_component else "Fig6_aux_centered_contribution"
    _record_axis_style(figure_id, init, "(a) SM-SHF", "timeseries", "panel/init-specific y-limits")
    for ax, (factor, title, ylabel) in zip(flat[1:6], [("sm_avg", "SM", "Soil moisture (m³ m⁻³)"), ("z500_anom_NCHN", "Z500 anomaly", "Z500 anomaly (gpm)"), ("SHF_Avg", "SHF", "W m⁻²"), ("NCVI_conc", "NCVI anomaly", "NCVI anomaly (PVU)"), ("TCC_Avg", "TCC", "TCC (fraction)")]):
        _raw_panel(ax, init, "Stage-I", factor, "2023-06-14", "2023-06-18", title, ylabel)
        _record_axis_style(figure_id, init, title, "timeseries", f"{factor}; panel/init-specific y-limits")
    _scatter_panel(flat[6], g, "sm_avg", "y_tmax", "SM vs Stage-I Tmax", _fig6_scatter_stats_position(init, "g"))
    _scatter_panel(flat[7], g, scatter_x, "y_tmax", scatter_title, _fig6_scatter_stats_position(init, "h"))
    _record_axis_style(figure_id, init, "(g) SM scatter", "scatter", "subtle x/y grid")
    _record_axis_style(figure_id, init, "(h) SM-SHF scatter", "scatter", "subtle x/y grid")
    flat[6].legend(
        loc=FIG6_SCATTER_GROUP_LEGEND["loc"],
        bbox_to_anchor=FIG6_SCATTER_GROUP_LEGEND["bbox_to_anchor"],
        frameon=False,
        fontsize=PAPER_STYLE["legend_size"],
        borderaxespad=0.0,
    )
    for ax, letter in zip(flat, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]):
        _add_panel_header(ax, letter, ax.get_title())
    for ax, letter in zip(flat, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]):
        _capture_main_y_axis("Fig6", letter, ax, init)
    handles, labels = flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=4, fontsize=PAPER_STYLE["legend_size"], frameon=False, columnspacing=1.0, handlelength=1.7)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.965, bottom=0.105, wspace=0.34, hspace=0.56)
    _verify_main_y_axes("Fig6", init)
    stem = f"stageI_land_surface_evolution{suffix}"
    return _save_paper_figure(fig, _standalone_fig_prefix("6", init, stem))


def _render_fig5_total_process_chain(init):
    fig, axes = plt.subplots(3, 2, figsize=(PAPER_FIG_WIDTH_IN, PAPER_FIG_HEIGHTS["Fig5"]), constrained_layout=False)
    flat = axes.ravel()
    for ax, (factor, title, ylabel) in zip(flat[:5], [("NCHN_tp", "TP", "Precipitation (mm day⁻¹)"), ("z500_anom_NCHN", "Z500 anomaly", "Z500 anomaly (gpm)"), ("sm_avg", "SM", "Soil moisture (m³ m⁻³)"), ("NCVI_conc", "NCVI anomaly", "NCVI anomaly (PVU)"), ("TCC_Avg", "TCC", "TCC (fraction)")]):
        _raw_panel(ax, init, "Total", factor, "2023-06-14", "2023-06-24", title, ylabel)
        _record_axis_style("Fig5", init, title, "timeseries", f"{factor}; panel/init-specific y-limits")
    note = flat[5]
    note.axis("off")
    handles, labels = flat[0].get_legend_handles_labels()
    boundary_handle = Line2D([0], [0], color="0.35", lw=0.8, ls="--", alpha=0.65, label="Stage-I boundary")
    note.legend(
        handles + [boundary_handle],
        labels + ["Stage-I boundary"],
        loc="center left",
        bbox_to_anchor=(0.08, 0.52),
        frameon=False,
        fontsize=PAPER_STYLE["legend_size"],
        handlelength=2.8,
        labelspacing=0.95,
        borderaxespad=0.0,
    )
    _record_axis_style("Fig5", init, "right-bottom legend panel", "legend_only", "axis off; compact legend only")
    for ax, letter in zip(flat[:5], ["(a)", "(b)", "(c)", "(d)", "(e)"]):
        _add_panel_header(ax, letter, ax.get_title())
        _capture_main_y_axis("Fig5", letter, ax, init)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.955, bottom=0.085, wspace=0.34, hspace=0.48)
    _verify_main_y_axes("Fig5", init)
    return _save_paper_figure(fig, _standalone_fig_prefix("5", init, "total_process_chain"))


for _init in [MAIN_INIT, CONTRAST_INIT, *AUX_INITS]:
    for _stage in PAPER_MLR_STAGES:
        _n_fitted = len(_df_slim_fitted[(_df_slim_fitted["init_date"] == _init) & (_df_slim_fitted["stage"] == _stage)])
        if _n_fitted != 51:
            raise RuntimeError(f"Standalone renderer expected 51 fitted members for {_init}/{_stage}, found {_n_fitted}")
        _standalone_qc.append(f"fitted member count {_init}/{_stage} = {_n_fitted}")
    _n_stagei = len(_df_stagei_member[_df_stagei_member["init_date"].astype(str) == _init])
    if _n_stagei != 51:
        raise RuntimeError(f"Standalone renderer expected 51 Stage-I audit members for {_init}, found {_n_stagei}")
    _standalone_qc.append(f"Stage-I audit member count {_init} = {_n_stagei}")
    for _figure_id, _renderer, _sources in [
        ("Fig3", _render_fig3, FIG34_INPUTS),
        ("Fig4", _render_fig4, FIG34_INPUTS),
        ("Fig5", _render_fig5_total_process_chain, FIG56_INPUTS),
        ("Fig6", lambda i: _render_fig6_stageI_land_surface(i, fitted_component=True), FIG56_INPUTS),
    ]:
        _png, _pdf = _renderer(_init)
        _meta = _saved_figure_metadata.get(os.path.splitext(os.path.basename(_png))[0], {})
        _standalone_qc.append(
            f"exported {_figure_id} {_init}: "
            f"figsize=({_meta.get('figsize_width_in')}, {_meta.get('figsize_height_in')}); "
            f"png_pixels=({_meta.get('png_width_px')}, {_meta.get('png_height_px')})"
        )
        _manifest_rows.append({
            "figure_id": _figure_id,
            "init_date": _init,
            "role": _standalone_role(_init),
            "source_tables": ";".join(_sources.values()),
            "output_png": _png,
            "output_pdf": _pdf,
            "exists_png": os.path.exists(_png),
            "exists_pdf": os.path.exists(_pdf),
            "pdf_exists": os.path.exists(_pdf),
            "width_in": _meta.get("figsize_width_in"),
            "height_in": _meta.get("figsize_height_in"),
            "figsize_width_in": _meta.get("figsize_width_in"),
            "figsize_height_in": _meta.get("figsize_height_in"),
            "png_width_px": _meta.get("png_width_px"),
            "png_height_px": _meta.get("png_height_px"),
        })

_expected_main_y_axis_lock_rows = 2 + 5 + 8
_main_y_axis_all_unchanged = (
    len(_main_y_axis_lock_rows) == _expected_main_y_axis_lock_rows
    and all(bool(row["unchanged"]) for row in _main_y_axis_lock_rows)
)
_standalone_qc.append(f"main_init_y_axis_lock_rows = {len(_main_y_axis_lock_rows)}")
_standalone_qc.append(f"main_init_y_axis_all_unchanged = {_main_y_axis_all_unchanged}")
if not _main_y_axis_all_unchanged:
    raise RuntimeError(
        "Standalone main-init y-axis lock QC failed: "
        f"expected {_expected_main_y_axis_lock_rows} unchanged panel rows, "
        f"found {len(_main_y_axis_lock_rows)}."
    )

_primary_png_width_values = []
for _row in _manifest_rows:
    _width = _row.get("png_width_px")
    if _width is not None and _width == _width:
        _primary_png_width_values.append(int(_width))
_primary_png_width_values = sorted(set(_primary_png_width_values))
_primary_png_width_consistent = _primary_png_width_values == [3900]
_standalone_qc.append(f"primary_png_width_consistent = {_primary_png_width_consistent}")
_standalone_qc.append(f"primary_png_width_values = {_primary_png_width_values}")
if not _primary_png_width_consistent:
    raise RuntimeError(f"Standalone primary PNG width check failed: expected [3900], found {_primary_png_width_values}")
for _row in _manifest_rows:
    _row["primary_png_width_consistent"] = _primary_png_width_consistent
    _row["primary_png_width_values"] = ";".join(str(v) for v in _primary_png_width_values)

_manifest_path = _standalone_assert_output(f"{PAPER_RENDER_TABLE_DIR}/table_standalone_paper_figures_manifest.csv")
_axis_style_qc_path = _standalone_assert_output(f"{PAPER_RENDER_TABLE_DIR}/table_standalone_paper_axis_style_qc.csv")
_main_y_axis_qc_path = _standalone_assert_output(f"{PAPER_RENDER_TABLE_DIR}/table_standalone_main_init_y_axis_lock_qc.csv")
_qc_path = _standalone_assert_output(f"{PAPER_RENDER_QC_DIR}/qc_standalone_paper_figures.txt")
_caption_path = _standalone_assert_output(f"{PAPER_RENDER_QC_DIR}/captions_fig3_fig6_final.md")
_standalone_qc.append(f"Final caption file = {_caption_path}")
pd.DataFrame(_manifest_rows).to_csv(_manifest_path, index=False)
pd.DataFrame(_axis_style_rows).to_csv(_axis_style_qc_path, index=False)
pd.DataFrame(_main_y_axis_lock_rows).to_csv(_main_y_axis_qc_path, index=False)
_caption_text = """# Final captions for standalone manuscript Figures 3–6

## Figure 3

**Multiple-linear-regression attribution of ensemble-member Tmax.** The upper row compares the target ECMWF ensemble-member mean Tmax with the corresponding slim multiple-linear-regression (MLR) fitted Tmax for 51 members, and the lower row shows LMG relative importance for the predictors retained in the same model. Columns show models fitted independently for Stage-I (14–17 June 2023) and the total period (14–24 June 2023). Within each initialization and stage, predictors were standardized before fitting; the plotted fitted values are in the response space (°C). The gray solid line in each scatter panel is a least-squares fitted line and is not a 1:1 reference line. Annotations report Pearson's r, model R², and the fitted-line slope; no confidence interval is shown. LMG partitions the complete-model R² among the correlated predictors by averaging incremental R² over predictor-entry orders. The two LMG panels use panel-specific x-axis ranges; cross-panel comparisons should therefore use the printed percentages and ranking rather than apparent bar length alone.

## Figure 4

**Robustness comparison of factor-importance diagnostics.** Rows show LMG relative importance, Johnson relative weights, and normalized positive leave-one-factor-out (LOFO) ΔR²; columns show Stage-I and the total period for the same exported slim-model results. Johnson relative weights use an orthogonal-predictor representation and map explained variation back to the original predictors, whereas LMG averages incremental R² over predictor-entry orders. For LOFO, negative ΔR² values are truncated at zero and the remaining positive ΔR² values are normalized to sum to 100%. Thus, LOFO is a complementary deletion-sensitivity diagnostic rather than the same variance decomposition as LMG. Each panel uses its own x-axis range. Comparisons across panels should be based on the printed values and factor ordering; exact agreement among methods is not assumed.

## Figure 5

**Total-period precipitation–circulation process diagnostics.** Panels show (a) total precipitation (mm day⁻¹), (b) Z500 anomaly (gpm), (c) soil moisture (m³ m⁻³), (d) NCVI anomaly (PVU), and (e) total cloud cover (fraction) from 14–24 June 2023 in Beijing time (BJT). The black line denotes ERA5 observations where available; colored lines denote means of the fixed Hot17, Middle17, and Cold17 groups (17 ECMWF ensemble members each), defined from the prescribed total-period regional-mean Tmax ranking in the exported grouping table. The gray dashed line at 17 June 12:00 marks the plotted boundary between Stage-I and the subsequent period. The panels diagnose co-evolution among the displayed quantities and do not by themselves establish a causal chain.

## Figure 6

**Stage-I land-surface thermal pathway diagnostics.** Panels show (a) the SM–SHF partial fitted Tmax component in the original Tmax space (°C), (b) soil moisture (m³ m⁻³), (c) Z500 anomaly (gpm), (d) sensible heat flux (W m⁻²), (e) NCVI anomaly (PVU), (f) total cloud cover (fraction), (g) member-mean soil moisture versus Stage-I Tmax, and (h) the SM–SHF partial fitted Tmax component versus Stage-I Tmax. The Stage-I statistics use 14–17 June 2023, while the time-series panels extend through 18 June to display the transition and retain the gray dashed boundary at 17 June 12:00 BJT. For each initialization, the centered term is C_SM–SHF = β_SM[SM − mean(SM)] + β_SHF[SHF_resid − mean(SHF_resid)], using exported raw-equivalent coefficients from the Stage-I slim MLR. The mean-state baseline is B = β₀ + Σ_k β_k mean(X_k), evaluated over the available full slim-model predictor set, and the plotted quantity is B + C_SM–SHF. SM is the Stage-I member-mean soil-moisture predictor, and SHF_resid is the portion of member-mean SHF remaining after the specified member-axis linear adjustment for SM across the 51-member Stage-I sample. The plotted component is in °C because the raw-equivalent coefficients and baseline map predictors to the absolute-Tmax response space. It is a baseline-anchored partial fitted diagnostic, not observed Tmax, the complete fitted Tmax, or an independent causal contribution. Each scatter point is one ECMWF ensemble member; colors identify the fixed Hot17, Middle17, and Cold17 groups (17 members each) from the prescribed Stage-I regional-mean Tmax ranking. Solid black lines are least-squares fits. The annotations give the member-axis Pearson correlation r and its square, R²; no confidence interval is shown.
"""
with open(_caption_path, "w", encoding="utf-8") as _f:
    _f.write(_caption_text)
with open(_qc_path, "w", encoding="utf-8") as _f:
    _f.write("\n".join(_standalone_qc))

print(f"[Standalone Fig.3-6 renderer] output root = {PAPER_RENDER_ROOT}")
print(f"[Standalone Fig.3-6 renderer] manifest = {_manifest_path}")
print(f"[Standalone Fig.3-6 renderer] axis style QC = {_axis_style_qc_path}")
print(f"[Standalone Fig.3-6 renderer] main-init y-axis lock QC = {_main_y_axis_qc_path}")
print(f"[Standalone Fig.3-6 renderer] QC = {_qc_path}")
print(f"[Standalone Fig.3-6 renderer] captions = {_caption_path}")
print("[Standalone Fig.3-6 renderer] No upstream data calculation performed = True")
