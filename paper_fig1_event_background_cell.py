# =============================================================================
# [Independent Cell 1] Paper Figure 1: Event evolution, observed stage definition,
# and historical extremeness
#
# This cell is intentionally independent from the Fig.2a / S2S / Hot17-Cold17
# workflow. It uses ERA5 observed Tmax and MSWEP observed precipitation only.
# =============================================================================

import glob
import importlib.util
import os
import struct

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import xarray as xr
if importlib.util.find_spec("scipy") is not None and importlib.util.find_spec("scipy.stats") is not None:
    from scipy.stats import gaussian_kde
else:
    gaussian_kde = None

if importlib.util.find_spec("cartopy") is not None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
else:
    ccrs = None
    cfeature = None
    LatitudeFormatter = None
    LongitudeFormatter = None

# -------------------------
# Fig.1 configuration
# -------------------------
PAPER_FIG1_DIR = "/data1/huangy/fig6/NC/v9/paper_figures/fig1/event_background_historical_pdf"
# The project plotting-only paper outputs use /v12 as the final-figure directory.
# Legacy Fig.1 CSV/QC products remain in PAPER_FIG1_DIR for compatibility, while
# the new manuscript figure, map cache, caption, and figure QC are written here.
PAPER_FIG1_JGR_DIR = "/data1/huangy/fig6/NC/v12"
ERA5_TMAX_FILE = "/data1/huangy/fig6/NC/ai_test/pangu/era5_T/era5_2m_temperature_6_2023.nc"
ERA5_TMAX_HIST_GLOB_1993_2023 = "/data1/huangy/fig6/NC/ai_test/pangu/era5_T/era5_2m_temperature_6_*.nc"
ERA5_TMAX_HIST_ROOT_1979_1992 = "/data6/wangsg/ERA5_meso2/monolevel/sfc_1hour/t2m"
MSWEP_ROOTS = {
    "past": "/data1/huangy/fig6/NC/MSE/MSWEP/Past_3hourly_2003_2020",
    "nrt": "/data1/huangy/fig6/NC/MSE/MSWEP/NRT_3hourly_2021_2022",
    "2023": "/data1/huangy/fig6/NC/MSE/MSWEP/3hourly-2023",
}

# Explicit Fig.1 NCHN box order: lon_min, lon_max, lat_min, lat_max
FIG1_NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
FIG1_TARGET_DATES = pd.date_range("2023-06-14", "2023-06-24", freq="D")
FIG1_WINDOWS = {
    "Total": ("06-14", "06-24"),
    "Stage-I": ("06-14", "06-17"),
    "Stage-II": ("06-18", "06-24"),
}
FIG1_STAGE_DATES = {
    "Stage-I": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-17")),
    "Stage-II": (pd.Timestamp("2023-06-18"), pd.Timestamp("2023-06-24")),
    "Total": (pd.Timestamp("2023-06-14"), pd.Timestamp("2023-06-24")),
}
FIG1_HIST_BACKGROUND_YEARS = range(1979, 2023)
FIG1_EVENT_YEAR = 2023

# No shared map extent/projection helper is present in this repository.  This
# explicit North-China context extent is therefore local to Fig.1 and is not
# inferred from an arbitrary raw-data domain.  The NCHN analysis box is drawn
# separately and is not changed by this plotting extent.
FIG1_MAP_EXTENT = (108.0, 125.0, 30.0, 48.0)  # lon_min, lon_max, lat_min, lat_max

# Final 6.5-inch JGR manuscript style (locally applied with plt.rc_context).
FIG1_WIDTH_IN = 6.5
FIG1_HEIGHT_IN = 6.15
FIG1_DPI = 600
NCHN_BOX_COLOR = "#2F4B7C"
FIG1_COLORBAR_RELATIVE_WIDTH = "78%"
FONT_PANEL_LETTER = 10.8
FONT_PANEL_TITLE = 9.6
FONT_AXIS_LABEL = 9.5
FONT_TICK = 8.7
FONT_LEGEND = 8.5
FONT_ANNOTATION = 8.4
FONT_COLORBAR = 8.7
FIG1_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _guess_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return ds[v]
    raise KeyError(f"Cannot find variable in {list(ds.data_vars)}; candidates={candidates}")


def _normalize_time_dim(da):
    if "time" in da.dims:
        return da
    if "valid_time" in da.dims:
        return da.rename({"valid_time": "time"})
    if "time" in da.coords and "time" not in da.dims:
        return da.expand_dims(time=da["time"].values)
    if "valid_time" in da.coords and "valid_time" not in da.dims:
        vt = da["valid_time"].values
        da = da.drop_vars("valid_time").expand_dims(time=[vt])
        return da
    raise KeyError(f"No time/valid_time found. dims={da.dims}, coords={list(da.coords)}")


def _standardize_lat_lon(da):
    rename = {}
    if "lat" not in da.dims and "lat" not in da.coords and ("latitude" in da.dims or "latitude" in da.coords):
        rename["latitude"] = "lat"
    if "lon" not in da.dims and "lon" not in da.coords and ("longitude" in da.dims or "longitude" in da.coords):
        rename["longitude"] = "lon"
    da = da.rename(rename) if rename else da
    if "lat" not in da.coords or "lon" not in da.coords:
        raise KeyError(f"Cannot find lat/lon after standardization. dims={da.dims}, coords={list(da.coords)}")
    lon = da["lon"]
    if float(lon.max().values) > 180.0:
        da = da.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
    if float(da["lat"][0].values) > float(da["lat"][-1].values):
        da = da.sortby("lat")
    return da


def _to_bjt_daily(da, how="sum"):
    da = da.assign_coords(time=pd.to_datetime(da.time.values) + pd.Timedelta(hours=8))
    if how == "sum":
        return da.resample(time="1D").sum(skipna=True)
    if how == "mean":
        return da.resample(time="1D").mean(skipna=True)
    if how == "max":
        return da.resample(time="1D").max(skipna=True)
    raise ValueError(how)


def _count_samples_per_bjt_day(time_index, target_dates):
    shifted = pd.DatetimeIndex(pd.to_datetime(time_index)) + pd.Timedelta(hours=8)
    s = pd.Series(1, index=shifted).resample("1D").sum()
    out = {}
    for d in pd.to_datetime(target_dates):
        out[d.strftime("%Y-%m-%d")] = int(s.get(d, 0))
    return out


def _subset_nchn(da, box=FIG1_NCHN_BOX):
    lon_min, lon_max, lat_min, lat_max = box
    return da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))


def _coslat_weighted_mean_fig1(da):
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(dim=["lat", "lon"])


def _to_celsius_if_needed_fig1(da):
    return da - 273.15 if float(da.mean()) > 150 else da


def _stage_for_date_fig1(dt):
    ts = pd.Timestamp(dt)
    if FIG1_STAGE_DATES["Stage-I"][0] <= ts <= FIG1_STAGE_DATES["Stage-I"][1]:
        return "Stage-I"
    if FIG1_STAGE_DATES["Stage-II"][0] <= ts <= FIG1_STAGE_DATES["Stage-II"][1]:
        return "Stage-II"
    return "Outside"


def _open_era5_tmax_file(path):
    ds = xr.open_dataset(path)
    da = _guess_var(ds, ["t2m", "2m_temperature", "air", "air_temperature", "T2M"])
    da.attrs["_fig1_variable_used"] = da.name
    da = _normalize_time_dim(da)
    da = _standardize_lat_lon(da)
    return da


def _find_mswep_2023_files():
    root = MSWEP_ROOTS["2023"]
    patterns = [
        os.path.join(root, "**", "*.nc"),
        os.path.join(root, "**", "*.nc4"),
        os.path.join(root, "**", "*.nc.gz"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    june_files = [f for f in files if ("202306" in os.path.basename(f) or "2023-06" in os.path.basename(f) or "/06/" in f)]
    selected = june_files if june_files else files
    if not selected:
        attempted = patterns
        print(f"[Fig1 QC] No MSWEP 2023 files found. Attempted: {attempted}")
        raise FileNotFoundError(f"No MSWEP 2023 precipitation files found. Attempted: {attempted}")
    return selected


def load_mswep_precip_daily_bjt_nchn():
    files = _find_mswep_2023_files()
    daily_inputs = []
    original_units = "unknown"
    unit_note = "unknown: no conversion applied"
    raw_times = []
    for path in files:
        ds = xr.open_dataset(path)
        da = _guess_var(ds, ["precipitation", "precip", "tp", "rain", "pcp"])
        da = _normalize_time_dim(da)
        da = _standardize_lat_lon(da)
        original_units = str(da.attrs.get("units", original_units))
        units_lower = original_units.lower()
        if units_lower in ["m", "meter", "metre", "meters", "metres"]:
            da = da * 1000.0
            da.attrs["units"] = "mm"
            unit_note = "m to mm (*1000)"
        elif units_lower in ["mm/3h", "mm 3h-1", "mm per 3h", "mm/3hr", "mm per 3hr"]:
            unit_note = "kept as mm/3h; daily sum gives mm day-1"
        elif units_lower in ["mm", "millimeter", "millimeters", "kg m-2", "kg m**-2", "kg/m^2", "kg m^-2"]:
            unit_note = f"kept as {original_units}"
        else:
            print(f"[Fig1 QC] Warning: unknown MSWEP precipitation units={original_units}; no conversion applied.")
        raw_times.extend(pd.to_datetime(da.time.values).tolist())
        daily_inputs.append(da)

    da_all = xr.concat(daily_inputs, dim="time").sortby("time")
    da_all = da_all.isel(time=~da_all.get_index("time").duplicated())
    sample_counts = _count_samples_per_bjt_day(da_all.time.values, FIG1_TARGET_DATES)
    da_daily = _to_bjt_daily(da_all, how="sum")
    da_daily = _subset_nchn(da_daily)
    da_daily = _coslat_weighted_mean_fig1(da_daily)
    da_daily = da_daily.sel(time=slice(FIG1_TARGET_DATES.min(), FIG1_TARGET_DATES.max())).sortby("time")
    return da_daily, {"source": "MSWEP", "files": files, "original_units": original_units, "unit_note": unit_note, "sample_counts": sample_counts}




def find_era5_t2m_files_for_year(year):
    if 1993 <= year <= 2023:
        direct = f"/data1/huangy/fig6/NC/ai_test/pangu/era5_T/era5_2m_temperature_6_{year}.nc"
        files = [direct] if os.path.exists(direct) else []
        if not files:
            files = [f for f in sorted(glob.glob(ERA5_TMAX_HIST_GLOB_1993_2023)) if str(year) in os.path.basename(f)]
        return sorted(set(files)), "1993_2023_glob"

    root = ERA5_TMAX_HIST_ROOT_1979_1992
    direct = os.path.join(root, f"air.2m.{year}06.nc")
    if os.path.exists(direct):
        print(f"[Fig1 QC] Year {year} direct monthly file found: {direct}")
        return [direct], "1979_1992_root"

    print(f"[Fig1 QC] Year {year} direct monthly file missing, fallback recursive search used")
    patterns = [
        os.path.join(root, "**", f"air.2m.{year}06.nc"),
        os.path.join(root, "**", f"*{year}06*.nc"),
        os.path.join(root, "**", f"*{year}*.nc"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files)), "1979_1992_root"


def _open_era5_tmax_files_for_year(paths, year):
    pieces = []
    variables = []
    units = []
    utc_start = pd.Timestamp(f"{year}-06-13 00:00")
    utc_end = pd.Timestamp(f"{year}-06-24 23:59")
    for path in paths:
        da = _open_era5_tmax_file(path)
        variables.append(str(da.attrs.get("_fig1_variable_used", da.name)))
        units.append(str(da.attrs.get("units", "unknown")))
        da = da.sel(time=slice(utc_start, utc_end))
        if da.sizes.get("time", 0) > 0:
            pieces.append(da)
    if not pieces:
        raise RuntimeError(f"No usable June 13–24 UTC t2m data found for {year}; files={paths}")
    da_year = xr.concat(pieces, dim="time").sortby("time")
    da_year = da_year.isel(time=~da_year.get_index("time").duplicated())
    metadata = {
        "variables": sorted(set(variables)),
        "units": sorted(set(units)),
    }
    return da_year, metadata


def load_era5_tmax_daily_nchn_for_year(year):
    files, source_group = find_era5_t2m_files_for_year(year)
    if not files:
        print(f"[Fig1 QC] Missing ERA5 t2m file for year {year}")
        raise FileNotFoundError(f"Missing ERA5 t2m file for year {year}")

    da, meta = _open_era5_tmax_files_for_year(files, year)
    raw_mean_before_unit_conversion = float(da.mean())
    da = _to_celsius_if_needed_fig1(da)
    mean_after_unit_conversion = float(da.mean())
    da = _subset_nchn(da)
    da = _to_bjt_daily(da, how="max")
    da = da.sel(time=slice(pd.Timestamp(f"{year}-06-14"), pd.Timestamp(f"{year}-06-24"))).sortby("time")
    da = _coslat_weighted_mean_fig1(da)

    expected_dates = pd.date_range(f"{year}-06-14", f"{year}-06-24", freq="D")
    got_dates = pd.DatetimeIndex(pd.to_datetime(da.time.values).normalize())
    missing_dates = [d.strftime("%Y-%m-%d") for d in expected_dates if d not in got_dates]
    ok_11_days = len(missing_dates) == 0 and da.sizes.get("time", 0) == 11
    if not ok_11_days:
        print(f"[Fig1 QC] Year {year} daily Tmax missing BJT dates: {missing_dates}; n_days={da.sizes.get('time', 0)}")
        raise RuntimeError(f"Year {year} does not have complete BJT daily Tmax for 06-14–06-24")

    qc_year = {
        "year": int(year),
        "source_group": source_group,
        "n_files": int(len(files)),
        "files": files,
        "data_variable_used": ";".join(meta["variables"]),
        "units": ";".join(meta["units"]),
        "raw_mean_before_unit_conversion": raw_mean_before_unit_conversion,
        "mean_after_unit_conversion": mean_after_unit_conversion,
        "n_days": int(da.sizes.get("time", 0)),
        "date_start": str(pd.to_datetime(da.time.values).min())[:10],
        "date_end": str(pd.to_datetime(da.time.values).max())[:10],
        "complete_11_days": bool(ok_11_days),
    }
    return da, qc_year


def load_era5_tmax_historical_daily_nchn():
    daily_list = []
    source_records = []
    missing_years = []
    for year in range(min(FIG1_HIST_BACKGROUND_YEARS), FIG1_EVENT_YEAR + 1):
        try:
            da_year, qc_year = load_era5_tmax_daily_nchn_for_year(year)
        except FileNotFoundError:
            missing_years.append(year)
            continue
        daily_list.append(da_year)
        source_records.append(qc_year)

    if missing_years:
        print(f"[Fig1 QC] Missing ERA5 Tmax years: {missing_years}")
        raise RuntimeError(f"Missing ERA5 Tmax years for 1979–2023 Fig.1 baseline: {missing_years}")
    if not daily_list:
        raise RuntimeError("No ERA5 Tmax daily data loaded for Fig.1 historical baseline.")

    tmax_hist_daily = xr.concat(daily_list, dim="time").sortby("time")
    tmax_hist_daily = tmax_hist_daily.isel(time=~tmax_hist_daily.get_index("time").duplicated())
    years = sorted(pd.DatetimeIndex(pd.to_datetime(tmax_hist_daily.time.values)).year.unique().tolist())
    required_years = set(range(min(FIG1_HIST_BACKGROUND_YEARS), FIG1_EVENT_YEAR + 1))
    missing_years = sorted(required_years - set(years))
    if missing_years:
        print(f"[Fig1 QC] Missing ERA5 Tmax years: {missing_years}")
        raise RuntimeError(f"Missing ERA5 Tmax years for 1979–2023 Fig.1 baseline: {missing_years}")
    return tmax_hist_daily, {"source_records": source_records, "years": years, "missing_years": missing_years}


def load_era5_total_period_mean_tmax_map_2023():
    """Return the mean of 11 BJT grid-point daily Tmax fields for 14–24 June 2023."""
    da = _open_era5_tmax_file(ERA5_TMAX_FILE)
    da = da.sel(time=slice(pd.Timestamp("2023-06-13 00:00"), pd.Timestamp("2023-06-24 23:59")))
    da = _to_celsius_if_needed_fig1(da)
    daily_tmax = _to_bjt_daily(da, how="max")
    daily_tmax = daily_tmax.sel(
        time=slice(FIG1_TARGET_DATES.min(), FIG1_TARGET_DATES.max())
    ).sortby("time")

    got_dates = pd.DatetimeIndex(pd.to_datetime(daily_tmax.time.values).normalize())
    missing_dates = [
        d.strftime("%Y-%m-%d") for d in FIG1_TARGET_DATES if d.normalize() not in got_dates
    ]
    n_days = int(daily_tmax.sizes.get("time", 0))
    if missing_dates or n_days != 11:
        raise RuntimeError(
            "Fig.1(d) requires all 11 BJT daily Tmax fields for 2023-06-14–24; "
            f"n_days={n_days}, missing_dates={missing_dates}"
        )

    mean_tmax = daily_tmax.mean(dim="time", skipna=True).rename("mean_daily_tmax")
    mean_tmax.attrs.update({
        "long_name": "Mean of BJT daily maximum 2-m temperature",
        "units": "degC",
        "period_start": "2023-06-14",
        "period_end": "2023-06-24",
        "timezone": "BJT (UTC+8)",
        "daily_statistic": "daily maximum",
        "temporal_statistic": "mean of daily maxima",
        "NCHN_BOX": "114–119E, 35–44N",
    })
    lon_min, lon_max, lat_min, lat_max = FIG1_MAP_EXTENT
    map_view = mean_tmax.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    if map_view.sizes.get("lon", 0) == 0 or map_view.sizes.get("lat", 0) == 0:
        raise RuntimeError(f"No ERA5 map grid points fall inside FIG1_MAP_EXTENT={FIG1_MAP_EXTENT}")
    qc = {
        "map_start_date": "2023-06-14",
        "map_end_date": "2023-06-24",
        "map_n_days": n_days,
        "map_units": "degC",
        "map_min": float(map_view.min(skipna=True)),
        "map_max": float(map_view.max(skipna=True)),
        "map_mean": float(map_view.mean(skipna=True)),
        "NCHN_box": FIG1_NCHN_BOX,
        "map_extent": FIG1_MAP_EXTENT,
        "daily_statistic": "maximum",
        "temporal_statistic": "mean of daily maxima",
    }
    return mean_tmax, qc


def compute_kde_pdf_and_cdf(values, event_value):
    if gaussian_kde is None:
        raise RuntimeError("scipy.stats.gaussian_kde is required for Fig.1 anomaly PDF curves.")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.nanstd(values) == 0:
        raise ValueError("At least three non-constant background values are required for KDE.")
    kde = gaussian_kde(values)
    xmin = min(values.min(), event_value)
    xmax = max(values.max(), event_value)
    pad = max(0.5, 0.12 * (xmax - xmin))
    x_grid = np.linspace(xmin - pad, xmax + pad, 1000)
    density = kde(x_grid)
    dx = x_grid[1] - x_grid[0]
    cdf = np.cumsum(density) * dx
    cdf = cdf / cdf[-1]
    event_density = float(kde([event_value])[0])
    event_cdf = float(np.interp(event_value, x_grid, cdf))
    return x_grid, density, cdf, event_density, event_cdf


def compute_fig1_historical_pdf_tables(tmax_hist_daily):
    recs = []
    time_index = pd.DatetimeIndex(pd.to_datetime(tmax_hist_daily.time.values))
    years = sorted(time_index.year.unique().tolist())
    background_years = set(FIG1_HIST_BACKGROUND_YEARS)
    for year in years:
        for window, (start_mmdd, end_mmdd) in FIG1_WINDOWS.items():
            start = pd.Timestamp(f"{year}-{start_mmdd}")
            end = pd.Timestamp(f"{year}-{end_mmdd}")
            expected_days = len(pd.date_range(start, end, freq="D"))
            sub = tmax_hist_daily.sel(time=slice(start, end))
            n_days = int(sub.sizes.get("time", 0))
            if n_days == expected_days:
                recs.append({
                    "window": window,
                    "year": int(year),
                    "source_group": "1979_1992_root" if year <= 1992 else "1993_2023_glob",
                    "start_mmdd": start_mmdd,
                    "end_mmdd": end_mmdd,
                    "stage_mean_tmax": float(sub.mean()),
                    "is_background_1979_2022": bool(year in background_years),
                    "is_2023": bool(year == FIG1_EVENT_YEAR),
                    "n_days": n_days,
                })
            else:
                print(f"[Fig1 QC] Skip {year} {window}: n_days={n_days}, expected={expected_days}")

    values_df = pd.DataFrame(recs)
    required_years = set(range(min(FIG1_HIST_BACKGROUND_YEARS), FIG1_EVENT_YEAR + 1))
    available_years = set(values_df["year"].unique()) if not values_df.empty else set()
    missing_years = sorted(required_years - available_years)
    if missing_years:
        print(f"[Fig1 QC] Missing ERA5 Tmax years: {missing_years}")
        raise RuntimeError(f"Missing required 1979–2023 years for Fig.1 anomaly PDF: {missing_years}")

    summary_recs = []
    for window in ["Total", "Stage-I", "Stage-II"]:
        wdf = values_df[values_df["window"] == window].copy()
        bg_mask = wdf["year"].isin(background_years)
        bg_abs = wdf.loc[bg_mask, "stage_mean_tmax"].dropna().values
        clim_mean = float(np.mean(bg_abs))
        wdf["climatology_mean_1979_2022"] = clim_mean
        wdf["stage_mean_tmax_anomaly"] = wdf["stage_mean_tmax"] - clim_mean
        values_df.loc[wdf.index, "climatology_mean_1979_2022"] = clim_mean
        values_df.loc[wdf.index, "stage_mean_tmax_anomaly"] = wdf["stage_mean_tmax_anomaly"]

        event_row = wdf[wdf["year"] == FIG1_EVENT_YEAR].iloc[0]
        event_abs = float(event_row["stage_mean_tmax"])
        event_anom = float(event_row["stage_mean_tmax_anomaly"])
        bg_anom = wdf.loc[bg_mask, "stage_mean_tmax_anomaly"].dropna().values
        all_anom = wdf[wdf["year"].isin(required_years)]["stage_mean_tmax_anomaly"].dropna().values
        _, _, _, _, kde_cdf = compute_kde_pdf_and_cdf(bg_anom, event_anom)
        empirical_percentile = 100.0 * np.mean(bg_anom <= event_anom)
        rank_descending = 1 + int(np.sum(all_anom > event_anom))
        summary_recs.append({
            "window": window,
            "background_year_start": int(min(FIG1_HIST_BACKGROUND_YEARS)),
            "background_year_end": int(max(FIG1_HIST_BACKGROUND_YEARS)),
            "n_background_years": int(len(bg_anom)),
            "event_year": int(FIG1_EVENT_YEAR),
            "event_stage_mean_tmax": event_abs,
            "climatology_mean_1979_2022": clim_mean,
            "event_anomaly": event_anom,
            "kde_cdf_percentile_2023": float(kde_cdf * 100.0),
            "empirical_percentile_2023": empirical_percentile,
            "rank_2023_descending": rank_descending,
            "rank_denominator": int(len(all_anom)),
            "background_anomaly_mean": float(np.mean(bg_anom)),
            "background_anomaly_std": float(np.std(bg_anom, ddof=1)),
            "background_anomaly_p95": float(np.percentile(bg_anom, 95)),
            "missing_years": ";".join(map(str, missing_years)),
        })
    return values_df, pd.DataFrame(summary_recs)


def _fig1_map_levels(map_field):
    """Create regular 2 °C levels that include the untruncated plotted range."""
    lon_min, lon_max, lat_min, lat_max = FIG1_MAP_EXTENT
    shown = map_field.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    data_min = float(shown.min(skipna=True))
    data_max = float(shown.max(skipna=True))
    level_min = 2.0 * np.floor(data_min / 2.0)
    level_max = 2.0 * np.ceil(data_max / 2.0)
    if level_max <= level_min:
        level_max = level_min + 2.0
    levels = np.arange(level_min, level_max + 0.1, 2.0)
    print(f"[Paper Fig1 map] color levels (°C): {levels.tolist()}")
    return levels


def _read_png_size(path):
    """Read PNG dimensions from its header without adding an imaging dependency."""
    try:
        with open(path, "rb") as f:
            header = f.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG file")
        return struct.unpack(">II", header[16:24])
    except Exception as exc:
        print(f"[Paper Fig1 QC] Warning: could not read PNG dimensions: {exc}")
        return None, None


def _set_fig1_panel_heading(ax, label, title, title_x=0.12):
    ax.text(
        0.0, 1.01, f"({label})", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=FONT_PANEL_LETTER, fontweight="bold",
    )
    ax.text(
        title_x, 1.01, title, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=FONT_PANEL_TITLE,
    )


def _write_fig1_caption(path):
    caption = (
        "Figure 1. Observed evolution, historical context, and spatial structure of the "
        "2023 North China heatwave. (a) NCHN regional-mean ERA5 daily maximum 2-m "
        "temperature (Tmax) and MSWEP daily precipitation for 14–24 June 2023, evaluated "
        "in Beijing time (BJT; UTC+8); the vertical line marks the end of Stage-I. "
        "(b) Kernel-density estimate of Total-period (14–24 June) stage-mean Tmax anomalies "
        "during the 1979–2022 background. (c) As in (b), but for Stage-I (14–17 June). "
        "For (b) and (c), anomalies are relative to the corresponding 1979–2022 same-window "
        "climatological mean, the red star denotes 2023, and ranks are evaluated over "
        "1979–2023 (n = 45). (d) ERA5 mean daily Tmax (°C) for 14–24 June 2023, computed as "
        "the mean of the 11 BJT daily Tmax fields; the dashed box denotes the NCHN analysis "
        "region (35–44°N, 114–119°E). Ocean areas in panel (d) are masked for visual clarity."
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")


def make_paper_fig1_event_background(
    event_daily_df,
    hist_values_df,
    hist_summary_df,
    map_field,
    map_qc,
    output_dir=PAPER_FIG1_JGR_DIR,
):
    if ccrs is None or cfeature is None:
        raise RuntimeError("cartopy is required to render the JGR Fig.1(d) map panel.")
    os.makedirs(output_dir, exist_ok=True)
    with plt.rc_context(FIG1_RC_PARAMS):
        fig = plt.figure(figsize=(FIG1_WIDTH_IN, FIG1_HEIGHT_IN), constrained_layout=False)
        # Preserve the finalized panel-(a)/legend geometry while allowing the
        # lower figure block to use independent horizontal manuscript margins.
        fig.subplots_adjust(left=0.07, right=0.94, top=0.965, bottom=0.09)
        outer_gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[1.18, 0.12, 1.42],
            hspace=0.24,
        )
        lower_slot = outer_gs[2].get_position(fig)
        lower_gs = fig.add_gridspec(
            2,
            2,
            left=0.11,
            right=0.965,
            bottom=lower_slot.y0,
            top=lower_slot.y1,
            height_ratios=[1.0, 1.0],
            width_ratios=[0.92, 1.78],
            hspace=0.72,
            wspace=0.24,
        )
        map_gs = lower_gs[:, 1].subgridspec(
            2,
            1,
            height_ratios=[1.0, 0.10],
            hspace=0.20,
        )
        cbar_gs = map_gs[1].subgridspec(
            3,
            3,
            width_ratios=[0.11, 0.78, 0.11],
            height_ratios=[0.275, 0.45, 0.275],
            wspace=0.0,
            hspace=0.0,
        )
        ax_event = fig.add_subplot(outer_gs[0])
        ax_legend = fig.add_subplot(outer_gs[1])
        ax_pdf_total = fig.add_subplot(lower_gs[0, 0])
        ax_pdf_stage1 = fig.add_subplot(lower_gs[1, 0], sharex=ax_pdf_total, sharey=ax_pdf_total)
        map_crs = ccrs.PlateCarree()
        ax_map = fig.add_subplot(map_gs[0], projection=map_crs)
        cax = fig.add_subplot(cbar_gs[1, 1])

        xdates = pd.to_datetime(event_daily_df["date"])
        ax_event.axvline(pd.Timestamp("2023-06-17 12:00"), color="0.25", lw=0.9, zorder=2)
        ax_event.plot(
            xdates,
            event_daily_df["era5_tmax_nchn"],
            color="#B22222",
            marker="o",
            lw=1.8,
            ms=3.5,
            label="ERA5 Tmax",
            zorder=4,
        )
        ax_event.set_ylabel("ERA5 Tmax (°C)", fontsize=FONT_AXIS_LABEL)
        ax_event.set_xlim(pd.Timestamp("2023-06-13 12:00"), pd.Timestamp("2023-06-24 12:00"))
        ax_event.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax_event.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax_event.tick_params(axis="both", labelsize=FONT_TICK, width=0.8, length=3.0)
        plt.setp(ax_event.get_xticklabels(), rotation=35, ha="right")
        ax_event.grid(ls=":", lw=0.45, alpha=0.30)
        _set_fig1_panel_heading(ax_event, "a", "Event evolution", title_x=0.055)

        ax_tp = ax_event.twinx()
        ax_tp.bar(
            xdates,
            event_daily_df["mswep_tp_nchn"],
            width=0.72,
            color="lightskyblue",
            alpha=0.42,
            label="MSWEP precipitation",
            zorder=1,
        )
        ax_tp.set_ylabel(
            "MSWEP precipitation (mm day$^{-1}$)",
            fontsize=FONT_AXIS_LABEL,
            labelpad=5,
        )
        ax_tp.tick_params(axis="y", labelsize=FONT_TICK, width=0.8, length=3.0)
        lines, labels = ax_event.get_legend_handles_labels()
        lines2, labels2 = ax_tp.get_legend_handles_labels()
        ax_legend.axis("off")
        ax_legend.legend(
            lines + lines2,
            labels + labels2,
            loc="center",
            ncol=2,
            fontsize=FONT_LEGEND,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.8,
            columnspacing=1.6,
        )

        order = [
            (ax_pdf_total, "Total", "(b) Total period"),
            (ax_pdf_stage1, "Stage-I", "(c) Stage-I"),
        ]
        pdf_cache = {}
        all_x = []
        max_density = 0.0
        for _, window, _ in order:
            bg = hist_values_df[
                (hist_values_df["window"] == window)
                & (hist_values_df["is_background_1979_2022"])
            ]
            vals = bg["stage_mean_tmax_anomaly"].dropna().values
            summ = hist_summary_df[hist_summary_df["window"] == window].iloc[0]
            event_anom = float(summ["event_anomaly"])
            x_grid, density, _, event_density, _ = compute_kde_pdf_and_cdf(vals, event_anom)
            pdf_cache[window] = (x_grid, density, event_anom, event_density, summ)
            all_x.extend(x_grid.tolist())
            max_density = max(max_density, float(np.max(density)), event_density)
        xlim = (float(np.min(all_x)), float(np.max(all_x)))
        ylim = (-0.035 * max_density, 1.12 * max_density)

        for ax, window, title in order:
            x_grid, density, event_anom, event_density, summ = pdf_cache[window]
            pdf_line, = ax.plot(x_grid, density, color="0.28", lw=1.45, label="1979–2022 PDF")
            ax.scatter(event_anom, event_density, color="#B22222", marker="*", s=58, zorder=5)
            panel_label, panel_title = title.split(") ", maxsplit=1)
            _set_fig1_panel_heading(ax, panel_label.lstrip("("), panel_title, title_x=0.21)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel("Tmax anomaly (°C)", fontsize=FONT_AXIS_LABEL)
            ax.tick_params(axis="both", labelsize=FONT_TICK, width=0.8, length=3.0)
            ax.grid(ls=":", lw=0.4, axis="y", alpha=0.28)
            ax.set_yticks([tick for tick in ax.get_yticks() if tick >= 0])
            rank_descending = int(summ["rank_2023_descending"])
            rank_total = int(summ["rank_denominator"])
            text_x = 0.96 if window == "Total" else 0.04
            text_ha = "right" if window == "Total" else "left"
            ax.text(
                text_x,
                0.93,
                f"2023 = {event_anom:+.1f}°C\nRank = {rank_descending}/{rank_total}",
                transform=ax.transAxes,
                ha=text_ha,
                va="top",
                fontsize=FONT_ANNOTATION,
                bbox=dict(facecolor="none", edgecolor="none", pad=0.0),
            )
            if window == "Stage-I":
                ax.legend(
                    handles=[pdf_line],
                    labels=["1979–2022\nPDF"],
                    loc="upper right",
                    bbox_to_anchor=(0.98, 0.97),
                    fontsize=FONT_LEGEND - 1.1,
                    frameon=False,
                    handlelength=1.0,
                    handletextpad=0.18,
                    borderpad=0.0,
                    borderaxespad=0.10,
                    labelspacing=0.05,
                )
        ax_pdf_total.set_ylabel("Probability density", fontsize=FONT_AXIS_LABEL)
        ax_pdf_stage1.set_ylabel("Probability density", fontsize=FONT_AXIS_LABEL)
        ax_pdf_stage1.tick_params(labelleft=True)

        lon_min, lon_max, lat_min, lat_max = FIG1_MAP_EXTENT
        shown_map = map_field.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        if shown_map.sizes.get("lon", 0) == 0 or shown_map.sizes.get("lat", 0) == 0:
            raise RuntimeError(f"No ERA5 map grid points fall inside FIG1_MAP_EXTENT={FIG1_MAP_EXTENT}")
        levels = _fig1_map_levels(map_field)
        mesh = ax_map.contourf(
            shown_map["lon"],
            shown_map["lat"],
            shown_map,
            levels=levels,
            cmap="YlOrRd",
            extend="neither",
            transform=map_crs,
            antialiased=True,
            zorder=1,
        )
        ax_map.set_extent(FIG1_MAP_EXTENT, crs=map_crs)
        # Display-only ocean masking: the cached and analyzed Tmax field remains unchanged.
        ax_map.add_feature(
            cfeature.OCEAN.with_scale("50m"),
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
        ax_map.coastlines(resolution="50m", color="0.25", linewidth=0.65, zorder=3)
        ax_map.add_feature(
            cfeature.BORDERS.with_scale("50m"), edgecolor="0.35", linewidth=0.45, zorder=3
        )
        box_lon_min, box_lon_max, box_lat_min, box_lat_max = FIG1_NCHN_BOX
        ax_map.add_patch(Rectangle(
            (box_lon_min, box_lat_min),
            box_lon_max - box_lon_min,
            box_lat_max - box_lat_min,
            fill=False,
            edgecolor=NCHN_BOX_COLOR,
            linewidth=1.25,
            linestyle="--",
            transform=map_crs,
            zorder=4,
        ))
        lon_ticks = np.arange(np.ceil(lon_min / 5) * 5, lon_max + 0.1, 5)
        lat_ticks = np.arange(np.ceil(lat_min / 5) * 5, lat_max + 0.1, 5)
        ax_map.set_xticks(lon_ticks, crs=map_crs)
        ax_map.set_yticks(lat_ticks, crs=map_crs)
        ax_map.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="°"))
        ax_map.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="°"))
        ax_map.tick_params(axis="both", labelsize=FONT_TICK, width=0.8, length=3.0)
        _set_fig1_panel_heading(ax_map, "d", "Mean Tmax", title_x=0.14)
        # Dedicated nested columns keep the 78%-width colorbar centered without
        # post-hoc axes-position patches.
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cbar.set_label("Mean daily Tmax (°C)", fontsize=FONT_COLORBAR)
        cbar.ax.tick_params(labelsize=FONT_TICK, width=0.8, length=2.8)

        out_png = os.path.join(output_dir, "fig1_event_background_jgr_final.png")
        out_pdf = os.path.join(output_dir, "fig1_event_background_jgr_final.pdf")
        fig.savefig(out_png, dpi=FIG1_DPI, facecolor="white")
        fig.savefig(out_pdf, facecolor="white")
        png_width, png_height = _read_png_size(out_png)
        plt.close(fig)

    figure_qc = {
        "figure_width_in": FIG1_WIDTH_IN,
        "figure_height_in": FIG1_HEIGHT_IN,
        "png_width_px": png_width,
        "png_height_px": png_height,
        "dpi": FIG1_DPI,
        "expected_png_width_px": int(FIG1_WIDTH_IN * FIG1_DPI),
        "bbox_tight_used": False,
        "suptitle_present": False,
        "panel_count": 4,
        "panel_labels": "a,b,c,d",
        "legend_uses_dedicated_row": True,
        "ocean_mask_display_only": True,
        "ocean_values_removed_from_cache": False,
        "NCHN_box_color": NCHN_BOX_COLOR,
        "colorbar_relative_width": FIG1_COLORBAR_RELATIVE_WIDTH,
        **map_qc,
        "map_levels_degC": levels.tolist(),
    }
    if png_width is not None and png_width != int(FIG1_WIDTH_IN * FIG1_DPI):
        raise RuntimeError(
            f"Unexpected JGR Fig.1 PNG width: {png_width}px; expected {int(FIG1_WIDTH_IN * FIG1_DPI)}px"
        )
    return out_png, out_pdf, figure_qc


def _process_era5_tmax_files_to_daily_nchn(paths, year):
    da, _ = _open_era5_tmax_files_for_year(paths, year)
    da = _to_celsius_if_needed_fig1(da)
    da = _subset_nchn(da)
    da = _to_bjt_daily(da, how="max")
    da = da.sel(time=slice(pd.Timestamp(f"{year}-06-14"), pd.Timestamp(f"{year}-06-24"))).sortby("time")
    da = _coslat_weighted_mean_fig1(da)
    return da


def run_overlap_source_consistency_check(output_dir=PAPER_FIG1_DIR):
    check_years = [1993, 2000, 2010, 2021]
    records = []
    for year in check_years:
        old_path = os.path.join(ERA5_TMAX_HIST_ROOT_1979_1992, f"air.2m.{year}06.nc")
        if not os.path.exists(old_path):
            print(f"[Fig1 QC] Overlap check warning: old-source file not found for {year}: {old_path}")
            continue
        new_files, _ = find_era5_t2m_files_for_year(year)
        if not new_files:
            print(f"[Fig1 QC] Overlap check warning: new-source file not found for {year}")
            continue
        try:
            old_daily = _process_era5_tmax_files_to_daily_nchn([old_path], year)
            new_daily = _process_era5_tmax_files_to_daily_nchn(new_files, year)
            old_s = pd.Series(old_daily.values, index=pd.to_datetime(old_daily.time.values).strftime("%Y-%m-%d"))
            new_s = pd.Series(new_daily.values, index=pd.to_datetime(new_daily.time.values).strftime("%Y-%m-%d"))
            common_dates = sorted(set(old_s.index).intersection(new_s.index))
            for date in common_dates:
                records.append({
                    "year": year,
                    "comparison_type": "daily",
                    "window": "daily",
                    "date": date,
                    "old_source_tmax": float(old_s.loc[date]),
                    "new_source_tmax": float(new_s.loc[date]),
                    "difference_old_minus_new": float(old_s.loc[date] - new_s.loc[date]),
                    "old_file": old_path,
                    "new_files": ";".join(new_files),
                })
            for window, (start_mmdd, end_mmdd) in FIG1_WINDOWS.items():
                dates = pd.date_range(f"{year}-{start_mmdd}", f"{year}-{end_mmdd}", freq="D").strftime("%Y-%m-%d")
                old_mean = float(old_s.reindex(dates).mean())
                new_mean = float(new_s.reindex(dates).mean())
                records.append({
                    "year": year,
                    "comparison_type": "stage_mean",
                    "window": window,
                    "date": "",
                    "old_source_tmax": old_mean,
                    "new_source_tmax": new_mean,
                    "difference_old_minus_new": old_mean - new_mean,
                    "old_file": old_path,
                    "new_files": ";".join(new_files),
                })
        except Exception as exc:
            print(f"[Fig1 QC] Overlap check warning: failed for {year}: {exc}")

    if not records:
        print("[Fig1 QC] Overlap check warning: no overlapping old-source files found; overlap_source_consistency_check.csv not written.")
        return None
    out_csv = os.path.join(output_dir, "overlap_source_consistency_check.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"[Fig1 QC] Overlap source consistency check CSV: {out_csv}")
    return out_csv


def run_paper_fig1_cell():
    os.makedirs(PAPER_FIG1_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG1_JGR_DIR, exist_ok=True)
    tmax_hist_daily, tmax_qc = load_era5_tmax_historical_daily_nchn()
    mswep_daily, mswep_qc = load_mswep_precip_daily_bjt_nchn()
    hist_values_df, hist_summary_df = compute_fig1_historical_pdf_tables(tmax_hist_daily)
    map_field, map_qc = load_era5_total_period_mean_tmax_map_2023()

    tmax_2023 = tmax_hist_daily.sel(time=slice(FIG1_TARGET_DATES.min(), FIG1_TARGET_DATES.max())).sortby("time")
    event_daily_df = pd.DataFrame({
        "date": pd.to_datetime(tmax_2023.time.values).strftime("%Y-%m-%d"),
        "era5_tmax_nchn": tmax_2023.values,
    })
    mswep_df = pd.DataFrame({
        "date": pd.to_datetime(mswep_daily.time.values).strftime("%Y-%m-%d"),
        "mswep_tp_nchn": mswep_daily.values,
    })
    event_daily_df = event_daily_df.merge(mswep_df, on="date", how="left")
    event_daily_df["stage"] = event_daily_df["date"].map(_stage_for_date_fig1)

    values_csv = os.path.join(PAPER_FIG1_DIR, "fig1_historical_tmax_pdf_values.csv")
    summary_csv = os.path.join(PAPER_FIG1_DIR, "fig1_historical_tmax_pdf_summary.csv")
    daily_csv = os.path.join(PAPER_FIG1_DIR, "fig1_event_evolution_daily_values.csv")
    qc_txt = os.path.join(PAPER_FIG1_DIR, "fig1_qc_summary.txt")
    map_nc = os.path.join(PAPER_FIG1_JGR_DIR, "fig1_total_period_mean_tmax_map_2023.nc")
    caption_md = os.path.join(PAPER_FIG1_JGR_DIR, "fig1_caption_jgr.md")
    jgr_qc_txt = os.path.join(PAPER_FIG1_JGR_DIR, "fig1_jgr_qc_summary.txt")
    hist_values_df.to_csv(values_csv, index=False)
    hist_summary_df.to_csv(summary_csv, index=False)
    event_daily_df.to_csv(daily_csv, index=False)
    map_ds = map_field.to_dataset(name="mean_daily_tmax")
    map_ds.attrs.update({
        "period_start": "2023-06-14",
        "period_end": "2023-06-24",
        "timezone": "BJT (UTC+8)",
        "daily_statistic": "daily maximum",
        "temporal_statistic": "mean of daily maxima",
        "units": "degC",
        "NCHN_BOX": "114–119E, 35–44N",
    })
    map_ds.to_netcdf(map_nc)
    _write_fig1_caption(caption_md)
    out_png, out_pdf, figure_qc = make_paper_fig1_event_background(
        event_daily_df,
        hist_values_df,
        hist_summary_df,
        map_field,
        map_qc,
        output_dir=PAPER_FIG1_JGR_DIR,
    )
    overlap_csv = run_overlap_source_consistency_check(PAPER_FIG1_DIR)

    years = tmax_qc["years"]
    missing_years = tmax_qc["missing_years"]
    complete_background = not missing_years and all(y in years for y in FIG1_HIST_BACKGROUND_YEARS)
    qc_lines = [
        "Figure 1 shows the observed evolution and historical context of the 2023 North China heatwave.",
        "The manuscript Fig.1 emphasizes Stage-I (14–17 June) and the Total period (14–24 June).",
        "Stage-II remains an internal legacy table row for downstream CSV compatibility and is not a public-facing Fig.1 panel.",
        f"ERA5 Tmax 1979–1992 root: {ERA5_TMAX_HIST_ROOT_1979_1992}",
        "1979–1992 historical Tmax file naming pattern: air.2m.YYYY06.nc",
        f"ERA5 Tmax 1993–2023 glob: {ERA5_TMAX_HIST_GLOB_1993_2023}",
        "ERA5 Tmax yearly source records:",
        *[f"  {r['year']} | {r['source_group']} | n_files={r['n_files']} | variable={r['data_variable_used']} | units={r['units']} | raw_mean={r['raw_mean_before_unit_conversion']:.2f} | mean_after_conversion={r['mean_after_unit_conversion']:.2f} | n_days={r['n_days']} | complete_11_days={r['complete_11_days']} | files={r['files']}" for r in tmax_qc["source_records"]],
        f"Available years: {years[0]}-{years[-1]} (n={len(years)})" if years else "Available years: none",
        f"Missing years: {missing_years}",
        f"1979–2022 background complete: {complete_background}",
        f"Background years: {min(FIG1_HIST_BACKGROUND_YEARS)}-{max(FIG1_HIST_BACKGROUND_YEARS)}",
        f"Event year: {FIG1_EVENT_YEAR}",
        "PDF variable: stage-mean Tmax anomaly relative to same-window 1979–2022 climatology, not absolute Tmax.",
        "Fig.1a precipitation source: MSWEP",
        f"MSWEP roots: {MSWEP_ROOTS}",
        f"MSWEP files used ({len(mswep_qc['files'])}):",
        *[f"  {f}" for f in mswep_qc["files"]],
        f"MSWEP original units: {mswep_qc['original_units']}",
        f"MSWEP unit conversion: {mswep_qc['unit_note']}",
        f"MSWEP BJT daily sample counts: {mswep_qc['sample_counts']}",
        f"NCHN_BOX (lon_min, lon_max, lat_min, lat_max): {FIG1_NCHN_BOX}",
        f"Manuscript windows: Stage-I={FIG1_WINDOWS['Stage-I']}; Total={FIG1_WINDOWS['Total']}",
        f"Output PNG: {out_png}",
        f"Output PDF: {out_pdf}",
        f"Map cache NetCDF: {map_nc}",
        f"JGR caption Markdown: {caption_md}",
        f"JGR figure QC TXT: {jgr_qc_txt}",
        f"Historical values CSV: {values_csv}",
        f"Historical summary CSV: {summary_csv}",
        f"Event daily values CSV: {daily_csv}",
        f"QC summary TXT: {qc_txt}",
        f"Overlap source consistency CSV: {overlap_csv if overlap_csv is not None else 'not written (no overlapping old-source files found)'}",
    ]
    with open(qc_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(qc_lines) + "\n")

    jgr_qc_lines = [
        "JGR manuscript Figure 1 QC",
        *[f"{key} = {value}" for key, value in figure_qc.items()],
        f"NCHN_BOX = {FIG1_NCHN_BOX}",
        f"map_extent = {FIG1_MAP_EXTENT}",
        "map_extent_source = local Fig.1 North-China context extent; no shared map helper was found in this repository",
        "daily_statistic = maximum",
        "temporal_statistic = mean of 11 daily maxima",
        "public_facing_periods = Stage-I, Total period",
        "legacy_Stage-II_table_row_retained = True (downstream CSV compatibility)",
        f"output_png = {out_png}",
        f"output_pdf = {out_pdf}",
        f"map_cache_netcdf = {map_nc}",
        f"caption_markdown = {caption_md}",
    ]
    with open(jgr_qc_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(jgr_qc_lines) + "\n")

    print("[Paper Fig1] outputs:")
    print(f"  PNG: {out_png}")
    print(f"  PDF: {out_pdf}")
    print(f"  Map cache NetCDF: {map_nc}")
    print(f"  Caption Markdown: {caption_md}")
    print(f"  JGR figure QC TXT: {jgr_qc_txt}")
    print(f"  Historical values CSV: {values_csv}")
    print(f"  Historical summary CSV: {summary_csv}")
    print(f"  Event daily values CSV: {daily_csv}")
    print(f"  QC summary TXT: {qc_txt}")
    print(f"  Overlap source consistency CSV: {overlap_csv if overlap_csv is not None else 'not written (no overlapping old-source files found)'}")
    n_early = sum(1 for r in tmax_qc["source_records"] if r["source_group"] == "1979_1992_root")
    n_late = sum(1 for r in tmax_qc["source_records"] if r["source_group"] == "1993_2023_glob")
    print("[Paper Fig1 QC] 1979–1992 historical Tmax file naming pattern: air.2m.YYYY06.nc")
    print(f"[Paper Fig1 QC] 1979–1992 source years loaded from {ERA5_TMAX_HIST_ROOT_1979_1992}: {n_early}/14")
    print(f"[Paper Fig1 QC] 1993–2023 source years loaded from {ERA5_TMAX_HIST_GLOB_1993_2023}: {n_late}/31")
    print(f"[Paper Fig1 QC] Found 1979–2023 ERA5 Tmax: {complete_background and FIG1_EVENT_YEAR in years}")
    print(f"[Paper Fig1 QC] KDE/PDF background years: {min(FIG1_HIST_BACKGROUND_YEARS)}-{max(FIG1_HIST_BACKGROUND_YEARS)}")
    print(f"[Paper Fig1 QC] 2023 is plotted as a separate event marker: True")
    print(f"[Paper Fig1 QC] Fig.1a precipitation source is MSWEP; conversion={mswep_qc['unit_note']}")
    print("[Paper Fig1 QC] Histogram removed; panels show anomaly KDE/PDF curves and 2023 markers.")
    for _, row in hist_summary_df[hist_summary_df["window"].isin(["Total", "Stage-I"])].iterrows():
        print(f"[Paper Fig1][{row['window']}] 2023 anomaly={row['event_anomaly']:+.2f} °C; "
              f"empirical percentile={row['empirical_percentile_2023']:.1f}; "
              f"rank={int(row['rank_2023_descending'])}/{int(row['rank_denominator'])}")
    print(f"[Paper Fig1 map QC] period={map_qc['map_start_date']}–{map_qc['map_end_date']}; "
          f"n_days={map_qc['map_n_days']}; units={map_qc['map_units']}; "
          f"range={map_qc['map_min']:.2f}–{map_qc['map_max']:.2f} °C")
    print(f"[Paper Fig1 figure QC] width={figure_qc['figure_width_in']:.2f} in; "
          f"PNG={figure_qc['png_width_px']}×{figure_qc['png_height_px']} px; "
          "bbox_tight_used=False; suptitle_present=False; panels=a,b,c,d")
    return {
        "png": out_png,
        "pdf": out_pdf,
        "historical_values_csv": values_csv,
        "historical_summary_csv": summary_csv,
        "event_daily_csv": daily_csv,
        "qc_txt": qc_txt,
        "map_cache_netcdf": map_nc,
        "caption_markdown": caption_md,
        "jgr_figure_qc_txt": jgr_qc_txt,
        "overlap_source_consistency_csv": overlap_csv,
    }


run_paper_fig1_cell()
