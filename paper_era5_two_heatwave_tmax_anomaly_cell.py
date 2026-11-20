# Independent notebook cell: two 2023 heatwave Tmax anomaly maps.
# Daily Tmax and source discovery follow paper_fig1_event_background_cell.py.
# No import of that cell: importing it would run the full Fig.1 workflow.
import glob
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

ERA5_TMAX_HIST_GLOB_1993_2023 = "/data1/huangy/fig6/NC/ai_test/pangu/era5_T/era5_2m_temperature_6_*.nc"
ERA5_TMAX_HIST_ROOT_1979_1992 = "/data6/wangsg/ERA5_meso2/monolevel/sfc_1hour/t2m"
BACKGROUND_YEARS = tuple(range(1979, 2023))
EVENT_YEAR = 2023
WINDOWS = {"first": ("06-14", "06-17"), "second": ("06-22", "06-24")}
MAP_EXTENT = (108.0, 125.0, 30.0, 48.0)
NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
OUT_DIR = Path("/data1/huangy/fig6/NC/v12")
PREFIX = "era5_2023_two_heatwave_tmax_anomaly_1979_2022"
FIG_SIZE = (6.5, 4.2)
DPI = 600
RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
      "pdf.fonttype": 42, "ps.fonttype": 42}

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

def _to_celsius_if_needed_fig1(da):
    return da - 273.15 if float(da.mean()) > 150 else da

def _open_era5_tmax_file(path):
    ds = xr.open_dataset(path)
    da = _guess_var(ds, ["t2m", "2m_temperature", "air", "air_temperature", "T2M"])
    da.attrs["_fig1_variable_used"] = da.name
    da = _normalize_time_dim(da)
    da = _standardize_lat_lon(da)
    return da

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

def load_window_maps(year):
    """Load only the map domain; require all hourly samples for each BJT day."""
    paths, source_group = find_era5_t2m_files_for_year(year)
    if not paths:
        raise FileNotFoundError(f"Missing ERA5 year {year}; cannot shorten the 44-year baseline")
    pieces, datasets = [], []
    try:
        for path in paths:
            ds = xr.open_dataset(path)
            datasets.append(ds)
            da = _standardize_lat_lon(_normalize_time_dim(
                _guess_var(ds, ["t2m", "2m_temperature", "air", "air_temperature", "T2M"])))
            da = da.sel(time=slice(f"{year}-06-13", f"{year}-06-24T23:59:59"),
                        lon=slice(MAP_EXTENT[0], MAP_EXTENT[1]),
                        lat=slice(MAP_EXTENT[2], MAP_EXTENT[3]))
            if da.sizes.get("time", 0):
                pieces.append(da.load())
        if not pieces:
            raise ValueError(f"No usable map data for {year}")
        raw = xr.concat(pieces, dim="time", join="exact").sortby("time")
        if raw.get_index("time").duplicated().any():
            raise ValueError(f"Duplicate timestamps in {year}; resolve overlapping inputs explicitly")
        raw = _to_celsius_if_needed_fig1(raw)
        maps, counts = {}, {}
        daily = _to_bjt_daily(raw, how="max")
        for key, (start, end) in WINDOWS.items():
            dates = pd.date_range(f"{year}-{start}", f"{year}-{end}")
            counts[key] = _count_samples_per_bjt_day(raw.time.values, dates)
            if any(count != 24 for count in counts[key].values()):
                raise ValueError(f"{year} {key}: expected 24 hourly samples per BJT day: {counts[key]}")
            window_raw = raw.assign_coords(time=pd.to_datetime(raw.time.values) + pd.Timedelta(hours=8))
            window_raw = window_raw.sel(time=slice(dates[0], dates[-1] + pd.Timedelta(hours=23)))
            expected_hours = pd.date_range(dates[0], dates[-1] + pd.Timedelta(hours=23), freq="h")
            if not pd.DatetimeIndex(window_raw.time.values).equals(expected_hours):
                raise ValueError(f"{year} {key}: timestamps must cover the complete hourly BJT window")
            if not np.isfinite(window_raw.values).all():
                raise ValueError(f"Nonfinite hourly grid values in {year} {key}")
            selected = daily.sel(time=dates)
            field = selected.mean("time", skipna=False).transpose("lat", "lon")
            if not field.size or not np.isfinite(field.values).all():
                raise ValueError(f"Invalid window map in {year} {key}")
            maps[key] = field
        return maps, {"year": year, "files": paths, "source_group": source_group,
                      "BJT_daily_sample_counts": counts}
    finally:
        for ds in datasets:
            ds.close()


def compute_anomaly_maps():
    event, event_qc = load_window_maps(EVENT_YEAR)
    historical = {key: [] for key in WINDOWS}
    records = []
    for year in BACKGROUND_YEARS:
        fields, qc = load_window_maps(year)
        records.append(qc)
        for key in WINDOWS:
            # Different source grids must be resolved explicitly, never silently
            # intersected or interpolated onto a smaller spatial domain.
            _, field = xr.align(event[key], fields[key], join="exact")
            historical[key].append(field.expand_dims(year=[year]))
    result = xr.Dataset()
    for key in WINDOWS:
        annual = xr.concat(historical[key], dim="year", join="exact")
        if annual.sizes["year"] != 44:
            raise ValueError("Expected exactly 44 climatology years")
        climatology = annual.mean("year", skipna=False)
        for suffix, field in {"event": event[key], "climatology": climatology,
                              "anomaly": event[key] - climatology}.items():
            result[f"{key}_{suffix}"] = field
            result[f"{key}_{suffix}"].attrs["units"] = "degC"
    result.attrs.update(baseline="1979–2022 (44 equally weighted annual window means)",
                        timezone="BJT (UTC+8)", daily_statistic="grid-point daily maximum",
                        windows=json.dumps(WINDOWS), map_extent=str(MAP_EXTENT), NCHN_box=str(NCHN_BOX))
    return result, {"event": event_qc, "background": records}


def render_anomaly_maps(data, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maximum = max(float(np.abs(data[f"{key}_anomaly"]).max()) for key in WINDOWS)
    limit = max(1.0, float(np.ceil(maximum)))
    levels = np.linspace(-limit, limit, 21)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    crs = ccrs.PlateCarree()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=FIG_SIZE)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.045], wspace=0.18, hspace=0.20)
        for index, (key, title) in enumerate(zip(WINDOWS, ["14–17 June 2023", "22–24 June 2023"])):
            ax = fig.add_subplot(gs[0, index], projection=crs)
            field = data[f"{key}_anomaly"]
            mesh = ax.contourf(field.lon, field.lat, field, levels=levels, norm=norm,
                               cmap="RdBu_r", transform=crs, extend="neither")
            ax.set_extent(MAP_EXTENT, crs=crs)
            # Match Fig.1's display-only white ocean mask.
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", edgecolor="none", zorder=2)
            ax.coastlines(resolution="50m", color="0.25", linewidth=0.65, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="0.35", linewidth=0.45, zorder=3)
            x0, x1, y0, y1 = NCHN_BOX
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#2F4B7C",
                                   linewidth=1.25, linestyle="--", transform=crs, zorder=4))
            ax.set_xticks([110, 115, 120, 125], crs=crs)
            ax.set_yticks([30, 35, 40, 45], crs=crs)
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(labelsize=8)
            ax.set_title(f"({chr(97+index)}) {title}", loc="left", fontsize=9.6)
        cbar = fig.colorbar(mesh, cax=fig.add_subplot(gs[1, :]), orientation="horizontal")
        cbar.set_label(r"$T_{\max}$ anomaly (°C)", fontsize=9.5)
        cbar.ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.18)
        png, pdf = out_dir / f"{PREFIX}.png", out_dir / f"{PREFIX}.pdf"
        fig.savefig(png, dpi=DPI, facecolor="white")
        fig.savefig(pdf, facecolor="white")
        plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "color_limits_degC": [-limit, limit]}


def run_two_heatwave_anomaly_cell():
    data, sources = compute_anomaly_maps()
    outputs = render_anomaly_maps(data)
    cache = OUT_DIR / f"{PREFIX}.nc"
    data.to_netcdf(cache)
    caption = ("ERA5 daily maximum 2-m temperature anomalies during the 2023 North China "
               "heatwaves: (a) 14–17 June and (b) 22–24 June. At each grid point, the 2023 "
               "window mean of BJT (UTC+8) daily maxima is referenced to the same-window "
               "1979–2022 climatology, obtained by equally averaging 44 annual window means. "
               "Both panels share a zero-centered symmetric color scale; dashed boxes mark "
               "114–119°E, 35–44°N. Ocean masking is display-only.")
    (OUT_DIR / f"{PREFIX}_caption.md").write_text(caption + "\n", encoding="utf-8")
    qc = {"baseline_years": list(BACKGROUND_YEARS), "n_years": 44, "event_year": EVENT_YEAR,
          "windows": WINDOWS, "sources": sources, "map_extent": MAP_EXTENT, "NCHN_box": NCHN_BOX,
          "timezone": "BJT (UTC+8)", "daily_samples_required": 24, "grid_alignment": "exact; no regridding",
          "formula": "2023 window mean of daily maxima minus mean of 44 historical annual window means",
          "figure_inches": FIG_SIZE, "dpi": DPI, "outputs": outputs, "cache": str(cache),
          "visual_QC": "manual inspection required: spacing, labels, clipping and geographic overlays"}
    (OUT_DIR / f"{PREFIX}_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(outputs)
    return outputs


# Notebook-cell entry point; run only where the configured ERA5 files exist.
run_two_heatwave_anomaly_cell()
