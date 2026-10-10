import os
import glob
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================
# Cell X: ERA5 + MSWEP daily precipitation–moisture evolution maps
# ==============================

# 1) Paths / constants (absolute paths only)
ERA5_TP_FILE = "/data1/huangy/fig6/NC/MSE/ERA5/tp/era5_tp_2023_06.nc"
# If these paths are not valid in your environment, fill them with actual ERA5 q/u/v hourly files.
ERA5_Q_FILE = "/data1/huangy/fig6/NC/MSE/ERA5/850 uv+q/era5_850_uvq_2023_06.nc"
ERA5_U_FILE = "/data1/huangy/fig6/NC/MSE/ERA5/850 uv+q/era5_850_uvq_2023_06.nc"
ERA5_V_FILE = "/data1/huangy/fig6/NC/MSE/ERA5/850 uv+q/era5_850_uvq_2023_06.nc"

MSWEP_ROOTS = {
    "past": "/data1/huangy/fig6/NC/MSE/MSWEP/Past_3hourly_2003_2020",
    "nrt": "/data1/huangy/fig6/NC/MSE/MSWEP/NRT_3hourly_2021_2022",
    "2023": "/data1/huangy/fig6/NC/MSE/MSWEP/3hourly-2023",
}

OUT_DIR = "/data1/huangy/fig6/NC/v7/factor_verification/daily_evolution_circulation_precip"
os.makedirs(OUT_DIR, exist_ok=True)

MAP_EXTENT = (60.0, 150.0, 0.0, 60.0)
NCHN_BOX = (114.0, 119.0, 35.0, 44.0)
TARGET_DATES = pd.to_datetime(["2023-06-17", "2023-06-18", "2023-06-19", "2023-06-20"])
VECTOR_STRIDE = 20
VECTOR_COLOR = "#00C853"
QFLUX_VECTOR_STRIDE = 24
QFLUX_MIN = 0.02
CONTOUR_REGION = (100.0, 130.0, 25.0, 50.0)
CONTOUR_LEVELS = [10, 20, 40, 80]


def parse_mswep_filename(path):
    basename = os.path.basename(path)
    stem = basename.replace(".nc", "")
    yyyyddd, hh = stem.split(".")
    year = int(yyyyddd[:4])
    doy = int(yyyyddd[4:])
    hour = int(hh)
    utc_time = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1) + pd.Timedelta(hours=hour)
    return utc_time


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
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    da = da.rename(rename)
    lon = da["lon"]
    if float(lon.max().values) > 180.0:
        da = da.assign_coords(lon=((lon + 180) % 360) - 180).sortby("lon")
    if float(da["lat"][0].values) > float(da["lat"][-1].values):
        da = da.sortby("lat")
    return da


def _sel_region(da, extent=MAP_EXTENT):
    x0, x1, y0, y1 = extent
    return da.sel(lon=slice(x0, x1), lat=slice(y0, y1))


def _area_mean(da, box=NCHN_BOX):
    x0, x1, y0, y1 = box
    sub = da.sel(lon=slice(x0, x1), lat=slice(y0, y1))
    w = np.cos(np.deg2rad(sub.lat))
    return sub.weighted(w).mean(("lat", "lon"))


def _plot_box(ax, box=NCHN_BOX):
    x0, x1, y0, y1 = box
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    ax.plot(xs, ys, transform=ccrs.PlateCarree(), color="k", lw=1.0)


def _setup_map(ax):
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", lw=0.8)
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    gl = ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.4, alpha=0.6)
    _ = gl


def _save_fig(fig, stem):
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _to_bjt_daily(da, how="sum"):
    da = da.assign_coords(time=da.time + np.timedelta64(8, "h"))
    return getattr(da.resample(time="1D"), how)()


def _select_850(da):
    for lev in ["isobaricInhPa", "isobaricInPa", "pressure_level", "level", "plev"]:
        if lev in da.dims or lev in da.coords:
            coord = da[lev]
            coord_vals = np.asarray(coord.values)
            cmax = float(np.nanmax(coord_vals))
            target = 850.0 if cmax < 2000 else 85000.0

            if lev in da.dims:
                sel = da.sel({lev: target}, method="nearest")
                return sel.squeeze(drop=True)

            if coord.ndim == 0:
                cval = float(coord.values)
                tol = 5.0 if target == 850.0 else 500.0
                if abs(cval - target) <= tol:
                    try:
                        return da.drop_vars(lev)
                    except Exception:
                        return da
                print(f"[Warning] scalar {lev}={cval} not close to target {target}; keep original field.")
                return da

            print(f"[Warning] coordinate {lev} exists but is not a dimension/scalar (shape={coord.shape}); keep original field.")
            return da

    print("No level coordinate found; assume field is already 850 hPa.")
    return da


def compute_divergence_lonlat(qu, qv):
    R = 6371000.0
    lat = np.deg2rad(qu.lat)
    coslat = np.cos(lat)
    dqu_dlambda = qu.differentiate("lon") * 180.0 / np.pi
    qv_cos = qv * coslat
    dqvcos_dphi = qv_cos.differentiate("lat") * 180.0 / np.pi
    div = (dqu_dlambda + dqvcos_dphi) / (R * coslat)
    return div


def load_era5_tp_daily_bjt():
    print("ERA5 TP file:", ERA5_TP_FILE)
    if not os.path.exists(ERA5_TP_FILE):
        raise FileNotFoundError(f"ERA5 TP file not found: {ERA5_TP_FILE}")
    ds = xr.open_dataset(ERA5_TP_FILE)
    tp = _guess_var(ds, ["tp", "total_precipitation", "precip", "precipitation"])
    tp = _normalize_time_dim(tp)
    tp = _standardize_lat_lon(tp)
    units = str(tp.attrs.get("units", ""))
    if units.lower() in ["m", "meter", "metre", "meters", "metres"]:
        tp = tp * 1000.0
        units = "mm"
    tp_counts = _count_samples_per_bjt_day(tp.time.values, TARGET_DATES)
    tp_daily = _to_bjt_daily(tp, "sum")
    tp_daily.attrs["units"] = "mm day^-1"
    return tp_daily, units, tp_counts


def load_era5_qflux_daily_bjt():
    for f in [ERA5_Q_FILE, ERA5_U_FILE, ERA5_V_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"ERA5 q/u/v file not found: {f}")
    uvq_file = ERA5_Q_FILE
    if not os.path.exists(uvq_file):
        raise FileNotFoundError(
            f"ERA5 850hPa q/u/v file missing: {uvq_file}. "
            "Please set ERA5_Q_FILE/ERA5_U_FILE/ERA5_V_FILE to this same valid hourly file."
        )
    print("ERA5 q/u/v unified file exists:", uvq_file)
    ds = xr.open_dataset(uvq_file)
    print("ERA5 uvq data_vars:", list(ds.data_vars))
    print("ERA5 uvq dims:", dict(ds.dims))
    print("ERA5 uvq coords:", list(ds.coords))

    q_name = next((v for v in ["q", "specific_humidity"] if v in ds.data_vars), None)
    u_name = next((v for v in ["u", "u_component_of_wind"] if v in ds.data_vars), None)
    v_name = next((v for v in ["v", "v_component_of_wind"] if v in ds.data_vars), None)
    print("Matched q variable:", q_name)
    print("Matched u variable:", u_name)
    print("Matched v variable:", v_name)

    q = _normalize_time_dim(_select_850(_guess_var(ds, ["q", "specific_humidity"])))
    u = _normalize_time_dim(_select_850(_guess_var(ds, ["u", "u_component_of_wind"])))
    v = _normalize_time_dim(_select_850(_guess_var(ds, ["v", "v_component_of_wind"])))
    q, u, v = map(_standardize_lat_lon, [q, u, v])
    qu = q * u
    qv = q * v
    qflux_counts = _count_samples_per_bjt_day(qu.time.values, TARGET_DATES)
    return _to_bjt_daily(qu, "mean"), _to_bjt_daily(qv, "mean"), qflux_counts


def _mswep_dir_for_year(year):
    if year <= 2020:
        return MSWEP_ROOTS["past"]
    if year <= 2022:
        return MSWEP_ROOTS["nrt"]
    return MSWEP_ROOTS["2023"]


def load_mswep_daily_bjt(start, end):
    year = start.year
    root = _mswep_dir_for_year(year)
    if year == 2023 and not os.path.isdir(MSWEP_ROOTS["2023"]):
        raise FileNotFoundError(f"MSWEP_2023_DIR not found: {MSWEP_ROOTS['2023']}")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"MSWEP directory not found for year {year}: {root}")

    all_files = sorted(glob.glob(os.path.join(root, "*.nc")))
    utc0 = start - pd.Timedelta(hours=8)
    utc1 = end + pd.Timedelta(days=1) - pd.Timedelta(hours=8)
    candidate_files = []
    for f in all_files:
        try:
            t = parse_mswep_filename(f)
            if utc0 <= t <= utc1:
                candidate_files.append(f)
        except Exception:
            pass
    print(f"MSWEP candidate files selected by filename: {len(candidate_files)} from {root}")
    if len(candidate_files) == 0:
        raise FileNotFoundError(
            "No MSWEP files found after UTC window filtering. "
            f"root={root}, utc_start={utc0}, utc_end={utc1}, bjt_start={start}, bjt_end={end}"
        )

    pr_list = []
    for fp in candidate_files:
        ds_i = xr.open_dataset(fp)
        pr_i = _guess_var(ds_i, ["precipitation", "precip", "pr", "rain", "tp"])
        pr_i = _standardize_lat_lon(pr_i)

        if "time" in pr_i.dims or "valid_time" in pr_i.dims or "time" in pr_i.coords or "valid_time" in pr_i.coords:
            pr_i = _normalize_time_dim(pr_i)
        else:
            t = parse_mswep_filename(fp)
            pr_i = pr_i.expand_dims(time=[np.datetime64(t.to_datetime64())])

        pr_list.append(pr_i)

    if len(pr_list) == 0:
        raise FileNotFoundError(f"No valid MSWEP precipitation data loaded from {root}")

    pr = xr.concat(pr_list, dim="time").sortby("time")
    mswep_counts = _count_samples_per_bjt_day(pr.time.values, TARGET_DATES)
    pr_daily = _to_bjt_daily(pr, "sum")
    pr_daily.attrs["units"] = "mm day^-1"
    return pr_daily, "mm", mswep_counts


def _count_samples_per_bjt_day(time_index, target_dates):
    shifted = pd.DatetimeIndex(pd.to_datetime(time_index)) + pd.Timedelta(hours=8)
    s = pd.Series(1, index=shifted).resample("1D").sum()
    out = {}
    for d in pd.to_datetime(target_dates):
        out[d.strftime("%Y-%m-%d")] = int(s.get(d, 0))
    return out



def _check_target_dates(da, name):
    times = pd.to_datetime(da.time.values)
    print(f"{name} daily times:", times)
    have = set(times.normalize())
    miss = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(TARGET_DATES) if d.normalize() not in have]
    if miss:
        raise RuntimeError(f"{name} missing target dates: {miss}")


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
    if qu_plot.size == 0:
        print(f"[Warning] No qflux vectors after filter (|qflux| >= {QFLUX_MIN:.3f}); skip quiver for this panel.")
        return None
    common_kwargs = dict(
        alpha=1.0,
        scale=1.0,
        pivot="mid",
        headwidth=6.2,
        headlength=7.2,
        headaxislength=6.2,
        transform=ccrs.PlateCarree(),
    )
    ax.quiver(x_plot, y_plot, qu_plot, qv_plot, color="white", width=0.0032, zorder=20, **common_kwargs)
    q = ax.quiver(x_plot, y_plot, qu_plot, qv_plot, color=VECTOR_COLOR, width=0.0023, zorder=21, **common_kwargs)
    return q

def run_daily_evolution_precip_moisture():
    era5_tp, era5_units, era5_tp_counts = load_era5_tp_daily_bjt()
    qu_d, qv_d, era5_quv_counts = load_era5_qflux_daily_bjt()
    mswep_tp, mswep_units, mswep_counts = load_mswep_daily_bjt(TARGET_DATES.min(), TARGET_DATES.max())

    _check_target_dates(era5_tp, "era5_tp")
    _check_target_dates(mswep_tp, "mswep_tp")
    _check_target_dates(qu_d, "qu_d")
    _check_target_dates(qv_d, "qv_d")

    era5_tp = _sel_region(era5_tp.sel(time=TARGET_DATES)).transpose("time", "lat", "lon")
    mswep_tp = _sel_region(mswep_tp.sel(time=TARGET_DATES)).transpose("time", "lat", "lon")
    qu_d = _sel_region(qu_d.sel(time=TARGET_DATES)).transpose("time", "lat", "lon")
    qv_d = _sel_region(qv_d.sel(time=TARGET_DATES)).transpose("time", "lat", "lon")
    mfc = (-compute_divergence_lonlat(qu_d, qv_d) * 1e5).transpose("time", "lat", "lon")
    print("mfc dims:", mfc.dims, "shape:", mfc.shape,
          "min:", float(np.nanmin(mfc.values)), "max:", float(np.nanmax(mfc.values)))

    qc = []
    for d in TARGET_DATES:
        ds = d.strftime("%Y-%m-%d")
        if era5_tp_counts[ds] != 24:
            print(f"[Warning] ERA5 TP samples on {ds}: {era5_tp_counts[ds]} (expected ~24)")
        if era5_quv_counts[ds] != 24:
            print(f"[Warning] ERA5 q/u/v samples on {ds}: {era5_quv_counts[ds]} (expected ~24)")
        if mswep_counts[ds] != 8:
            print(f"[Warning] MSWEP samples on {ds}: {mswep_counts[ds]} (expected ~8)")
        era5_mean = float(_area_mean(era5_tp.sel(time=d)))
        mswep_mean = float(_area_mean(mswep_tp.sel(time=d)))
        qc.append({
            "date": d.strftime("%Y-%m-%d"),
            "era5_tp_nchn_mean": era5_mean,
            "mswep_tp_nchn_mean": mswep_mean,
            "era5_minus_mswep_nchn_mean": era5_mean - mswep_mean,
            "mfc_nchn_mean": float(_area_mean(mfc.sel(time=d))),
            "era5_tp_max": float(era5_tp.sel(time=d).max()),
            "mswep_tp_max": float(mswep_tp.sel(time=d).max()),
            "mfc_min": float(mfc.sel(time=d).min()),
            "mfc_max": float(mfc.sel(time=d).max()),
            "number_of_era5_hourly_tp_samples": era5_tp_counts[d.strftime("%Y-%m-%d")],
            "number_of_era5_hourly_quv_samples": era5_quv_counts[d.strftime("%Y-%m-%d")],
            "number_of_mswep_3hourly_samples": mswep_counts[d.strftime("%Y-%m-%d")],
            "era5_tp_units_after_conversion": "mm day^-1",
            "mswep_units_after_conversion": "mm day^-1",
            "bjt_start": f"{d.strftime('%Y-%m-%d')} 00:00",
            "bjt_end": (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00"),
            "utc_start": (d - pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
            "utc_end": (d + pd.Timedelta(hours=15)).strftime("%Y-%m-%d %H:%M"),
        })
    pd.DataFrame(qc).to_csv(os.path.join(OUT_DIR, "daily_evolution_precip_moisture_qc.csv"), index=False)

    def _panel_fill_quiver(fill, stem, cbar_label):
        fig, axs = plt.subplots(1, 4, figsize=(20, 4.2), subplot_kw={"projection": ccrs.PlateCarree()})
        combo = np.r_[era5_tp.values.flatten(), mswep_tp.values.flatten()]
        if np.all(np.isnan(combo)):
            raise ValueError("All precipitation values are NaN; cannot draw precipitation maps.")
        vmax = max(float(np.nanpercentile(combo, 98)), 1.0)
        levels = np.linspace(0, vmax, 13)
        q_for_key = None
        for i, d in enumerate(TARGET_DATES):
            ax = axs[i]; _setup_map(ax)
            cf = ax.contourf(fill.lon, fill.lat, fill.sel(time=d), levels=levels, cmap="Blues", extend="max", transform=ccrs.PlateCarree())
            q = _plot_qflux850_quiver(ax, qu_d.lon, qu_d.lat, qu_d.sel(time=d), qv_d.sel(time=d))
            if i == 0:
                q_for_key = q
            panel_mean = float(_area_mean(fill.sel(time=d)))
            ax.text(0.02, 0.03, f"NCHN mean = {panel_mean:.1f} mm/day", transform=ax.transAxes,
                    fontsize=8, ha="left", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            _plot_box(ax)
            ax.set_title(d.strftime("%Y-%m-%d BJT"))
        if q_for_key is not None:
            axs[0].quiverkey(q_for_key, 0.72, 0.08, 0.02, "0.02 kg kg$^{-1}$ m s$^{-1}$", labelpos="E")
        fig.colorbar(cf, ax=axs, orientation="horizontal", pad=0.07, shrink=0.8, label=cbar_label)
        _save_fig(fig, stem)

    _panel_fill_quiver(era5_tp, "figure_daily_evolution_era5tp_qflux_era5", "ERA5 daily precipitation (mm day$^{-1}$)")
    _panel_fill_quiver(mswep_tp, "figure_daily_evolution_mswep_qflux_era5", "MSWEP daily precipitation (mm day$^{-1}$)")

    def _panel_mfc_contour(contour_tp, stem):
        fig, axs = plt.subplots(1, 4, figsize=(20, 4.2), subplot_kw={"projection": ccrs.PlateCarree()})
        mfc_abs = np.abs(mfc.values)
        if np.all(np.isnan(mfc_abs)):
            raise ValueError("All MFC values are NaN; cannot draw MFC maps.")
        mm = max(float(np.nanpercentile(mfc_abs, 98)), 1e-4)
        levels = np.linspace(-mm, mm, 15)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-mm, vmax=mm)
        tp_smoothed = contour_tp.rolling(lat=3, lon=3, center=True, min_periods=1).mean()
        tp_smoothed_reg = _sel_region(tp_smoothed, CONTOUR_REGION)
        for i, d in enumerate(TARGET_DATES):
            ax = axs[i]; _setup_map(ax)
            cf = ax.contourf(mfc.lon, mfc.lat, mfc.sel(time=d), levels=levels, cmap="RdBu_r", norm=norm, extend="both", transform=ccrs.PlateCarree())
            tp_day_reg = tp_smoothed_reg.sel(time=d)
            if tp_day_reg.size == 0 or np.all(np.isnan(tp_day_reg.values)):
                print(f"[Warning] No valid TP data in contour region for {d.strftime('%Y-%m-%d')}; skip contour.")
                pr_levels = []
            else:
                daily_max = float(tp_day_reg.max())
                pr_levels = [x for x in CONTOUR_LEVELS if x <= daily_max]
            if pr_levels:
                ax.contour(tp_smoothed_reg.lon, tp_smoothed_reg.lat, tp_day_reg, levels=pr_levels, colors="k", linewidths=0.7, alpha=0.65, transform=ccrs.PlateCarree())
            mfc_mean = float(_area_mean(mfc.sel(time=d)))
            tp_mean = float(_area_mean(contour_tp.sel(time=d)))
            ax.text(0.02, 0.03, f"NCHN MFC = {mfc_mean:.1f}\nNCHN TP = {tp_mean:.1f} mm/day", transform=ax.transAxes,
                    fontsize=8, ha="left", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            if i == 0:
                ax.text(0.02, 0.95, "TP contours over N China: 10/20/40/80 mm/day", transform=ax.transAxes,
                        fontsize=7, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            _plot_box(ax)
            ax.set_title(d.strftime("%Y-%m-%d BJT"))
        fig.colorbar(cf, ax=axs, orientation="horizontal", pad=0.07, shrink=0.8, label="MFC = -div(qV) × 1e5 (scaled)")
        _save_fig(fig, stem)

    _panel_mfc_contour(era5_tp, "figure_daily_evolution_mfc_era5tp_era5")
    _panel_mfc_contour(mswep_tp, "figure_daily_evolution_mfc_mswep_era5")


if __name__ == "__main__":
    run_daily_evolution_precip_moisture()
