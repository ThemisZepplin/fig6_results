import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==============================
# Cell Y: ERA5 circulation–precipitation daily evolution maps
# ==============================
ERA5_TP_FILE = "/data1/huangy/fig6/NC/MSE/ERA5/tp/era5_tp_2023_06.nc"
ERA5_Z500_2023_FILE = "/data1/huangy/fig6/NC/MSE/ERA5/Z500/era5_z500_2023_06.nc"
ERA5_Z500_DIR = "/data1/huangy/fig6/NC/MSE/ERA5/Z500"
ERA5_UVQ_FILE_NEW = "/data1/huangy/fig6/NC/MSE/ERA5/850uv+q/era5_850_uvq_2023_06.nc"
ERA5_UVQ_FILE_OLD = "/data1/huangy/fig6/NC/MSE/ERA5/850 uv+q/era5_850_uvq_2023_06.nc"
MSWEP_2023_DIR = "/data1/huangy/fig6/NC/MSE/MSWEP/3hourly-2023"
OUT_DIR = "/data1/huangy/fig6/NC/v6/factor_verification/daily_evolution_circulation_precip"
os.makedirs(OUT_DIR, exist_ok=True)

MAP_EXTENT = (60.0, 150.0, 10.0, 60.0)
TARGET_DATES = pd.to_datetime(["2023-06-17", "2023-06-18", "2023-06-19", "2023-06-20"])
CLIM_YEARS = list(range(2003, 2023))
REQUIRE_Z500_CLIM_FOR_ANOMALY = True
VECTOR_COLOR = "#00C853"
WIND_VECTOR_STRIDE = 24
WIND_MIN_MS = 1.5

HCNC_BOX = (114.0, 119.0, 35.0, 44.0)
print("HCNC_BOX not found in external context, fallback to (114,119,35,44)")




def _open_dataset_safe(path):
    try:
        return xr.open_dataset(path)
    except Exception as e:
        print(f"[Error] Failed to open dataset: {path}")
        raise RuntimeError(f"Failed to open dataset: {path}. Original error: {repr(e)}") from e

def _guess_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return ds[v], v
    raise KeyError(f"Cannot find variable in {list(ds.data_vars)}; candidates={candidates}")


def _normalize_time_dim(da):
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


def _standardize_lat_lon(da):
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    da = da.rename(rename)
    if float(da.lon.max()) > 180.0:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")
    return da


def _select_level(da, target_hpa=850.0):
    for lev in ["pressure_level", "level", "isobaricInhPa", "isobaricInPa", "plev"]:
        if lev in da.dims or lev in da.coords:
            c = da[lev]
            vals = np.asarray(c.values)
            use = target_hpa if np.nanmax(vals) < 2000 else target_hpa * 100.0
            if lev in da.dims:
                out = da.sel({lev: use}, method="nearest").squeeze(drop=True)
                print(f"Selected level from dim {lev}: target={use}, actual={float(out[lev]) if lev in out.coords else 'dropped'}")
                return out
            if c.ndim == 0:
                print(f"Scalar level coordinate {lev}={float(c.values)}; keep as-is.")
                try:
                    return da.drop_vars(lev)
                except Exception:
                    return da
            print(f"Warning: level coord {lev} non-scalar non-dim; keep as-is")
            return da
    print("No level coord found; assume single-level field.")
    return da


def _to_bjt_daily(da, how):
    da = da.assign_coords(time=da.time + np.timedelta64(8, "h"))
    return getattr(da.resample(time="1D"), how)()


def _area_mean(da, box):
    x0, x1, y0, y1 = box
    sub = da.sel(lon=slice(x0, x1), lat=slice(y0, y1))
    w = np.cos(np.deg2rad(sub.lat))
    return sub.weighted(w).mean(("lat", "lon"))


def _sel_region(da, extent):
    x0, x1, y0, y1 = extent
    return da.sel(lon=slice(x0, x1), lat=slice(y0, y1))


def parse_mswep_filename(path):
    stem = os.path.basename(path).replace(".nc", "")
    yyyyddd, hh = stem.split(".")
    year = int(yyyyddd[:4]); doy = int(yyyyddd[4:]); hour = int(hh)
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1, hours=hour)


def _check_target_dates(da, name):
    t = pd.to_datetime(da.time.values)
    print(f"{name} daily times:", t)
    missing = [d.strftime("%Y-%m-%d") for d in TARGET_DATES if d not in set(t)]
    if missing:
        raise RuntimeError(f"{name} missing target dates: {missing}")


def _plot_box(ax, box):
    x0, x1, y0, y1 = box
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="gold", lw=1.3, transform=ccrs.PlateCarree())


def _setup_map(ax):
    ax.set_extent(MAP_EXTENT, ccrs.PlateCarree())
    ax.coastlines("110m", lw=0.8)
    ax.add_feature(cfeature.BORDERS, lw=0.5)
    ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.4, alpha=0.6)


def _save_fig(fig, stem):
    fig.savefig(os.path.join(OUT_DIR, stem + ".png"), dpi=250, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _count_samples(time_values):
    shifted = pd.to_datetime(time_values) + pd.Timedelta(hours=8)
    s = pd.Series(1, index=shifted).resample("1D").sum()
    return {d.strftime("%Y-%m-%d"): int(s.get(d, 0)) for d in TARGET_DATES}


def _find_uvq_file():
    if os.path.exists(ERA5_UVQ_FILE_NEW):
        return ERA5_UVQ_FILE_NEW
    if os.path.exists(ERA5_UVQ_FILE_OLD):
        return ERA5_UVQ_FILE_OLD
    raise FileNotFoundError(f"Both uvq paths missing: {ERA5_UVQ_FILE_NEW}, {ERA5_UVQ_FILE_OLD}")


def load_era5_tp():
    if not os.path.exists(ERA5_TP_FILE):
        raise FileNotFoundError(ERA5_TP_FILE)
    ds = _open_dataset_safe(ERA5_TP_FILE)
    print("ERA5 TP data_vars/dims/coords:", list(ds.data_vars), dict(ds.dims), list(ds.coords))
    tp, name = _guess_var(ds, ["tp", "total_precipitation", "precip", "precipitation"])
    print("ERA5 TP matched var:", name)
    tp = _standardize_lat_lon(_normalize_time_dim(tp))
    units = str(tp.attrs.get("units", ""))
    if units.lower() in ["m", "meter", "metre", "meters", "metres"]:
        tp = tp * 1000.0
    tp_daily = _to_bjt_daily(tp, "sum")
    tp_daily.attrs["units"] = "mm day^-1"
    return tp_daily, _count_samples(tp.time.values)


def load_mswep_tp():
    if not os.path.isdir(MSWEP_2023_DIR):
        raise FileNotFoundError(MSWEP_2023_DIR)
    files = sorted(glob.glob(os.path.join(MSWEP_2023_DIR, "*.nc")))
    utc0 = TARGET_DATES.min() - pd.Timedelta(hours=8)
    utc1 = TARGET_DATES.max() + pd.Timedelta(days=1) - pd.Timedelta(hours=8)
    files = [f for f in files if utc0 <= parse_mswep_filename(f) <= utc1]
    if not files:
        raise FileNotFoundError(f"No MSWEP files in window: {utc0} to {utc1}")
    arrs = []
    for f in files:
        dsi = _open_dataset_safe(f)
        pri, v = _guess_var(dsi, ["precipitation", "precip", "pr", "rain", "tp"])
        pri = _standardize_lat_lon(pri)
        if any(x in pri.dims or x in pri.coords for x in ["time", "valid_time"]):
            pri = _normalize_time_dim(pri)
        else:
            pri = pri.expand_dims(time=[parse_mswep_filename(f).to_datetime64()])
        arrs.append(pri)
    pr = xr.concat(arrs, dim="time").sortby("time")
    print("MSWEP matched var:", v)
    pr_daily = _to_bjt_daily(pr, "sum")
    pr_daily.attrs["units"] = "mm day^-1"
    return pr_daily, _count_samples(pr.time.values)


def load_z500_2023_daily():
    if not os.path.exists(ERA5_Z500_2023_FILE):
        raise FileNotFoundError(ERA5_Z500_2023_FILE)
    ds = _open_dataset_safe(ERA5_Z500_2023_FILE)
    print("Z500 2023 data_vars/dims/coords:", list(ds.data_vars), dict(ds.dims), list(ds.coords))
    z, zn = _guess_var(ds, ["z", "geopotential", "z500", "gh", "hgt"])
    print("Z500 matched var:", zn)
    z = _standardize_lat_lon(_normalize_time_dim(_select_level(z, 500.0)))
    u = str(z.attrs.get("units", "")).lower()
    conv = "none"
    if "m**2" in u or "m2 s-2" in u or "m^2 s^-2" in u or u.strip() in ["m2/s2", "m^2/s^2"]:
        z = z / 9.80665
        conv = "geopotential->gpm (divide by 9.80665)"
    elif "geopotential" in str(z.attrs.get("long_name", "")).lower() and (u == "" or "s" in u):
        z = z / 9.80665
        conv = "geopotential->gpm (heuristic divide by 9.80665)"
    else:
        conv = "already height-like, no conversion"
    print("Z500 units/conversion:", z.attrs.get("units", ""), conv)
    z_daily = _to_bjt_daily(z, "mean")
    reg = _sel_region(z_daily, MAP_EXTENT)
    print("Z500 converted range over map:", float(np.nanmin(reg.values)), float(np.nanmax(reg.values)))
    return z_daily, _count_samples(z.time.values), conv


def load_uv850_daily():
    uvq = _find_uvq_file()
    print("Using uvq file:", uvq)
    ds = _open_dataset_safe(uvq)
    print("UVQ data_vars/dims/coords:", list(ds.data_vars), dict(ds.dims), list(ds.coords))
    u, un = _guess_var(ds, ["u", "u_component_of_wind"])
    v, vn = _guess_var(ds, ["v", "v_component_of_wind"])
    print("U/V matched vars:", un, vn)
    u = _standardize_lat_lon(_normalize_time_dim(_select_level(u, 850.0)))
    v = _standardize_lat_lon(_normalize_time_dim(_select_level(v, 850.0)))
    return _to_bjt_daily(u, "mean"), _to_bjt_daily(v, "mean"), _count_samples(u.time.values)


def load_z500_clim_daily_2003_2022():
    missing = []
    files = []
    broken_files = []
    date_missing_errors = []

    for y in CLIM_YEARS:
        f = os.path.join(ERA5_Z500_DIR, f"era5_z500_{y}_06.nc")
        if not os.path.exists(f):
            missing.append((y, f))
        else:
            files.append((y, f))

    if missing:
        msg = "Missing climatology files\n" + "\n".join([f"year={y}, file={fp}" for y, fp in missing])
        raise FileNotFoundError(msg)

    out = []
    for y, f in files:
        try:
            ds = _open_dataset_safe(f)
        except Exception as e:
            broken_files.append((y, f, repr(e)))
            continue

        z, _ = _guess_var(ds, ["z", "geopotential", "z500", "gh", "hgt"])
        z = _standardize_lat_lon(_normalize_time_dim(_select_level(z, 500.0)))
        u = str(z.attrs.get("units", "")).lower()
        if "m**2" in u or "m2 s-2" in u or "m^2 s^-2" in u or u.strip() in ["m2/s2", "m^2/s^2"]:
            z = z / 9.80665

        z_daily = _to_bjt_daily(z, "mean")
        hist_dates = pd.to_datetime([f"{y}-06-17", f"{y}-06-18", f"{y}-06-19", f"{y}-06-20"])
        have = set(pd.to_datetime(z_daily.time.values))
        miss_dates = [d.strftime("%Y-%m-%d") for d in hist_dates if d not in have]

        print(f"clim year={y}, hist_dates={list(hist_dates.strftime('%Y-%m-%d'))}, "
              f"available_daily_range=({pd.to_datetime(z_daily.time.values).min()}, {pd.to_datetime(z_daily.time.values).max()})")

        if miss_dates:
            date_missing_errors.append((y, miss_dates))
            continue

        z_hist = z_daily.sel(time=hist_dates)
        print(f"clim year={y}, selected_dates={list(pd.to_datetime(z_hist.time.values).strftime('%Y-%m-%d'))}")
        z_hist = z_hist.assign_coords(time=TARGET_DATES)
        out.append(z_hist)

    if broken_files:
        lines = [f"year={y}, file={f}, error={err}" for y, f, err in broken_files]
        raise RuntimeError("Broken climatology files detected\n" + "\n".join(lines))

    if date_missing_errors:
        lines = [f"year={y} missing same-MMDD target dates: {miss}" for y, miss in date_missing_errors]
        raise RuntimeError("Climatology daily dates missing\n" + "\n".join(lines))

    stack = xr.concat(out, dim="clim_year")
    clim = stack.mean("clim_year")
    return clim, len(files), []


def _prepare_uv850_quiver_arrays(lon, lat, u2d, v2d):
    lon_sub = lon.values[::WIND_VECTOR_STRIDE]
    lat_sub = lat.values[::WIND_VECTOR_STRIDE]
    u_sub = u2d.values[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE]
    v_sub = v2d.values[::WIND_VECTOR_STRIDE, ::WIND_VECTOR_STRIDE]

    lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)
    ws = np.sqrt(u_sub**2 + v_sub**2)
    mask = ws >= WIND_MIN_MS
    return lon2d[mask], lat2d[mask], u_sub[mask], v_sub[mask]


def _plot_uv850_quiver(ax, lon, lat, u2d, v2d):
    x_plot, y_plot, u_plot, v_plot = _prepare_uv850_quiver_arrays(lon, lat, u2d, v2d)

    if u_plot.size == 0:
        print(f"[Warning] No UV850 vectors after weak-wind filter (ws >= {WIND_MIN_MS:.1f} m/s); skip quiver for this panel.")
        return None

    common_kwargs = dict(
        alpha=1.0,
        scale=85,
        pivot="mid",
        headwidth=6.2,
        headlength=7.2,
        headaxislength=6.2,
        transform=ccrs.PlateCarree(),
    )
    ax.quiver(
        x_plot, y_plot, u_plot, v_plot,
        color="white", width=0.0048,
        zorder=20, **common_kwargs
    )
    q = ax.quiver(
        x_plot, y_plot, u_plot, v_plot,
        color=VECTOR_COLOR, width=0.0035,
        zorder=21, **common_kwargs
    )
    return q


def plot_precip_abs_z500_uv(precip, z500, u850, v850, stem, ttl, precip_vmax):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.4), subplot_kw={"projection": ccrs.PlateCarree()})
    cpool = np.r_[precip.values.flatten()]
    if np.all(np.isnan(cpool)): raise ValueError("All precip is NaN")
    pvmax = max(float(precip_vmax), 1.0)
    plev = np.linspace(0, pvmax, 13)
    qobj = None
    for i, d in enumerate(TARGET_DATES):
        ax = axs[i]; _setup_map(ax)
        cf = ax.contourf(precip.lon, precip.lat, precip.sel(time=d), levels=plev, cmap="Blues", extend="max", transform=ccrs.PlateCarree())
        zz = z500.sel(time=d)
        zlv = np.arange(5400, 6021, 40)
        ax.contour(zz.lon, zz.lat, zz, levels=zlv, colors="0.2", linewidths=0.7, transform=ccrs.PlateCarree())
        c5880 = ax.contour(zz.lon, zz.lat, zz, levels=[5880], colors="purple", linewidths=1.8, transform=ccrs.PlateCarree())
        if len(c5880.allsegs[0]) == 0:
            print(f"Warning: 5880 contour absent on {d.strftime('%Y-%m-%d')}")
        else:
            ax.clabel(c5880, fmt="5880", fontsize=7)
        q = _plot_uv850_quiver(ax, u850.lon, u850.lat, u850.sel(time=d), v850.sel(time=d))
        if i == 0 and q is not None: qobj = q
        tp_mean = float(_area_mean(precip.sel(time=d), HCNC_BOX)); z_mean = float(_area_mean(zz, HCNC_BOX))
        ax.text(0.02, 0.03, f"HCNC TP = {tp_mean:.1f} mm/day\nHCNC Z500 = {z_mean:.0f} gpm", transform=ax.transAxes,
                fontsize=8, ha="left", va="bottom", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        _plot_box(ax, HCNC_BOX); ax.set_title(d.strftime("%Y-%m-%d BJT"))
    if qobj is not None:
        axs[0].quiverkey(qobj, 0.08, -0.06, 5.0, "5 m s$^{-1}$", labelpos="E")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.16, wspace=0.08)
    fig.suptitle(ttl, y=0.93, fontsize=12)
    fig.colorbar(cf, ax=axs, orientation="horizontal", pad=0.04, shrink=0.88, label="Precipitation (mm day$^{-1}$)")
    _save_fig(fig, stem)


def plot_precip_z500anom_uv(precip, z500_abs, z500_anom, u850, v850, stem, ttl, precip_vmax):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.4), subplot_kw={"projection": ccrs.PlateCarree()})
    cpool = np.r_[precip.values.flatten()]
    if np.all(np.isnan(cpool)): raise ValueError("All precip is NaN")
    pvmax = max(float(precip_vmax), 1.0)
    plev = np.linspace(0, pvmax, 13)
    anom_levels = np.array([-120, -100, -80, -60, -40, -20, 20, 40, 60, 80, 100, 120])
    qobj = None
    for i, d in enumerate(TARGET_DATES):
        ax = axs[i]; _setup_map(ax)
        cf = ax.contourf(precip.lon, precip.lat, precip.sel(time=d), levels=plev, cmap="Blues", extend="max", transform=ccrs.PlateCarree())
        aa = z500_anom.sel(time=d)
        neg = [l for l in anom_levels if l < 0 and l >= float(np.nanmin(aa.values))]
        pos = [l for l in anom_levels if l > 0 and l <= float(np.nanmax(aa.values))]
        if neg: ax.contour(aa.lon, aa.lat, aa, levels=neg, colors="royalblue", linestyles="--", linewidths=0.9, transform=ccrs.PlateCarree())
        if pos: ax.contour(aa.lon, aa.lat, aa, levels=pos, colors="firebrick", linestyles="-", linewidths=0.9, transform=ccrs.PlateCarree())
        c5880 = ax.contour(z500_abs.lon, z500_abs.lat, z500_abs.sel(time=d), levels=[5880], colors="purple", linewidths=1.8, transform=ccrs.PlateCarree())
        if len(c5880.allsegs[0]) == 0:
            print(f"Warning: 5880 contour absent on {d.strftime('%Y-%m-%d')}")
        q = _plot_uv850_quiver(ax, u850.lon, u850.lat, u850.sel(time=d), v850.sel(time=d))
        if i == 0 and q is not None: qobj = q
        tp_mean = float(_area_mean(precip.sel(time=d), HCNC_BOX))
        z_mean = float(_area_mean(z500_abs.sel(time=d), HCNC_BOX))
        za_mean = float(_area_mean(aa, HCNC_BOX))
        ax.text(0.02, 0.03, f"HCNC TP = {tp_mean:.1f} mm/day\nHCNC Z500 = {z_mean:.0f} gpm\nHCNC Z500 anom = {za_mean:.1f} gpm",
                transform=ax.transAxes, fontsize=8, ha="left", va="bottom", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        _plot_box(ax, HCNC_BOX); ax.set_title(d.strftime("%Y-%m-%d BJT"))
    if qobj is not None:
        axs[0].quiverkey(qobj, 0.08, -0.06, 5.0, "5 m s$^{-1}$", labelpos="E")
    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.16, wspace=0.08)
    fig.suptitle(ttl + "\nZ500 anomaly = 2023 - ERA5 2003–2022 daily climatology", y=0.94, fontsize=11)
    fig.colorbar(cf, ax=axs, orientation="horizontal", pad=0.04, shrink=0.88, label="Precipitation (mm day$^{-1}$)")
    _save_fig(fig, stem)


def run_daily_evolution_circulation_precip():
    era5_tp, era5_tp_samples = load_era5_tp()
    mswep_tp, mswep_samples = load_mswep_tp()
    z500_2023, z500_samples, zconv = load_z500_2023_daily()
    u850, v850, uv_samples = load_uv850_daily()

    for nm, da in [("era5_tp", era5_tp), ("mswep_tp", mswep_tp), ("z500_2023", z500_2023), ("u850", u850), ("v850", v850)]:
        _check_target_dates(da, nm)

    era5_tp = _sel_region(era5_tp.sel(time=TARGET_DATES), MAP_EXTENT).transpose("time", "lat", "lon")
    mswep_tp = _sel_region(mswep_tp.sel(time=TARGET_DATES), MAP_EXTENT).transpose("time", "lat", "lon")
    z500_2023 = _sel_region(z500_2023.sel(time=TARGET_DATES), MAP_EXTENT).transpose("time", "lat", "lon")
    u850 = _sel_region(u850.sel(time=TARGET_DATES), MAP_EXTENT).transpose("time", "lat", "lon")
    v850 = _sel_region(v850.sel(time=TARGET_DATES), MAP_EXTENT).transpose("time", "lat", "lon")


    z500_clim = None
    z500_anom = None
    clim_count = 0
    clim_missing = []

    pmax = max(float(np.nanpercentile(era5_tp.values, 98)), float(np.nanpercentile(mswep_tp.values, 98)), 1.0)
    era5_tp_plot = era5_tp.copy(); mswep_tp_plot = mswep_tp.copy()
    era5_tp_plot.values = np.clip(era5_tp_plot.values, 0, pmax)
    mswep_tp_plot.values = np.clip(mswep_tp_plot.values, 0, pmax)

    plot_precip_abs_z500_uv(era5_tp_plot, z500_2023, u850, v850, "figure_daily_evolution_era5tp_z500_uv850_abs", "ERA5 TP + ERA5 Z500(abs) + ERA5 UV850", pmax)
    plot_precip_abs_z500_uv(mswep_tp_plot, z500_2023, u850, v850, "figure_daily_evolution_mswep_z500_uv850_abs", "MSWEP TP + ERA5 Z500(abs) + ERA5 UV850", pmax)

    try:
        z500_clim, clim_count, clim_missing = load_z500_clim_daily_2003_2022()
        _check_target_dates(z500_clim, "z500_clim")
        z500_clim = _sel_region(z500_clim.sel(time=TARGET_DATES), MAP_EXTENT).transpose("time", "lat", "lon")
        z500_anom = z500_2023 - z500_clim
    except Exception as e:
        if REQUIRE_Z500_CLIM_FOR_ANOMALY:
            raise
        print(f"[Warning] Skip anomaly plots because climatology unavailable: {repr(e)}")
        z500_clim = None
        z500_anom = None
        clim_count = 0
        clim_missing = ["unknown"]


    qc = []
    for d in TARGET_DATES:
        ds = d.strftime("%Y-%m-%d")
        print(f"Sample check {ds}: ERA5 TP={era5_tp_samples[ds]}, MSWEP={mswep_samples[ds]}, Z500={z500_samples[ds]}, UV={uv_samples[ds]}")
        z500_clim_mean = np.nan
        z500_anom_mean = np.nan
        if z500_clim is not None and z500_anom is not None:
            z500_clim_mean = float(_area_mean(z500_clim.sel(time=d), HCNC_BOX))
            z500_anom_mean = float(_area_mean(z500_anom.sel(time=d), HCNC_BOX))
        qc.append({
            "bjt_date": ds,
            "era5_tp_hcnc_mean": float(_area_mean(era5_tp.sel(time=d), HCNC_BOX)),
            "mswep_tp_hcnc_mean": float(_area_mean(mswep_tp.sel(time=d), HCNC_BOX)),
            "z500_hcnc_mean": float(_area_mean(z500_2023.sel(time=d), HCNC_BOX)),
            "z500_clim_hcnc_mean": z500_clim_mean,
            "z500_anom_hcnc_mean": z500_anom_mean,
            "u850_hcnc_mean": float(_area_mean(u850.sel(time=d), HCNC_BOX)),
            "v850_hcnc_mean": float(_area_mean(v850.sel(time=d), HCNC_BOX)),
            "era5_tp_samples": era5_tp_samples[ds],
            "mswep_tp_samples": mswep_samples[ds],
            "z500_2023_samples": z500_samples[ds],
            "uv850_samples": uv_samples[ds],
            "z500_clim_year_count": clim_count,
            "z500_clim_missing_years": ";".join(map(str, clim_missing)),
            "z500_unit_after_conversion": "gpm",
            "bjt_start": f"{ds} 00:00",
            "bjt_end": (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00"),
            "utc_start": (d - pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M"),
            "utc_end": (d + pd.Timedelta(hours=16)).strftime("%Y-%m-%d %H:%M"),
        })
    pd.DataFrame(qc).to_csv(os.path.join(OUT_DIR, "daily_evolution_era5_z500_uv850_tp_qc.csv"), index=False)

    if z500_anom is not None:
        plot_precip_z500anom_uv(era5_tp_plot, z500_2023, z500_anom, u850, v850, "figure_daily_evolution_era5tp_z500anom_uv850", "ERA5 TP + ERA5 Z500(anom) + ERA5 UV850", pmax)
        plot_precip_z500anom_uv(mswep_tp_plot, z500_2023, z500_anom, u850, v850, "figure_daily_evolution_mswep_z500anom_uv850", "MSWEP TP + ERA5 Z500(anom) + ERA5 UV850", pmax)


if __name__ == "__main__":
    run_daily_evolution_circulation_precip()
