import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm_api
import xarray as xr
from relativeImp import relativeImp
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler

import swanlab

# 忽略 xarray 的 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 0. 全局设置与数据一次性读取
# =============================================================================
start_date = "2023-06-12"
start_time = pd.Timestamp(start_date)
study_start = pd.Timestamp("2023-06-14")
study_end = pd.Timestamp("2023-06-24")

swanlab.init(
    project="Heatwave-Subseasonal-Forecast",
    experiment_name=f"RIA_Model_Compare_S2S_{start_date}",
    config={
        "start_date": start_date,
        "study_period": f"{study_start.date()} to {study_end.date()}",
        "model_type": "OLS + Dominance Analysis",
        "description": "双模型对比(纯S2S版): 将 S2S 模式前3天平均作为记忆因子",
    },
)

print("正在加载基础变量数据 (S2S)...")
# 1. T2m
tmxfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_mx2t6_2023-06.grb"
tmx_ds = xr.open_dataset(tmxfile, engine="cfgrib")
tmx = tmx_ds["mx2t6"].loc[:, start_date, :, 44:35, 114:119] - 273.16
daily_max_tmx = tmx.resample(step="D").max()

# 2. SM20 (土壤湿度)
smfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_sm2023-06.grb"
sm_ds = xr.open_dataset(smfile, engine="cfgrib")
sm = sm_ds["sm20"].loc[:, start_date, :, 44:35, 114:119]
daily_max_sm = sm.resample(step="D").max()

# 3. TP (降水)
tpfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_tp2023-06.grb"
tp_ds = xr.open_dataset(tpfile, engine="cfgrib")
NCHN_tp_data = tp_ds["tp"].loc[:, start_date, :, 44:35, 114:119]
daily_max_NCHNtp = NCHN_tp_data.resample(step="D").max().diff(dim="step")

# 4. Z500 & WNPSH
zfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"
z_ds = xr.open_dataset(zfile, engine="cfgrib")
reforecastfile = f"/data1/huangy/fig6/NC/pl_z03_22/merged_{start_date}.nc"
reforecast = xr.open_dataset(reforecastfile)

z_NCHN = z_ds["gh"].loc[:, start_date, :, 500, 44:35, 114:119]
daily_max_z_NCHN = z_NCHN.resample(step="D").max()
z_refo_NCHN = reforecast["gh"].loc[:, :, :, 500, 44:35, 114:119]
daily_max_z_refo_NCHN = z_refo_NCHN.resample(step="D").max()

z_WNPSH = z_ds["gh"].loc[:, start_date, :, 850, 25:15, 115:150]
daily_max_z_WNPSH = z_WNPSH.resample(step="D").max()

# 5. SSR (辐射)
ssr_file = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_radiation2023-06.grb"
ssr_ds = xr.open_dataset(ssr_file, engine="cfgrib")
ssr = ssr_ds["ssr"].loc[:, start_date, :, 44:35, 114:119]
daily_mean_ssr = ssr.resample(step="D").mean()

# 6. SSHF (感热通量)
sshf_file = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_hflux2023-06.grb"
sshf_ds = xr.open_dataset(sshf_file, engine="cfgrib")
sshf = sshf_ds["sshf"].loc[:, start_date, :, 44:35, 114:119]
daily_mean_sshf = sshf.resample(step="D").mean()

# 7. Low-level Stability
cp, Lv, g = 1005.0, 2.5e6, 9.81
t1000 = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_pl1000_130_2022-08.grib", engine="cfgrib"
)["t"].loc[:, start_date, :, 44:35, 114:119]
q1000 = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_pl1000_133_2022-08.grib", engine="cfgrib"
)["q"].loc[:, start_date, :, 44:35, 114:119]
z1000 = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_pl1000_156_2022-08.grib", engine="cfgrib"
)["gh"].loc[:, start_date, :, 44:35, 114:119]
mse_1000 = cp * t1000 + Lv * q1000 + g * z1000

mse_star_list = []
for lev in [925, 850, 700, 500]:
    t_lev = xr.open_dataset(
        f"/data1/huangy/fig6/NC/MSE/ecmf_pl{lev}_130_2022-08.grib", engine="cfgrib"
    )["t"].loc[:, start_date, :, 44:35, 114:119]
    z_lev = xr.open_dataset(
        f"/data1/huangy/fig6/NC/MSE/ecmf_pl{lev}_156_2022-08.grib", engine="cfgrib"
    )["gh"].loc[:, start_date, :, 44:35, 114:119]
    t_c = t_lev - 273.15
    es_pa = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5)) * 100.0
    p_pa = lev * 100.0
    q_star_lev = (0.622 * es_pa) / (p_pa - 0.378 * es_pa)
    mse_star_list.append(cp * t_lev + Lv * q_star_lev + g * z_lev)

mse_star_max = xr.concat(mse_star_list, dim="level").max(dim="level")
stability_index = mse_1000 - mse_star_max
daily_max_stability = stability_index.resample(step="D").max()


# =============================================================================
# 8. 提取 S2S 前兆因子（改为起报后初始3天平均）
# =============================================================================
print("正在加载 S2S 前兆因子数据 (初始3天平均)...")
dir_s2s_antecedent = "/data1/huangy/fig6/NC/MSE/"


def select_init_time(data_array: xr.DataArray, init_date: pd.Timestamp) -> xr.DataArray:
    """统一将因子筛选到指定起报时间（如 2023-06-12）。"""
    for time_dim in ["time", "forecast_reference_time", "valid_time", "date"]:
        if time_dim in data_array.dims:
            return data_array.sel({time_dim: init_date.strftime("%Y-%m-%d")})
    return data_array


def extract_first_3day_mean(data_array: xr.DataArray) -> np.ndarray:
    """对输入因子取起报后初始3天平均，并保留50个集合成员。"""
    data_array = select_init_time(data_array, start_time)
    if "step" in data_array.dims:
        data_array = data_array.isel(step=slice(0, 3))
    dims_to_mean = [d for d in data_array.dims if d != "number"]
    return data_array.mean(dim=dims_to_mean).values.flatten()


# (1) NCVI (华北冷涡 PV)
pv_file = f"{dir_s2s_antecedent}ecmf_pf_60_2023-06.grib"
pv_ds = xr.open_dataset(pv_file, engine="cfgrib")
pv_region = pv_ds["pv"].sel(latitude=slice(43, 36), longitude=slice(113, 122))
ncvi_arr = extract_first_3day_mean(pv_region)

# (2) ISM (印度夏季风降水)
ism_file = f"{dir_s2s_antecedent}ecmf_pf_228228_2023-06.grib"
ism_ds = xr.open_dataset(ism_file, engine="cfgrib")
ism_region = ism_ds["tp"].sel(latitude=slice(25, 5), longitude=slice(65, 85))
ism_arr = extract_first_3day_mean(ism_region)

# (3) SST Gradient (印太暖池海温梯度)
sst_file = f"{dir_s2s_antecedent}ecmf_pf_34_2023-06.grib"
sst_ds = xr.open_dataset(sst_file, engine="cfgrib")
sst_wp = sst_ds["sst"].sel(latitude=slice(15, 0), longitude=slice(125, 145))
sst_io = sst_ds["sst"].sel(latitude=slice(10, -10), longitude=slice(60, 80))
sst_grad_arr = extract_first_3day_mean(sst_wp) - extract_first_3day_mean(sst_io)

# =============================================================================
# 9. ERA5 观测基准数据
# =============================================================================
print("正在加载 ERA5 观测基准数据...")
era5_file = "/data1/huangy/fig6/NC/MSE/T2m_era5_2023.6_NC.nc"
_era5_ds = xr.open_dataset(era5_file)
# 取数据集中第一个（亦即温度）变量
_era5_var = list(_era5_ds.data_vars)[0]
# ERA5 纬度坐标为降序（北→南），与 S2S GRIB 数据保持一致
_era5_t2m = _era5_ds[_era5_var].sel(latitude=slice(44, 35), longitude=slice(114, 119))
# 自动检测时间维度名称（ERA5 可能为 "time" 或 "valid_time" 等）
_time_dim = next(
    (d for d in _era5_t2m.dims if "time" in d.lower()),
    None,
)
if _time_dim is None:
    raise ValueError(
        f"无法在 ERA5 变量 '{_era5_var}' 的维度 {list(_era5_t2m.dims)} 中找到时间维度，"
        "请确认数据文件格式。"
    )
# 若时间维度名不是 "time"，重命名为 "time" 以统一后续操作
if _time_dim != "time":
    _era5_t2m = _era5_t2m.rename({_time_dim: "time"})
# 根据变量单位属性判断是否需要 K→°C 换算
_units = _era5_ds[_era5_var].attrs.get("units", "")
_sample_val = float(_era5_t2m.isel({d: 0 for d in _era5_t2m.dims}).values)
if _units.lower() in ("k", "kelvin") or (not _units and _sample_val > 200):
    _era5_t2m = _era5_t2m - 273.15
# 取逐日最高气温（若已为逐日数据则 resample 不改变结果）
_era5_daily = _era5_t2m.resample(time="1D").max()
# 仅保留研究时段（2023-06-14 至 2023-06-24）
_era5_study = _era5_daily.sel(time=slice(study_start, study_end))
# 取研究区域空间均值，得到形状 (n_days,) 的 1-D 数组
_spatial_dims = [d for d in _era5_study.dims if d != "time"]
era5_obs_ts = _era5_study.mean(dim=_spatial_dims).values  # shape: (n_days,)


# =============================================================================
# 公共处理函数
# =============================================================================
def extract_period_mean(daily_data, is_reforecast=False):
    step_vals = daily_data.step
    dates = [start_time + pd.Timedelta(days=s) for s in range(1, len(step_vals) + 1)]
    dates_ts = pd.to_datetime(dates, format="%Y-%m-%d")
    selected = daily_data.sel(
        step=[s for s, d in zip(step_vals.values, dates_ts) if study_start <= d <= study_end]
    )
    if is_reforecast:
        return selected.mean(dim=list(selected.dims)).values.flatten()[0]

    dims_to_mean = [d for d in selected.dims if d != "number"]
    return selected.mean(dim=dims_to_mean).values.flatten()


def extract_daily_timeseries(daily_data, is_reforecast=False):
    """提取集合均值的逐日时间序列，返回形状 (n_days,) 的数组。"""
    step_vals = daily_data.step
    dates = [start_time + pd.Timedelta(days=s) for s in range(1, len(step_vals) + 1)]
    dates_ts = pd.to_datetime(dates, format="%Y-%m-%d")
    selected = daily_data.sel(
        step=[s for s, d in zip(step_vals.values, dates_ts) if study_start <= d <= study_end]
    )
    dims_to_mean = [d for d in selected.dims if d != "step"]
    return selected.mean(dim=dims_to_mean).values.flatten()


def extract_daily_timeseries_per_member(daily_data):
    """提取每个集合成员的逐日时间序列，返回形状 (n_members, n_days) 的数组。

    对空间维度求平均，保留 number（集合成员）和 step（时间步）两个维度。
    """
    step_vals = daily_data.step
    dates = [start_time + pd.Timedelta(days=s) for s in range(1, len(step_vals) + 1)]
    dates_ts = pd.to_datetime(dates, format="%Y-%m-%d")
    selected = daily_data.sel(
        step=[s for s, d in zip(step_vals.values, dates_ts) if study_start <= d <= study_end]
    )
    # 仅对空间维度（非 step 和 number）求平均
    dims_to_mean = [d for d in selected.dims if d not in ("step", "number")]
    if dims_to_mean:
        result = selected.mean(dim=dims_to_mean)
    else:
        result = selected
    # 确保输出维度顺序为 (number, step)
    if "number" in result.dims and "step" in result.dims:
        result = result.transpose("number", "step")
    return result.values  # shape: (n_members, n_days)


# =============================================================================
# 核心：执行双模型对比实验
# =============================================================================
def run_experiment(period_name, exp_config):
    model_name = exp_config["name"]
    extra_cols = exp_config["extra_cols"]
    print(f"\n{'=' * 50}\n正在运行模型: {model_name}\n{'=' * 50}")

    # 1. 基础特征提取 (50个成员的 1D 数组)
    temp_mean_val = extract_period_mean(daily_max_tmx)
    sm_avg_val = extract_period_mean(daily_max_sm)
    NCHN_tp_val = extract_period_mean(daily_max_NCHNtp)
    WNPSH_avg_val = extract_period_mean(daily_max_z_WNPSH)
    SSR_Avg_val = extract_period_mean(daily_mean_ssr)
    Stability_NCHN_val = extract_period_mean(daily_max_stability)
    SHF_Avg_val = extract_period_mean(daily_mean_sshf)
    z500_2023_NCHN_val = extract_period_mean(daily_max_z_NCHN)
    z500_mean_NCHN_val = extract_period_mean(daily_max_z_refo_NCHN, is_reforecast=True)
    z500_anom_NCHN_val = z500_2023_NCHN_val - z500_mean_NCHN_val

    data_dict = {
        "temp_mean": temp_mean_val,
        "NCHN_tp": NCHN_tp_val,
        "sm_avg": sm_avg_val,
        "WNPSH": WNPSH_avg_val,
        "SSR_Avg": SSR_Avg_val,
        "Stability_NCHN": Stability_NCHN_val,
        "SHF_Avg": SHF_Avg_val,
        "z500_anom_NCHN": z500_anom_NCHN_val,
        "NCVI": ncvi_arr,
        "ISM": ism_arr,
        "SST_Grad": sst_grad_arr,
    }

    factor_cols = [
        "NCHN_tp",
        "sm_avg",
        "WNPSH",
        "SSR_Avg",
        "Stability_NCHN",
        "SHF_Avg",
        "z500_anom_NCHN",
    ] + extra_cols

    data_new = pd.DataFrame(data_dict).dropna()
    y_final = data_new["temp_mean"]
    X_final = data_new[factor_cols]

    label_map = {
        "temp_mean": "T2max",
        "NCHN_tp": "NCHN\nPrecip",
        "sm_avg": "SM\n(Avg)",
        "WNPSH": "WNPSH\n(Avg)",
        "SSR_Avg": "SSR\n(Avg)",
        "Stability_NCHN": "Stability\nNCHN",
        "SHF_Avg": "Sensible\nHeat",
        "z500_anom_NCHN": "NCHN\nZ500",
        "NCVI": "NCVI\n(S2S)",
        "ISM": "ISM\n(S2S)",
        "SST_Grad": "SST Grad\n(S2S)",
    }

    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X_final)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X_final.columns)
    X_scaled_ols = sm_api.add_constant(X_scaled)

    model_full = sm_api.OLS(y_final, X_scaled_ols).fit()
    r_squared_full = model_full.rsquared
    r_squared_adj = model_full.rsquared_adj
    print(f"[{model_name}] OLS R²: {r_squared_full:.4f} | Adj R²: {r_squared_adj:.4f}")

    df_ria = pd.concat([y_final.reset_index(drop=True), X_scaled.reset_index(drop=True)], axis=1)
    ria_results = relativeImp(df_ria, outcomeName="temp_mean", driverNames=factor_cols)

    # 2. 逐日序列预测 (Time Series)
    # --- 集合均值时间序列 ---
    ts_y = extract_daily_timeseries(daily_max_tmx)
    days_len = len(ts_y)
    ts_z500_refo_mean = extract_daily_timeseries(daily_max_z_refo_NCHN, is_reforecast=True)

    ts_data_dict = {
        "NCHN_tp": extract_daily_timeseries(daily_max_NCHNtp),
        "sm_avg": extract_daily_timeseries(daily_max_sm),
        "WNPSH": extract_daily_timeseries(daily_max_z_WNPSH),
        "SSR_Avg": extract_daily_timeseries(daily_mean_ssr),
        "Stability_NCHN": extract_daily_timeseries(daily_max_stability),
        "SHF_Avg": extract_daily_timeseries(daily_mean_sshf),
        "z500_anom_NCHN": extract_daily_timeseries(daily_max_z_NCHN) - ts_z500_refo_mean,
    }
    ts_data_dict["NCVI"] = np.repeat(ncvi_arr.mean(), days_len)
    ts_data_dict["ISM"] = np.repeat(ism_arr.mean(), days_len)
    ts_data_dict["SST_Grad"] = np.repeat(sst_grad_arr.mean(), days_len)

    ts_X_final = pd.DataFrame(ts_data_dict)[factor_cols]
    ts_X_scaled_arr = scaler.transform(ts_X_final)
    ts_X_scaled = pd.DataFrame(ts_X_scaled_arr, columns=ts_X_final.columns)
    ts_X_scaled_ols = sm_api.add_constant(ts_X_scaled, has_constant="add")
    ts_y_pred = model_full.predict(ts_X_scaled_ols)

    # --- 逐成员时间序列（用于集合展布可视化）---
    ts_tmx_members = extract_daily_timeseries_per_member(daily_max_tmx)   # (50, n_days)
    ts_z500_members = extract_daily_timeseries_per_member(daily_max_z_NCHN)  # (50, n_days)

    n_members = ts_tmx_members.shape[0]

    # 为每个成员构建特征并预测
    ts_member_preds = np.full((n_members, days_len), np.nan)
    per_member_feature_arrays = {
        "NCHN_tp": extract_daily_timeseries_per_member(daily_max_NCHNtp),
        "sm_avg": extract_daily_timeseries_per_member(daily_max_sm),
        "WNPSH": extract_daily_timeseries_per_member(daily_max_z_WNPSH),
        "SSR_Avg": extract_daily_timeseries_per_member(daily_mean_ssr),
        "Stability_NCHN": extract_daily_timeseries_per_member(daily_max_stability),
        "SHF_Avg": extract_daily_timeseries_per_member(daily_mean_sshf),
        "z500_anom_NCHN": ts_z500_members - ts_z500_refo_mean,  # broadcast (50, n_days) - scalar
    }

    for m in range(n_members):
        member_dict = {key: arr[m] for key, arr in per_member_feature_arrays.items()}
        member_dict["NCVI"] = np.repeat(ncvi_arr[m], days_len)
        member_dict["ISM"] = np.repeat(ism_arr[m], days_len)
        member_dict["SST_Grad"] = np.repeat(sst_grad_arr[m], days_len)

        ts_X_m = pd.DataFrame(member_dict)[factor_cols]
        ts_X_scaled_m = scaler.transform(ts_X_m)
        ts_X_scaled_ols_m = sm_api.add_constant(
            pd.DataFrame(ts_X_scaled_m, columns=ts_X_m.columns),
            has_constant="add",
        )
        ts_member_preds[m] = model_full.predict(ts_X_scaled_ols_m).values

    # 3. 绘图部分
    fig_main, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(18, 7))

    y_pred = model_full.predict(X_scaled_ols)
    slope, intercept, r_value, _, _ = linregress(y_final, y_pred)

    ax_scatter.scatter(y_final, y_pred, color="blue", alpha=0.6, s=100)
    ax_scatter.plot(y_final, slope * y_final + intercept, color="black", alpha=0.5, lw=2)

    text_str = f"Corr: {r_value:.2f}\n$R^2$: {r_squared_full:.2f}\nAdj $R^2$: {r_squared_adj:.2f}"
    ax_scatter.text(
        0.05,
        0.95,
        text_str,
        transform=ax_scatter.transAxes,
        fontsize=14,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ax_scatter.set_title(f"Predicted vs ECMWF T2max ({model_name})", fontsize=16, pad=15)
    ax_scatter.set_xlabel("ECMWF T2max (°C)", fontsize=14)
    ax_scatter.set_ylabel("Predicted T2max (°C)", fontsize=14)
    ax_scatter.text(-0.05, 1.05, "a)", transform=ax_scatter.transAxes, fontsize=20, weight="bold")

    drivers = ria_results["driver"].values
    norm_vals = ria_results["normRelaImpt"].values
    sorted_indices = np.argsort(norm_vals)[::-1]
    display_labels = [label_map.get(l, l) for l in drivers[sorted_indices]]

    bars = ax_bar.bar(display_labels, norm_vals[sorted_indices], color="royalblue", alpha=0.7)
    ax_bar.set_title(f"Factor Relative Importance ({model_name})", fontsize=16, pad=15)
    ax_bar.set_ylabel("Normalized Relative Importance (%)", fontsize=14)

    ax_bar.set_xticks(range(len(display_labels)))
    ax_bar.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)

    for i, bar in enumerate(bars):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{norm_vals[sorted_indices][i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax_bar.text(-0.05, 1.05, "b)", transform=ax_bar.transAxes, fontsize=20, weight="bold")

    plt.tight_layout()
    plt.savefig(
        f"/data1/huangy/fig6/NC/1.s2s.fig_scatter_bar_{model_name}_{start_date}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # -------------------------------------------------------------------------
    # 时间序列图：集合展布（意大利面图）+ 集合均值
    # -------------------------------------------------------------------------
    fig_ts, ax_ts = plt.subplots(figsize=(14, 7))

    plot_dates = pd.date_range(start=study_start, end=study_end)

    # 绘制 50 个成员的 ECMWF T2max 曲线（浅红，半透明）
    for m in range(n_members):
        label = "Ensemble Members (ECMWF)" if m == 0 else None
        ax_ts.plot(
            plot_dates,
            ts_tmx_members[m],
            color="lightcoral",
            alpha=0.25,
            lw=0.8,
            label=label,
        )

    # 绘制 50 个成员的预测 T2max 曲线（浅蓝，半透明）
    for m in range(n_members):
        label = "Ensemble Members (Predicted)" if m == 0 else None
        ax_ts.plot(
            plot_dates,
            ts_member_preds[m],
            color="lightskyblue",
            alpha=0.25,
            lw=0.8,
            label=label,
        )

    # 绘制集合均值 ECMWF T2max（粗红线）
    ax_ts.plot(
        plot_dates,
        ts_y,
        marker="o",
        color="crimson",
        lw=2.5,
        zorder=5,
        label="ECMWF $T_{max}$ (Ensemble Mean)",
    )

    # 绘制集合均值预测 T2max（粗蓝虚线）
    ax_ts.plot(
        plot_dates,
        ts_y_pred.values,
        marker="s",
        linestyle="--",
        color="dodgerblue",
        lw=2.5,
        zorder=5,
        label="Predicted $T_{max}$ (Ensemble Mean)",
    )

    # 绘制 ERA5 观测基准（黑色实线）
    ax_ts.plot(
        plot_dates,
        era5_obs_ts,
        marker="^",
        color="black",
        lw=2.5,
        zorder=6,
        label="ERA5 Observation",
    )

    ax_ts.set_title(f"Time Series Evolution ({model_name})", fontsize=16, pad=15)
    ax_ts.set_ylabel("T2max (°C)", fontsize=14)

    rmse = np.sqrt(np.mean((ts_y - ts_y_pred.values) ** 2))
    ax_ts.text(
        0.02,
        0.92,
        f"RMSE: {rmse:.2f} °C",
        transform=ax_ts.transAxes,
        fontsize=16,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig_ts.autofmt_xdate(rotation=30)
    ax_ts.legend(fontsize=12)
    ax_ts.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        f"/data1/huangy/fig6/NC/1.s2s.fig_ts_{model_name}_{start_date}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))

    features_with_target = pd.concat(
        [y_final.rename("temp_mean").reset_index(drop=True), X_final.reset_index(drop=True)],
        axis=1,
    )
    corr_matrix = features_with_target.corr()

    clean_labels = [label_map.get(col, col).replace("\n", " ") for col in corr_matrix.columns]
    corr_matrix.columns = clean_labels
    corr_matrix.index = clean_labels

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax_corr,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
    )

    ax_corr.set_title(f"Feature and T2max Correlation Matrix ({model_name})", fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(
        f"/data1/huangy/fig6/NC/1.s2s.fig_corr_{model_name}_{start_date}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


experiments = [
    {"name": "Model_A_ISM", "extra_cols": ["NCVI", "ISM"]},
    {"name": "Model_B_SST", "extra_cols": ["NCVI", "SST_Grad"]},
]

for exp in experiments:
    run_experiment("Heatwave Period", exp)

swanlab.finish()
print("S2S 纯预报对比模型全部运行完毕！")
