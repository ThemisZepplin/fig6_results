"""
标准 LMG / Shapley Relative Importance 分析脚本 (S2S 版)

用途：
  对10个前兆因子 [NCHN_tp, sm_avg, WNPSH, SSR_Avg, SHF_Avg,
  z500_anom_NCHN, Stability_NCHN, NCVI, ISM, SST_Grad]
  使用标准 LMG 方法计算相对重要性：
    - 枚举(或蒙特卡洛采样)所有可能的变量进入顺序
    - 计算每个变量在该顺序下带来的增量 R²
    - 对所有顺序取平均，得到该变量的 LMG 相对重要性
"""

import warnings
from itertools import permutations
from math import factorial
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm_api
import xarray as xr
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

FACTOR_COLS = [
    "NCHN_tp",
    "sm_avg",
    "WNPSH",
    "SSR_Avg",
    "SHF_Avg",
    "z500_anom_NCHN",
    "Stability_NCHN",
    "NCVI",
    "ISM",
    "SST_Grad",
]

LABEL_MAP = {
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

# 蒙特卡洛采样置换数（当 p! 过大时使用）
MC_SAMPLES = 10000
# 变量数 ≤ MAX_EXACT_P 时使用精确枚举，否则蒙特卡洛近似
MAX_EXACT_P = 8

swanlab.init(
    project="Heatwave-Subseasonal-Forecast",
    experiment_name=f"RIA_LMG_S2S_{start_date}",
    config={
        "start_date": start_date,
        "study_period": f"{study_start.date()} to {study_end.date()}",
        "model_type": "OLS + LMG Relative Importance",
        "factors": FACTOR_COLS,
        "description": "标准 LMG/Shapley 相对重要性分析 (S2S 版)",
        "mc_samples": MC_SAMPLES,
    },
)

# =============================================================================
# 1. 加载气象数据
# =============================================================================
print("正在加载基础变量数据 (S2S)...")

# 1a. T2m（日最高气温）
tmxfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_mx2t6_2023-06.grb"
tmx_ds = xr.open_dataset(tmxfile, engine="cfgrib")
tmx = tmx_ds["mx2t6"].loc[:, start_date, :, 44:35, 114:119] - 273.16
daily_max_tmx = tmx.resample(step="D").max()

# 1b. SM20（土壤湿度）
smfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_sm2023-06.grb"
sm_ds = xr.open_dataset(smfile, engine="cfgrib")
sm = sm_ds["sm20"].loc[:, start_date, :, 44:35, 114:119]
daily_max_sm = sm.resample(step="D").max()

# 1c. TP（降水）
tpfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_tp2023-06.grb"
tp_ds = xr.open_dataset(tpfile, engine="cfgrib")
NCHN_tp_data = tp_ds["tp"].loc[:, start_date, :, 44:35, 114:119]
daily_max_NCHNtp = NCHN_tp_data.resample(step="D").max().diff(dim="step")

# 1d. Z500 & WNPSH
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

# 1e. SSR（地面短波辐射）
ssr_file = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_radiation2023-06.grb"
ssr_ds = xr.open_dataset(ssr_file, engine="cfgrib")
ssr = ssr_ds["ssr"].loc[:, start_date, :, 44:35, 114:119]
daily_mean_ssr = ssr.resample(step="D").mean()

# 1f. SSHF（感热通量）
sshf_file = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_hflux2023-06.grb"
sshf_ds = xr.open_dataset(sshf_file, engine="cfgrib")
sshf = sshf_ds["sshf"].loc[:, start_date, :, 44:35, 114:119]
daily_mean_sshf = sshf.resample(step="D").mean()

# 1g. 低层大气稳定度（MSE 方法）
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
# 2. 加载 S2S 前兆因子（起报后初始3天平均）
# =============================================================================
print("正在加载 S2S 前兆因子数据 (初始3天平均)...")
dir_s2s_antecedent = "/data1/huangy/fig6/NC/MSE/"


def select_init_time(data_array: xr.DataArray, init_date: pd.Timestamp) -> xr.DataArray:
    """统一将因子筛选到指定起报时间。"""
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


# NCVI（华北冷涡 PV）
pv_ds = xr.open_dataset(f"{dir_s2s_antecedent}ecmf_pf_60_2023-06.grib", engine="cfgrib")
pv_region = pv_ds["pv"].sel(latitude=slice(43, 36), longitude=slice(113, 122))
ncvi_arr = extract_first_3day_mean(pv_region)

# ISM（印度夏季风降水）
ism_ds = xr.open_dataset(
    f"{dir_s2s_antecedent}ecmf_pf_228228_2023-06.grib", engine="cfgrib"
)
ism_region = ism_ds["tp"].sel(latitude=slice(25, 5), longitude=slice(65, 85))
ism_arr = extract_first_3day_mean(ism_region)

# SST Gradient（印太暖池海温梯度）
sst_ds = xr.open_dataset(
    f"{dir_s2s_antecedent}ecmf_pf_34_2023-06.grib", engine="cfgrib"
)
sst_wp = sst_ds["sst"].sel(latitude=slice(15, 0), longitude=slice(125, 145))
sst_io = sst_ds["sst"].sel(latitude=slice(10, -10), longitude=slice(60, 80))
sst_grad_arr = extract_first_3day_mean(sst_wp) - extract_first_3day_mean(sst_io)

# =============================================================================
# 3. ERA5 观测基准数据
# =============================================================================
print("正在加载 ERA5 观测基准数据...")
era5_file = "/data1/huangy/fig6/NC/MSE/T2m_era5_2023.6_NC.nc"
_era5_ds = xr.open_dataset(era5_file)
_era5_var = list(_era5_ds.data_vars)[0]
_era5_t2m = _era5_ds[_era5_var].sel(latitude=slice(44, 35), longitude=slice(114, 119))
_time_dim = next((d for d in _era5_t2m.dims if "time" in d.lower()), None)
if _time_dim is None:
    raise ValueError(
        f"无法在 ERA5 变量 '{_era5_var}' 的维度 {list(_era5_t2m.dims)} 中找到时间维度。"
    )
if _time_dim != "time":
    _era5_t2m = _era5_t2m.rename({_time_dim: "time"})
_units = _era5_ds[_era5_var].attrs.get("units", "")
_sample_val = float(_era5_t2m.isel({d: 0 for d in _era5_t2m.dims}).values)
if _units.lower() in ("k", "kelvin") or (not _units and _sample_val > 200):
    _era5_t2m = _era5_t2m - 273.15
_era5_daily = _era5_t2m.resample(time="1D").max()
_era5_study = _era5_daily.sel(time=slice(study_start, study_end))
_spatial_dims = [d for d in _era5_study.dims if d != "time"]
era5_obs_ts = _era5_study.mean(dim=_spatial_dims).values  # shape: (n_days,)


# =============================================================================
# 公共处理函数
# =============================================================================
def extract_period_mean(daily_data, is_reforecast=False):
    """提取研究时段（study_start ~ study_end）的集合成员均值，返回长度50的一维数组。"""
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
    """提取每个集合成员的逐日时间序列，返回形状 (n_members, n_days) 的数组。"""
    step_vals = daily_data.step
    dates = [start_time + pd.Timedelta(days=s) for s in range(1, len(step_vals) + 1)]
    dates_ts = pd.to_datetime(dates, format="%Y-%m-%d")
    selected = daily_data.sel(
        step=[s for s, d in zip(step_vals.values, dates_ts) if study_start <= d <= study_end]
    )
    dims_to_mean = [d for d in selected.dims if d not in ("step", "number")]
    result = selected.mean(dim=dims_to_mean) if dims_to_mean else selected
    if "number" in result.dims and "step" in result.dims:
        result = result.transpose("number", "step")
    return result.values  # shape: (n_members, n_days)


# =============================================================================
# 4. 标准 LMG / Shapley 相对重要性
# =============================================================================

def _r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    """拟合 OLS 并返回 R²；若只有截距（X 为空）则返回 0。"""
    if X.shape[1] == 0:
        return 0.0
    X_c = sm_api.add_constant(X, has_constant="add")
    try:
        return sm_api.OLS(y, X_c).fit().rsquared
    except Exception:
        return 0.0


def compute_lmg(y: np.ndarray, X: np.ndarray, col_names: List[str],
                mc_samples: int = MC_SAMPLES,
                max_exact_p: int = MAX_EXACT_P,
                random_state: int = 42) -> pd.DataFrame:
    """
    计算标准 LMG (Lindeman-Merenda-Gold) / Shapley 相对重要性。

    对每种可能的变量进入顺序，计算各变量带来的增量 R²，然后取所有顺序的平均值。

    参数
    ----
    y           : 因变量（一维数组）
    X           : 自变量矩阵（二维数组，列数 = p）
    col_names   : 各列的名称列表（长度 = p）
    mc_samples  : 当 p > max_exact_p 时，随机采样的置换数量
    max_exact_p : p ≤ 该值时使用精确枚举，否则使用蒙特卡洛近似
    random_state: 随机种子

    返回
    ----
    DataFrame 含列 ['driver', 'rawRelaImpt', 'normRelaImpt']
    """
    rng = np.random.default_rng(random_state)
    p = X.shape[1]
    indices = list(range(p))

    # 选择枚举策略
    if p <= max_exact_p:
        # 精确枚举所有 p! 种顺序
        order_list = list(permutations(indices))
        print(f"  LMG 精确枚举：{len(order_list)} 种顺序（p={p}）")
    else:
        # 蒙特卡洛随机采样置换
        order_list = [tuple(rng.permutation(indices)) for _ in range(mc_samples)]
        print(f"  LMG 蒙特卡洛近似：{mc_samples} 次随机采样（p={p}，p!={factorial(p):,}）")

    # 累计各变量的增量 R²
    incremental_r2 = np.zeros(p)

    for order in order_list:
        prev_r2 = 0.0
        for pos, var_idx in enumerate(order):
            # 当前顺序中，位于 var_idx 之前（含 var_idx）的变量子集
            subset_with = np.array(list(order[: pos + 1]))
            r2_with = _r2_ols(y, X[:, subset_with])
            incremental_r2[var_idx] += r2_with - prev_r2
            prev_r2 = r2_with

    # 取各顺序的平均
    lmg_raw = incremental_r2 / len(order_list)

    # 归一化为百分比（总和 = 全模型 R²）
    total_lmg = lmg_raw.sum()
    if total_lmg > 0:
        lmg_norm = (lmg_raw / total_lmg) * 100.0
    else:
        lmg_norm = np.zeros(p)

    return pd.DataFrame(
        {"driver": col_names, "rawRelaImpt": lmg_raw, "normRelaImpt": lmg_norm}
    )


# =============================================================================
# 5. 主实验流程
# =============================================================================
print("\n" + "=" * 60)
print("开始提取特征，建立单一模型（所有10个因子）...")
print("=" * 60)

# --- 提取各因子（50个成员的集合均值，代表研究时段均值）---
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
    "SHF_Avg": SHF_Avg_val,
    "z500_anom_NCHN": z500_anom_NCHN_val,
    "Stability_NCHN": Stability_NCHN_val,
    "NCVI": ncvi_arr,
    "ISM": ism_arr,
    "SST_Grad": sst_grad_arr,
}

data_df = pd.DataFrame(data_dict).dropna()
y_final = data_df["temp_mean"].values
X_df = data_df[FACTOR_COLS]

# --- 标准化 ---
scaler = StandardScaler()
X_scaled_arr = scaler.fit_transform(X_df)
X_scaled = pd.DataFrame(X_scaled_arr, columns=FACTOR_COLS)

# --- 全模型 OLS ---
X_scaled_ols = sm_api.add_constant(X_scaled)
model_full = sm_api.OLS(y_final, X_scaled_ols).fit()
r_squared_full = model_full.rsquared
r_squared_adj = model_full.rsquared_adj
print(f"全模型 OLS  R²: {r_squared_full:.4f} | Adj R²: {r_squared_adj:.4f}")

# --- LMG 相对重要性 ---
print("\n计算 LMG 相对重要性（标准 Shapley 分解）...")
lmg_results = compute_lmg(
    y=y_final,
    X=X_scaled_arr,
    col_names=FACTOR_COLS,
)
print("\nLMG 相对重要性结果：")
print(lmg_results.to_string(index=False))

swanlab.log(
    {
        "R2_full": r_squared_full,
        "R2_adj": r_squared_adj,
        **{f"LMG_{row['driver']}": row["rawRelaImpt"] for _, row in lmg_results.iterrows()},
        **{f"LMG_norm_{row['driver']}": row["normRelaImpt"] for _, row in lmg_results.iterrows()},
    }
)

# =============================================================================
# 6. 图1：散点图 + LMG 条形图
# =============================================================================
fig_main, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(18, 7))

y_pred_full = model_full.predict(X_scaled_ols)
slope, intercept, r_value, _, _ = linregress(y_final, y_pred_full)

ax_scatter.scatter(y_final, y_pred_full, color="blue", alpha=0.6, s=100)
ax_scatter.plot(y_final, slope * y_final + intercept, color="black", alpha=0.5, lw=2)
text_str = (
    f"Corr: {r_value:.2f}\n$R^2$: {r_squared_full:.2f}\nAdj $R^2$: {r_squared_adj:.2f}"
)
ax_scatter.text(
    0.05, 0.95, text_str,
    transform=ax_scatter.transAxes, fontsize=14, va="top",
    bbox=dict(facecolor="white", alpha=0.7),
)
ax_scatter.set_title("Predicted vs ECMWF T2max (All 10 Factors)", fontsize=16, pad=15)
ax_scatter.set_xlabel("ECMWF T2max (°C)", fontsize=14)
ax_scatter.set_ylabel("Predicted T2max (°C)", fontsize=14)
ax_scatter.text(-0.05, 1.05, "a)", transform=ax_scatter.transAxes, fontsize=20, weight="bold")

# LMG 条形图（按重要性降序排列）
sorted_lmg = lmg_results.sort_values("normRelaImpt", ascending=False).reset_index(drop=True)
display_labels = [LABEL_MAP.get(d, d) for d in sorted_lmg["driver"]]
bars = ax_bar.bar(display_labels, sorted_lmg["normRelaImpt"], color="royalblue", alpha=0.7)
ax_bar.set_title("LMG Relative Importance (All 10 Factors)", fontsize=16, pad=15)
ax_bar.set_ylabel("Normalized Relative Importance (%)", fontsize=14)
ax_bar.set_xticks(range(len(display_labels)))
ax_bar.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
for bar, val in zip(bars, sorted_lmg["normRelaImpt"]):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=11,
    )
ax_bar.text(-0.05, 1.05, "b)", transform=ax_bar.transAxes, fontsize=20, weight="bold")

plt.tight_layout()
out_scatter = f"/data1/huangy/fig6/NC/1.s2s.fig_scatter_bar_LMG_{start_date}.png"
plt.savefig(out_scatter, dpi=300, bbox_inches="tight")
plt.show()
print(f"散点图 + 条形图已保存至: {out_scatter}")

# =============================================================================
# 7. 逐日时间序列预测
# =============================================================================
ts_y = extract_daily_timeseries(daily_max_tmx)
days_len = len(ts_y)
ts_z500_refo_mean = extract_daily_timeseries(daily_max_z_refo_NCHN, is_reforecast=True)

ts_data_dict = {
    "NCHN_tp": extract_daily_timeseries(daily_max_NCHNtp),
    "sm_avg": extract_daily_timeseries(daily_max_sm),
    "WNPSH": extract_daily_timeseries(daily_max_z_WNPSH),
    "SSR_Avg": extract_daily_timeseries(daily_mean_ssr),
    "SHF_Avg": extract_daily_timeseries(daily_mean_sshf),
    "z500_anom_NCHN": extract_daily_timeseries(daily_max_z_NCHN) - ts_z500_refo_mean,
    "Stability_NCHN": extract_daily_timeseries(daily_max_stability),
    "NCVI": np.repeat(ncvi_arr.mean(), days_len),
    "ISM": np.repeat(ism_arr.mean(), days_len),
    "SST_Grad": np.repeat(sst_grad_arr.mean(), days_len),
}
ts_X_final = pd.DataFrame(ts_data_dict)[FACTOR_COLS]
ts_X_scaled_arr = scaler.transform(ts_X_final)
ts_X_scaled = pd.DataFrame(ts_X_scaled_arr, columns=FACTOR_COLS)
ts_X_scaled_ols = sm_api.add_constant(ts_X_scaled, has_constant="add")
ts_y_pred = model_full.predict(ts_X_scaled_ols)

# --- 逐成员时间序列 ---
ts_tmx_members = extract_daily_timeseries_per_member(daily_max_tmx)    # (50, n_days)
ts_z500_members = extract_daily_timeseries_per_member(daily_max_z_NCHN)  # (50, n_days)
n_members = ts_tmx_members.shape[0]

ts_member_preds = np.full((n_members, days_len), np.nan)
per_member_feature_arrays = {
    "NCHN_tp": extract_daily_timeseries_per_member(daily_max_NCHNtp),
    "sm_avg": extract_daily_timeseries_per_member(daily_max_sm),
    "WNPSH": extract_daily_timeseries_per_member(daily_max_z_WNPSH),
    "SSR_Avg": extract_daily_timeseries_per_member(daily_mean_ssr),
    "SHF_Avg": extract_daily_timeseries_per_member(daily_mean_sshf),
    "z500_anom_NCHN": ts_z500_members - ts_z500_refo_mean,
    "Stability_NCHN": extract_daily_timeseries_per_member(daily_max_stability),
}
for m in range(n_members):
    member_dict = {key: arr[m] for key, arr in per_member_feature_arrays.items()}
    member_dict["NCVI"] = np.repeat(ncvi_arr[m], days_len)
    member_dict["ISM"] = np.repeat(ism_arr[m], days_len)
    member_dict["SST_Grad"] = np.repeat(sst_grad_arr[m], days_len)
    ts_X_m = pd.DataFrame(member_dict)[FACTOR_COLS]
    ts_X_scaled_m = scaler.transform(ts_X_m)
    ts_X_scaled_ols_m = sm_api.add_constant(
        pd.DataFrame(ts_X_scaled_m, columns=FACTOR_COLS), has_constant="add"
    )
    ts_member_preds[m] = model_full.predict(ts_X_scaled_ols_m).values

# =============================================================================
# 8. 图2：时间序列图（集合展布）
# =============================================================================
fig_ts, ax_ts = plt.subplots(figsize=(14, 7))
plot_dates = pd.date_range(start=study_start, end=study_end)

for m in range(n_members):
    label = "Ensemble Members (ECMWF)" if m == 0 else None
    ax_ts.plot(plot_dates, ts_tmx_members[m], color="lightcoral", alpha=0.25, lw=0.8, label=label)

for m in range(n_members):
    label = "Ensemble Members (Predicted)" if m == 0 else None
    ax_ts.plot(plot_dates, ts_member_preds[m], color="lightskyblue", alpha=0.25, lw=0.8, label=label)

ax_ts.plot(plot_dates, ts_y, marker="o", color="crimson", lw=2.5, zorder=5,
           label="ECMWF $T_{max}$ (Ensemble Mean)")
ax_ts.plot(plot_dates, era5_obs_ts, marker="^", color="black", lw=2.5, zorder=6,
           label="ERA5 Observation")
ax_ts.plot(plot_dates, ts_y_pred.values, marker="s", linestyle="--", color="dodgerblue",
           lw=2.5, zorder=5, label="Predicted $T_{max}$ (Ensemble Mean)")

ax_ts.set_title("Time Series Evolution (All 10 Factors, LMG Model)", fontsize=16, pad=15)
ax_ts.set_ylabel("T2max (°C)", fontsize=14)

rmse = np.sqrt(np.mean((ts_y - ts_y_pred.values) ** 2))
ax_ts.text(
    0.02, 0.92, f"RMSE: {rmse:.2f} °C",
    transform=ax_ts.transAxes, fontsize=16,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig_ts.autofmt_xdate(rotation=30)
ax_ts.legend(fontsize=12)
ax_ts.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
out_ts = f"/data1/huangy/fig6/NC/1.s2s.fig_ts_LMG_{start_date}.png"
plt.savefig(out_ts, dpi=300, bbox_inches="tight")
plt.show()
print(f"时间序列图已保存至: {out_ts}")

# =============================================================================
# 9. 图3：特征相关矩阵热力图
# =============================================================================
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
features_with_target = pd.concat(
    [pd.DataFrame({"T2max": y_final}), X_df.reset_index(drop=True)], axis=1
)
corr_matrix = features_with_target.corr()
clean_labels = [LABEL_MAP.get(col, col).replace("\n", " ") for col in corr_matrix.columns]
corr_matrix.columns = clean_labels
corr_matrix.index = clean_labels

sns.heatmap(
    corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1,
    square=True, ax=ax_corr, cbar_kws={"shrink": 0.8}, annot_kws={"size": 10},
)
ax_corr.set_title("Feature and T2max Correlation Matrix (All 10 Factors)", fontsize=16, pad=20)
plt.tight_layout()
out_corr = f"/data1/huangy/fig6/NC/1.s2s.fig_corr_LMG_{start_date}.png"
plt.savefig(out_corr, dpi=300, bbox_inches="tight")
plt.show()
print(f"相关矩阵图已保存至: {out_corr}")

swanlab.finish()
print("\nLMG 相对重要性分析全部完成！")
