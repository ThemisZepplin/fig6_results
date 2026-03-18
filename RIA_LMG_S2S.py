"""
标准 LMG / Shapley Relative Importance 分析脚本 (S2S 版)

用途：
  对12个前兆因子 [NCHN_tp, sm_avg, WNPSH, SSR_Avg, SHF_Avg,
  z500_anom_NCHN, MSEstar_max_NCHN, MSEstar500_NCHN, Barrier_NCHN,
  NCVI, ISM, SST_Grad]
  使用标准 LMG 方法计算相对重要性：
    - 枚举(或蒙特卡洛采样)所有可能的变量进入顺序
    - 计算每个变量在该顺序下带来的增量 R²
    - 对所有顺序取平均，得到该变量的 LMG 相对重要性

热力因子定义参考 Li & Tamarin-Brodsky (Sci Adv, 2026) MSE 框架：
  - MSEs        = cp*T2m + Lv*q2m + Phi_s  (近地面 MSE，使用 2 m 变量)
  - MSEstar500  = cp*T500 + Lv*qsat(T500,500hPa) + g*z500
  - MSEstar_max = at or above LCL、below 300 hPa 的下对流层最大 MSE*
  - Barrier     = MSEstar_max - MSEs  (残余能量障碍)
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
    "MSEstar_max_NCHN",   # Li & Tamarin-Brodsky (2026): MSE*_max in lower free troposphere
    "MSEstar500_NCHN",    # Li & Tamarin-Brodsky (2026): MSE*_500 (CAPE scaling reference)
    "Barrier_NCHN",       # Li & Tamarin-Brodsky (2026): residual energy barrier = MSE*_max - MSEs
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
    "SHF_Avg": "Sensible\nHeat",
    "z500_anom_NCHN": "NCHN\nZ500",
    "MSEstar_max_NCHN": "MSE*max\nNCHN",
    "MSEstar500_NCHN": "MSE*500\nNCHN",
    "Barrier_NCHN": "Barrier\nNCHN",
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
        "description": "标准 LMG/Shapley 相对重要性分析 (S2S 版, MSE 热力因子框架)",
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

# 1g. MSE 热力因子框架（Li & Tamarin-Brodsky, Sci Adv 2026）
# -----------------------------------------------------------------------
# 物理常数
cp, Lv, g, Rd = 1005.0, 2.5e6, 9.81, 287.05  # cp J/kg/K, Lv J/kg, g m/s², Rd J/kg/K

# 压力层列表（可扩展）；MSEstar_max 的搜索范围为 at or above LCL、below 300 hPa
# 即满足：300 < level_hPa <= LCL_hPa
MSE_LEVELS = [1000, 925, 850, 700, 500, 300]  # hPa


def _qsat(T_K: xr.DataArray, p_hPa: float) -> xr.DataArray:
    """
    计算饱和比湿 qsat(T, p)，使用 Bolton (1980) 近似。

    对应论文 MSE* = cp*T + Lv*qsat(T, p) + g*z（Li & Tamarin-Brodsky 2026）。
    注意：此函数只用温度和压力，不使用实际比湿 q，避免混淆。

    参数
    ----
    T_K   : 温度 (K)
    p_hPa : 压力 (hPa, 标量)
    """
    t_c = T_K - 273.15
    es_Pa = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5)) * 100.0  # 饱和水汽压 (Pa)
    p_Pa = p_hPa * 100.0
    return (0.622 * es_Pa) / (p_Pa - 0.378 * es_Pa)


def _lcl_pressure(
    T2m_K: xr.DataArray,
    Td2m_K: xr.DataArray,
    sp_Pa: xr.DataArray,
) -> xr.DataArray:
    """
    Bolton (1980) 近似计算 LCL 压力（hPa）。

    公式（Bolton 1980, eq. 22 & 21，温度单位 K）：
      LCL_T = 56 + 1 / (1/(Td - 56) + ln(T/Td)/800)
      LCL_P = sp * (LCL_T / T)^(cp/Rd)

    参数
    ----
    T2m_K  : 2 m 温度 (K)
    Td2m_K : 2 m 露点温度 (K)
    sp_Pa  : 地面气压 (Pa)

    返回
    ----
    LCL 压力（hPa）
    """
    lcl_T = 56.0 + 1.0 / (1.0 / (Td2m_K - 56.0) + np.log(T2m_K / Td2m_K) / 800.0)
    _lcl_p_Pa = sp_Pa * (lcl_T / T2m_K) ** (cp / Rd)  # intermediate result in Pa
    return _lcl_p_Pa / 100.0  # → hPa，与 MSE_LEVELS 单位一致


# --- 载入 2 m 地表变量（用于 MSEs 和 LCL 计算）---
# MSEs = cp*T2m + Lv*q2m + Phi_s
#   q2m ≈ qsat(Td2m, sp)：近地面比湿用露点近似（在露点时空气对水汽饱和）
#   Phi_s = surface geopotential = g*z_s（若 zs 是几何高度 m），
#           或 Phi_s = zs（若 zs 已是 geopotential m^2/s^2）
#
# TODO: 确认以下文件路径，ECMWF GRIB 参数编号对应关系：
#   167 = 2m temperature (t2m), 单位 K
#   168 = 2m dew-point temperature (d2m), 单位 K
#   134 = surface pressure (sp), 单位 Pa
#   129 = surface geopotential (z), 单位 m^2 s^-2（即 Phi_s = gz_s）

t2m_sfc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_sfc_167_2023-06.grib", engine="cfgrib"
)
# TODO: 确认变量名；ECMWF GRIB 2m temperature 通常为 "t2m"
t2m_sfc = t2m_sfc_ds["t2m"].loc[:, start_date, :, 44:35, 114:119]  # K

td2m_sfc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_sfc_168_2023-06.grib", engine="cfgrib"
)
# TODO: 确认变量名；ECMWF GRIB 2m dewpoint 通常为 "d2m"
td2m_sfc = td2m_sfc_ds["d2m"].loc[:, start_date, :, 44:35, 114:119]  # K

sp_sfc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_sfc_134_2023-06.grib", engine="cfgrib"
)
# TODO: 确认变量名；ECMWF GRIB surface pressure 通常为 "sp"
sp_sfc = sp_sfc_ds["sp"].loc[:, start_date, :, 44:35, 114:119]  # Pa

# 地面位势（surface geopotential, param 129）
# ECMWF param 129 "z" 的单位为 m^2 s^-2（即已含 g，Phi_s = g*z_s）
# 因此 MSEs 中直接加 zs_phi，无需再乘 g
# 若实际数据单位为几何高度(m)，请将下方 mse_s 改为 cp*t2m_sfc + Lv*q2m + g*zs_phi
# TODO: 确认文件路径和变量名（param 129, 通常变量名 "z" 或 "zs"）
# TODO: 确认是否为随时间变化的场或静态场（orography 通常为静态）
zs_phi_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_sfc_129_2023-06.grib", engine="cfgrib"
)
# NOTE: ECMWF param 129 "z" is geopotential (Phi = g*z_geometric), units m^2/s^2.
#       We add it directly to MSE without multiplying by g again.
zs_phi = zs_phi_ds["z"].loc[:, start_date, :, 44:35, 114:119]  # m^2 s^-2

# --- MSEs（近地面静态能量）---
# 对应论文：MSEs = cp*T2m + Lv*q2m + Phi_s  （Li & Tamarin-Brodsky 2026）
# 注：q2m ≈ qsat(Td2m, sp)，利用 2 m 露点温度代替实际比湿
q2m_approx = _qsat(td2m_sfc, sp_sfc / 100.0)  # sp 由 Pa 转为 hPa
mse_s = cp * t2m_sfc + Lv * q2m_approx + zs_phi  # J/kg
daily_mean_mse_s = mse_s.resample(step="D").mean()

# --- 各压力层 MSE*（仅用 T 和 gh，不使用实际 q）---
# NOTE: ECMWF GRIB 变量 "gh"（param 156）为位势高度（geopotential height, gpm ≈ m）。
#       MSE* = cp*T + Lv*qsat(T,p) + g*gh，其中 g*gh 将几何高度转换为位势能 (m^2/s^2)。
#       若文件中的 "gh" 实为 geopotential (m^2/s^2)，请将 g * z_lev 改为 z_lev。
# TODO: 用实际数据加载后打印 gh 量级（~几千 m 则为几何高度；~几万则为 geopotential）
# TODO: 确认 300 hPa 层数据文件是否存在（路径：ecmf_pl300_130_*.grib, ecmf_pl300_156_*.grib）
# NOTE: 文件名中的 "2022-08" 与地表文件 "2023-06" 不同，这与原代码保持一致；
#       该命名约定可能源于再预报（reforecast）档案，请确认实际数据来源。
mse_star_by_level: dict = {}
for _lev in MSE_LEVELS:
    _t_lev = xr.open_dataset(
        f"/data1/huangy/fig6/NC/MSE/ecmf_pl{_lev}_130_2022-08.grib", engine="cfgrib"
    )["t"].loc[:, start_date, :, 44:35, 114:119]  # K
    _z_lev = xr.open_dataset(
        f"/data1/huangy/fig6/NC/MSE/ecmf_pl{_lev}_156_2022-08.grib", engine="cfgrib"
    )["gh"].loc[:, start_date, :, 44:35, 114:119]  # m (geopotential height, param 156)
    _qs_lev = _qsat(_t_lev, float(_lev))
    # MSE* = cp*T + Lv*qsat(T,p) + g*z  （z 为几何高度 m，g*z 转为能量项）
    mse_star_by_level[_lev] = cp * _t_lev + Lv * _qs_lev + g * _z_lev  # J/kg

# --- MSEstar500 ---
# 对应论文 CAPE scaling：CAPE ∝ MSEs - MSE*500  （Li & Tamarin-Brodsky 2026）
mse_star_500 = mse_star_by_level[500]
daily_max_mse_star500 = mse_star_500.resample(step="D").max()

# --- LCL 压力（Bolton 1980）---
lcl_p_da = _lcl_pressure(t2m_sfc, td2m_sfc, sp_sfc)  # hPa

# --- MSEstar_max ---
# 对应论文 "maximum MSE* in the lower free troposphere"（Li & Tamarin-Brodsky 2026）
# 搜索范围：at or above LCL（level_hPa <= LCL_hPa）且 below 300 hPa（level_hPa > 300）
# 即：300 < level_hPa <= LCL_hPa
_mse_star_list_new = [
    mse_star_by_level[_lev].assign_coords(level=_lev) for _lev in MSE_LEVELS
]
mse_star_all = xr.concat(_mse_star_list_new, dim="level")  # dim=(level, number, step, lat, lon)

# level_coord 广播到 (level, number, step, lat, lon)，lcl_p_da 广播到 (level, ...)
_level_coord = mse_star_all["level"]  # DataArray, dim=(level,)
# 有效层掩码：高于 LCL（低压）且低于 300 hPa（高压侧边界）
_valid_mask = (_level_coord > 300) & (_level_coord <= lcl_p_da)
mse_star_max_da = mse_star_all.where(_valid_mask).max(dim="level")  # J/kg
daily_max_mse_star_max = mse_star_max_da.resample(step="D").max()

# --- Barrier = MSEstar_max - MSEs ---
# 对应论文"residual energy barrier"（Li & Tamarin-Brodsky 2026）
# Barrier > 0：上层 MSE* 高于近地面 MSE，存在阻碍深对流的能量障碍
# 注：Barrier = MSEstar_max - MSEs（不要写反）
barrier_da = mse_star_max_da - mse_s  # J/kg
daily_max_barrier = barrier_da.resample(step="D").max()

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
print("开始提取特征，建立单一模型（所有12个因子）...")
print("=" * 60)

# --- 提取各因子（50个成员的集合均值，代表研究时段均值）---
temp_mean_val = extract_period_mean(daily_max_tmx)
sm_avg_val = extract_period_mean(daily_max_sm)
NCHN_tp_val = extract_period_mean(daily_max_NCHNtp)
WNPSH_avg_val = extract_period_mean(daily_max_z_WNPSH)
SSR_Avg_val = extract_period_mean(daily_mean_ssr)
SHF_Avg_val = extract_period_mean(daily_mean_sshf)
z500_2023_NCHN_val = extract_period_mean(daily_max_z_NCHN)
z500_mean_NCHN_val = extract_period_mean(daily_max_z_refo_NCHN, is_reforecast=True)
z500_anom_NCHN_val = z500_2023_NCHN_val - z500_mean_NCHN_val
# MSE 热力因子（Li & Tamarin-Brodsky 2026）
MSEstar_max_NCHN_val = extract_period_mean(daily_max_mse_star_max)
MSEstar500_NCHN_val = extract_period_mean(daily_max_mse_star500)
Barrier_NCHN_val = extract_period_mean(daily_max_barrier)

data_dict = {
    "temp_mean": temp_mean_val,
    "NCHN_tp": NCHN_tp_val,
    "sm_avg": sm_avg_val,
    "WNPSH": WNPSH_avg_val,
    "SSR_Avg": SSR_Avg_val,
    "SHF_Avg": SHF_Avg_val,
    "z500_anom_NCHN": z500_anom_NCHN_val,
    "MSEstar_max_NCHN": MSEstar_max_NCHN_val,
    "MSEstar500_NCHN": MSEstar500_NCHN_val,
    "Barrier_NCHN": Barrier_NCHN_val,
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
ax_scatter.set_title("Predicted vs ECMWF T2max (All 12 Factors)", fontsize=16, pad=15)
ax_scatter.set_xlabel("ECMWF T2max (°C)", fontsize=14)
ax_scatter.set_ylabel("Predicted T2max (°C)", fontsize=14)
ax_scatter.text(-0.05, 1.05, "a)", transform=ax_scatter.transAxes, fontsize=20, weight="bold")

# LMG 条形图（按重要性降序排列）
sorted_lmg = lmg_results.sort_values("normRelaImpt", ascending=False).reset_index(drop=True)
display_labels = [LABEL_MAP.get(d, d) for d in sorted_lmg["driver"]]
bars = ax_bar.bar(display_labels, sorted_lmg["normRelaImpt"], color="royalblue", alpha=0.7)
ax_bar.set_title("LMG Relative Importance (All 12 Factors)", fontsize=16, pad=15)
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
    # MSE 热力因子逐日时间序列（Li & Tamarin-Brodsky 2026）
    "MSEstar_max_NCHN": extract_daily_timeseries(daily_max_mse_star_max),
    "MSEstar500_NCHN": extract_daily_timeseries(daily_max_mse_star500),
    "Barrier_NCHN": extract_daily_timeseries(daily_max_barrier),
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
    # MSE 热力因子逐成员时间序列（Li & Tamarin-Brodsky 2026）
    "MSEstar_max_NCHN": extract_daily_timeseries_per_member(daily_max_mse_star_max),
    "MSEstar500_NCHN": extract_daily_timeseries_per_member(daily_max_mse_star500),
    "Barrier_NCHN": extract_daily_timeseries_per_member(daily_max_barrier),
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

ax_ts.set_title("Time Series Evolution (All 12 Factors, LMG Model)", fontsize=16, pad=15)
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
ax_corr.set_title("Feature and T2max Correlation Matrix (All 12 Factors)", fontsize=16, pad=20)
plt.tight_layout()
out_corr = f"/data1/huangy/fig6/NC/1.s2s.fig_corr_LMG_{start_date}.png"
plt.savefig(out_corr, dpi=300, bbox_inches="tight")
plt.show()
print(f"相关矩阵图已保存至: {out_corr}")

swanlab.finish()
print("\nLMG 相对重要性分析全部完成！")
