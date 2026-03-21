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

import os
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
_N_STUDY_DAYS = (study_end - study_start).days + 1  # 研究时段总天数（含首尾）= 11

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
    "sm_avg": "SM",
    "WNPSH": "WNPSH",
    "SSR_Avg": "SSR",
    "SHF_Avg": "Sensible\nHeat",
    "z500_anom_NCHN": "NCHN\nZ500",
    "MSEstar_max_NCHN": "MSE*max\nNCHN",
    "MSEstar500_NCHN": "MSE*500\nNCHN",
    "Barrier_NCHN": "Barrier\nNCHN",
    "NCVI": "NCVI",
    "ISM": "ISM",
    "SST_Grad": "SST Grad",
}

# 所有图表统一输出目录
OUT_DIR = "/data1/huangy/fig6/NC/v3"

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
# 数据验证工具函数
# =============================================================================
def _validate_dataset(
    ds: xr.Dataset,
    ds_name: str,
    skip_init_check: bool = False,
    skip_coverage_check: bool = False,
) -> None:
    """验证数据集的起报时间和预报时段覆盖范围。

    检查项：
      1. 起报时间坐标（time / forecast_reference_time / valid_time / date）中需包含 start_date。
         月度 GRIB 文件含多个起报时刻（如 2023-06-01…2023-06-30），仅需 start_date 在其中即可。
         若 skip_init_check=True（如多年再预报气候态数据），跳过此项。
      2. 通过 step 坐标推算的有效时刻（start_time + step）应覆盖
         study_start（2023-06-14）至 study_end（2023-06-24）。
         若 skip_coverage_check=True（如仅取前 3 天的前兆因子数据），跳过此项。

    若验证失败则发出 UserWarning；通过则打印确认信息。
    """
    # --- 1. 起报时间验证 ---
    if skip_init_check:
        print(f"  ─ [{ds_name}] 跳过起报时间验证（气候态/多年再预报数据）")
    else:
        init_time_found = False
        for coord_name in ("time", "forecast_reference_time", "valid_time", "date"):
            if coord_name in ds.coords:
                raw = ds.coords[coord_name].values
                # 支持标量和数组坐标：搜索整个时间数组，而非仅取首值
                times = pd.to_datetime(raw.flat if np.ndim(raw) > 0 else [raw])
                expected = pd.Timestamp(start_date)
                date_set = {t.date() for t in times}
                if expected.date() not in date_set:
                    warnings.warn(
                        f"[{ds_name}] 起报时间验证失败：期望 {start_date}，"
                        f"坐标 '{coord_name}' 未包含该日期"
                        f"（范围 {times.min().date()} ~ {times.max().date()}）",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    print(
                        f"  ✓ [{ds_name}] 起报时间验证通过："
                        f"{expected.date()} 存在于 '{coord_name}' 坐标中"
                    )
                init_time_found = True
                break
        if not init_time_found:
            warnings.warn(
                f"[{ds_name}] 未找到起报时间坐标（time / forecast_reference_time 等），跳过起报时间验证。",
                UserWarning,
                stacklevel=2,
            )

    # --- 2. 预报时段覆盖验证 ---
    if skip_coverage_check:
        print(f"  ─ [{ds_name}] 跳过预报时段覆盖验证（前兆因子数据，仅需前 3 步）")
    elif "step" in ds.coords:
        step_vals = ds.coords["step"].values
        valid_dates = pd.to_datetime(
            [start_time + pd.Timedelta(s) for s in step_vals]
        )
        if not (valid_dates.min() <= study_start and valid_dates.max() >= study_end):
            warnings.warn(
                f"[{ds_name}] 预报时段覆盖不足：研究时段 "
                f"{study_start.date()} ~ {study_end.date()}，"
                f"数据有效时刻范围 {valid_dates.min().date()} ~ {valid_dates.max().date()}",
                UserWarning,
                stacklevel=2,
            )
        else:
            print(
                f"  ✓ [{ds_name}] 预报时段验证通过："
                f"覆盖 {study_start.date()} ~ {study_end.date()}"
            )


# =============================================================================
# 1. 加载气象数据
# =============================================================================
print("正在加载基础变量数据 (S2S)...")

# -----------------------------------------------------------------------
# 数据步长类型说明（适用于 1a–1f 所有变量）：
#
# 【瞬时/累计数据】（step = 0h, 24h, 48h, ... 整点瞬时值）：
#   T2m (mx2t6), TP (tp), Z500/Z850 (gh), SSR (ssr), SSHF (sshf)
#   经 resample(step="D") 后，step 坐标变为 0d, 1d, 2d, ...
#     valid_time[i] = start_time + i_days
#     研究时段 study_start(2023-06-14) = start_time + 2d → step=2d
#             study_end  (2023-06-24) = start_time + 12d → step=12d
#
# 【日均数据】（step = 24h, 48h, ... 表示 0-24h、24-48h 窗口均值）：
#   SM20 (sm20), 以及 MSE 用 2t/2d
#   经 resample(step="D") 后，step 坐标从 1d 开始（因首个日均值在 step=24h）：
#     valid_time[i] = start_time + (i+1)_days（i 从 0 开始）
#     研究时段首步：step=2d（对应窗口 24-48h，覆盖 2023-06-13~14）
#
# 提取函数（extract_period_mean 等）统一使用：
#   dates[i] = start_time + pd.Timedelta(step_vals[i])
# 自动适配两种步长类型，无需手动区分。
# -----------------------------------------------------------------------

# 1a. T2m（日最高气温）
# 数据类型：瞬时（mx2t6，6 小时滚动最大值），step = 6h, 12h, 18h, 24h, ...
# resample(step="D").max() → step 坐标 = 0d, 1d, 2d, ...（每日最高气温）
# 研究时段对应 step 2d（2023-06-14）~ 12d（2023-06-24）
tmxfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_mx2t6_2023-06.grb"
tmx_ds = xr.open_dataset(tmxfile, engine="cfgrib")
_validate_dataset(tmx_ds, "mx2t6 (tmx)")
tmx = tmx_ds["mx2t6"].loc[:, start_date, :, 44:35, 114:119] - 273.16
daily_max_tmx = tmx.resample(step="D").max()

# 1b. SM20（土壤湿度）
# 数据类型：日均（step = 24h, 48h, ...，每步为 24h 窗口均值）
# resample(step="D").max() → step 坐标从 1d 开始（首个日均步在 24h bin 内）
# 研究时段对应 step 2d（2023-06-14）~ 12d（2023-06-24）
smfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_sm2023-06.grb"
sm_ds = xr.open_dataset(smfile, engine="cfgrib")
_validate_dataset(sm_ds, "sm20 (soil moisture)")
sm = sm_ds["sm20"].loc[:, start_date, :, 44:35, 114:119]
daily_max_sm = sm.resample(step="D").max()
# 修正：sm20 的 step 坐标标在 24h 窗口右边界，向左平移 1 天对齐物理有效时刻
daily_max_sm = daily_max_sm.assign_coords(
    step=daily_max_sm["step"] - pd.Timedelta(days=1)
)

# 1c. TP（降水）
# 数据类型：累计量（step = 0h, 24h, 48h ... 自起报时刻的累计降水，Pa）
# resample(step="D").max() 取每日最大累计值，.diff() 差分得日降水量
# diff 后 step[0] 对应 NaN；研究时段起始 step 2d 对应 2023-06-14
tpfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_tp2023-06.grb"
tp_ds = xr.open_dataset(tpfile, engine="cfgrib")
_validate_dataset(tp_ds, "tp (NCHN precipitation)")
NCHN_tp_data = tp_ds["tp"].loc[:, start_date, :, 44:35, 114:119]
daily_max_NCHNtp = NCHN_tp_data.resample(step="D").max().diff(dim="step")

# 1d. Z500 & WNPSH
# 数据类型：瞬时（gh，位势高度，step = 0h, 24h, 48h, ...）
# resample(step="D").max() → step 坐标 = 0d, 1d, 2d, ...
# 研究时段对应 step 2d（2023-06-14）~ 12d（2023-06-24）
zfile = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_pl_zqt2023-06.grb"
z_ds = xr.open_dataset(zfile, engine="cfgrib")
_validate_dataset(z_ds, "gh z500/z850")
reforecastfile = f"/data1/huangy/fig6/NC/pl_z03_22/merged_{start_date}.nc"
reforecast = xr.open_dataset(reforecastfile)
_validate_dataset(reforecast, "z500 reforecast (climatology)", skip_init_check=True)

z_NCHN = z_ds["gh"].loc[:, start_date, :, 500, 44:35, 114:119]
daily_max_z_NCHN = z_NCHN.resample(step="D").max()
z_refo_NCHN = reforecast["gh"].loc[:, :, :, 500, 44:35, 114:119]
daily_max_z_refo_NCHN = z_refo_NCHN.resample(step="D").max()

z_WNPSH = z_ds["gh"].loc[:, start_date, :, 850, 25:15, 115:150]
daily_max_z_WNPSH = z_WNPSH.resample(step="D").max()

# 1e. SSR（地面短波辐射）
# 数据类型：瞬时/累计（ssr，step = 0h, 24h, 48h, ...）
# resample(step="D").mean() → 每日所有瞬时值的均值；step 坐标 = 0d, 1d, 2d, ...
# 若 ssr 仅有每日一个瞬时值，均值等价于该值
# 研究时段对应 step 2d（2023-06-14）~ 12d（2023-06-24）
ssr_file = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_radiation2023-06.grb"
ssr_ds = xr.open_dataset(ssr_file, engine="cfgrib")
_validate_dataset(ssr_ds, "ssr (SSR_Avg)")
ssr = ssr_ds["ssr"].loc[:, start_date, :, 44:35, 114:119]
daily_mean_ssr = ssr.resample(step="D").mean()

# 1f. SSHF（感热通量）
# 数据类型：瞬时/累计（sshf，step = 0h, 24h, 48h, ...）
# resample(step="D").mean() → 每日均值；step 坐标 = 0d, 1d, 2d, ...
# 研究时段对应 step 2d（2023-06-14）~ 12d（2023-06-24）
sshf_file = "/data1/huangy/fig6/NC/2023-06/ecmf_rel_pf_sfc_hflux2023-06.grb"
sshf_ds = xr.open_dataset(sshf_file, engine="cfgrib")
_validate_dataset(sshf_ds, "sshf (SHF_Avg)")
sshf = sshf_ds["sshf"].loc[:, start_date, :, 44:35, 114:119]
daily_mean_sshf = sshf.resample(step="D").mean()

# 1g. MSE 热力因子框架（Li & Tamarin-Brodsky, Sci Adv 2026）
# -----------------------------------------------------------------------
# 数据约束说明（S2S ECMWF perturbed forecast 数据现实情况）：
#   - 2t (param 167)、2d (param 168)：daily-averaged step（每步代表 24h 均值，如 0-24h、24-48h）
#   - sp (param 134)：instantaneous step（瞬时量，如 0h、24h、48h）
#   - 地面位势（orography）：无逐成员预报，使用控制成员(cf)的静态 orography
#   因此，以下 MSEs_2m 为 daily-mean 2m MSE proxy，
#   非严格意义上的逐时峰值 MSEs（但与 Li & Tamarin-Brodsky 2026 原始定义最接近）。
#
# 物理常数
cp, Lv, g, Rd = 1005.0, 2.5e6, 9.81, 287.05  # J/kg/K, J/kg, m/s², J/kg/K

# 压力层列表（可扩展）；MSEstar_max 的搜索范围为 at or above LCL、below 300 hPa
# 即满足：300 < level_hPa <= LCL_hPa
MSE_LEVELS = [1000, 925, 850, 700, 500, 300]  # hPa

# orography 单位判断阈值（m^2/s^2）：
#   典型地面 geopotential (m^2/s^2) 量级 ≈ 0–30000（海拔 3000 m ≈ 29430 m^2/s^2）
#   典型几何高度 (m) 量级 ≈ 0–3000
#   阈值 9000 稳健区分两者；若区域海拔极高（>918 m 均值）或数据非标准，请手动覆盖
_GEOPOTENTIAL_THRESHOLD = 9000.0  # m^2 s^-2


def _qsat(T_K: xr.DataArray, p_hPa) -> xr.DataArray:
    """
    计算饱和比湿 qsat(T, p)，使用 Bolton (1980) 近似。

    对应论文 MSE* = cp*T + Lv*qsat(T, p) + g*z（Li & Tamarin-Brodsky 2026）。
    注意：此函数只用温度和压力，不使用实际比湿 q，避免混淆。

    参数
    ----
    T_K   : 温度 (K)；xr.DataArray 或 ndarray
    p_hPa : 压力 (hPa)；标量 float 或与 T_K 广播兼容的 DataArray（如 daily_mean_sp/100）
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
    _lcl_p_Pa = sp_Pa * (lcl_T / T2m_K) ** (cp / Rd)  # 中间量，单位 Pa
    return _lcl_p_Pa / 100.0  # 转为 hPa，与 MSE_LEVELS 单位一致


def _daily_mean_sp(sp_inst: xr.DataArray) -> xr.DataArray:
    """
    从 instantaneous sp（瞬时气压）构造近似日均值，与 daily-mean t2m/d2m 的 step 坐标对齐。

    S2S 中 sp 为瞬时量（step = 0h, 24h, 48h, ...）。
    对于第 N 日（日窗 [(N-1)*24, N*24] h），用前后边界瞬时值平均作为日均近似：
      sp_day[N] ≈ (sp[(N-1)*24h] + sp[N*24h]) / 2
    若前边界不存在（如第 1 日 step=0h 缺失），则回退到仅使用日末时刻 sp。

    注意：此为 daily-mean MSEs proxy 的近似气压分量，非精确日均值。

    参数
    ----
    sp_inst : instantaneous surface pressure DataArray，step 维度为 timedelta，单位 Pa

    返回
    ----
    DataArray，step 坐标为日末时刻 timedelta（与 daily-mean t2m 步长一致），单位 Pa
    """
    # 将 step timedelta 坐标转为小时整数，便于对齐逻辑
    step_hours = np.array([
        int(pd.Timedelta(s).total_seconds() / 3600)
        for s in sp_inst.step.values
    ])

    # 找出所有 24h 整数倍时刻作为日窗边界（0h, 24h, 48h, ...）
    daily_edges = sorted({int(h) for h in step_hours if h % 24 == 0})

    result_list, result_step_hours = [], []
    for i in range(len(daily_edges) - 1):
        h_start, h_end = daily_edges[i], daily_edges[i + 1]
        idx0 = np.where(step_hours == h_start)[0]
        idx1 = np.where(step_hours == h_end)[0]
        if idx0.size and idx1.size:
            # 优先：前后边界平均（最佳日均近似）
            sp_day = (
                sp_inst.isel(step=int(idx0[0])) + sp_inst.isel(step=int(idx1[0]))
            ) / 2.0
        elif idx1.size:
            # 回退：仅用日末时刻（若前边界不可用，见注释说明）
            sp_day = sp_inst.isel(step=int(idx1[0]))
        else:
            continue
        result_list.append(sp_day.drop_vars("step", errors="ignore"))
        result_step_hours.append(h_end)

    if not result_list:
        raise ValueError(
            "_daily_mean_sp: 无法从 sp 构造日均值，请检查 step 坐标是否包含 24h 整数倍时刻。"
        )

    # 沿 step 维拼合，赋值与 daily-mean t2m 一致的 timedelta 坐标
    result = xr.concat(result_list, dim="step")
    result = result.assign_coords(
        step=[pd.Timedelta(hours=int(h)) for h in result_step_hours]
    )
    return result  # Pa，step 坐标为 timedelta（日末时刻）


# -----------------------------------------------------------------------
# 载入 2 m 地表变量：2t（日均温）、2d（日均露点）
# -----------------------------------------------------------------------
# S2S pf 数据中，2t/2d 以 daily-averaged step 下发（每步已是 24h 均值，如 step=24h 对应 0-24h 均值）。
# TODO: 确认文件路径；ECMWF param 167 = 2m temperature (t2m), param 168 = 2m dewpoint (d2m)
# TODO: 确认变量名（cfgrib 读取后通常为 "t2m" 和 "d2m"；可运行 list(ds.data_vars) 核验）
t2m_sfc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_pf_sfc_2t_2023-06.grib", engine="cfgrib"
)
_validate_dataset(t2m_sfc_ds, "2t (t2m)")
t2m_sfc = t2m_sfc_ds["t2m"].loc[:, start_date, :, 44:35, 114:119]  # K, daily-mean steps

td2m_sfc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_pf_sfc_2td_2023-06.grib", engine="cfgrib"
)
_validate_dataset(td2m_sfc_ds, "2td (d2m)")
td2m_sfc = td2m_sfc_ds["d2m"].loc[:, start_date, :, 44:35, 114:119]  # K, daily-mean steps

# 每个 step 值本身已是日均窗口（resample 等效于原值，保留以维持代码风格一致性）
daily_mean_t2m = t2m_sfc.resample(step="D").mean()   # K, shape: (number, step, lat, lon)
# 修正：2t 的 step 坐标标在 24h 窗口右边界，向左平移 1 天对齐物理有效时刻
daily_mean_t2m = daily_mean_t2m.assign_coords(
    step=daily_mean_t2m["step"] - pd.Timedelta(days=1)
)
daily_mean_td2m = td2m_sfc.resample(step="D").mean()  # K
# 修正：2td 的 step 坐标标在 24h 窗口右边界，向左平移 1 天对齐物理有效时刻
daily_mean_td2m = daily_mean_td2m.assign_coords(
    step=daily_mean_td2m["step"] - pd.Timedelta(days=1)
)

# -----------------------------------------------------------------------
# 载入 sp（瞬时量）并对齐为日均代理
# -----------------------------------------------------------------------
# S2S pf 数据中，sp 为 instantaneous step（0h, 24h, 48h, ...），单位 Pa。
# TODO: 确认文件路径；ECMWF param 134 = surface pressure (sp)
# TODO: 确认变量名（cfgrib 读取后通常为 "sp"）
sp_inst_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_pf_sfc_sp_2023-06.grib", engine="cfgrib"
)
_validate_dataset(sp_inst_ds, "sp (surface pressure)")
sp_inst = sp_inst_ds["sp"].loc[:, start_date, :, 44:35, 114:119]  # Pa, instantaneous
# 用相邻瞬时值均值构造日均 sp proxy，step 坐标对齐到 daily_mean_t2m
daily_mean_sp = _daily_mean_sp(sp_inst)  # Pa, daily-mean proxy

# -----------------------------------------------------------------------
# 载入静态 orography（控制成员 cf，非逐成员预报）
# -----------------------------------------------------------------------
# 由于无逐成员 surface geopotential 预报，使用控制成员的静态 orography。
# xarray 广播规则：gz_s 为 (latitude, longitude) 二维场，计算 MSEs 时
# 会自动广播到 (number, step, latitude, longitude)，不会引入 silent bug。
#
# 单位判断（二选一，取决于实际数据文件）：
#   ECMWF param 129 "z"   → surface geopotential，单位 m^2 s^-2，直接使用（Phi_s = z）
#   ECMWF param 228 "orog"→ geometric height (orography)，单位 m，须乘 g（Phi_s = g*z）
# 文件含变量 "orog"（ECMWF param 228，几何高度，单位 m）；
# 若文件含时间/step等多余维，下方 ndim>2 分支会自动去除首个非空间维。
orog_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/ecmf_cf_orog_2023-06.grib", engine="cfgrib"
)
# orography 为静态场，无 step 覆盖需求；仅验证起报时间坐标（如有）。
_validate_dataset(orog_ds, "orog (static orography)")
# 实际变量名为 "orog"（cfgrib 读取 param 228 时映射为该名称）
_orog_raw = orog_ds["orog"].sel(latitude=slice(44, 35), longitude=slice(114, 119))
# 若含多余维（time 等），压缩为 (latitude, longitude)
if _orog_raw.ndim > 2:
    _extra_dims = [d for d in _orog_raw.dims if d not in ("latitude", "longitude")]
    _orog_raw = _orog_raw.isel({d: 0 for d in _extra_dims}).squeeze()

# 稳健单位检查：
#   典型地面 geopotential (m^2/s^2) 量级 ≈ 0–30000（海拔 3000 m 对应约 29430）
#   典型几何高度 (m) 量级 ≈ 0–3000
#   阈值 9000 介于两者之间，可可靠区分（如误判请手动注释覆盖）
_orog_sample = float(_orog_raw.mean().values)
if _orog_sample > _GEOPOTENTIAL_THRESHOLD:
    # 判断为 geopotential (m^2 s^-2)，直接作为势能项 Phi_s
    print(
        f"  INFO: orography 判断为 geopotential (m^2/s^2)，"
        f"区域均值 = {_orog_sample:.0f}，直接使用"
    )
    gz_s = _orog_raw  # Phi_s = z (m^2 s^-2)
else:
    # 判断为几何高度 (m)，乘以 g 转为势能 Phi_s = g*z
    print(
        f"  INFO: orography 判断为几何高度 (m)，"
        f"区域均值 = {_orog_sample:.0f} m，使用 g*z"
    )
    gz_s = g * _orog_raw  # Phi_s = g * z (m^2 s^-2)

# -----------------------------------------------------------------------
# MSEs_2m（daily-mean near-surface MSE proxy）
# -----------------------------------------------------------------------
# 对应 Li & Tamarin-Brodsky (2026): MSEs = cp*T2m + Lv*q2m + Phi_s
#   - T2m  : daily-mean 2m temperature (K)
#   - q2m  : 近地面比湿，以露点温度代替实际比湿（qsat(Td2m, sp) ≈ q2m）
#   - Phi_s: gz_s，静态 orography 势能项（见上方单位判断）
# 重要说明：因使用 daily-mean 变量和静态 orography，本量为 daily-mean 2m MSE proxy，
#           而非严格意义上的瞬时近地面 MSEs。
q2m_proxy = _qsat(daily_mean_td2m, daily_mean_sp / 100.0)  # daily_mean_sp Pa → hPa
# gz_s 为 (lat, lon) 静态场，xarray 自动广播到 (number, step, lat, lon)
mse_s_daily = cp * daily_mean_t2m + Lv * q2m_proxy + gz_s  # J/kg, daily-mean MSE proxy

# -----------------------------------------------------------------------
# 各压力层 MSE*（仅用 T 和 gh，不使用实际 q）
# -----------------------------------------------------------------------
# 对应论文：MSE*_lev = cp*T_lev + Lv*qsat(T_lev, p_lev) + g*gh_lev
# NOTE: ECMWF GRIB 变量 "gh"（param 156）为位势高度（geopotential height），单位 m（gpm）。
#       MSE* = cp*T + Lv*qsat(T,p) + g*gh，其中 g*gh 将几何高度转换为势能 (m^2/s^2)。
#       若实际文件中 "gh" 为 geopotential (m^2/s^2)，请将下方 g * _z_lev 改为 _z_lev。
# TODO: 加载后确认量级：~百至千 m → 几何高度（gpm）；~万级 → geopotential (m^2/s^2)
# TODO: 确认 300 hPa 层数据文件存在（ecmf_pl300_130_*.grib 和 ecmf_pl300_156_*.grib）
# NOTE: 压力层文件 "ecmf_pl{lev}_130/156_2023-06.grib" 与主预报使用相同年月。
mse_star_by_level: dict = {}
for _lev in MSE_LEVELS:
    _t_ds = xr.open_dataset(
        f"/data1/huangy/fig6/NC/MSE/ecmf_pl{_lev}_130_2023-06.grib", engine="cfgrib"
    )
    _validate_dataset(_t_ds, f"T@{_lev}hPa (param 130)")
    _t_lev = _t_ds["t"].loc[:, start_date, :, 44:35, 114:119]  # K
    _z_ds = xr.open_dataset(
        f"/data1/huangy/fig6/NC/MSE/ecmf_pl{_lev}_156_2023-06.grib", engine="cfgrib"
    )
    _validate_dataset(_z_ds, f"gh@{_lev}hPa (param 156)")
    _z_lev = _z_ds["gh"].loc[:, start_date, :, 44:35, 114:119]  # m（geopotential height, param 156）
    _qs_lev = _qsat(_t_lev, float(_lev))
    # MSE*_lev = cp*T + Lv*qsat(T,p) + g*gh  （gh 为几何高度 m，g*gh 为势能 m^2/s^2）
    mse_star_by_level[_lev] = cp * _t_lev + Lv * _qs_lev + g * _z_lev  # J/kg

# -----------------------------------------------------------------------
# MSEstar500（CAPE scaling 参考层）
# -----------------------------------------------------------------------
# 对应论文 CAPE scaling：CAPE ∝ MSEs - MSE*_500（Li & Tamarin-Brodsky 2026）
mse_star_500 = mse_star_by_level[500]
daily_max_mse_star500 = mse_star_500.resample(step="D").max()  # J/kg, daily max

# -----------------------------------------------------------------------
# LCL 压力（Bolton 1980）
# -----------------------------------------------------------------------
# 使用 daily-mean T2m、daily-mean Td2m、daily-mean sp proxy 计算 LCL。
# LCL 用于界定 MSEstar_max 的下边界（at or above LCL）。
lcl_p_daily = _lcl_pressure(daily_mean_t2m, daily_mean_td2m, daily_mean_sp)  # hPa

# -----------------------------------------------------------------------
# MSEstar_max（下对流层最大 MSE*）
# -----------------------------------------------------------------------
# 对应论文"maximum MSE* in the lower free troposphere"（Li & Tamarin-Brodsky 2026）
# 搜索范围：at or above LCL（level_hPa <= LCL_hPa）且 below 300 hPa（level_hPa > 300）
# 即：300 < level_hPa <= LCL_hPa
_mse_star_list_new = [
    mse_star_by_level[_lev].assign_coords(level=_lev) for _lev in MSE_LEVELS
]
# 沿 level 维拼合：shape = (level, number, step, latitude, longitude)
mse_star_all = xr.concat(_mse_star_list_new, dim="level")

# 掩码：level_coord 沿 level 维广播，lcl_p_daily 沿剩余维广播（xarray 自动对齐）
_level_coord = mse_star_all["level"]  # DataArray, dim=(level,)
_valid_mask = (_level_coord > 300) & (_level_coord <= lcl_p_daily)
# 取满足条件层的最大 MSE*；不满足条件的层设为 NaN。
# 边界情况说明：若某格点 LCL 低于 300 hPa（极干/高海拔情况，NCHN 夏季极少见），
# 则该格点的 valid_mask 全为 False，mse_star_max_da 对应位置为 NaN。
# 下游 extract_period_mean 会对 NaN 进行均值忽略，不会引发 silent 数值错误。
mse_star_max_da = mse_star_all.where(_valid_mask).max(dim="level")  # J/kg
daily_max_mse_star_max = mse_star_max_da.resample(step="D").max()   # J/kg, daily max

# -----------------------------------------------------------------------
# LCL 的 step 维可能与 pressure-level 数据的 step 维不完全相同，
# 因此用 daily_max_mse_star_max 和 daily-resample 后的 mse_s_daily 计算 Barrier，
# 确保两者 step 坐标对齐（若有偏差，xarray 会报 MergeError，便于调试）。
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Barrier_NCHN = MSEstar_max - MSEs（残余能量障碍）
# -----------------------------------------------------------------------
# 对应论文"residual energy barrier"（Li & Tamarin-Brodsky 2026）。
# Barrier > 0：上层 MSE* 高于近地面 MSE，存在阻碍深对流的能量障碍。
# 注意：Barrier = MSEstar_max - MSEs（不要写反）。
# 两侧均已重采样到 daily，step 坐标需一致；若不一致请先 .reindex_like() 对齐。
daily_max_barrier = daily_max_mse_star_max - mse_s_daily.resample(step="D").mean()  # J/kg

# =============================================================================
# 2. 加载 S2S 前兆因子（异常值模式：V' = V_rt - V̄_hc）
# =============================================================================
# 对 NCVI、ISM、SST_Grad 三个前兆因子实施异常值计算范式，消除 S2S 模式系统性漂移。
# 双轨时间提取：
#   Logic A（OLS/相对重要性/散点图）：初始前 3 天均值 → 每成员一个标量
#   Logic B（逐日时间序列图）       ：全 13 天逐日序列 → 每成员 13 步数组
print("正在加载 S2S 前兆因子数据（模式漂移校正：实时预报 - 历史回报气候态）...")
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


def _hindcast_clim(hc_da: xr.DataArray) -> xr.DataArray:
    """沿年份维（forecast_reference_time/time）和成员维（number）求均值，构建气候态背景场。

    返回仅含 step（可能还有 lat/lon）维的气候态 DataArray。
    """
    mean_dims = [
        d for d in hc_da.dims
        if d in ("forecast_reference_time", "time", "date", "valid_time", "number")
    ]
    return hc_da.mean(dim=mean_dims) if mean_dims else hc_da


def _anomaly_daily(
    rt_da: xr.DataArray,
    hc_clim_da: xr.DataArray,
) -> xr.DataArray:
    """对齐 step 坐标后计算 rt - hc_clim 异常值，保留 number 维（集合成员）。"""
    common_steps = np.intersect1d(rt_da.step.values, hc_clim_da.step.values)
    if len(common_steps) == 0:
        warnings.warn(
            "_anomaly_daily: rt 与 hc_clim 无公共 step 坐标，返回全 NaN 异常值",
            UserWarning,
            stacklevel=2,
        )
        return rt_da * np.nan
    return rt_da.sel(step=common_steps) - hc_clim_da.sel(step=common_steps)


def _extract_3day_mean_anom(anom_da: xr.DataArray) -> np.ndarray:
    """取异常 DataArray 前 3 个日步均值（Logic A），每成员返回一个标量。

    anom_da 应已完成空间平均，dims 为 (number, step)，step 为日尺度 timedelta。
    """
    da = anom_da.isel(step=slice(0, 3))
    dims_to_mean = [d for d in da.dims if d != "number"]
    return da.mean(dim=dims_to_mean).values.flatten()


# ═══════════════════════════════════════════════════════════════════════
# 因子一：华北冷涡 PV 异常（NCVI）
# 空间范围：[35°N–50°N, 115°E–130°E]（纬度降序 GRIB → slice(50, 35)）
# ═══════════════════════════════════════════════════════════════════════
pv_ds = xr.open_dataset(f"{dir_s2s_antecedent}ecmf_pf_60_2023-06.grib", engine="cfgrib")
_validate_dataset(pv_ds, "pv (NCVI rt)", skip_coverage_check=True)
pv_region = pv_ds["pv"].sel(latitude=slice(50, 35), longitude=slice(115, 130))
_pv_rt = select_init_time(pv_region, start_time).resample(step="D").mean()
_pv_rt_spatial = _pv_rt.mean(
    dim=[d for d in _pv_rt.dims if d not in ("number", "step")]
)  # (number, step_daily)

# 文件含 2003-2022 年（20年）多年历史回报数据，按同一起报日（6月12日）和相同预报步长归档。
# 若修改 start_date，需同步更新文件路径（路径中的 "2023-06-12" 对应起报日期）。
ncvi_hc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/Antecedent/ecmf_pf_NCVI_2023-06-12.grib",
    engine="cfgrib",
)
_validate_dataset(
    ncvi_hc_ds, "pv (NCVI hc)", skip_init_check=True, skip_coverage_check=True
)
_pv_hc = ncvi_hc_ds["pv"].sel(latitude=slice(50, 35), longitude=slice(115, 130))
_pv_hc_daily = _pv_hc.resample(step="D").mean()
_pv_hc_spatial = _pv_hc_daily.mean(
    dim=[d for d in _pv_hc_daily.dims if d not in (
        "forecast_reference_time", "time", "date", "valid_time", "number", "step"
    )]
)  # spatial-avg: (year/number, step)
ncvi_hc_clim = _hindcast_clim(_pv_hc_spatial)  # (step,)

# Logic B DataArray（number, step_daily）：供逐日时间序列图使用
ncvi_anom_da = _anomaly_daily(_pv_rt_spatial, ncvi_hc_clim)
# Logic A 标量数组（供 OLS + 相对重要性分析）
ncvi_arr = _extract_3day_mean_anom(ncvi_anom_da)

# ═══════════════════════════════════════════════════════════════════════
# 因子二：印度夏季风降水异常（ISM）
# 空间范围：[10°N–25°N, 70°E–90°E]（纬度降序 → slice(25, 10)）
# ═══════════════════════════════════════════════════════════════════════
ism_ds = xr.open_dataset(
    f"{dir_s2s_antecedent}ecmf_pf_228228_2023-06.grib",
    engine="cfgrib",
    backend_kwargs={"errors": "ignore"},  # 跳过文件末尾损坏的 GRIB 消息
)
_validate_dataset(ism_ds, "tp (ISM rt)", skip_coverage_check=True)
ism_region = ism_ds["tp"].sel(latitude=slice(25, 10), longitude=slice(70, 90))
_ism_rt = select_init_time(ism_region, start_time).resample(step="D").mean()
_ism_rt_spatial = _ism_rt.mean(
    dim=[d for d in _ism_rt.dims if d not in ("number", "step")]
)  # (number, step_daily)

# 文件含 2003-2022 年多年历史回报数据（同一起报日 6月12日）。
ism_hc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/Surface/ecmf_pf_ISM_NCHN_2023-06-12.grib",
    engine="cfgrib",
    backend_kwargs={"errors": "ignore"},
)
_validate_dataset(
    ism_hc_ds, "tp (ISM hc)", skip_init_check=True, skip_coverage_check=True
)
_ism_hc = ism_hc_ds["tp"].sel(latitude=slice(25, 10), longitude=slice(70, 90))
_ism_hc_daily = _ism_hc.resample(step="D").mean()
_ism_hc_spatial = _ism_hc_daily.mean(
    dim=[d for d in _ism_hc_daily.dims if d not in (
        "forecast_reference_time", "time", "date", "valid_time", "number", "step"
    )]
)
ism_hc_clim = _hindcast_clim(_ism_hc_spatial)  # (step,)

ism_anom_da = _anomaly_daily(_ism_rt_spatial, ism_hc_clim)
ism_arr = _extract_3day_mean_anom(ism_anom_da)

# ═══════════════════════════════════════════════════════════════════════
# 因子三：印太暖池纬向海温梯度异常（SST_Grad = SSTA_AS - SSTA_WNP）
# 阿拉伯海 AS ：[5°N–25°N,  50°E– 90°E]（纬度降序 → slice(25,  5)）
# 西北太平洋 WNP：[5°N–25°N, 120°E–150°E]（纬度降序 → slice(25,  5)）
# ═══════════════════════════════════════════════════════════════════════
sst_ds = xr.open_dataset(
    f"{dir_s2s_antecedent}ecmf_pf_34_2023-06.grib", engine="cfgrib"
)
_validate_dataset(sst_ds, "sst (SST rt)", skip_coverage_check=True)
sst_as  = sst_ds["sst"].sel(latitude=slice(25, 5),   longitude=slice(50, 90))
sst_wnp = sst_ds["sst"].sel(latitude=slice(25, 5),   longitude=slice(120, 150))
# 保留 sst_wp / sst_io 别名以兼容后续可能的引用
sst_wp = sst_wnp
sst_io = sst_as

_sst_as_rt  = select_init_time(sst_as,  start_time).resample(step="D").mean()
_sst_wnp_rt = select_init_time(sst_wnp, start_time).resample(step="D").mean()
_sst_as_rt_spatial  = _sst_as_rt.mean(
    dim=[d for d in _sst_as_rt.dims  if d not in ("number", "step")]
)
_sst_wnp_rt_spatial = _sst_wnp_rt.mean(
    dim=[d for d in _sst_wnp_rt.dims if d not in ("number", "step")]
)

# 文件含 2003-2022 年多年历史回报（AS 阿拉伯海区域，同一起报日 6月12日）。
sst_as_hc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/SST/ecmf_pf_34_AS_2023-06-12.grib", engine="cfgrib"
)
_validate_dataset(
    sst_as_hc_ds, "sst (SST AS hc)", skip_init_check=True, skip_coverage_check=True
)
_sst_as_hc = sst_as_hc_ds["sst"].sel(latitude=slice(25, 5), longitude=slice(50, 90))
_sst_as_hc_daily = _sst_as_hc.resample(step="D").mean()
_sst_as_hc_spatial = _sst_as_hc_daily.mean(
    dim=[d for d in _sst_as_hc_daily.dims if d not in (
        "forecast_reference_time", "time", "date", "valid_time", "number", "step"
    )]
)
sst_as_hc_clim = _hindcast_clim(_sst_as_hc_spatial)  # (step,)

# 文件含 2003-2022 年多年历史回报（WNP 西北太平洋区域，同一起报日 6月12日）。
sst_wnp_hc_ds = xr.open_dataset(
    "/data1/huangy/fig6/NC/MSE/SST/ecmf_pf_34_WNP_2023-06-12.grib", engine="cfgrib"
)
_validate_dataset(
    sst_wnp_hc_ds, "sst (SST WNP hc)", skip_init_check=True, skip_coverage_check=True
)
_sst_wnp_hc = sst_wnp_hc_ds["sst"].sel(latitude=slice(25, 5), longitude=slice(120, 150))
_sst_wnp_hc_daily = _sst_wnp_hc.resample(step="D").mean()
_sst_wnp_hc_spatial = _sst_wnp_hc_daily.mean(
    dim=[d for d in _sst_wnp_hc_daily.dims if d not in (
        "forecast_reference_time", "time", "date", "valid_time", "number", "step"
    )]
)
sst_wnp_hc_clim = _hindcast_clim(_sst_wnp_hc_spatial)  # (step,)

# 先各自求异常，再差分构建梯度（SSTA_AS - SSTA_WNP）
sst_as_anom_da  = _anomaly_daily(_sst_as_rt_spatial,  sst_as_hc_clim)
sst_wnp_anom_da = _anomaly_daily(_sst_wnp_rt_spatial, sst_wnp_hc_clim)
sst_grad_anom_da = sst_as_anom_da - sst_wnp_anom_da
sst_grad_arr = _extract_3day_mean_anom(sst_grad_anom_da)

# =============================================================================
# 3. ERA5 观测基准数据
# =============================================================================
print("正在加载 ERA5 观测基准数据...")
era5_file = "/data1/huangy/fig6/NC/MSE/T2m_era5_2023.6_NC.nc"
# ERA5 为再分析观测资料，时间轴为日历时间（非起报 + step 结构），
# 不适用 _validate_dataset 的起报时间 / step 覆盖检查；
# 时段覆盖由下方 _era5_study = _era5_daily.sel(time=slice(...)) 隐式保证。
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
    """提取研究时段（study_start ~ study_end）的集合成员均值，返回长度50的一维数组。

    日期映射规则（适配瞬时与日均两类数据）：
    ---------------------------------------------------------------
    经 resample(step="D") 后，step 坐标为 timedelta（如 0d, 1d, 2d, ...）。
    对应有效时刻：valid_time = start_time + step_timedelta
      - 瞬时数据（原始 step=0h,24h,...）→ resample 后 step 从 0d 开始
          step=2d → 2023-06-14 = study_start  step=12d → 2023-06-24 = study_end
      - 日均数据（原始 step=24h,48h,...）→ resample 后 step 从 1d 开始
          step=2d → 2023-06-14 = study_start  step=12d → 2023-06-24 = study_end
    两类数据的研究时段筛选均为：study_start <= start_time+step <= study_end
    """
    step_vals = daily_data.step
    # 使用 step 的 timedelta 值直接推算有效日期，避免 range(1,N+1) 的索引偏差
    dates = [start_time + pd.Timedelta(sv) for sv in step_vals.values]
    dates_ts = pd.to_datetime(dates)
    selected = daily_data.sel(
        step=[s for s, d in zip(step_vals.values, dates_ts)
              if study_start.date() <= d.date() <= study_end.date()]
    )
    if is_reforecast:
        return selected.mean(dim=list(selected.dims)).values.flatten()[0]
    dims_to_mean = [d for d in selected.dims if d != "number"]
    return selected.mean(dim=dims_to_mean).values.flatten()


def extract_daily_timeseries(daily_data, is_reforecast=False):
    """提取集合均值的逐日时间序列，返回形状 (_N_STUDY_DAYS,) 的数组。

    日期映射规则同 extract_period_mean：valid_time = start_time + step_timedelta
    研究时段 2023-06-14 ~ 2023-06-24 对应 step 2d ~ 12d（瞬时数据）
    或 step 2d ~ 12d（日均数据，step 从 1d 开始时同样覆盖该时段）。

    当输入数据的步长范围不足以覆盖完整研究时段（如高空文件步长较短）时，
    缺失日期以 NaN 填充，确保返回长度始终为 _N_STUDY_DAYS，
    避免不同变量数组长度不一致导致 DataFrame 构建失败。
    """
    step_vals = daily_data.step
    # 使用 step 的 timedelta 值直接推算有效日期
    dates = [start_time + pd.Timedelta(sv) for sv in step_vals.values]
    dates_ts = pd.to_datetime(dates)
    avail_pairs = [
        (s, d) for s, d in zip(step_vals.values, dates_ts)
        if study_start.date() <= d.date() <= study_end.date()
    ]

    if not avail_pairs:
        return np.full(_N_STUDY_DAYS, np.nan)

    avail_steps, _ = zip(*avail_pairs)
    selected = daily_data.sel(step=list(avail_steps))
    dims_to_mean = [d for d in selected.dims if d != "step"]
    avail_values = selected.mean(dim=dims_to_mean).values.flatten()  # shape: (n_avail,)

    if len(avail_values) == _N_STUDY_DAYS:
        return avail_values

    # 缺失日期用 NaN 填充，确保输出长度始终为 _N_STUDY_DAYS
    result = np.full(_N_STUDY_DAYS, np.nan)
    study_start_day = (study_start - start_time).days
    for i, step in enumerate(avail_steps):
        day_idx = int(pd.Timedelta(step).days) - study_start_day
        if 0 <= day_idx < _N_STUDY_DAYS:
            result[day_idx] = avail_values[i]

    # 末尾 NaN 前向填充：若 study_end 步次缺失（数据文件覆盖不足），
    # 用最近一个有效日的值填充，并发出警告，确保时间序列完整至 study_end。
    for i in range(1, _N_STUDY_DAYS):
        if np.isnan(result[i]) and not np.isnan(result[i - 1]):
            # 只在连续末尾缺失时进行前向填充
            is_trailing = all(np.isnan(result[i:]))
            if is_trailing:
                fill_date = (study_start + pd.Timedelta(days=i)).date()
                warnings.warn(
                    f"extract_daily_timeseries: {fill_date} 步次缺失，"
                    f"用前一有效日值前向填充（数据文件覆盖不足）",
                    UserWarning,
                    stacklevel=2,
                )
                result[i:] = result[i - 1]
                break
    return result


def extract_daily_timeseries_per_member(daily_data):
    """提取每个集合成员的逐日时间序列，返回形状 (n_members, _N_STUDY_DAYS) 的数组。

    日期映射规则同 extract_period_mean：valid_time = start_time + step_timedelta
    保留 number 维（每成员），step 维按研究时段筛选后保留。

    当输入数据步长不足以覆盖完整研究时段时，缺失日期以 NaN 填充，
    确保列数（天数）始终为 _N_STUDY_DAYS。
    """
    step_vals = daily_data.step
    # 使用 step 的 timedelta 值直接推算有效日期
    dates = [start_time + pd.Timedelta(sv) for sv in step_vals.values]
    dates_ts = pd.to_datetime(dates)
    avail_pairs = [
        (s, d) for s, d in zip(step_vals.values, dates_ts)
        if study_start.date() <= d.date() <= study_end.date()
    ]

    if not avail_pairs:
        n_members = daily_data.sizes.get("number", 1)
        return np.full((n_members, _N_STUDY_DAYS), np.nan)

    avail_steps, _ = zip(*avail_pairs)
    selected = daily_data.sel(step=list(avail_steps))
    dims_to_mean = [d for d in selected.dims if d not in ("step", "number")]
    result_sel = selected.mean(dim=dims_to_mean) if dims_to_mean else selected
    if "number" in result_sel.dims and "step" in result_sel.dims:
        result_sel = result_sel.transpose("number", "step")
    avail_values = result_sel.values  # shape: (n_members, n_avail_steps)

    if avail_values.shape[1] == _N_STUDY_DAYS:
        return avail_values

    # 缺失日期用 NaN 填充，确保列数为 _N_STUDY_DAYS
    n_members = avail_values.shape[0]
    result = np.full((n_members, _N_STUDY_DAYS), np.nan)
    study_start_day = (study_start - start_time).days
    for i, step in enumerate(avail_steps):
        day_idx = int(pd.Timedelta(step).days) - study_start_day
        if 0 <= day_idx < _N_STUDY_DAYS:
            result[:, day_idx] = avail_values[:, i]

    # 末尾列 NaN 前向填充：若 study_end 步次缺失（数据文件覆盖不足），
    # 用最近一个有效列的值填充，确保所有成员的时间序列完整至 study_end。
    for i in range(1, _N_STUDY_DAYS):
        col = result[:, i]
        if np.all(np.isnan(col)) and not np.all(np.isnan(result[:, i - 1])):
            is_trailing = all(np.all(np.isnan(result[:, j])) for j in range(i, _N_STUDY_DAYS))
            if is_trailing:
                fill_date = (study_start + pd.Timedelta(days=i)).date()
                warnings.warn(
                    f"extract_daily_timeseries_per_member: {fill_date} 步次缺失，"
                    f"用前一有效日值前向填充（数据文件覆盖不足）",
                    UserWarning,
                    stacklevel=2,
                )
                for j in range(i, _N_STUDY_DAYS):
                    result[:, j] = result[:, i - 1]
                break
    return result  # shape: (n_members, _N_STUDY_DAYS)


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

# --- relativeIMP 相对重要性（Johnson 相对权重法，基于相关矩阵特征分解）---
print("\n计算 relativeIMP 相对重要性（Johnson 相对权重法）...")
ri_df = data_df[["temp_mean"] + FACTOR_COLS].copy()
ri_results = relativeImp(ri_df, outcomeName="temp_mean", driverNames=FACTOR_COLS)
print("\nrelativeIMP 相对重要性结果：")
print(ri_results.to_string(index=False))

swanlab.log(
    {
        **{f"RI_{row['driver']}": row["rawRelaImpt"] for _, row in ri_results.iterrows()},
        **{f"RI_norm_{row['driver']}": row["normRelaImpt"] for _, row in ri_results.iterrows()},
    }
)

# =============================================================================
# 6. 图1+图2：LMG 预测散点图 + 重要性棒格图（合并保存）
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)

fig_lmg, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(18, 7))

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
ax_scatter.set_title("LMG: Predicted vs ECMWF T2max (All 12 Factors)", fontsize=16, pad=15)
ax_scatter.set_xlabel("ECMWF T2max (°C)", fontsize=14)
ax_scatter.set_ylabel("Predicted T2max (°C)", fontsize=14)

# LMG 条形图（按重要性降序排列）
sorted_lmg = lmg_results.sort_values("normRelaImpt", ascending=False).reset_index(drop=True)
display_labels = [LABEL_MAP.get(d, d) for d in sorted_lmg["driver"]]
bars = ax_bar.bar(display_labels, sorted_lmg["normRelaImpt"], color="royalblue", alpha=0.7)
ax_bar.set_title("LMG Relative Importance (All 12 Factors)", fontsize=16, pad=15)
ax_bar.set_ylabel("Normalized Relative Importance (%)", fontsize=14)
ax_bar.set_xticks(range(len(display_labels)))
ax_bar.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
ax_bar.set_ylim(0, 25)
for bar, (_, row) in zip(bars, sorted_lmg.iterrows()):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        f"raw: {row['rawRelaImpt']:.3f}\nnorm: {row['normRelaImpt']:.1f}%",
        ha="center", va="bottom", fontsize=10, rotation=90,
    )
plt.tight_layout()
out_combined_lmg = f"{OUT_DIR}/1.s2s.fig_combined_LMG_{start_date}.png"
plt.savefig(out_combined_lmg, dpi=300, bbox_inches="tight")
plt.show()
print(f"LMG 散点+棒格合并图已保存至: {out_combined_lmg}")

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

rmse = np.sqrt(np.nanmean((ts_y - ts_y_pred.values) ** 2))
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
out_ts = f"{OUT_DIR}/1.s2s.fig_ts_LMG_{start_date}.png"
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
out_corr = f"{OUT_DIR}/1.s2s.fig_corr_LMG_{start_date}.png"
plt.savefig(out_corr, dpi=300, bbox_inches="tight")
plt.show()
print(f"相关矩阵图已保存至: {out_corr}")

# =============================================================================
# 10. 图4+图5：Johnson (relativeIMP) 预测散点图 + 重要性棒格图（合并保存）
# =============================================================================
print("\n生成 Johnson (relativeIMP) 方法分析图...")

fig_ri, (ax_ri_scatter, ax_ri_bar) = plt.subplots(1, 2, figsize=(18, 7))
slope_c, intercept_c, r_value_c, _, _ = linregress(y_final, y_pred_full)
ax_ri_scatter.scatter(y_final, y_pred_full, color="steelblue", alpha=0.65, s=90, zorder=3)
ax_ri_scatter.plot(
    y_final, slope_c * y_final + intercept_c,
    color="black", alpha=0.5, lw=2,
)
text_ri = (
    f"Corr: {r_value_c:.2f}\n$R^2$: {r_squared_full:.2f}\nAdj $R^2$: {r_squared_adj:.2f}"
)
ax_ri_scatter.text(
    0.05, 0.95, text_ri,
    transform=ax_ri_scatter.transAxes, fontsize=14, va="top",
    bbox=dict(facecolor="white", alpha=0.7),
)
ax_ri_scatter.set_title(
    "Johnson (relativeIMP): Predicted vs ECMWF T2max\n(OLS, All 12 Factors)",
    fontsize=15, pad=12,
)
ax_ri_scatter.set_xlabel("ECMWF T2max (°C)", fontsize=14)
ax_ri_scatter.set_ylabel("Predicted T2max (°C)", fontsize=14)

sorted_ri = ri_results.sort_values("normRelaImpt", ascending=False).reset_index(drop=True)
display_labels_ri = [LABEL_MAP.get(d, d) for d in sorted_ri["driver"]]
bars_ri_only = ax_ri_bar.bar(
    display_labels_ri, sorted_ri["normRelaImpt"], color="darkorange", alpha=0.75,
)
ax_ri_bar.set_title(
    "Johnson (relativeIMP) Relative Importance (All 12 Factors)", fontsize=15, pad=15,
)
ax_ri_bar.set_ylabel("Normalized Relative Importance (%)", fontsize=14)
ax_ri_bar.set_xticks(range(len(display_labels_ri)))
ax_ri_bar.set_xticklabels(display_labels_ri, rotation=45, ha="right", fontsize=12)
ax_ri_bar.set_ylim(0, 25)
for bar, (_, row) in zip(bars_ri_only, sorted_ri.iterrows()):
    ax_ri_bar.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        f"raw: {row['rawRelaImpt']:.3f}\nnorm: {row['normRelaImpt']:.1f}%",
        ha="center", va="bottom", fontsize=10, rotation=90,
    )
plt.tight_layout()
out_combined_ri = f"{OUT_DIR}/1.s2s.fig_combined_RI_{start_date}.png"
plt.savefig(out_combined_ri, dpi=300, bbox_inches="tight")
plt.show()
print(f"Johnson 散点+棒格合并图已保存至: {out_combined_ri}")

# =============================================================================
# 10c. 图6：Johnson (relativeIMP) 模型逐日时间序列图（独立保存）
# =============================================================================
fig_ts_ri, ax_ts_ri = plt.subplots(figsize=(14, 7))
for m in range(n_members):
    lbl = "Ensemble Members (ECMWF)" if m == 0 else None
    ax_ts_ri.plot(
        plot_dates, ts_tmx_members[m],
        color="lightcoral", alpha=0.2, lw=0.7, label=lbl,
    )
for m in range(n_members):
    lbl = "Ensemble Members (Predicted)" if m == 0 else None
    ax_ts_ri.plot(
        plot_dates, ts_member_preds[m],
        color="#add8e6", alpha=0.15, lw=0.7, label=lbl,
    )
ax_ts_ri.plot(
    plot_dates, ts_y,
    marker="o", color="crimson", lw=2.2, zorder=5,
    label="ECMWF $T_{max}$ (Ensemble Mean)",
)
ax_ts_ri.plot(
    plot_dates, era5_obs_ts,
    marker="^", color="black", lw=2.2, zorder=6, label="ERA5 Observation",
)
ax_ts_ri.plot(
    plot_dates, ts_y_pred.values,
    marker="s", linestyle="--", color="#add8e6", lw=2.2, zorder=5,
    label="OLS Prediction (Johnson / relativeIMP)",
)
rmse_ri = np.sqrt(np.nanmean((ts_y - ts_y_pred.values) ** 2))
ax_ts_ri.text(
    0.02, 0.92, f"RMSE: {rmse_ri:.2f} °C",
    transform=ax_ts_ri.transAxes, fontsize=16,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)
ax_ts_ri.set_title(
    "Time Series Evolution (All 12 Factors, Johnson / relativeIMP Model)",
    fontsize=15, pad=12,
)
ax_ts_ri.set_ylabel("T2max (°C)", fontsize=14)
ax_ts_ri.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig_ts_ri.autofmt_xdate(rotation=30)
ax_ts_ri.legend(fontsize=12, loc="lower right")
ax_ts_ri.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
out_ts_ri = f"{OUT_DIR}/1.s2s.fig_ts_RI_{start_date}.png"
plt.savefig(out_ts_ri, dpi=300, bbox_inches="tight")
plt.show()
print(f"Johnson 时间序列图已保存至: {out_ts_ri}")

# =============================================================================
# 11. 图7+图8：12 个因子 S2S 集合预报时间序列图（全程 6.12–6.24, 13 天）
# =============================================================================
print("\n生成 12 个因子时间序列图（6.12–6.24）...")

_FULL_DAYS = (study_end - start_time).days + 1  # = 13
plot_dates_full = pd.date_range(start=start_time, end=study_end)


def _extract_full13_ts_per_member(daily_data):
    """从 start_time (6.12) 到 study_end (6.24) 提取每成员逐日序列。
    返回 (n_members, _FULL_DAYS) 数组，缺失步次填 NaN。
    day_idx = int(Timedelta(step).days) 直接作为 0-based 天数偏移。
    """
    step_vals = daily_data.step
    dates = [start_time + pd.Timedelta(sv) for sv in step_vals.values]
    dates_ts = pd.to_datetime(dates)
    avail_pairs = [
        (s, d) for s, d in zip(step_vals.values, dates_ts)
        if start_time.date() <= d.date() <= study_end.date()
    ]
    if not avail_pairs:
        n_members = daily_data.sizes.get("number", 1)
        return np.full((n_members, _FULL_DAYS), np.nan)
    avail_steps, _ = zip(*avail_pairs)
    selected = daily_data.sel(step=list(avail_steps))
    dims_to_mean = [d for d in selected.dims if d not in ("step", "number")]
    result_sel = selected.mean(dim=dims_to_mean) if dims_to_mean else selected
    if "number" in result_sel.dims and "step" in result_sel.dims:
        result_sel = result_sel.transpose("number", "step")
    avail_values = result_sel.values  # (n_members, n_avail)
    n_members = avail_values.shape[0]
    result = np.full((n_members, _FULL_DAYS), np.nan)
    for _i, _sv in enumerate(avail_steps):
        _di = int(pd.Timedelta(_sv).days)
        if 0 <= _di < _FULL_DAYS:
            result[:, _di] = avail_values[:, _i]
    return result


# 第一类：9 个动态因子（直接从已有日变量提取全 13 天）
_ts13_nchn_tp  = _extract_full13_ts_per_member(daily_max_NCHNtp)
_ts13_sm       = _extract_full13_ts_per_member(daily_max_sm)
_ts13_wnpsh    = _extract_full13_ts_per_member(daily_max_z_WNPSH)
_ts13_ssr      = _extract_full13_ts_per_member(daily_mean_ssr)
_ts13_shf      = _extract_full13_ts_per_member(daily_mean_sshf)
_ts13_z500_raw = _extract_full13_ts_per_member(daily_max_z_NCHN)
_ts13_mse_max  = _extract_full13_ts_per_member(daily_max_mse_star_max)
_ts13_mse500   = _extract_full13_ts_per_member(daily_max_mse_star500)
_ts13_barrier  = _extract_full13_ts_per_member(daily_max_barrier)

# Z500 异常：逐成员减去气候态均值（延伸至全 13 天）
_step_refo_all  = daily_max_z_refo_NCHN.step.values
_refo_dates_all = pd.to_datetime([start_time + pd.Timedelta(sv) for sv in _step_refo_all])
_refo_mask_full = (
    (_refo_dates_all >= pd.Timestamp(start_time))
    & (_refo_dates_all <= pd.Timestamp(study_end))
)
_refo_clim_13 = np.full(_FULL_DAYS, np.nan)
if _refo_mask_full.any():
    _refo_sel13 = daily_max_z_refo_NCHN.sel(step=_step_refo_all[_refo_mask_full])
    _refo_mean13_vals = _refo_sel13.mean(
        dim=[d for d in _refo_sel13.dims if d != "step"]
    ).values.flatten()
    for _j, _sv in enumerate(_step_refo_all[_refo_mask_full]):
        _di = int(pd.Timedelta(_sv).days)
        if 0 <= _di < _FULL_DAYS:
            _refo_clim_13[_di] = _refo_mean13_vals[_j]
_ts13_z500_anom = _ts13_z500_raw - _refo_clim_13[np.newaxis, :]

# 第二类：NCVI、ISM、SST_Grad（直接使用 Section 2 计算的逐日异常 DataArray，Logic B）
_ts13_ncvi  = _extract_full13_ts_per_member(ncvi_anom_da)
_ts13_ism   = _extract_full13_ts_per_member(ism_anom_da)
_ts13_sst_grad = _extract_full13_ts_per_member(sst_grad_anom_da)

# 整理为有序字典（顺序与 FACTOR_COLS 一致）
_factor_ts13 = {
    "NCHN_tp":          _ts13_nchn_tp,
    "sm_avg":           _ts13_sm,
    "WNPSH":            _ts13_wnpsh,
    "SSR_Avg":          _ts13_ssr,
    "SHF_Avg":          _ts13_shf,
    "z500_anom_NCHN":   _ts13_z500_anom,
    "MSEstar_max_NCHN": _ts13_mse_max,
    "MSEstar500_NCHN":  _ts13_mse500,
    "Barrier_NCHN":     _ts13_barrier,
    "NCVI":             _ts13_ncvi,
    "ISM":              _ts13_ism,
    "SST_Grad":         _ts13_sst_grad,
}

# 两张图（3行×2列），每张绘制 6 个因子
_out_factor_ts = {}
for _grp_label, _grp_cols in [("A", FACTOR_COLS[:6]), ("B", FACTOR_COLS[6:])]:
    fig_fts, axes_fts = plt.subplots(3, 2, figsize=(14, 12))
    for _idx, _col in enumerate(_grp_cols):
        _ax = axes_fts.flat[_idx]
        _ts = _factor_ts13[_col]           # (n_members, 13)
        _n_mem = _ts.shape[0]
        # 集合成员细线（背景展布）
        for _m in range(_n_mem):
            _ax.plot(plot_dates_full, _ts[_m], color="lightsteelblue", alpha=0.2, lw=0.6)
        # 集合均值粗线
        _ax.plot(
            plot_dates_full, np.nanmean(_ts, axis=0),
            color="dodgerblue", lw=2.5, alpha=1.0,
            marker="o", markersize=5, label="Ensemble Mean",
        )
        _ax.set_title(LABEL_MAP.get(_col, _col).replace("\n", " "), fontsize=13)
        _ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(_ax.get_xticklabels(), rotation=30, ha="right")
        _ax.legend(fontsize=9, loc="best")
        _ax.grid(True, linestyle=":", alpha=0.6)
    plt.suptitle(
        f"S2S Ensemble Forecast: 12 Factors Time Series"
        f" (2023-06-12 ~ 2023-06-24) [{_grp_label}]",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _out_f = f"{OUT_DIR}/1.s2s.fig_factor_ts_{_grp_label}_{start_date}.png"
    plt.savefig(_out_f, dpi=300, bbox_inches="tight")
    plt.show()
    _out_factor_ts[_grp_label] = _out_f
    print(f"因子时间序列图 [{_grp_label}] 已保存至: {_out_f}")

swanlab.log({
    "combined_LMG": swanlab.Image(out_combined_lmg),
    "ts_LMG": swanlab.Image(out_ts),
    "corr_LMG": swanlab.Image(out_corr),
    "combined_RI": swanlab.Image(out_combined_ri),
    "ts_RI": swanlab.Image(out_ts_ri),
    "factor_ts_A": swanlab.Image(_out_factor_ts["A"]),
    "factor_ts_B": swanlab.Image(_out_factor_ts["B"]),
})

swanlab.finish()
print("\nLMG 相对重要性分析全部完成！")
