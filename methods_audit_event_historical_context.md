# Methods audit: event historical context and Fig.1 KDE/PDF

**Scope audited.** This report audits only the accessible code in `paper_fig1_event_background_cell.py`. It does not infer behavior from unavailable notebooks or HPC outputs. No code, parameters, data, or figure outputs were modified.

**Primary code file.** `paper_fig1_event_background_cell.py`.

**Methods explicitly found in this window.** ERA5 historical Tmax daily preprocessing, MSWEP precipitation daily accumulation, NCHN cosine-latitude regional mean, 1979--2022 same-window climatology, stage-mean Tmax anomaly, KDE/PDF and CDF percentile, empirical percentile, historical rank using 1979--2023, 2023 event marker, and overlap-source consistency check.

**Methods not found in this window.** S2S forecast-error metrics, Hot17/Cold17 grouping, MLR, LMG, Johnson relative weights, LOFO, residualized predictors, SM--SHF partial fitted component, Hovmöller diagnostics, and spatial significance are **not implemented in this Fig.1 code window**.

---

## 1. Method summary table

| Method | Code location | Input | Sample dimension | Formula | Output unit | Main purpose | Interpretation boundary | Needs citation |
|---|---|---|---|---|---|---|---|---|
| ERA5 historical Tmax loading | `load_era5_tmax_historical_daily_nchn()` and yearly helpers | ERA5 2 m temperature files, 1979--2023 | 45 years × 11 BJT days | yearly file read, UTC-to-BJT daily max, K-to-C, NCHN weighted mean | °C | Build historical context for event Tmax | Depends on available yearly files and variable detection | ERA5 data citation |
| MSWEP precipitation daily accumulation | `load_mswep_precip_daily_bjt_nchn()` | MSWEP 2023 3-hourly/hourly precipitation | 2023-06-14--24 | UTC-to-BJT daily sum, unit handling, NCHN weighted mean | mm day⁻¹ | Observed precipitation context in Fig.1a | Precipitation is descriptive, not proof of causality | MSWEP citation |
| Cos-lat regional mean | `_coslat_weighted_mean_fig1()` | gridded Tmax/precip | NCHN grid cells | weighted average using \(\cos\phi\) | native variable unit | Area-weighted regional quantity | Box-average only, no subregional heterogeneity | None |
| Stage windows | `FIG1_WINDOWS` | BJT daily series | Total, Stage-I, Stage-II windows | same-calendar windows | days | Event-period definitions | Fig.1 plot currently displays Total and Stage-I PDFs only, but table computes Stage-II too | None |
| 1979--2022 climatology | `compute_fig1_historical_pdf_tables()` | annual stage-mean Tmax | 44 background years | mean over 1979--2022 for each window | °C | Same-window baseline | Different windows have different climatological means | Climatology convention if needed |
| Stage-mean Tmax anomaly | `compute_fig1_historical_pdf_tables()` | annual stage means and climatology | year × window | stage mean minus same-window climatology | °C | Historical extremeness on anomaly scale | Not absolute Tmax | None |
| KDE/PDF and KDE CDF | `compute_kde_pdf_and_cdf()` | 1979--2022 anomaly samples | 44 values per window | `gaussian_kde`, grid density, integrated CDF | density; percentile | Smooth background distribution and event CDF | Sensitive to KDE bandwidth; requires SciPy | KDE/statistics reference if discussed |
| Empirical percentile | `compute_fig1_historical_pdf_tables()` | background anomalies and event anomaly | 44 background years | `100 * mean(bg_anom <= event_anom)` | percentile | Nonparametric event position | Resolution limited by 44 samples | None |
| Historical rank | `compute_fig1_historical_pdf_tables()` | all anomalies 1979--2023 | 45 years | `1 + count(all_anom > event_anom)` | rank | Event rank including 2023 | Rank denominator differs from PDF background by design | None |
| PDF background excluding 2023 | `compute_fig1_historical_pdf_tables()` and QC lines | 1979--2022 background; 2023 event | 44 + 1 years | 2023 not in KDE background | °C anomaly | Avoid including event in reference PDF | Rank still includes 2023 | None |
| Overlap source consistency check | `run_overlap_source_consistency_check()` | old and new ERA5 source files for selected years | daily and stage means | old minus new source comparison | °C | QC for data-source transition | Only runs where overlapping old-source files exist | None |

---

## 2. Symbols used below

| Symbol | Meaning in audited code |
|---|---|
| \(y\) | Calendar year. |
| \(w\) | Window: Total, Stage-I, or Stage-II. |
| \(t\) | BJT daily date within a window. |
| \(N_w\) | Number of days in window \(w\). |
| \(T_{y,t}\) | NCHN regional ERA5 daily Tmax for year \(y\), date \(t\). |
| \(\overline{T}_{y,w}\) | Stage/window mean Tmax for year \(y\), window \(w\). |
| \(C_w\) | 1979--2022 same-window climatological mean. |
| \(A_{y,w}\) | Stage-mean Tmax anomaly, \(\overline{T}_{y,w}-C_w\). |
| \(B_w\) | Background anomaly sample for 1979--2022. |
| \(A_{2023,w}\) | 2023 event anomaly for window \(w\). |

---

## 3. Method cards

## ERA5 historical Tmax preprocessing

### 1. Scientific purpose
To construct a consistent 1979--2023 historical record of NCHN stage-mean Tmax for evaluating the extremeness of the 2023 heatwave.

### 2. Code evidence
- File: `paper_fig1_event_background_cell.py`.
- Functions: `find_era5_t2m_files_for_year()`, `load_era5_tmax_daily_nchn_for_year()`, `load_era5_tmax_historical_daily_nchn()`, `_open_era5_tmax_files_for_year()`.
- Constants: `ERA5_TMAX_HIST_ROOT_1979_1992`, `ERA5_TMAX_HIST_GLOB_1993_2023`, `FIG1_HIST_BACKGROUND_YEARS`, `FIG1_EVENT_YEAR`, `FIG1_TARGET_DATES`.
- QC fields: source group, files, variable, units, raw mean, converted mean, `n_days`, `complete_11_days`.

### 3. Current code definition
The code reads ERA5 Tmax year by year, not by opening all years at once. Years 1979--1992 are searched first as `air.2m.YYYY06.nc` under the old root, with recursive fallback. Years 1993--2023 are found via the configured glob. For each year, the code opens June data, standardizes time and coordinates, converts units if mean > 150, subsets NCHN, shifts to BJT, computes daily max, selects 06-14--06-24, and applies a cos-lat regional mean. Missing required years raise an error before final plotting.

### 4. Mathematical formula
For grid point \(g\), year \(y\), and BJT date \(t\),
\[
T_{g,y,t}^{\max}=\max_{h\in t_{BJT}}T_{g,y,h}.
\]
The regional mean is
\[
T_{y,t}=\frac{\sum_{g\in\Omega}\cos(\phi_g)T_{g,y,t}^{\max}}{\sum_{g\in\Omega}\cos(\phi_g)}.
\]

### 5. Method principle
The event is compared against same-calendar historical daily Tmax values after applying the same BJT daily maximum and regional averaging procedure to all years.

### 6. Output and interpretation
Output is NCHN regional mean daily Tmax in °C for each year from 1979 to 2023 and dates 06-14--06-24. It is a reanalysis-based regional diagnostic, not station-observed point Tmax.

### 7. Difference from adjacent methods
This preprocessing produces daily Tmax. Stage means and anomalies are computed later and should not be confused with daily maxima.

### 8. Assumptions and limitations
- K-to-°C conversion uses the mean > 150 heuristic.
- File discovery assumes the configured file naming conventions.
- Source consistency is checked only for selected overlap years if files exist.

### 9. Methods-ready wording
**中文草稿：** ERA5 历史 Tmax 按年份逐年读取，避免一次性合并全部原始逐小时文件。对每一年，先将时间转换到北京时间并在格点上计算 6 月 14--24 日的日最高 2 m temperature，必要时由 Kelvin 转为 Celsius，然后在华北区域内采用 \(\cos(\phi)\) 权重求区域平均。若任一年缺少完整的 6 月 14--24 日序列，代码会在 QC 阶段报错而不继续生成最终图。

**English JGR-style draft:** Historical ERA5 Tmax was processed year by year to avoid loading all hourly files simultaneously. For each year, 2-m temperature was shifted to the BJT calendar, daily maxima were computed at each grid point for 14--24 June, temperatures were converted to degrees Celsius when required, and North China regional means were calculated using cosine-latitude weights. The workflow requires complete daily values for all years from 1979 to 2023.

### 10. Items needing confirmation or references
- Cite ERA5.
- Confirm final manuscript domain definition and coordinate order for NCHN.

---

## MSWEP precipitation daily accumulation

### 1. Scientific purpose
To show observed precipitation evolution alongside ERA5 Tmax in Fig.1a, providing event-context information without attributing causality.

### 2. Code evidence
- File: `paper_fig1_event_background_cell.py`.
- Function: `load_mswep_precip_daily_bjt_nchn()`.
- Constants: `MSWEP_ROOTS`, `FIG1_TARGET_DATES`.
- Output field: `mswep_tp_nchn` in `fig1_event_evolution_daily_values.csv`.
- QC fields: original units, conversion note, sample counts.

### 3. Current code definition
MSWEP files for June 2023 are found from the configured `2023` root. The code guesses precipitation variables, standardizes time and coordinates, converts units only when units indicate meters, treats `mm/3h` as already millimeters per accumulation interval, shifts time to BJT, sums to daily totals, subsets NCHN, and applies cos-lat mean.

### 4. Mathematical formula
For precipitation accumulation \(P_{g,h}\), BJT daily precipitation is
\[
P_{g,t}=\sum_{h\in t_{BJT}}P_{g,h}.
\]
Regional daily precipitation is
\[
P_t=\frac{\sum_{g\in\Omega}\cos(\phi_g)P_{g,t}}{\sum_{g\in\Omega}\cos(\phi_g)}.
\]

### 5. Method principle
Accumulated precipitation should be summed over the target daily period, not averaged over sub-daily timesteps.

### 6. Output and interpretation
Output is mm day⁻¹. It is used as observed event context in Fig.1a and should not be described as proving that precipitation caused temperature changes.

### 7. Difference from adjacent methods
MSWEP precipitation is not used in the historical Tmax KDE/PDF calculations.

### 8. Assumptions and limitations
- Unit interpretation relies on metadata strings.
- Daily sample counts are printed to QC and should be checked in the HPC output.

### 9. Methods-ready wording
**中文草稿：** Fig.1a 中降水采用 MSWEP 数据。原始逐小时或 3 小时降水先转换到北京时间日历日，并按日累计得到 mm day⁻¹，再在华北区域内采用 \(\cos(\phi)\) 权重求区域平均。该降水序列仅用于描述事件演变背景，不用于证明因果机制。

**English JGR-style draft:** MSWEP precipitation was used to describe the observed event evolution in Fig. 1a. Sub-daily precipitation accumulations were shifted to the BJT calendar and summed to daily totals before cosine-latitude weighted averaging over North China. The precipitation series is used as contextual information and is not interpreted as a causal attribution of the temperature evolution.

### 10. Items needing confirmation or references
- Cite MSWEP.
- Confirm sample counts in `fig1_qc_summary.txt` from the HPC run.

---

## 1979--2022 climatology and stage-mean Tmax anomaly

### 1. Scientific purpose
To quantify how unusual the 2023 Total and Stage-I heat conditions were relative to same-calendar historical conditions.

### 2. Code evidence
- Function: `compute_fig1_historical_pdf_tables()`.
- Constants: `FIG1_WINDOWS`, `FIG1_HIST_BACKGROUND_YEARS`, `FIG1_EVENT_YEAR`.
- Output CSVs: `fig1_historical_tmax_pdf_values.csv`, `fig1_historical_tmax_pdf_summary.csv`.
- Fields: `stage_mean_tmax`, `climatology_mean_1979_2022`, `stage_mean_tmax_anomaly`, `is_background_1979_2022`, `is_2023`.

### 3. Current code definition
For each year and each window, the code computes the mean of daily NCHN Tmax over the same calendar dates. For each window separately, the climatological mean is the mean of 1979--2022 stage means. The anomaly for every year is stage-mean Tmax minus that same-window climatology.

### 4. Mathematical formula
Stage mean:
\[
\overline{T}_{y,w}=\frac{1}{N_w}\sum_{t\in w}T_{y,t}.
\]
Same-window climatology:
\[
C_w=\frac{1}{44}\sum_{y=1979}^{2022}\overline{T}_{y,w}.
\]
Anomaly:
\[
A_{y,w}=\overline{T}_{y,w}-C_w.
\]

### 5. Method principle
Same-calendar anomalies remove the average seasonal timing of the selected June window. Each window has its own climatological baseline.

### 6. Output and interpretation
Anomalies are in °C. A positive anomaly indicates the stage-mean Tmax was warmer than the 1979--2022 same-window mean.

### 7. Difference from adjacent methods
The KDE/PDF is plotted for anomaly values, not absolute Tmax. The event marker is the 2023 anomaly.

### 8. Assumptions and limitations
- Background period is 1979--2022.
- Single-window climatology does not remove long-term trends unless detrending is implemented; no detrending is present in this code.
- The event year 2023 is excluded from the climatological mean.

### 9. Methods-ready wording
**中文草稿：** 对每个历史年和每个事件窗口，先计算该窗口内 NCHN 区域平均日 Tmax 的阶段平均值。每个窗口分别以 1979--2022 年同日历窗口的阶段平均 Tmax 作为气候态。2023 年及历史各年的阶段平均 Tmax anomaly 定义为该年阶段平均值减去对应窗口的 1979--2022 气候态。

**English JGR-style draft:** For each year and event window, stage-mean Tmax was calculated by averaging the NCHN daily Tmax over the corresponding calendar dates. A same-window climatology was then computed as the mean stage-mean Tmax over 1979--2022. Stage-mean Tmax anomalies were defined as departures from this same-window climatology, with a separate climatological mean for each window.

### 10. Items needing confirmation or references
If trend effects are important, authors should decide whether to discuss the absence of detrending.

---

## KDE/PDF, KDE CDF percentile, empirical percentile, and historical rank

### 1. Scientific purpose
To place the 2023 event and Stage-I anomalies within a smoothed historical anomaly distribution and provide percentile/rank diagnostics.

### 2. Code evidence
- Functions: `compute_kde_pdf_and_cdf()`, `compute_fig1_historical_pdf_tables()`, `make_paper_fig1_event_background()`.
- KDE uses `scipy.stats.gaussian_kde` when available; otherwise the function raises for Fig.1 anomaly curves.
- Output fields: `kde_cdf_percentile_2023`, `empirical_percentile_2023`, `rank_2023_descending`, `rank_denominator`, `background_anomaly_p95`.

### 3. Current code definition
The KDE/PDF background sample is 1979--2022 stage-mean Tmax anomalies. The 2023 anomaly is not included in the KDE background and is plotted as a separate marker. KDE density is evaluated on a grid padded beyond the sample/event range. The CDF is numerically integrated from the KDE density and normalized. The empirical percentile is computed as the fraction of background anomalies not exceeding the 2023 anomaly. Rank is computed using all 1979--2023 anomaly values, so the denominator is 45.

### 4. Mathematical formula
KDE density is represented generally as
\[
\hat{f}_w(x)=\mathrm{KDE}(B_w),
\]
where \(B_w=\{A_{y,w}:y=1979,\ldots,2022\}\).
The KDE CDF percentile is
\[
100\times \hat{F}_w(A_{2023,w}),
\]
where \(\hat{F}\) is obtained by numerical integration of \(\hat{f}\).
The empirical percentile is
\[
100\times \frac{1}{44}\sum_{y=1979}^{2022}\mathbf{1}(A_{y,w}\le A_{2023,w}).
\]
The descending rank is
\[
1+\sum_{y=1979}^{2023}\mathbf{1}(A_{y,w}>A_{2023,w}).
\]

### 5. Method principle
KDE provides a smoothed estimate of the historical anomaly probability density. The empirical percentile and rank provide nonparametric checks that do not depend on the KDE smoothing.

### 6. Output and interpretation
- KDE/PDF y-axis is probability density.
- Percentiles are relative to the 1979--2022 background.
- Rank is relative to all available 1979--2023 years.
- A rank of 1/45 means the 2023 anomaly is the largest among 1979--2023 for that window.

### 7. Difference from adjacent methods
The PDF background excludes 2023 to avoid estimating the historical distribution with the event itself. The rank includes 2023 because the event must be ranked among all years including itself.

### 8. Assumptions and limitations
- KDE shape is sensitive to bandwidth chosen internally by `gaussian_kde`.
- The background sample has 44 values, which limits tail precision.
- No detrending or block bootstrap is implemented.

### 9. Methods-ready wording
**中文草稿：** 历史 PDF 使用 1979--2022 年同窗口 Tmax anomaly 估计，不包含 2023 年事件本身。代码采用 `scipy.stats.gaussian_kde` 对 1979--2022 anomaly 样本进行平滑密度估计，并通过对密度曲线数值积分得到 2023 anomaly 的 KDE CDF percentile。同时，经验 percentile 定义为 1979--2022 背景样本中不超过 2023 anomaly 的比例。历史 rank 则在 1979--2023 共 45 年中计算，因此 2023 不参与 PDF 背景估计，但参与 rank 排序。

**English JGR-style draft:** Historical anomaly PDFs were estimated from the 1979--2022 same-window Tmax anomalies, excluding the 2023 event from the background distribution. We used `scipy.stats.gaussian_kde` to obtain a smoothed density estimate and numerically integrated the density to calculate the KDE-based CDF percentile of the 2023 anomaly. We also reported an empirical percentile based on the fraction of 1979--2022 anomalies not exceeding the 2023 anomaly. Historical rank was computed over 1979--2023, so the event year was excluded from the PDF background but included in the ranking denominator.

### 10. Items needing confirmation or references
- Add a general KDE/statistical reference if the journal requires it.
- Confirm whether to report KDE CDF percentile, empirical percentile, or both in the final Methods/Results.

---

## Overlap source consistency check

### 1. Scientific purpose
To assess whether the old and new ERA5 data sources produce consistent NCHN daily and stage-mean Tmax for selected overlapping years.

### 2. Code evidence
- Function: `run_overlap_source_consistency_check()`.
- Check years: 1993, 2000, 2010, 2021.
- Output CSV: `overlap_source_consistency_check.csv` if overlapping files exist.

### 3. Current code definition
For each check year, the code attempts to process an old-source file and new-source files with the same daily Tmax pipeline and records daily and stage-mean differences `old - new`. If no overlapping old-source files are found, it prints a warning and does not write the CSV.

### 4. Mathematical formula
Daily difference:
\[
D_{y,t}=T_{y,t}^{old}-T_{y,t}^{new}.
\]
Stage-mean difference:
\[
D_{y,w}=\overline{T}_{y,w}^{old}-\overline{T}_{y,w}^{new}.
\]

### 5. Method principle
This is a source-consistency QC check, not a scientific diagnostic of the event.

### 6. Output and interpretation
Differences are in °C. Small differences support source consistency; large differences would require data-source investigation.

### 7. Difference from adjacent methods
The check does not enter the KDE calculations unless the processed historical data source itself changes.

### 8. Assumptions and limitations
- Only selected years are checked.
- The QC depends on whether overlapping old-source files exist.

### 9. Methods-ready wording
**中文草稿：** 为检查 ERA5 历史数据源切换的一致性，代码对若干重叠年份分别使用旧源和新源执行相同的日 Tmax 处理流程，并输出日尺度和阶段平均 Tmax 的差值。该步骤作为 QC 使用，不参与事件极端性指标的计算。

**English JGR-style draft:** As a data-source consistency check, selected overlapping years were processed with both the old and new ERA5 sources using the same daily Tmax workflow. Daily and stage-mean differences between the two sources were written as QC output. This check was used only to document source consistency and was not itself an event-extremeness metric.

### 10. Items needing confirmation or references
Review the generated overlap CSV from the HPC run if cited in Supplementary Methods.

---

## 4. Chinese Methods draft (Fig.1 event historical context)

本文使用 ERA5 2 m temperature 构建 1979--2023 年 6 月 14--24 日华北区域 Tmax 历史序列。所有年份均按相同流程处理：先转换到北京时间，在格点上计算日最高温，必要时由 Kelvin 转为 Celsius，然后在华北区域内采用 \(\cos(\phi)\) 权重求区域平均。1979--1992 年和 1993--2023 年分别来自不同 ERA5 文件路径，代码逐年读取并在 QC 中记录每年的数据源、变量名、单位和完整性。

Fig.1a 的降水采用 MSWEP 数据。原始逐小时或 3 小时降水转换到北京时间后按日累计，并同样在华北区域内采用 \(\cos(\phi)\) 权重求区域平均。该降水序列用于展示事件演变背景，不用于证明降水对 Tmax 演变的因果作用。

对每个历史年和事件窗口，先计算 NCHN 区域平均日 Tmax 的阶段平均值。每个窗口分别以 1979--2022 年同日历窗口阶段平均 Tmax 的平均值作为气候态。阶段平均 Tmax anomaly 定义为该年阶段平均值减去对应窗口的 1979--2022 气候态。历史 PDF 使用 1979--2022 年 anomaly 样本通过 Gaussian KDE 估计，2023 年不参与背景 PDF 估计，而作为单独事件标记绘制。经验 percentile 定义为 1979--2022 背景样本中不超过 2023 anomaly 的比例；历史 rank 则使用 1979--2023 共 45 年计算。

---

## 5. English JGR-style Methods draft (Fig.1 event historical context)

We constructed a 1979--2023 historical record of North China Tmax from ERA5 2-m temperature for 14--24 June. All years were processed consistently: fields were shifted to the Beijing-time calendar, daily maxima were computed at each grid point, temperatures were converted to degrees Celsius when required, and regional means were calculated using cosine-latitude weights. The 1979--1992 and 1993--2023 ERA5 records were read from separate configured data sources on a year-by-year basis, and the workflow recorded source, variable, unit, and completeness information for QC.

MSWEP precipitation was used in Fig. 1a to describe the observed event evolution. Sub-daily precipitation accumulations were shifted to the BJT calendar, summed to daily totals, and averaged over North China using the same cosine-latitude weights. This precipitation time series was used as contextual information and was not interpreted as causal evidence for the Tmax evolution.

For each historical year and event window, we computed the stage-mean NCHN daily Tmax. A same-window climatology was calculated from the 1979--2022 stage means, separately for each window. Stage-mean Tmax anomalies were then defined as departures from this same-window climatology. Historical PDFs were estimated from the 1979--2022 anomaly samples using Gaussian KDE, while the 2023 event anomaly was excluded from the PDF background and plotted as a separate marker. Empirical percentiles were computed as the fraction of 1979--2022 anomalies not exceeding the 2023 anomaly, whereas historical ranks were computed over all years from 1979 to 2023.

---

## 6. Interpretation boundaries

1. Tmax anomaly is relative to a same-window 1979--2022 climatology, not an annual or seasonal anomaly unless otherwise stated.
2. The KDE/PDF background excludes 2023; rank includes 2023. This is intentional and must be explained.
3. KDE smooths a 44-sample background and is sensitive to bandwidth.
4. Rank 1/45 means highest among available 1979--2023 values for that window, not necessarily an all-time observational record.
5. MSWEP precipitation in Fig.1a is event context and should not be described as a causal driver unless supported elsewhere.
6. No detrending is implemented in the audited code.
7. Stage-II historical PDF is still computed in the tables but is not shown in the current main Fig.1 layout.

---

## 7. Literature and citation checklist

- ERA5 data reference.
- MSWEP data reference.
- KDE / kernel density estimation reference if the method needs explicit citation.
- If discussing historical-event ranking, cite relevant climate-extremes methodology only after deciding final framing.

Suggested search keywords: `ERA5 citation`, `MSWEP precipitation citation`, `Gaussian KDE`, `kernel density estimation climatological anomalies`, `event rank percentile historical distribution`.

---

## 8. Code audit issue checklist

| Issue | Status | Evidence / note |
|---|---|---|
| ERA5 Tmax uses BJT daily max before regional mean | Confirmed consistent | `_to_bjt_daily(..., how="max")` and cos-lat mean in yearly processing. |
| MSWEP uses BJT daily sum | Confirmed consistent | `_to_bjt_daily(..., how="sum")` in MSWEP loader. |
| Cos-lat regional mean used | Confirmed consistent | `_coslat_weighted_mean_fig1()`. |
| 1979--2022 climatology | Confirmed consistent | `FIG1_HIST_BACKGROUND_YEARS` and climatology fields. |
| 2023 excluded from PDF background | Confirmed consistent | KDE background uses `is_background_1979_2022`. |
| Rank denominator includes 2023 | Confirmed consistent | `required_years` through event year and `rank_denominator`. |
| PDF variable is anomaly, not absolute Tmax | Confirmed consistent | QC line states anomaly variable; plot x-label is anomaly. |
| Stage-II computed but not plotted in current Fig.1 | Potential inconsistency | Tables compute Stage-II; current figure only plots Total and Stage-I. Methods/caption should state what is shown. |
| KDE bandwidth explicitly configured | Needs clarification | Code uses `gaussian_kde` default bandwidth; no manual bandwidth parameter. |
| Detrending | Not found in current window | No detrending implemented. |
| MSWEP precipitation causal interpretation | Confirmed consistent if described cautiously | QC says observed evolution only; Methods should avoid causal wording. |
| Overlap source consistency output always exists | Needs clarification | CSV is not written if no overlapping old-source files are found. |
| S2S forecast methods | Not found in current window | Use forecast-error audit file. |
| MLR/LMG/Johnson/LOFO/partial fitted component | Not found in current window | Must audit separate code windows. |

---

## 9. Methods section placement suggestion

- **2.1 Data sources:** ERA5 Tmax and MSWEP precipitation.
- **2.2 Observed event evolution:** BJT daily Tmax and precipitation, NCHN regional averaging, event windows.
- **2.3 Historical context:** 1979--2022 same-window climatology, stage-mean Tmax anomaly, KDE/PDF, percentile and rank.
- **2.4 Quality control:** yearly source records, complete-year checks, MSWEP sample counts, overlap source consistency.
- **2.5 Forecast-error methods:** not from this Fig.1 window; use the forecast-error audit.
- **2.6 Regression and process diagnostics:** not from this Fig.1 window; audit separate windows.

---

## 10. Methods not covered in this window but requiring audit elsewhere

- S2S daily errors, Bias, RMSE, member-wise RMSE, ensemble spread, and Hot17/Cold17 grouping.
- MLR, residualized predictors, LMG, Johnson relative weights, LOFO.
- SM--SHF partial fitted components and centered contributions.
- Composite maps, Hot17--Cold17 spatial differences, significance tests, Hovmöller, propagation diagnostics, and daily synoptic composites.
