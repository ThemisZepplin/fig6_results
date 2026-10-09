# Methods audit: forecast-error metrics and member grouping (Fig.2 / Fig.10)

**Scope audited.** This report audits only the accessible code in `tmax_error_stage_eval_cell.py` and `tmax_error_stage_eval_plot_only_cell.py`. It does not infer behavior from unavailable notebooks or HPC outputs. No code, parameters, data, or figure outputs were modified.

**Primary code files.**

- Heavy science workflow: `tmax_error_stage_eval_cell.py`.
- Standalone plotting-only renderer: `tmax_error_stage_eval_plot_only_cell.py`.

**Methods explicitly found in this window.** ERA5/S2S Tmax daily preprocessing, ensemble quantiles, daily error, signed Bias, ensemble-mean RMSE, member-wise RMSE, ensemble spread, coverage diagnostics, Hot17/Middle17/Cold17 grouping, fixed Stage-I grouping, Fig.10 lead-time summary, and Fig.10 RMSE-definition audit.

**Methods not found in this window.** Multiple linear regression (MLR), LMG relative importance, Johnson relative weights, LOFO, residualized predictors, SM--SHF partial fitted component, centered contribution, fitted component returned to Tmax space, spatial significance, Hovmöller diagnostics, propagation diagnostics, and daily synoptic composites are **not implemented in the audited code window**. They should be audited in the separate windows that actually contain those methods.

---

## 1. Method summary table

| Method | Code location | Input | Sample dimension | Formula | Output unit | Main purpose | Interpretation boundary | Needs citation |
|---|---|---|---|---|---|---|---|---|
| ERA5 daily Tmax preprocessing | `load_era5_tmax_daily()` | ERA5 `t2m` file | dates 2023-06-14--24 over NCHN | BJT-shifted gridpoint daily max, K-to-C if needed, cos-lat regional mean | °C | Observed verification target | Observation/reanalysis reference, not perfect truth | Data citation for ERA5 |
| S2S daily Tmax members | `load_s2s_tmax_daily_members()` | ECMWF S2S `mx2t6` CF+PF | 51 members by date and init | BJT valid time from init + step + 8 h; daily max; K-to-C; cos-lat mean | °C | Forecast-member Tmax series | Depends on existing helper `load_s2s_ensemble_with_cf`; no raw GRIB audit here | ECMWF S2S data citation |
| Daily ensemble statistics | `compute_daily_error_table()` | ERA5 daily Tmax; S2S member daily Tmax | date × init × 51 members | ensemble mean, std, min/max, p05/p25/p75/p95; error = ensmean - ERA5 | °C and boolean coverage | Daily forecast error and spread | Percentile rank is empirical over 51 members | None beyond data |
| Signed Bias | `compute_stage_summary()` | `daily_error` | date samples within stage | \(\mathrm{Bias}=N^{-1}\sum_t e_t\) | °C | Mean signed forecast error | Positive/negative daily errors can cancel | Standard forecast verification text |
| Ensemble-mean RMSE | `compute_stage_summary()` and `audit_fig10_rmse_definition()` | `daily_error` | date samples within stage | \(\sqrt{N^{-1}\sum_t e_t^2}\) | °C | Magnitude of daily ensemble-mean errors | Not `abs(stage-mean S2S - stage-mean ERA5)` except algebraically related to Bias only before squaring | Standard forecast verification text |
| Member-wise Bias/MAE/RMSE | `compute_member_stage_metrics()` | each member vs ERA5 | date samples within stage for each member | per-member mean, mean absolute, RMS error | °C | Distribution of member skill/error | Not ensemble-mean error; each member is evaluated separately | Standard forecast verification text |
| Member RMSE summary | `compute_stage_summary()` | member-wise RMSE | 51 members within init × stage | median, IQR, min, max | °C | Ensemble member divergence in error magnitude | Not equal to ensemble-mean RMSE | None |
| Ensemble spread | `compute_daily_error_table()`; `compute_stage_summary()`; `build_fig2b_leadtime_total_summary()` | S2S members | daily or stage-mean member samples | daily std over members; stage mean of daily spread; Fig.10 auxiliary spread = std over member Total stage-mean Tmax | °C | Ensemble dispersion | Different spread definitions are used in different outputs; must label explicitly | Forecast ensemble verification references if used |
| Coverage diagnostics | `compute_daily_error_table()`; `compute_stage_summary()` | ERA5 and ensemble range/quantiles | date samples | fraction of dates ERA5 falls inside min--max or p05--p95 | fraction | Reliability/range diagnostic | Not a calibrated probability statement with 51 members | Ensemble verification if discussed |
| Hot17/Middle17/Cold17 grouping | `compute_member_hotcold_groups_by_stage()` | member stage-mean Tmax | 51 members per init × stage | sort descending by member stage-mean Tmax; ranks 1--17 Hot17, 18--34 Middle17, 35--51 Cold17 | group labels; °C metrics | Diagnose warm/cold ensemble subgroups | Grouping is target-variable-dependent; not independent causal grouping | None, but describe clearly |
| Fixed Stage-I grouping | `compute_fixed_stageI_groups()` | Hot/Cold labels from Stage-I | member labels merged to all stages | use Stage-I group labels as fixed reference | group labels | Test persistence of Stage-I hot/cold classification | Fixed labels are defined from Stage-I only | None |
| Fig.10 RMSE audit | `audit_fig10_rmse_definition()` | daily and stage summary CSVs | four inits, Total stage | recompute Bias and RMSE from daily errors; compare to summary | °C and boolean pass/fail | Demonstrate RMSE definition | Audit only checks existing CSV consistency | None |
| MLR | not found | not found | not found | not found | not found | 本窗口未使用 | Must audit in MLR window | Needs original method refs if used elsewhere |
| LMG | not found | not found | not found | not found | not found | 本窗口未使用 | Must audit in relative-importance window | Needs original method refs if used elsewhere |
| Johnson relative weights | not found | not found | not found | not found | not found | 本窗口未使用 | Must audit in relative-importance window | Needs original method refs if used elsewhere |
| LOFO | not found | not found | not found | not found | not found | 本窗口未使用 | Must audit in attribution/sensitivity window | Needs clear definition and references if used elsewhere |
| SM--SHF partial fitted component | not found | not found | not found | not found | not found | 本窗口未使用 | Must audit in process-diagnostic window | Needs method description if used elsewhere |

---

## 2. Symbols used below

| Symbol | Meaning in audited code |
|---|---|
| \(i\) | S2S initialization date; requested inits are 2023-06-01, 2023-06-05, 2023-06-08, and 2023-06-12. |
| \(t\) | BJT calendar date in the target period. |
| \(s\) | Stage label: Stage-I, Stage-II, or Total. |
| \(m\) | Ensemble member number; member 0 is CF and members 1--50 are PF. |
| \(M\) | Number of members; code enforces \(M=51\). |
| \(N_s\) | Number of days in stage \(s\): 4 for Stage-I, 7 for Stage-II, 11 for Total. |
| \(T^{\mathrm{ERA5}}_t\) | ERA5 NCHN area-mean BJT daily Tmax. |
| \(T_{m,t}^{\mathrm{S2S}}\) | S2S member \(m\) NCHN area-mean BJT daily Tmax. |
| \(\bar{T}^{\mathrm{S2S}}_t\) | Daily ensemble mean, \(M^{-1}\sum_m T_{m,t}^{\mathrm{S2S}}\). |
| \(e_t\) | Daily ensemble-mean forecast error, \(\bar{T}^{\mathrm{S2S}}_t-T^{\mathrm{ERA5}}_t\). |
| \(e_{m,t}\) | Daily member forecast error, \(T_{m,t}^{\mathrm{S2S}}-T^{\mathrm{ERA5}}_t\). |

---

## 3. Method cards

## ERA5 and S2S Tmax preprocessing

### 1. Scientific purpose
To put observed and forecast Tmax on the same regional, daily, BJT calendar basis before computing forecast errors and member divergence.

### 2. Code evidence
- File: `tmax_error_stage_eval_cell.py`.
- Functions: `load_era5_tmax_daily()`, `load_s2s_tmax_daily_members()`, `_coslat_weighted_mean()`, `_to_celsius_if_needed()`.
- Constants: `ERA5_TMAX_FILE`, `S2S_PF_FILE`, `S2S_CF_FILE`, `TARGET_DATES`, `STAGES`, `EXPECTED_NDAYS`.
- Output fields downstream: `ERA5_Tmax`, `S2S_ensmean_Tmax`, `daily_error`, member `bias`, `mae`, `rmse`.
- Key logic: ERA5 is BJT-shifted by `time + 8h`, resampled with `.max()`, converted from K to °C when mean > 150, and regionally averaged with cosine-latitude weighting. S2S uses `init time + step + 8h`, `resample(time="1D").max()`, K-to-°C if needed, cos-lat mean, and enforces `number == 51`.

### 3. Current code definition
- Time window: 2023-06-14 through 2023-06-24.
- Stages: Stage-I = 2023-06-14--17; Stage-II = 2023-06-18--24; Total = 2023-06-14--24.
- Space: NCHN via `safe_slice_region(..., *NCHN_BOX)` followed by cosine-latitude weighted mean. The audited file assumes `NCHN_BOX` exists in the notebook environment.
- Members: S2S must contain `number` dimension with 51 members; member 0 is interpreted as CF and 1--50 as PF in member metrics.
- Missing values: explicit target-date coverage checks raise `ValueError`; no imputation is implemented.

### 4. Mathematical formula
For each grid point, daily Tmax is first calculated on BJT calendar days:
\[
T_{g,t}^{\max}=\max_{h\in t_{\mathrm{BJT}}} T_{g,h}.
\]
The NCHN regional value is then
\[
T_t=\frac{\sum_{g\in\Omega} \cos(\phi_g)T_{g,t}^{\max}}{\sum_{g\in\Omega}\cos(\phi_g)},
\]
where \(\Omega\) is the NCHN domain and \(\phi_g\) is latitude.

### 5. Method principle
General principle: daily maxima should be computed before spatial aggregation when the target variable is daily Tmax. Current implementation follows this by taking BJT daily maxima on gridded data and then applying the area-weighted regional mean.

### 6. Output and interpretation
Output values are regional daily Tmax in °C. They should be interpreted as NCHN regional mean daily maximum temperature, not as point-station maxima.

### 7. Difference from adjacent methods
This preprocessing differs from stage-mean diagnostics: daily values are retained first, and stage metrics are computed later from daily errors or daily member values.

### 8. Assumptions and limitations
- ERA5 is used as the verification reference.
- The NCHN box is assumed to be correctly defined in prior notebook state.
- K-to-°C conversion is inferred from mean > 150.
- S2S CF/PF merge logic is delegated to an existing helper not defined in this file.

### 9. Methods-ready wording
**中文草稿：** ERA5 和 ECMWF S2S Tmax 均首先转换到北京时间日历日。对每个格点先计算北京时间日最高 2 m temperature，再在华北区域内采用 \(\cos(\phi)\) 纬度权重进行区域平均。若温度场均值大于 150，则按 Kelvin 转换为 Celsius。S2S 预报的有效时间由起报时间、预报步长和 8 h 时区偏移共同确定，并保留 CF0 与 PF1--50 共 51 个成员。

**English JGR-style draft:** ERA5 and ECMWF S2S Tmax were placed on a common Beijing-time daily calendar before verification. At each grid point, daily maximum 2-m temperature was first computed for each BJT day and then averaged over the North China domain using cosine-latitude weights. Temperature fields were converted from Kelvin to degrees Celsius when required. S2S valid times were derived from initialization time plus lead time and an 8-h offset, and all 51 members (CF0 and PF1--PF50) were retained for member-level diagnostics.

### 10. Items needing confirmation or references
- Confirm the exact coordinate order and physical domain of `NCHN_BOX` in the upstream notebook.
- Cite ERA5 and ECMWF S2S data sources in the final Methods.

---

## Daily ensemble statistics and daily error

### 1. Scientific purpose
To quantify, day by day, how the S2S ensemble mean and ensemble distribution compare with ERA5 during the heatwave target period.

### 2. Code evidence
- File: `tmax_error_stage_eval_cell.py`.
- Function: `compute_daily_error_table()`.
- Key variables: `ens_mean`, `ens_spread`, `ens_min`, `ens_max`, quantiles `[0.05,0.25,0.75,0.95]`, `daily_error`, `era5_percentile_rank`.
- Output CSV: `tables/tmax_daily_error_timeseries.csv`.

### 3. Current code definition
For every successful init and each target date, the code computes the ensemble mean, ensemble standard deviation across `number`, min/max, p05/p25/p75/p95, ERA5 coverage flags, and percentile rank. Dates must overlap ERA5 and S2S for all target days.

### 4. Mathematical formula
\[
\bar{T}^{\mathrm{S2S}}_t = \frac{1}{M}\sum_{m=1}^{M}T_{m,t}^{\mathrm{S2S}},\quad M=51.
\]
\[
e_t = \bar{T}^{\mathrm{S2S}}_t - T_t^{\mathrm{ERA5}}.
\]
The empirical ERA5 percentile rank is
\[
P_t=100\times \frac{1}{M}\sum_{m=1}^{M}\mathbf{1}(T_{m,t}^{\mathrm{S2S}}\le T_t^{\mathrm{ERA5}}).
\]

### 5. Method principle
The method separates the central forecast tendency (ensemble mean) from ensemble distribution diagnostics (spread and quantiles). The percentile rank places ERA5 relative to the forecast-member distribution.

### 6. Output and interpretation
- `daily_error` is in °C and signed.
- `ens_spread`, quantiles, min/max are in °C.
- Coverage flags are boolean, later averaged as rates.
- Percentile rank is empirical with 51 members, not a continuous calibrated CDF.

### 7. Difference from adjacent methods
Daily error is the input to stage Bias/RMSE. It is not a stage-mean error until averaged over time.

### 8. Assumptions and limitations
- 51 members are treated as the ensemble sample.
- The percentile rank has coarse resolution because \(M=51\).
- Coverage of min--max or p05--p95 is descriptive and not a formal calibration test.

### 9. Methods-ready wording
**中文草稿：** 对每个起报日和每个验证日，本文从 51 个 S2S 成员计算集合均值、集合标准差、最小值、最大值以及 5/25/75/95 分位数。日尺度集合均值误差定义为 S2S 集合平均 Tmax 减 ERA5 Tmax。ERA5 在集合分布中的经验百分位用 51 个成员中不超过 ERA5 的比例表示。

**English JGR-style draft:** For each initialization and verification day, we computed the ensemble mean, standard deviation, minimum, maximum, and the 5th, 25th, 75th, and 95th percentiles across the 51 S2S members. The daily ensemble-mean error was defined as the S2S ensemble-mean Tmax minus ERA5 Tmax. The empirical percentile rank of ERA5 was computed as the fraction of members not exceeding the ERA5 value.

### 10. Items needing confirmation or references
No method-specific reference is required beyond standard ensemble-verification terminology, unless coverage/rank is emphasized in Results.

---

## Signed Bias

### 1. Scientific purpose
To diagnose whether the ensemble-mean forecast is systematically warmer or colder than ERA5 over a stage.

### 2. Code evidence
- File: `tmax_error_stage_eval_cell.py`.
- Function: `compute_stage_summary()`.
- Key variable: `ensmean_bias = float(dsub["daily_error"].mean())`.
- Output field: `ensmean_bias` in `tmax_stage_summary_metrics.csv`; Fig.10 `bias` in `fig2b_leadtime_total_tmax_error_bias_rmse.csv`.

### 3. Current code definition
Bias is computed from daily ensemble-mean errors within each stage after filtering dates by the stage start and end dates. It is signed and uses daily errors, not member errors.

### 4. Mathematical formula
For stage \(s\) with \(N_s\) days,
\[
\mathrm{Bias}_{i,s}=\frac{1}{N_s}\sum_{t\in s}\left(\bar{T}^{\mathrm{S2S}}_{i,t}-T^{\mathrm{ERA5}}_t\right)=\frac{1}{N_s}\sum_{t\in s}e_{i,t}.
\]

### 5. Method principle
General principle: signed Bias measures average directional error. Current implementation uses daily ensemble-mean errors and then averages them in time.

### 6. Output and interpretation
Output is °C. Negative Bias indicates a cold ensemble-mean forecast relative to ERA5; positive Bias indicates a warm forecast. Because it is signed, positive and negative daily errors can compensate.

### 7. Difference from adjacent methods
Bias is not RMSE. A small Bias can occur even if daily errors are large, because opposite-signed errors cancel. Bias is also not a member-wise metric.

### 8. Assumptions and limitations
- Sensitive to sign cancellation.
- For single-event stages with 4, 7, or 11 days, values are descriptive diagnostics rather than climatological skill estimates.

### 9. Methods-ready wording
**中文草稿：** 阶段平均 Bias 定义为该阶段内逐日集合平均 Tmax 误差的算术平均，即 S2S 集合均值减 ERA5。该指标保留正负号，因此可诊断集合平均预报的冷/暖偏差，但正负日误差可能相互抵消。

**English JGR-style draft:** The stage-mean signed Bias was computed as the arithmetic mean of daily ensemble-mean Tmax errors over the stage, where the daily error is the S2S ensemble-mean Tmax minus ERA5 Tmax. This metric preserves the sign of the error and therefore diagnoses cold or warm ensemble-mean bias, but opposite-signed daily errors can compensate.

### 10. Items needing confirmation or references
No special method citation is needed, but standard forecast-verification terminology may be cited.

---

## Ensemble-mean RMSE and Fig.10 RMSE audit

### 1. Scientific purpose
To quantify the magnitude of daily ensemble-mean Tmax errors without sign cancellation and to document for Fig.10 that RMSE is based on daily errors rather than on a single stage-mean difference.

### 2. Code evidence
- File: `tmax_error_stage_eval_cell.py`.
- Function: `compute_stage_summary()` uses `np.sqrt(np.mean(dsub["daily_error"] ** 2))`.
- Function: `audit_fig10_rmse_definition()` recomputes RMSE from daily errors and compares it to the stage summary using a tolerance of `1e-6`.
- Output fields: `ensmean_rmse`, `rmse_recomputed_from_daily_error`, `abs_stage_mean_difference`, `daily_error_std`, `rmse_definition_pass`.

### 3. Current code definition
The code first computes daily ensemble-mean errors, squares those daily errors, averages them over time, and then takes the square root. The audit also computes `abs_stage_mean_difference` as a contrast, where the stage-mean difference equals the mean daily error, so its absolute value equals `abs(Bias)`.

### 4. Mathematical formula
\[
\mathrm{RMSE}_{i,s}^{\mathrm{ensmean}}=\sqrt{\frac{1}{N_s}\sum_{t\in s} e_{i,t}^2},
\quad e_{i,t}=\bar{T}_{i,t}^{\mathrm{S2S}}-T_t^{\mathrm{ERA5}}.
\]
The audited alternative is
\[
\left|\overline{\bar{T}^{\mathrm{S2S}}}_{i,s}-\overline{T^{\mathrm{ERA5}}}_{s}\right| = \left|\frac{1}{N_s}\sum_{t\in s}e_{i,t}\right|=|\mathrm{Bias}_{i,s}|.
\]
This is not the RMSE except when daily errors have no variability around their mean in sign/magnitude.

### 5. Method principle
General principle: RMSE measures the root mean squared magnitude of errors and penalizes large errors. Current implementation applies this principle to daily ensemble-mean errors.

### 6. Output and interpretation
Output is °C. It is the daily-error magnitude of the ensemble mean over a stage. It is not the mean of member RMSE values, not a stage-mean absolute difference, and not an ERA5 average.

### 7. Difference from adjacent methods
- Bias is signed and can cancel.
- RMSE squares errors before averaging and is not subject to sign cancellation.
- Member-wise RMSE evaluates each member separately; ensemble-mean RMSE evaluates the ensemble mean time series.

### 8. Assumptions and limitations
- RMSE is sensitive to large daily errors.
- Single-event stages have small time samples.
- It summarizes magnitude but not timing or sign pattern separately.

### 9. Methods-ready wording
**中文 Methods 草稿：** 集合平均 RMSE 用逐日集合平均误差计算。具体地，先在每个验证日计算 S2S 集合平均 Tmax 与 ERA5 Tmax 的差值，再对这些逐日误差平方、按阶段取平均，最后开平方。该 RMSE 不等于阶段平均 S2S Tmax 与阶段平均 ERA5 Tmax 差值的绝对值；后者等于 signed Bias 的绝对值，仅反映平均误差。

**English JGR-style Methods draft:** The ensemble-mean RMSE was evaluated from daily ensemble-mean errors. For each initialization and stage, we first computed the daily error as the S2S ensemble-mean Tmax minus ERA5 Tmax, squared these daily errors, averaged them over the stage, and then took the square root. Thus, the RMSE is \(\sqrt{\mathrm{mean}_t(e_t^2)}\), not the absolute difference between stage-mean S2S Tmax and stage-mean ERA5 Tmax.

**Results/Discussion-ready Bias--RMSE distinction:** Signed Bias and RMSE can show different lead-time behavior because they summarize different aspects of the daily error sequence. Signed Bias measures the mean error and can be reduced by compensation between positive and negative daily errors. RMSE measures the magnitude of daily errors after squaring and is therefore larger when day-to-day errors are variable or when individual daily errors are large, even if the signed mean error is modest.

### 10. Items needing confirmation or references
Standard forecast-verification citation may be added. No additional code confirmation is needed for the RMSE order; it is explicitly implemented and audited.

---

## Member-wise Bias, MAE, and RMSE

### 1. Scientific purpose
To quantify how each individual S2S member performs relative to ERA5, allowing member-level divergence and grouping diagnostics.

### 2. Code evidence
- File: `tmax_error_stage_eval_cell.py`.
- Function: `compute_member_stage_metrics()`.
- Output CSV: `tmax_member_error_by_stage.csv`.
- Fields: `init`, `stage`, `member`, `member_type`, `bias`, `mae`, `rmse`, `n_days`.

### 3. Current code definition
For each init, stage, and member, the code selects stage dates, subtracts ERA5 daily Tmax from the member daily Tmax, and calculates mean error, mean absolute error, and RMSE. Member 0 is labeled `cf`; members 1--50 are labeled `pf`.

### 4. Mathematical formula
\[
\mathrm{Bias}_{m,s}=\frac{1}{N_s}\sum_{t\in s}e_{m,t},
\quad
\mathrm{MAE}_{m,s}=\frac{1}{N_s}\sum_{t\in s}|e_{m,t}|,
\]
\[
\mathrm{RMSE}_{m,s}=\sqrt{\frac{1}{N_s}\sum_{t\in s}e_{m,t}^2},
\quad e_{m,t}=T_{m,t}^{\mathrm{S2S}}-T_t^{\mathrm{ERA5}}.
\]

### 5. Method principle
Each member is treated as an individual forecast trajectory. This exposes member-to-member variation that the ensemble mean can mask.

### 6. Output and interpretation
All error metrics are in °C. Member-wise RMSE is used to form distributions and boxplots; it is not the same as ensemble-mean RMSE.

### 7. Difference from adjacent methods
The black diamond in Fig.2 panel (c) is `ensmean_rmse` from the stage summary, not a statistic of the member RMSE distribution. The boxplot represents member-wise RMSE values.

### 8. Assumptions and limitations
- Members are not guaranteed statistically independent.
- Sample size is 51 members but only 4/7/11 daily verification times depending on stage.

### 9. Methods-ready wording
**中文草稿：** 对每个集合成员分别计算阶段内 Bias、MAE 和 RMSE。成员误差定义为该成员逐日 Tmax 与 ERA5 逐日 Tmax 的差值。该成员尺度评估保留了 51 个成员之间的离散程度，并与集合平均误差分开报告。

**English JGR-style draft:** Member-wise error metrics were computed separately for each of the 51 S2S members. For each stage, the member error was defined as the member daily Tmax minus ERA5 daily Tmax, from which signed Bias, MAE, and RMSE were calculated. These member-wise metrics describe the spread of individual forecast errors and are distinct from the RMSE of the ensemble-mean forecast.

### 10. Items needing confirmation or references
No special citation needed beyond standard forecast verification.

---

## Ensemble spread and quantiles

### 1. Scientific purpose
To describe the ensemble distribution around the ensemble mean and to show forecast uncertainty/member divergence.

### 2. Code evidence
- Daily spread and quantiles in `compute_daily_error_table()`.
- Stage summary mean spread in `compute_stage_summary()`.
- Fig.10 auxiliary spread in `build_fig2b_leadtime_total_summary()`.

### 3. Current code definition
There are two distinct spread definitions:
1. Daily spread: standard deviation across members for each date, stored as `ens_spread`; stage `mean_spread` is the mean of daily spread values.
2. Fig.10 auxiliary spread: standard deviation across the 51 member Total-period stage-mean Tmax values with `ddof=0`; it is written to an auxiliary CSV and is not plotted in the revised Fig.10.

### 4. Mathematical formula
Daily spread:
\[
\sigma_t=\sqrt{\frac{1}{M}\sum_{m=1}^{M}\left(T_{m,t}^{\mathrm{S2S}}-\bar{T}_t^{\mathrm{S2S}}\right)^2}.
\]
Stage mean spread:
\[
\overline{\sigma}_{s}=\frac{1}{N_s}\sum_{t\in s}\sigma_t.
\]
Auxiliary Total spread for Fig.10:
\[
\sigma_{\mathrm{stage\ mean}}=\mathrm{std}_{m}\left(\frac{1}{N_{Total}}\sum_{t\in Total}T_{m,t}^{\mathrm{S2S}}\right),
\]
with `ddof=0`.

### 5. Method principle
Spread measures dispersion among ensemble members. Daily spread describes instantaneous/day-specific dispersion; stage-mean spread describes dispersion in member-average heat amplitude.

### 6. Output and interpretation
Spread outputs are in °C. They do not directly measure forecast error unless compared with verification; they quantify ensemble dispersion.

### 7. Difference from adjacent methods
Spread is not RMSE. RMSE compares forecasts to ERA5; spread measures member dispersion around the ensemble distribution.

### 8. Assumptions and limitations
- The ensemble is treated as a sample of 51 members.
- Daily and stage-mean spread are not interchangeable.

### 9. Methods-ready wording
**中文草稿：** 集合离散度在日尺度上定义为 51 个成员 Tmax 的标准差；阶段平均离散度为该日尺度标准差在阶段内的平均。Fig.10 的辅助 spread 另定义为每个成员 Total 阶段平均 Tmax 在成员维上的标准差，该量仅作为辅助 QC 输出，修订后的主图不展示。

**English JGR-style draft:** Ensemble spread was computed as the standard deviation across the 51 members. Daily spread was first calculated for each verification day and then averaged over a stage for stage summaries. For the lead-time summary, an auxiliary spread diagnostic was also computed as the standard deviation across members of their Total-period stage-mean Tmax; this auxiliary quantity was retained for QC but not shown in the revised main figure.

### 10. Items needing confirmation or references
If spread is interpreted probabilistically, add ensemble forecast-verification references.

---

## Hot17/Middle17/Cold17 grouping and fixed Stage-I labels

### 1. Scientific purpose
To identify whether the warmest and coldest forecast members differ systematically in error and to support later physical-process comparisons between warm and cold ensemble subgroups.

### 2. Code evidence
- File: `tmax_error_stage_eval_cell.py`.
- Function: `compute_member_hotcold_groups_by_stage()`.
- Summary functions: `compute_hotcold_group_summary_by_stage()`, `compute_fixed_stageI_groups()`, `compute_fixed_stageI_group_summary()`.
- Output CSVs: `tmax_member_hotcold_group_by_stage.csv`, `tmax_hotcold_group_summary_by_stage.csv`, fixed Stage-I group tables.

### 3. Current code definition
For each init × stage, the code computes each member's stage-mean Tmax, ERA5 stage-mean Tmax, Bias, MAE, and RMSE. Members are sorted descending by `member_stage_mean_Tmax`. Ranks 1--17 are `Hot17`, ranks 18--34 are `Middle17`, and ranks 35--51 are `Cold17`. Fixed Stage-I grouping merges each member's Stage-I group label onto all stages for the same init.

### 4. Mathematical formula
Member stage mean:
\[
\overline{T}_{m,s}^{\mathrm{S2S}}=\frac{1}{N_s}\sum_{t\in s}T_{m,t}^{\mathrm{S2S}}.
\]
Ranking:
\[
r_{m,s}=\mathrm{rank}_{\mathrm{descending}}\left(\overline{T}_{m,s}^{\mathrm{S2S}}\right).
\]
Group assignment:
\[
G_{m,s}=\begin{cases}
\mathrm{Hot17}, & r_{m,s}\le 17,\\
\mathrm{Middle17}, & 18\le r_{m,s}\le 34,\\
\mathrm{Cold17}, & r_{m,s}\ge 35.
\end{cases}
\]

### 5. Method principle
This is a rank-based ensemble stratification by forecast heat amplitude. It is not based on ERA5 error or RMSE; it is based on the members' own stage-mean Tmax.

### 6. Output and interpretation
Group summaries include mean Tmax, mean Bias, mean MAE, mean RMSE, median RMSE, and min/max Tmax. These describe differences between forecast-warm and forecast-cold subsets.

### 7. Difference from adjacent methods
Hot17/Cold17 is not the same as Best17/Worst17 by RMSE. A hot member can still have large error. Fixed Stage-I grouping differs from per-stage grouping because labels are defined only from Stage-I and then propagated.

### 8. Assumptions and limitations
- Groups depend on the forecast target variable, so they are not independent of Tmax amplitude.
- Group size is fixed at 17 members because total M=51.
- Per-stage grouping changes labels by stage; fixed Stage-I grouping uses Stage-I as a reference and should be named accordingly.

### 9. Methods-ready wording
**中文草稿：** 为诊断集合内温度强度分化，本文按每个成员自身的阶段平均 Tmax 对 51 个成员降序排序。最暖的 17 个成员定义为 Hot17，中间 17 个为 Middle17，最冷的 17 个为 Cold17。该分组不使用 ERA5 误差作为排序依据。另构建固定 Stage-I 分组，将 Stage-I 中得到的 Hot/Middle/Cold 标签应用于后续阶段，以评估成员温度强度分组的持续性。

**English JGR-style draft:** To diagnose member divergence in forecast heat amplitude, the 51 members were ranked by their own stage-mean Tmax for each initialization and stage. The 17 warmest members were labeled Hot17, the middle 17 Middle17, and the 17 coldest Cold17. This grouping was based solely on forecast stage-mean Tmax and did not use ERA5 errors. A fixed Stage-I grouping was also constructed by applying the Stage-I labels to subsequent stages for the same members, providing a persistence diagnostic of the Stage-I warm/cold classification.

### 10. Items needing confirmation or references
No external method citation is required, but authors should ensure all figures distinguish per-stage Hot17/Cold17 from fixed Stage-I groups.

---

## Fig.10 lead-time summary and RMSE audit

### 1. Scientific purpose
To summarize how Total-period signed Bias and RMSE vary with initialization date and to provide an audit trail for the RMSE definition.

### 2. Code evidence
- Functions: `build_fig2b_leadtime_total_summary()`, `audit_fig10_rmse_definition()`, `plot_fig2b_leadtime_total_summary()`, `run_fig2b_leadtime_total_summary_cell()`.
- Inputs: `tmax_stage_summary_metrics.csv`, `tmax_member_hotcold_group_by_stage.csv`, `tmax_daily_error_timeseries.csv`.
- Outputs: `fig2b_leadtime_total_tmax_error_bias_rmse.csv`, `fig2b_leadtime_total_tmax_spread_auxiliary.csv`, `fig10_rmse_definition_audit.csv`, Fig.10 PNG/PDF.

### 3. Current code definition
For each of four inits, the code reads Total-period `ensmean_bias` and `ensmean_rmse` from stage summary fields, computes auxiliary spread from member stage-mean Tmax, and separately audits that RMSE equals recomputed daily-error RMSE. The revised Fig.10 plots only signed Bias and RMSE.

### 4. Mathematical formula
The plotted values are
\[
\mathrm{Bias}_{i,Total}=\frac{1}{11}\sum_{t=\mathrm{Jun14}}^{\mathrm{Jun24}}e_{i,t},
\]
\[
\mathrm{RMSE}_{i,Total}=\sqrt{\frac{1}{11}\sum_{t=\mathrm{Jun14}}^{\mathrm{Jun24}}e_{i,t}^2}.
\]
The audit computes
\[
\mathrm{daily\_error\_std}_i=\mathrm{std}_{t}(e_{i,t}),
\]
with `ddof=0` in code, and checks summary RMSE against recomputed RMSE.

### 5. Method principle
The lead-time summary is a compact forecast-verification comparison across initialization dates. The audit separates the plotted RMSE from the alternative absolute stage-mean difference.

### 6. Output and interpretation
Fig.10 values are in °C. Signed Bias indicates direction; RMSE indicates magnitude. Auxiliary spread is written for QC but not plotted.

### 7. Difference from adjacent methods
Fig.10 does not show Hot17/Cold17 groups and does not show ensemble spread in the revised main figure.

### 8. Assumptions and limitations
- Only four initialization dates are summarized.
- Total period is a single 11-day case-study window.
- The code relies on previously generated CSVs.

### 9. Methods-ready wording
**中文草稿：** 为概括预报提前期对 Total 阶段误差的影响，本文对四个起报日分别报告 Total 阶段的集合平均 signed Bias 和 RMSE。Bias 与 RMSE 均来自阶段汇总表，并通过独立审计函数从逐日误差表重新计算 RMSE，以确认 RMSE 的计算顺序为逐日误差、平方、时间平均、开平方。

**English JGR-style draft:** To summarize the lead-time dependence of Total-period forecast error, we reported the signed Bias and RMSE of the ensemble-mean forecast for four initialization dates. Both metrics were read from the stage-summary table, and an independent audit recomputed RMSE from the daily-error time series to verify that RMSE was calculated as daily error, square, time mean, and square root.

### 10. Items needing confirmation or references
Confirm whether to refer to this figure as Fig.10 or Fig.2b in final manuscript files and captions.

---

## MLR, LMG, Johnson relative weights, LOFO, residual predictors, and SM--SHF partial fitted component

### 1. Scientific purpose
These are mentioned in the requested audit scope, but this code window does not implement them.

### 2. Code evidence
Searches of the audited files found no implementation of MLR fitting, LMG order averaging, Johnson orthogonalization, LOFO deletion metrics, residualized predictors, or SM--SHF partial fitted components.

### 3. Current code definition
本窗口未使用.

### 4. Mathematical formula
No formula can be attributed to this code window.

### 5. Method principle
General principles should not be written as current implementation until the relevant code window is audited.

### 6. Output and interpretation
No output fields in the audited files correspond to MLR coefficients, LMG percentages, Johnson relative weights, LOFO losses, residual predictors, or SM--SHF partial fitted Tmax.

### 7. Difference from adjacent methods
Forecast-error metrics in this file are verification diagnostics, not regression attribution or physical-process decomposition.

### 8. Assumptions and limitations
The absence of these methods in this file means the Methods section must be completed from other code windows.

### 9. Methods-ready wording
**中文草稿：** 本窗口未包含 MLR、LMG、Johnson relative weights、LOFO 或 partial fitted component 的实现，因此不能基于本窗口给出这些方法的可复现定义。

**English JGR-style draft:** The audited forecast-error scripts do not implement MLR, LMG, Johnson relative weights, LOFO, residualized predictors, or partial fitted components. These methods should be documented from the separate code sections where they are actually implemented.

### 10. Items needing confirmation or references
- Locate the regression/relative-importance/process-diagnostic code windows.
- Suggested search keywords: `LinearRegression`, `statsmodels`, `lmg`, `relative`, `Johnson`, `LOFO`, `drop`, `residual`, `partial fitted`, `centered contribution`, `SM`, `SHF`.
- Add original method references only after confirming the actual implementation.

---

## 4. Chinese Methods draft (forecast-error section)

本文首先将 ERA5 与 ECMWF S2S Tmax 统一到北京时间日尺度。ERA5 2 m temperature 和 S2S Tmax 均先在格点上计算北京时间日最高值，必要时由 Kelvin 转换为 Celsius，然后在华北区域内采用 \(\cos(\phi)\) 纬度权重进行区域平均。S2S 有效时间由起报时间、预报步长和 8 h 时区偏移确定，并保留 CF0 与 PF1--50 共 51 个成员。

对每个起报日和验证日，集合平均 Tmax 定义为 51 个成员的算术平均。日尺度集合平均误差定义为 S2S 集合平均 Tmax 减 ERA5 Tmax。阶段平均 signed Bias 为阶段内逐日集合平均误差的平均值；集合平均 RMSE 则先对逐日误差平方，再在阶段内求平均并开平方。因此，RMSE 表示逐日误差幅度，不等于阶段平均 S2S Tmax 与阶段平均 ERA5 Tmax 差值的绝对值。后者等于 signed Bias 的绝对值，可能因正负日误差抵消而较小。

成员尺度误差分别对每个成员计算。对成员 \(m\)，阶段内 Bias、MAE 和 RMSE 均基于该成员逐日 Tmax 与 ERA5 逐日 Tmax 的差值计算。成员 RMSE 的箱线图反映 51 个成员误差幅度的分布，而图中的黑色菱形表示集合平均预报的 RMSE，不是成员 RMSE 的平均值或中位数。

为诊断集合内温度强度分化，本文按成员自身的阶段平均 Tmax 对 51 个成员降序排序。最暖 17 个成员定义为 Hot17，中间 17 个为 Middle17，最冷 17 个为 Cold17。该分组仅依据 S2S 成员自身的阶段平均 Tmax，不使用 ERA5 误差排序。另构建固定 Stage-I 分组，将 Stage-I 中得到的成员组别应用于其他阶段，以评估 Stage-I 暖/冷成员分类在后续阶段的持续性。

---

## 5. English JGR-style Methods draft (forecast-error section)

ERA5 and ECMWF S2S Tmax were first placed on a common Beijing-time daily calendar. For both datasets, daily maximum temperature was computed at each grid point before spatial averaging. Temperatures were converted from Kelvin to degrees Celsius when required, and North China regional means were computed using cosine-latitude weights. S2S valid times were obtained from initialization time plus lead time and an 8-h offset. All 51 ensemble members were retained, with member 0 representing the control forecast and members 1--50 representing perturbed forecasts.

For each initialization and verification day, the ensemble-mean Tmax was computed as the arithmetic mean across the 51 members. The daily ensemble-mean error was defined as the S2S ensemble-mean Tmax minus ERA5 Tmax. Stage-mean signed Bias was computed as the mean of these daily errors over the corresponding stage. The ensemble-mean RMSE was computed by squaring the daily errors, averaging them over the stage, and taking the square root. Therefore, RMSE quantifies the magnitude of daily errors and is not the absolute difference between stage-mean S2S Tmax and stage-mean ERA5 Tmax, which is equivalent to the absolute value of signed Bias.

Member-wise error metrics were calculated separately for each member. For a given member and stage, Bias, MAE, and RMSE were computed from the member's daily Tmax errors relative to ERA5. The member-wise RMSE distribution therefore describes the spread of individual member errors, whereas the black diamond in the paper figure denotes the RMSE of the ensemble-mean forecast.

To diagnose ensemble divergence in forecast heat amplitude, members were ranked by their own stage-mean Tmax for each initialization and stage. The 17 warmest members were labeled Hot17, the middle 17 Middle17, and the 17 coldest Cold17. This grouping was based only on forecast stage-mean Tmax and did not use ERA5 errors. A fixed Stage-I grouping was additionally constructed by applying Stage-I group labels to subsequent stages, providing a diagnostic of whether the Stage-I warm/cold classification persists.

---

## 6. Interpretation boundaries

1. Bias is signed and can be reduced by compensation between positive and negative daily errors.
2. RMSE measures daily-error magnitude and is not subject to sign cancellation.
3. Ensemble-mean RMSE is not the average or median of member-wise RMSE.
4. Member-wise RMSE distributions do not directly measure ensemble-mean forecast error.
5. Spread is member dispersion, not forecast error against ERA5.
6. Hot17/Cold17 groups are defined by forecast Tmax amplitude and should not be interpreted as externally imposed physical regimes.
7. Fixed Stage-I groups should be clearly distinguished from per-stage Hot17/Cold17 groups.
8. All diagnostics are based on a single event and a small number of stage days.
9. None of these verification metrics establishes causality.

---

## 7. Literature and citation checklist

- ERA5 data citation.
- ECMWF S2S data citation.
- Standard forecast verification reference for Bias, RMSE, MAE, spread, and ensemble quantiles.
- If later Methods include MLR/LMG/Johnson/LOFO, add original method references after auditing actual implementation.

Suggested search keywords for missing method windows: `multiple linear regression`, `relative importance`, `LMG`, `Lindeman Merenda Gold`, `Johnson relative weights`, `leave one factor out`, `LOFO`, `residualized predictor`, `partial fitted component`.

---

## 8. Code audit issue checklist

| Issue | Status | Evidence / note |
|---|---|---|
| Bias definition is consistent with daily ensemble-mean error mean | Confirmed consistent | `compute_stage_summary()` uses `dsub["daily_error"].mean()`. |
| RMSE definition is daily error -> square -> time mean -> sqrt | Confirmed consistent | `compute_stage_summary()` and `audit_fig10_rmse_definition()` implement this. |
| RMSE is not abs(stage-mean difference) | Confirmed consistent | Audit explicitly computes `abs_stage_mean_difference` as contrast. |
| Why abs(stage-mean difference) equals abs(Bias) | Confirmed consistent | Difference of stage means equals mean daily error when both use same dates. |
| Why RMSE usually exceeds abs(Bias) | Confirmed consistent | RMSE includes daily-error variance; audit records `daily_error_std`. |
| Ensemble-mean RMSE vs member-wise RMSE terminology | Confirmed consistent | Separate fields: `ensmean_rmse` and member `rmse`; Fig.2 black diamond uses `ensmean_rmse`. |
| Black diamond definition | Confirmed consistent | Paper Fig.2 renderer plots `ensmean_rmse` as black diamond. |
| Stage-I, Stage-II, Total n_days | Confirmed consistent | `EXPECTED_NDAYS = {4,7,11}` and stage checks raise errors. |
| Hot17/Cold17 per-stage regrouping | Confirmed consistent | Per-stage groups sorted by `member_stage_mean_Tmax`. |
| Fixed Stage-I grouping distinction | Confirmed consistent | Separate `fixed_stageI_group` generated by merge from Stage-I labels. |
| Ensemble spread definition consistency | Needs clarification | Daily/stage `mean_spread` differs from Fig.10 auxiliary stage-mean spread; Methods must label each. |
| Standardized vs raw coefficients | Not found in current window | No regression coefficients present. |
| MLR / LMG / Johnson / LOFO terminology | Not found in current window | Must audit other code windows. |
| Partial fitted component as causal contribution | Not found in current window | Must audit process-diagnostic code before writing. |
| Fig.10 as Fig.2b vs Fig.10 terminology | Needs clarification | Function names retain Fig.2b while text says Fig.10. |
| Plot-only CSV renderer changes science values | Confirmed consistent | Plot-only file validates and plots existing CSVs only. |

---

## 9. Methods section placement suggestion

- **2.1 Data and preprocessing:** ERA5/S2S Tmax, BJT daily max, K-to-°C, NCHN cos-lat mean, CF/PF 51-member handling.
- **2.2 Event windows and ensemble-member grouping:** Stage-I, Stage-II, Total; Hot17/Middle17/Cold17; fixed Stage-I grouping.
- **2.3 Forecast-error metrics:** daily error, signed Bias, RMSE, member-wise RMSE, ensemble spread, coverage.
- **2.4 Lead-time summary and QC:** four-init Total-period Bias/RMSE and RMSE-definition audit.
- **2.5 Regression and relative-importance attribution:** not supported by this window; fill after auditing MLR/LMG/Johnson/LOFO code.
- **2.6 Process diagnostics:** not supported by this window; fill after auditing SM--SHF and composite code.
- **2.7 Robustness and quality control:** date completeness, 51-member checks, RMSE audit, input CSV validation in plotting-only renderer.

---

## 10. Methods not covered in this window but requiring audit elsewhere

- Multiple linear regression, including response/predictor definitions, intercept, standardization, coefficients, residuals, and R².
- Residual predictors such as SHF residual, TCC residual, or SM residual.
- LMG relative importance and whether all predictor orderings/submodels are averaged.
- Johnson relative weights and orthogonal-predictor mapping.
- LOFO and its exact loss or \(\Delta R^2\) definition.
- SM--SHF partial fitted component, centered contribution, baseline at mean predictor state, and return to Tmax space.
- Spatial composites, Hot17--Cold17 differences, significance testing, Hovmöller and propagation diagnostics.
