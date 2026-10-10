# Methods audit: MLR attribution, LMG, Johnson, LOFO, and forecast-error metrics

Scope: this audit is based only on the accessible `stage_specific_residualized_ols_shf.py` window. It does not modify code, does not run the HPC/data pipeline, and does not infer methods that are not implemented or referenced in this file.

## 0. Files scanned

- `stage_specific_residualized_ols_shf.py`

## 1. Method summary table

| Method | Code location | Input | Sample dimension | Formula | Output unit | Main purpose | Interpretation boundary | Needs citation |
|---|---|---|---|---|---|---|---|---|
| Stage/member response `y_tmax` | `extract_window_only(..., agg_func="mean")`; raw rows written with `y_tmax` (`stage_specific_residualized_ols_shf.py:L976-L994`) | S2S member daily Tmax over each stage window | member axis, typically 51 members per init-stage | \(y_m=\overline{T_{m,t}}_{t\in W_s}\) | °C | Response for member-axis MLR | Not a daily skill metric; it is a stage-window member mean | No, unless data-source citation needed |
| Main residualized OLS / MLR | `fit_ols_lmg`; `sm.OLS(y, sm.add_constant(X))` (`L1132-L1160`) | stage-member predictor table; residualized predictors | member axis, 51 rows expected | \(y_m=\beta_0+\sum_k\beta_kX_{m,k}+\epsilon_m\) | y/fitted in °C; coefficients predictor-dependent | Quantify member-to-member association between predictors and Tmax | Linear association, not proof of causality | Standard OLS citation optional |
| Final slim MLR with standardized fitting and raw-equivalent coefficients | `StandardScaler`; `_slim_fit_ols_from_standardized`; raw-equivalent coefficient conversion (`L5234-L5290`) | final slim predictor set, `y_tmax` | member axis, per init × stage | fit in standardized X; raw coefficient \(\beta_k=\beta_k^*/\sigma_k\); raw intercept \(\beta_0=\beta_0^* - \sum_k\beta_k^*\mu_k/\sigma_k\) | fitted/y/RMSE in °C; coefficients raw units and standardized units | Final attribution model used by paper Fig.3–4 | Raw-equivalent coefficients are derived from standardized fit; do not mix coefficient spaces | Standard scaling/OLS citation optional |
| Residualized predictors | `calc_resid`; stage-specific residualization (`L1092-L1100`, `L1281-L1315`) | raw predictor and control predictor(s) | member axis within init-stage | \(R(Y\mid C)=Y-\widehat{Y}(C)\) | same unit as residualized variable | Reduce linear collinearity in selected predictor pairs | Residual means “not linearly explained by controls,” not independent causal forcing | Residualization/partial regression citation optional |
| LMG relative importance | `compute_lmg_mc` (`L1102-L1122`); slim version called with `mc_samples=2000` (`L5243-L5244`) | predictor matrix and `y_tmax` | member axis | Monte-Carlo average of non-negative incremental \(R^2\) over random predictor orders, normalized to 100% | % of model-explained importance after normalization | Partition model explanatory importance among correlated predictors | Monte-Carlo approximation; relative importance, not causal contribution | Yes: original LMG/relative-importance method literature |
| Johnson relative weights | `table_method_importance_comparison.csv`; `johnson_pct` consumed in Cell 14 (`L9361-L9364`); Johnson implementation block starts around `L5947-L6106` | final slim MLR predictors and response | member axis | orthogonal-predictor relative-weight method; exact transformation must be checked in full Johnson block | % when `johnson_pct` is used | Independent robustness check for factor importance | This window confirms outputs but the full orthogonalization algebra requires author check against implementation block | Yes: Johnson relative weights literature |
| LOFO sensitivity | final slim MLR LOFO rows (`L5303-L5327`); Cell 14 uses positive-normalized `lofo_positive_pct` (`L9349-L9355`) | full model and reduced models omitting one predictor | member axis | \(\Delta R^2_k=R^2_{full}-R^2_{-k}\); positive parts normalized to 100% for plotting | raw \(\Delta R^2\), \(\Delta RMSE\), normalized % | Complementary sensitivity diagnostic | Shared information can make \(\Delta R^2\) small/negative; not identical to LMG | No specific citation required unless LOFO methodology discussed |
| Bias | `Pred_vs_ECMWF_Bias`; daily S2S/ERA5 bias (`L1148-L1150`, `L1924-L1930`) | fitted vs member target, or daily S2S/ERA5 series | member axis or daily time axis depending table | \(\overline{a-b}\) | °C | Signed error tendency | Positive/negative daily errors can cancel | No |
| RMSE | `Pred_vs_ECMWF_RMSE`, `RMSE_S2S_vs_ERA5` (`L1148-L1150`, `L1924-L1930`) | fitted vs member target, or daily S2S/ERA5 series | member axis or daily time axis | \(\sqrt{\overline{(a-b)^2}}\) | °C | Magnitude of mismatch | Not absolute stage-mean difference; no sign | No |
| Correlation, R², slope | `safe_corr`; plotting scatter stats (`L1127-L1130`, `L9282-L9314`, `L9483-L9514`) | paired x/y values | members or daily values depending call | Pearson \(r\), \(r^2\), OLS slope | dimensionless r/R²; slope unit ratio | Association and display diagnostics | Association, not causation | No |

## 2. Method cards

## Stage-window member response (`y_tmax`)

### 1. Scientific purpose
This variable defines the target whose member-to-member spread is diagnosed: why ECMWF S2S members differ in stage-mean North China Tmax during June 2023 event windows.

### 2. Code evidence
- File: `stage_specific_residualized_ols_shf.py`.
- Logic: `extract_window_only(da_tmax_m, win_start, win_end, agg_func="mean")` and row field `y_tmax` (`L976-L994`).
- Output fields: `init_date`, `stage`, `member`, `y_tmax` in raw/residualized/fitted CSVs.

### 3. Current code definition
For each initialization, stage window, and ensemble member, Tmax is averaged across the stage window. The response is not an ensemble mean; each member is a separate sample. Stage windows include `Stage-I_dry`, `Stage-II_wet`, and `Total` in earlier pipeline cells, while paper MLR focuses on `Stage-I_dry` and `Total` (`PAPER_MLR_STAGES`).

### 4. Mathematical formula
Let \(m=1,\ldots,M\) index ensemble members, \(t\) index dates, and \(W_s\) be stage window \(s\). The response is
\[
y_{m,s}=\frac{1}{|W_s|}\sum_{t\in W_s}T_{m,t},
\]
where \(T_{m,t}\) is regional daily Tmax for member \(m\) on date \(t\).

### 5. Method principle
General principle: the response in a member-axis regression is a scalar outcome per ensemble member. Current implementation: the scalar is stage-mean regional Tmax, not daily Tmax and not forecast error.

### 6. Output and interpretation
Output is °C. It represents ECMWF member stage-mean Tmax. It should not be described as daily forecast skill or as a climatological anomaly unless an anomaly transformation is explicitly applied elsewhere.

### 7. Difference from adjacent methods
Different from daily error metrics, which compare daily S2S/ERA5 time series. Different from fitted values, which are model-predicted member Tmax.

### 8. Assumptions and limitations
Only the member dimension supplies the regression sample size; most fits expect 51 members. This is a single-event member-spread diagnosis.

### 9. Methods text
Chinese: 本文将每个起报日、每个阶段窗口内的华北区域平均日最高气温先沿时间求平均，得到每个集合成员一个阶段平均 Tmax 标量，作为成员轴回归的响应变量。

English: For each initialization and event window, daily regional Tmax was averaged over the window for each ensemble member. The resulting member-level stage-mean Tmax was used as the response variable in the member-axis regression analyses.

### 10. To confirm / cite
Confirm exact spatial mask and Tmax unit in the data-preprocessing section. Data-source citations are needed for ECMWF S2S and ERA5.

## Multiple linear regression (main and slim MLR)

### 1. Scientific purpose
MLR quantifies how much variation in member-level stage-mean Tmax is linearly associated with selected physical predictors across the 51-member ensemble.

### 2. Code evidence
- Main OLS: `fit_ols_lmg(df, factor_names, model_type, residualization_strategy)` (`L1132-L1160`).
- Final slim MLR: standardized predictors are created with `StandardScaler`; OLS fit and raw-equivalent coefficient export are at `L5234-L5290`.
- Key variables: `y_tmax`, `predictors`, `X_scaled`, `model.rsquared`, `model.fittedvalues`, `table_slim_model_summary.csv`, `table_slim_model_coefficients.csv`, `table_slim_fitted_members.csv`.

### 3. Current code definition
The main OLS uses raw predictor arrays with intercept. The final slim MLR fits predictors after standardization but writes raw-equivalent coefficients and intercept. The response `y_tmax` remains in °C and is not standardized in the slim MLR block. Models are fit separately for each initialization and stage. The paper figure renderer uses `Stage-I_dry` and `Total`.

### 4. Mathematical formula
For member \(m\):
\[
y_m = \beta_0 + \sum_{k=1}^{K}\beta_k X_{m,k}+\epsilon_m.
\]
For standardized fitting:
\[
Z_{m,k}=\frac{X_{m,k}-\mu_k}{\sigma_k},\quad y_m=\beta_0^*+\sum_k\beta_k^*Z_{m,k}+\epsilon_m.
\]
Raw-equivalent coefficients are
\[
\beta_k=\frac{\beta_k^*}{\sigma_k},\quad \beta_0=\beta_0^* - \sum_k \frac{\beta_k^*\mu_k}{\sigma_k}.
\]

### 5. Method principle
General principle: OLS estimates the best linear least-squares relation between predictors and response. Current implementation: final slim MLR standardizes predictors for fitting, then converts coefficients back to raw predictor units for coefficient tables and process diagnostics.

### 6. Output and interpretation
Outputs include \(R^2\), adjusted \(R^2\), RMSE in °C, fitted member Tmax in °C, residuals in °C, standardized and raw-equivalent coefficients. These quantify statistical association in the ensemble; they do not prove physical causality.

### 7. Difference from adjacent methods
MLR estimates fitted values and coefficients. LMG/Johnson/LOFO summarize predictor importance using the fitted model or related re-fits. Residualized predictors are transformations used before fitting.

### 8. Assumptions and limitations
Linear relation; sample size about 51; correlated predictors; single-event ensemble; predictor selection affects results; residualization changes predictor interpretation.

### 9. Methods text
Chinese: 对每个起报日和阶段，本文以 51 个集合成员为样本，将阶段平均 Tmax 作为响应变量，并以阶段平均物理量作为 predictors 拟合含截距的多元线性回归。最终 slim MLR 在标准化 predictor 空间中拟合，并将系数转换回原始量纲用于解释和过程图诊断。

English: For each initialization and event window, a multiple linear regression was fitted across the 51 ensemble members, with stage-mean Tmax as the response and the selected stage-mean physical diagnostics as predictors. The final slim MLR was fitted using standardized predictors, while raw-equivalent coefficients and intercepts were exported for interpretation and process diagnostics.

### 10. To confirm / cite
Confirm final predictor set from `table_slim_predictor_sets.csv` generated in the target run. Cite standard regression only if required by journal style.

## Residualized predictors

### 1. Scientific purpose
Residualization reduces specific linear overlaps among physically related predictors, such as SM with SHF or TP with SM/SHF, before member-axis attribution.

### 2. Code evidence
- `calc_resid(y_col, x_cols, df)` computes residuals from OLS with constant (`L1092-L1100`).
- Stage-I residualizes `SSR_Avg` and `SHF_Avg` against `sm_avg`; Stage-II/Total residualize `sm_avg`, `SSR_Avg`, and `SHF_Avg` against `NCHN_tp` (`L1281-L1315`).
- Residual audit writes `residualization_audit.csv` and predictor-list tables (`L1370-L1375`).

### 3. Current code definition
Residualization is performed within each init-stage member table. It uses all available member rows in that table. If the response variable has zero standard deviation, residuals are zeros; if controls have zero summed standard deviation, the demeaned variable is returned.

### 4. Mathematical formula
For a variable \(Y_m\) and controls \(C_{m,j}\):
\[
Y_m=\alpha_0+\sum_j\alpha_j C_{m,j}+u_m,
\]
\[
Y^{resid}_m=u_m=Y_m-\widehat{Y}_m.
\]

### 5. Method principle
General principle: residualization removes the part linearly explained by controls. Current implementation: residuals are produced by statsmodels OLS with intercept on the member axis.

### 6. Output and interpretation
Residuals have the same units as the original variable. A residual predictor represents the component not linearly explained by specified controls in this ensemble sample. It is not an independent causal effect.

### 7. Difference from adjacent methods
Residualization changes predictor definitions before MLR; LMG/Johnson/LOFO then evaluate importance of the transformed predictors.

### 8. Assumptions and limitations
Only linear control is removed; residuals depend on chosen controls and sample; residualization can create suppressor-like changes in correlation.

### 9. Methods text
Chinese: 为减弱特定物理量之间的线性共变，本文在成员轴上对部分 predictors 做残差化处理。例如 Stage-I 中 SHF_resid 表示 SHF 中不能由 SM 线性解释的剩余部分。

English: To reduce selected linear dependencies among predictors, some variables were residualized along the ensemble-member dimension. For example, in Stage-I, SHF_resid denotes the part of SHF not linearly explained by SM within the same initialization and stage.

### 10. To confirm / cite
Confirm in prose that residuals are semipartial diagnostics, not causal isolation.

## LMG relative importance

### 1. Scientific purpose
LMG estimates the relative share of the model-explained member-to-member Tmax variance associated with each predictor when predictors are correlated.

### 2. Code evidence
- `compute_lmg_mc(X, y, feature_names, mc_samples=2000)` (`L1102-L1122`).
- It standardizes X and y, samples random predictor orders, accumulates non-negative incremental \(R^2\), and normalizes to 100%.
- Slim MLR calls `_slim_compute_lmg_mc(..., mc_samples=2000)` (`L5243-L5244`) and writes `LMG_percent` (`L5292-L5301`).

### 3. Current code definition
The visible implementation is a Monte-Carlo approximation to order-averaged incremental \(R^2\), using 2000 random permutations. Negative increments are truncated at zero via `max(0, r2 - prev_r2)`. The resulting values are normalized to sum to 100% if total positive increment is nonzero.

### 4. Mathematical formula
For predictor order \(\pi\), let \(S_{i-1}=\{\pi_1,\ldots,\pi_{i-1}\}\). The increment for predictor \(k=\pi_i\) is
\[
\Delta R^2_k(\pi)=\max\{0, R^2(S_{i-1}\cup\{k\})-R^2(S_{i-1})\}.
\]
The code estimates
\[
LMG_k=100\times \frac{\sum_{b=1}^{B}\Delta R^2_k(\pi_b)}{\sum_j\sum_{b=1}^{B}\Delta R^2_j(\pi_b)},\quad B=2000.
\]

### 5. Method principle
General LMG averages incremental explained variance over predictor orderings. Current implementation uses random permutations rather than enumerating all orders and truncates negative increments to zero.

### 6. Output and interpretation
Output is percent relative importance. It partitions normalized explanatory importance, not degrees Celsius and not causal contribution.

### 7. Difference from adjacent methods
Johnson relative weights use an orthogonalization approach. LOFO removes predictors and evaluates performance loss. LMG averages incremental \(R^2\) over orderings.

### 8. Assumptions and limitations
Monte-Carlo approximation depends on random seed/permutation count; correlated predictors share variance; truncation of negative increments is a code-specific choice.

### 9. Methods text
Chinese: LMG 相对重要性通过随机 predictor 进入顺序的 Monte-Carlo 近似，计算每个 predictor 对模型 \(R^2\) 的平均非负增量，并归一化为百分比。

English: LMG relative importance was estimated using a Monte-Carlo approximation over random predictor entry orders. For each order, the non-negative incremental increase in model \(R^2\) was accumulated for each predictor and then normalized to percentages.

### 10. To confirm / cite
Needs original LMG/relative-importance citation. Do not cite from memory; retrieve bibliographic details separately.

## Johnson relative weights

### 1. Scientific purpose
Johnson relative weights provide an independent robustness diagnostic for predictor importance under collinearity.

### 2. Code evidence
- Method comparison table includes `johnson_pct` and is plotted in Fig.4 (`L9361-L9364`).
- Johnson validation block starts around `L5947-L6106`, including a function named `johnson_relative_weights` and method-comparison merge.

### 3. Current code definition
The accessible window confirms Johnson outputs are read as `johnson_pct` and compared with LMG/LOFO in `table_method_importance_comparison.csv`. The full algebraic details should be verified directly in the Johnson function block before final Methods wording.

### 4. Mathematical formula
General Johnson relative weights transform correlated predictors into orthogonal components, estimate explained variance in the orthogonal space, and map the weights back to original predictors. The exact current-code matrix formula must be confirmed from the full implementation block.

### 5. Method principle
General method: decompose predictor correlation structure to attribute model \(R^2\) while accounting for collinearity. Current implementation: exported `johnson_pct` is used as a robustness method alongside LMG and LOFO.

### 6. Output and interpretation
Output is percent relative weight. It is a robustness attribution measure, not causal effect or °C contribution.

### 7. Difference from adjacent methods
Unlike LMG order averaging, Johnson uses orthogonal components. Unlike LOFO, it does not remove predictors one at a time.

### 8. Assumptions and limitations
Sensitive to predictor correlation structure; requires correct implementation of orthogonal transformation; still linear and ensemble-sample based.

### 9. Methods text
Chinese: Johnson relative weights 被用作与 LMG 独立的相对重要性稳健性诊断，用于检验 predictor 重要性排序是否依赖于某一种方差分解方法。

English: Johnson relative weights were used as an independent robustness diagnostic for predictor importance, complementary to the LMG decomposition and LOFO sensitivity tests.

### 10. To confirm / cite
Needs original Johnson relative weights citation and a final check of the exact orthogonalization equations in the code.

## LOFO sensitivity

### 1. Scientific purpose
LOFO tests how model fit changes when one predictor is omitted, providing a complementary sensitivity diagnostic.

### 2. Code evidence
- Final slim MLR LOFO loop removes one predictor and refits the reduced model (`L5303-L5320`).
- Ranking and positive normalized share are computed (`L5321-L5327`).
- Cell 14 recomputes `lofo_positive_pct` from positive `lofo_delta_r2` for Fig.4 plotting (`L9349-L9355`).

### 3. Current code definition
For each predictor, the code fits a reduced model excluding that predictor, computes \(\Delta R^2\), \(\Delta Adj.R^2\), and \(\Delta RMSE\), ranks predictors by \(\Delta R^2\), and normalizes positive \(\Delta R^2\) values to percentages for plotting.

### 4. Mathematical formula
\[
\Delta R^2_k=R^2_{full}-R^2_{(-k)},
\]
\[
\Delta RMSE_k=RMSE_{(-k)}-RMSE_{full}.
\]
For plotting positive shares:
\[
LOFO^+_k=100\times\frac{\max(\Delta R^2_k,0)}{\sum_j\max(\Delta R^2_j,0)}.
\]

### 5. Method principle
General principle: a predictor is considered influential if deleting it reduces model fit. Current implementation: LOFO is a re-fit sensitivity diagnostic, not a decomposition of full-model \(R^2\).

### 6. Output and interpretation
Outputs are \(\Delta R^2\), \(\Delta Adj.R^2\), \(\Delta RMSE\), ranks, and positive normalized percentages. Negative values are possible before positive-only normalization.

### 7. Difference from adjacent methods
LOFO differs from LMG and Johnson because it tests model degradation after deletion. It can understate variables that share information with retained predictors.

### 8. Assumptions and limitations
Affected by correlated predictors and model re-fitting; deletion may allow remaining predictors to absorb shared signal; not a causal contribution.

### 9. Methods text
Chinese: LOFO 通过逐一删除 predictor 并重新拟合模型，计算完整模型与删减模型之间的 \(R^2\)、调整 \(R^2\) 和 RMSE 差异，用作补充稳健性诊断。

English: A leave-one-factor-out (LOFO) diagnostic was computed by refitting the slim MLR after removing each predictor in turn and comparing the reduced model with the full model in terms of \(R^2\), adjusted \(R^2\), and RMSE.

### 10. To confirm / cite
Usually no special citation required, but terminology should be defined clearly.

## Bias and RMSE

### 1. Scientific purpose
Bias and RMSE summarize signed and unsigned error characteristics in daily S2S/ERA5 comparisons and fitted-vs-member diagnostics.

### 2. Code evidence
- MLR fitted-vs-member metrics: `Pred_vs_ECMWF_RMSE`, `Pred_vs_ECMWF_Bias` (`L1148-L1150`).
- Daily S2S-vs-ERA5 metrics: `Bias_S2S_vs_ERA5`, `RMSE_S2S_vs_ERA5`, and OLS stage-mean diagnostics (`L1924-L1930`).

### 3. Current code definition
Bias is a signed mean difference. RMSE is square root of mean squared differences. The daily S2S-vs-ERA5 code computes daily errors before averaging. MLR RMSE is member-axis fitted-vs-`y_tmax` RMSE.

### 4. Mathematical formula
Daily bias:
\[
Bias=\frac{1}{N}\sum_{t=1}^N(F_t-O_t).
\]
Daily RMSE:
\[
RMSE=\sqrt{\frac{1}{N}\sum_{t=1}^N(F_t-O_t)^2}.
\]
Member-axis fitted RMSE:
\[
RMSE_{fit}=\sqrt{\frac{1}{M}\sum_{m=1}^M(\hat{y}_m-y_m)^2}.
\]

### 5. Method principle
Bias measures signed mean error and can cancel positive/negative errors. RMSE squares errors before averaging and penalizes larger errors.

### 6. Output and interpretation
Unit is °C. RMSE is not the absolute difference of stage means; it follows error → square → mean → square root.

### 7. Difference from adjacent methods
Member-wise RMSE would compute each member separately over time. Ensemble-mean RMSE first averages members each day, then computes daily RMSE against ERA5. The current code includes daily ensemble-mean diagnostics but exact member-wise RMSE tables are not confirmed in this window.

### 8. Assumptions and limitations
Daily time-axis RMSE and member-axis regression RMSE are different quantities and must not be mixed.

### 9. Methods text
Chinese: Bias 定义为预报与 ERA5 的有符号平均差，RMSE 定义为逐日误差平方的时间平均后开方；因此 RMSE 不是阶段平均预报与阶段平均 ERA5 之差的绝对值。

English: Bias was computed as the signed mean forecast-minus-ERA5 difference, whereas RMSE was computed by squaring daily errors, averaging them over time, and taking the square root.

### 10. To confirm / cite
If member-wise RMSE is discussed, confirm its table/field in the run outputs.

## Correlation, R², slope, p value

### 1. Scientific purpose
These statistics diagnose member-wise or daily associations used in scatter panels and QC tables.

### 2. Code evidence
- `safe_corr` (`L1127-L1130`).
- Cell 14 Fig.3 scatter uses Pearson-like correlation from numpy and model \(R^2\) from summary (`L9282-L9314`).
- Fig.5 scatter helper uses Pearson correlation, fitted line, and \(R^2=r^2\) (`L9483-L9514`).

### 3. Current code definition
Correlation is Pearson correlation when at least two non-constant samples exist. Fig.5 scatter \(R^2\) is \(r^2\). The fitted scatter line is an OLS line in x-y display space.

### 4. Mathematical formula
\[
r=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2\sum_i(y_i-\bar{y})^2}},\quad R^2=r^2
\]
for simple scatter diagnostics. Slope \(b\) is from \(y=a+bx\).

### 5. Method principle
Correlation measures linear association. Slope gives change in y per unit x in the fitted line. Current implementation uses these as diagnostics, not hypothesis-test proof.

### 6. Output and interpretation
r is dimensionless; \(R^2\) is dimensionless; slope units depend on x and y. p values appear in some helper functions but must be checked per output table.

### 7. Difference from adjacent methods
Scatter \(R^2=r^2\) is not the same as multivariate MLR \(R^2\).

### 8. Assumptions and limitations
Linear association; small member sample; outliers can matter.

### 9. Methods text
Chinese: 散点图中的相关系数和斜率仅用于描述成员间线性关联，不作为因果证明。

English: Correlation coefficients and fitted slopes in the scatter diagnostics were used to summarize member-wise linear associations and were not interpreted as causal effects.

### 10. To confirm / cite
No special citation required.

## 3. Formula and symbol table

| Symbol | Meaning |
|---|---|
| \(m\) | Ensemble member index |
| \(M\) | Number of members, expected 51 in main checks |
| \(t\) | Date index |
| \(N\) | Number of dates in a daily metric |
| \(s\) | Stage/window index |
| \(W_s\) | Set of dates in stage \(s\) |
| \(y_m\) | Member-level stage-mean Tmax response |
| \(X_{m,k}\) | Predictor \(k\) for member \(m\) |
| \(\hat{y}_m\) | MLR fitted value |
| \(R^2\) | Coefficient of determination |
| \(\Delta R^2_k\) | LOFO full-minus-reduced R² for predictor \(k\) |

## 4. Suggested Methods section structure

1. **Data and preprocessing**: ECMWF S2S, ERA5, variables, units, BJT daily aggregation, regional means.
2. **Event windows and member grouping**: initialization list, stage windows, 51 members, Hot17/Middle17/Cold17 if used.
3. **Forecast-error metrics**: Bias and RMSE definitions; distinguish daily ensemble-mean and member-axis quantities.
4. **Member-axis MLR**: response, predictors, standardization, intercept, fitted values and residuals.
5. **Residualized predictors**: stage-specific residualization and interpretation boundaries.
6. **Relative-importance attribution**: LMG, Johnson, and LOFO as three separate diagnostics.
7. **Robustness and QC**: VIF/condition number, output-path audit, consistency checks.

## 5. Code audit issue list

| Issue | Status | Note |
|---|---|---|
| Code, figure captions, and terminology consistency | Needs clarification | Display labels are standardized in Cell 14, but manuscript captions must match CSV fields. |
| Same quantity has different definitions | Potential inconsistency | `RMSE` appears both as member-axis MLR RMSE and daily S2S-vs-ERA5 RMSE; Methods must distinguish. |
| Units are consistent | Needs clarification | y and fitted values are °C; residual predictors retain original units; verify all figure labels. |
| Standardized and raw coefficients mixed | Confirmed consistent in slim block | Fit uses standardized X; raw-equivalent coefficients are exported. Methods must state this. |
| Bias/RMSE ensemble-mean vs member-level mixed | Potential inconsistency | Code has both daily ensemble-mean diagnostics and member-axis fitted RMSE. |
| Stage-I/Stage-II/Total terminology | Needs clarification | Paper MLR uses Stage-I and Total; broader pipeline also includes Stage-II. |
| Hot17/Cold17 regrouping | Not found in this MLR window | Grouping is mainly Cell 12/Fig.5–6 concern. |
| LMG/Johnson/LOFO called “contribution” | Potential inconsistency | They are relative-importance/sensitivity diagnostics, not causal contributions. |
| Partial fitted component as causal contribution | Not found in this MLR card | Addressed in process audit. |
| Legend/CSV/Methods names consistency | Needs clarification | Use display names such as “Init MSE*”, “SHF resid”, “TCC resid”. |

## 6. Not used or not confirmed in current window

- Spatial significance: not confirmed.
- Hovmöller diagnostics: not confirmed.
- Historical KDE / percentile / rank: not confirmed.
- Exact Johnson orthogonalization algebra: implementation block present but should be manually verified before final prose.
- Member-wise RMSE table: not confirmed as a dedicated output in the current scan.
