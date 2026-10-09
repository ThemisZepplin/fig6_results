# Methods audit: Stage-I / Total process diagnostics and SM–SHF fitted components

Scope: this audit is based only on the accessible `stage_specific_residualized_ols_shf.py` window, especially Cell 12 and Cell 14 process-figure logic. It does not modify code or re-run the data pipeline.

## 0. Files scanned

- `stage_specific_residualized_ols_shf.py`

## 1. Method summary table

| Method | Code location | Input | Sample dimension | Formula | Output unit | Main purpose | Interpretation boundary | Needs citation |
|---|---|---|---|---|---|---|---|---|
| Hot17/Middle17/Cold17 grouping | group file path and stage-specific group use (`L7897-L7933`, Cell 14 group columns `L8963-L8964`) | fixed grouping CSV | 17/17/17 members per init-stage expected | sort-based grouping not recalculated in current window | group labels | Compare hot/middle/cold member composites | Groups are target-conditioned and not independent samples | No, but define clearly |
| Fig.5 raw process time series | `_paper_raw_timeseries_panel`; Cell 14 `_raw_panel` (`L7439-L7459`, `L9422-L9438`) | `daily_timeseries_all_factors.csv` | daily group means | plot `era5`, `hot17_mean`, `middle17_mean`, `cold17_mean` | variable-specific | Show process evolution by group | Composite diagnostics; not causal proof | No |
| Fig.6 Total process chain | Cell 14 `_render_fig6` (`L9536-L9560`) | raw daily time-series table | daily group means | same group mean plotting over Total range | variable-specific | Show TP/circulation/SM/NCVI/TCC evolution in Total period | Descriptive group-composite process view | No |
| SM–SHF centered contribution | `_add_lsm_variables` (`L9403-L9420`) and Cell 12 audit rows (`L8026-L8107`) | slim coefficients, Stage-I member predictors | 51 members per init | \(\beta_{SM}(SM-\bar{SM})+\beta_{SHF}(SHF_{resid}-\overline{SHF_{resid}})\) | °C when raw-equivalent coefficients and y in °C | Diagnose SM–SHF member deviations relative to mean predictor state | Not standalone causal warming | No; regression methods citation maybe |
| MLR baseline at mean state | `_add_lsm_variables` (`L9407-L9418`) | raw-equivalent coefficients and member predictor means | per init-stage | \(\beta_0+\sum_k\beta_k\bar{X}_k\) | °C | Anchor partial fitted component in Tmax space | Baseline is full-model mean-state fitted value, not SM–SHF effect | No |
| SM_SHF_partial_fitted_Tmax | `_add_lsm_variables` (`L9418-L9419`) | baseline plus centered contribution | 51 members per init | baseline + centered SM–SHF contribution | °C | Display partial fitted Tmax component | Not full fitted Tmax; other predictors held at mean reference | No |
| Raw/with-intercept SM–SHF audit variables | Cell 12 audit fields (`L8026-L8107`) | member-level audit table | 51 members per init | raw, centered, intercept-inclusive variants | °C or coefficient-derived units | QC of contribution definition and scale | raw and with-intercept versions not main figure variables | No |
| Hot17 − Cold17 difference | Cell 12 QC/summary fields (`L8014-L8017`, `L8081-L8084`) | group means | group means per init | \(\bar{x}_{Hot17}-\bar{x}_{Cold17}\) | variable-specific | Quantify group contrast | Composite contrast, not causal attribution | No |
| Bias/RMSE process diagnostics | daily metric block (`L1924-L1930`) | daily S2S/ERA5 or OLS stage means | daily time axis | signed mean error; square-mean-root error | °C | Forecast error summary | Different from member-axis MLR RMSE | No |
| Historical/KDE/spatial/Hovmöller | not found in current scan | — | — | — | — | — | Not in this window | If used elsewhere, cite separately |

## 2. Method cards

## Hot17 / Middle17 / Cold17 grouping

### 1. Scientific purpose
The grouping separates ensemble members by their predicted temperature state, enabling process composites that contrast hotter, middle, and colder members.

### 2. Code evidence
- Stage-I/Total grouping checks and QC in Cell 12 (`L7897-L7933`).
- Group colors/order in Cell 14: `_STANDALONE_GROUP_ORDER = ["Hot17", "Middle17", "Cold17"]` (`L8963-L8964`).
- Fig.5 uses Stage-I grouping and Fig.6 uses Total grouping in Cell 14 renderers (`L9505-L9560`).

### 3. Current code definition
The current window uses an external fixed grouping table; it does not recompute groups. QC expects Hot17, Middle17, and Cold17 groups and checks member counts. Fig.5 is Stage-I-focused, while Fig.6 uses Total-period grouping. The grouping is fixed for plotting within each figure context.

### 4. Mathematical formula
If members are sorted by a target scalar \(q_m\), groups are conceptually:
\[
Hot17=\{m: q_m\text{ in highest 17}\},\quad Cold17=\{m: q_m\text{ in lowest 17}\},
\]
with Middle17 the remaining middle members. The actual sorting operation is not recalculated in the current Cell 14 code.

### 5. Method principle
Composite grouping compares conditional subsets of ensemble members. Current implementation consumes fixed labels and does not perform new sorting.

### 6. Output and interpretation
Output is group label. Group contrasts show associations with hot/cold predicted outcomes and should not be interpreted as independent experimental treatments.

### 7. Difference from adjacent methods
Grouping is descriptive/composite; MLR is a regression model; LMG/Johnson/LOFO are model-importance diagnostics.

### 8. Assumptions and limitations
Groups are target-conditioned; sample size per group is 17; grouping stage matters and must be stated.

### 9. Methods text
Chinese: 本文使用预先生成的 Hot17、Middle17 和 Cold17 成员分组进行过程合成分析。Fig.5 使用 Stage-I 分组，Fig.6 使用 Total 时段分组；本窗口不重新计算成员分组。

English: Hot17, Middle17, and Cold17 member groups were read from a fixed grouping table rather than recalculated in the plotting cell. Fig.5 used Stage-I grouping, whereas Fig.6 used Total-period grouping.

### 10. To confirm / cite
Confirm in the grouping-generation script which target scalar and tie-handling rule were used.

## Fig.5 and Fig.6 raw process time series

### 1. Scientific purpose
These panels diagnose how key physical variables evolve across hot, middle, and cold member groups during Stage-I and Total windows.

### 2. Code evidence
- Cell 12 raw panel uses `era5`, `hot17_mean`, `middle17_mean`, `cold17_mean` (`L7439-L7459`).
- Cell 14 `_raw_panel` uses the same line fields and fixed date limits (`L9422-L9438`).
- Required raw daily table columns are checked (`L9195-L9206`).

### 3. Current code definition
For each init, stage, factor, and date, the renderer reads daily values from `daily_timeseries_all_factors.csv`. It plots ERA5 where available and group means for Hot17, Middle17, and Cold17. Fig.5 uses 2023-06-14 to 2023-06-18; Fig.6 uses 2023-06-14 to 2023-06-24. A Stage-I boundary line is displayed.

### 4. Mathematical formula
For group \(G\) and factor \(X\):
\[
\bar{X}_{G,t}=\frac{1}{|G|}\sum_{m\in G}X_{m,t}.
\]
The plotted lines are \(\bar{X}_{Hot17,t}\), \(\bar{X}_{Middle17,t}\), \(\bar{X}_{Cold17,t}\), and ERA5 when available.

### 5. Method principle
Composite time series summarize conditional group evolution. Current implementation reads precomputed daily group means and does not recompute the daily member values.

### 6. Output and interpretation
Units are variable-specific: TP in mm day⁻¹, SM in m³ m⁻³, SHF in W m⁻², Z500 anomaly in gpm, NCVI anomaly in PVU, TCC fraction. These are descriptive diagnostics.

### 7. Difference from adjacent methods
Raw time series show daily evolution; stage-mean MLR predictors collapse windows to one scalar per member.

### 8. Assumptions and limitations
Group means hide within-group spread; grouping depends on target outcome; daily curves are not causal pathways.

### 9. Methods text
Chinese: 过程图直接读取预先计算的日尺度组均值，分别绘制 ERA5（若存在）以及 Hot17、Middle17 和 Cold17 组均值，用于描述不同成员组的过程演变。

English: Process time-series panels were rendered from exported daily group-mean tables. ERA5 values, where available, and Hot17, Middle17, and Cold17 means were plotted to diagnose the temporal evolution of physical variables in each group.

### 10. To confirm / cite
Confirm the upstream daily aggregation rules for each variable in the data-preprocessing Methods.

## SM–SHF centered contribution

### 1. Scientific purpose
This diagnostic isolates the linear SM–SHF part of the Stage-I slim MLR relative to each initialization’s mean predictor state, to interpret land-surface thermal pathway differences among members.

### 2. Code evidence
- `_stagei_coef_info` reads raw-equivalent coefficients `beta_SM`, `beta_SHF`, intercept, means (`L9388-L9400`).
- `_add_lsm_variables` computes `LSM_SHF_partial_centered` (`L9403-L9406`).
- Cell 12 audit writes member and summary fields for `LSM_SHF_partial_centered` (`L8026-L8107`).

### 3. Current code definition
For each init, the code reads Stage-I slim MLR raw-equivalent coefficients for `sm_avg` and `SHF_resid`. It centers each member’s SM and SHF_resid by the per-init Stage-I ensemble mean and forms a linear combination. The response target is documented in QC as Stage-I member mean Tmax in °C.

### 4. Mathematical formula
For member \(m\):
\[
C_m=\beta_{SM}(SM_m-\bar{SM})+\beta_{SHF}(SHF^{resid}_m-\overline{SHF^{resid}}).
\]
Here \(\beta_{SM}\) and \(\beta_{SHF}\) are raw-equivalent slim MLR coefficients, and bars denote per-init Stage-I ensemble means.

### 5. Method principle
General principle: a partial linear component evaluates selected terms of a fitted linear model. Current implementation centers predictors to express deviations from the mean predictor state.

### 6. Output and interpretation
Unit is °C if coefficients are raw-equivalent and the response is absolute Tmax in °C. A zero value means the member is at the ensemble-mean SM and SHF_resid reference state for those two predictors. It is not a causal SM–SHF warming estimate.

### 7. Difference from adjacent methods
Centered contribution excludes intercept/baseline. The partial fitted component adds a full-model mean-state baseline. Raw contribution is uncentered and used for QC, not the main interpretation.

### 8. Assumptions and limitations
Depends on linear MLR, predictor centering, and raw-equivalent coefficient correctness; other predictors are not represented in this centered term.

### 9. Methods text
Chinese: Stage-I 的 SM–SHF centered contribution 定义为 `sm_avg` 与 `SHF_resid` 相对该起报日 Stage-I 成员平均状态的线性偏离项，并使用 final slim MLR 的原始量纲等价系数计算。

English: The Stage-I centered SM–SHF contribution was computed as the sum of the SM and SHF_resid linear terms relative to the per-initialization Stage-I ensemble-mean predictor state, using the raw-equivalent coefficients from the final slim MLR.

### 10. To confirm / cite
Confirm that `table_slim_model_coefficients.csv` always contains raw-equivalent coefficients for all inits and stages.

## MLR baseline at mean state

### 1. Scientific purpose
The baseline places the centered SM–SHF deviation on the fitted Tmax scale by anchoring it to the full slim MLR prediction at the mean predictor state.

### 2. Code evidence
- `_add_lsm_variables` initializes baseline from intercept and adds all available predictor mean terms (`L9407-L9418`).
- Audit fields include `MLR_baseline_at_mean_state`, `baseline_predictor_count`, and `baseline_predictor_list` (`L8041-L8044`, `L8071-L8077`).

### 3. Current code definition
For each init, baseline begins with the raw-equivalent intercept and adds \(\beta_k\bar{X}_k\) for each predictor present in the member table. Missing predictors are recorded.

### 4. Mathematical formula
\[
B=\beta_0+\sum_{k\in K}\beta_k\bar{X}_k.
\]

### 5. Method principle
A linear model prediction at mean predictor values provides a reference fitted level. Current implementation uses all available slim-model raw-equivalent coefficients, not only SM and SHF_resid.

### 6. Output and interpretation
Unit is °C. Baseline is the model’s fitted Tmax at the mean predictor state. It is not an effect of SM or SHF alone.

### 7. Difference from adjacent methods
Baseline is a reference level; centered contribution is a deviation; partial fitted component is their sum.

### 8. Assumptions and limitations
Only predictors available in the audit/member table are included; missing predictors are recorded and require checking.

### 9. Methods text
Chinese: mean-state baseline 定义为完整 slim MLR 在该起报日 Stage-I predictor 均值处的拟合 Tmax，用于将 centered SM–SHF 项放回温度空间。

English: The mean-state baseline was defined as the full slim MLR prediction evaluated at the per-initialization Stage-I ensemble-mean predictor values.

### 10. To confirm / cite
Confirm no missing predictors in the baseline predictor list for final figures.

## SM_SHF_partial_fitted_Tmax

### 1. Scientific purpose
This quantity shows the fitted Tmax component obtained by adding SM–SHF deviations to the full-model mean-state baseline.

### 2. Code evidence
- `_add_lsm_variables` computes `SM_SHF_partial_fitted_Tmax = baseline + LSM_SHF_partial_centered` (`L9418-L9419`).
- Cell 12 summary writes `partial_fitted_*` fields and group means (`L8078-L8087`).
- Cell 14 displays it in Fig.5 fitted-component version (`L9524-L9536`).

### 3. Current code definition
Per member and init, the component equals the baseline at mean predictor state plus the centered SM–SHF contribution.

### 4. Mathematical formula
\[
F^{SM-SHF}_m=B+C_m.
\]
Substituting:
\[
F^{SM-SHF}_m=\beta_0+\sum_k\beta_k\bar{X}_k+\beta_{SM}(SM_m-\bar{SM})+\beta_{SHF}(SHF^{resid}_m-\overline{SHF^{resid}}).
\]

### 5. Method principle
General principle: partial fitted component combines a reference fitted state with selected predictor deviations. Current implementation holds all non-SM/SHF terms at their mean-state contribution.

### 6. Output and interpretation
Unit is °C. It is a diagnostic partial fitted Tmax component, not the full fitted Tmax and not a causal SM–SHF-only temperature effect.

### 7. Difference from adjacent methods
Different from centered contribution because it includes the baseline. Different from full fitted value because only SM and SHF_resid vary around the baseline.

### 8. Assumptions and limitations
Depends on MLR linearity and coefficient validity; other predictors are fixed at reference mean state by construction.

### 9. Methods text
Chinese: `SM_SHF_partial_fitted_Tmax` 是将 centered SM–SHF contribution 加到完整 slim MLR 的 mean-state baseline 后得到的诊断量，表示在其他 predictors 固定于参考均值时 SM 与 SHF_resid 偏离对应的部分拟合 Tmax。

English: The SM–SHF partial fitted Tmax was defined as the sum of the full-model mean-state baseline and the centered SM–SHF contribution. It represents a diagnostic partial fitted quantity with non-SM–SHF predictors held at their mean-state reference values.

### 10. To confirm / cite
Confirm whether the manuscript should show centered contribution, fitted component, or both.

## Raw and with-intercept SM–SHF audit variables

### 1. Scientific purpose
These variables audit whether centering and intercept handling change interpretation.

### 2. Code evidence
Cell 12 writes `LSM_SHF_partial_raw`, `LSM_SHF_partial_centered`, `LSM_SHF_with_intercept`, and `StageI_Tmax` fields (`L8026-L8052`). Summary fields include min/max/mean and correlations (`L8088-L8101`).

### 3. Current code definition
The raw term uses uncentered predictor values, the centered term subtracts predictor means, and the with-intercept term adds the intercept to the raw two-predictor term.

### 4. Mathematical formula
\[
R_m=\beta_{SM}SM_m+\beta_{SHF}SHF^{resid}_m,
\]
\[
C_m=\beta_{SM}(SM_m-\bar{SM})+\beta_{SHF}(SHF^{resid}_m-\overline{SHF^{resid}}),
\]
\[
I_m=\beta_0+R_m.
\]

### 5. Method principle
These are algebraic variants used for QC, not separate models.

### 6. Output and interpretation
Raw and with-intercept versions are QC quantities. The centered term is preferred for relative contribution around the mean state.

### 7. Difference from adjacent methods
With-intercept includes a model baseline term and must not be described as pure SM–SHF contribution.

### 8. Assumptions and limitations
Same as the underlying MLR; interpretation sensitive to centering and intercept choice.

### 9. Methods text
Chinese: 为避免将截距或未中心化常数项误解释为 SM–SHF 贡献，代码同时输出 raw、centered 和 with-intercept 三种代数形式作为 QC。

English: Raw, centered, and intercept-inclusive SM–SHF terms were exported as QC diagnostics to distinguish the member-varying centered contribution from constant baseline terms.

### 10. To confirm / cite
No literature citation; ensure figure captions identify the displayed variant.

## Hot17 − Cold17 differences

### 1. Scientific purpose
Hot-minus-cold differences summarize group contrasts in variables or partial fitted quantities.

### 2. Code evidence
Cell 12 QC lines and summary rows compute Hot17 and Cold17 means and differences for centered and fitted components (`L8014-L8017`, `L8081-L8084`).

### 3. Current code definition
For each init and variable, group means are calculated and subtracted.

### 4. Mathematical formula
\[
D_X=\frac{1}{17}\sum_{m\in Hot17}X_m-\frac{1}{17}\sum_{m\in Cold17}X_m.
\]

### 5. Method principle
Composite difference quantifies contrast between conditional groups.

### 6. Output and interpretation
Units follow the variable. It does not imply causality and depends on grouping criteria.

### 7. Difference from adjacent methods
Different from regression coefficients; it is a group-composite statistic.

### 8. Assumptions and limitations
Small group size; target-conditioned groups; no independent random assignment.

### 9. Methods text
Chinese: Hot17−Cold17 差值定义为 Hot17 组平均减去 Cold17 组平均，用于描述两类成员的过程量差异。

English: Hot17-minus-Cold17 differences were computed as the difference between the Hot17 and Cold17 group means for each diagnostic variable.

### 10. To confirm / cite
No citation required.

## Bias and RMSE in process diagnostics

### 1. Scientific purpose
Bias and RMSE describe S2S/ERA5 forecast-error behavior and OLS stage-mean diagnostic mismatch.

### 2. Code evidence
Daily bias/RMSE are computed at `L1924-L1930`. The code comment states OLS stage-mean RMSE is an external consistency diagnostic, not daily forecast skill (`L1928-L1930`).

### 3. Current code definition
Bias is signed mean difference. RMSE is square root of mean squared daily differences. OLS stage-mean diagnostics are not daily forecast skill.

### 4. Mathematical formula
\[
Bias=\frac{1}{N}\sum_t(F_t-O_t),\quad RMSE=\sqrt{\frac{1}{N}\sum_t(F_t-O_t)^2}.
\]

### 5. Method principle
Bias captures sign; RMSE captures magnitude.

### 6. Output and interpretation
Unit °C. Signed bias can cancel daily errors; RMSE is not absolute stage-mean difference.

### 7. Difference from adjacent methods
Member-axis MLR RMSE is computed across members, not dates.

### 8. Assumptions and limitations
Must clearly state whether the averaging axis is daily time or ensemble members.

### 9. Methods text
Chinese: Bias 和 RMSE 分别作为有符号误差与误差幅度指标；代码中明确区分日尺度预报误差与 OLS 阶段均值外部一致性诊断。

English: Bias and RMSE were used as signed and magnitude-based error diagnostics, with daily forecast-error metrics kept distinct from member-axis MLR fit diagnostics.

### 10. To confirm / cite
Confirm which error tables are used in the final paper.

## 3. Formula and symbol table

| Symbol | Meaning |
|---|---|
| \(m\) | ensemble member |
| \(G\) | member group, e.g., Hot17 |
| \(t\) | date |
| \(X_{m,t}\) | daily member value of variable X |
| \(\bar{X}_{G,t}\) | group mean at date t |
| \(SM_m\) | Stage-I member mean soil moisture |
| \(SHF^{resid}_m\) | Stage-I SHF residual predictor |
| \(\beta_{SM},\beta_{SHF}\) | raw-equivalent slim MLR coefficients |
| \(B\) | full-model mean-state baseline |
| \(C_m\) | centered SM–SHF contribution |
| \(F^{SM-SHF}_m\) | partial fitted Tmax component |

## 4. Suggested Methods section structure

1. **Data and preprocessing**: BJT daily values, regions, variables, unit conversions.
2. **Event windows and ensemble-member grouping**: Stage-I/Total definitions; fixed Hot17/Middle17/Cold17 group labels.
3. **Group-composite process diagnostics**: daily group means and Hot17−Cold17 differences.
4. **MLR-derived SM–SHF diagnostics**: coefficient source, centered contribution, mean-state baseline, partial fitted component.
5. **Interpretation boundaries and QC**: no causal language, group dependence, coefficient-space checks.

## 5. Code audit issue list

| Issue | Status | Note |
|---|---|---|
| Code, figure captions, and terminology consistency | Needs clarification | Fig.5/6 display names differ from raw field names; Methods should map both. |
| Same quantity has different definitions | Potential inconsistency | SM–SHF appears as centered contribution and partial fitted component; captions must specify which. |
| Units are consistent | Needs clarification | Partial fitted component is °C if raw-equivalent coefficients and y in °C; verify coefficient table in run. |
| Standardized and raw coefficients mixed | Confirmed consistent in current code | Code records raw-equivalent coefficient space for Fig.5. |
| Bias/RMSE ensemble-mean vs member-level mixed | Potential inconsistency | Must distinguish daily and member-axis metrics. |
| Stage-I, Stage-II, Total terminology | Needs clarification | Fig.5 Stage-I and Fig.6 Total use different grouping stages. |
| Hot17/Cold17 re-grouping | Confirmed consistent in renderer | Cell 14 reads fixed group/time-series outputs and does not regroup. |
| LMG/Johnson/LOFO called contribution | Not central in this process window | Use “relative importance” for model methods. |
| Partial fitted component as causal contribution | Potential inconsistency | Must call it diagnostic partial fitted quantity, not causal effect. |
| Legend/CSV/Methods names consistency | Needs clarification | Ensure `NCVI_conc` is described as NCVI anomaly and `TCC_Avg` as TCC. |

## 6. Not used or not confirmed in current window

- Spatial significance: not found.
- Hovmöller: not found.
- Propagation diagnosis: not found.
- Daily synoptic composites: not confirmed beyond process time series.
- Historical climatology, KDE, percentile, historical rank: not found.
- Exact upstream grouping sort variable and tie handling: not recalculated in this window; confirm from grouping-generation code.
