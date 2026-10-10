# English caption drafts

Each figure number has separate raw and residualized versions with the indicated
filename suffix. These drafts describe definitions, not scientific conclusions.

**Figure 3 — Total-period member regression.** Window-mean absolute North China
daily maximum 2-m temperature (Tmax, °C) for 14–24 June 2023 (inclusive, BJT)
from each of 51 ECMWF ensemble members versus its MLR fitted value. Panels (a–d)
show initializations on 1, 5, 8 and 12 June. The raw version uses the six original
predictor columns; the residualized version substitutes residual TCC and SM after
separate intercept regressions against TP while retaining TP in the full model.
Trend lines, Pearson r, R² and slope are **in-sample member-regression diagnostics**,
not independent predictive validation or forecast skill against ERA5. Corresponding
raw and residualized panels use identical axes. All models include an intercept
and independently standardized predictors; Tmax is not standardized.

**Figure 4 — Total-period relative importance and supplementary diagnostics.**
LMG (top row, green), Johnson relative weights (middle row, blue), and normalized
positive LOFO ΔR² (bottom row, orange) for the 51-member regressions in Figure 3
(14–24 June 2023, inclusive BJT). Columns show 1, 5, 8 and 12 June initializations.
LMG is the primary relative-importance indicator: averaged positive incremental
R² over 2000 variable-entry permutations (seed42), normalized to 100%. Johnson
percentages normalize the original relative weights; their unnormalized sum is
checked against full-model R². LOFO deletes one fixed full-model column at a time
and normalizes positive full-minus-reduced R² across factors. LOFO percentages
are not LMG contributions; the three percentage definitions are not equivalent
variance decompositions. Johnson and LOFO provide supplementary robustness
diagnostics. All rows and both versions share each column’s residualized LMG
position order. Bold one-decimal labels are visual emphasis, not significance.
Raw versus residualized labels distinguish original SM/TCC from SM resid/TCC resid.
TP is window-accumulated North China precipitation; SM is mean soil moisture;
TCC is mean total cloud cover; Z500 is the North China 500-hPa geopotential-height
anomaly; NCVI is the stage-concurrent cold-vortex diagnostic; Init MSE* denotes
Initial_MSEstar_max_NCHN (antecedent thermodynamic predictor). Regression
associations are not interpreted as causal contributions.

**Figure 9 — Stage-I member regression.** As in Figure 3, for 14–17 June 2023
(inclusive, BJT) and all 51 ECMWF members per initialization. The response is
window-mean absolute North China daily Tmax (°C). Panels (a–d) correspond to
1, 5, 8 and 12 June. Raw predictors retain SHF and TCC; the residualized version
replaces them by SHF resid and TCC resid from separate intercept regressions
against SM, while retaining SM in the full model. Other predictor positions are
unchanged. The fitted trend and r/R²/slope describe in-sample member regression
only; they are not independent predictive performance or ERA5 forecast skill.
Corresponding raw/residualized panels share limits and ticks.

**Figure 10 — Stage-I relative importance and supplementary diagnostics.**
As in Figure 4 for Stage-I (14–17 June 2023, inclusive BJT), with 51 ECMWF
members, window-mean absolute Tmax as response, and initializations on 1, 5, 8
and 12 June. Rows display LMG (primary indicator, green), Johnson relative weights
(blue) and normalized positive LOFO ΔR² (orange). Raw and residualized versions
use the same predictor positions and each column’s residualized LMG display order.
SHF denotes window-mean sensible heat flux; SHF resid and TCC resid are the
parts remaining after intercept regression on SM. Remaining abbreviations follow
Figure 4. LMG averages positive incremental R² across the common 2000-permutation
sequence; Johnson normalizes its relative weights; LOFO normalizes positive R²
losses after deletion of fixed columns. These percentages have distinct meanings,
and positive-normalized LOFO ΔR² is not a LMG contribution. Johnson and LOFO are
supplementary diagnostics; bold labels are visual emphasis only. No causal
interpretation, significance claim or preferred model representation is implied.
