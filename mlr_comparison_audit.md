# Source and interface audit

Development start: `dev/jgr-major-revision`,
`096cad22cc84cf8b14a5ee70a5db80c762891a65` (remote verified before branch creation).
Initial local checkout: clean `work`, `297c8cc40b5c89fef4c499bf30a4adea5865b541`.
Feature branch: `feature/mlr-raw-residual-comparison`.
Source PR #10: `7d94fb1d8ca630eafb65cfa203dbaa7890c1580a`.
The development tree did not contain the four MLR files; exact file-level import
was used, without merging a historical branch. All six development entries,
including Fig.1/2 code and audits, remain unchanged.

| Historical filename | Preserved blob SHA |
|---|---|
| stage_specific_residualized_ols_shf.py | a132b0e1875dba812c2be6bceb9c423bd9d2d702 |
| stage_specific_residualized_ols_shf.ipynb | e9fa7882fbf6da3aac700b2460a6f26ec4936a6d |
| methods_audit_mlr_attribution.md | 7ca196d87d46d1a47d1bfae813409bdd02f6244c |
| methods_audit_stageI_total_process.md | 79a1e900ca224c5802598c8f974ab0424d2be2b7 |

## Audited source sections

The historical notebook has **3 physical cells** containing **14 logical sections**;
its contents and outputs are retained without restructuring.
Line references below are to the preserved `.py` source.

- Cell 8 starts at line 4870. `SLIM_PREDICTORS_BY_STAGE` at 4944 matches the user
  Stage-I/Total lists and order exactly. Stage-I raw controls SM; Total controls TP.
  `_slim_resid_series` (4982) fits intercept statsmodels OLS. Preparation (5074)
  reads stage factors and separately joins NCVI_concurrent. Member IDs originated
  as `range(51)` (994); the stricter comparison checks exactly 0–50.
- `_slim_compute_lmg_mc` (5000) uses global seed 42 and 2000 default permutations,
  nanmean/nanstd + 1e-8, sklearn LinearRegression, positive incremental R² clipping
  and normalization over accumulated increments. New code uses a local compatible
  RandomState sequence, records it and caches repeated subset R². This performance
  change preserves the method and is checked against the original pure function.
- OLS loop (5235 onward) uses independent `StandardScaler`, intercept statsmodels,
  unscaled target, and raw coefficient conversion with scale + 1e-12. LOFO retains
  the existing full-model standardized columns. Historical zero-denominator LOFO
  exported 0%; the new workflow follows the explicit user requirement of NA and
  reports undefined normalization without altering the historical source.
- Cell 9 starts at 5823. `_johnson_relative_weights` (5966) is copied verbatim as
  a safe pure function: ddof=1, correlation matrix, eigh, eigenvalue floor,
  pseudoinverse, weight threshold and sum-based percentages. Historical method
  comparison fields include `johnson_raw`, `johnson_pct`, `lofo_delta_r2` (6122).
  Original implicit ranks remain in that function; new export ranks use the user’s
  explicit stable position-based tie handling.
- Cell 13 (8257) confirms Stage-I/Total six-factor final lists and CSV interfaces.
  Its historical filters/grouped process analyses are not called or rewritten.
- Cell 14 (8831) provides 6.5-inch canvas, 600 dpi, method colors, dotted grids,
  blue scatter size22/alpha0.62, fitted gray trend and r/R²/slope box. New layouts
  follow the requested 2×2 four-init scatter and 3×4 importance organization;
  numbering is Total Fig.3/4 and Stage-I Fig.9/10. Old Fig.3–6 are untouched.

## Input and baseline interface

Primary paths are Cell 8’s two exported stage-factor and concurrent-NCVI CSVs,
under the v12 four-init root. NCVI export (4225 onward) includes window_start/end,
whose values are checked when available. There is no daily-table, v11 process-table,
NetCDF/GRIB or previous-notebook kernel dependency in the new workflow.

History schemas: summary R2/Adj_R2/RMSE/n/predictor_list; coefficients
predictor/standardized_coef/coef; fitted member/y_tmax/y_fitted/residual; LMG
predictor/LMG_percent; LOFO predictor/Delta_R2/positive_delta_R2_share_percent;
method table factor/johnson_raw/johnson_pct; predictor sets
stage/predictors_raw_name. Reproduction checks match explicit row keys and never
reuse historical scientific results in new outputs.

Only synthetic calculations and renderer validation are possible in this environment.
The provided source establishes interfaces and algorithms; it does not establish
real-data reproduction, actual raw/residualized importance differences or a paper
conclusion. Those remain explicitly marked for HPC validation.
