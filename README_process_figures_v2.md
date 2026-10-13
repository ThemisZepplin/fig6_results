# Independent Fig.5 / Fig.11 / Fig.S11 process workflow

Development base: `dev/jgr-major-revision` at
`096cad22cc84cf8b14a5ee70a5db80c762891a65`.
Before editing: clean `feature/mlr-raw-residual-comparison` at
`7f3b9df5d1ea9583fab1c77b0125d7ffcd097e62`.
New branch: `feature/process-fig5-fig11-refactor`.
No old Fig.5/6 code, notebook or historical branch is imported, edited or merged.
Source interfaces were inspected in historical `stage_specific_residualized_ols_shf.py`
(Cell 11/14) and the paired comparison modules in PR #12. The latest development
branch does not contain those modules; this new workflow is self-contained.

## Notebook and existing exports

Open `process_figures_v2.ipynb` from the repository, edit its first cell's paths,
Restart Kernel and execute cells in order: paths → source checks → projections and
numerical tables → figure rendering → report. Helpers safely import without reading,
fitting or writing. Dependencies: Python ≥3.10, numpy, pandas, matplotlib and Jupyter;
nbformat is useful for static notebook schema checks.

Required existing exports (no GRIB/NetCDF reads or upstream pipeline execution):

- v11 `hotcold_stage_group_timeseries_tables_multiinit_totalfig6/tables/daily_timeseries_all_factors.csv`:
  audited aliases `init`/`init_date`, `grouping_stage`/`stage`/`window`, `factor`, `date`,
  `era5`, `hot17_mean`, `middle17_mean`, `cold17_mean`. Exact inclusive BJT daily
  coverage is required, 11 days for Total and 5 days for the Stage-I display.
  Physical factors are TP, Z500 anomaly, SM, SHF (Stage-I), NCVI anomaly and TCC.
  `NCVI_conc` is an explicit historical daily-factor alias; its values are retained,
  not recomputed or substituted by the model's stage scalar.
- v9 `tmax_error_stage_eval_multiinit/tables/tmax_member_hotcold_group_by_stage.csv`:
  init, stage, member and hotcold_group (documented aliases accepted). For both
  06-08 and 06-12, **both Stage-I and Total** must have member IDs 0–50 exactly
  once and Hot17/Middle17/Cold17 each n=17. Membership is read, never re-ranked.
- Previously computed paired MLR output root, typically
  `/data1/huangy/fig6/NC/v12/four_init_full_pipeline_NCVIconcurrent/mlr_raw_residual_comparison`:
  `qc/comparison_manifest.json`, `tables/table_mlr_model_summary.csv`,
  `table_mlr_model_coefficients.csv`, `table_mlr_design_members.csv`,
  `table_mlr_residualization_audit.csv`. Its computation QC must be PASS and all
  four model-table hashes must match. Run the separate comparison workflow on HPC
  first if these paired raw/residualized model/scaler exports do not yet exist.

The aggregate daily table must have been created using the same window-specific
membership as the supplied group file. Its `grouping_stage` distinguishes Total
groups from Stage-I groups. Counts and window labels are checked, but an aggregate
table cannot independently prove its member-level averaging. Confirm common source
provenance on HPC; do not supply a table grouped using a different window or init.
Inputs and SHA-256 provenance are recorded; no source files are written.
Where `grouping_stage` coexists with a per-day `stage`, the explicit grouping
window takes precedence, matching the old renderer. The 14–17 June daily SM/SHF
group means must also reproduce the model-member Stage-I group means at absolute
tolerance 1e-6; incompatible grouping, units or variable definitions block projection.

## Exact diagnostic entry and interpretation

`process_timeseries_core.partial_fitted_tmax(sm_value, shf_value, model, relation)`
is the actual calculation entry, called for daily group means and each Stage-I
training member. It uses the **saved full six-factor model** for the corresponding
init/version. With standardized intercept α, standardized coefficients β and that
version's training means μ and population standard deviations σ:

`diagnostic = α + β_SM*(SM−μ_SM)/σ_SM + β_heat*(heat−μ_heat)/σ_heat`.

All other predictors are fixed at their own training means, so their standardized
terms are exactly zero. The raw heat predictor is SHF_Avg. Residualized heat is
`SHF_Avg − (a + b*sm_avg)` using the **saved Stage-I member-fitted** intercept a
and slope b. Those fixed values must reproduce the exported 51-member SHF_resid
and pass residual mean/SM orthogonality checks. No daily residual model is fitted;
no MLR, scaler or relative-importance algorithm is re-fitted.

The entire training design is checked against saved predictor order, means, ddof=0
scales, member coverage and shared response. The full standardized intercept is
used rather than the raw-coordinate intercept without its mean-state correction.
The audit exports all six coefficients/means/scales, held values, α, a/b and source
paths/hashes. Day 18 is a display projection beyond the Stage-I **training** window
(14–17 June), explicitly identified in the audit.

Raw/residualized physical daily series (SM/SHF/Z500/NCVI/TCC) are copied identically.
Tmax versus raw SM member scatter is also identical, including the groups. Only
the partial fitted diagnostic and its member scatter are evaluated separately with
each model's own parameters. Their differences indicate **parameterization
sensitivity**, not stronger/weaker land-surface physics or causal attribution.
If ERA5 SM/SHF exist, their projected diagnostic is labelled as model-informed ERA5
covariate projection, **not observed ERA5 Tmax or forecast skill**. A wholly absent
ERA5 factor is explicitly reported in tables/plots; partially missing reference
series and any missing ensemble group means fail rather than being dropped.

## New numerical outputs and figures

Long-form numerical CSVs under an independent new run root's `tables/`:

1. `table_fig5_total_daily_timeseries.csv`: init/date/Total/group/factor/value.
2. `table_fig11_stageI_daily_timeseries.csv`: init/date/Stage-I/group/version,
   physical factors and partial diagnostic, projected SHF, a/b and parameter source.
3. `table_figS11_stageI_scatter_members.csv`: init/member/Stage-I/group/version,
   y_tmax, raw SM/SHF, projected heat and partial diagnostic.
4. `table_fig11_partial_fitted_audit.csv`: init/version/predictor, all parameters,
   held values and the fixed projection rule. No means from daily display samples.

New figures, each PDF with embedded TrueType text and fixed-width 600-dpi PNG:

- `Fig5_Total_daily_0608_0612_process_v2.{png,pdf}`: 5×2, rows TP/Z500/SM/NCVI/TCC,
  14–24 June BJT; columns 8 and 12 June; respective Total groups.
- `Fig11_StageI_daily_{raw,residualized}_0608_0612_process_v2.{png,pdf}`: 6×2,
  rows partial diagnostic/SM/Z500/SHF/NCVI/TCC, 14–18 June BJT; respective Stage-I
  groups, dashed boundary at 17 June 12:00 (between the 17/18 calendar points).
- `FigS11_StageI_scatter_{raw,residualized}_0608_0612_process_v2.{png,pdf}`: 2×2,
  rows 8 and 12 June; columns raw SM vs Tmax and partial diagnostic vs Tmax.
  Both axes show member absolute values, all 51 members, no group subsetting.

Daily colors: ERA5 black, Hot17 red, Middle17 blue, Cold17 green; a single figure
legend replaces repeated panel legends. Each factor uses a common, data-derived
y range across both inits and, for Fig.11, both versions. Regular MaxNLocator ticks
are visible in both columns; boxes have equal physical width/height per figure.
All five figures are QC-checked before export; crop/collision/box failures raise.
PNG width is 3900 px at 6.5 inches, with no tight-canvas cropping.

Default new run root: v12 root `/process_figures_v2/run_001`. Choose a fresh run
directory each time. Nonempty run roots and existing figure names are refused,
preserving both historical figures and earlier new runs. `qc/process_manifest.json`
records input/table hashes; rendering checks these hashes without rewriting tables.
`qc/process_figure_qc.json` records figure hashes, dimensions and layout checks.

## Caption drafts

**Fig.5.** Daily North China process evolution over 14–24 June 2023 (inclusive BJT)
for initializations on 8 June (left) and 12 June (right). Rows show precipitation,
500-hPa height anomaly, soil moisture, cold-vortex anomaly and total cloud cover.
Ensemble means use each initialization's Total-window Hot17/Middle17/Cold17 groups;
black denotes the ERA5 reference where available. This figure has one physical-
variable representation, with no raw/residualized model distinction.

**Fig.11.** Stage-I process evolution and SM–SHF partial fitted Tmax diagnostic
for 8 and 12 June initializations, shown over 14–18 June (BJT). The dashed line
marks the boundary between 17 and 18 June; model training uses 14–17 June only.
Physical-variable group means use fixed Stage-I groups and are identical between
versions. The diagnostic uses each full model's saved intercept, standardized
coefficients and training scaler, holding all other predictors at their training
means. Residualized daily SHF uses a fixed stage-member residual relation.
An ERA5-covariate projection, if available, is a model diagnostic rather than
observed Tmax. Version differences describe parameterization sensitivity.

**Fig.S11.** Stage-I member-wise absolute Tmax versus raw soil moisture (left) and
the SM–SHF partial fitted diagnostic (right), for 8 June (top) and 12 June (bottom).
Colors identify the 17-member Stage-I groups; all 51 members are retained. The SM
scatter is identical in the two versions. Diagnostic coordinates use each version's
own full-model parameters and fixed mean-state rule; no causal strength comparison
or independent prediction/forecast-skill claim is implied.

## Validation status

HPC inputs are not present in the Codex environment. Per the task constraint,
only code/static checks were performed: AST, notebook schema and import-side-effect
inspection. No diagnostic values, numerical CSVs or new scientific figures were
generated. No synthetic paper results are supplied. Pending HPC checks: actual
export schemas/units and grouping provenance; model/scaler/residual replay; version
equality gates; daily/date/member completeness; all five actual PNG/PDF exports,
shared axes and final text/legend readability. The layout QC is implemented but
has not been certified on real data.
