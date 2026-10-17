## Fig.4/Fig.10 bottom x-axis revision (v5)

This styling-only revision restores a visible bottom spine in all 12 LMG, Johnson, and LOFO importance subpanels. Its line is 0.8 pt in gray 0.25; numerical bottom ticks remain outward and visible. Top/right/left spines remain hidden (no four-sided box). Bar lengths, percentage and ΔR² labels, ordering, typography, 0.6-pt white halo and native PDF text extraction are unchanged. Fig.3 and Fig.9 scatter renderers are unchanged. Outputs use `lofo_deltaR2_v5` filenames and corresponding QC/manifest, preserving the previous v4 results; use the updated `mlr_comparison_figures.ipynb` with a restarted kernel. The corresponding synthetic axis assertions are in `tests/test_mlr_comparison_plotting.py`. Real HPC figure exports and PDF QA remain to be rerun.

# Final MLR raw versus residualized comparison

PDF QA compatibility revision (v4): the initial v3 white-stroke text path effect
converted some inside value labels into PDF vector outlines, causing
PyMuPDF to miss 5 labels (Fig10 observed 44/48 percentages and 23/24 LOFO
values). A layered text artist now paints the 0.6 pt white stroke first
and a native editable black text glyph second; v4 keeps numeric bars and
layout unchanged, retains every strict PDF QA threshold, and uses fresh
`lofo_deltaR2_v4` output filenames to preserve prior v3 files.
The updated source/tests are committed; actual HPC PDF QC remains pending.


The latest Fig.4/10 **v3 halo-only** microrevision draws a 0.6 pt white
Matplotlib path-effect stroke behind **inside-centered** black bar-value labels.
Outside/end-adjacent labels have no halo, and no value label uses a white
background box. Positioning, encoded bar lengths, numerical results, model
calculations and Fig.3/9 remain unchanged. The output files and
revision-specific QC/manifest now use `lofo_deltaR2_v3`; existing v2 outputs
are not overwritten. The notebook version guard rejects stale v2 imports.
Synthetic test code has been updated but HPC end-to-end rendering/PDF QC
has not yet been executed for v3.


The current Fig.4/10 microrevision keeps 8 pt factor names and applies semibold
only in the first column. It exports `_layout_v3` figure filenames and
`figure_layout_v3_qc.json` / `table_figure_layout_v3_manifest.csv`; the v2 figures
and QC described below remain preserved for comparison. All other layout and
numerical definitions are unchanged.

Open `mlr_raw_residual_comparison.ipynb`, configure paths in its first code cell,
then Restart Kernel → Run All. After numerical QC passes, open
`mlr_comparison_figures.ipynb` and Restart Kernel → Run All with the same paths.
The repository should be the Jupyter working directory; otherwise set
`CODE_ROOT` (or `MLR_CODE_ROOT`) to the cloned repository directory.

Dependencies: Python ≥3.10, numpy, pandas, scipy, scikit-learn, statsmodels,
matplotlib, Jupyter. See `requirements_mlr_comparison.txt`; nbformat and pytest
are used for schema validation and synthetic tests; nbclient/ipykernel support the
optional fresh-kernel Notebook tests. Helpers are safe to import:
no CSV reading, fitting, writes or global matplotlib style changes happen at import.
No original pipeline import, `%run`, raw GRIB/NetCDF access or hidden kernel state
is required. Original notebooks and outputs are preserved.

## HPC paths and inputs

Default `INPUT_ROOT`:
`/data1/huangy/fig6/NC/v12/four_init_full_pipeline_NCVIconcurrent`.
Default `OUTPUT_ROOT`: `INPUT_ROOT/mlr_raw_residual_comparison`.
Edit notebook variables or set `MLR_INPUT_ROOT` and `MLR_OUTPUT_ROOT` before Jupyter.
The computation notebook reads these two existing exports under `INPUT_ROOT/tables/`:

- `table_stage_member_raw_factors_TCC.csv`: `init_date`, `stage`, `member`,
  `y_tmax`, `sm_avg`, `SHF_Avg`, `TCC_Avg`, `NCHN_tp`, `z500_anom_NCHN`,
  `Initial_MSEstar_max_NCHN`.
- `table_ncvi_concurrent_stage_values.csv`: `init_date`, `stage`, `member`,
  `NCVI_concurrent`. Existing `ncvi_concurrent_window_start/end` metadata is checked
  against the inclusive BJT windows. A raw-table NCVI column is never used instead
  of this table. Export schemas lacking date metadata cannot independently establish
  window provenance; use the audited original exports.

Stage names in historical CSVs are `Stage-I_dry` and `Total`; new output `window`
names are `Stage-I` and `Total`. Stage-II rows are ignored and never fitted.
Each selected group must contain member IDs **0–50 exactly once**, in both tables;
all required numerical fields must be finite. Rows are paired by
`init_date, stage, member`, sorted by member. No silent sample deletion occurs.
`y_tmax` retains its historical definition: North China member window-mean absolute
daily Tmax (°C), not ERA5 error, a standardized target or a Hot17/Cold17 subset.

Optional historical CSVs are all seven listed in `core.HISTORY_FILES` (predictor
sets, model summary, coefficients, fitted members, LMG, LOFO and method comparison).
If present, predictor order must match before calculation. History comparison
checks every available metric/member/coefficient/importance field; missing files
or fields are explicitly `NOT_AVAILABLE`, giving `INCOMPLETE`, never reproduction
success. A mismatched available baseline gives `FAIL` and blocks figure export.
Historical const standardized coefficient was NA; standardized intercept is saved
in the new summary and coefficient table and the historical raw intercept is checked.

## Fixed model definitions

Four inits: 1, 5, 8, 12 June 2023. Stage-I: 14–17 June; Total: 14–24 June, both
inclusive BJT. Both versions use all 51 members, six predictors, intercept OLS,
unstandardized `y_tmax`, independent predictor `StandardScaler(ddof=0)` fits.

| Window | Residualized positions 1–6 | Raw positions 1–6 |
|---|---|---|
| Stage-I | sm_avg; SHF_resid; z500_anom_NCHN; Initial_MSEstar_max_NCHN; TCC_resid; NCVI_concurrent | sm_avg; SHF_Avg; z500_anom_NCHN; Initial_MSEstar_max_NCHN; TCC_Avg; NCVI_concurrent |
| Total | NCHN_tp; z500_anom_NCHN; NCVI_concurrent; Initial_MSEstar_max_NCHN; TCC_resid; SM_resid | NCHN_tp; z500_anom_NCHN; NCVI_concurrent; Initial_MSEstar_max_NCHN; TCC_Avg; sm_avg |

The only final configuration is `RESIDUALIZED`, with position-preserving raw mappings
in `COUNTERPARTS`. No SSR, WNPSH, Initial Barrier or early NCVI enters these models.
Stage-I: SHF_resid = SHF_Avg − fitted(SHF_Avg ~ 1 + sm_avg),
TCC_resid = TCC_Avg − fitted(TCC_Avg ~ 1 + sm_avg).
Total: TCC_resid = TCC_Avg − fitted(TCC_Avg ~ 1 + NCHN_tp),
SM_resid = sm_avg − fitted(sm_avg ~ 1 + NCHN_tp).
Residual regressions use the same 51 members, following Cell 8 statsmodels OLS.

## Numerical methods and QC

- Full design rank must be 7. Rank failures name the group, predictors and singular
  values. Scalers, coefficient conversion (including Cell 8's 1e-12 denominator),
  intercepts, standardized matrices, member fitted values and residuals are saved.
- Raw/residualized fits are computed independently. R², fitted/residual differences,
  ranks and QR projection-matrix differences are checked at absolute tolerance 1e-8.
  Failure is reported, never fixed by assigning identical fits. Ill-conditioning
  remains visible in condition numbers and singular values.
- LMG keeps Cell 8's 2000 permutations, seed 42, MT19937 (`RandomState`, compatible
  with `np.random.seed`), epsilon standardization, positive incremental-R² clipping
  and percentage normalization. Both versions use the same saved sequence, whose
  SHA-256 is recorded per model. A 63-subset R² cache avoids repeat fits; parity
  against the original pure function is tested to 1e-10 on synthetic matrices.
  Permutations are not regenerated from a different RNG family.
- Johnson's pure `_johnson_relative_weights` function is copied exactly from Cell 9:
  ddof=1, correlation matrix, `eigh`, eigenvalue floor 1e-12, `pinv`, raw-weight
  threshold, original normalization. Raw sum−full R² is reported; differences >1e-5
  are warnings, never silently repaired. Explicit stable ordinal rankings replace
  implicit sorting only in the exported comparison tables.
- LOFO removes one already-constructed full-design column at a time. No remaining
  variable is residualized again. Unclipped ΔR² and reduced-model diagnostics are
  saved; plots use normalized positive ΔR². Zero positive denominator returns NA,
  is explicitly `UNDEFINED`, and blocks figures, as required for this comparison.
- Rank ties are resolved by predictor position (stable ordinal descending order).
  `difference_pp = raw_pct - residualized_pct`. LOFO-only delta fields are blank
  in the LMG/Johnson comparison rows. Per-method rankings/top three and sums are
  saved in cross-init QC. Method percentages have different definitions and are
  not interchangeable variance decompositions.
- Historical comparison uses absolute tolerance 1e-6, zero relative tolerance.
  Failures include values/differences and advice to inspect provenance, sample,
  scaler and numerical library versions; no historical CSV is substituted for
  a new calculation. Missing history is distinguished from failed reproduction.

## Outputs and repeat runs

Only the independent output root is writable. Historical `tables/`, `figures/`,
`qc/` and `paper_figures_standalone/` beneath the input root are protected.
Existing unowned output directories and symlink write targets are rejected. Re-running may replace this
workflow's own comparison tables/figures when its manifest is present; use a new
output directory to preserve earlier comparison runs. All input/historical CSV
SHA-256 values and numerical library versions are recorded in the manifest.

- `tables/`: 11 numerical tables (see `TABLE_FILES`), including full model summaries,
  coefficients/scalers, all 816 member fitted rows, design values, residualization
  audit, all 288 method-importance rows, predictor mapping, 8 fitting-equivalence
  rows, 144 paired comparison rows, model and cross-init QC.
- `qc/`: manifest with table checksums and gate, historical reproduction CSV,
  actual 2000×6 permutation sequence, rendered layout QC, figure-file manifest.
- `figures/`: Fig3_Total, Fig4_Total, Fig9_StageI, Fig10_StageI, each suffixed
  `_residualized` and `_raw`, each PNG/PDF (16 files). Fig.4/10 new exports also
  have `_layout_v2` suffixes; prior formal figures are preserved. Only these eight figures
  are produced; no difference heatmaps.

The figure notebook rejects edited/incomplete/failed result tables. Scatter layout
is 2×2 in date order; each paired init/window shares limits and ticks, while different
inits can use different ranges. Cell 14 styles, fitted trend and r/R²/slope diagnostics
are retained; no 1:1 line is drawn. Axis boxes have equal physical dimensions,
without equal data-unit aspect constraints. Importance layout is 3×4, with common
per-method scales across both windows/all inits/versions.
For each window, predictor positions are fixed according to the descending LMG importance in the 12 June 2023 residualized model and retained across all initialization dates, methods, and raw/residualized counterparts.
The CSV-derived reference order must match the confirmed order exactly:
Total = TP, Z500, NCVI, TCC resid, Init MSE*, SM resid;
Stage-I = SHF resid, SM, Z500, Init MSE*, NCVI, TCC resid.
Raw names use the same positions (SHF resid→SHF, TCC resid→TCC, SM resid→SM).
Mismatch raises an error, with no fallback to per-date/per-method sorting.
Existing numerical comparison CSV `display_order` fields are not rewritten;
the renderer derives its v2 order from the actual 12 June LMG rows.
Green/blue/orange are fixed by method. Only panels (a), (e), (i) show factor names.
Black bold 8-point one-decimal values are measured using rendered text extents and
prefer the bar-end exterior. Labels that cannot fit outside move inside with a small,
borderless near-white backing; short/zero bars retain true lengths. LOFO uses the exact
heading `Normalized positive ΔR² (%)`. Width is 6.5 inch, PNG is 600 dpi/3900 pixels,
and PDF uses the same complete fixed canvas (no tight cropping).

Layout v2 uses a 6.5×8.1-inch GridSpec canvas with equal relative axis widths
approximately 0.1838 (1.195 inches). Margins: left .145, right .985, bottom .09,
top .92; GridSpec wspace .19, hspace .60. Factor labels are 8 pt, panel/date headings
9 pt, method row headings 10 pt. No x ticks/grid; only a pale-grey left zero line
remains. TrueType embedding is explicitly selected at PDF export. These parameters
are checked against rendered geometry, not used as a substitute for final-file QA.

New outputs: `Fig4_Total_{raw,residualized}_layout_v2.{png,pdf}` and
`Fig10_StageI_{raw,residualized}_layout_v2.{png,pdf}`. Use the existing figure Notebook;
or, to redraw only the four importance figures, call `make_figures(data, plan,
'importance')` then `save_figures(built, INPUT_ROOT, OUTPUT_ROOT)` after `load_results`
and `figure_plan`. Existing Fig.3/9 pairs are retained during full exports.
New QC filenames are `qc/figure_layout_v2_qc.json` and
`qc/table_figure_layout_v2_manifest.csv`; old figure QC is not overwritten.
Computational CSVs, their checksums, and their manifest are read-only in rendering.

Automatic QC checks text collisions/clipping, value containment, equal boxes,
paired actual ticks/limits and exported file sizes. Real HPC figures still need
manual visual review for annotations covering data and manuscript readability.
Caption drafts are in `mlr_comparison_captions.md`. Bold labels imply no significance.

The v2 QA follows the [nature-figure reference](https://github.com/Yuan1z0825/nature-skills/blob/main/skills/nature-figure/README.md)
for final-size alignment, text and data integrity, adapted to JGR: Atmospheres.
Measured row/column edges, repeated gaps and panel title anchors must agree within
1.5 pt. Exported importance PDFs are read with **PyMuPDF** (`pip install pymupdf`):
actual font sizes ≥7.5 pt, embedded fonts, editable vector text, all 72 percentage
labels, no clipping or unexpected line-box collisions are required. Failures block
the acceptance manifest; they are not suppressed with clipping or ignored QC.
Outside labels use the white page background; white backing is confined to labels
that must go inside. Final real-data visual review remains necessary.

To run actual-range QA without writing into historical data, set
`MLR_REAL_OUTPUT_ROOT` to the existing, checksum-approved comparison output directory
and run `pytest -q tests/test_mlr_comparison_plotting.py`. The genuine CSV test
renders to a separate temporary directory and runs PDF/PNG QC. Without that path,
the real-data test is explicitly skipped. Synthetic layout fixtures use declared
test-only reference percentages; they do not establish actual scientific rankings.

## Tests and limitations

From the repository: `python -m pytest -q tests/test_mlr_comparison.py`.
All tests use clearly synthetic matrices/schema fixtures; they are not paper results.
Tests cover original-function parity, RNG sequence, member matching, scaler/rank,
fitting-space equivalence, fixed-column LOFO, normalization edge cases, history
failure detection, output protection, safe import, notebook schemas and original
blob preservation. `tests/test_mlr_comparison_plotting.py` additionally checks
synthetic figure layout/files. Test artifacts live in temporary directories only.
`tests/test_mlr_notebooks.py` checks both notebooks in separate fresh kernels using
synthetic CSVs, verifies input hashes, and checks the explicit missing-input failure
before output creation. It requires nbclient/ipykernel and permission to open local
Jupyter kernel sockets.

The Codex environment has no `/data1/huangy/...` CSVs. **The 16 genuine models,
eight genuine figure pairs, historical reproduction and actual scientific ranking
changes are pending HPC execution.** No conclusions or preferred model version
are selected by this implementation.


## JGR figure revision — original LOFO ΔR² (branch feature/mlr-figures-lofo-raw-delta-r2)

This isolated branch starts from `dev/jgr-major-revision` and imports audited PR #12 MLR helpers without merging old history. Its output policy supersedes earlier eight-figure and layout-v3 directions in this README.

- **Paper figures only:** `Fig3_Total_residualized_lofo_deltaR2_v1`, `Fig4_Total_residualized_lofo_deltaR2_v1`, `Fig9_June14_17_residualized_lofo_deltaR2_v1`, and `Fig10_June14_17_residualized_lofo_deltaR2_v1`; each as PNG 600 dpi and vector PDF.
- **Fig.3 and Fig.9:** x-axis `ECMWF Tmax (°C)`; y-axis `MLR predicted Tmax (°C)`; model R² and fitted slope only, no Pearson r annotation. The fitted member values/MLR remain unchanged. Figure title uses 14–17 June rather than Stage-I.
- **Fig.4 and Fig.10:** first row LMG %, second row Johnson %, third row original LOFO ΔR² = R²(full) − R²(delete predictor k), **not** normalized to percentages. Predictor standardization, residualization, sample 51, and same-predictor OLS deletion refit are unmodified. LMG/Johnson percentages still sum to 100%. LOFO does not partition model R² and is not commensurate with percentage-based weights. All three rows show numeric x ticks; method-specific shared scales across four inits, fixed predictor positions.
- **Audit output:** `tables/table_mlr_lofo_original_delta_r2_residualized.csv` preserves full/deleted R² and original ΔR². `importance_pct` in the imported comparison table retains its **legacy historical comparison role**, never used for LOFO in new paper figures. A nested in-sample OLS ∆R² below −1e-8 is a QC error.
- **Provenance:** `SOURCE_COMMIT` points to original historical MLR method definition, not to HPC HEAD. This branch never overwrites historical outputs. Use a **fresh** `MLR_OUTPUT_ROOT` rather than `mlr_raw_residual_comparison` already used by PR #12. Historical raw-versus-residualized numerical comparisons remain in source for audit; the figure notebook exports only four residualized plots. Genuine HPC results, actual figure appearance, and PDF font checks remain to be evaluated on HPC.

Run order: first `mlr_raw_residual_comparison.ipynb` to generate and audit model tables in a **new** output root, then `mlr_comparison_figures.ipynb` with the same root. Both notebooks use precomputed HPC stage CSVs, not GRIB files. The independent plot renderer does not refit models. Check the new `qc/figure_lofo_deltaR2_v1_qc.json` and `qc/table_figure_lofo_deltaR2_v1_manifest.csv`, and send genuine PNG/PDF for final review.


### Fig.4/10 label-placement microrevision (v2)

- The new files are `Fig4_Total_residualized_lofo_deltaR2_v2.png/.pdf` and `Fig10_June14_17_residualized_lofo_deltaR2_v2.png/.pdf`. Earlier v1 files are preserved verbatim; the official Fig.3/9 v1 output is not rerendered or overwritten by this notebook's export cell.
- If a short bar has enough room to the right for its actual rendered value label plus clearance, the label stays outside. If not, the label is centered within the colored bar. No white label background covers a bar endpoint. Bar length, numeric x scale and actual model importance are never modified.
- Positive `0 < ΔR² < 0.0005` values are displayed as `<0.001` rather than `0.000`; source values remain unchanged.
- Every bar/label pair is checked in rendered pixels by `importance_label_placement_qc`. Revisions fail on any label extending outside an inside bar, or overlap with a bar tip when outside. All earlier alignment, text-collision, PDF vector text/font and SHA checks remain active.
- Execute the updated `mlr_comparison_figures.ipynb` with `MLR_OUTPUT_ROOT` pointing to the verified computed tables. It constructs all figures for analysis, but exports **only Fig.4 and Fig.10 v2** to avoid duplicate v1 export failures. The new QC outputs are `qc/figure_lofo_deltaR2_v2_qc.json` and `qc/table_figure_lofo_deltaR2_v2_manifest.csv`. No rerun of MLR is needed for this styling change.
