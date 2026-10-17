# Implementation validation and HPC checklist

This report records **code and synthetic validation**, not a scientific-data run.
The real directory `/data1/huangy/fig6/NC/v12/four_init_full_pipeline_NCVIconcurrent`
was absent in the Codex environment. No original pipeline or historical notebook
cell was executed; no raw GRIB/NetCDF was read. No scientific conclusion was changed.

## Completed checks

- 20 computation/interface/preservation tests and 2 plotting tests: **22 passed**.
- Fresh-kernel Notebook checks after final output guards: **2 passed**. Each notebook
  ran independently with clearly synthetic temporary CSVs. The plotting notebook
  read saved tables and did not refit. A separate absent-input execution raised
  the explicit exported-CSV error before any output directory was created.
- **24 tests passed in total** across the two runs. Actual statistical helpers were
  exercised, not mocked. Original-function LMG and Johnson checks extracted only
  pure function ASTs; they never imported or executed the 9911-line pipeline.
- Both new notebooks validate against nbformat v4 schema; all new Python and all
  notebook code cells parse with Python AST. New notebooks are delivered unexecuted,
  with no embedded synthetic results or images.
- Cell 8 final lists match the sole new configuration source. Synthetic member
  pairing, invalid/duplicate IDs, missing fields/nonfinite values, wrong NCVI
  windows, rank failures, scaler ddof, coefficient conversion and QR fitting-space
  equivalence are checked. LOFO retains columns and returns NA for zero denominator.
- Seed42 MT19937 permutation arrays match the historical sequence exactly. New
  cached LMG agrees with the original function at absolute tolerance 1e-10 in the
  tested synthetic matrices. Johnson pure-function outputs match exactly.
- History-comparator fixtures detect member fitted/coefficient drift beyond
  tolerance and report missing history as INCOMPLETE. These fixtures test the
  interface, **not reproduction of the actual historical HPC scientific results**.
- Synthetic renderer produced eight temporary PNG/PDF pairs, verified 3900-pixel
  width, 6.5-inch PDF MediaBox, paired actual axes/ticks, equal boxes, true short/zero
  bar lengths, labels and text collision/clipping checks. Synthetic layout was also
  visually inspected. These are test artifacts and are not committed or presented
  as paper figures. Genuine data may require layout review; checks fail rather than
  alter scientific values.
- Four imported historical blobs match PR #10 exactly, including original notebook
  contents/outputs. Every development-tree entry (Fig.1/2 included) retains identical
  content. No large model-result CSV, PNG or PDF is included in the change.

## Recommended HPC sequence

1. Open `mlr_raw_residual_comparison.ipynb` from the repository. Set CODE_ROOT,
   INPUT_ROOT and independent OUTPUT_ROOT; use a new output root for a new run.
2. Restart Kernel → Run All. Inspect input hashes/window metadata, predictor order,
   residual regressions, rank/condition numbers and all 16 model results.
3. Inspect `table_mlr_historical_reproduction.csv` for the complete set of historical
   metrics, member values, coefficients and importance diagnostics. Missing files
   or fields cannot be certified; failures block rendering.
4. Inspect raw/residualized R², fitted/residual and projection differences,
   normalization/Johnson-sum diagnostics, top-three rankings and difference_pp.
   Same R² does not establish equivalent physical attribution.
5. Open `mlr_comparison_figures.ipynb`, use the same paths, Restart Kernel → Run All.
   Confirm eight PNG/PDF pairs, their manifests, paired axes and text QC.
6. Visually inspect genuine figures for data/annotation occlusion, font readability
   at publication size and agreement with the four caption drafts. Resolve any QC
   failures without modifying data to accommodate layout.

**Pending:** 16 genuine models, eight genuine scientific figure pairs, actual
historical residualized reproduction, actual fitted/R² differences and actual
changes to first/top-three LMG/Johnson/LOFO factors. No preferred raw/residualized
representation or manuscript conclusion has been selected.
