# NCVI precision tracing — diagnostic only

Copy the single full code Cell in `ncvi_precision_diagnostic.ipynb` into HPC Jupyter after successful standalone Cells1–5. It embeds all diagnostic functions/imports and reads only the existing in-memory config and ref table; no repository modules/Git required. Set its separate PRECISION_OUTPUT to a new directory. It does not assign daily, replay, result, STATE or ATOL, overwrite existing files or change PASS/FAIL.

## Original numerical order confirmed by source inspection

Preserved PR12 `compute_ncvi_concurrent` computes regional SI arithmetic means, then hc_window.mean(dim=list(hc_window.dims), skipna=True) in the native dtype. It subtracts that scalar from realtime SI, averages all non-number dimensions with skipna=True, THEN casts the resulting 51 values to float (float64) and multiplies by 1e6. Casting occurs after the native stage reduction, not before. The exporter converts regional arrays to PVU before daily_replay casts to float64. float32 reduction/subtraction/conversion ordering can therefore produce different roundoff; this is a hypothesis until the diagnostic is run on the actual inputs.

The diagnostic rereads pinned GRIB using the same validated fields, reference samples, CF/PF order, region and BJT windows. It retains original reforecast time/number/step axes, concatenates raw CF/PF grids before spatial averaging, and uses original default spatial mean/skipna=True stage means. Existing finite-value checks refuse missing grids; no missing-value repair is made. Original MLR skipna behavior is reproduced on these finite sources. It refuses already-PVU sources rather than reverse-converting them to fictitious original SI data.

Four paths per 204 members:

1. old_mlr_native_si: literal native SI scalar/subtraction/stage reduction, cast to float64, PVU last.
2. exporter_pvu_float64: regional PVU conversion, (step, sample) stacking, float64 stage reductions, subtraction.
3. si_float64_then_pvu: promotes SI regional data before scalar/subtraction/stage reduction, PVU last.
4. native_pvu_order: native PVU scalar/subtraction/stage mean without float64 promotion.

The different paths are diagnostics only. Array dimension order and scalar/subtraction/mean/conversion dtypes are recorded. For exact reproduction of an archived binary result, package versions, memory layout, GRIB decoder and original selected field must also agree; persistent discrepancies cannot be automatically attributed to roundoff.

Outputs: ncvi_precision_member_paths.csv (816 rows =204×4), summary by init/window/path (max absolute error, signed mean/std ddof0, value mean/std), intermediates CSV, source hashes/attributes CSV, executed Cell snapshot and full precision manifest. No formal process manifest is replaced.

## User-reported HPC evidence, not independently rerun here

Regional arrays realtime (47,51) and reforecast (47,220), both float32. User reports daily equal-day/time-weighted vs new scalar differences <=1.11e-16 PVU, while new scalar vs saved model differences are 3.5177e-08, 4.7483e-08, 3.6543e-08 and 2.5594e-08 PVU for 06-08 Stage-I/Total and 06-12 Stage-I/Total. With fixed ATOL=1e-8, existing strict QC is still FAIL. No raw HPC outputs were supplied in this turn; these numbers are reported evidence, not cloud validation.

## Acceptance recommendation if roundoff is proven

Keep two separate checks: (a) archived MLR numerical reproduction using the original native-SI ordering at the existing tolerance; (b) daily anomaly temporal aggregation equivalence under the explicitly documented exporter precision/order. Save both errors and dtypes. A native replay PASS plus daily-to-new-scalar equivalence does not silently change the current daily-to-saved-MLR strict FAIL. If accepting a new precision contract is scientifically justified, make that an explicit separately reviewed acceptance rule; preserve old results and thresholds and report the cross-contract discrepancy. Never offset daily anomalies, replace archived stage values or refit to hide numerical differences.

Cloud tests exercise numerical expressions with synthetic float32 arrays; real HPC diagnostic remains pending.
