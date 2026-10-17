# Standalone Fig.5 / Fig.11 / Fig.S11 diagnostic workflow

`process_fig5_fig11_diagnostic_standalone.ipynb` is a self-contained six-cell HPC workflow. It reads existing member-daily CSVs and saved model tables; it does not refit MLR, scalers, residualization, LMG, Johnson or LOFO. Run all six cells in order with a **new** output directory. The notebook in GitHub is intentionally unexecuted; real figures require HPC inputs.

## Current figure selection

- **Fig.5 Total:** 8 June left / 12 June right, 14–24 June daily evolution, using each initialization's Total Hot17/Middle17/Cold17 membership.
- **Fig.11:** residualized model only; **5 June left / 8 June right** in one 6×2 process figure. Rows are SM–SHF partial fitted Tmax, SM, Z500 anomaly, SHF, NCVI anomaly and Total cloud cover. Display dates remain 14–18 June. The saved training/grouping window is 14–17 June; 18 June remains a frozen-parameter projection outside that window.
- **Fig.S11:** one asymmetric 2×2 mechanism comparison. The 5 June left column shows Tmax–Z500 anomaly above Tmax–NCVI anomaly. The 8 June right column shows Tmax–SM above Tmax–SM–SHF partial fitted diagnostic. These are 51-member **14–17 June scalars**, not group means or daily points. NCVI scalars are taken directly from the saved model design, not from an unverified daily aggregate.

All scatter y labels read `6.14–17 Tmax (°C)`. Statistics show Pearson r, descriptive R²=r² and n=51; this R² is not full-model R². Stats boxes are at the upper right except the existing partial fitted row, which retains its upper-left box. Colors, markers and line styles are unchanged. Raw model figure/plot-table outputs are disabled.

The 5/8 June pair is selected because the 5 June members show the clearest Z500-related separation, while the 8 June members show the strongest SM–SHF partial-fitted relationship. `TOTAL_INITS` remains 8/12 June. All four dates stay in the source-integrity checks because the immutable NCVI export directories contain 1/5 and 8/12 June together.

## Required NCVI daily exports for the added dates

The old exporter was configured for 8/12 June. Those two dates cannot supply daily curves for 1/5 June. Cell 1 now explicitly uses two immutable source directories:

```python
NCVI_RUN = V12 / 'ncvi_daily_standalone/run_001'
NCVI_EARLY_RUN = V12 / 'ncvi_daily_standalone/run_early_0601_0605'
NCVI_RUNS = (NCVI_RUN, NCVI_EARLY_RUN)
```

If the early export does not already exist, run the prepared **`ncvi_daily_export_early_0601_0605.ipynb`** first. It copies the existing exporter's algorithms unchanged and changes only Cell 1 to:

```python
INITS = ('2023-06-01', '2023-06-05')
OUTPUT_ROOT = V12 / 'ncvi_daily_standalone/run_early_0601_0605'
```

Run that exporter’s six cells first. Its Cells 2–6 are byte-for-byte identical to the existing export notebook. Its two-date coverage checks and algorithms already support this pair. It reads the corresponding pinned realtime/reforecast sources; it does not refit models or recalculate ERA5. The directory must be new. If the export is already available elsewhere, configure its actual directory instead. The process notebook validates the export hashes, source PASS, exact date/member coverage, no overlap between exports, and daily internal replay. It fails explicitly on missing dates; it never interpolates, substitutes stage scalars for daily data, or fabricates curves. A missing early daily source means HPC validation is pending, not that its curves have been generated.

The historical `FAIL_STAGE_EQUIVALENCE` and ERA5 provenance limitations remain intact. New source manifests are recorded separately. Rendering does not promote any scientific FAIL/PENDING status to PASS.

## Fig.5 axes and cloud-cover units

Requested ascending bounds and tick steps are:

| Factor | Preferred y range | Step |
| --- | --- | --- |
| Z500 anomaly | −50 to 100 gpm | 25 |
| TP | −1 to 10 mm day⁻¹ | 1 |
| NCVI anomaly | −0.30 to 0.25 PVU | 0.10 |
| Total cloud cover | 0 to 100% | 25 |

Each panel checks all plotted groups including ERA5. Before plotting, every central value and any supplied uncertainty bounds are checked. Bounds expand outward on the specified tick grid when necessary. If an expanded range makes numeric labels collide, label density is reduced while every tick position is retained; the stride is recorded in QC. The default target ranges retain every label. TP keeps −1 as bottom whitespace but suppresses negative ticks if all data/bands are nonnegative. The existing day18 shading denotes time outside the training window; it is not a vertical uncertainty band. Optional vertical bands use `value_lower` / `value_upper` in the same source units as `value`, and are validated, converted and drawn together. No bands are invented if those columns are absent.

Cloud cover is converted on a **plotting copy only**, separately for S2S and ERA5. `TCC_SOURCE_UNITS` accepts `auto`, `fraction` or `percent`. Auto detects [0,1] vs [0,100] by range; values already in percent are not multiplied again. A percentage sample entirely at or below 1 is inherently ambiguous, so use an explicit `percent` override when source metadata says so. The source-unit inference/override and multiplier are saved in QC. Source CSVs, saved training scales and coefficients retain their original units and values. Invalid/nonfinite values fail.

## Fig.11 panel-specific axes

Fig.11 checks each initialization panel separately, including all central curves and any supplied uncertainty bounds. Preferred limits expand outward on the stated tick grid whenever any displayed value or bound falls outside them.

| Initialization | Factor | Preferred y range | Step |
| --- | --- | --- | --- |
| 5 June | SM–SHF diagnostic | 28 to 32 °C | 1 °C |
| 8 June | SM–SHF diagnostic | 32 to 35 °C | 0.5 °C |
| 5 June | Z500 anomaly | 25 to 125 gpm | 25 gpm |
| 5 June | SHF | 50 to 80 W m⁻² | 10 W m⁻² |
| 5 June | NCVI anomaly | −0.40 to 0.10 PVU | 0.10 PVU |

With the uploaded HPC plotting table, the 5 June SM–SHF series reaches 27.92 °C and its Z500 series reaches −5.09 gpm. The range check therefore expands those two panels to 27–32 °C and −25–125 gpm, respectively. This behavior is recorded in `axis_range_qc.csv`.

Date markers use the exact daily datetime nodes and ticks. Fig.5 labels every day 06-14 through 06-24 at 70° on its bottom row; Fig.11 labels 06-14 through 06-18. There is no half-day marker offset.

## Cells, outputs and validation

1. Configure actual input paths, comparison pairs, cloud-cover units and a new output directory.
2. Verify source hashes, 51-member coverage, original descending Tmax ranking/groups, daily-to-stage values and saved models.
3. Load and validate existing daily NCVI exports and independent ERA5 references.
4. Reuse saved residualized intercept, coefficients, training means/scales and SHF residual relation for partial fitted diagnostics. Export three consolidated plot tables:
   - `fig5_total_timeseries_plot.csv` (8/12 June only)
   - `fig11_stageI_timeseries_residualized_plot.csv` (all four source dates; Fig.11 selects 5/8 June)
   - `figS11_stageI_scatter_residualized.csv` (all four source dates, including Z500/NCVI scalars; Fig.S11 selects 5/8 June)
5. Draw the three selected figures and enforce renderer text collision/clipping, matched panel dimensions, exact daily ticks/markers, and curve/band range checks.
6. Export **600-dpi PNG, vector PDF and SVG**, immutable executed-code snapshots, input/output hashes, source manifests and independent audit tables. Recheck all inputs and both NCVI directories for mutation. PDF font/size/clipping QC remains mandatory where PyMuPDF is available; missing PyMuPDF is explicitly NOT_RUN.

Additional QC includes `axis_range_qc.csv`, `tcc_display_units_qc.csv`, `scatter_descriptive_statistics.csv`, `layout_qc.csv` and `pdf_font_qc.csv`. Figure filenames include both date columns and the residualized version. Fresh output directories prevent overwriting old figures/tables. Old repository scripts and audit results are not modified.

## ERA5 and methods distinctions

The mandatory independent ERA5 320-K NCVI daily anomaly source is:

```text
/data1/huangy/fig6/NC/MSE/ERA5/NCVI_theta320/processed/era5_ncvi_theta320_daily_anomaly_2023.csv
```

Configured columns are `date` and `ncvi_theta320_anom_pvu`; absolute/climatological PV is never silently substituted. The reader requires exactly 14–24 June, one finite daily value per day, and PVU. The same reference appears under each initialization. Optional `ERA5_PHYSICAL` supplies other variables; `None` omits only those other lines. ERA5 NCVI remains mandatory.

ERA5 NCVI uses cos(latitude) weighting according to the earlier method, whereas the S2S source uses arithmetic spatial averaging. Independent ERA5 climatology/unit/time/region provenance remains `PLOTTED_PROVENANCE_PENDING`. TCC source labels retain their original 08:00–08:00 BJT interval convention; they are not silently relabeled. All outputs remain `PRELIMINARY_DIAGNOSTIC`, `formal_process_allowed=False` until the scientific issues are resolved.
