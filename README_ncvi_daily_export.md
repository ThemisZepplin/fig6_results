# Independent daily NCVI exporter (HPC execution pending)

No GRIB or ERA5 source CSV is available in this cloud workspace. Actual physical-source, daily anomaly and 204-member replay QC remain **NOT RUN**. Existing uploaded stage values are not daily values. No scientific output is fabricated.

## Source review

Reference PR12 preserved source: `stage_specific_residualized_ols_shf.py` at PR12 HEAD 7f3b9df5d1ea9583fab1c77b0125d7ffcd097e62, functions `_ncvi_concurrent_select_bjt_window`, `_ncvi_concurrent_candidate_files`, `compute_ncvi_concurrent`, and loader/init-selector dependencies. Uploaded timeseries Notebook was inspected as text, including daily helpers and ERA5 CSV reader/source audit. Neither source was executed or modified.

MLR selects BJT dates using initialization midnight + lead +8h, arithmetic regional grid means, realtime member means over selected effective times, minus a reforecast scalar over all selected dimensions. Its hindcast selector takes the first exact init-date match; absent an exact date it uses all records. New exporter mirrors this selection but requires an explicit retained reference-time list, matching month/day and midnight initialization. An unexpected sample set fails rather than falling back silently. Full GRIB layout remains an HPC interface check: unexpected dimensions/coordinates stop execution.

Old Notebook uses cos(latitude) weights, daily reforecast climatology, nearest-layer and magnitude-based unit heuristics, and may fall back to absolute PV. New exporter allows none of those substitutions: exact pv/theta320 filter, proven raw units, explicit PVU conversion once, complete finite region/samples, CF0/PF1–50, immutable input paths/hashes. Current MLR multiplies SI PV by 1e6 unconditionally; already-PVU sources are converted only once here and any incompatibility is exposed by scalar replay.

## Configuration and run

Install xarray, cfgrib/ecCodes, numpy and pandas on HPC. Create `ncvi_daily_hpc_config.json` (not tracked; paths/hashes/sample lists must come from actual source inspection). Run the unexecuted `ncvi_daily_export.ipynb` or call `ncvi_daily_exporter.export(config)`.

Configuration keys:

- `output_root`: independent new directory; an existing directory is refused.
- `stage_table`, `design_table`, `comparison_manifest`: verified PR12 CSV/manifest paths.
- `era5_csv`: existing `era5_ncvi_theta320_daily_anomaly_2023.csv`, columns date and ncvi_theta320_anom_pvu. Eleven unique finite dates required. Never recomputed here.
- `era5_provenance`: JSON with csv_sha256, units=PVU, level_K=320, climatology_sources (actual source files/description), baseline_period. Must be supplied from existing ERA5 audit, not invented.
- `sources`: keys 2023-06-08 and 2023-06-12. Each has realtime and reforecast, each containing cf and pf objects with path, sha256, selected_reference_times (full timestamp strings, e.g. 2023-06-08 00:00:00; reforecast actual selected years must be inspected). Reforecast paths must exactly match the stored climatology_file_used string. Realtime use the preserved MLR `MSE/ecmf_pf_60_2023-06.grib` and `cf_2023-6/ecmf_cf_60_2023-06.grib`, not a candidate search. No implicit candidate/file replacement.

## Outputs and failure evidence

A successful source read writes:

- table_ncvi_concurrent_daily_members.csv: 1122 rows, realtime/climatology/anomaly PVU, member IDs, daily effective-time counts, source reference and QC status.
- table_ncvi_daily_climatology.csv: 22 rows, daily climatology, realtime/reforecast time and sample/value counts.
- table_ncvi_daily_source_audit.csv: actual GRIB hashes, attributes, PV units/conversion, exact reference samples, variable/level and arithmetic-region convention.
- table_ncvi_daily_stage_replay_qc.csv: all 204 comparisons with saved stage NCVI, independently reconstructed stage scalar anomaly, daily equal-day mean and daily realtime-time-weighted mean, raw differences and stage time/sample counts.
- ncvi_daily_export_manifest.json: code commit/hash, config, original input hashes, output hashes, fixed 1e-8 PVU tolerance, all maximum differences and formal_process_allowed gate.

All daily anomalies are literal daily realtime minus daily climatology. No offsets or per-day corrections are applied. If source checks fail, export stops before products. If scalar or daily replay fails, products are saved only as FAIL_STAGE_EQUIVALENCE audit evidence. No QC thresholds are relaxed. Inspect scalar replay first (files, time/sample set, units, spatial definition); then daily equal/time-weight differences. Uniform effective-time/day weights can yield equivalence naturally; unequal weights can break it. Changing baseline or MLR is a separate scientific task, never an exporter correction.

## Process handoff

`process_v12_adapter.verified_ncvi_groups(export_root, groups)` requires immutable PASS manifest/output hashes and exact 1122 member/day keys. It computes v12 Tmax-group means from verified anomaly values, identically for raw/residualized physical sequences. It never substitutes legacy NCVI or stage scalar copies. Fig.S11 existing member diagnostics are unchanged. Formal plotting stays blocked until genuine HPC source and stage QC PASS is reviewed.

Synthetic numerical tests check natural equivalence, unequal-day weight failure and missing/nonfinite source rejection; these tests do not validate real HPC GRIB or climate baselines.

## Read-only preflight and independent QC (added after e5f20f8)

`ncvi_source_preflight.py` reads the actual stage table's climatology_file_used, rather than guessing candidate reforecast paths. Realtime PF/CF paths are explicit CLI inputs. It opens pv/theta320 with cfgrib indexpath disabled (no sidecar edits), discovers reference times/sample years and lead/member coordinates, computes input hashes, and validates the discovered selection with the same strict source loader. Only a new report directory is written. Unknown S2S metadata or missing inputs produce a FAIL report and no config. Existing inputs are read-only.

Example command in HPC py310 (replace placeholders with actual supplied paths):

```bash
conda run -n py310 python ncvi_source_preflight.py \
  --stage-table STAGE_CSV --design-table DESIGN_CSV \
  --comparison-manifest COMPARISON_MANIFEST \
  --realtime-pf ACTUAL_PF_GRIB --realtime-cf ACTUAL_CF_GRIB \
  --report-dir NEW_PREFLIGHT_DIR --export-root NEW_EXPORT_DIR \
  --era5-csv ACTUAL_ERA5_CSV --era5-provenance EXISTING_ERA5_AUDIT_JSON
```

Review ncvi_source_preflight_report.json and the generated ncvi_daily_hpc_config.json before Notebook execution. Point CONFIG_PATH in the Notebook at that generated config; then use py310 with nbconvert/nbclient or Jupyter. Do not overwrite the original Notebook when saving an executed copy.

ERA5 flags are optional for independent S2S preflight. Missing/invalid ERA5 is explicitly FAIL, not assumed metadata: the config contains only supplied paths, with no fabricated provenance. The exporter now continues independent S2S numeric reconstruction even if ERA5 is absent. Manifest separately records s2s_daily_source_qc, s2s_stage_replay_qc, era5_reference_qc and formal_process_allowed. All three must PASS for formal_process_allowed=True. Source preflight cannot claim stage replay PASS: it records NOT_RUN until numerical export. This supersedes the earlier statement that missing ERA5 stops all export.

Current cloud runtime: no /data1/huangy, no conda/py310, no configured HPC credentials or remote execution capability. Real preflight/1122-member coverage/204-stage QC and actual maximum differences remain NOT RUN; no real manifest exists for this change. Synthetic tests remain separate evidence.
