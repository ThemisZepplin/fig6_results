# Process source audit — 14 October 2026

Status: BLOCKED before process reconstruction and publication export. Branch: feature/process-fig5-fig11-refactor, starting HEAD 91bda1851892b587e102be7274a6081ee41e8ba8. No original CSV, model, notebook or figure was changed. No model fitting or importance calculation was performed.

## Verified inputs and saved-model replay

`hash_checks.csv` compares SHA256 for all five supplied model tables and the v12 raw stage table against comparison_manifest.json; all six match. The raw table covers 0–50 exactly once for each target initialization/window. `member_stage_differences.csv` contains 612 individual comparisons (51 members × four init/window combinations × Tmax/SM/SHF); difference means v12 minus legacy. These are stage-level comparisons, not fabricated daily differences.

`stageI_model_replay.csv` records the four complete Stage-I model reconstructions from saved parameters and six-predictor design. Existing core checks verify saved scaler ddof=0, targets, fitted residual identities, training-mean baseline and fitted values. Maximum fitted replay error is 5.69e-14 degrees Celsius. Frozen SHF residual projections reproduce stored residualized design within 1.47e-14 W m^-2. These checks validate replay of stored models, not scientific model validity or new daily projections.

## Source-chain discrepancy

The supplied Notebook exporter (physical cell index 3, execution_count 5) uses `_sel_init_member_cube` to construct valid BJT time with UTC+8, then `get_s2s_tmax_member_daily` performs gridpoint daily maximum followed by `_area_weighted_region_mean` with cosine-latitude weights. The preserved scientific source on PR #12 (`stage_specific_residualized_ols_shf.py`, branch HEAD 7f3b9df5d1ea9583fab1c77b0125d7ffcd097e62) performs the same BJT daily maximum but `process_s2s_tmax_bjt_daily_max` uses an arithmetic spatial mean. Window definitions agree: Stage-I 14–17 June and Total 14–24 June, arithmetic temporal mean of daily Tmax.

Thus a concrete spatial aggregation difference is established. Its precise contribution to each numerical difference cannot be isolated without underlying grids/daily sources and execution provenance. Source variable candidates also differ (legacy admits t2m; v12 candidates mx2t6/mx2t/tmax); actual selected variables, grid/mask and file equivalence remain unverified. Do not attribute all differences to weighting or relax tolerances.

Stage-I maximum absolute old-versus-v12 differences: 06-08 Tmax 0.1091750482 C, SM approximately 0.001143 m3 m-3 (see CSV for exact values), SHF 0.669329 W m^-2; 06-12 Tmax 0.0567880722 C, SM approximately 0.001413 m3 m-3, SHF 0.438061 W m^-2. Total Tmax maxima are 0.1132884130 C and 0.1196942655 C. Exact per-member differences for all variables are in the CSV.

06-08 Stage-I: v12 descending Tmax ranks member43 at 17 (Hot17, legacy Middle17), member39 at 18 (Middle17, legacy Hot17). Other three target memberships agree. No daily reaggregation has been performed: no member-daily upload is present. Stage tables or group means cannot reconstruct individual daily members.

## NCVI and unresolved prerequisites

Saved Notebook outputs for the exporter explicitly show NCVI anomaly for both target initializations, 51 members × 11 dates, plus ERA5 anomaly coverage 14–24 June. The code subtracts same-init reforecast daily climatology and allows absolute fallback; a generic value/PVU unit label alone proves neither mode. Saved logs support anomaly use, but supplied daily CSVs have no per-date anomaly/source flags and no immutable binding to those logs or source attributes. Actual reforecast files, selected variables, units, climatology sample selection and every-date anomaly provenance require dedicated export evidence. No absolute fallback is permitted in the new flow.

Required before exports:

1. Upload the claimed candidate member-daily CSV with IDs, BJT dates, factors, units and provenance. Compare its 14–17 June member Tmax/SM/SHF against v12. Since legacy stage values already disagree, obtain v12-derived daily Tmax, SM and SHF if candidate replay fails. Do not correct old daily values to force agreement.
2. Prefer the existing v12 `table_daily_member_raw_factors_SHF.csv` and `table_daily_tmax_SHF.csv` (preserved source exports), including 14–24 June for both initializations. Confirm exact schema/coverage before reuse. Reaggregate with v12 groups, comparing all four groups against old daily exports. Check other physical factors before accepting a mixed-source process figure.
3. Supply daily NCVI anomaly source/QC metadata and the concurrent NCVI stage audit (anomaly_used, climatology_file_used, fallback_reason), plus reforecast reference and date/member coverage. If NCVI source chain differs, export verified v12-conformant daily anomalies rather than silently substituting.

No Fig.5/11/S11 plot CSV or figures are generated. Existing static workflow remains intact. PR #12 is not merged or imported; only its saved outputs and preserved source are inspected. PR #14 remains a Draft and is not merged.
