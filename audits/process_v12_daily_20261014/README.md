# v12 daily-source audit and internal-only diagnostics

**v12 daily-to-stage aggregation consistency PASS. Formal figure status BLOCKED: daily NCVI anomaly missing.**

The uploaded raw daily, TCC daily and Tmax daily sources independently cover 2 target initializations × 11 dates × 51 members without duplicates or omissions. Tmax values in the two source tables are exactly equal (max difference 0.0 C). `source_hashes.csv` preserves SHA256 and sizes of the inputs; none are altered. The new NCVI stage table matches the comparison manifest provenance hash.

`daily_to_stage_qc.csv` records all 1224 member/variable comparisons across four initialization/window combinations. Maximum absolute error is 1.4210854715202004e-14. TP is summed over the window, while Tmax, SM, SHF, Z500 and TCC are averaged, preserving existing definitions. A fixed 1e-8 tolerance is used; no threshold was relaxed.

`process_v12_adapter.py` checks exact keys and coverage, converts only six specified physical factors from wide to long format, recomputes groups from v12 absolute stage Tmax, refuses ambiguous boundary ties, and computes means from exactly 17 daily members. NCVI, which is an old absolute/stage variable in the raw daily table, is deliberately excluded. The v12 physical route does not depend on legacy grouped curves. `current_v12_groups.csv` includes ranks. Member43 is Hot17 and member39 Middle17 in 06-08 Stage-I.

`legacy_daily_group_comparison.csv` compares reaggregated v12 curves with all available legacy group means for each target date/group/factor, including the other three membership-identical combinations. Legacy values are retained solely for comparison. `internal_physical_daily.csv` contains physical series for both grouping windows, with status INTERNAL_ONLY_NCVI_MISSING. Display limits remain Total 14–24 and Stage-I 14–18; extra Stage-I-grouped days are internal audit data.

`internal_partial_daily.csv` and `internal_scatter.csv` replay saved raw/residualized Stage-I models via `parameters`, `residual_relation` and `partial_fitted_tmax` in process_timeseries_core.py. The daily SM/SHF inputs are v12 group means; saved full-model intercept, standardized coefficients and ddof=0 training scalers are used. Other predictors stay at model training means. Residualized SHF uses fixed stage intercept/slope; no daily or full-model fitting occurs. Day18 is marked outside_training_window. Member Tmax and raw SM are identical across versions. These intermediate tables are not final plot products.

## NCVI source audit

`ncvi_stage_design_qc.csv` confirms the supplied concurrent stage values equal PR12 model design NCVI for all target members and both versions. anomaly_used is true. climatology_file_used identifies same-init PF/CF reforecast references; no recorded absolute fallback is accepted. This establishes stage provenance only: it does not establish every daily value, unit attribute, reforecast sample/date coverage or daily-to-stage replay. The raw daily `NCVI` column is not `NCVI_concurrent`. Neither repeating stage NCVI nor substituting legacy daily curves is permitted.

Required: export actual daily concurrent NCVI anomalies for 06-08/06-12 × 14–24 June × 51 members, including units, anomaly_used, fallback_reason, actual realtime and reforecast source variables/files/attributes, climatology averaging dimensions and sample/date coverage. Verify Stage-I/Total member means against the supplied concurrent stage table. Use the frozen scientific definition; do not invent a new baseline. Existing stage source CSV and PR12 implementation can locate this exporter, but the named HPC GRIB files are not accessible here.

## Methods consistency requiring author decision

Tmax: preserved v12 uses UTC+8 valid times, gridpoint daily maximum, arithmetic gridcell spatial mean, and arithmetic window mean. Legacy process Notebook uses cosine-latitude spatial weights. These are different definitions. The new internal route follows v12 unchanged. If the manuscript requires area weighting, the impact includes stage Tmax ranking/groups, all dependent MLR fits/results and diagnostics; that scientific revision is outside this task.

TCC: source date 2023-06-14 has window_start_utc 2023-06-14, window_end_utc 2023-06-15, window_center_bjt 2023-06-14 20:00. Thus the recorded interval is 08:00–08:00 BJT, not a midnight-to-midnight BJT natural day. Source labels and values remain unchanged to preserve exact MLR aggregation. Changing TCC daily windows would affect stage TCC, residualization, model fits and diagnostics. The Methods/captions must disclose this distinction, or a separately authorized scientific revision must precede harmonization. No silent timestamp shifting is performed.

No formal Fig.5/11/S11 CSV or PDF/PNG exports, runtime figure QC PASS, PR merge, MLR refit, LMG/Johnson/LOFO recomputation, or historical overwrite occurs.
