# Final read-only float32 provenance audit

Copy the ONE complete code Cell from `ncvi_float32_final_audit.ipynb` into HPC Jupyter. It is self-contained (standard scientific packages only) and may run in a fresh kernel. Edit RUN_ROOT and the NEW precision output directory at the top. It reads the saved run manifest/config/model source, so no repository module, Git or separate configuration file is needed. Existing run files and hashes are checked before and after; all audit files go to a new sibling directory. No existing results or manifest are written.

The Cell rereads actual pinned GRIB in SI, concatenates CF/PF grids then xarray arithmetic regional mean, retaining native float32. It reproduces original dimension order, source selector, BJT window, native reforecast scalar mean, realtime subtraction, native stage mean, float64 conversion and final multiplication by 1e6. It refuses an already-PVU source or non-float32 regional/intermediate dtype; it never reverse-converts cached PVU or rounds/casts to force historical equivalence. Complete finite sources are required; original xarray skipna behavior is retained. Compare exporter PVU-before-float64 and auxiliary SI64/native-PVU paths separately.

Output member paths contain 204 comparisons for each path (816 total), original errors, max/mean/std ddof0 summaries, intermediate dtypes/scalars, actual source metadata/hashes, executed code snapshot and full manifest. No correction is applied to saved anomalies or historical stage values.

Three independent states:

- daily_internal_aggregation_qc: saved equal-day and weighted daily aggregates compared with their own new scalar replay. This does not compare against the historical model.
- historical_mlr_float32_replay_qc: the 204 old native-SI-order results compared with saved MLR using unchanged 1e-8 PVU threshold.
- era5_reference_qc: preserved existing provenance status. No missing evidence is invented.

The original formal_process_allowed and all strict FAIL results remain unchanged, even if the diagnostic historical replay passes. A separate reviewed acceptance contract would be required before formal process output. Native-history reproduction plus daily-internal equivalence must be reported as separate evidence; neither silently replaces the saved daily-versus-model strict check.

Reference provenance: diagnostic reference_commit=705063b identifies its reviewed baseline, not an HPC HEAD. The updated standalone source reference is 4af3590, which includes the reforecast-member correction; 544c901 was the earlier incorrect baseline label after that change. These labels never imply HPC repository execution. Actual execution version is the captured Cell text/hash. Do not alter an already saved run_001 manifest to change its old label.

User-reported HPC run evidence (not independently reproduced in cloud): source PASS; 1122 daily rows,22 climatology rows,204 comparisons; daily/new-scalar agreement about 1e-16; historical maximum 4.748331733361866e-08 PVU,139/204 above 1e-8; ERA5 evidence incomplete. Reverse float32 quantization reconstructs all204 historical values around1e-16. This strongly supports float32 storage provenance but does not prove source/operation identity: true SI replay must still establish that. No real precision CSV or manifest was uploaded here. Tests exercise synthetic native float32 ordering only.

Return the audit summary, intermediate dtype CSV, member-path CSV and full manifest after HPC execution. If native replay fails, investigate decoder/package versions, exact original field selection/dimension ordering/sample membership before attributing error solely to arithmetic order. Do not modify thresholds, daily anomaly offsets, rounding or MLR fits.
