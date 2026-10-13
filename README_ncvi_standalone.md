# Copy-by-cell HPC Notebook

Use `ncvi_daily_export_standalone.ipynb`. It contains exactly six unexecuted code cells. Copy each complete cell into your existing HPC Jupyter Notebook and run 1–6 in order under py310. Do not copy only fragments. No Git, CLI, external configuration JSON, local repository modules or `__file__` is needed. Installed scientific packages (numpy, pandas, xarray, cfgrib/ecCodes) and existing HPC input files are required.

## Cell prerequisites and outputs

1. Environment and paths: edit all paths and a new OUTPUT_ROOT here. ERA5_EVIDENCE starts explicitly unknown; fill only from actual independent audit. Missing evidence blocks formal acceptance, not S2S. Existing comparison_manifest.json is scientific input provenance, not an external configuration file. Prints runtime versions and reference commit.
2. Metadata: requires existing stage CSV and realtime/reforecast GRIB. Reforecast paths come from stored climatology_file_used; no candidate replacement. Prints actual PV field/layer/units, full path/hash, selected reference samples and years; builds config in memory. No file writes.
3. Hash/sample/member validation: requires comparison manifest/design CSV. Executes embedded strict source loader and retains validated arrays. Prints source PASS and independent ERA5 status. Unknown S2S fields fail immediately. ERA5 CSV/provenance cannot be guessed; no additional JSON evidence file is required.
4. Computation function: embeds daily_replay verbatim from exporter. Prints fixed 1e-8 PVU tolerance. No scientific calculation yet.
5. Actual calculation: requires validated arrays and sequential cells. Rechecks GRIB hashes; prints 1122-row coverage and 204-comparison maxima. No product files yet. Failed equivalence is retained, not corrected.
6. Audit export: refuses existing directory; outputs the four CSVs, manifest, full preflight report and executed-code snapshots. Prints all three independent QC states, maxima and formal_process_allowed. No plots generated, regardless of PASS.

## Execution provenance

REFERENCE_COMMIT=544c901 identifies the reviewed code baseline; execution_git_head=null explicitly avoids representing it as the HPC HEAD. IPython execution history records the exact source pasted into each of the six cells. These snapshots are saved with a SHA256, including paths/parameters and inline ERA5 evidence. If execution-history capture is unavailable, the cell stops rather than claiming code provenance. Do not disable IPython history. Run with `%run`/plain Python is not supported; this deliverable targets interactive Jupyter cells.

Core scientific functions `inspect_grib`, `load_pair`, `daily_replay` and `digest` are embedded without alteration and tested for AST equality with their reviewed module equivalents. Only orchestration/config/provenance and the point of export differ. The original modules and Notebook remain available and unchanged.

Daily PV, daily reforecast climatology and literal anomaly remain separate. Region=35–50N/115–130E, exact pv/theta320, arithmetic grid averaging, BJT init+step+8h, CF0/PF1–50, explicit same-init reforecast sample sets and fixed thresholds are retained. No fallback, model fitting or anomaly correction. Equal-day/time-weighted and stage-scalar differences are all preserved. raw/residualized model designs must match the saved NCVI stage table before replay.

ERA5 curve acceptance requires real coverage and verified inline provenance; unknown evidence leaves era5_reference_qc FAIL and formal_process_allowed false. S2S numerical audit proceeds independently. A failed stage replay also blocks formal acceptance. Existing files are read-only; rerunning Cell6 cannot overwrite a previous export. Choose a new run directory and restart Cells1–6 for another run.

HPC execution is not possible from this cloud environment. All cells are unexecuted; synthetic tests and schema checks are code-preservation evidence only. Please return Cell2/3 selections and Cell5/6 QC logs, the replay CSV and full manifest after HPC execution. No GRIB upload is needed.
