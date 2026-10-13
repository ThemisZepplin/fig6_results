# Preliminary Fig.5 / Fig.11 — manual HPC Jupyter execution

Copy the SIX complete code cells of `process_fig5_fig11_diagnostic_standalone.ipynb` into HPC Jupyter and execute in order. All functions are embedded; no custom Python modules, Git or CLI is required. Scientific packages numpy/pandas/matplotlib and existing source CSVs are required. Edit paths only in Cell1; choose a fresh output directory outside NCVI run_001.

1. Configure v12 daily raw/TCC/Tmax and stage tables, five stored model tables plus their input provenance manifest, NCVI run_001 and optional legacy ERA5 physical reference CSV. Existing comparison/NCVI manifests are scientific inputs, not external configuration. ERA5_PHYSICAL=None explicitly omits all ERA5; otherwise missing or incomplete configured reference aborts. NCVI ERA5 is always excluded. Unit definitions remain v12 source conventions; CSV agreement checks do not independently validate GRIB unit attributes.
2. Verify model/source hashes, exact 2×11×51 coverage, unchanged physical stage aggregation (1224 comparisons), frozen saved-model provenance and v12 rank groups. No legacy S2S group curves are reused.
3. Verify all run_001 manifest output hashes and S2S source PASS, exact daily anomaly keys/units and daily-internal replay. Current historical FAIL is allowed exclusively by this user-authorized diagnostic route, copied to the new manifest and not edited. NCVI groups use 17 original members each. Optional ERA5 physical values are read independently from era5 column; s2s_all_mean never substitutes Middle17. This reference retains its original processing convention and is not promoted to a newly verified physical source.
4. Replay saved full models and compute frozen Stage-I SM–SHF partial fits for both versions, including day18 out-of-window. Other predictors remain at saved training means. Physical factor values are copied identically across versions. No fitting or importance calculation.
5. Draw Fig5 Total (5×2) and Fig11 Stage-I raw/residualized (6×2), left06-08/right06-12. Colors black/red/blue/green for ERA5/Hot17/Middle17/Cold17; common row y-ranges, equal axes boxes, centralized legend, fixed width6.5in. Stage-I boundary06-18 00:00 BJT and shaded day18. All figures explicitly PRELIMINARY/DIAGNOSTIC, with NCVI historical FAIL/ERA5-unverified annotation. True renderer text collision and clipping QC must pass; it is not bypassed. Inspect plotted NCVI curves and physical-variable relationships manually after the actual HPC run.
6. Recheck input/run_001 hashes, export new tables/QC/manifest, PNG600dpi and editable vector PDF. No formal acceptance is claimed. Missing provenance, scientific FAIL or unrun verification are preserved distinctly from diagnostic layout PASS. Original run_001 files are never changed.

Products in new output directory:

- Fig5_Total_DIAGNOSTIC.png/.pdf
- Fig11_StageI_raw_DIAGNOSTIC.png/.pdf
- Fig11_StageI_residualized_DIAGNOSTIC.png/.pdf
- table_fig5_total_DIAGNOSTIC.csv, table_fig11_stageI_DIAGNOSTIC.csv
- v12_daily_stage_qc.csv, current_groups.csv, partial_fitted_audit.csv, layout_qc.csv
- executed_cells.json and diagnostic_process_manifest.json (formal_process_allowed=false, immutable original status/hashes)

Methods distinctions are preserved: v12 Tmax arithmetic spatial average, TCC source date convention represents08:00–08:00 BJT, Stage-I day18 is out-of-training projection. Partial-fit differences reflect parameterization sensitivity. Nothing redefines members or changes MLR results. FigS11 existing inputs/outputs are not touched.

Current HPC evidence is user-reported: source PASS,1122 daily NCVI rows,204 stage comparisons, internal agreement about1e-16, archived-model difference4.748331733361866e-08 and strict1e-8 FAIL, ERA5 provenance incomplete. Cloud lacks run_001 so real curves/PDFs are not generated here. Cloud tests use synthetic series to check the six-cell independence, axes counts/width/alignment, renderer text extents and stage boundary; they are not scientific validation. Return Cell2–6 logs, new plotting CSVs, manifest and PNG/PDF after running on HPC.

## Updated Fig.5 / Fig.11 / Fig.S11 organization

The same complete six-cell Notebook now generates all five figures. FigS11 raw and residualized are each2×2, columns06-08/06-12 and rows Tmax–raw SM / Tmax–version-specific partial fit. Both member diagnostics are independently recomputed with each saved full model; Tmax and raw SM are asserted identical across versions. Physical Fig11 series are also asserted identical. No MLR fitting occurs.

Cell4 writes FIVE dedicated plot CSVs into a fresh output directory:

- fig5_total_timeseries_plot.csv
- fig11_stageI_timeseries_raw_plot.csv
- fig11_stageI_timeseries_residualized_plot.csv
- figS11_stageI_scatter_raw.csv
- figS11_stageI_scatter_residualized.csv

Cell5 verifies hashes and reads those files back before rendering. Plotting does not depend on old Notebook objects. Cell6 adds group_membership_audit.csv, layout_qc.csv, partial_fitted_audit.csv and consistency_manifest.json, including original NCVI manifest/FAIL and explicit formal_process_allowed=false. Existing output directories/files cannot be overwritten. If Cell4 succeeds but a later cell fails, fix/retry the later cell without rerunning Cell4 or choose a new directory and restart the entire workflow.

Figure faces no longer contain PRELIMINARY/DIAGNOSTIC watermark text. Diagnostic status stays in figure filenames (_DIAGNOSTIC), this caption note and the manifest. ERA5 NCVI remains omitted pending provenance. PNG/PDF sizes, colors, row/common ranges, renderer QC and day18 shading remain unchanged. Two additional filenames: FigS11_StageI_raw_DIAGNOSTIC and FigS11_StageI_residualized_DIAGNOSTIC.

Caption draft: Curves show the current v12 stage-ranked groups; physical series are unchanged across raw/residualized models. The SM–SHF partial fitted Tmax diagnostic uses each full model's intercept, standardized coefficients and training scalers, holding all other predictors at training means. Daily residualized heat flux is projected with frozen Stage-I residualization parameters. June18 is outside the14–17 training window. Scatter first-row Tmax/raw SM is identical across versions. Diagnostic outputs retain unresolved historical NCVI strict replay and ERA5 provenance status; they are not formal paper acceptance.

This updated subsection supersedes the older product names and figure-watermark description above.

## Final plotting polish (baseline90b4982)

All calculation cells and five plot CSV definitions remain unchanged. Scatter descriptive statistics use all51 individual member pairs for each initialization/version, regardless of colored groups. The annotation is Pearson r, descriptive R²=r² and n51; a simple least-squares line is drawn. This r² is NOT the full six-predictor MLR R². Statistics and line coefficients are saved in scatter_descriptive_statistics.csv. No scientific model is fitted/refitted.

Top headers show the two initialization dates once; panels use concise letters. Time-series date labels appear only on the bottom row to reduce repetition. Shared variable scales remain; scatter row scales now pool both initialization columns and both versions. Physical width6.5in; heights9.0in(Total),10.5in(Stage-I),5.9in(scatter). Base/tick typography9pt, labels8pt, headers10pt and descriptive annotations8pt. Source values/groups and date windows do not change. Day18 shading and frozen projection boundary remain.

Visual encoding: ERA5 black circle/solid; Hot17 red upright triangle/solid; Middle17 blue square/dashed; Cold17 green downward triangle/dotted. Single legend per figure. PDF uses embedded TrueType; SVG uses editable text; PNG remains600dpi. Renderer collision/cropping/equal-box QC is mandatory. If PyMuPDF is available, Cell6 also reads each actual PDF, verifies6.5in physical width and text clipping/editable spans and records actual font sizes in pdf_font_qc.csv. Without it, PDF audit is explicitly NOT_RUN, never assumed PASS. Install PyMuPDF in the HPC environment if that additional actual PDF check is desired; synthetic/cloud QA does not validate genuine HPC plots.

The manifest now identifies90b4982 as the reviewed baseline (not HPC HEAD) and captures the actual six-cell code text SHA256; the snapshot is the authoritative execution version. History statuses and formal_process_allowed=false are preserved. SVG is additional output. No large diagnostic watermark appears on figures.

## AGU/JGR export size repair (source205c4b8)

Only Cell5–6 change in this repair. Figure physical dimensions: Fig5 165.1×218.44mm (6.5×8.6in); Fig11 each165.1×226.06mm (6.5×8.9in); FigS11 each165.1×149.86mm (6.5×5.9in). All within requested170×228mm maximum. Fonts are preserved, not scaled with canvas; full-size renderer clipping/collision/equal-axis tests rerun on shorter canvases.

Actual PDF QC verifies maximum width/height AND exact intended canvas, editable text and clipping. Readability criteria require ordinary spans>=8pt and PDF-flagged superscript/subscript spans>=5.5pt, recorded separately. These explicit workflow criteria are not represented as an AGU-mandated minimum. Unflagged Unicode superscripts embedded in ordinary labels remain ordinary spans. Undersized fonts/oversized pages produce FAIL with reasons, saved in pdf_font_qc.csv/manifest, then stop acceptance; PyMuPDF missing remains NOT_RUN. Renderer QA is not actual PDF QA. All export formats PDF/SVG/PNG600 remain, with no overwrite of existing SVG or other outputs.

Manifest notebook_source_commit=205c4b8 identifies the actual source before this repair; historical_reference_baseline=90b4982 is separate. Neither is HPC HEAD; execution_git_head remains null. Exact repaired code is identified by executed_code_sha256 and executed_cells.json. Calculations, five plot-table definitions, original NCVI FAIL and ERA5 exclusion are unchanged.
