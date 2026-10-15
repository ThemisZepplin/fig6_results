# Fig.6 / Fig.7 revision source inventory (2026-10-15)

Recorded before implementation. Base dev/jgr-major-revision:
096cad22cc84cf8b14a5ee70a5db80c762891a65.
Selected source PR #13: 2f97ecc6bbe6544aa08f6dc36ec3c74b20185058.
No previous PR is merged. No AGENTS.md exists at the development-branch root.
No executor/worktree is available in this session; remote branch and SHA checks
replace local Git status inspection. Runtime checks must run in CI/server.

## Numbering and scope

- New main Fig.6: four initializations June 1/5/8/12, BJT June 18–20 mean.
- Fig.7: existing seven-panel Total Hovmoller, June 8 left / June 12 right.
- Former daily Fig.6/S6: preserve 12-panel renderers as SI candidates; final SI
  numbering is deliberately deferred. No existing output file is removed.
- Former Fig.9 in previous conversations = former manuscript Fig.8: four-init
  June 14–24 mean map. Archive it; it is not the new early-window MLR Fig.8.
- Do not modify MLR code or assign a spatial product to the final MLR Fig.8.

## Scientific source and native grids

`cell_leadtime_total_hotcold_composite_maps.py` is copied unchanged from PR #13.
Raw readers, GRIB variable/level selection, CF0+PF1–50, TP step differencing,
BJT calendar-day apportionment, unit conversions and Total grouping CSV
selection are reused in an isolated namespace. The new exporter sets only that
namespace's target analysis dates to June 18/19/20; it never calls historical
11-day export functions or changes their default globals in another session.
GROUP_FILE remains the source v9 file; grouping_stage remains Total, with the
grouping window June 14–24. Four independent Hot17/Cold17 pairs are retained.

New target fields, after three valid BJT days per selected member:
- tp_hot/cold/diff(init,s2s_lat,s2s_lon): mean daily TP, mm day^-1.
- z500_abs_hot/cold/diff(init,s2s_lat,s2s_lon): absolute Z500 or signed group
  difference of absolute Z500, gpm. Never climatological anomalies.
- tp_obs_ref(mswep_lat,mswep_lon): MSWEP mean of three daily amounts.
- z500_abs_obs_ref(era5_lat,era5_lon): ERA5 mean absolute height, same three days.
- tp_abs_levels / tp_diff_levels: common levels computed across four inits and
  the observed reference using the existing source level functions.
No raw members, forecast steps or hourly arrays are stored in the plot cache.
Native observation and forecast grids remain separate; no interpolation.

Local two-sided Welch tests use 17 member-level three-day means per group,
independently for each initialization and each variable. The existing Welch/BH
kernels are reused. No pooling days/inits, averaging p-values/masks, or reuse of
the 11-day significance. Display TP local p<0.05 only; retain Z500 statistics
as an independent archive without rendering Z500 significance.

## Files and products planned

- New exporter: cell_fig6_interruption_totalgroup_export.py
- New renderer: paper_fig6_interruption_from_cache.py
- Revised Hovmoller renderer: paper_fig8_from_cache.py (historical filename)
- Preserved daily renderer: paper_fig7_from_cache.py (historical filename)
- Preserved Total map renderer: paper_fig9_from_cache.py (historical filename)
- Hovmoller scientific source: cell_hovmoller_tp_z500_obs_s2s_stageII.py,
  copied unchanged, available only if server Hovmoller files are missing.
- New wrapper: paper_fig67_revision_from_cache.py, separate unambiguous entries.
- Instructions: fig6_fig7_revision_jupyter.md, including paths and missing-file
  checks, first export and later from-cache-only redraws.
- Focused tests and CI: synthetic exports, strict window/group/mask validation,
  actual Matplotlib geometry and closed-contour label rendering. Synthetic
  results must not be described as real-data PNG/PDF validation.

Root: /data1/huangy/fig6/NC/v12/paper_figures/
New stem: fig6_interruption_0618_0620_totalgroup_fourinit
cache/<stem>_plot_cache.nc and _cache_qc.csv;
significance/<stem>_significance_tp.nc, _significance_z500.nc,
_significance_qc.csv; final_from_cache/jgr_6p5in/<stem>.png/.pdf.
Hovmoller input filenames remain unchanged. Its revised output uses a new
_fig7..._revised heading/date stem to preserve prior finalized figures.
