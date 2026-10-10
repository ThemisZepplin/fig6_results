# JGR 6.5-inch Layout Audit: Figures 7, 8, and 10

## Scope and provenance

This pass changes only the three from-cache plotting scripts. No science-processing script, plot-ready NetCDF, significance NetCDF, QC CSV, or scientific audit was modified.

| Final figure | Plotting script | Historical plot cache | Significance inputs |
|---|---|---|---|
| Fig. 7 | `paper_fig7_from_cache.py` | `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig7_daily_synoptic_plot_cache.nc` | `fig7_spatial_significance_tp.nc`; `fig7_spatial_significance_z500.nc` |
| Fig. 8 | `paper_fig8_from_cache.py` | `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig8_hovmoller_plot_cache.nc` | `fig8_hovmoller_significance_tp.nc`; `fig8_hovmoller_significance_z500.nc` |
| Fig. 10 | `paper_fig9_from_cache.py` | `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig9_leadtime_tp_z500_plot_cache.nc` | `fig9_spatial_significance_tp.nc`; `fig9_spatial_significance_z500.nc` |

The Fig. 9-named cache is retained as a historical workflow artifact; the formal publication output is Fig. 10.

## Fixed physical layout

| Figure | Figure size | Expected 600-dpi PNG | Expected PDF page width | Panel order | Colorbars |
|---|---:|---:|---:|---|---|
| Fig. 7 | 6.5 × 7.1 in | 3900 × 4260 px | 468 pt | 4 rows × 3 dates, (a)–(l) | Two horizontal bars directly below the maps |
| Fig. 8 | 6.5 × 5.1 in | 3900 × 3060 px | 468 pt | Observed, Hot17, Hot17 − Cold17, Cold17 | Two horizontal bars aligned beneath the two columns |
| Fig. 10 | 6.5 × 6.4 in | 3900 × 3840 px | 468 pt | Four initialization rows × three groups, plus (m) Observed | Two stacked horizontal bars to the right of (m) |

All descriptive figure-level titles were removed. The top margins now retain only short column or panel titles. Exports do not use `tight_layout()` or `bbox_inches="tight"`.

## Typography

- Panel letters: 10.5 pt, bold, compact white backing.
- Column headers: 10.0 pt.
- Row headers and panel titles: 9.5 pt.
- Axis labels: 9.5 pt.
- Geographic/date ticks: 8.5 pt.
- Colorbar ticks: 8.5 pt; colorbar labels: 9.0 pt.
- Z500 contour labels: 8.5 pt with the existing white halo.
- PDF and PostScript font type: 42.

## Significance display

The main figures retain uncorrected local *p* < 0.05 from two-sided Welch independent-samples tests with member as the sample dimension (Hot17 = 17; Cold17 = 17). TP and Z500 masks remain independent. TP is shown by black stippling; significant Z500 segments are red solid for positive differences and blue dashed for negative differences. Significance appears only in Hot17-minus-Cold17 panels.

Runtime counts are read from the prepared significance NetCDFs. They were not available in this container because the `/data1` prepared files and scientific Python dependencies are absent. The expected audited Fig. 8 counts are TP = 38 and Z500 = 81; Fig. 7 and Fig. 10 counts must be confirmed against their existing runtime QC on HPC.

## Outputs

- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/fig7_daily_synoptic_evolution.png`
- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/fig7_daily_synoptic_evolution.pdf`
- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/fig8_hovmoller_propagation.png`
- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/fig8_hovmoller_propagation.pdf`
- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/fig10_leadtime_tp_z500_composites.png`
- `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/fig10_leadtime_tp_z500_composites.pdf`

## Validation status

- Python byte-compilation: passed for all three from-cache scripts.
- `git diff --check`: passed.
- Raw-data/grouping-read grep: passed with no matches.
- Cache/significance SHA256 before/after: unavailable in this container because no prepared NetCDFs are mounted; no code path writes these inputs.
- PNG pixel and PDF page-size checks: pending HPC rendering.
- Font embedding/selectability: pending HPC rendering.
- Visual checks for clipping, overlap, whitespace, contour-label readability, NCHN placement, and PNG/PDF equivalence: pending HPC rendering.

No automatic visual claim is made without the prepared files and plotting dependencies.
