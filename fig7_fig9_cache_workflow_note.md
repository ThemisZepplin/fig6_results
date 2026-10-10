# Fig.7–Fig.9 cache workflow note

## Purpose
This workflow separates expensive science processing from final style/layout plotting. Run the source figure scripts to export plot-ready NetCDF caches, then use the `paper_fig*_from_cache.py` scripts for fast style-only revisions.

## Source files and caches
- Fig.7 source: `cell_fig7_stageII_daily_synoptic_paper.py`
  - Cache: `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig7_daily_synoptic_plot_cache.nc`
  - QC: `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig7_daily_synoptic_cache_qc.csv`
  - From-cache script: `paper_fig7_from_cache.py`
- Fig.8 source: `cell_hovmoller_tp_z500_obs_s2s_stageII.py`
  - Cache: `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig8_hovmoller_plot_cache.nc`
  - QC: `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig8_hovmoller_cache_qc.csv`
  - From-cache script: `paper_fig8_from_cache.py`
- Fig.9 source: `cell_leadtime_total_hotcold_composite_maps.py`
  - Cache: `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig9_leadtime_tp_z500_plot_cache.nc`
  - QC: `/data1/huangy/fig6/NC/v12/paper_figures/cache/fig9_leadtime_tp_z500_cache_qc.csv`
  - From-cache script: `paper_fig9_from_cache.py`

## Minimal redraw
Use `paper_fig789_from_cache_example.py` after caches exist. It calls all three from-cache plotting functions and writes PNG/PDF files to `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/`.

## Style-only revision targets
After caches are exported, update only the from-cache scripts to adjust figure size, colorbar placement, labels, line widths, panel spacing, or typography. Do not rerun raw ERA5/MSWEP/S2S processing unless the science definition or grouping changes.
