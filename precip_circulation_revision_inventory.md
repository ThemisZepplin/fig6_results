# JGR precipitation–circulation revision inventory

Base: `dev/jgr-major-revision` at `096cad22cc84cf8b14a5ee70a5db80c762891a65`.
Source: PR #11 head `e477e6f74261b493b3c4d23af3ce9cfadd7231c6`.
The latest base contains none of the six scripts below. Only these six source
files were extracted, rather than merging PR #11. No historical branch or PR
was changed. No Fig.10 subperiod or MLR code was introduced.

| Historical → final | Science cell | Historical plotting file → final entry | Initialization / BJT dates |
| --- | --- | --- | --- |
| Fig.7 → Fig.6 | `cell_fig7_stageII_daily_synoptic_paper.py` | `paper_fig7_from_cache.py` → `plot_fig6_from_cache()` | June 12 / June 18–20 |
| Fig.7 appendix → Fig.S6 | same | same → `plot_figS6_from_cache()` | June 8 / June 18–20 |
| Fig.8 → Fig.7 | `cell_hovmoller_tp_z500_obs_s2s_stageII.py` | `paper_fig8_from_cache.py` → `plot_fig7_hovmoller_from_cache()` | June 8 + June 12 / June 14–24; 35–44N |
| Fig.10 → Fig.9 | `cell_leadtime_total_hotcold_composite_maps.py` | `paper_fig9_from_cache.py`: historical `plot_fig10_from_cache()` → `plot_fig9_from_cache()` | June 1, 5, 8, 12 / June 14–24 |

`paper_precip_circulation_revision_from_cache.py` provides unambiguous final
`plot_fig6_from_cache`, `plot_figS6_from_cache`, `plot_fig7_from_cache`, and
`plot_fig9_from_cache` entries. The old daily-map `plot_fig7_from_cache` alias
still outputs Fig.6; do not confuse it with the new Hovmöller Fig.7 entry.

## Cache and QC manifest

All paths below are relative to `/data1/huangy/fig6/NC/v12/paper_figures/`.
The June 12 and leadtime paths are retained for historical compatibility.

| Product | Plot-ready cache | Cache QC | TP / Z500 significance | Significance QC |
| --- | --- | --- | --- | --- |
| Fig.6 | `cache/fig7_daily_synoptic_plot_cache.nc` | `cache/fig7_daily_synoptic_cache_qc.csv` | `significance/fig7_spatial_significance_tp.nc` / `fig7_spatial_significance_z500.nc` | `significance/fig7_spatial_significance_qc.csv` |
| Fig.S6 | `cache/fig7_daily_synoptic_init2023-06-08_plot_cache.nc` | `cache/fig7_daily_synoptic_init2023-06-08_cache_qc.csv` | `significance/fig7_init2023-06-08_spatial_significance_tp.nc` / `fig7_init2023-06-08_spatial_significance_z500.nc` | `significance/fig7_init2023-06-08_spatial_significance_qc.csv` |
| Fig.7 June 12 | `cache/fig8_hovmoller_plot_cache.nc` | `cache/fig8_hovmoller_cache_qc.csv` | `significance/fig8_hovmoller_significance_tp.nc` / `fig8_hovmoller_significance_z500.nc` | `significance/fig8_hovmoller_significance_qc.csv` |
| Fig.7 June 8 | `cache/fig8_hovmoller_init2023-06-08_plot_cache.nc` | `cache/fig8_hovmoller_init2023-06-08_cache_qc.csv` | `significance/fig8_init2023-06-08_hovmoller_significance_tp.nc` / `fig8_init2023-06-08_hovmoller_significance_z500.nc` | `significance/fig8_init2023-06-08_hovmoller_significance_qc.csv` |
| Fig.9 | `cache/fig9_leadtime_tp_z500_plot_cache.nc` | `cache/fig9_leadtime_tp_z500_cache_qc.csv` | `significance/fig9_spatial_significance_tp.nc` / `fig9_spatial_significance_z500.nc` | `significance/fig9_spatial_significance_qc.csv` |

Both significance filenames in each row reside in `significance/`. Final plots
read only TP significance; independent Z500 diagnostics remain available for
traceability. Daily significance runtime audit sections are separated by init
in `significance/fig7_fig9_spatial_significance_audit.md` (see the source
`SPATIAL_SIGNIFICANCE_AUDIT_PATH` constant for the configured audit filename).
Hovmöller June 12 retains `significance/fig8_hovmoller_significance_audit.md`;
June 8 diagnostics and provenance are in its independent QC CSV and NetCDF.

June 8 previously participated in common color-level calculation only. This
revision adds equivalent daily independent cache/significance functionality
under the requested export names, and separate Hovmöller June 8 exports.
Both daily caches include the full identical MSWEP/ERA5 observation row.
Both export routines calculate absolute and difference TP levels from both
initializations. No grouping is recomputed for the diagnostic window.

## Final output paths

Directory: `/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/`.
Each stem produces `.png` and `.pdf`, without replacing historical figure stems:

- `fig6_daily_synoptic_init2023-06-12`
- `figS6_daily_synoptic_init2023-06-08`
- `fig7_hovmoller_total_init2023-06-08_2023-06-12`
- `fig9_leadtime_tp_z500_composites`
