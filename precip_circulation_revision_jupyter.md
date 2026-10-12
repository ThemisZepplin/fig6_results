# Jupyter workflow

Use a checked-out directory containing all seven scripts. Scientific exports
require the existing ERA5/MSWEP/S2S files and Total-period Tmax grouping CSV at
the paths configured in the science cells. Dependencies: numpy, pandas, xarray,
scipy, statsmodels, matplotlib, cartopy, cfgrib/eccodes for GRIB reads, and a
NetCDF backend. The existing data definitions and processing rules are retained.

Running a code cell defines its functions in that cell's namespace; it does not
install a same-named Python module. Use absolute paths and separate namespaces
as below to avoid working-directory dependencies and colliding helper names.
Replace the example checkout path with the actual local checkout.

```python
from pathlib import Path
import runpy
repo = Path('/absolute/path/to/fig6_results').resolve()
```

## First-time cache generation

Load the complete daily science cell, then run the following functions.
The revised science cells do not automatically export on loading. Existing
June 12 caches can be retained if metadata, fields and stored shared levels
validate; regenerate both when shared levels are absent or inconsistent.

```python
daily = runpy.run_path(str(repo / 'cell_fig7_stageII_daily_synoptic_paper.py'))
daily['export_fig7_plot_cache']()                  # June 12 + common levels
daily['export_fig7_spatial_significance']()        # June 12 TP/Z500 + QC
daily['export_fig7_20230608_plot_cache']()         # complete June 8 + obs
daily['export_fig7_20230608_spatial_significance']() # June 8 TP/Z500 + QC
```

Load the complete Hovmöller science cell next. Welch inputs are member-level
cosine-latitude-weighted fields; group-average fields are never its samples.

```python
hov = runpy.run_path(str(repo / 'cell_hovmoller_tp_z500_obs_s2s_stageII.py'))
hov['export_fig8_plot_cache']()                    # June 12 + common levels
hov['export_fig8_hovmoller_significance']()        # June 12 TP/Z500 + QC
hov['export_fig8_20230608_plot_cache']()           # June 8 + common obs
hov['export_fig8_20230608_hovmoller_significance']() # June 8 TP/Z500 + QC
```

For Fig.9, load the complete leadtime cell and generate only the established
Total-period cache/significance if they are missing:

```python
lead = runpy.run_path(str(repo / 'cell_leadtime_total_hotcold_composite_maps.py'))
lead['export_fig9_plot_cache']()
lead['export_fig9_spatial_significance']()
```

Review the cache/QC manifest in `precip_circulation_revision_inventory.md`.
QC records provenance, dimensions, units, finite values, group sizes and
member-composite consistency. June 8 outputs have independent filenames.

## Render and later style-only changes

Run just this cell after the caches exist; reload it after editing styles:

```python
figures = runpy.run_path(str(repo / 'paper_precip_circulation_revision_from_cache.py'))
figures['plot_fig6_from_cache']()
figures['plot_figS6_from_cache']()
figures['plot_fig7_from_cache']()
figures['plot_fig9_from_cache']()
```

These entries read only plot-ready caches and TP significance NetCDF files.
They neither read raw data/group CSVs nor recalculate groups, composites,
Hovmöllers or tests. Fig.S6 can render using its own cache and TP file alone.
The Hovmöller figure verifies common observations and color levels between
both caches. All final entries preserve ordinary labeled Z500 difference
contours without any significance-width overlay.

Legacy June 12 significance without initialization/member metadata is accepted
only as June 12, with coordinate and exact composite matching. New exports add
explicit init and group-member metadata. Cache variables, dates, units and
Total grouping are checked. Wrong init, window, coordinates or composite
differences fail instead of silently mixing fields. Keep the grouping file
unchanged between exports of the paired initializations.
