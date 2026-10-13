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

## Existing files: reload only the final plotting entry

Server logs confirm the June 8 daily cache and TP/Z500 significance already
exist under `/data1/huangy/fig6/NC/v12/paper_figures/` (not `paper_figures.0`).
This is sufficient to proceed with the daily redraw; do not rerun science
exports or edit NetCDF values to repair metadata compatibility.

```python
figures = runpy.run_path(str(repo / 'paper_precip_circulation_revision_from_cache.py'))
figures['plot_figS6_from_cache']()
# If the existing June 12 files are also available:
# figures['plot_fig6_from_cache']()
```

Fig.S6 uses these existing inputs, with all 12 panels (a–l):

- `cache/fig7_daily_synoptic_init2023-06-08_plot_cache.nc`
- `significance/fig7_init2023-06-08_spatial_significance_tp.nc`

It writes `final_from_cache/jgr_6p5in/figS6_daily_synoptic_init2023-06-08.png`
and `.pdf`. Z500 significance is not loaded or rendered. Ordinary labeled
Z500 difference contours and TP local two-sided Welch p < 0.05 markers remain.
Reload this plotting cell for future layout changes; no scientific export is
needed when its input files already exist and pass validation.

June 8 Hovmöller files must be checked separately on the server. A successful
daily Fig.S6 redraw does not establish that Hovmöller exports were generated.
The final Fig.7 needs both initializations' Hovmöller caches and TP tests:

```python
root = Path('/data1/huangy/fig6/NC/v12/paper_figures')
hov_inputs = [
    root / 'cache/fig8_hovmoller_init2023-06-08_plot_cache.nc',
    root / 'significance/fig8_init2023-06-08_hovmoller_significance_tp.nc',
    root / 'cache/fig8_hovmoller_plot_cache.nc',
    root / 'significance/fig8_hovmoller_significance_tp.nc',
]
missing = [str(path) for path in hov_inputs if not path.is_file()]
if missing:
    print('Hovmöller inputs requiring separate verification:', missing)
else:
    figures['plot_fig7_from_cache']()  # also validates metadata/coords/fields
```

Only confirmed missing products need the relevant export below. If an existing
file fails a coordinate, grouping or difference-consistency guard, investigate
that mismatch first; this compatibility fix is not a reason to overwrite data.

## First-time cache generation

Only for genuinely missing daily products, load the complete science cell and
run the corresponding missing-product export; skip files already generated.
The revised science cells do not automatically export on loading. Existing
June 8 and June 12 caches are retained when their metadata, fields and stored
shared levels validate. Do not regenerate scientific fields merely because
legacy initialization attribute names differ.

```python
daily = runpy.run_path(str(repo / 'cell_fig7_stageII_daily_synoptic_paper.py'))
daily['export_fig7_plot_cache']()                  # June 12 + common levels
daily['export_fig7_spatial_significance']()        # June 12 TP/Z500 + QC
daily['export_fig7_20230608_plot_cache']()         # complete June 8 + obs
daily['export_fig7_20230608_spatial_significance']() # June 8 TP/Z500 + QC
```

Only if the separate Hovmöller file check confirms missing products, load its
complete science cell and execute the needed exports. Welch inputs are member-level
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

Cache and TP significance resolve the first valid date from `initialization`,
then `init_date`, then `main_init_date`. Thus old June 8 `init_date` overrides
a stale June 12 `main_init_date`. A valid but incorrect higher-priority date
is rejected. Blank or invalid primary values allow a valid lower-priority
attribute. Fully missing significance metadata retains only the historical
June 12 compatibility case; it is never inferred to be June 8. Invalid-only
metadata is rejected. Date, coordinate, grouping and difference-field checks
remain mandatory. New exports add
explicit init and group-member metadata. Cache variables, dates, units and
Total grouping are checked. Wrong init, window, coordinates or composite
differences fail instead of silently mixing fields. Keep the grouping file
unchanged between exports of the paired initializations.
