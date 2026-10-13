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

Each function generates **one initialization only**: `plot_fig6_from_cache()`
produces June 12 Fig.6, and `plot_figS6_from_cache()` produces June 8 Fig.S6.
Calling either function never generates the other figure. Both use the same
6.5 × 7.1-inch, four-row/three-column layout with panels a–l.

### Check the four daily inputs before either method

This check lists exact missing paths without exporting caches, changing files,
or substituting another initialization. Readiness is independent for each figure.

```python
from pathlib import Path
root = Path('/data1/huangy/fig6/NC/v12/paper_figures')
daily_inputs = {
    'Fig.6': [
        root / 'cache/fig7_daily_synoptic_plot_cache.nc',
        root / 'significance/fig7_spatial_significance_tp.nc',
    ],
    'Fig.S6': [
        root / 'cache/fig7_daily_synoptic_init2023-06-08_plot_cache.nc',
        root / 'significance/fig7_init2023-06-08_spatial_significance_tp.nc',
    ],
}
ready = {}
for figure, paths in daily_inputs.items():
    missing = [str(path) for path in paths if not path.is_file()]
    ready[figure] = not missing
    if missing:
        print(figure + ' missing inputs:')
        for path in missing:
            print('  ' + path)
    else:
        print(figure + ': both input files exist')
```

File existence alone is not validation; rendering still performs the existing
initialization, date, coordinate, unit, grouping and difference-field checks.
The correct root is `paper_figures/`, never the historical `paper_figures.1/`.

### Method 1: paste the full updated daily plotting script into the notebook

Paste the entire latest `paper_fig7_from_cache.py` into a notebook cell and run
it to define the functions. Then run both explicit calls below. This does not
install a Python module or require importing the script by module name.

```python
if ready['Fig.6']:
    fig6_paths = plot_fig6_from_cache()
    print("Fig.6:", fig6_paths)

if ready['Fig.S6']:
    figS6_paths = plot_figS6_from_cache()
    print("Fig.S6:", figS6_paths)
```

The two blocks are independent and may also be run in separate cells.
Missing inputs are reported by the preceding check; no automatic regeneration
or replacement takes place.

### Method 2: load the updated scripts on disk through the unified runpy entry

Ensure the **disk copy** of `paper_fig7_from_cache.py` and the unified entry
`paper_precip_circulation_revision_from_cache.py` are updated to this revision.
Pasting definitions into a notebook does not update either disk file. Use the
absolute checkout path, not the notebook working directory, and call both entries:

```python
import runpy
repo = Path('/absolute/path/to/fig6_results').resolve()
figures = runpy.run_path(str(repo / 'paper_precip_circulation_revision_from_cache.py'))
if ready['Fig.6']:
    fig6_paths = figures['plot_fig6_from_cache']()
    print("Fig.6:", fig6_paths)

if ready['Fig.S6']:
    figS6_paths = figures['plot_figS6_from_cache']()
    print("Fig.S6:", figS6_paths)
```

Both methods write PNG/PDF under
`/data1/huangy/fig6/NC/v12/paper_figures/final_from_cache/jgr_6p5in/`:

- `fig6_daily_synoptic_init2023-06-12.png` / `.pdf`
- `figS6_daily_synoptic_init2023-06-08.png` / `.pdf`

Z500 significance is not loaded or rendered. Ordinary labeled Z500 difference
contours and TP local two-sided Welch p < 0.05 markers remain. For subsequent
layout changes, reload only the updated plotting definitions/entry and repeat
both explicit calls. Existing scientific products are reused without re-export.

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

## TP hatch / contour-label style update: redraw four final figures only

Update the disk copies of all three `paper_fig*_from_cache.py` scripts before
using runpy. Do not load science cells or call any export function for this
style revision. Reuse the existing plot-ready caches and TP significance.
After verifying each figure's inputs, explicitly call each entry separately:

```python
figures = runpy.run_path(str(repo / 'paper_precip_circulation_revision_from_cache.py'))
fig6_paths = figures['plot_fig6_from_cache']()
print('Fig.6:', fig6_paths)
figS6_paths = figures['plot_figS6_from_cache']()
print('Fig.S6:', figS6_paths)
fig7_paths = figures['plot_fig7_from_cache']()
print('Fig.7 Hovmöller:', fig7_paths)
fig9_paths = figures['plot_fig9_from_cache']()
print('Fig.9:', fig9_paths)
```

Each entry renders only its own figure. If inputs are missing or validation
fails, stop and investigate the named product; do not export automatically or
substitute another initialization. Fig.7 still requires both Hovmöller inits,
independent of the daily-map files. All output names and figure layouts remain.

For full-cell paste use, paste the latest complete corresponding plotting file
and invoke `plot_fig6_from_cache()` / `plot_figS6_from_cache()` (daily script),
`plot_fig7_hovmoller_from_cache()` (Hovmöller script), or
`plot_fig9_from_cache()` (leadtime script). The visual helpers are intentionally
included in each file: no new project-local module is required. Existing
Matplotlib supplies the geometry helper; scipy/statistics are not recomputed.

Black dotted hatching remains the local two-sided Welch p < 0.05 mask, with
17 members per group and no multiple-comparison correction for the displayed
markers. Hatch density, pattern, covered area and order are unchanged; 0.60 pt
is the updated stroke default. The number of hatch dots is not a count of
significant grid points. Labels use the same number format, font and white
stroke; placement occurs on actual contour paths after layout draw. One safe
label per selected contour level/panel avoids redundant repetitions. Contours
with no safe location are retained with an explicit placement log.

Until the four real figures are redrawn at their final JGR/AGU dimensions,
0.60 pt is a proposed default rather than verified dark-fill readability.
Inspect real PNG/PDF for label clipping/collisions and dark-fill contrast.
