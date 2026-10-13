# Layout and validation audit

- Fig.6 and Fig.S6: 4 rows × 3 columns, 12 data panels labeled a–l in row
  order; complete observations, Hot17, Cold17, and Hot17−Cold17. Date columns
  remain June 18, 19, 20. Shared TP levels are generated using both inits.
- Fig.7: seven data axes, a shared centered observation (a), then June 8/12
  Hot17 (b/c), Cold17 (d/e), difference (f/g). All axes span equal GridSpec
  rows and two equal column slots; the top panel is offset by one slot to
  center it. Runtime verifies equal width/height. Longitude/time limits are
  common; 114E/119E reference lines and 35–44N averaging are preserved.
- Fig.9: original four initialization rows × three groups plus observation
  reference, a–m, unchanged order/window. Only final numbering/output stem,
  TP-only significance loading, and removal of Z500 overlay change rendering.
- No final renderer adds a suptitle. Ordinary signed, styled, labeled Z500
  difference contours remain. Only TP local Welch masks are rendered.

## Checks actually executed

All eight newly added Python files (six extracted/revised scripts, final entry
wrapper, and test script) passed `python -m py_compile`. `git diff --check`
passed. Two checks in `tests/check_precip_circulation_revision.py` passed:
static rendering/processing contracts, and real Matplotlib seven-axis geometry
using the actual renderer's GridSpec definition. These are structural checks,
not a plot-ready NetCDF or scientific-data integration run.

AST comparison against the exact PR #11 source found 214 existing science
functions unchanged. Changed existing functions are only daily cache and
significance export orchestration, Hovmöller cache/significance export
orchestration and its audit text, and leadtime significance audit text.
Raw loaders, BJT aggregation, TP differencing/calendar-day apportionment,
CF/PF merging, group parsers/membership, composite/latitude-average kernels,
period averaging, and Welch/BH kernels remain identical. No scientific
calculation method changed. June 8 calls the same established kernels with
its own initialization and Total members; newly stored metadata/QC identify it.

## Unexecuted validation

`/data1/huangy/fig6/NC/v12/paper_figures/` is absent in this environment.
Scientific export, real NetCDF validation and PNG/PDF redraw were not run.
The environment also lacks xarray, cartopy and statsmodels, so py_compile
does not establish successful runtime dependency import. Real-data colorbars,
contour labels, map layout, mask counts and output files must be checked in
the scientific Jupyter environment using the provided explicit workflow.

Git HTTPS fetch/push is unavailable because the configured proxy endpoint
cannot be reached. GitHub connector reads supplied the latest branch and exact
base objects, verified by Git hashes; publication uses GitHub Git-data APIs
and a normal fast-forward ref update. No reset, force push, merge, or deletion
is used. The local base and GitHub commit provenance remain reviewable.

## PR #13 legacy initialization compatibility follow-up

Both daily and Hovmöller final renderers now resolve the first valid date in
`initialization`, `init_date`, `main_init_date` order for cache and TP tests.
A valid incorrect primary date fails; legacy June 8 `init_date` takes priority
over stale June 12 `main_init_date`. Missing metadata cannot identify June 8.
No original science processing or NetCDF values were changed. Daily audit
removed nonexistent run flags; Hovmöller audit removed the obsolete Z500
significance-contour description. Final rendering logs report TP only.

Nine targeted tests execute the actual production validators with synthetic
NumPy dataset fixtures: old/new/invalid/missing metadata, incorrect dates,
longitude/latitude coordinates, differences, grouping and TP units. All pass,
as do the two existing static/Matplotlib geometry checks, py_compile for the
five changed Python files, and git diff --check. These tests do not open real
NetCDF or execute full maps. The server-reported June 8 daily files are accepted
as existing; they are not mounted here, so no real redraw was executed in this
environment. Hovmöller availability is independent and must be checked on the
server as documented, without rerunning already generated daily products.
