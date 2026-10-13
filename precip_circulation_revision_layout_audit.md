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

## Shared Fig.6/Fig.S6 compact-layout follow-up

The shared renderer restores the requested baseline: 6.5 × 7.1 inches;
subplots left=0.105, right=0.99, top=0.965, bottom=0.145,
wspace=0.035, hspace=0.075. Both colorbars move from y=0.055 to y=0.085,
with unchanged x positions, widths (0.28/0.30), thickness (0.011), ticks,
units, colors and fonts. Maps retain their existing fixed geographic aspect.

After canvas draw, row labels use actual first-column axis centers and tight
bounds including latitude labels. A 3-point gap separates row names from the
latitude labels and canvas edge. If actual font/tick bounds require it, only
the shared left margin may increase by at most 0.015; larger adjustments fail
explicitly. The baseline suffices in the synthetic geometry check. Both
initializations call the same layout helper; no separate layout parameters.
No suptitle is added, and dates, observations, group rows and a–l remain.

The Jupyter instructions provide full-script paste and absolute-path runpy
methods, four exact input-path checks, and explicit separate Fig.6/Fig.S6
calls. Each call generates just its corresponding initialization.

Validation: both changed Python files pass py_compile; nine identity tests and
four layout/static checks pass; git diff --check passes. The added daily check
executes the actual layout helper on fixed-aspect Matplotlib axes with latitude
labels and checks centering, clipping, collision clearance, map ratio and
colorbar geometry. It is not a Cartopy/NetCDF integration run or a real figure.
No science code, cache, grouping CSV or NetCDF values are changed.

The named attachment and comparison PNGs were not exposed as readable files in
this session; layout was based on the explicit old parameters in the request.
Server-confirmed daily products remain valid existing inputs, but their real
cache files are not mounted here. Neither real Fig.6 nor real Fig.S6 PNG/PDF
was redrawn or visually verified in this environment. Server checks of actual
latitude-label clearance, row/colorbar spacing and final PNG/PDF remain pending.

## TP hatch and contour-label visual follow-up

Reference principles: nature-figure README
https://github.com/Yuan1z0825/nature-skills/blob/main/skills/nature-figure/README.md
(final-size hierarchy, full text extents, alignment and collision review).
Submission sizes and layout remain JGR/AGU; no Nature-specific resizing or
new figure contract is applied.

Only three final renderers, the existing layout test and these usage/audit
documents change. The daily compact layout, seven-panel Hovmöller layout,
leadtime/observation structure, fonts, levels, ticks, colorbars and output
filenames are retained. Science scripts, all validation methods, masks,
NetCDF/QC values and grouping inputs are unchanged.

TP hatch linewidth changes from 0.45 to 0.60. Black `...`, transparent fills,
local significant_local_p005, mask extent/density and overlay zorders stay the
same. Matplotlib 3.10.8 Collection stores `_hatch_linewidth` at creation; the
PDF hatch pattern uses that artist/graphics-context value. Creation and final
draw/savefig both run inside local rc_context, covering older draw-time readers
as well. No global hatch rc setting is changed. Tests decoded synthetic PDF
streams and confirmed `0.6 w`; artist width and rc restoration also passed.
Older Matplotlib versions were not installed/tested separately.

Contour geometry/levels/color/style/width, selected label levels, sign format,
font size and the 5880 feature line remain unchanged. Clabel is deferred until
each figure's final layout. Candidates are sampled along actual contour paths,
then clabel snaps anchors to those paths; accepted labels are checked for the
requested level and full rotated text extents plus the existing 3.2-pt stroke.
Placement avoids panel-edge clipping, existing text/panel labels and reference
line paths including NCHN and 114/119E. Actual contour/text intersection is
intentional and is not reported as a collision. No free text translation is
used. Non-inline placement preserves continuous paths, so discarded trials
never leave unnumbered breaks. At most one label per selected level/panel is
kept; a no-space contour is retained and the omitted label is explicitly logged.
After final draw, accepted labels are checked again for clipping and collisions.

Validation: four changed Python files pass py_compile; nine identity tests and
five static/layout/visual tests pass; Jupyter snippets compile and git diff
--check passes. Synthetic PNG/PDF tests run identical helpers from each script,
confirm effective PDF hatch width, all signed test levels, on-contour anchors,
unchanged path vertices and rc restoration. They verify rendering mechanisms,
not real scientific figures or dark-fill readability.

No true cache inputs or final scientific PDFs are mounted in this executor.
Not completed: real Fig.6, Fig.S6, Fig.7 and Fig.9 redraws; final-size dark-fill
contrast and geographic contour/text collision review; nature-figure PDF font
and collision audit of the real deliverables; per-case review of intentional
map text/contour overlays. Synthetic PDFs were used only for hatch mechanism
checks and cannot replace these real-figure acceptance items.
