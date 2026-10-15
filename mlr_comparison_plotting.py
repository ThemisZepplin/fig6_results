"""CSV-only Fig.3/4/9/10 renderer, with no fitting and no import-time I/O/style changes."""
from pathlib import Path
import hashlib
import json
import re
import struct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from mlr_comparison_core import (INITS, WINDOWS, VERSIONS, METHODS, LABELS,
                                 TABLE_FILES, predictors, output_guard, ATOL)

WIDTH = 6.5
DPI = 600
COLORS = {"LMG": "#6AA84F", "Johnson": "#3C78D8", "LOFO": "#E69138"}
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
         "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
         "pdf.fonttype": 42, "ps.fonttype": 42,
         "axes.linewidth": 0.8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
         "axes.labelsize": 9.5, "axes.titlesize": 9.5}
FIGURE_IDS = {"Total": (3, 4, "Total"), "Stage-I": (9, 10, "June14_17")}
FIGURE_VERSION = "residualized"
FIGURE_REVISION = "lofo_deltaR2_v1"
IMPORTANCE_REVISION = "lofo_deltaR2_v2"  # Style-only; Fig3/9 preserve v1
IMPORTANCE_REFERENCE_ORDER = {
    'Total': ('NCHN_tp', 'z500_anom_NCHN', 'NCVI_concurrent', 'TCC_resid',
              'Initial_MSEstar_max_NCHN', 'SM_resid'),
    'Stage-I': ('SHF_resid', 'sm_avg', 'z500_anom_NCHN',
                'Initial_MSEstar_max_NCHN', 'NCVI_concurrent', 'TCC_resid'),
}


def importance_order(data, window):
    """Derive the 12 June order from CSV values; reject an unexpected reference."""
    g = data['importance']
    g = g[(g.init_date == '2023-06-12') & (g.window == window) &
          (g.version == 'residualized') & (g.method == 'LMG')].copy()
    canonical = predictors(window, 'residualized')
    if (len(g) != 6 or g.position.duplicated().any() or
            g.sort_values('position').predictor.tolist() != canonical or
            g.sort_values('position').position.tolist() != list(range(1, 7)) or
            not np.isfinite(g.importance_pct).all()):
        raise ValueError(f'Invalid 12 June residualized LMG reference for {window}')
    ordered = g.sort_values(['importance_pct', 'position'], ascending=[False, True], kind='stable')
    observed = tuple(ordered.predictor)
    if observed != IMPORTANCE_REFERENCE_ORDER[window]:
        raise ValueError(f'12 June residualized LMG order mismatch for {window}: '
                         f'observed={observed}; expected={IMPORTANCE_REFERENCE_ORDER[window]}. '
                         'Check the genuine CSV; no fallback sorting is permitted.')
    return ordered.position.astype(int).tolist()


def load_results(output_root):
    """Require complete, verified result tables before creating any figure."""
    root = Path(output_root).expanduser().resolve()
    manifest = json.loads((root / 'qc/comparison_manifest.json').read_text())
    if manifest.get('computational_status') != 'PASS':
        raise ValueError('Computation QC failed or importance is undefined. Inspect saved QC; figure export blocked.')
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported comparison schema')
    data = {}
    for key, name in TABLE_FILES.items():
        path = root / 'tables' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['table_sha256'][name]:
            raise ValueError(f'Table changed after computation QC: {path}; rerun calculation/QC')
        data[key] = pd.read_csv(path)
    # Backward-compatible, FAIL-CLOSED use of existing, SHA256-verified PR12
    # numerical exports. The original LOFO delta R² already lives in the
    # importance table's raw_value column. A new derivative audit CSV is
    # welcome but must never be a prerequisite for reusing verified results.
    lofo_name='table_mlr_lofo_original_delta_r2_residualized.csv'
    lofo_path=root/'tables'/lofo_name
    lofo_hash=manifest.get('table_sha256',{}).get(lofo_name)
    primary=data['importance'].query("method=='LOFO' and version=='residualized'").copy()
    model_summary=data['summary'].query("version=='residualized'")[
        ['init_date','window','R2','n']].copy()
    lofo=primary.merge(model_summary,on=['init_date','window'],validate='many_to_one')
    if len(lofo)!=48 or lofo.duplicated(['init_date','window','predictor','position']).any():
        raise ValueError('Incomplete/duplicate residualized LOFO entries')
    if not lofo['n'].eq(51).all() or not np.isfinite(
        lofo[['R2','R2_without_factor','raw_value']].to_numpy(float)).all():
        raise ValueError('Invalid 51-member LOFO model and deletion statistics')
    if not np.allclose(lofo.raw_value,lofo.R2-lofo.R2_without_factor,rtol=0,atol=1e-8):
        raise ValueError('Original LOFO values fail R²(full)−R²(reduced) audit')
    if (lofo.raw_value < -1e-8).any():
        raise ValueError('Negative LOFO deletion loss inconsistent with nested in-sample OLS')
    if lofo_hash is not None:
        if not lofo_path.is_file() or hashlib.sha256(lofo_path.read_bytes()).hexdigest()!=lofo_hash:
            raise ValueError('Audited original LOFO file is missing or its SHA256 changed')
        documented=pd.read_csv(lofo_path)
        compare=primary.merge(documented,on=['init_date','window','predictor','position'],
                              validate='one_to_one',suffixes=('','_audit'))
        if len(compare)!=48 or not np.allclose(compare.raw_value,compare.lofo_delta_r2,
                                                 atol=1e-10,rtol=0):
            raise ValueError('Independent LOFO audit CSV disagrees with original raw_value')
        audit_mode='INDEPENDENT_HASHED_CSV'
    else:
        if lofo_path.is_file():
            raise ValueError('Unmanifested LOFO audit CSV exists; refuse unverified source')
        audit_mode='LEGACY_HASHED_IMPORTANCE_AND_REDUCED_R2'
    # Attach presentation audit metadata in memory only; never change historical
    # manifest, input files, or hashes.
    manifest={**manifest,'lofo_plot_input_audit_mode':audit_mode}
    s, f, i = data['summary'], data['fitted'], data['importance']
    if len(s) != 16 or len(f) != 16 * 51 or len(i) != 16 * 18:
        raise ValueError('Incomplete 16-model result set')
    for init in INITS:
        for w in WINDOWS:
            member_groups = []
            for v in VERSIONS:
                subset = lambda d: d[(d.init_date == init) & (d.window == w) & (d.version == v)]
                ss, ff, ii = subset(s), subset(f).sort_values('member'), subset(i)
                if len(ss) != 1 or ss.iloc[0]['n'] != 51 or ss.iloc[0].design_rank != 7:
                    raise ValueError(f'Invalid summary {init}/{w}/{v}')
                if ss.iloc[0].predictor_list != ';'.join(predictors(w, v)):
                    raise ValueError(f'Predictor mapping mismatch {init}/{w}/{v}')
                if not np.array_equal(ff.member.to_numpy(), np.arange(51)):
                    raise ValueError(f'Invalid member IDs {init}/{w}/{v}')
                if not np.isfinite(ff[['y_tmax', 'y_fitted', 'residual']].to_numpy()).all():
                    raise ValueError('Nonfinite fitted result')
                member_groups.append(ff)
                for method in METHODS:
                    im = ii[ii.method == method].sort_values('position')
                    if im.predictor.tolist() != predictors(w, v) or im.position.tolist() != list(range(1, 7)):
                        raise ValueError('Importance mapping mismatch')
                    if method == 'LOFO':
                        delta = im.raw_value.to_numpy(float)
                        if not np.isfinite(delta).all() or (delta < -1e-8).any():
                            raise ValueError('LOFO full-minus-reduced R2 invalid')
                        if not np.allclose(delta, ss.iloc[0].R2-im.R2_without_factor.to_numpy(float),rtol=0,atol=1e-8):
                            raise ValueError('LOFO delta R2 audit mismatch')
                    elif not np.isfinite(im.importance_pct).all() or abs(im.importance_pct.sum() - 100) > 1e-7:
                        raise ValueError('Invalid LMG/Johnson percent shares')
            r, a = member_groups
            if not np.array_equal(r.y_tmax.to_numpy(), a.y_tmax.to_numpy()) or np.max(np.abs(r.y_fitted.to_numpy() - a.y_fitted.to_numpy())) > ATOL:
                raise ValueError(f'Paired fitting QC failed {init}/{w}')
    return data, manifest


def figure_plan(data):
    """Shared per-init scatter axes; shared per-method bars over all 8 figures."""
    scatter = {}
    for w in WINDOWS:
        for init in INITS:
            g = data['fitted'][(data['fitted'].window == w) & (data['fitted'].init_date == init)]
            vals = np.r_[g.y_tmax, g.y_fitted]
            span = float(np.ptp(vals))
            pad = max(span * 0.04, 0.1)
            step = 0.5 if w == 'Stage-I' and span + 2 * pad <= 5 else 1.0
            limits = (float(np.floor((vals.min() - pad) / step) * step),
                      float(np.ceil((vals.max() + pad) / step) * step))
            xticks = MaxNLocator(nbins=5).tick_values(*limits)
            xticks = xticks[(xticks >= limits[0]) & (xticks <= limits[1])]
            yticks = np.arange(limits[0], limits[1] + step / 2, step)
            scatter[(init, w)] = {'limits': limits, 'xticks': xticks.tolist(), 'yticks': yticks.tolist()}
    bars = {}
    for method in METHODS:
        column = 'raw_value' if method == 'LOFO' else 'importance_pct'
        vals = data['importance'].loc[(data['importance'].method==method)&(data['importance'].version==FIGURE_VERSION),column].to_numpy(float)
        if not np.isfinite(vals).all(): raise ValueError(f'Nonfinite {method} values')
        bars[method] = max(float(vals.max())*1.35, 0.01 if method=='LOFO' else 10.0)
    order = {}
    for w in WINDOWS:
        reference = importance_order(data, w)
        for init in INITS:
            order[(init, w)] = reference.copy()
    return {'scatter': scatter, 'bars': bars, 'order': order}


def _base(ax, bars=False):
    ax.tick_params(axis='both', width=0.8, length=3.2)
    for spine in ax.spines.values():
        spine.set_color('0.25')
    ax.grid(True, axis='x' if bars else 'both', linestyle=':', linewidth=0.45,
            alpha=0.12 if bars else 0.18, zorder=0)
    ax.set_axisbelow(True)


def scatter_figure(data, plan, window, version):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.6), dpi=DPI)
    for p, (ax, init) in enumerate(zip(axes.ravel(), INITS)):
        g = data['fitted']
        g = g[(g.init_date == init) & (g.window == window) & (g.version == version)].sort_values('member')
        s = data['summary']
        s = s[(s.init_date == init) & (s.window == window) & (s.version == version)].iloc[0]
        y, fitted = g.y_tmax.to_numpy(), g.y_fitted.to_numpy()
        ax.scatter(y, fitted, s=22, color='#2f6db3', alpha=0.62, edgecolor='white', linewidth=0.30)
        limits = plan['scatter'][(init, window)]
        slope, intercept = np.polyfit(y, fitted, 1)
        xx = np.asarray(limits['limits'])
        ax.plot(xx, slope * xx + intercept, color='0.38', lw=0.9, alpha=0.78)
        ax.text(0.04, 0.96, f'R² = {s.R2:.2f}\nslope = {slope:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8.0,
                bbox=dict(facecolor='white', edgecolor='0.80', alpha=0.86, boxstyle='round,pad=0.25'))
        ax.set_xlim(limits['limits']); ax.set_ylim(limits['limits'])
        ax.set_xticks(limits['xticks']); ax.set_yticks(limits['yticks'])
        ax.set_xlabel('ECMWF Tmax (°C)')
        ax.set_ylabel('MLR predicted Tmax (°C)')
        ax.set_title(f'({chr(97+p)}) {pd.Timestamp(init).day} Jun', loc='left', pad=8)
        _base(ax)
    # Explicit equal axes boxes; aspect remains automatic, no data-unit aspect lock.
    fig.subplots_adjust(left=0.115, right=0.985, top=0.94, bottom=0.10, wspace=0.34, hspace=0.40)
    return fig, axes.ravel().tolist(), []


def _safe_importance_xticks(axis_max):
    """Use readable numerical ticks strictly inside each narrow importance panel.

    MaxNLocator used directly on an axis may return a last tick ABOVE xlim,
    rendering a '90' or '0.6' label across the neighboring panel. Restrict
    the tick *positions*, not the data or the axis limits; the full bar extent
    and method-shared scale are unchanged.
    """
    upper=float(axis_max)
    if not np.isfinite(upper) or upper<=0:
        raise ValueError('Invalid importance axis limit')
    safe_end=0.76*upper
    candidates=MaxNLocator(nbins=4,min_n_ticks=3).tick_values(0.0,safe_end)
    ticks=np.unique(candidates[(candidates>=0) & (candidates<=safe_end+1e-12)])
    if ticks.size<3:
        # Degenerate float limits are not anticipated; fail rather than silently
        # removing the numerical axis required by the supervisor.
        raise ValueError('Importance axis cannot accommodate three numeric ticks')
    return ticks


def importance_figure(data, plan, window, version):
    order = importance_order(data, window)
    if any(plan['order'][(init, window)] != order for init in INITS):
        raise ValueError(f'Importance plan does not retain the fixed reference positions for {window}')
    fig = plt.figure(figsize=(WIDTH, 8.1), dpi=DPI)
    grid = fig.add_gridspec(3, 4, left=0.145, right=0.985, bottom=0.09,
                          top=0.92, wspace=0.19, hspace=0.60)
    axes, value_texts = [], []
    row_headers = ('LMG relative importance (%)', 'Johnson relative weights (%)',
                   'R² reduction (ΔR²)')
    for row, method in enumerate(METHODS):
        for col, init in enumerate(INITS):
            ax = fig.add_subplot(grid[row, col])
            if col == 0:
                fig.text(0.565, ax.get_position().y1 + 0.052, row_headers[row],
                         ha='center', va='center', fontsize=10.0)
            g = data['importance']
            g = g[(g.init_date == init) & (g.window == window) & (g.version == version) & (g.method == method)].set_index('position')
            g = g.loc[order]
            vals = g.raw_value.to_numpy(float) if method=='LOFO' else g.importance_pct.to_numpy(float)
            if not np.isfinite(vals).all() or (method=='LOFO' and np.min(vals)<-1e-8):raise ValueError('Invalid bar values')
            ax.barh(np.arange(6), vals, color=COLORS[method], alpha=0.88, height=0.66)
            ax.set_yticks(np.arange(6), [LABELS[n] for n in g.predictor], fontsize=8.0)
            # Slightly emphasize factor names in the first column.
            if col == 0:
                for tick in ax.get_yticklabels():
                    tick.set_fontweight('semibold')
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=(col == 0))
            ax.invert_yaxis()
            ax.set_xlim(0, plan['bars'][method])
            ax.set_title(f'({chr(97+row*4+col)}) {pd.Timestamp(init).day} Jun', loc='left', fontsize=9.0, pad=7)
            _base(ax, bars=True)
            for side in ('top', 'right', 'bottom'):
                ax.spines[side].set_visible(False)
            ax.spines['left'].set_color('0.80')
            ax.spines['left'].set_linewidth(0.6)
            # Same method uses the same numeric ticks in all four init panels.
            # Never allow MaxNLocator to draw an out-of-range 90/0.6 tick.
            ax.set_xticks(_safe_importance_xticks(plan['bars'][method]))
            ax.tick_params(axis='x',bottom=True,labelbottom=True,labelsize=8)
            for y, v in enumerate(vals):
                # Preserve positive sub-0.0005 values instead of rounding to 0.000.
                label = (('<0.001' if 0 < v < 0.0005 else f'{v:.3f}')
                         if method == 'LOFO' else f'{v:.1f}%')
                t = ax.text(v, y, label, va='center', ha='left', color='black',
                            fontweight='bold', fontsize=8.0)
                value_texts.append((ax, t, float(v), y))
            axes.append(ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Measure at FINAL 6.5-inch canvas width. Short bars carry values outside;
    # when a label will not fit to the right, place it at the bar CENTER
    # WITHOUT a white rectangle that visually shortens the encoded bar.
    clearance_px = fig.dpi / 72.0 * 2.0
    for ax, t, v, y in value_texts:
        bounds=ax.get_window_extent(renderer)
        text_width=t.get_window_extent(renderer).width
        zero_px=ax.transData.transform((0.0,y))[0]
        end_px=ax.transData.transform((v,y))[0]
        room_right = bounds.x1 - end_px
        fit_outside = room_right >= text_width + 2.0*clearance_px
        fit_inside = end_px-zero_px >= text_width + 2.0*clearance_px
        if fit_outside:
            label_px=end_px + clearance_px
            t.set_x(ax.transData.inverted().transform(
                (label_px,ax.transData.transform((0.0,y))[1]))[0])
            t.set_ha('left')
            t._mlr_placement='outside'
        elif fit_inside:
            t.set_x(v/2.0)
            t.set_ha('center')
            t._mlr_placement='inside'
        else:
            raise ValueError(f'Cannot place importance label {t.get_text()!r} '
                             f'without covering bar endpoint: value={v:.8g}; '
                             'increase panel width or revise tick/layout spacing')
        # NO bbox for any importance value: color must remain visible to the tip.
        if t.get_bbox_patch() is not None:
            raise ValueError('A label background is obscuring a data bar')
    fig.canvas.draw()
    return fig, axes, [t for _, t, _, _ in value_texts]


def layout_qc(fig, axes, values):
    """Rendered text extents, inter-text collisions, annotation containment and boxes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in axes:
        texts += [ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label]
        texts += list(ax.texts) + list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    texts = list({id(t): t for t in texts if t.get_visible() and t.get_text()}.values())
    extents = [(t, t.get_window_extent(renderer)) for t in texts]
    clipped = [t.get_text() for t, bb in extents if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height]
    collisions = []
    for i, (a, aa) in enumerate(extents):
        for b, bb in extents[i + 1:]:
            # Positive-area overlap (avoid reporting a shared boundary).
            if min(aa.x1, bb.x1) - max(aa.x0, bb.x0) > 1 and min(aa.y1, bb.y1) - max(aa.y0, bb.y0) > 1:
                collisions.append([a.get_text(), b.get_text()])
    outside_axes = []
    for t in values:
        bb, ab = t.get_window_extent(renderer), t.axes.get_window_extent(renderer)
        if bb.x0 < ab.x0 or bb.x1 > ab.x1 or bb.y0 < ab.y0 or bb.y1 > ab.y1:
            outside_axes.append(t.get_text())
    sizes = np.array([[ax.get_position().width * WIDTH, ax.get_position().height * fig.get_figheight()] for ax in axes])
    return {'clipped_text': clipped, 'text_collisions': collisions,
            'value_labels_outside_axes': outside_axes,
            'equal_axes_boxes': bool(np.allclose(sizes, sizes[0], atol=1e-10)),
            'axes_box_inches': sizes.tolist(), 'width_inches': fig.get_figwidth(),
            'height_inches': fig.get_figheight(), 'status': 'PASS' if not clipped and not collisions and not outside_axes and np.allclose(sizes, sizes[0], atol=1e-10) else 'FAIL',
            'manual_review': 'Inspect genuine HPC figures for text/data occlusion; extent checks do not establish scientific visual adequacy.'}


def importance_label_placement_qc(fig, axes):
    """Audit every value label against the ACTUAL bar and rendered text extents.

    Non-interference with bar endpoints is a separate scientific legibility gate,
    because generic text-to-text collision checks do not detect bar occlusion.
    """
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    slack_px=fig.dpi/72.0*0.35
    records=[]; failures=[]
    for panel,ax in enumerate(axes):
        patches=list(ax.patches)
        texts=list(ax.texts)
        if len(patches)!=6 or len(texts)!=6:
            failures.append(f'panel {panel}: expected 6 bars and 6 labels, '
                            f'found {len(patches)} and {len(texts)}')
            continue
        frame=ax.get_window_extent(renderer)
        for row,(bar,label) in enumerate(zip(patches,texts)):
            bounds=label.get_window_extent(renderer)
            bar_bounds=bar.get_window_extent(renderer)
            placement=getattr(label,'_mlr_placement',None)
            if placement=='inside':
                valid=(bounds.x0>=bar_bounds.x0-slack_px and
                       bounds.x1<=bar_bounds.x1+slack_px and
                       bounds.y0>=bar_bounds.y0-slack_px and
                       bounds.y1<=bar_bounds.y1+slack_px and
                       label.get_ha()=='center')
            elif placement=='outside':
                valid=(bounds.x0>=bar_bounds.x1-slack_px and
                       bounds.x1<=frame.x1+slack_px and
                       label.get_ha()=='left')
            else:
                valid=False
            valid=bool(valid and label.get_bbox_patch() is None and
                       label.get_color()=='black' and
                       label.get_fontweight()=='bold')
            records.append({'panel':panel,'row':row,'text':label.get_text(),
                            'placement':placement,'valid':valid})
            if not valid:
                failures.append(f'panel {panel}, row {row}: {label.get_text()!r}, '
                                f'placement={placement}; label/bar geometry invalid')
    return {'label_placement_status':'PASS' if not failures and len(records)==72 else 'FAIL',
            'label_placement_count':len(records),
            'label_placement_inside':sum(r['placement']=='inside' for r in records),
            'label_placement_outside':sum(r['placement']=='outside' for r in records),
            'label_placement_failures':failures,
            'label_placement_records':records}


def importance_alignment_qc(fig, axes):
    """Measure grid edges, repeated gaps and title anchors at final size (1.5 pt)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = np.array([a.get_window_extent(renderer).extents for a in axes]) * 72 / fig.dpi
    titles = np.array([a._left_title.get_window_extent(renderer).extents for a in axes]) * 72 / fig.dpi
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    deviations = [np.ptp(widths), np.ptp(heights)]
    for row in range(3):
        b, t = boxes[row*4:row*4+4], titles[row*4:row*4+4]
        deviations += [np.ptp(b[:, 1]), np.ptp(b[:, 3]),
                       np.ptp(b[1:, 0] - b[:-1, 2]), np.ptp(t[:, 1]),
                       np.ptp(t[:, 0] - b[:, 0])]
    for col in range(4):
        b = boxes[col::4]
        deviations += [np.ptp(b[:, 0]), np.ptp(b[:, 2]),
                       np.ptp(b[:-1, 1] - b[1:, 3])]
    maximum = float(max(deviations))
    return {'alignment_tolerance_pt': 1.5, 'alignment_max_deviation_pt': maximum,
            'measured_axes_width_inches': (widths / 72).tolist(),
            'alignment_status': 'PASS' if maximum <= 1.5 else 'FAIL'}


def pdf_importance_qc(path):
    """Audit exported vector text/font size and collisions, not source rcParams."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('PDF layout QA requires PyMuPDF: pip install pymupdf') from exc
    with fitz.open(path) as doc:
        page = doc[0]
        lines, sizes = [], []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                text = ''.join(s['text'] for s in line['spans'])
                if text.strip():
                    lines.append((text, fitz.Rect(line['bbox'])))
                    sizes.extend(s['size'] for s in line['spans'] if s['text'].strip())
        collisions = []
        clipped = []
        for index, (text, rect) in enumerate(lines):
            if not page.rect.contains(rect):
                clipped.append(text)
            for other, box in lines[index+1:]:
                overlap = rect & box
                if overlap.width > 0.3 and overlap.height > 0.3:
                    collisions.append([text, other])
        fonts = page.get_fonts(full=True)
        embedded = all(bool(doc.extract_font(f[0])[3]) for f in fonts)
        vector = bool(lines) and not page.get_images() and all(f[2] != 'Type3' for f in fonts)
        percentages = re.findall(r'\d+\.\d%', '\n'.join(t for t, _ in lines))
        # Expect 48 LMG/Johnson percent value labels and 24 LOFO decimal labels.
        numeric_decimal = re.findall(r'(?<![\d.])\d+\.\d{3}(?!\d)', '\n'.join(t for t, _ in lines))
        subthreshold_labels = re.findall(r'<0\.001', '\n'.join(t for t, _ in lines))
        minimum = float(min(sizes)) if sizes else 0.0
        return {'pdf_min_font_pt': minimum, 'pdf_embedded_fonts': embedded,
                'pdf_editable_vector_text': vector, 'pdf_percentage_labels': len(percentages),
                'pdf_lofo_decimal_labels':len(numeric_decimal),
                'pdf_lofo_subthreshold_labels':len(subthreshold_labels),
                'pdf_text_collisions': collisions, 'pdf_clipped_text': clipped,
                'pdf_status': 'PASS' if minimum >= 7.5 and embedded and vector and
                len(percentages) >= 48 and len(numeric_decimal)+len(subthreshold_labels) >= 24 and not clipped and not collisions else 'FAIL'}


def make_figures(data, plan, kind):
    built = []
    with plt.rc_context(STYLE):
        for window in WINDOWS:
            scatter_id, importance_id, stem = FIGURE_IDS[window]
            for version in (FIGURE_VERSION,):
                number = scatter_id if kind == 'scatter' else importance_id
                factory = scatter_figure if kind == 'scatter' else importance_figure
                fig, axes, values = factory(data, plan, window, version)
                qc = layout_qc(fig, axes, values)
                if kind == 'importance':
                    qc.update(importance_alignment_qc(fig, axes))
                    qc.update(importance_label_placement_qc(fig, axes))
                    if (qc['alignment_status'] != 'PASS' or
                            qc['label_placement_status'] != 'PASS'):
                        qc['status'] = 'FAIL'
                qc.update({'figure': f'Fig{number}', 'window': window, 'version': version,
                           'basename': f'Fig{number}_{stem}_{version}_' +
                                       (IMPORTANCE_REVISION if kind=='importance' else FIGURE_REVISION),
                           'axes': [{'xlim': list(a.get_xlim()), 'ylim': list(a.get_ylim()),
                                     'xticks': a.get_xticks().tolist(), 'yticks': a.get_yticks().tolist()} for a in axes]})
                built.append({'fig': fig, 'qc': qc})
    return built


def save_figures(built, input_root, output_root):
    out = output_guard(input_root, output_root)
    numbers = {b['qc']['figure'] for b in built}
    if len(built)!=2 or numbers!={'Fig4','Fig10'}:
        raise ValueError('Style revision exports only the two new importance figures; Fig3/9 v1 untouched')
    if any(x['qc']['version']!=FIGURE_VERSION for x in built):
        raise ValueError('Raw versions are excluded from main figures')
    qcs = [b['qc'] for b in built]
    (out / 'qc').mkdir(parents=True, exist_ok=True)
    qc_path = out / ('qc/figure_'+IMPORTANCE_REVISION+'_qc.json')
    qc_path.write_text(json.dumps(qcs, indent=2), encoding='utf-8')
    if any(q['status'] != 'PASS' for q in qcs):
        raise ValueError('Rendered figure layout QC failed; inspect revision-specific QC file. No figure files exported.')
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    manifest = []
    for b in built:
        fig, qc = b['fig'], b['qc']
        stem = out / 'figures' / qc['basename']
        if stem.with_suffix('.png').exists() or stem.with_suffix('.pdf').exists():
            raise FileExistsError(f'No figure overwrite allowed: {stem}')
        fig.savefig(stem.with_suffix('.png'), dpi=DPI, bbox_inches=None)
        with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
            fig.savefig(stem.with_suffix('.pdf'), bbox_inches=None)
        png, pdf = stem.with_suffix('.png'), stem.with_suffix('.pdf')
        width, height = struct.unpack('>II', png.read_bytes()[16:24])
        box = re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)', pdf.read_bytes())
        pdf_width = float(box.group(1)) / 72 if box else np.nan
        pdf_height = float(box.group(2)) / 72 if box else np.nan
        if (width != 3900 or height != round(fig.get_figheight() * DPI) or
                not np.isclose(pdf_width, WIDTH, atol=1e-8) or
                not np.isclose(pdf_height, fig.get_figheight(), atol=1e-8)):
            raise ValueError(f'Export dimensions/file check failed {stem}')
        if qc['figure'] in ('Fig4', 'Fig10'):
            qc.update(pdf_importance_qc(pdf))
            qc_path.write_text(json.dumps(qcs, indent=2), encoding='utf-8')
            if qc['pdf_status'] != 'PASS':
                raise ValueError(f'Exported PDF QA failed {pdf}; inspect {qc_path}. No acceptance manifest written.')
        manifest.append({**{k: qc[k] for k in ['figure', 'window', 'version', 'basename']},
                         'png_width_px': width, 'png_height_px': height,
                         'width_inches': WIDTH, 'height_inches': fig.get_figheight(),
                         'dpi': DPI, 'pdf_bytes': pdf.stat().st_size,
                         'pdf_width_inches': pdf_width, 'pdf_height_inches': pdf_height,
                         'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
                         'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'status': 'PASS'})
        plt.close(fig)
    pd.DataFrame(manifest).to_csv(out / ('qc/table_figure_'+IMPORTANCE_REVISION+'_manifest.csv'), index=False)
    return pd.DataFrame(manifest)
