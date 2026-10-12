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
FIGURE_IDS = {"Total": (3, 4, "Total"), "Stage-I": (9, 10, "StageI")}


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
                    if not np.isfinite(im.importance_pct).all() or abs(im.importance_pct.sum() - 100) > 1e-7:
                        raise ValueError('Undefined/unnormalized importance; do not render')
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
    bars = {m: max(float(data['importance'].loc[data['importance'].method == m, 'importance_pct'].max()) * 1.55, 10.0)
            for m in METHODS}
    order = {}
    for init in INITS:
        for w in WINDOWS:
            g = data['importance']
            g = g[(g.init_date == init) & (g.window == w) & (g.version == 'residualized') & (g.method == 'LMG')]
            order[(init, w)] = g.sort_values(['importance_pct', 'position'], ascending=[False, True], kind='stable').position.tolist()
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
        ax.text(0.04, 0.96, f'r = {np.corrcoef(y, fitted)[0, 1]:.2f}\nR² = {s.R2:.2f}\nslope = {slope:.2f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=8.0,
                bbox=dict(facecolor='white', edgecolor='0.80', alpha=0.86, boxstyle='round,pad=0.25'))
        ax.set_xlim(limits['limits']); ax.set_ylim(limits['limits'])
        ax.set_xticks(limits['xticks']); ax.set_yticks(limits['yticks'])
        ax.set_xlabel('ECMWF member Tmax (°C)')
        ax.set_ylabel('MLR fitted Tmax (°C)')
        ax.set_title(f'({chr(97+p)}) {pd.Timestamp(init).day} Jun', loc='left', pad=8)
        _base(ax)
    # Explicit equal axes boxes; aspect remains automatic, no data-unit aspect lock.
    fig.subplots_adjust(left=0.115, right=0.985, top=0.94, bottom=0.10, wspace=0.34, hspace=0.40)
    return fig, axes.ravel().tolist(), []


def importance_figure(data, plan, window, version):
    fig = plt.figure(figsize=(WIDTH, 8.4), dpi=DPI)
    axes, value_texts = [], []
    row_headers = ('LMG relative importance (%)', 'Johnson relative weights (%)',
                   'Normalized positive ΔR² (%)')
    row_bottoms = (0.69, 0.395, 0.10)
    for row, method in enumerate(METHODS):
        y0 = row_bottoms[row]
        fig.text(0.54, y0 + 0.258, row_headers[row], ha='center', va='center', fontsize=9.0)
        for col, init in enumerate(INITS):
            ax = fig.add_axes([0.115 + col * 0.245, y0, 0.13, 0.225])
            g = data['importance']
            g = g[(g.init_date == init) & (g.window == window) & (g.version == version) & (g.method == method)].set_index('position')
            g = g.loc[plan['order'][(init, window)]]
            vals = g.importance_pct.to_numpy()
            ax.barh(np.arange(6), vals, color=COLORS[method], alpha=0.88, height=0.66)
            ax.set_yticks(np.arange(6), [LABELS[n] for n in g.predictor], fontsize=7.0)
            ax.invert_yaxis()
            ax.set_xlim(0, plan['bars'][method])
            ax.set_title(f'({chr(97+row*4+col)}) {pd.Timestamp(init).day} Jun', loc='left', fontsize=8.5, pad=5)
            _base(ax, bars=True)
            # Bar labels carry exact percentages; retain the shared internal scale.
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
            ax.tick_params(axis='x', which='both', bottom=False, top=False,
                           labelbottom=False, labeltop=False)
            ax.grid(False, axis='x', which='both')
            for y, v in enumerate(vals):
                t = ax.text(v, y, f'{v:.1f}%', va='center', ha='left', color='black',
                            fontweight='bold', fontsize=7.0)
                value_texts.append((ax, t, float(v), y))
            axes.append(ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Measure actual rendered widths; place inside only if the bar can hold text.
    for ax, t, v, y in value_texts:
        text_width = t.get_window_extent(renderer).width
        zero_px = ax.transData.transform((0, y))[0]
        end_px = ax.transData.transform((v, y))[0]
        pad_px = fig.dpi / 72 * 1.8
        inside = v > 0 and end_px - zero_px >= text_width + 2 * pad_px
        label_px = end_px - pad_px if inside else end_px + pad_px
        xpos = ax.transData.inverted().transform((label_px, ax.transData.transform((0, y))[1]))[0]
        t.set_x(xpos); t.set_ha('right' if inside else 'left')
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


def make_figures(data, plan, kind):
    built = []
    with plt.rc_context(STYLE):
        for window in WINDOWS:
            scatter_id, importance_id, stem = FIGURE_IDS[window]
            for version in VERSIONS:
                number = scatter_id if kind == 'scatter' else importance_id
                factory = scatter_figure if kind == 'scatter' else importance_figure
                fig, axes, values = factory(data, plan, window, version)
                qc = layout_qc(fig, axes, values)
                qc.update({'figure': f'Fig{number}', 'window': window, 'version': version,
                           'basename': f'Fig{number}_{stem}_{version}',
                           'axes': [{'xlim': list(a.get_xlim()), 'ylim': list(a.get_ylim()),
                                     'xticks': a.get_xticks().tolist(), 'yticks': a.get_yticks().tolist()} for a in axes]})
                built.append({'fig': fig, 'qc': qc})
    return built


def save_figures(built, input_root, output_root):
    out = output_guard(input_root, output_root)
    if len(built) != 8:
        raise ValueError('Expected all eight figures before export')
    # Verify the final raw/resid axes and actual ticks rather than merely intentions.
    for number in (3, 4, 9, 10):
        pair = [b for b in built if b['qc']['figure'] == f'Fig{number}']
        if len(pair) != 2 or pair[0]['qc']['axes'] != pair[1]['qc']['axes']:
            raise ValueError(f'Raw/residualized axes mismatch Fig{number}')
    qcs = [b['qc'] for b in built]
    (out / 'qc').mkdir(parents=True, exist_ok=True)
    (out / 'qc/figure_layout_qc.json').write_text(json.dumps(qcs, indent=2), encoding='utf-8')
    if any(q['status'] != 'PASS' for q in qcs):
        raise ValueError('Rendered figure layout QC failed; inspect figure_layout_qc.json. No figure files exported.')
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    manifest = []
    for b in built:
        fig, qc = b['fig'], b['qc']
        stem = out / 'figures' / qc['basename']
        # Fixed canvas: never bbox_inches="tight". Explicit None avoids inherited rc.
        fig.savefig(stem.with_suffix('.png'), dpi=DPI, bbox_inches=None)
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
        manifest.append({**{k: qc[k] for k in ['figure', 'window', 'version', 'basename']},
                         'png_width_px': width, 'png_height_px': height,
                         'width_inches': WIDTH, 'height_inches': fig.get_figheight(),
                         'dpi': DPI, 'pdf_bytes': pdf.stat().st_size,
                         'pdf_width_inches': pdf_width, 'pdf_height_inches': pdf_height,
                         'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
                         'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'status': 'PASS'})
        plt.close(fig)
    pd.DataFrame(manifest).to_csv(out / 'qc/table_figure_manifest.csv', index=False)
    return pd.DataFrame(manifest)
