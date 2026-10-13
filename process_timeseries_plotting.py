"""New process figure layouts only; old Fig.5/6 renderers are never imported."""
from pathlib import Path
import json
import struct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from process_timeseries_core import INITS, GROUPS, VERSIONS, FACTORS5, FACTORS11, OUTPUT_TABLES, sha256

COLORS = {'ERA5': 'black', 'Hot17': 'firebrick', 'Middle17': '#1f77b4', 'Cold17': 'seagreen'}
LABELS = {'NCHN_tp': 'TP (mm day⁻¹)', 'z500_anom_NCHN': 'Z500 anomaly (gpm)',
          'sm_avg': 'SM (m³ m⁻³)', 'SHF_Avg': 'SHF (W m⁻²)',
          'NCVI_concurrent': 'NCVI anomaly (PVU)', 'TCC_Avg': 'TCC (fraction)',
          'partial_fitted_Tmax': 'SM–SHF diagnostic (°C)'}
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
         'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
         'xtick.labelsize': 8, 'ytick.labelsize': 8,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None}


def load_tables(output_root):
    root = Path(output_root)
    manifest = json.loads((root / 'qc/process_manifest.json').read_text())
    if manifest.get('workflow') != 'process_figures_v2' or manifest.get('qc_status') != 'PASS':
        raise ValueError('Process table QC is not PASS')
    for folder in ('tables', 'figures', 'qc'):
        child = root / folder
        if child.is_symlink() or any(p.is_symlink() for p in child.iterdir()):
            raise ValueError(f'Unsafe symlink in output folder: {child}')
    result = {}
    for name in OUTPUT_TABLES:
        path = root / 'tables' / name
        if sha256(path) != manifest['table_sha256'][name]:
            raise ValueError(f'Process numerical table checksum changed: {path}')
        result[name] = pd.read_csv(path)
    return result


def limits(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        raise ValueError('No finite values for panel range')
    lo, hi = float(a.min()), float(a.max())
    pad = max((hi-lo)*0.10, abs(lo)*0.01, 0.001)
    locator = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(lo-pad, hi+pad)
    return (float(ticks[0]), float(ticks[-1])), ticks


def legend(fig, reference=True):
    names = ('ERA5', *GROUPS) if reference else GROUPS
    handles = [Line2D([0], [0], color=COLORS[g], marker='o', lw=1.4, label=g) for g in names]
    fig.legend(handles=handles, loc='lower center', ncol=len(names), frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=8)


def style(ax):
    ax.grid(True, linestyle=':', alpha=0.2, linewidth=0.4)
    ax.tick_params(width=0.7, length=3)
    for spine in ax.spines.values():
        spine.set_color('0.5'); spine.set_linewidth(0.6)


def timeseries_figure(df, factors, window, version=None):
    rows = len(factors)
    fig, axes = plt.subplots(rows, 2, figsize=(6.5, 9.0 if rows == 5 else 10.5), dpi=600)
    data = df if version is None else df[df.model_version == version]
    end = '2023-06-24' if window == 'Total' else '2023-06-18'
    # Pool both inits and both parameterizations for identical row scales.
    for row, factor in enumerate(factors):
        ylim, ticks = limits(df[df.factor == factor].value)
        for col, init in enumerate(INITS):
            ax = axes[row, col]
            s = data[(data.init_date == init) & (data.factor == factor)].copy()
            s['date'] = pd.to_datetime(s.date)
            for group in ('ERA5', *GROUPS):
                g = s[s.group == group].sort_values('date')
                if g.value.notna().all():
                    # Calendar-day aggregates are shown at the BJT day midpoint.
                    ax.plot(g.date + pd.Timedelta(hours=12), g.value, color=COLORS[group], marker='o', ms=2.6, lw=1.3)
                elif group == 'ERA5' and g.value.isna().all():
                    ax.text(0.02, 0.03, 'ERA5 unavailable', transform=ax.transAxes, fontsize=7, color='0.4')
                else:
                    raise ValueError('Incomplete process series cannot be silently omitted')
            ax.set_ylim(ylim); ax.set_yticks(ticks)
            ax.set_xlim(pd.Timestamp('2023-06-14'), pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2 if window == 'Total' else 1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if window == 'Stage-I':
                ax.axvline(pd.Timestamp('2023-06-18 00:00'), color='0.4', linestyle='--', lw=0.8)
            ax.set_title(f'({chr(97+row*2+col)}) {pd.Timestamp(init).day} Jun', loc='left')
            ax.set_ylabel(LABELS[factor] if col == 0 else '')
            style(ax)
    legend(fig)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.07, wspace=0.28, hspace=0.60)
    return fig, axes.ravel().tolist()


def scatter_figure(df, version):
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.9), dpi=600)
    for row, factor in enumerate(('sm_avg', 'partial_fitted_Tmax')):
        for col, init in enumerate(INITS):
            ax = axes[row, col]
            all_versions = df[df.init_date == init]
            g = all_versions[all_versions.model_version == version]
            for group in GROUPS:
                s = g[g.group == group]
                ax.scatter(s[factor], s.y_tmax, s=18, color=COLORS[group], alpha=0.7,
                           edgecolor='white', linewidth=0.3)
            # Paired versions share limits; the SM panel has identical input points.
            xlim, xticks = limits(all_versions[factor])
            ylim, yticks = limits(all_versions.y_tmax)
            ax.set_xlim(xlim); ax.set_xticks(xticks)
            ax.set_ylim(ylim); ax.set_yticks(yticks)
            ax.set_xlabel(LABELS[factor]); ax.set_ylabel('Stage-I Tmax (°C)')
            ax.set_title(f'({chr(97+row*2+col)}) {pd.Timestamp(init).day} Jun', loc='left')
            style(ax)
    legend(fig, reference=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.14, wspace=0.40, hspace=0.50)
    return fig, axes.ravel().tolist()


def figure_qc(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sizes = np.array([[a.get_position().width, a.get_position().height] for a in axes])
    if not np.allclose(sizes, sizes[0], rtol=0, atol=1e-10):
        raise ValueError('Unequal process-panel axes boxes')
    texts = list(fig.texts)
    for ax in axes:
        texts += [ax._left_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts if t.get_visible() and t.get_text()]
    for label, b in boxes:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height:
            raise ValueError(f'Clipped text in process figure: {label}')
    for i, (label, a) in enumerate(boxes):
        for other, b in boxes[i+1:]:
            if min(a.x1,b.x1)-max(a.x0,b.x0)>1 and min(a.y1,b.y1)-max(a.y0,b.y0)>1:
                raise ValueError(f'Overlapping process labels: {label}; {other}')
    return {'equal_boxes': True, 'text_extent_qc': 'PASS', 'width_inches': fig.get_figwidth(),
            'height_inches': fig.get_figheight(), 'axes_count': len(axes)}


def render_all(output_root):
    root = Path(output_root)
    tables = load_tables(root)
    built = []
    with plt.rc_context(STYLE):
        fig, axes = timeseries_figure(tables[OUTPUT_TABLES[0]], FACTORS5, 'Total')
        built.append(('Fig5_Total_daily_0608_0612_process_v2', fig, axes))
        for version in VERSIONS:
            fig, axes = timeseries_figure(tables[OUTPUT_TABLES[1]], ('partial_fitted_Tmax', *FACTORS11), 'Stage-I', version)
            built.append((f'Fig11_StageI_daily_{version}_0608_0612_process_v2', fig, axes))
            fig, axes = scatter_figure(tables[OUTPUT_TABLES[2]], version)
            built.append((f'FigS11_StageI_scatter_{version}_0608_0612_process_v2', fig, axes))
        # All layouts pass before any new figure is accepted for export.
        qcs = [dict(filename=name, **figure_qc(fig, axes)) for name, fig, axes in built]
        for name, fig, axes in built:
            png, pdf = root/'figures'/f'{name}.png', root/'figures'/f'{name}.pdf'
            if png.exists() or pdf.exists():
                raise ValueError('Figure already exists; choose a new output run to preserve it')
        for (name, fig, axes), qc in zip(built, qcs):
            png, pdf = root/'figures'/f'{name}.png', root/'figures'/f'{name}.pdf'
            fig.savefig(png, dpi=600, bbox_inches=None)
            fig.savefig(pdf, bbox_inches=None)
            width, height = struct.unpack('>II', png.read_bytes()[16:24])
            if width != 3900 or height != round(fig.get_figheight()*600):
                raise ValueError('PNG dimensions incorrect')
            qc.update(png_width_px=width, png_height_px=height, pdf_bytes=pdf.stat().st_size,
                      png_sha256=sha256(png), pdf_sha256=sha256(pdf))
            plt.close(fig)
    (root/'qc/process_figure_qc.json').write_text(json.dumps(qcs, indent=2))
    return pd.DataFrame(qcs)
