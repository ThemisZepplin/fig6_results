"""Synthetic layout/export tests; artifacts are temporary and not scientific results."""
import json
import struct

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import mlr_comparison_core as core
import mlr_comparison_plotting as plotting
from test_mlr_comparison import synthetic_inputs


@pytest.fixture(scope='module')
def render_input(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_layout_only')
    inp, out = root / 'input', root / 'comparison'
    raw, nc = synthetic_inputs()
    groups, ra = core.prepare_pairs(raw, nc)
    models = core.fit_pairs(groups)
    perm = core.lmg_permutations()
    imp, qc = core.importance(models, perm)
    result = core.assemble_results(groups, models, imp, qc, ra)
    h = core.compare_history(result, {})
    core.save_results(result, h, inp, out, [{'synthetic_layout_only': True}], perm)
    data, _ = plotting.load_results(out)
    return inp, out, data, plotting.figure_plan(data)


def test_rendered_layout_axes_and_exports(render_input):
    inp, out, data, plan = render_input
    built = plotting.make_figures(data, plan, 'scatter') + plotting.make_figures(data, plan, 'importance')
    for item in built:
        assert item['qc']['status'] == 'PASS', item['qc']
    manifest = plotting.save_figures(built, inp, out)
    assert len(manifest) == 8
    assert (manifest.png_width_px == 3900).all()
    assert len(list((out / 'figures').glob('*.png'))) == 8
    assert len(list((out / 'figures').glob('*.pdf'))) == 8
    qcs = json.loads((out / 'qc/figure_layout_qc.json').read_text())
    for num in (3, 4, 9, 10):
        p = [q for q in qcs if q['figure'] == f'Fig{num}']
        assert p[0]['axes'] == p[1]['axes']
    # PDF MediaBox may be compressed in other backends; matplotlib emits it plain.
    import re
    for pdf in (out / 'figures').glob('*.pdf'):
        m = re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)', pdf.read_bytes())
        assert m and abs(float(m.group(1)) / 72 - 6.5) < 1e-8


def test_short_and_zero_bars_keep_true_lengths(render_input):
    _, _, data, plan = render_input
    data = dict(data)
    data['importance'] = data['importance'].copy()
    mask = ((data['importance'].init_date == core.INITS[0]) &
            (data['importance'].window == 'Stage-I') &
            (data['importance'].version == 'raw') &
            (data['importance'].method == 'LOFO'))
    data['importance'].loc[mask, 'importance_pct'] = [0, 0.01, 0.05, 1, 48.94, 50]
    with plotting.plt.rc_context(plotting.STYLE):
        fig, axes, labels = plotting.importance_figure(data, plan, 'Stage-I', 'raw')
    ax = axes[8]
    widths = sorted(p.get_width() for p in ax.patches)
    np.testing.assert_allclose(widths, [0, 0.01, 0.05, 1, 48.94, 50])
    assert {'0.0%', '0.1%'} <= {t.get_text() for t in ax.texts}
    plotting.plt.close(fig)
