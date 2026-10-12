"""Synthetic layout/export tests; artifacts are temporary and not scientific results."""
import json
import os
from pathlib import Path

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
    # Explicit synthetic layout fixture, not a reproduction of scientific rankings.
    # Set the reference percentages to exercise the user-confirmed display contract.
    for window, order in plotting.IMPORTANCE_REFERENCE_ORDER.items():
        for predictor, percent in zip(order, [35, 25, 18, 12, 7, 3]):
            mask = ((result['importance'].init_date == '2023-06-12') &
                    (result['importance'].window == window) &
                    (result['importance'].version == 'residualized') &
                    (result['importance'].method == 'LMG') &
                    (result['importance'].predictor == predictor))
            result['importance'].loc[mask, 'importance_pct'] = percent
    h = core.compare_history(result, {})
    core.save_results(result, h, inp, out, [{'synthetic_layout_only': True}], perm)
    data, _ = plotting.load_results(out)
    return inp, out, data, plotting.figure_plan(data)


def test_rendered_layout_axes_and_exports(render_input):
    inp, out, data, plan = render_input
    built = plotting.make_figures(data, plan, 'scatter') + plotting.make_figures(data, plan, 'importance')
    for item in built:
        assert item['qc']['status'] == 'PASS', item['qc']
        if item['qc']['figure'] in ('Fig4', 'Fig10'):
            axes = item['fig'].axes
            assert len(axes) == 12
            for index, ax in enumerate(axes):
                method = core.METHODS[index // 4]
                assert ax.get_xlim() == (0, plan['bars'][method])
                assert len(ax.get_xticks()) == len(ax.get_xticks(minor=True)) == 0
                assert not ax.get_xticklabels() and not ax.get_xticklabels(minor=True)
                assert not any(t.get_visible() for t in ax.xaxis.get_ticklines())
                assert not any(g.get_visible() for g in ax.get_xgridlines())
                assert len(ax.get_yticklabels()) == (6 if index % 4 == 0 else 0)
                window, version = item['qc']['window'], item['qc']['version']
                expected_order = plotting.importance_order(data, window)
                init = core.INITS[index % 4]
                source = data['importance']
                source = source[(source.window == window) & (source.version == version) &
                                (source.init_date == init) & (source.method == method)].set_index('position').loc[expected_order]
                np.testing.assert_array_equal([p.get_width() for p in ax.patches], source.importance_pct)
                np.testing.assert_array_equal([p.get_y() + p.get_height()/2 for p in ax.patches], np.arange(6))
                if index % 4 == 0:
                    assert [t.get_text() for t in ax.get_yticklabels()] == [core.LABELS[p] for p in source.predictor]
                assert all(not ax.spines[s].get_visible() for s in ('top', 'right', 'bottom'))
                assert len(ax.texts) == 6
                assert all(t.get_color() == 'black' and t.get_fontweight() == 'bold'
                           for t in ax.texts)
            assert item['qc']['value_labels_outside_axes'] == []
            assert item['qc']['alignment_status'] == 'PASS'
            assert all(0.18 <= a.get_position().width <= 0.19 for a in axes)
    manifest = plotting.save_figures(built, inp, out)
    assert len(manifest) == 8
    assert (manifest.png_width_px == 3900).all()
    assert len(list((out / 'figures').glob('*.png'))) == 8
    assert len(list((out / 'figures').glob('*.pdf'))) == 8
    qcs = json.loads((out / 'qc/figure_layout_v2_qc.json').read_text())
    for num in (3, 4, 9, 10):
        p = [q for q in qcs if q['figure'] == f'Fig{num}']
        assert p[0]['axes'] == p[1]['axes']
        if num in (4, 10):
            assert all(q['pdf_status'] == 'PASS' and q['pdf_min_font_pt'] >= 7.5 and
                       q['pdf_editable_vector_text'] and q['pdf_percentage_labels'] == 72 for q in p)
            assert all(q['basename'].endswith('_layout_v2') for q in p)
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
    assert plotting.layout_qc(fig, axes, labels)['value_labels_outside_axes'] == []
    plotting.plt.close(fig)


def test_wrong_reference_order_is_rejected(render_input):
    _, _, data, _ = render_input
    data = dict(data)
    data['importance'] = data['importance'].copy()
    mask = ((data['importance'].init_date == '2023-06-12') &
            (data['importance'].window == 'Total') &
            (data['importance'].version == 'residualized') &
            (data['importance'].method == 'LMG'))
    data['importance'].loc[mask, 'importance_pct'] = [1, 2, 3, 4, 5, 85]
    with pytest.raises(ValueError, match='12 June residualized LMG order mismatch'):
        plotting.figure_plan(data)


def test_v2_export_preserves_old_figures_and_numerical_tables(render_input):
    inp, out, data, plan = render_input
    import hashlib
    old = out / 'figures/Fig4_Total_raw.png'
    old.write_bytes(b'old formal figure sentinel')
    table_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / 'tables').glob('*.csv')}
    manifest_bytes = (out / 'qc/comparison_manifest.json').read_bytes()
    built = plotting.make_figures(data, plan, 'importance')
    plotting.save_figures(built, inp, out)
    assert old.read_bytes() == b'old formal figure sentinel'
    assert table_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / 'tables').glob('*.csv')}
    assert manifest_bytes == (out / 'qc/comparison_manifest.json').read_bytes()


@pytest.mark.skipif(not os.environ.get('MLR_REAL_OUTPUT_ROOT'), reason='Genuine HPC CSV root not supplied; real-data layout acceptance pending')
def test_genuine_csv_layout(tmp_path):
    root = Path(os.environ['MLR_REAL_OUTPUT_ROOT'])
    data, manifest = plotting.load_results(root)
    assert not any(p.get('synthetic') or p.get('synthetic_layout_only') for p in manifest['provenance'])
    plan = plotting.figure_plan(data)
    built = plotting.make_figures(data, plan, 'importance')
    # Export only to a separate temporary root; never write into the real dataset.
    temporary = tmp_path / 'real_csv_layout_qa'
    temporary.mkdir()
    (temporary / 'qc').mkdir()
    (temporary / 'qc/comparison_manifest.json').write_text(json.dumps(manifest))
    plotting.save_figures(built, root, temporary)
