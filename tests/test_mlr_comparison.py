"""Synthetic interface/math checks only. No HPC data or paper results."""
import ast
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import mlr_comparison_core as core


def synthetic_inputs():
    rng = np.random.RandomState(2718)
    raw, nc = [], []
    for init in core.INITS:
        for window, stage in core.STAGES.items():
            z = rng.normal(size=(51, 7))
            for member in range(51):
                a = z[member]
                smv = 0.25 + 0.03 * a[0]
                tp = 15 + 2 * a[1]
                shf = 80 + 25 * a[0] + 7 * a[2]
                tcc = 0.5 + 0.03 * a[0] + 0.025 * a[1] + 0.03 * a[3]
                row = {'init_date': init, 'stage': stage, 'member': member,
                       'sm_avg': smv, 'NCHN_tp': tp, 'SHF_Avg': shf, 'TCC_Avg': tcc,
                       'z500_anom_NCHN': 30 * a[4], 'Initial_MSEstar_max_NCHN': 300 + 3 * a[5],
                       'y_tmax': 35 + 0.5 * a[0] - 0.35 * a[1] + 0.3 * a[2] + 0.4 * a[3] + 0.35 * a[4] + 0.2 * a[5] + 0.45 * a[6] + rng.normal(scale=0.25)}
                raw.append(row)
                nc.append({'init_date': init, 'stage': stage, 'member': member,
                           'NCVI_concurrent': a[6],
                           'ncvi_concurrent_window_start': core.WINDOWS[window][0],
                           'ncvi_concurrent_window_end': core.WINDOWS[window][1]})
    # Scrambled input rows must not affect member pairing.
    return pd.DataFrame(raw).sample(frac=1, random_state=11), pd.DataFrame(nc).sample(frac=1, random_state=19)


@pytest.fixture(scope='module')
def calculated():
    raw, nc = synthetic_inputs()
    groups, ra = core.prepare_pairs(raw, nc)
    models = core.fit_pairs(groups)
    permutations = core.lmg_permutations()
    imp, qc = core.importance(models, permutations)
    results = core.assemble_results(groups, models, imp, qc, ra)
    return raw, nc, groups, models, permutations, results


def original_functions(*names):
    """Compile selected pure function ASTs, never import/run the historical pipeline."""
    source = Path('stage_specific_residualized_ols_shf.py').read_text()
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'np': np, 'pd': pd, 'sm': sm, 'LinearRegression': LinearRegression,
                 'JOHNSON_DISPLAY_LABELS': core.LABELS}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<selected pure historical functions>', 'exec'), namespace)
    return namespace


def test_configuration_matches_original_and_raw_mapping():
    tree = ast.parse(Path('stage_specific_residualized_ols_shf.py').read_text())
    original = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SLIM_PREDICTORS_BY_STAGE' for t in n.targets))
    for w, stage in core.STAGES.items():
        assert original[stage] == core.predictors(w, 'residualized')
    assert core.predictors('Stage-I', 'raw') == ['sm_avg', 'SHF_Avg', 'z500_anom_NCHN', 'Initial_MSEstar_max_NCHN', 'TCC_Avg', 'NCVI_concurrent']
    assert core.predictors('Total', 'raw') == ['NCHN_tp', 'z500_anom_NCHN', 'NCVI_concurrent', 'Initial_MSEstar_max_NCHN', 'TCC_Avg', 'sm_avg']


def test_pairs_ols_scaler_and_equivalence(calculated):
    _, _, groups, models, _, results = calculated
    assert len(groups) == 8 and len(models) == 16
    assert (results['equivalence'].status == 'PASS').all()
    for (init, w, v), m in models.items():
        assert m['rank'] == 7
        np.testing.assert_allclose(m['scaler'].scale_, np.std(m['X'], axis=0, ddof=0), rtol=1e-12)
        np.testing.assert_allclose(m['raw_intercept'] + m['X'] @ m['raw_slopes'], m['fitted'], atol=1e-8)
        independent = LinearRegression().fit(m['X'], m['y'])
        np.testing.assert_allclose(independent.predict(m['X']), m['fitted'], atol=1e-9)
    assert results['fitted'].shape[0] == 816
    assert results['comparison'].shape[0] == 144


@pytest.mark.parametrize('window', ['Stage-I', 'Total'])
def test_residuals_and_johnson_exact_original(calculated, window):
    _, _, groups, models, _, _ = calculated
    original = original_functions('_slim_resid_series', '_johnson_relative_weights')
    g = groups[(core.INITS[0], window)]
    for target, source, control in core.RESIDUALIZATION[window]:
        np.testing.assert_array_equal(g[target], original['_slim_resid_series'](g, source, [control]))
    for version in core.VERSIONS:
        m = models[(core.INITS[0], window, version)]
        args = (pd.DataFrame(m['X'], columns=m['names']), m['y'], m['names'])
        old = original['_johnson_relative_weights'](*args)
        new = core._johnson_relative_weights(*args)
        pd.testing.assert_frame_equal(old[0], new[0])
        assert old[1:] == new[1:]
        assert abs(new[1] - m['R2']) < 1e-5


@pytest.mark.parametrize('version', core.VERSIONS)
def test_lmg_original_sequence_and_algorithm(calculated, version):
    _, _, _, models, perms, results = calculated
    m = models[(core.INITS[0], 'Stage-I', version)]
    old = original_functions('_slim_compute_lmg_mc')['_slim_compute_lmg_mc'](m['X'], m['y'], m['names'], mc_samples=2000)
    new, _ = core.lmg(m['X'], m['y'], perms)
    np.testing.assert_allclose(new, [old[n] for n in m['names']], atol=1e-10, rtol=0)
    np.random.seed(42)
    expected = np.array([np.random.permutation(6) for _ in range(2000)])
    np.testing.assert_array_equal(perms, expected)
    assert results['model_qc'].lmg_permutation_sha256.nunique() == 1


def test_lofo_fixed_columns_and_zero_denominator(calculated):
    _, _, _, models, _, results = calculated
    m = models[(core.INITS[0], 'Total', 'raw')]
    rows = results['importance']
    rows = rows[(rows.init_date == core.INITS[0]) & (rows.window == 'Total') & (rows.version == 'raw') & (rows.method == 'LOFO')].sort_values('position')
    for j in range(6):
        keep = [k for k in range(6) if k != j]
        independent = LinearRegression().fit(m['X'][:, keep], m['y'])
        assert abs(rows.iloc[j].raw_value - (m['R2'] - independent.score(m['X'][:, keep], m['y']))) < 1e-10
    pct, denominator = core.positive_lofo_percent([0, -1e-14, 0])
    assert denominator == 0 and np.isnan(pct).all()
    np.testing.assert_array_equal(core.stable_ranks([20, 20, 10]), [1, 2, 3])


@pytest.mark.parametrize('fault', ['duplicate', 'missing_member', 'wrong_member', 'nonfinite', 'missing_field', 'wrong_window'])
def test_input_failures(fault):
    raw, nc = synthetic_inputs()
    if fault == 'duplicate':
        raw = pd.concat([raw, raw.iloc[[0]]])
    elif fault == 'missing_member':
        nc = nc.iloc[1:]
    elif fault == 'wrong_member':
        raw.loc[raw.index[0], 'member'] = 51
    elif fault == 'nonfinite':
        raw.loc[raw.index[0], 'TCC_Avg'] = np.inf
    elif fault == 'missing_field':
        raw = raw.drop(columns='SHF_Avg')
    else:
        nc.loc[nc.index[0], 'ncvi_concurrent_window_end'] = '2023-06-25'
    with pytest.raises(ValueError):
        core.prepare_pairs(raw, nc)


def test_rank_deficiency_diagnostic():
    x = np.random.RandomState(8).normal(size=(51, 6))
    x[:, 5] = x[:, 4]
    with pytest.raises(ValueError, match='singular_values'):
        core.fit_model(x, np.arange(51), list('abcdef'), 'test/init/window/raw')


def test_output_symlinks_cannot_redirect_writes(tmp_path):
    inp, out = tmp_path / 'input', tmp_path / 'comparison'
    (inp / 'tables').mkdir(parents=True)
    (out / 'qc').mkdir(parents=True)
    (out / 'qc/comparison_manifest.json').write_text(json.dumps(
        {'schema_version': 1, 'source_commit': core.SOURCE_COMMIT}))
    (out / 'tables').symlink_to(inp / 'tables', target_is_directory=True)
    with pytest.raises(ValueError, match='Unsafe output subdirectory'):
        core.output_guard(inp, out)


def history_fixture(results):
    """CSV-schema fixtures for the comparator; NOT an independent scientific baseline."""
    history = {}
    for key in ['summary', 'coefficients', 'fitted']:
        d = results[key]
        d = d[d.version == 'residualized'].copy()
        d['stage'] = d.window.map(core.STAGES)
        history[key] = d
    i = results['importance']
    i = i[i.version == 'residualized'].copy()
    i['stage'] = i.window.map(core.STAGES)
    history['lmg'] = i[i.method == 'LMG'].rename(columns={'importance_pct': 'LMG_percent'})
    history['lofo'] = i[i.method == 'LOFO'].rename(columns={'raw_value': 'Delta_R2', 'importance_pct': 'positive_delta_R2_share_percent'})
    history['methods'] = i[i.method == 'Johnson'].rename(columns={'predictor': 'factor', 'raw_value': 'johnson_raw', 'importance_pct': 'johnson_pct'})
    history['predictors'] = pd.DataFrame([{'stage': stage, 'predictors_raw_name': p}
                                        for w, stage in core.STAGES.items() for p in core.predictors(w, 'residualized')])
    return history


def test_history_detects_fitted_and_coefficient_drift(calculated):
    results = calculated[-1]
    history = history_fixture(results)
    assert core.check_predictor_history(history) == 'PASS'
    good = core.compare_history(results, history)
    assert (good.status == 'PASS').all()
    history['fitted'].loc[history['fitted'].index[0], 'y_fitted'] += 0.01
    history['coefficients'].loc[history['coefficients'].index[0], 'standardized_coef'] += 0.01
    bad = core.compare_history(results, history)
    assert (bad.status == 'FAIL').sum() >= 2
    assert core.qc_status(results, bad)['computational_status'] == 'FAIL'
    absent = core.compare_history(results, {})
    assert core.qc_status(results, absent)['history_status'] == 'INCOMPLETE'
    history['predictors'] = history['predictors'].iloc[::-1]
    with pytest.raises(ValueError, match='order mismatch'):
        core.check_predictor_history(history)


def test_output_protection_and_saved_results(calculated, tmp_path):
    results, perms = calculated[-1], calculated[-2]
    inp = tmp_path / 'input'
    for out in [inp, inp / 'tables/new', inp / 'figures/new', inp / 'paper_figures_standalone/new', tmp_path]:
        with pytest.raises(ValueError):
            core.output_guard(inp, out)
    out = inp / 'mlr_raw_residual_comparison'
    h = core.compare_history(results, {})
    manifest = core.save_results(results, h, inp, out, [{'synthetic': True}], perms)
    assert manifest['computational_status'] == 'PASS'
    assert manifest['history_status'] == 'INCOMPLETE'
    import mlr_comparison_plotting as plotting
    data, _ = plotting.load_results(out)
    # Arbitrary synthetic scientific rankings must not silently replace the confirmed reference.
    with pytest.raises(ValueError, match='12 June residualized LMG order mismatch'):
        plotting.figure_plan(data)
    with (out / 'tables/table_mlr_fitted_members.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='changed after computation QC'):
        plotting.load_results(out)


def test_import_has_no_analysis_io_or_global_style(tmp_path):
    code = """
import pathlib, matplotlib, matplotlib.pyplot
matplotlib.use('Agg')
before = matplotlib.rcParams.copy()
import mlr_comparison_core, mlr_comparison_plotting
assert dict(before) == dict(matplotlib.rcParams)
assert not list(pathlib.Path('.').iterdir())
"""
    env = dict(os.environ, PYTHONPATH=str(Path.cwd()),
               MPLCONFIGDIR=str(tmp_path.parent / 'matplotlib_cache'))
    # The optional test dependency path is appended, never shadowing installed numpy.
    extra = os.environ.get('MLR_TEST_EXTRA_SITE')
    bootstrap = f"import site; site.addsitedir({extra!r});\n" if extra else ''
    subprocess.run([sys.executable, '-c', bootstrap + code], cwd=tmp_path, env=env, check=True)


def test_notebooks_schema_and_ast():
    import nbformat
    for name in ['mlr_raw_residual_comparison.ipynb', 'mlr_comparison_figures.ipynb']:
        nb = nbformat.read(name, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                assert cell.execution_count is None and cell.outputs == []
                assert '__file__' not in cell.source and '%run' not in cell.source


def test_original_blobs_preserved():
    expected = {'stage_specific_residualized_ols_shf.py': 'a132b0e1875dba812c2be6bceb9c423bd9d2d702',
                'stage_specific_residualized_ols_shf.ipynb': 'e9fa7882fbf6da3aac700b2460a6f26ec4936a6d',
                'methods_audit_mlr_attribution.md': '7ca196d87d46d1a47d1bfae813409bdd02f6244c',
                'methods_audit_stageI_total_process.md': '79a1e900ca224c5802598c8f974ab0424d2be2b7'}
    for name, sha in expected.items():
        b = Path(name).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == sha
