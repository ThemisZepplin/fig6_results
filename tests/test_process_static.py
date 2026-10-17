"""Static checks only: no input-data projections or synthetic scientific figures."""
import ast
import json
from pathlib import Path
import subprocess


def test_modules_parse_and_have_no_top_level_analysis_calls():
    for name in ('process_timeseries_core.py', 'process_timeseries_plotting.py'):
        tree = ast.parse(Path(name).read_text())
        assert all(isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.Expr)) for n in tree.body)
        for n in tree.body:
            if not isinstance(n, ast.FunctionDef):
                assert not any(isinstance(x, ast.Call) for x in ast.walk(n))


def test_notebook_static_entry_and_no_embedded_results():
    notebook = json.loads(Path('process_figures_v2.ipynb').read_text())
    assert notebook['nbformat'] == 4
    codes = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(codes) == 6
    for cell in codes:
        source = ''.join(cell['source'])
        ast.parse(source)
        assert cell['execution_count'] is None and cell['outputs'] == []
        assert '__file__' not in source and '%run' not in source and 'exec(' not in source
    source = '\n'.join(''.join(c['source']) for c in codes)
    assert 'core.build_tables' in source and 'plotting.render_all' in source


def test_no_model_fitting_or_upstream_scientific_file_imports():
    for name in ('process_timeseries_core.py', 'process_timeseries_plotting.py'):
        tree = ast.parse(Path(name).read_text())
        imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert 'stage_specific_residualized_ols_shf' not in imported
        assert 'mlr_comparison_core' not in imported
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        assert 'fit' not in calls and 'fit_transform' not in calls and 'polyfit' not in calls


def test_expected_outputs_and_projection_fields_are_declared():
    tree = ast.parse(Path('process_timeseries_core.py').read_text())
    output = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'OUTPUT_TABLES' for t in n.targets))
    assert len(output) == 4 and len(set(output)) == 4
    text = Path('process_timeseries_core.py').read_text()
    for field in ('standardized_intercept', 'standardized_coef', 'scale_ddof0', 'residual_intercept',
                  'residual_slope', 'held_value', 'parameter_source', 'model_train_end_bjt'):
        assert field in text
    assert "def partial_fitted_tmax(" in text


def test_development_files_are_unchanged():
    base = '096cad22cc84cf8b14a5ee70a5db80c762891a65'
    files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', base], text=True).splitlines()
    for name in files:
        expected = subprocess.check_output(['git', 'show', f'{base}:{name}'])
        assert Path(name).read_bytes() == expected


def test_group_source_and_anomaly_gates_are_declared_without_fitting():
    source = Path('process_timeseries_core.py').read_text()
    for entry in ('probe_sources', 'audit_exporter_notebook', 'current_group_audit',
                  'check_units', 'reaggregate_member_daily'):
        assert f'def {entry}(' in source
    for token in ('middle17_mean', 'legacy_match', 'reaggregate_member_daily',
                  'absolute fallback', 'mean_state_prediction', 'full_fit_max_abs_difference'):
        assert token in source
    assert "'residualization', 'fitted'" in source


def test_corrected_scatter_orientation_and_stage_boundary():
    source = Path('process_timeseries_plotting.py').read_text()
    assert "for row, factor in enumerate(('sm_avg', 'partial_fitted_Tmax'))" in source
    assert 'for col, init in enumerate(INITS)' in source
    assert "pd.Timestamp('2023-06-18 00:00')" in source
    assert '2023-06-17 12:00' not in source
