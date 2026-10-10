"""Fresh-kernel Run All checks on temporary, explicitly synthetic CSVs."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

nbformat = pytest.importorskip('nbformat')
nbclient = pytest.importorskip('nbclient')
pytest.importorskip('ipykernel')
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

from test_mlr_comparison import synthetic_inputs


def execute_notebook(name, root, inp, out):
    kernel_dir = root / 'kernels/python3'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    extra = os.environ.get('MLR_TEST_EXTRA_SITE')
    bootstrap = f'import site; site.addsitedir({extra!r}); ' if extra else ''
    env = {
        'MLR_CODE_ROOT': str(Path.cwd()), 'MLR_INPUT_ROOT': str(inp),
        'MLR_OUTPUT_ROOT': str(out), 'MPLBACKEND': 'Agg',
        'MPLCONFIGDIR': str(root / 'matplotlib'), 'IPYTHONDIR': str(root / 'ipython'),
        'JUPYTER_RUNTIME_DIR': str(root / 'runtime'),
    }
    kernel_dir.joinpath('kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-c', bootstrap +
                 'from ipykernel.kernelapp import IPKernelApp; IPKernelApp.launch_instance()',
                 '-f', '{connection_file}'],
        'display_name': 'Synthetic test kernel', 'language': 'python', 'env': env}))
    manager = KernelManager(kernel_name='python3',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(root / 'kernels')]))
    nb = nbformat.read(name, as_version=4)
    # A new NotebookClient/KernelManager is used for every notebook execution.
    client = nbclient.NotebookClient(nb, km=manager, timeout=120,
                                    resources={'metadata': {'path': str(Path.cwd())}})
    client.execute()
    return nb


def test_both_notebooks_restart_run_all_without_hidden_state(tmp_path):
    inp, out = tmp_path / 'synthetic_input', tmp_path / 'synthetic_comparison'
    tables = inp / 'tables'
    tables.mkdir(parents=True)
    raw, nc = synthetic_inputs()
    raw.to_csv(tables / 'table_stage_member_raw_factors_TCC.csv', index=False)
    nc.to_csv(tables / 'table_ncvi_concurrent_stage_values.csv', index=False)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.iterdir()}
    execute_notebook('mlr_raw_residual_comparison.ipynb', tmp_path, inp, out)
    manifest = json.loads((out / 'qc/comparison_manifest.json').read_text())
    assert manifest['computational_status'] == 'PASS'
    assert manifest['history_status'] == 'INCOMPLETE'
    execute_notebook('mlr_comparison_figures.ipynb', tmp_path, inp, out)
    assert len(list((out / 'figures').glob('*.png'))) == 8
    assert len(list((out / 'figures').glob('*.pdf'))) == 8
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.iterdir()}


def test_missing_hpc_inputs_fail_explicitly_before_output(tmp_path):
    inp, out = tmp_path / 'absent_input', tmp_path / 'absent_output'
    with pytest.raises(nbclient.exceptions.CellExecutionError, match='Missing exported CSV'):
        execute_notebook('mlr_raw_residual_comparison.ipynb', tmp_path, inp, out)
    assert not out.exists()
