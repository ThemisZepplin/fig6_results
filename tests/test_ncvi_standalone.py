import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def notebook():
    return json.loads((ROOT/'ncvi_daily_export_standalone.ipynb').read_text())


def test_six_standalone_cells_and_dependencies():
    nb=notebook()
    assert len(nb['cells'])==6
    for i,c in enumerate(nb['cells'],1):
        assert c['execution_count'] is None and c['outputs']==[]
        source=''.join(c['source']);tree=ast.parse(source)
        assert f'capture_cell({i})' in source
        assert 'print(' in source
        assert '__file__' not in source and 'git rev-parse' not in source
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):
                assert n.module not in ['ncvi_daily_exporter','ncvi_source_preflight','process_timeseries_core']
    assert 'CONFIG_PATH' not in ''.join(''.join(c['source']) for c in nb['cells'])


def test_scientific_functions_identical_to_reviewed_modules():
    functions={}
    for c in notebook()['cells']:
        for n in ast.parse(''.join(c['source'])).body:
            if isinstance(n,ast.FunctionDef):functions[n.name]=ast.dump(n,include_attributes=False)
    for file,names in [('ncvi_daily_exporter.py',['digest','load_pair','daily_replay']),('ncvi_source_preflight.py',['inspect_grib'])]:
        nodes={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((ROOT/file).read_text()).body if isinstance(n,ast.FunctionDef)}
        for name in names:assert functions[name]==nodes[name]


def test_notebook_embedded_replay_runs_without_repository_imports():
    tree=ast.parse(''.join(notebook()['cells'][3]['source']))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='daily_replay')
    env={'np':np,'pd':pd,'INITS':('2023-06-08','2023-06-12'),'DATES':pd.date_range('2023-06-14','2023-06-24'),'ATOL':1e-8}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'embedded','exec'),env)
    dates=env['DATES'];steps=(dates-pd.Timestamp('2023-06-08')-pd.Timedelta(hours=8)).to_numpy()
    hc=np.zeros((11,3));rt=np.ones((11,51))
    ref=pd.DataFrame([{'init_date':'2023-06-08','window':w,'member':m,'NCVI_concurrent':1.} for w in ['Stage-I','Total'] for m in range(51)])
    d,c,q=env['daily_replay']('2023-06-08',rt,steps,hc,steps,ref)
    assert len(d)==561 and len(q)==102 and q.scalar_pass.all() and q.daily_pass.all()
