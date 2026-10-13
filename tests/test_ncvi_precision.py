import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]


def embedded():
    nb=json.loads((ROOT/'ncvi_precision_diagnostic.ipynb').read_text())
    return ''.join(nb['cells'][0]['source'])


def test_precision_cell_is_readonly_for_existing_results():
    tree=ast.parse(embedded())
    assert '__file__' not in embedded() and 'subprocess' not in embedded()
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom):assert n.module not in ['ncvi_daily_exporter','ncvi_source_preflight']
        if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store):assert n.id not in ['daily','replay','result','STATE','ATOL','config','ref']


def test_native_order_matches_original_expression():
    node=next(n for n in ast.parse(embedded()).body if isinstance(n,ast.FunctionDef) and n.name=='precision_paths')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'precision','exec'),env)
    rng=np.random.default_rng(819)
    rt=xr.DataArray(rng.normal(3e-6,2e-7,(51,4)).astype('float32'),dims=['number','step'],coords={'number':range(51),'step':range(4)})
    hc=xr.DataArray(rng.normal(2e-6,1e-7,(20,11,4)).astype('float32'),dims=['time','number','step'],coords={'time':pd.date_range('2003-06-08',periods=20,freq=pd.DateOffset(years=1)),'number':range(11),'step':range(4)})
    paths,m=env['precision_paths'](rt,hc)
    scalar=hc.mean(dim=list(hc.dims),skipna=True)
    expected=np.asarray((rt-scalar).mean(dim=['step'],skipna=True).sel(number=np.arange(51)).values,dtype=float).squeeze()*1e6
    np.testing.assert_array_equal(paths['old_mlr_native_si'],expected)
    assert m['rt_si_dtype']=='float32' and m['hc_samples']==220
    assert max(abs(paths['old_mlr_native_si']-paths['exporter_pvu_float64']))>0
