import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import pytest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=''.join(json.loads((ROOT/'ncvi_float32_final_audit.ipynb').read_text())['cells'][0]['source'])


def embedded(name):
    fn=next(n for n in ast.parse(SOURCE).body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'np':np,'pd':pd,'Path':Path}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'audit','exec'),env)
    return env[name]


def test_float32_order_reproduces_saved_native_expression():
    rng=np.random.default_rng(51)
    rt=xr.DataArray(rng.normal(3e-6,1e-7,(51,5)).astype('float32'),dims=['number','step'],coords={'number':range(51)})
    hc=xr.DataArray(rng.normal(2e-6,1e-7,(20,11,5)).astype('float32'),dims=['time','number','step'])
    paths,meta=embedded('precision_paths')(rt,hc)
    saved=np.asarray((rt-hc.mean(list(hc.dims),skipna=True)).mean('step',skipna=True).values,dtype=float)*1e6
    np.testing.assert_array_equal(paths['old_mlr_native_si'],saved)
    np.testing.assert_allclose(np.float32(saved/1e6).astype(float)*1e6,saved,rtol=0,atol=1e-15)
    assert meta['native_stage_mean_dtype']=='float32'
    with pytest.raises(ValueError,match='not float32'):embedded('precision_paths')(rt.astype('float64'),hc)


def test_no_reverse_si_conversion_or_existing_result_writes():
    ast.parse(SOURCE)
    assert "regional=merged.mean(dim=['latitude','longitude'])" in SOURCE
    assert "already-PVU" not in SOURCE # rejects by explicit original-SI requirement
    assert "Old MLR SI replay requires actual SI source" in SOURCE
    assert 'original_run_sha256' in SOURCE and 'historical_mlr_float32_replay_qc' in SOURCE
    for n in ast.walk(ast.parse(SOURCE)):
        if isinstance(n,ast.ImportFrom):assert n.module not in ['ncvi_daily_exporter','ncvi_source_preflight']
    assert 'daily_internal_aggregation_qc' in SOURCE and 'era5_reference_qc' in SOURCE
