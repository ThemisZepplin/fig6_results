import numpy as np
import pandas as pd
import pytest
from ncvi_daily_exporter import daily_replay


def case(unequal=False):
    times=pd.date_range('2023-06-14',periods=11,freq='D')
    if unequal:times=times.insert(1,pd.Timestamp('2023-06-14 06:00'))
    steps=(times-pd.Timestamp('2023-06-08')-pd.Timedelta(hours=8)).to_numpy()
    clim=np.arange(len(times),dtype=float)[:,None]+np.zeros((len(times),3))
    rt=clim[:,[0]]+np.arange(51)[None,:]
    if unequal: rt=rt+np.arange(len(times))[:,None]*0.2
    ref=[]
    for window,end in [('Stage-I','2023-06-17'),('Total','2023-06-24')]:
        mask=times.normalize()<=pd.Timestamp(end)
        for m in range(51):ref.append(dict(init_date='2023-06-08',window=window,member=m,NCVI_concurrent=float(rt[mask,m].mean()-clim[mask].mean())))
    return rt,steps,clim,steps,pd.DataFrame(ref)


def test_daily_scalar_equivalence_uniform_samples():
    d,c,q=daily_replay('2023-06-08',*case())
    assert len(d)==561 and len(q)==102 and q.scalar_pass.all() and q.daily_pass.all()
    assert np.array_equal(d.daily_ncvi_anomaly_pvu.to_numpy(),np.tile(np.arange(51),11))


def test_unequal_day_counts_not_corrected_to_pass():
    d,c,q=daily_replay('2023-06-08',*case(True))
    assert q.scalar_pass.all() and not q.daily_pass.all()
    assert q.daily_difference.abs().max()>0


def test_missing_day_and_nonfinite_fail_closed():
    rt,s,h,hs,ref=case()
    with pytest.raises(ValueError,match='coverage'):
        daily_replay('2023-06-08',rt[:-1],s[:-1],h,hs,ref)
    rt[0,0]=np.nan
    with pytest.raises(ValueError,match='Missing'):
        daily_replay('2023-06-08',rt,s,h,hs,ref)


def test_missing_era5_is_independent_failure():
    from ncvi_daily_exporter import era5_reference_qc
    q=era5_reference_qc({})
    assert q['status']=='FAIL' and q['reason']
    _,_,stage=daily_replay('2023-06-08',*case())
    assert stage.scalar_pass.all() and stage.daily_pass.all()


def test_preflight_missing_inputs_reports_no_config(tmp_path):
    from argparse import Namespace
    from ncvi_source_preflight import build
    args=Namespace(report_dir=str(tmp_path/'report'),export_root=str(tmp_path/'export'),
                   stage_table=str(tmp_path/'missing.csv'),design_table=str(tmp_path/'design.csv'),
                   comparison_manifest=str(tmp_path/'manifest.json'),era5_csv=None,era5_provenance=None,
                   realtime_pf='missing-pf.grib',realtime_cf='missing-cf.grib')
    with pytest.raises(RuntimeError):build(args)
    assert (tmp_path/'report'/'ncvi_source_preflight_report.json').is_file()
    assert not (tmp_path/'report'/'ncvi_daily_hpc_config.json').exists()
