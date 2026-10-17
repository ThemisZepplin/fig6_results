import pandas as pd
from process_v12_adapter import group_daily
import pytest


def test_reject_incomplete_group_instead_of_partial_mean():
    d=pd.DataFrame({'init_date':['2023-06-08']*16,'window':['Stage-I']*16,
                    'date':['2023-06-14']*16,'group':['Hot17']*16,
                    'factor':['sm_avg']*16,'value':[1.0]*16})
    with pytest.raises(ValueError,match='17 members'):
        group_daily(d)


def test_saved_real_audit_and_group_invariants():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'audits/process_v12_daily_20261014'
    q=pd.read_csv(root/'daily_to_stage_qc.csv')
    assert len(q)==51*4*6 and q.passed.all()
    assert q.difference.abs().max()<1e-8
    g=pd.read_csv(root/'current_v12_groups.csv')
    assert (g.groupby(['init_date','window','group']).size()==17).all()
    s=g[(g.init_date=='2023-06-08')&(g.window=='Stage-I')].set_index('member')
    assert s.loc[43,'group']=='Hot17' and s.loc[39,'group']=='Middle17'
    x=pd.read_csv(root/'internal_scatter.csv')
    p=x.pivot(index=['init_date','member'],columns='model_version',values=['y_tmax','sm_avg'])
    for col in ['y_tmax','sm_avg']:
        assert (p[col]['raw']==p[col]['residualized']).all()
    assert x.status.eq('INTERNAL_ONLY_NCVI_MISSING').all()
