import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]


def nb():return json.loads((ROOT/'process_fig5_fig11_diagnostic_standalone.ipynb').read_text())


def test_selfcontained_and_explicit_diagnostic_contract():
    n=nb();assert len(n['cells'])==6
    whole=''.join(''.join(c['source']) for c in n['cells'])
    assert "'formal_process_allowed':False" in whole and 'table_ncvi_concurrent_daily_members.csv' in whole
    assert 'daily_ncvi_anomaly_pvu' in whole and "if factor=='NCVI_concurrent':continue" in whole
    for c in n['cells']:
        assert c['execution_count'] is None and not c['outputs']
        for x in ast.walk(ast.parse(''.join(c['source']))):
            if isinstance(x,ast.ImportFrom):assert not x.module.startswith(('process_','ncvi_'))
    assert '__file__' not in whole and 'subprocess' not in whole


def test_synthetic_panel_counts_alignment_text_and_boundary():
    env=dict(np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,INITS=('2023-06-08','2023-06-12'),GROUPS=('Hot17','Middle17','Cold17'))
    nodes=[n for n in ast.parse(''.join(nb()['cells'][4]['source'])).body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (isinstance(n,ast.FunctionDef) or any(isinstance(t,ast.Name) and t.id in ['COLORS','LABELS','STYLE','MARKERS','LINESTYLES'] for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'layout','exec'),env)
    for window,factors,end,count in [('Total',('NCHN_tp','z500_anom_NCHN','sm_avg','NCVI_concurrent','TCC_Avg'),'2023-06-24',10),('Stage-I',('partial_fitted_Tmax','sm_avg','z500_anom_NCHN','SHF_Avg','NCVI_concurrent','TCC_Avg'),'2023-06-18',12)]:
        records=[dict(init_date=init,date=date,group=g,factor=f,value=0.5+i*.02,model_version='raw') for init in env['INITS'] for f in factors for g in env['GROUPS'] for i,date in enumerate(pd.date_range('2023-06-14',end))]
        with plt.rc_context(env['STYLE']):
            fig,axes=env['timeseries_figure'](pd.DataFrame(records),factors,window,'raw')
            assert len(axes)==count and fig.get_figwidth()==6.5
            assert env['figure_qc'](fig,axes)['text_extent_qc']=='PASS'
            if window=='Stage-I':assert all(len(a.patches)==1 for a in axes)
            plt.close(fig)


def test_scatter_layout_and_explicit_plot_tables():
    whole=''.join(''.join(c['source']) for c in nb()['cells'])
    for name in ['fig5_total_timeseries_plot.csv','fig11_stageI_timeseries_','figS11_stageI_scatter_','group_membership_audit.csv','consistency_manifest.json']:
        assert name in whole
    assert 'fig.suptitle' not in whole
    env=dict(np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,INITS=('2023-06-08','2023-06-12'),GROUPS=('Hot17','Middle17','Cold17'))
    nodes=[n for n in ast.parse(''.join(nb()['cells'][4]['source'])).body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (isinstance(n,ast.FunctionDef) or any(isinstance(t,ast.Name) and t.id in ['COLORS','LABELS','STYLE','MARKERS','LINESTYLES'] for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'scatter','exec'),env)
    records=[dict(init_date=init,member=m,group=env['GROUPS'][m//17],model_version=v,sm_avg=.15+m*.001,y_tmax=30+m*.08,partial_fitted_Tmax=31+m*.04+(.1 if v=='raw' else 0)) for init in env['INITS'] for v in ['raw','residualized'] for m in range(51)]
    with plt.rc_context(env['STYLE']):
        fig,axes=env['scatter_figure'](pd.DataFrame(records),'raw')
        assert len(axes)==4 and axes[0].get_xlabel().startswith('SM (')
        assert axes[2].get_xlabel().startswith('SM–SHF')
        assert env['figure_qc'](fig,axes)['text_extent_qc']=='PASS'
        plt.close(fig)


def test_statistics_use_individual_51_members_and_no_group_mean():
    env=dict(np=np,pd=pd)
    fn=next(n for n in ast.parse(''.join(nb()['cells'][4]['source'])).body if isinstance(n,ast.FunctionDef) and n.name=='scatter_statistics')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'stats','exec'),env)
    x=np.arange(51,dtype=float);y=2*x+np.sin(x)*5
    d=pd.DataFrame(dict(init_date=['2023-06-08']*51,model_version=['raw']*51,member=np.arange(51),sm_avg=x,y_tmax=y))
    q=env['scatter_statistics'](d,'raw','2023-06-08','sm_avg')
    assert q['n']==51 and np.isclose(q['pearson_r'],np.corrcoef(x,y)[0,1])
    assert q['descriptive_R2']==q['pearson_r']**2
    assert np.isclose(q['trend_slope'],np.polyfit(x,y,1)[0])
    import pytest
    with pytest.raises(ValueError,match='51 original'):env['scatter_statistics'](d.iloc[:17],'raw','2023-06-08','sm_avg')
