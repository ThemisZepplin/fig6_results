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
    nodes=[n for n in ast.parse(''.join(nb()['cells'][4]['source'])).body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (isinstance(n,ast.FunctionDef) or any(isinstance(t,ast.Name) and t.id in ['COLORS','LABELS','STYLE'] for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'layout','exec'),env)
    for window,factors,end,count in [('Total',('NCHN_tp','z500_anom_NCHN','sm_avg','NCVI_concurrent','TCC_Avg'),'2023-06-24',10),('Stage-I',('partial_fitted_Tmax','sm_avg','z500_anom_NCHN','SHF_Avg','NCVI_concurrent','TCC_Avg'),'2023-06-18',12)]:
        records=[dict(init_date=init,date=date,group=g,factor=f,value=0.5+i*.02,model_version='raw') for init in env['INITS'] for f in factors for g in env['GROUPS'] for i,date in enumerate(pd.date_range('2023-06-14',end))]
        with plt.rc_context(env['STYLE']):
            fig,axes=env['timeseries_figure'](pd.DataFrame(records),factors,window,'raw')
            assert len(axes)==count and fig.get_figwidth()==6.5
            assert env['figure_qc'](fig,axes)['text_extent_qc']=='PASS'
            if window=='Stage-I':assert all(len(a.patches)==1 for a in axes)
            plt.close(fig)
