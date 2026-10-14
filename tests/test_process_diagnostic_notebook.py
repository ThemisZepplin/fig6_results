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
    assert 'load_era5_ncvi_reference' in whole and 'PLOTTED_PROVENANCE_PENDING' in whole
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
            assert fig.get_figheight()*25.4<=228
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


def test_pdf_qc_rejects_size_and_small_body_and_classifies_scripts():
    from types import SimpleNamespace
    fn=next(n for n in ast.parse(''.join(nb()['cells'][4]['source'])).body if isinstance(n,ast.FunctionDef) and n.name=='pdf_page_qc')
    env={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pdf','exec'),env)
    class Page:
        def __init__(self,height=8.9,body=8,script=5.6):
            self.rect=SimpleNamespace(width=6.5*72,height=height*72)
            self.spans=[dict(text='Tmax',size=body,flags=0,bbox=(10,10,30,20)),dict(text='2',size=script,flags=1,bbox=(31,10,35,15))]
        def get_text(self,mode):return {'blocks':[{'lines':[{'spans':self.spans}]}]}
    assert env['pdf_page_qc'](Page(),8.9)['status']=='PASS'
    assert env['pdf_page_qc'](Page(body=7),8.9)['status']=='FAIL'
    assert env['pdf_page_qc'](Page(script=4),8.9)['status']=='FAIL'
    assert env['pdf_page_qc'](Page(height=10.5),10.5)['status']=='FAIL'


def test_cell4_member_diagnostic_branch_executes_dict_y(tmp_path):
    source=''.join(nb()['cells'][3]['source'])
    # Execute the actual notebook branch from stage-input loading through pairing.
    branch=source[source.index('stage_members=pd.read_csv(STAGE)'):source.index('for variable in FACTORS11:')]
    stage=pd.DataFrame([dict(init_date=init,stage='Stage-I',member=m,y_tmax=30+m*.1,sm_avg=.15+m*.001,SHF_Avg=50+m) for init in ['2023-06-08','2023-06-12'] for m in range(51)])
    path=tmp_path/'stage.csv';stage.to_csv(path,index=False)
    groups=stage[['init_date','member']].copy();groups['window']='Stage-I';groups['group']=['Hot17' if m<17 else 'Middle17' if m<34 else 'Cold17' for m in groups.member]
    calls=[]
    def parameters(data,init,version):
        calls.append((init,version))
        return {'y':stage[stage.init_date==init].set_index('member').y_tmax.copy(),'version':version}
    env=dict(np=np,pd=pd,STAGE=path,INITS=('2023-06-08','2023-06-12'),VERSIONS=('raw','residualized'),
             normalize_window=lambda s:s,parameters=parameters,data={},current_groups=groups,
             residual_relation=lambda data,init:(0.,1.),
             partial_fitted_tmax=lambda sm,shf,model,relation:(np.asarray(sm)+np.asarray(shf)+(1 if model['version']=='raw' else 0),np.asarray(shf)))
    exec(compile(branch,'actual-cell4-member-branch','exec'),env)
    assert len(calls)==4 and len(env['scatter_table'])==204
    assert 'atol=1e-8' in branch
    import pytest
    def mismatched_parameters(data,init,version):
        result=parameters(data,init,version);result['y']=result['y']+.001;return result
    env['parameters']=mismatched_parameters
    with pytest.raises(ValueError,match='target/model mismatch'):
        exec(compile(branch,'actual-cell4-member-branch','exec'),env)


def test_daily_markers_land_exactly_on_labelled_ticks_and_scatter_stats_use_right_side():
    """Exercise true plotting functions, not only their static strings."""
    env=dict(np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,
             INITS=('2023-06-08','2023-06-12'),GROUPS=('Hot17','Middle17','Cold17'))
    ast_nodes=[node for node in ast.parse(''.join(nb()['cells'][4]['source'])).body
               if isinstance(node,(ast.FunctionDef,ast.Assign)) and
               (isinstance(node,ast.FunctionDef) or
                any(isinstance(t,ast.Name) and t.id in
                    ('COLORS','LABELS','STYLE','MARKERS','LINESTYLES') for t in node.targets))]
    exec(compile(ast.Module(body=ast_nodes,type_ignores=[]),'layout','exec'),env)
    for window,factors,end,expect_ticks in [
        ('Total',('NCHN_tp','z500_anom_NCHN','sm_avg','NCVI_concurrent','TCC_Avg'),'2023-06-24',11),
        ('Stage-I',('partial_fitted_Tmax','sm_avg','z500_anom_NCHN','SHF_Avg','NCVI_concurrent','TCC_Avg'),'2023-06-18',5)]:
        records=[dict(init_date=init,date=date,group=group,factor=factor,value=1+i*.02,
                      model_version='raw')
                 for init in env['INITS'] for factor in factors for group in env['GROUPS']
                 for i,date in enumerate(pd.date_range('2023-06-14',end))]
        with plt.rc_context(env['STYLE']):
            fig,axes=env['timeseries_figure'](pd.DataFrame(records),factors,window,'raw')
            result=env['figure_qc'](fig,axes)
            assert result['daily_marker_tick_alignment'] is True
            for ax in axes:
                assert len(ax.get_xticks())==expect_ticks
                assert len(ax.lines[0].get_xdata(orig=True))==expect_ticks
            if window=='Total':
                assert result['fig5_all_11_days_labelled'] is True
                assert [x.get_text() for x in axes[-1].get_xticklabels()]==[
                    d.strftime('%m-%d') for d in pd.date_range('2023-06-14','2023-06-24')]
            plt.close(fig)
    df=pd.DataFrame([
        dict(init_date=init,model_version=version,member=m,
             group=env['GROUPS'][m//17],sm_avg=.12+.001*m,
             y_tmax=32+.05*m,partial_fitted_Tmax=31+.07*m)
        for init in env['INITS'] for version in ['raw','residualized'] for m in range(51)])
    with plt.rc_context(env['STYLE']):
        fig,axes=env['scatter_figure'](df,'raw')
        for ax in axes:
            text_boxes=[obj for obj in ax.texts if 'n = 51' in obj.get_text()]
            assert len(text_boxes)==1
            assert text_boxes[0].get_position()[0] >= .95
            assert text_boxes[0].get_ha()=='right'
        plt.close(fig)


def test_era5_ncvi_loader_rejects_ambiguous_missing_days_and_bad_units(tmp_path):
    """Execute the actual Cell3 loader with mocked file and independently audited shape."""
    tree=ast.parse(''.join(nb()['cells'][2]['source']))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)
            and x.name=='load_era5_ncvi_reference')
    import hashlib
    env=dict(pd=pd,np=np,Path=Path,
             finite=lambda x,description: np.asarray(x,dtype=float),
             sha256=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest())
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'era5_ncvi_loader','exec'),env)
    run=env['load_era5_ncvi_reference']
    dates=pd.date_range('2023-06-14','2023-06-24')
    correct=pd.DataFrame(dict(date_bjt=dates,ncvi_anomaly_pvu=np.arange(11)/10,units=['PVU']*11))
    path=tmp_path/'ncvi.csv';correct.to_csv(path,index=False)
    # Real HPC schema: date, month_day, abs, clim, ncvi_theta320_anom_pvu, ...
    actual=correct.rename(columns={'date_bjt':'date','ncvi_anomaly_pvu':'ncvi_theta320_anom_pvu'}).drop(columns=['units'])
    actual['ncvi_theta320_abs_pvu']=1.0
    actual['ncvi_theta320_clim_pvu']=1.0-actual['ncvi_theta320_anom_pvu']
    actual.to_csv(path,index=False)
    real_daily,real_audit=run(path)
    assert real_audit['value_column']=='ncvi_theta320_anom_pvu'
    assert len(real_daily)==11
    correct.to_csv(path,index=False)
    daily,audit=run(path)
    assert len(daily)==11 and audit['status']=='PLOTTED_PROVENANCE_PENDING'
    assert audit['value_column']=='ncvi_anomaly_pvu'
    assert audit['n_target_days']==11
    bad=correct.copy();bad['ncvi_theta320_anomaly_pvu']=bad.ncvi_anomaly_pvu
    bad.to_csv(path,index=False)
    import pytest
    with pytest.raises(ValueError,match='ambiguous'):run(path)
    daily,audit=run(path,value_column='ncvi_anomaly_pvu')
    assert len(daily)==11
    bad=correct.iloc[:-1];bad.to_csv(path,index=False)
    with pytest.raises(ValueError,match='exactly one'):run(path)
    bad=correct.copy();bad.loc[3,'date_bjt']=bad.loc[2,'date_bjt'];bad.to_csv(path,index=False)
    with pytest.raises(ValueError,match='exactly one'):run(path)
    bad=correct.copy();bad['units']='m2/s2';bad.to_csv(path,index=False)
    with pytest.raises(ValueError,match='PVU'):run(path)
    bad=correct.copy();bad.loc[3,'ncvi_anomaly_pvu']='oops';bad.to_csv(path,index=False)
    with pytest.raises((ValueError,TypeError)):run(path)
    # An arbitrary numerical column is NEVER auto-accepted as an NCVI anomaly.
    bad=pd.DataFrame(dict(date_bjt=dates,pv_absolute_pvu=np.arange(11)))
    bad.to_csv(path,index=False)
    with pytest.raises(ValueError,match='column ambiguous/missing'):run(path)


def test_era5_ncvi_is_in_all_physical_and_missing_curve_is_fatal():
    source=''.join(nb()['cells'][2]['source'])
    fig=''.join(nb()['cells'][4]['source'])
    assert 'all_physical=pd.concat([all_physical,era5_ncvi_long]' in source
    assert "hashes['era5_ncvi']" in source
    assert "if factor == 'NCVI_concurrent':" in fig
    assert 'Mandatory ERA5 NCVI reference missing' in fig
    assert "'era5_ncvi_source_qc':era5_ncvi_audit" in ''.join(nb()['cells'][5]['source'])
