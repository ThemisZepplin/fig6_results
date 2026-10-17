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
    env=dict(TCC_SOURCE_UNITS={'S2S':'auto','ERA5':'auto'},BAND_COLUMNS=('value_lower','value_upper'),np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,INITS=('2023-06-08','2023-06-12'),GROUPS=('Hot17','Middle17','Cold17'))
    nodes=[n for n in ast.parse(''.join(nb()['cells'][4]['source'])).body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (isinstance(n,ast.FunctionDef) or any(isinstance(t,ast.Name) and t.id in ['COLORS','LABELS','STYLE','MARKERS','LINESTYLES'] for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'layout','exec'),env)
    for window,factors,end,count in [('Total',('NCHN_tp','z500_anom_NCHN','sm_avg','NCVI_concurrent','TCC_Avg'),'2023-06-24',10),('Stage-I',('partial_fitted_Tmax','sm_avg','z500_anom_NCHN','SHF_Avg','NCVI_concurrent','TCC_Avg'),'2023-06-18',12)]:
        records=[dict(init_date=init,date=date,group=g,factor=f,value=0.5+i*.02,model_version='raw') for init in env['INITS'] for f in factors for g in env['GROUPS'] for i,date in enumerate(pd.date_range('2023-06-14',end))]
        records += [dict(init_date=init,date=date,group='ERA5',factor='NCVI_concurrent',value=.01*i,model_version='raw') for init in env['INITS'] for i,date in enumerate(pd.date_range('2023-06-14',end))]
        with plt.rc_context(env['STYLE']):
            fig,axes=env['timeseries_figure'](pd.DataFrame(records),factors,window,'raw')
            assert len(axes)==count and fig.get_figwidth()==6.5
            assert fig.get_figheight()*25.4<=228
            assert env['figure_qc'](fig,axes)['text_extent_qc']=='PASS'
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            legend_box=fig.legends[0].get_window_extent(renderer)
            heading_boxes=[t.get_window_extent(renderer) for t in fig.texts if 'June 2023' in t.get_text()]
            assert len(heading_boxes)==2
            assert legend_box.y1 < min(box.y0 for box in heading_boxes)
            assert legend_box.y0 > max(ax.get_window_extent(renderer).y1 for ax in axes)
            if window=='Stage-I':assert all(len(a.patches)==1 for a in axes)
            plt.close(fig)


def test_scatter_layout_and_explicit_plot_tables():
    whole=''.join(''.join(c['source']) for c in nb()['cells'])
    for name in ['fig5_total_timeseries_plot.csv','fig11_stageI_timeseries_','figS11_stageI_scatter_','group_membership_audit.csv','consistency_manifest.json']:
        assert name in whole
    assert 'fig.suptitle' not in whole
    env=dict(TCC_SOURCE_UNITS={'S2S':'auto','ERA5':'auto'},BAND_COLUMNS=('value_lower','value_upper'),np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,INITS=('2023-06-05','2023-06-08'),GROUPS=('Hot17','Middle17','Cold17'),SCATTER_FACTORS_BY_INIT={'2023-06-05':('z500_anom_NCHN','NCVI_concurrent'),'2023-06-08':('sm_avg','partial_fitted_Tmax')})
    nodes=[n for n in ast.parse(''.join(nb()['cells'][4]['source'])).body if isinstance(n,(ast.FunctionDef,ast.Assign)) and (isinstance(n,ast.FunctionDef) or any(isinstance(t,ast.Name) and t.id in ['COLORS','LABELS','STYLE','MARKERS','LINESTYLES'] for t in n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'scatter','exec'),env)
    records=[dict(init_date=init,member=m,group=env['GROUPS'][m//17],model_version=v,sm_avg=.15+m*.001,y_tmax=30+m*.08,partial_fitted_Tmax=31+m*.04+(.1 if v=='raw' else 0),z500_anom_NCHN=-20+m*2,NCVI_concurrent=-.3+m*.01) for init in env['INITS'] for v in ['raw','residualized'] for m in range(51)]
    with plt.rc_context(env['STYLE']):
        fig,axes=env['scatter_figure'](pd.DataFrame(records),'raw')
        assert len(axes)==4 and axes[0].get_xlabel()=='Z500 anomaly (gpm)'
        assert axes[1].get_xlabel().startswith('SM (')
        assert axes[2].get_xlabel()=='NCVI anomaly (PVU)'
        assert axes[3].get_xlabel().startswith('SM–SHF')
        assert env['figure_qc'](fig,axes)['text_extent_qc']=='PASS'

        # The prior y=.975 placed headings 1.13 PDF pt above page top.
        # Keep both initialization headings inside the rendered PDF canvas.
        headings=[t for t in fig.texts if 'June 2023' in t.get_text()]
        assert len(headings)==2
        assert all(abs(t.get_position()[1]-.960)<1e-10 for t in headings)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        assert all(t.get_window_extent(renderer).y1 <= fig.bbox.height for t in headings)
        legend_box=fig.legends[0].get_window_extent(renderer)
        assert legend_box.y1 < min(ax.get_window_extent(renderer).y0 for ax in axes)
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
    branch=source[source.index('stage_members=pd.read_csv(STAGE)'):source.index('# Validate saved member scalars')]
    stage=pd.DataFrame([dict(init_date=init,stage='Stage-I',member=m,y_tmax=30+m*.1,sm_avg=.15+m*.001,SHF_Avg=50+m,z500_anom_NCHN=20+m) for init in ['2023-06-08','2023-06-12'] for m in range(51)])
    path=tmp_path/'stage.csv';stage.to_csv(path,index=False)
    groups=stage[['init_date','member']].copy();groups['window']='Stage-I';groups['group']=['Hot17' if m<17 else 'Middle17' if m<34 else 'Cold17' for m in groups.member]
    calls=[]
    def parameters(data,init,version):
        calls.append((init,version))
        return {'y':stage[stage.init_date==init].set_index('member').y_tmax.copy(),'version':version,'design':pd.DataFrame({'NCVI_concurrent':np.linspace(-.1,.1,51)},index=np.arange(51))}
    env=dict(np=np,pd=pd,STAGE=path,STAGEI_INITS=('2023-06-08','2023-06-12'),VERSIONS=('residualized',),
             normalize_window=lambda s:s,parameters=parameters,data={},current_groups=groups,
             residual_relation=lambda data,init:(0.,1.),
             partial_fitted_tmax=lambda sm,shf,model,relation:(np.asarray(sm)+np.asarray(shf)+(1 if model['version']=='raw' else 0),np.asarray(shf)))
    exec(compile(branch,'actual-cell4-member-branch','exec'),env)
    assert len(calls)==2 and len(env['scatter_table'])==102
    assert set(env['scatter_table'].model_version)=={'residualized'}
    assert np.allclose(env['scatter_table'].NCVI_concurrent.iloc[:51],np.linspace(-.1,.1,51))
    assert 'atol=1e-8' in branch
    import pytest
    def mismatched_parameters(data,init,version):
        result=parameters(data,init,version);result['y']=result['y']+.001;return result
    env['parameters']=mismatched_parameters
    with pytest.raises(ValueError,match='target/model mismatch'):
        exec(compile(branch,'actual-cell4-member-branch','exec'),env)


def test_daily_markers_land_exactly_on_labelled_ticks_and_scatter_stats_use_right_side():
    """Exercise true plotting functions, not only their static strings."""
    env=dict(TCC_SOURCE_UNITS={'S2S':'auto','ERA5':'auto'},BAND_COLUMNS=('value_lower','value_upper'),np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,
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
        records += [dict(init_date=init,date=date,group='ERA5',factor='NCVI_concurrent',value=.01*i,model_version='raw') for init in env['INITS'] for i,date in enumerate(pd.date_range('2023-06-14',end))]
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
    env['SCATTER_FACTORS_BY_INIT']={'2023-06-08':('sm_avg','partial_fitted_Tmax'),'2023-06-12':('sm_avg','partial_fitted_Tmax')}
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
            if ax.get_xlabel().startswith('SM ('):
                assert text_boxes[0].get_position()[0] >= .95
                assert text_boxes[0].get_ha()=='right'
            else:
                assert text_boxes[0].get_position()[0] <= .05
                assert text_boxes[0].get_ha()=='left'
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
    assert "if factor=='NCVI_concurrent':" in fig
    assert 'Mandatory ERA5 NCVI reference missing' in fig
    assert "'era5_ncvi_source_qc':era5_ncvi_audit" in ''.join(nb()['cells'][5]['source'])


def test_tcc_fixed_physical_ylim_and_no_extra_date_axis_title():
    env=dict(TCC_SOURCE_UNITS={'S2S':'auto','ERA5':'auto'},BAND_COLUMNS=('value_lower','value_upper'),np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,
             INITS=('2023-06-08','2023-06-12'),GROUPS=('Hot17','Middle17','Cold17'))
    nodes=[x for x in ast.parse(''.join(nb()['cells'][4]['source'])).body
           if isinstance(x,(ast.FunctionDef,ast.Assign)) and
           (isinstance(x,ast.FunctionDef) or
            any(isinstance(t,ast.Name) and t.id in
                ('COLORS','LABELS','STYLE','MARKERS','LINESTYLES') for t in x.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'tcc','exec'),env)
    factors=('NCHN_tp','z500_anom_NCHN','sm_avg','NCVI_concurrent','TCC_Avg')
    records=[dict(init_date=init,date=date,group=group,factor=factor,
                  value=(0.4+i*.02 if factor=='TCC_Avg' else i*.02+1.0))
             for init in env['INITS'] for factor in factors
             for group in env['GROUPS']
             for i,date in enumerate(pd.date_range('2023-06-14','2023-06-24'))]
    # Required independent ERA5 NCVI line, with identical dates for both inits.
    records += [dict(init_date=init,date=date,group='ERA5',factor='NCVI_concurrent',value=.01*i)
                for init in env['INITS']
                for i,date in enumerate(pd.date_range('2023-06-14','2023-06-24'))]
    df=pd.DataFrame(records)
    with plt.rc_context(env['STYLE']):
        fig,axes=env['timeseries_figure'](df,factors,'Total')
        assert axes[8].get_ylim()==(0.0,100.0)
        assert axes[9].get_ylim()==(0.0,100.0)
        assert np.array_equal(axes[8].get_yticks(),np.array([0,25,50,75,100]))
        assert axes[8].get_ylabel()=='Total cloud cover (%)'
        assert all(a.get_xlabel()=='' for a in axes)
        assert env['figure_qc'](fig,axes)['text_extent_qc']=='PASS'
        plt.close(fig)
    bad=df.copy()
    bad.loc[(bad.factor=='TCC_Avg')&(bad.group=='Hot17'),'value']=101
    with plt.rc_context(env['STYLE']):
        import pytest
        with pytest.raises(ValueError,match=r'outside \[0,100\]'):
            env['timeseries_figure'](bad,factors,'Total')


def plotting_env():
    env=dict(np=np,pd=pd,plt=plt,mdates=mdates,Line2D=Line2D,MaxNLocator=MaxNLocator,
             INITS=('2023-06-08','2023-06-12'),GROUPS=('Hot17','Middle17','Cold17'),
             TCC_SOURCE_UNITS={'S2S':'auto','ERA5':'auto'},BAND_COLUMNS=('value_lower','value_upper'),
             SCATTER_FACTORS_BY_INIT={'2023-06-05':('z500_anom_NCHN','NCVI_concurrent'),'2023-06-08':('sm_avg','partial_fitted_Tmax')})
    nodes=[node for node in ast.parse(''.join(nb()['cells'][4]['source'])).body
           if isinstance(node,ast.FunctionDef) or isinstance(node,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id in ('COLORS','LABELS','STYLE','MARKERS','LINESTYLES') for t in node.targets)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'plot','exec'),env)
    return env


def test_percentage_sources_are_not_multiplied_twice_and_csv_is_unchanged():
    env=plotting_env()
    source=pd.DataFrame(dict(factor=['TCC_Avg']*4,group=['Hot17','Cold17','ERA5','ERA5'],
                             value=[.2,.8,20.,80.],value_lower=[.1,.7,10.,70.],value_upper=[.3,.9,30.,90.]))
    before=source.copy(deep=True)
    result,audit=env['display_units'](source)
    pd.testing.assert_frame_equal(source,before)
    assert result.value.tolist()==[20.,80.,20.,80.]
    assert result.value_lower.tolist()==[10.,70.,10.,70.]
    assert {r['source']:r['multiplier'] for r in audit}=={'S2S':100.,'ERA5':1.}
    env['TCC_SOURCE_UNITS']['ERA5']='percent'
    tiny=source.iloc[2:].copy();tiny['value']=[.2,.8]
    result,_=env['display_units'](tiny)
    assert result.value.tolist()==[.2,.8]


def test_requested_limits_expand_to_include_all_curves_and_uncertainty_bands():
    env=plotting_env()
    frame=pd.DataFrame(dict(init_date=['2023-06-08']*2,group=['ERA5']*2,value=[0.,80.],
                            value_lower=[-75.,70.],value_upper=[5.,125.]))
    bounds,ticks,info=env['row_limits'](frame,'z500_anom_NCHN','Total')
    assert bounds==(-75.,125.) and info['expanded']
    assert np.allclose(np.diff(ticks),25)
    for factor,expected,step in [('NCHN_tp',(-1.,10.),1),('NCVI_concurrent',(-.3,.25),.10)]:
        f=frame.drop(columns=['value_lower','value_upper']).copy();f['value']=[0.,.1]
        bounds,ticks,info=env['row_limits'](f,factor,'Total')
        assert bounds==expected and not info['expanded']
        assert np.allclose(np.diff(ticks),step)
        if factor=='NCHN_tp':assert np.min(ticks)==0
    import pytest
    invalid=frame.copy();invalid.loc[0,'value_lower']=1.
    with pytest.raises(ValueError,match='do not contain'):env['row_limits'](invalid,'z500_anom_NCHN','Total')

    stage=pd.DataFrame(dict(init_date=['2023-06-05']*2,group=['Cold17','Hot17'],value=[27.91,31.92]))
    bounds,ticks,info=env['row_limits'](stage,'partial_fitted_Tmax','Stage-I',init='2023-06-05')
    assert bounds==(27.,32.) and info['expanded'] and np.allclose(np.diff(ticks),1.)
    stage['init_date']='2023-06-08';stage['value']=[32.28,34.61]
    bounds,ticks,info=env['row_limits'](stage,'partial_fitted_Tmax','Stage-I',init='2023-06-08')
    assert bounds==(32.,35.) and not info['expanded'] and np.allclose(np.diff(ticks),.5)
    stage['init_date']='2023-06-05';stage['value']=[-5.1,83.9]
    bounds,ticks,info=env['row_limits'](stage,'z500_anom_NCHN','Stage-I',init='2023-06-05')
    assert bounds==(-25.,125.) and info['expanded'] and np.allclose(np.diff(ticks),25.)


def test_selected_mechanism_scatter_uses_asymmetric_factors_and_window_label(tmp_path):
    env=plotting_env();pair=('2023-06-05','2023-06-08')
    df=pd.DataFrame([dict(init_date=init,member=m,group=env['GROUPS'][m//17],model_version='residualized',
                         y_tmax=30+m*.07+np.sin(m)*.2,z500_anom_NCHN=-30+m*2,
                         NCVI_concurrent=-.15+m*.005,sm_avg=.1+m*.002,partial_fitted_Tmax=31+m*.04)
                     for init in pair for m in range(51)])
    with plt.rc_context(env['STYLE']):
        fig,axes=env['scatter_figure'](df,'residualized',inits=pair,factors_by_init=env['SCATTER_FACTORS_BY_INIT'])
        assert axes[0].get_xlabel()=='Z500 anomaly (gpm)'
        assert axes[1].get_xlabel()=='SM (m³ m⁻³)'
        assert axes[2].get_xlabel()=='NCVI anomaly (PVU)'
        assert axes[3].get_xlabel()=='SM–SHF diagnostic (°C)'
        assert all(a.get_ylabel()=='6.14–17 Tmax (°C)' for a in axes)
        assert [t.get_text() for t in fig.texts]==['05 June 2023','08 June 2023']
        boxes=[next(t for t in a.texts if 'n = 51' in t.get_text()) for a in axes]
        assert [t.get_ha() for t in boxes]==['right','right','right','left']
        assert env['figure_qc'](fig,axes)['text_extent_qc']=='PASS'
        path=tmp_path/'circulation.pdf';fig.savefig(path)
        import fitz
        with fitz.open(path) as doc:assert env['pdf_page_qc'](doc[0],fig.get_figheight())['status']=='PASS'
        plt.close(fig)


def test_config_separates_total_and_selected_process_pair_and_frozen_math():
    source=''.join(nb()['cells'][0]['source'])
    assert "TOTAL_INITS=('2023-06-08','2023-06-12')" in source
    assert "FIG11_INITS=('2023-06-05','2023-06-08')" in source
    assert "PROCESS_PAIRS=(FIG11_INITS,)" in source
    assert "VERSIONS=('residualized',)" in source
    # These numerical definitions must remain independent of the plotting repair.
    tree=ast.parse(''.join(nb()['cells'][1]['source']))
    names={'parameters','residual_relation','partial_fitted_tmax'}
    assert names <= {node.name for node in tree.body if isinstance(node,ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'fit','lstsq','polyfit'}


def test_early_ncvi_export_changes_configuration_only():
    baseline=json.loads((ROOT/'ncvi_daily_export_standalone.ipynb').read_text())
    early=json.loads((ROOT/'ncvi_daily_export_early_0601_0605.ipynb').read_text())
    assert len(early['cells'])==6
    assert early['cells'][1:]==baseline['cells'][1:]
    config=''.join(early['cells'][0]['source'])
    assert "INITS = ('2023-06-01', '2023-06-05')" in config
    assert 'ncvi_daily_standalone/run_early_0601_0605' in config
    for cell in early['cells']:
        ast.parse(''.join(cell['source']))
        assert cell['execution_count'] is None and not cell['outputs']


def test_multiple_ncvi_exports_require_exact_coverage_and_preserve_historical_fail(tmp_path):
    import hashlib,pytest
    source=''.join(nb()['cells'][2]['source'])
    branch=source[source.index('def snapshot_ncvi_runs():'):source.index('pieces=[]')]
    runs=[]
    for k,pair in enumerate([('2023-06-08','2023-06-12'),('2023-06-01','2023-06-05')]):
        run=tmp_path/str(k);run.mkdir();runs.append(run)
        daily=pd.DataFrame([dict(init_date=init,date=date,member=m,units='PVU',anomaly_used=True,daily_ncvi_anomaly_pvu=.1)
                            for init in pair for date in pd.date_range('2023-06-14','2023-06-24') for m in range(51)])
        replay=pd.DataFrame([dict(init_date=init,window=w,member=m,daily_equal_day_aggregate=.1,
                                 daily_time_weighted_aggregate=.1,mlr_scalar_replay=.1)
                             for init in pair for w in ('Stage-I','Total') for m in range(51)])
        daily.to_csv(run/'table_ncvi_concurrent_daily_members.csv',index=False)
        replay.to_csv(run/'table_ncvi_daily_stage_replay_qc.csv',index=False)
        manifest=dict(status='FAIL_STAGE_EQUIVALENCE',s2s_daily_source_qc={'status':'PASS'},
                      s2s_stage_replay_qc={'status':'FAIL'},
                      outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()})
        (run/'ncvi_daily_export_manifest.json').write_text(json.dumps(manifest))
    env=dict(pd=pd,np=np,Path=Path,json=json,NCVI_RUNS=tuple(runs),NCVI_RUN=runs[0],
             INITS=('2023-06-08','2023-06-12','2023-06-01','2023-06-05'),KEY=['init_date','date','member'],
             finite=lambda x,msg:np.asarray(x,float),sha256=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
    before={str(p):p.read_bytes() for run in runs for p in run.iterdir()}
    exec(compile(branch,'actual-multi-export-loader','exec'),env)
    assert len(env['ncvi'])==2244 and len(env['rq'])==408
    assert env['nm']['status']=='FAIL_STAGE_EQUIVALENCE'
    assert len(env['ncvi_manifests'])==2
    assert before=={str(p):p.read_bytes() for run in runs for p in run.iterdir()}
    env['NCVI_RUNS']=(runs[0],)
    with pytest.raises(ValueError,match='coverage'):exec(compile(branch,'missing-early','exec'),env)
    env['NCVI_RUNS']=(runs[0],runs[1],runs[1])
    with pytest.raises(ValueError,match='coverage'):exec(compile(branch,'duplicate-early','exec'),env)
