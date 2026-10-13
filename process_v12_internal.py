"""Internal diagnostics only; usage: python process_v12_internal.py INPUT_ROOT NEW_OUTPUT_ROOT."""
import pathlib,sys,json,hashlib,pandas as pd,numpy as np
sys.path.insert(0,str(pathlib.Path.cwd()))
from process_v12_adapter import adapt_v12_daily,group_daily
from process_timeseries_core import parameters,residual_relation,partial_fitted_tmax

def main():
    root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
    def source(n):
        found=list(root.rglob(n))
        if len(found)!=1: raise ValueError(f'Require exactly one source for {n}, got {len(found)}')
        return found[0]
    load=lambda n:pd.read_csv(source(n))
    raw=load('table_daily_member_raw_factors_SHF.csv');tcc=load('table_daily_member_tcc_SHF.csv');tmax=load('table_daily_tmax_SHF.csv');stage=load('table_stage_member_raw_factors_TCC.csv')
    long,groups,qc=adapt_v12_daily(raw,tcc,tmax,stage)
    qc.to_csv(out/'daily_to_stage_qc.csv',index=False);groups.to_csv(out/'current_v12_groups.csv',index=False)
    g=group_daily(long);g['status']='INTERNAL_ONLY_NCVI_MISSING';g.to_csv(out/'internal_physical_daily.csv',index=False)
    data={k:load(n) for k,n in {'summary':'table_mlr_model_summary.csv','coefficients':'table_mlr_model_coefficients.csv','design':'table_mlr_design_members.csv','fitted':'table_mlr_fitted_members.csv','residualization':'table_mlr_residualization_audit.csv'}.items()}
    rows=[];scatter=[]
    for init in ['2023-06-08','2023-06-12']:
     for version in ['raw','residualized']:
      model=parameters(data,init,version);relation=residual_relation(data,init)
      p=g[(g.init_date==init)&(g.window=='Stage-I')].pivot(index=['date','group'],columns='factor',values='value').reset_index();p=p[p.date<='2023-06-18']
      y,heat=partial_fitted_tmax(p.sm_avg,p.SHF_Avg,model,relation)
      for i,r in p.iterrows():rows.append(dict(init_date=init,date=r.date,group=r.group,model_version=version,partial_fitted_tmax=y[i],projected_heat=heat[i],intercept=model['intercept'],resid_intercept=relation[0],resid_slope=relation[1],outside_training_window=r.date>pd.Timestamp('2023-06-17'),status='INTERNAL_ONLY_NCVI_MISSING'))
      m=stage[(stage.init_date==init)&(stage.stage=='Stage-I_dry')].sort_values('member'); y,heat=partial_fitted_tmax(m.sm_avg,m.SHF_Avg,model,relation)
      for j,r in enumerate(m.itertuples()):scatter.append(dict(init_date=init,member=r.member,model_version=version,y_tmax=r.y_tmax,sm_avg=r.sm_avg,partial_fitted_tmax=y[j],status='INTERNAL_ONLY_NCVI_MISSING'))
    pd.DataFrame(rows).to_csv(out/'internal_partial_daily.csv',index=False);pd.DataFrame(scatter).to_csv(out/'internal_scatter.csv',index=False)
    pro=[]
    for p in root.rglob('*'):
     if p.suffix in ['.csv','.json','.ipynb']:pro.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    pd.DataFrame(pro).to_csv(out/'source_hashes.csv',index=False)
    ncvi=load('table_ncvi_concurrent_stage_values.csv');man=json.loads(source('comparison_manifest.json').read_text());path=source('table_ncvi_concurrent_stage_values.csv');assert hashlib.sha256(path.read_bytes()).hexdigest()==next(x['sha256'] for x in man['provenance'] if x['name']=='ncvi')
    print(qc.groupby(['init_date','window','factor']).difference.apply(lambda x:abs(x).max()).to_string());print('NCVI flags',ncvi.anomaly_used.unique());print('TCC windows',tcc[['date','window_start_utc','window_end_utc','window_center_bjt']].head(1).to_dict('records'))

if __name__ == "__main__":
    main()
