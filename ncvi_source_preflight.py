"""Read-only HPC input inspection; writes reports/config only in fresh directory."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import numpy as np
import pandas as pd
from ncvi_daily_exporter import INITS, digest, load_pair, era5_reference_qc


def inspect_grib(path, init, hindcast):
    import xarray as xr
    path=Path(path).resolve(strict=True)
    with xr.open_dataset(path,engine='cfgrib',backend_kwargs={'indexpath':'','filter_by_keys':{'shortName':'pv','typeOfLevel':'theta','level':320}}) as ds:
        a=ds['pv']; refs=pd.DatetimeIndex(np.atleast_1d(a.time.values))
        matches=np.where(refs.normalize()==pd.Timestamp(init))[0]
        selected=refs[[matches[0]]] if matches.size else refs
        if not hindcast and len(matches)!=1:raise ValueError('Realtime init must match uniquely')
        record={'path':str(path),'sha256':digest(path),'selected_reference_times':[str(x) for x in selected]}
        audit=dict(record,shortName=a.attrs.get('GRIB_shortName'),typeOfLevel=a.attrs.get('GRIB_typeOfLevel'),
                   units=a.attrs.get('units'),level_coordinates={k:np.asarray(a[k]).tolist() for k in ['theta','potentialTemperature'] if k in a.coords},
                   forecast_reference_times=[str(x) for x in refs],selected_years=sorted(set(selected.year.tolist())),
                   dimensions=dict(a.sizes),coordinates={k:[str(v) for v in np.atleast_1d(a[k].values)] for k in ['step','number','time'] if k in a.coords})
        if audit['shortName']!='pv' or audit['typeOfLevel']!='theta':raise ValueError('Uncertain GRIB variable/level')
        return record,audit


def build(args):
    out=Path(args.report_dir)
    if out.exists():raise ValueError('Report directory must be new')
    out.mkdir(parents=True)
    report={'status':'FAIL','errors':[],'grib':[],'s2s_daily_source_qc':{'status':'NOT_RUN'},'s2s_stage_replay_qc':{'status':'NOT_RUN'},'formal_process_allowed':False}
    config={'output_root':args.export_root,'stage_table':str(Path(args.stage_table).resolve()),'design_table':str(Path(args.design_table).resolve()),
            'comparison_manifest':str(Path(args.comparison_manifest).resolve()),'sources':{}}
    if args.era5_csv:config['era5_csv']=str(Path(args.era5_csv).resolve())
    if args.era5_provenance:config['era5_provenance']=str(Path(args.era5_provenance).resolve())
    report['era5_reference_qc']=era5_reference_qc(config)
    try:
        manifest=json.loads(Path(args.comparison_manifest).read_text());ref=pd.read_csv(args.stage_table)
        if digest(args.stage_table)!=next(p['sha256'] for p in manifest['provenance'] if p['name']=='ncvi') or digest(args.design_table)!=manifest['table_sha256']['table_mlr_design_members.csv']:
            raise ValueError('Model-table hashes do not match manifest')
        report['input_hashes']={p:digest(p) for p in [args.stage_table,args.design_table,args.comparison_manifest]}
        for init in INITS:
            rows=ref[(ref.init_date==init)&ref.stage.isin(['Stage-I_dry','Stage-I','Total'])]
            if len(rows)!=102 or not rows.anomaly_used.eq(True).all() or rows.fallback_reason.fillna('').ne('').any():raise ValueError('Stage source is incomplete/non-anomaly')
            names=rows.climatology_file_used.unique()
            if len(names)!=1:raise ValueError('Ambiguous stored reforecast paths')
            match=re.fullmatch(r'pf=(.+); cf=(.+)',names[0])
            if not match:raise ValueError('Cannot parse actual MLR climatology paths')
            paths={'realtime':{'pf':args.realtime_pf,'cf':args.realtime_cf},'reforecast':{'pf':match[1],'cf':match[2]}}
            config['sources'][init]={}
            for kind,pair in paths.items():
                config['sources'][init][kind]={}
                for role,path in pair.items():
                    item,audit=inspect_grib(path,init,kind=='reforecast')
                    # Retain exact stored pathname for exporter source equality; hash canonical target.
                    item['path']=str(Path(path).absolute())
                    config['sources'][init][kind][role]=item
                    report['grib'].append(dict(audit,init_date=init,kind=kind,role=role,mlr_climatology_file_used=names[0]))
                _,_,validation=load_pair(config['sources'][init][kind],init,kind=='reforecast')
                report.setdefault('validated_sources',[]).extend(validation)
        report['s2s_daily_source_qc']={'status':'PASS','reason':'Actual selected source metadata and finite samples verified'}
        report['status']='PASS' if report['era5_reference_qc']['status']=='PASS' else 'S2S_READY_ERA5_BLOCKED'
        # Config allows standalone S2S audit even if ERA5 is missing. Unknown S2S fields never emitted.
        (out/'ncvi_daily_hpc_config.json').write_text(json.dumps(config,indent=2))
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        report['errors'].append(str(exc))
    report['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (out/'ncvi_source_preflight_report.json').write_text(json.dumps(report,indent=2,default=str))
    if report['errors']:raise RuntimeError('; '.join(report['errors']))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['stage-table','design-table','comparison-manifest','realtime-pf','realtime-cf','report-dir','export-root']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--era5-csv');parser.add_argument('--era5-provenance')
    print(json.dumps(build(parser.parse_args()),indent=2,default=str))
