"""Strict daily NCVI export and saved-MLR replay. No I/O on import."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

INITS = ('2023-06-08', '2023-06-12')
DATES = pd.date_range('2023-06-14', '2023-06-24')
ATOL = 1e-8


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def daily_replay(init, rt, rt_steps, hc, hc_steps, reference):
    """Inputs are regional PVU: realtime[step,51], climatology[step,sample].

    Stage scalars use equal effective-time/sample weights, exactly as preserved
    MLR; daily anomaly remains unadjusted. Both equal-day and time-weighted
    replays are retained. Variable timestep counts cannot be hidden by rescaling.
    """
    if init not in INITS:
        raise ValueError('Unsupported init')
    arrays = [np.asarray(rt,float), np.asarray(hc,float)]
    if arrays[0].ndim != 2 or arrays[0].shape[1] != 51 or arrays[1].ndim != 2:
        raise ValueError('Require realtime 51 members and explicit hindcast samples')
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('Missing PV values are forbidden')
    rt,hc = arrays
    times=[]
    for steps, values in [(rt_steps,rt),(hc_steps,hc)]:
        steps=np.asarray(steps)
        if not np.issubdtype(steps.dtype,np.timedelta64):
            raise ValueError('Lead coordinates must be explicit timedeltas')
        t=pd.DatetimeIndex(pd.Timestamp(init)+pd.to_timedelta(steps)+pd.Timedelta(hours=8))
        if len(t)!=len(values) or t.has_duplicates or not t.is_monotonic_increasing:
            raise ValueError('Duplicated/unsorted or inconsistent valid times')
        times.append(t)
    rtime,htime=times
    rows=[];clims=[]
    for date in DATES:
        rmask=rtime.normalize()==date;hmask=htime.normalize()==date
        if not rmask.any() or not hmask.any():
            raise ValueError(f'{init}/{date}: missing realtime/reforecast daily coverage')
        clim=float(hc[hmask].mean());real=rt[rmask].mean(axis=0)
        clims.append(dict(init_date=init,date=str(date.date()),daily_climatology_pvu=clim,
                          realtime_time_count=int(rmask.sum()),reforecast_time_count=int(hmask.sum()),
                          reforecast_sample_count=hc.shape[1],reforecast_value_count=int(hmask.sum()*hc.shape[1])))
        for member in range(51):
            rows.append(dict(init_date=init,date=str(date.date()),member=member,realtime_pv_pvu=real[member],
                             daily_climatology_pvu=clim,daily_ncvi_anomaly_pvu=real[member]-clim,
                             units='PVU',anomaly_used=True,source_qc_reference=f'{init}:source_audit',
                             realtime_time_count=int(rmask.sum()),reforecast_time_count=int(hmask.sum())))
    daily=pd.DataFrame(rows);qc=[]
    for window,end in [('Stage-I','2023-06-17'),('Total','2023-06-24')]:
        rm=(rtime.normalize()>=DATES[0])&(rtime.normalize()<=pd.Timestamp(end))
        hm=(htime.normalize()>=DATES[0])&(htime.normalize()<=pd.Timestamp(end))
        scalar=float(hc[hm].mean());replay=rt[rm].mean(axis=0)-scalar
        q=daily[daily.date<=end];equal=q.groupby('member').daily_ncvi_anomaly_pvu.mean()
        weighted=q.assign(term=q.daily_ncvi_anomaly_pvu*q.realtime_time_count).groupby('member').agg(term=('term','sum'),count=('realtime_time_count','sum'))
        targets=reference[(reference.init_date==init)&(reference.window==window)].set_index('member')
        if len(targets)!=51 or set(targets.index)!=set(range(51)):
            raise ValueError('Incomplete stage reference members')
        for member in range(51):
            target=float(targets.loc[member,'NCVI_concurrent']);w=float(weighted.loc[member,'term']/weighted.loc[member,'count'])
            qc.append(dict(init_date=init,window=window,member=member,model_stage_ncvi=target,
                           mlr_scalar_replay=replay[member],stage_climatology_scalar=scalar,
                           daily_equal_day_aggregate=equal.loc[member],daily_time_weighted_aggregate=w,
                           scalar_difference=replay[member]-target,daily_difference=equal.loc[member]-target,
                           weighted_difference=w-target,scalar_pass=abs(replay[member]-target)<=ATOL,
                           daily_pass=abs(equal.loc[member]-target)<=ATOL and abs(w-target)<=ATOL,
                           realtime_times=int(rm.sum()),reforecast_times=int(hm.sum()),
                           reforecast_samples=hc.shape[1],difference_analysis='Compare scalar replay, daily equal/time weights and per-day sample/time counts; no correction applied'))
    return daily,pd.DataFrame(clims),pd.DataFrame(qc)


def load_pair(spec, init, hindcast=False):
    """Open explicitly pinned GRIB files only; validate level/units/member identity.

    Reforecast selector mirrors MLR: first exact init-date match, otherwise all
    records. The retained forecast-reference list must exactly match config.
    Nonmatching month/day or hour rejects the MLR fallback rather than using it.
    """
    import xarray as xr
    parts=[];audit=[]
    for role in ['cf','pf']:
        item=spec[role];path=Path(item['path'])
        if digest(path)!=item['sha256']:
            raise ValueError('GRIB source hash differs from pinned config')
        with xr.open_dataset(path,engine='cfgrib',backend_kwargs={'indexpath':'','filter_by_keys':{'shortName':'pv','typeOfLevel':'theta','level':320}}) as ds:
            if 'pv' not in ds:
                raise ValueError('Actual variable must be pv')
            a=ds.pv
            if a.attrs.get('GRIB_typeOfLevel')!='theta':
                raise ValueError('Require proven isentropic theta level')
            level=next((c for c in ['theta','potentialTemperature'] if c in a.coords),None)
            if level is None or np.size(a[level])!=1 or float(np.asarray(a[level]).ravel()[0])!=320:
                raise ValueError('Require exact 320 K; no nearest-layer substitution')
            if level in a.dims:a=a.squeeze(level,drop=True)
            units=str(a.attrs.get('units',''));normalized=units.lower().replace(' ','').replace('**','^')
            if normalized not in ['km^2kg^-1s^-1','k*m^2*kg^-1*s^-1','pvu']:
                raise ValueError(f'Unknown PV units: {units}; no magnitude heuristic')
            factor=1. if normalized=='pvu' else 1e6
            if 'time' not in a.coords:raise ValueError('Missing forecast-reference time')
            refs=pd.DatetimeIndex(np.atleast_1d(a.time.values))
            matches=np.where(refs.normalize()==pd.Timestamp(init))[0]
            if not hindcast and len(matches)!=1:raise ValueError('Realtime init not unique')
            if matches.size:
                chosen=refs[[matches[0]]]
                if 'time' in a.dims:a=a.isel(time=int(matches[0]),drop=True)
            else:
                if not hindcast:raise ValueError('Missing realtime init')
                chosen=refs
            if [str(x) for x in chosen]!=item['selected_reference_times']:
                raise ValueError('Selected reference-time sample set differs from explicit config')
            if any((t.month,t.day,t.hour,t.minute)!=(pd.Timestamp(init).month,pd.Timestamp(init).day,0,0) for t in chosen):
                raise ValueError('Reforecast init-day/hour mismatch')
            if role=='cf':
                if 'number' in a.dims:
                    if a.sizes['number']!=1:raise ValueError('CF must have one member')
                    a=a.assign_coords(number=[0])
                else:a=a.expand_dims(number=[0])
            elif 'number' not in a.dims or not np.array_equal(a.number.values,np.arange(1,51)):
                raise ValueError('PF IDs must be 1..50 in order; no renumbering')
            lat=a.latitude;lon=a.longitude
            a=a.sel(latitude=lat[(lat>=35)&(lat<=50)],longitude=lon[(lon>=115)&(lon<=130)])
            if a.sizes['latitude']==0 or a.sizes['longitude']==0 or not np.isfinite(a.values).all():
                raise ValueError('Empty region/missing grid values; no silent skip')
            attrs=dict(a.attrs)
            regional=(a.mean(['latitude','longitude'],skipna=False)*factor).load()
            parts.append(regional)
            audit.append(dict(init_date=init,role=('reforecast_' if hindcast else 'realtime_')+role,
                              path=str(path),sha256=item['sha256'],variable='pv',level=320,units=units,
                              pvu_multiplier=factor,reference_times=json.dumps([str(t) for t in chosen]),
                              sample_coordinates=json.dumps({d:[str(x) for x in regional[d].values] for d in regional.dims if d not in ['step','number']}),
                              attributes=json.dumps(attrs,default=str),spatial_mean='arithmetic grid-cell mean',
                              legacy_spatial_mean='cos(latitude) weighted',region='35-50N 115-130E'))
    merged=xr.concat(parts,dim='number',join='exact',compat='equals').sortby('number')
    if not np.array_equal(merged.number.values,np.arange(51)):
        raise ValueError('CF0 + PF1..50 required')
    if hindcast:
        extra=[d for d in merged.dims if d!='step']
        merged=merged.stack(sample=extra).transpose('step','sample')
    else:
        if set(merged.dims)!={'step','number'}:raise ValueError('Unexpected realtime dimensions')
        merged=merged.transpose('step','number')
    return merged.values,merged.step.values,audit


def era5_reference_qc(config):
    """Missing ERA5 blocks formal acceptance, not independent S2S replay."""
    try:
        csv=Path(config['era5_csv']); provenance=Path(config['era5_provenance'])
        era=pd.read_csv(csv); era=era[pd.to_datetime(era.date).isin(DATES)]
        ep=json.loads(provenance.read_text())
        if len(era)!=11 or era.date.duplicated().any() or not np.isfinite(era.ncvi_theta320_anom_pvu).all() or ep['csv_sha256']!=digest(csv) or ep['units']!='PVU' or ep['level_K']!=320 or not ep['climatology_sources'] or not ep['baseline_period']:
            raise ValueError('ERA5 coverage/climatology provenance invalid')
        return {'status':'PASS','reason':'Coverage and supplied provenance verified','fields':list(era.columns)}
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        return {'status':'FAIL','reason':str(exc)}


def export(config):
    """Always save failed replay evidence; formal consumer requires all gates PASS."""
    out=Path(config['output_root'])
    if out.exists():raise ValueError('Output root must not exist; history is protected')
    paths={k:Path(config[k]) for k in ['stage_table','design_table','comparison_manifest']}
    era_qc=era5_reference_qc(config)
    for key in ['era5_csv','era5_provenance']:
        if config.get(key) and Path(config[key]).is_file(): paths[key]=Path(config[key])
    manifest=json.loads(paths['comparison_manifest'].read_text())
    if digest(paths['stage_table'])!=next(p['sha256'] for p in manifest['provenance'] if p['name']=='ncvi') or digest(paths['design_table'])!=manifest['table_sha256']['table_mlr_design_members.csv']:
        raise ValueError('Saved-model provenance hash mismatch')
    ref=pd.read_csv(paths['stage_table']);ref['window']=ref.stage.replace({'Stage-I_dry':'Stage-I'})
    ref=ref[ref.init_date.isin(INITS)&ref.window.isin(['Stage-I','Total'])]
    if len(ref)!=204 or not ref.anomaly_used.eq(True).all() or ref.fallback_reason.fillna('').ne('').any():
        raise ValueError('Require 204 anomaly stage references without fallback')
    design=pd.read_csv(paths['design_table']);q=design[(design.predictor=='NCVI_concurrent')&design.init_date.isin(INITS)&design.window.isin(['Stage-I','Total'])].merge(ref,on=['init_date','window','member'],validate='many_to_one')
    if len(q)!=408 or q.version.nunique()!=2 or not np.allclose(q.value,q.NCVI_concurrent,rtol=0,atol=ATOL):
        raise ValueError('MLR design differs from concurrent stage table')
    daily=[];clim=[];qc=[];audit=[]
    for init in INITS:
        spec=config['sources'][init]
        for role, suffix in [('pf','MSE/ecmf_pf_60_2023-06.grib'),('cf','cf_2023-6/ecmf_cf_60_2023-06.grib')]:
            if not str(spec['realtime'][role]['path']).endswith(suffix):
                raise ValueError('Realtime source differs from preserved MLR file convention')
        expected=ref[ref.init_date==init].climatology_file_used.unique()
        pinned=f"pf={spec['reforecast']['pf']['path']}; cf={spec['reforecast']['cf']['path']}"
        if len(expected)!=1 or expected[0]!=pinned:raise ValueError('Actual reforecast files must match stored MLR source')
        rt,rs,ra=load_pair(spec['realtime'],init)
        hc,hs,ha=load_pair(spec['reforecast'],init,True)
        d,c,q=daily_replay(init,rt,rs,hc,hs,ref);daily.append(d);clim.append(c);qc.append(q);audit+=ra+ha
    d,c,q=pd.concat(daily),pd.concat(clim),pd.concat(qc)
    if len(d)!=1122 or d.duplicated(['init_date','date','member']).any() or len(q)!=204:raise ValueError('Export coverage invalid')
    s2s_pass=bool(q.scalar_pass.all() and q.daily_pass.all())
    passed=s2s_pass and era_qc['status']=='PASS'
    d['qc_status']='PASS' if passed else ('FAIL_ERA5_REFERENCE' if s2s_pass else 'FAIL_STAGE_EQUIVALENCE')
    out.mkdir(parents=True,exist_ok=False)
    tables={'table_ncvi_concurrent_daily_members.csv':d,'table_ncvi_daily_climatology.csv':c,'table_ncvi_daily_source_audit.csv':pd.DataFrame(audit),'table_ncvi_daily_stage_replay_qc.csv':q}
    for name,table in tables.items():table.to_csv(out/name,index=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    record={'status':'PASS' if passed else ('FAIL_ERA5_REFERENCE' if s2s_pass else 'FAIL_STAGE_EQUIVALENCE'),
            's2s_daily_source_qc':{'status':'PASS','reason':'Strict GRIB source checks passed'},
            's2s_stage_replay_qc':{'status':'PASS' if s2s_pass else 'FAIL','reason':'Scalar, equal-day and time-weighted comparisons at fixed 1e-8 PVU'},
            'era5_reference_qc':era_qc,'formal_process_allowed':passed,'atol':ATOL,'code_commit':commit,
            'code_sha256':digest(__file__),'config':config,'inputs':{k:{'path':str(p),'sha256':digest(p)} for k,p in paths.items()},
            'outputs':{n:digest(out/n) for n in tables},'max_scalar_difference':float(q.scalar_difference.abs().max()),
            'max_daily_difference':float(q.daily_difference.abs().max()),'max_weighted_difference':float(q.weighted_difference.abs().max()),
            'recommendation':'If mismatch: retain saved MLR; examine effective-time counts/sample selection first. Uniform daily climatology may be equivalent only with matching day weights. Any change to stage baseline or MLR requires a separate scientific revision; never adjust daily anomalies.'}
    (out/'ncvi_daily_export_manifest.json').write_text(json.dumps(record,indent=2))
    return record
