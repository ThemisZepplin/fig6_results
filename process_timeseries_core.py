"""Independent exported-table process diagnostics. Import performs no I/O or fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

INITS = ('2023-06-08', '2023-06-12')
GROUPS = ('Hot17', 'Middle17', 'Cold17')
VERSIONS = ('residualized', 'raw')
FACTORS5 = ('NCHN_tp', 'z500_anom_NCHN', 'sm_avg', 'NCVI_concurrent', 'TCC_Avg')
FACTORS11 = ('sm_avg', 'z500_anom_NCHN', 'SHF_Avg', 'NCVI_concurrent', 'TCC_Avg')
PREDICTORS = {
    'residualized': ('sm_avg', 'SHF_resid', 'z500_anom_NCHN', 'Initial_MSEstar_max_NCHN', 'TCC_resid', 'NCVI_concurrent'),
    'raw': ('sm_avg', 'SHF_Avg', 'z500_anom_NCHN', 'Initial_MSEstar_max_NCHN', 'TCC_Avg', 'NCVI_concurrent'),
}
OUTPUT_TABLES = ('table_fig5_total_daily_timeseries.csv',
                 'table_fig11_stageI_daily_timeseries.csv',
                 'table_figS11_stageI_scatter_members.csv',
                 'table_fig11_partial_fitted_audit.csv')
UNITS = {'NCHN_tp': 'mm', 'z500_anom_NCHN': 'gpm', 'sm_avg': 'm3 m-3',
         'SHF_Avg': 'W m-2', 'NCVI_concurrent': 'PVU', 'TCC_Avg': 'fraction'}


def probe_sources(paths):
    """Inventory all sources before any calculations, including missing files."""
    rows = []
    for name, value in paths.items():
        p = Path(value).expanduser() if value is not None else None
        record = {'source': name, 'path': str(p), 'exists': bool(p and p.is_file())}
        if record['exists']:
            record['sha256'] = sha256(p)
            if p.suffix == '.csv':
                try:
                    d = pd.read_csv(p)
                    record.update(rows=len(d), columns=json.dumps(list(d.columns)), parse_status='PASS')
                except (ValueError, pd.errors.ParserError) as exc:
                    record.update(parse_status='FAIL', error=str(exc))
        rows.append(record)
    return pd.DataFrame(rows)


def audit_exporter_notebook(path):
    """Read Cell 3 as text only; never execute the exporter or its raw-file code."""
    nb = json.loads(Path(path).read_text())
    if len(nb.get('cells', [])) < 3:
        raise ValueError('Exporter Notebook has no physical Cell 3; supply the actual source')
    names = ('daily_timeseries_all_factors.csv', 'member_stage_values_for_tests.csv', 'group_member_audit.csv')
    code_indices = [i for i, c in enumerate(nb['cells']) if c.get('cell_type') == 'code']
    candidates = {2}
    if len(code_indices) >= 3:
        candidates.add(code_indices[2])
    matches = [i for i in candidates if nb['cells'][i].get('cell_type') == 'code' and
               all(name in ''.join(nb['cells'][i].get('source', [])) for name in names)]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely identify physical/third-code Cell 3 with all three exports; inspect the actual notebook numbering')
    cell_index = matches[0]
    source = ''.join(nb['cells'][cell_index].get('source', []))
    if 'middle17_mean' not in source:
        raise ValueError('Exporter Cell 3 does not document middle17_mean; inspect schema before reuse')
    return {'source': 'exporter_cell3', 'path': str(path), 'sha256': sha256(path),
            'cell_index': cell_index, 'execution': 'none', 'review_text': source}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(df, names, context):
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f'{context}: missing fields {sorted(missing)}')


def finite(values, context):
    a = np.asarray(values, float)
    if not np.isfinite(a).all():
        raise ValueError(f'{context}: missing or nonfinite values; no rows dropped')
    return a


def normalize_window(series):
    return series.replace({'Stage-I_dry': 'Stage-I', 'Stage_I': 'Stage-I'})


def canonical(df, aliases, context):
    """Resolve audited export aliases; ambiguous fields are rejected."""
    df = df.copy()
    for target, choices in aliases.items():
        found = [c for c in choices if c in df]
        if len(found) != 1:
            raise ValueError(f'{context}: need exactly one field for {target}, found {found}')
        df = df.rename(columns={found[0]: target})
    return df


def read_sources(paths):
    """Read explicit paths only; do not search for or generate upstream data."""
    data, provenance = {}, []
    for name, value in paths.items():
        if name == 'member_daily' and (value is None or not Path(value).is_file()):
            data[name] = None
            continue
        if value is None:
            raise FileNotFoundError(f'Configure the required existing source path: {name}')
        p = Path(value).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f'Missing existing export {name}: {p}. Configure paths; no upstream pipeline is run.')
        data[name] = audit_exporter_notebook(p) if name == 'source_notebook' else (json.loads(p.read_text()) if p.suffix == '.json' else pd.read_csv(p))
        record = {'source': name, 'path': str(p), 'sha256': sha256(p), 'exists': True}
        if isinstance(data[name], pd.DataFrame):
            record.update(rows=len(data[name]), columns=json.dumps(list(data[name].columns)))
        if name == 'source_notebook':
            record.update(cell_index=data[name]['cell_index'], notebook_execution='none')
        provenance.append(record)
    manifest = data['model_manifest']
    if manifest.get('computational_status') != 'PASS':
        raise ValueError('Paired MLR computation QC is not PASS')
    for name in ('summary', 'coefficients', 'design', 'residualization', 'fitted'):
        p = Path(paths[name]).resolve()
        if manifest.get('table_sha256', {}).get(p.name) != sha256(p):
            raise ValueError(f'Model source checksum mismatch: {p}')
    return data, provenance


def truth(values, context):
    result = values.astype(str).str.lower().map({'true': True, '1': True, '1.0': True,
                                              'false': False, '0': False, '0.0': False})
    if result.isna().any():
        raise ValueError(f'{context}: need explicit boolean anomaly metadata, not an inferred label')
    return result


def check_units(data):
    """Metadata must come from the actual exporter, not a manually guessed unit."""
    m = data['variable_metadata'].copy()
    require(m, ['factor', 'units', 'is_anomaly', 'evidence_source'], 'exporter variable metadata')
    m['factor'] = m.factor.replace({'NCVI_conc': 'NCVI_concurrent'})
    for factor, expected in UNITS.items():
        row = m[m.factor == factor]
        if not len(row) or not (row.units == expected).all() or row.evidence_source.isna().any() or not row.evidence_source.astype(str).str.strip().ne('').all():
            raise ValueError(f'{factor}: unverified/incompatible units; require {expected} and an exporter evidence source (no silent conversion)')
    nc_metadata = m[m.factor == 'NCVI_concurrent'].copy()
    require(nc_metadata, ['init_date', 'window', 'date'], 'actual per-case NCVI anomaly metadata')
    nc_metadata['window'] = normalize_window(nc_metadata.window)
    nc_metadata['date'] = pd.to_datetime(nc_metadata.date)
    for init in INITS:
        for window, end in [('Total','2023-06-24'), ('Stage-I','2023-06-18')]:
            expected_dates = pd.date_range('2023-06-14', end)
            q = nc_metadata[(nc_metadata.init_date == init) & (nc_metadata.window == window) & nc_metadata.date.isin(expected_dates)].sort_values('date')
            if len(q) != len(expected_dates) or not np.array_equal(q.date.to_numpy(), expected_dates.to_numpy()):
                raise ValueError(f'{init}/{window}: need actual NCVI anomaly/fallback evidence for every target date, not a global assumed mode')
            if not truth(q.is_anomaly, 'NCVI metadata').all():
                raise ValueError('NCVI absolute fallback cannot be labelled or plotted as anomaly')
    # If an export itself carries a fallback flag, it overrides no sidecar declaration.
    for name in ('daily', 'member_daily', 'member_stage'):
        d = data.get(name)
        if d is None or 'factor' not in d:
            continue
        nc = d[d.factor.isin(('NCVI_conc', 'NCVI_concurrent'))]
        init_col = next((c for c in ('init_date','init') if c in nc), None)
        if init_col:
            nc = nc[nc[init_col].isin(INITS)]
        window_col = next((c for c in ('grouping_stage','window','stage') if c in nc), None)
        if window_col:
            nc = nc[normalize_window(nc[window_col]).isin(('Stage-I','Total'))]
        if 'date' in nc:
            dates = pd.to_datetime(nc.date)
            keep = dates.between('2023-06-14','2023-06-24')
            if window_col:
                keep &= ~((normalize_window(nc[window_col]) == 'Stage-I') & (dates > pd.Timestamp('2023-06-18')))
            nc = nc[keep]
        for column in ('ncvi_is_anomaly', 'anomaly_used', 'is_anomaly'):
            if column in nc and len(nc) and not truth(nc[column], name+' NCVI').all():
                raise ValueError(f'{name}: NCVI absolute fallback detected')
        if 'units' in d:
            for factor, rows in d.groupby('factor'):
                canonical_factor = 'NCVI_concurrent' if factor == 'NCVI_conc' else factor
                if canonical_factor in UNITS and not (rows.units == UNITS[canonical_factor]).all():
                    raise ValueError(f'{name}/{factor}: unit metadata disagrees with source metadata')
    m['status'] = 'PASS'
    return m


def current_group_audit(data):
    """Compare legacy exports with v12 absolute Tmax rankings, never alter definitions."""
    current = canonical(data['stage_input'], {'init_date': ('init_date', 'init'),
                         'window': ('window', 'stage'), 'member': ('member',)}, 'v12 stage input')
    current['window'] = normalize_window(current.window)
    require(current, ['y_tmax', 'sm_avg', 'SHF_Avg'], 'v12 stage input')
    legacy = prepare_groups(data['groups'])
    old = canonical(data['member_stage'], {'init_date': ('init_date', 'init'),
                     'window': ('window', 'stage', 'grouping_stage'), 'member': ('member',),
                     'y_tmax': ('y_tmax', 'member_stage_mean_Tmax', 'stage_mean_Tmax')}, 'legacy member-stage export')
    old['window'] = normalize_window(old.window)
    groups, checks = [], []
    for init in INITS:
        for window in ('Stage-I', 'Total'):
            s = current[(current.init_date == init) & (current.window == window)].sort_values('member').copy()
            if len(s) != 51 or not np.array_equal(s.member, np.arange(51)):
                raise ValueError(f'v12 {init}/{window}: incomplete or duplicate member IDs')
            finite(s.y_tmax, 'v12 ranking Tmax')
            # Same descending absolute-Tmax definition as the old grouping routine.
            ranked = s.sort_values('y_tmax', ascending=False).copy()
            yy = ranked.y_tmax.to_numpy()
            if yy[16] == yy[17] or yy[33] == yy[34]:
                raise ValueError('Tmax tie across a 17-member boundary: require an explicit original tie audit; do not change group definition')
            ranked['rank'] = np.arange(1, 52)
            ranked['group'] = np.where(ranked['rank'] <= 17, 'Hot17', np.where(ranked['rank'] >= 35, 'Cold17', 'Middle17'))
            o = old[(old.init_date == init) & (old.window == window)].copy()
            if o.duplicated('member').any() or len(o) != 51:
                raise ValueError(f'{init}/{window}: legacy member-stage adapter needs one absolute Tmax per member; audit the actual Cell 3 schema')
            finite(o.y_tmax, 'legacy member Tmax')
            o = o.sort_values('y_tmax', ascending=False)
            old_rank = dict(zip(o.member, np.arange(1, 52)))
            old_y = dict(zip(o.member, o.y_tmax))
            legacy_case = legacy[(legacy.init_date == init) & (legacy.window == window)].set_index('member')
            old_group = legacy_case['group']
            for row in ranked.itertuples():
                same = row.member in old_y and abs(old_y[row.member]-row.y_tmax) <= 1e-6 and old_rank[row.member] == row.rank and old_group.loc[row.member] == row.group
                for column in ('rank_by_stage_mean_Tmax','rank'):
                    if column in legacy_case:
                        same = same and legacy_case.loc[row.member, column] == row.rank
                for column in ('member_stage_mean_Tmax','y_tmax'):
                    if column in legacy_case:
                        same = same and np.isclose(legacy_case.loc[row.member,column], row.y_tmax, atol=1e-6, rtol=0)
                checks.append({'init_date': init, 'window': window, 'member': row.member,
                               'current_y_tmax': row.y_tmax, 'legacy_y_tmax': old_y.get(row.member),
                               'current_rank': row.rank, 'legacy_rank': old_rank.get(row.member),
                               'current_group': row.group, 'legacy_group': old_group.loc[row.member],
                               'legacy_match': bool(same)})
            groups.append(ranked[['init_date', 'window', 'member', 'group']])
            fit = data['fitted']
            fit = fit[(fit.init_date == init) & (fit.window == window)]
            for version in VERSIONS:
                f = fit[fit.version == version].sort_values('member')
                if len(f) != 51 or not np.array_equal(f.member, s.member) or not np.allclose(f.y_tmax, s.y_tmax, atol=1e-8, rtol=0):
                    raise ValueError(f'{init}/{window}/{version}: saved fit target differs from v12 input')
    audit = pd.DataFrame(checks)
    audit['action'] = np.where(audit.groupby(['init_date','window']).legacy_match.transform('all'), 'reuse_verified_legacy', 'reaggregate_member_daily')
    return pd.concat(groups, ignore_index=True), audit, current


def reaggregate_member_daily(data, groups, init, window):
    """Recover group means from existing per-member daily CSVs only, never GRIB."""
    member = data.get('member_daily')
    if member is None:
        raise ValueError(f'{init}/{window}: legacy grouping is incompatible; provide a verified member_daily CSV, no GRIB fallback')
    d = canonical(member, {'init_date': ('init_date', 'init'), 'date': ('date',),
                          'member': ('member',), 'factor': ('factor',), 'value': ('value',)}, 'member daily')
    require(d, ['units'], 'member daily')
    d['date'] = pd.to_datetime(d.date)
    if d.date.dt.tz is not None or not (d.date == d.date.dt.normalize()).all():
        raise ValueError('Member daily source must use BJT calendar dates')
    d['factor'] = d.factor.replace({'NCVI_conc': 'NCVI_concurrent'})
    dates = pd.date_range('2023-06-14', '2023-06-24' if window == 'Total' else '2023-06-18')
    factors = FACTORS5 if window == 'Total' else FACTORS11
    g = groups[(groups.init_date == init) & (groups.window == window)]
    rows = []
    for factor in factors:
        for date in dates:
            s = d[(d.init_date == init) & (d.factor == factor) & (d.date == date)]
            if len(s) != 51 or not np.array_equal(np.sort(s.member), np.arange(51)):
                raise ValueError(f'{init}/{window}/{date}/{factor}: require 51 complete member-daily values')
            finite(s.value, 'member-daily aggregation')
            joined = s.merge(g[['member','group']], on='member', validate='one_to_one')
            for group in GROUPS:
                mean_value = float(joined[joined.group == group].value.mean())
                finite(mean_value, 'reaggregated member-daily group mean')
                rows.append({'init_date': init, 'window': window, 'date': date, 'group': group,
                             'factor': factor, 'value': mean_value,
                             'available': True, 'grouping_source': 'member-daily export reaggregated using verified v12 absolute-Tmax groups'})
    return pd.DataFrame(rows)


def prepare_groups(df):
    g = canonical(df, {'init_date': ('init_date', 'init', 'forecast_init'),
                       'window': ('window', 'stage', 'stage_name'),
                       'member': ('member', 'number', 'ensemble_member'),
                       'group': ('group', 'hotcold_group', 'member_group', 'tmax_group')}, 'member groups')
    g['window'] = normalize_window(g.window)
    g = g[g.init_date.astype(str).isin(INITS) & g.window.isin(('Stage-I', 'Total'))].copy()
    for init in INITS:
        for w in ('Stage-I', 'Total'):
            s = g[(g.init_date == init) & (g.window == w)]
            members = finite(s.member, f'{init}/{w} members')
            if len(s) != 51 or not np.array_equal(np.sort(members), np.arange(51)):
                raise ValueError(f'{init}/{w}: require member IDs 0..50 exactly once')
            if s.groupby('group').size().to_dict() != {k: 17 for k in GROUPS}:
                raise ValueError(f'{init}/{w}: require Hot17/Middle17/Cold17 each n=17')
    return g


def prepare_daily(df):
    """Adapt v11 grouped long table, preserving stage-specific grouping provenance."""
    # grouping_stage is the audited grouping-window field, not a per-day stage label.
    if 'grouping_stage' in df:
        df = df.drop(columns=['stage', 'window'], errors='ignore')
    d = canonical(df, {'init_date': ('init_date', 'init'),
                       'window': ('window', 'grouping_stage', 'stage'),
                       'date': ('date', 'valid_date', 'time'), 'factor': ('factor',)}, 'daily series')
    require(d, ['era5', 'hot17_mean', 'middle17_mean', 'cold17_mean'], 'daily series')
    d['window'] = normalize_window(d.window)
    d['factor'] = d.factor.replace({'NCVI_conc': 'NCVI_concurrent'})
    d['date'] = pd.to_datetime(d.date, errors='raise')
    if d.date.dt.tz is not None or not (d.date == d.date.dt.normalize()).all():
        raise ValueError('Daily dates must already represent BJT calendar dates, without time/timezone')
    rows = []
    for init in INITS:
        for w, end, factors in [('Total', '2023-06-24', FACTORS5), ('Stage-I', '2023-06-18', FACTORS11)]:
            dates = pd.date_range('2023-06-14', end)
            for factor in factors:
                s = d[(d.init_date == init) & (d.window == w) & (d.factor == factor) & d.date.isin(dates)].sort_values('date')
                if len(s) != len(dates) or not np.array_equal(s.date.to_numpy(), dates.to_numpy()):
                    raise ValueError(f'{init}/{w}/{factor}: duplicated/missing daily dates; grouping window must match')
                finite(s[['hot17_mean', 'middle17_mean', 'cold17_mean']], f'{init}/{w}/{factor} group means')
                reference = s.era5.to_numpy(float)
                if not (np.isnan(reference).all() or np.isfinite(reference).all()):
                    raise ValueError(f'{init}/{w}/{factor}: ERA5 must be complete or explicitly entirely unavailable')
                for group, column in [('ERA5', 'era5'), ('Hot17', 'hot17_mean'), ('Middle17', 'middle17_mean'), ('Cold17', 'cold17_mean')]:
                    for date, value in zip(s.date, s[column]):
                        rows.append({'init_date': init, 'window': w, 'date': date,
                                     'group': group, 'factor': factor, 'value': value,
                                     'available': bool(np.isfinite(float(value))),
                                     'grouping_source': 'exported daily table, matching window; not re-ranked'})
    return pd.DataFrame(rows)


def parameters(data, init, version):
    """Load a complete saved Stage-I model and validate its 51-member design."""
    summary, coef, design = data['summary'], data['coefficients'], data['design']
    require(summary, ['init_date', 'window', 'version', 'n', 'design_rank', 'predictor_list', 'standardized_intercept'], 'model summary')
    require(coef, ['init_date', 'window', 'version', 'predictor', 'position', 'standardized_coef', 'mean', 'scale_ddof0'], 'coefficients')
    require(design, ['init_date', 'window', 'version', 'member', 'predictor', 'value', 'y_tmax'], 'training design')
    select = lambda d: d[(d.init_date == init) & (d.window == 'Stage-I') & (d.version == version)].copy()
    s = select(summary)
    c = select(coef)
    x = select(design)
    if len(s) != 1 or s.iloc[0]['n'] != 51 or s.iloc[0].design_rank != 7 or s.iloc[0].predictor_list != ';'.join(PREDICTORS[version]):
        raise ValueError(f'{init}/{version}: invalid complete final Stage-I model')
    c = c[c.predictor != 'const'].sort_values('position')
    if c.predictor.tolist() != list(PREDICTORS[version]) or c.position.tolist() != list(range(1, 7)):
        raise ValueError(f'{init}/{version}: coefficients are not the final six predictors')
    finite(c[['standardized_coef', 'mean', 'scale_ddof0']], 'saved scaler/coefficients')
    if (c.scale_ddof0 <= 0).any():
        raise ValueError('Saved scaler must have positive scales')
    if len(x) != 306 or x.duplicated(['member', 'predictor']).any():
        raise ValueError(f'{init}/{version}: incomplete training design')
    for row in c.itertuples():
        q = x[x.predictor == row.predictor].sort_values('member')
        if not np.array_equal(q.member.to_numpy(), np.arange(51)):
            raise ValueError('Training design member coverage mismatch')
        a = finite(q.value, 'training predictors')
        if not np.isclose(a.mean(), row.mean, rtol=0, atol=1e-8) or not np.isclose(a.std(ddof=0), row.scale_ddof0, rtol=0, atol=1e-8):
            raise ValueError('Training design does not reproduce saved scaler')
    pivot = x.pivot(index='member', columns='predictor', values='value').loc[:, list(PREDICTORS[version])]
    y = x.pivot(index='member', columns='predictor', values='y_tmax')
    finite(y, 'training y_tmax')
    if not (y.to_numpy() == y.iloc[:, 0].to_numpy()[:, None]).all():
        raise ValueError('Training response differs by predictor position')
    intercept = float(s.iloc[0].standardized_intercept)
    finite(intercept, 'standardized intercept')
    fitted = select(data['fitted']).sort_values('member')
    require(fitted, ['member','y_tmax','y_fitted','residual'], 'saved fitted members')
    if len(fitted) != 51 or not np.array_equal(fitted.member, np.arange(51)):
        raise ValueError('Saved fitted members incomplete')
    reconstructed = intercept + ((pivot.to_numpy() - c['mean'].to_numpy()) /
                                  c.scale_ddof0.to_numpy()) @ c.standardized_coef.to_numpy()
    finite(fitted[['y_tmax','y_fitted','residual']], 'saved fitted members')
    if not np.allclose(reconstructed, fitted.y_fitted, atol=1e-6, rtol=0) or not np.allclose(y.iloc[:,0], fitted.y_tmax, atol=1e-8, rtol=0):
        raise ValueError('Full-model replay differs from saved fit/Stage-I training response; coefficients are not replaced')
    if not np.allclose(fitted.y_tmax-fitted.y_fitted, fitted.residual, atol=1e-8, rtol=0):
        raise ValueError('Saved fitted residual identity failed')
    mean_state_prediction = float(intercept + ((pivot.mean().to_numpy()-c['mean'].to_numpy()) / c.scale_ddof0.to_numpy()) @ c.standardized_coef.to_numpy())
    if not np.isclose(mean_state_prediction, intercept, atol=1e-8, rtol=0) or not np.isclose(mean_state_prediction, fitted.y_fitted.mean(), atol=1e-8, rtol=0):
        raise ValueError('Diagnostic baseline does not equal full-model prediction at training mean state')
    return {'intercept': intercept, 'coef': c.set_index('predictor'),
            'design': pivot, 'y': y.iloc[:, 0].sort_index(), 'version': version,
            'mean_state_prediction': mean_state_prediction,
            'full_fit_max_abs_difference': float(np.max(np.abs(reconstructed-fitted.y_fitted.to_numpy())))}


def residual_relation(data, init):
    a = data['residualization']
    require(a, ['init_date', 'window', 'target', 'source', 'control', 'intercept', 'slope'], 'residualization audit')
    a = a[(a.init_date == init) & (a.window == 'Stage-I') & (a.target == 'SHF_resid')]
    if len(a) != 1 or a.iloc[0].source != 'SHF_Avg' or a.iloc[0].control != 'sm_avg':
        raise ValueError(f'{init}: require saved Stage-I SHF_Avg ~ intercept + sm_avg residual relation')
    return tuple(finite(a[['intercept', 'slope']], 'fixed residual relation')[0])


def partial_fitted_tmax(sm_value, shf_value, model, relation):
    """Actual diagnostic entry: alpha_std + beta_SM*z_SM + beta_SHF*z_SHF.

    All other six-factor full-model predictors remain at their own training means.
    Saved stage-member residual parameters are used without any daily fitting.
    """
    smv, shfv = np.asarray(sm_value, float), np.asarray(shf_value, float)
    factor = 'SHF_resid' if model['version'] == 'residualized' else 'SHF_Avg'
    projected = shfv - (relation[0] + relation[1] * smv) if factor == 'SHF_resid' else shfv
    c = model['coef']
    value = (model['intercept'] + c.loc['sm_avg', 'standardized_coef'] *
             (smv - c.loc['sm_avg', 'mean']) / c.loc['sm_avg', 'scale_ddof0'] +
             c.loc[factor, 'standardized_coef'] *
             (projected - c.loc[factor, 'mean']) / c.loc[factor, 'scale_ddof0'])
    return value, projected


def build_tables(data, provenance):
    unit_qc = check_units(data)
    groups, group_qc, current = current_group_audit(data)
    old_daily = prepare_daily(data['daily'])
    pieces = []
    for init in INITS:
        for window in ('Stage-I','Total'):
            q = group_qc[(group_qc.init_date == init) & (group_qc.window == window)]
            old = old_daily[(old_daily.init_date == init) & (old_daily.window == window)]
            if q.legacy_match.all():
                if data.get('member_daily') is not None:
                    rebuilt = reaggregate_member_daily(data, groups, init, window)
                    comparison = old[old.group != 'ERA5'].merge(
                        rebuilt[['date','group','factor','value']], on=['date','group','factor'],
                        validate='one_to_one', suffixes=('_old','_member'))
                    difference = float(np.max(np.abs(comparison.value_old-comparison.value_member)))
                    group_qc.loc[q.index, 'daily_mean_max_abs_difference'] = difference
                    if difference > 1e-6:
                        pieces.extend([rebuilt, old[old.group == 'ERA5']])
                        group_qc.loc[q.index, 'action'] = 'reaggregate_numeric_mismatch'
                    else:
                        pieces.append(old)
                    group_qc.loc[q.index, 'member_daily_reconstruction_checked'] = True
                else:
                    pieces.append(old)
                    group_qc.loc[q.index, 'member_daily_reconstruction_checked'] = False
            else:
                new = reaggregate_member_daily(data, groups, init, window)
                # ERA5 observations are independent of member group reassignments.
                pieces.extend([new, old[old.group == 'ERA5']])
                group_qc.loc[q.index, 'member_daily_reconstruction_checked'] = True
    daily = pd.concat(pieces, ignore_index=True)
    source_json = json.dumps(provenance, sort_keys=True)
    total = daily[daily.window == 'Total'].copy()
    total['parameter_source'] = 'not model dependent'
    stage = daily[daily.window == 'Stage-I'].copy()
    stage_tables, scatter_rows, audit_rows = [], [], []
    for init in INITS:
        models = {v: parameters(data, init, v) for v in VERSIONS}
        raw = models['raw']['design']
        resid = models['residualized']['design']
        a, b = residual_relation(data, init)
        if not np.array_equal(models['raw']['y'], models['residualized']['y']):
            raise ValueError('Raw/residualized training responses differ')
        if not np.allclose(raw.sm_avg, resid.sm_avg, rtol=0, atol=1e-10):
            raise ValueError('Raw/residualized SM training samples differ')
        original = current[(current.init_date == init) & (current.window == 'Stage-I')].sort_values('member')
        for factor in ('sm_avg','SHF_Avg'):
            if not np.allclose(original[factor], raw[factor], atol=1e-8, rtol=0):
                raise ValueError(f'v12 {factor} does not match saved Stage-I model design')
        expected = raw.SHF_Avg - (a + b * raw.sm_avg)
        if not np.allclose(expected, resid.SHF_resid, rtol=0, atol=1e-8):
            raise ValueError('Fixed residual relation does not reproduce saved Stage-I SHF_resid')
        if abs(expected.mean()) > 1e-8 or abs(np.dot(raw.sm_avg - raw.sm_avg.mean(), expected)) > 1e-6:
            raise ValueError('Saved residual relation fails intercept/control orthogonality')
        g = groups[(groups.init_date == init) & (groups.window == 'Stage-I')].set_index('member')
        base = stage[stage.init_date == init].copy()
        pair = base[base.factor.isin(('sm_avg', 'SHF_Avg'))].pivot(index=['date', 'group'], columns='factor', values='value').sort_index()
        # A daily export from a different grouping/variable definition must not be projected.
        for group in GROUPS:
            members = g[g.group == group].index
            dates = pair.loc[(slice(pd.Timestamp('2023-06-14'), pd.Timestamp('2023-06-17')), group), :]
            for factor in ('sm_avg', 'SHF_Avg'):
                if not np.isclose(dates[factor].mean(), raw.loc[members, factor].mean(), rtol=0, atol=1e-6):
                    raise ValueError(f'{init}/{group}/{factor}: daily Stage-I mean differs from model-member group mean; check provenance/units/grouping')
        for version, model in models.items():
            d = base.copy()
            d['model_version'] = version
            d['parameter_source'] = source_json
            values, projected = partial_fitted_tmax(pair.sm_avg, pair.SHF_Avg, model, (a, b))
            finite(values[pair.index.get_level_values('group') != 'ERA5'], 'daily group partial fitted diagnostic')
            p = pair.reset_index()[['date', 'group']]
            p['value'] = values
            p['projected_SHF'] = projected
            p['init_date'], p['window'], p['model_version'] = init, 'Stage-I', version
            p['factor'] = 'partial_fitted_Tmax'
            p['available'] = np.isfinite(values)
            p['residual_intercept'], p['residual_slope'] = a, b
            p['parameter_source'] = source_json
            p['reference_role'] = np.where(p.group == 'ERA5', 'model diagnostic projected from ERA5 SM/SHF, not observed Tmax', 'member-group diagnostic')
            p['outside_training_window'] = p.date > pd.Timestamp('2023-06-17')
            stage_tables += [d, p]
            values, projected = partial_fitted_tmax(raw.sm_avg, raw.SHF_Avg, model, (a, b))
            finite(values, 'member partial fitted diagnostic')
            for member, value, shf in zip(raw.index, values, projected):
                scatter_rows.append({'init_date': init, 'window': 'Stage-I', 'member': int(member),
                                     'group': g.loc[member, 'group'], 'model_version': version,
                                     'y_tmax': float(model['y'].loc[member]), 'sm_avg': float(raw.loc[member, 'sm_avg']),
                                     'SHF_Avg': float(raw.loc[member, 'SHF_Avg']), 'projected_SHF': float(shf),
                                     'partial_fitted_Tmax': float(value), 'parameter_source': source_json,
                                     'residual_intercept': a, 'residual_slope': b})
            heat = 'SHF_resid' if version == 'residualized' else 'SHF_Avg'
            for predictor, row in model['coef'].iterrows():
                audit_rows.append({'init_date': init, 'window': 'Stage-I', 'model_version': version,
                                   'predictor': predictor, 'included_in_partial': predictor in ('sm_avg', heat),
                                   'standardized_intercept': model['intercept'],
                                   'standardized_coef': row.standardized_coef, 'training_mean': row['mean'],
                                   'training_scale_ddof0': row.scale_ddof0,
                                   'held_value': row['mean'] if predictor not in ('sm_avg', heat) else np.nan,
                                   'residual_intercept': a, 'residual_slope': b, 'parameter_source': source_json,
                                   'formula': 'alpha_std + beta_SM*(SM-mu_SM)/sigma_SM + beta_heat*(heat-mu_heat)/sigma_heat',
                                   'projection_rule': ('fixed stage SHF relation; no daily residual regression' if version == 'residualized' else 'raw SHF direct; no residual projection'),
                                   'model_train_start_bjt': '2023-06-14', 'model_train_end_bjt': '2023-06-17',
                                   'display_end_bjt': '2023-06-18', 'n_training_members': 51})
                audit_rows[-1].update(baseline_prediction_at_training_means=model['mean_state_prediction'],
                                      baseline_difference=model['mean_state_prediction']-model['intercept'],
                                      full_fit_max_abs_difference=model['full_fit_max_abs_difference'])
    fig11 = pd.concat(stage_tables, ignore_index=True)
    scatter = pd.DataFrame(scatter_rows)
    # Explicit equality gates: raw daily covariates and Tmax/SM scatter never vary by version.
    for init in INITS:
        d = fig11[(fig11.init_date == init) & (fig11.factor != 'partial_fitted_Tmax')]
        r = d[d.model_version == 'raw'].sort_values(['factor', 'group', 'date'])
        s = d[d.model_version == 'residualized'].sort_values(['factor', 'group', 'date'])
        if not np.array_equal(r.value.to_numpy(), s.value.to_numpy(), equal_nan=True):
            raise ValueError('Daily physical series changed between model versions')
        r = scatter[(scatter.init_date == init) & (scatter.model_version == 'raw')].sort_values('member')
        s = scatter[(scatter.init_date == init) & (scatter.model_version == 'residualized')].sort_values('member')
        if not np.array_equal(r[['y_tmax', 'sm_avg']].to_numpy(), s[['y_tmax', 'sm_avg']].to_numpy()):
            raise ValueError('Tmax/raw-SM scatter changed between model versions')
    total['source_provenance'] = source_json
    result = dict(zip(OUTPUT_TABLES, (total, fig11, scatter, pd.DataFrame(audit_rows))))
    result.update({'table_process_group_consistency_qc.csv': group_qc,
                   'table_process_variable_units_qc.csv': unit_qc,
                   'table_process_source_qc.csv': pd.DataFrame(provenance)})
    return result


def output_guard(output_root, input_paths):
    """Require an independent empty run directory; reruns use a new root."""
    out = Path(output_root).expanduser().resolve()
    for path in input_paths.values():
        if path is None:
            continue
        p = Path(path).expanduser().resolve()
        if (out == p.parent or out in p.parents or p == out or
                out == p.parent.parent or p.parent.parent in out.parents):
            raise ValueError('Output root must be separate from source directories')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output run directory is not empty: {out}; choose a new run root to preserve outputs')
    return out


def save_tables(tables, output_root, input_paths, provenance):
    out = output_guard(output_root, input_paths)
    for d in ('tables', 'figures', 'qc'):
        (out / d).mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / 'tables' / name, index=False)
    manifest = {'workflow': 'process_figures_v2', 'provenance': provenance,
                'table_sha256': {name: sha256(out / 'tables' / name) for name in tables},
                'qc_status': 'PASS', 'interpretation': 'parameterization sensitivity, not land-surface causal-strength change'}
    (out / 'qc/process_manifest.json').write_text(json.dumps(manifest, indent=2))
    return out
