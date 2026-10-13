"""Audited v12 physical-source adapter; never fills missing daily NCVI."""
import numpy as np
import pandas as pd
from process_timeseries_core import INITS, GROUPS, normalize_window, finite

FACTORS = ('y_tmax', 'NCHN_tp', 'sm_avg', 'SHF_Avg', 'z500_anom_NCHN', 'TCC_Avg')
KEY = ['init_date', 'date', 'member']


def adapt_v12_daily(raw, tcc, tmax, stage):
    """Return physical long data, current groups and daily-to-stage numeric audit.

    Dates retain source labels: TCC UTC interval metadata is preserved and
    explicitly audited separately, rather than silently shifted to BJT days.
    """
    frames = []
    dates = pd.date_range('2023-06-14', '2023-06-24')
    expected = pd.MultiIndex.from_product([INITS, dates, range(51)], names=KEY)
    for name, source in [('raw', raw), ('tcc', tcc), ('tmax', tmax)]:
        d = source[source.init_date.isin(INITS)].copy()
        d['date'] = pd.to_datetime(d.date, errors='raise')
        if d.duplicated(KEY).any() or not pd.MultiIndex.from_frame(d[KEY]).sort_values().equals(expected.sort_values()):
            raise ValueError(f'{name}: require exactly 2 inits x 11 dates x 51 integer members')
        frames.append(d)
    d, c, t = frames
    d = d.merge(c[KEY + ['TCC_Avg']], on=KEY, validate='one_to_one')
    check = d.merge(t[KEY + ['s2s_tmax']], on=KEY, validate='one_to_one')
    if not np.array_equal(check.y_tmax.to_numpy(), check.s2s_tmax.to_numpy()):
        raise ValueError('Daily Tmax tables differ')
    finite(d[list(FACTORS)], 'v12 physical factors')
    s = stage.copy(); s['window'] = normalize_window(s.stage)
    groups, audit, pieces = [], [], []
    for init in INITS:
        for window, end in [('Stage-I', '2023-06-17'), ('Total', '2023-06-24')]:
            target = s[(s.init_date == init) & (s.window == window)].set_index('member').sort_index()
            if len(target) != 51 or not np.array_equal(target.index, np.arange(51)):
                raise ValueError('Stage input member coverage invalid')
            ranking = target.y_tmax.sort_values(ascending=False)
            if ranking.iloc[16] == ranking.iloc[17] or ranking.iloc[33] == ranking.iloc[34]:
                raise ValueError('Boundary tie requires explicit scientific ranking rule')
            g = pd.DataFrame({'member': ranking.index, 'rank': range(1,52)})
            g['group'] = [GROUPS[i//17] for i in range(51)]
            g['init_date'], g['window'] = init, window
            groups.append(g)
            q = d[(d.init_date == init) & (d.date <= end)]
            for factor in FACTORS:
                rebuilt = q.groupby('member')[factor].sum() if factor == 'NCHN_tp' else q.groupby('member')[factor].mean()
                for member in range(51):
                    delta = float(rebuilt.loc[member] - target.loc[member, factor])
                    audit.append(dict(init_date=init, window=window, member=member, factor=factor,
                                      daily_aggregate=rebuilt.loc[member], stage_value=target.loc[member,factor],
                                      difference=delta, aggregation='sum' if factor=='NCHN_tp' else 'mean',
                                      passed=abs(delta)<=1e-8))
            display = d[d.init_date == init].copy()
            display = display.merge(g[['member','group']], on='member', validate='many_to_one')
            display['window'] = window
            long = display.melt(id_vars=KEY+['window','group'], value_vars=list(FACTORS), var_name='factor', value_name='value')
            pieces.append(long)
    qc = pd.DataFrame(audit)
    if not qc.passed.all():
        raise ValueError('v12 daily-to-stage consistency FAILED; do not export')
    return pd.concat(pieces, ignore_index=True), pd.concat(groups, ignore_index=True), qc


def group_daily(long):
    """Group means use current v12 membership, never legacy grouped curves."""
    keys = ['init_date','window','date','group','factor']
    counts = long.groupby(keys).size()
    if not (counts == 17).all():
        raise ValueError('Each physical group requires exactly 17 members')
    return long.groupby(keys, as_index=False).value.mean()
