"""Paired final slim MLR from exported CSVs; importing performs no analysis/I/O.

Scientific definitions trace to PR #10, 7d94fb1, Cells 8/9. No upstream reads.
"""
from pathlib import Path
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

SOURCE_COMMIT = "7d94fb1d8ca630eafb65cfa203dbaa7890c1580a"
INITS = ("2023-06-01", "2023-06-05", "2023-06-08", "2023-06-12")
WINDOWS = {"Stage-I": ("2023-06-14", "2023-06-17"),
           "Total": ("2023-06-14", "2023-06-24")}
STAGES = {"Stage-I": "Stage-I_dry", "Total": "Total"}
VERSIONS = ("residualized", "raw")
METHODS = ("LMG", "Johnson", "LOFO")
# The only final-model configuration; raw lists are mapped position by position.
RESIDUALIZED = {
    "Stage-I": ("sm_avg", "SHF_resid", "z500_anom_NCHN",
                "Initial_MSEstar_max_NCHN", "TCC_resid", "NCVI_concurrent"),
    "Total": ("NCHN_tp", "z500_anom_NCHN", "NCVI_concurrent",
              "Initial_MSEstar_max_NCHN", "TCC_resid", "SM_resid"),
}
COUNTERPARTS = {"SHF_resid": "SHF_Avg", "TCC_resid": "TCC_Avg",
                "SM_resid": "sm_avg"}
RESIDUALIZATION = {"Stage-I": (("SHF_resid", "SHF_Avg", "sm_avg"),
                               ("TCC_resid", "TCC_Avg", "sm_avg")),
                   "Total": (("TCC_resid", "TCC_Avg", "NCHN_tp"),
                             ("SM_resid", "sm_avg", "NCHN_tp"))}
LABELS = {"sm_avg": "SM", "SM_resid": "SM resid", "SHF_Avg": "SHF",
          "SHF_resid": "SHF resid", "TCC_Avg": "TCC", "TCC_resid": "TCC resid",
          "NCHN_tp": "TP", "z500_anom_NCHN": "Z500", "NCVI_concurrent": "NCVI",
          "Initial_MSEstar_max_NCHN": "Init MSE*"}
JOHNSON_DISPLAY_LABELS = LABELS
LMG_SEED = 42
LMG_COUNT = 2000
LMG_ALGORITHM = "Cell8-MT19937-permutation-2000-seed42-subset-cache-v1"
RANK_RULE = "ordinal descending importance; exact ties resolved by predictor position"
ATOL = 1e-8
HISTORY_ATOL = 1e-6
HISTORY_FILES = {
    "predictors": "table_slim_predictor_sets.csv",
    "summary": "table_slim_model_summary.csv",
    "coefficients": "table_slim_model_coefficients.csv",
    "fitted": "table_slim_fitted_members.csv",
    "lmg": "table_slim_lmg_long.csv",
    "lofo": "table_slim_lofo_long.csv",
    "methods": "table_method_importance_comparison.csv",
}
TABLE_FILES = {
    "summary": "table_mlr_model_summary.csv",
    "coefficients": "table_mlr_model_coefficients.csv",
    "fitted": "table_mlr_fitted_members.csv",
    "importance": "table_mlr_importance_long.csv",
    "mapping": "table_mlr_predictor_mapping.csv",
    "design": "table_mlr_design_members.csv",
    "residualization": "table_mlr_residualization_audit.csv",
    "model_qc": "table_mlr_model_qc.csv",
    "equivalence": "table_mlr_fitting_equivalence.csv",
    "comparison": "table_mlr_raw_vs_residualized_importance_comparison.csv",
    "cross_init": "table_mlr_cross_init_qc.csv",
}


def predictors(window, version):
    if version not in VERSIONS:
        raise ValueError(f"Unsupported version {version}")
    names = RESIDUALIZED[window]
    return list(names if version == "residualized" else
                tuple(COUNTERPARTS.get(n, n) for n in names))


def predictor_mapping():
    return pd.DataFrame([
        {"window": w, "position": i, "predictor_residualized": r,
         "predictor_raw": a, "start_bjt": WINDOWS[w][0], "end_bjt": WINDOWS[w][1]}
        for w in WINDOWS for i, (r, a) in enumerate(zip(
            predictors(w, "residualized"), predictors(w, "raw")), 1)])


def _require(df, columns, context):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{context}: missing columns {missing}; available={list(df.columns)}")


def _finite(df, columns, context):
    _require(df, columns, context)
    try:
        values = df[columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: nonnumeric fields {columns}") from exc
    if not np.isfinite(values).all():
        bad = [(int(i), columns[int(j)]) for i, j in np.argwhere(~np.isfinite(values))]
        raise ValueError(f"{context}: missing/nonfinite values (row, column)={bad}")
    return values


def _members(g, context):
    _require(g, ["member"], context)
    m = _finite(g, ["member"], context).ravel()
    if len(m) != 51 or not np.array_equal(np.sort(m), np.arange(51)):
        raise ValueError(f"{context}: require exactly member IDs 0..50 once each; got={m.tolist()}")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs(input_root):
    """Read only two exported stage CSVs and optional historical CSVs."""
    root = Path(input_root).expanduser().resolve()
    paths = {"raw": root / "tables/table_stage_member_raw_factors_TCC.csv",
             "ncvi": root / "tables/table_ncvi_concurrent_stage_values.csv"}
    data = {}
    provenance = []
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing exported CSV: {path}. Copy existing exports or correct INPUT_ROOT; do not run upstream GRIB/NetCDF processing.")
        data[name] = pd.read_csv(path)
        _require(data[name], ["init_date", "stage", "member"], str(path))
        provenance.append({"name": name, "path": str(path), "sha256": _digest(path),
                           "rows": len(data[name]), "columns": list(data[name].columns)})
    history = {}
    for key, name in HISTORY_FILES.items():
        path = root / "tables" / name
        history[key] = pd.read_csv(path) if path.is_file() else None
        provenance.append({"name": key, "path": str(path), "exists": path.is_file(),
                           "sha256": _digest(path) if path.is_file() else None})
    return data["raw"], data["ncvi"], history, provenance


def check_predictor_history(history):
    df = history.get("predictors")
    if df is None:
        return "NOT_AVAILABLE"
    _require(df, ["stage", "predictors_raw_name"], "historical predictor sets")
    for w, stage in STAGES.items():
        seen = df.loc[df.stage == stage, "predictors_raw_name"].tolist()
        if seen != predictors(w, "residualized"):
            raise ValueError(f"Historical final predictor order mismatch {w}: {seen}; expected={predictors(w, 'residualized')}")
    return "PASS"


def prepare_pairs(raw, ncvi):
    """Validate all 8 groups, join by member, recompute historical residuals."""
    for name, df in [("raw", raw), ("NCVI", ncvi)]:
        _require(df, ["init_date", "stage", "member"], name)
    groups, audit = {}, []
    for init in INITS:
        for window, stage in STAGES.items():
            context = f"{init}/{window}"
            g = raw.loc[(raw.init_date.astype(str) == init) & (raw.stage == stage)].copy()
            n = ncvi.loc[(ncvi.init_date.astype(str) == init) & (ncvi.stage == stage)].copy()
            _members(g, context + " raw")
            _members(n, context + " NCVI")
            _finite(g, ["y_tmax"] + [p for p in predictors(window, "raw")
                                      if p != "NCVI_concurrent"], context)
            _finite(n, ["NCVI_concurrent"], context)
            for column, expected in [('ncvi_concurrent_window_start', WINDOWS[window][0]),
                                     ('ncvi_concurrent_window_end', WINDOWS[window][1])]:
                if column in n and not (n[column].astype(str) == expected).all():
                    raise ValueError(f'{context}: {column} must be {expected} (inclusive BJT)')
            # A stale NCVI column is never substituted for the actual concurrent table.
            g = g.drop(columns=["NCVI_concurrent"], errors="ignore")
            g = g.merge(n[["init_date", "stage", "member", "NCVI_concurrent"]],
                        on=["init_date", "stage", "member"], how="left", validate="one_to_one")
            g = g.sort_values("member").reset_index(drop=True)
            for target, source, control in RESIDUALIZATION[window]:
                design = sm.add_constant(g[[control]].astype(float), has_constant="add")
                model = sm.OLS(g[source].astype(float), design).fit()
                residual = g[source].astype(float) - model.predict(design)
                prior_diff = float(np.max(np.abs(g[target].to_numpy(float) - residual))) if target in g else np.nan
                g[target] = residual
                ortho = np.asarray(design).T @ residual.to_numpy()
                audit.append({"init_date": init, "window": window, "target": target,
                              "source": source, "control": control, "intercept": float(model.params.iloc[0]),
                              "slope": float(model.params.iloc[1]), "max_abs_design_dot_residual": float(np.max(np.abs(ortho))),
                              "prior_residual_column_max_difference": prior_diff})
            _finite(g, ["y_tmax"] + predictors(window, "residualized"), context)
            if np.std(g.y_tmax.to_numpy(float)) == 0:
                raise ValueError(f"{context}: constant y_tmax; R² and importance undefined")
            groups[(init, window)] = g
    return groups, pd.DataFrame(audit)


def fit_model(X, y, names, context):
    """Cell 8 StandardScaler(ddof=0), unscaled response, intercept OLS."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{context}: nonfinite design or response")
    if np.any(np.std(X, axis=0) == 0):
        raise ValueError(f"{context}: constant predictors {np.asarray(names)[np.std(X, axis=0) == 0].tolist()}")
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    design = sm.add_constant(Xz, has_constant="add")
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    if rank != len(names) + 1:
        raise ValueError(f"{context}: rank deficient rank={rank}; predictors={names}; singular_values={singular.tolist()}")
    model = sm.OLS(y, design).fit()
    fitted = np.asarray(model.predict(design))
    params = np.asarray(model.params)
    # Keep Cell 8's 1e-12 coefficient-conversion convention.
    slopes = params[1:] / (scaler.scale_ + 1e-12)
    intercept = float(params[0] - np.sum(params[1:] * scaler.mean_ / (scaler.scale_ + 1e-12)))
    return {"model": model, "X": X, "Xz": Xz, "design": design, "scaler": scaler,
            "y": y.copy(), "names": list(names), "fitted": fitted, "residual": y - fitted,
            "R2": float(model.rsquared), "Adj_R2": float(model.rsquared_adj),
            "RMSE": float(np.sqrt(np.mean((y - fitted) ** 2))), "rank": rank,
            "condition_number": float(np.linalg.cond(design)), "singular_values": singular,
            "raw_slopes": slopes, "raw_intercept": intercept}


def fit_pairs(groups):
    models = {}
    for (init, w), g in groups.items():
        for version in VERSIONS:
            names = predictors(w, version)
            models[(init, w, version)] = fit_model(g[names], g.y_tmax, names, f"{init}/{w}/{version}")
    return models


def lmg_permutations():
    # RandomState is MT19937, matching np.random.seed(42)/permutation in Cell 8.
    rng = np.random.RandomState(LMG_SEED)
    return np.array([rng.permutation(6) for _ in range(LMG_COUNT)], dtype=np.int64)


def lmg(X, y, permutations):
    """Cell 8 MC increments, epsilon scaling, positive clipping and normalization.

    Repeated subset R² is cached (all 63 nonempty subsets). Reordering a subset
    does not change its fitted space; parity is tested against the original loop.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    if permutations.shape != (LMG_COUNT, 6) or not np.array_equal(
            permutations, lmg_permutations()):
        raise ValueError("LMG requires the exact Cell 8 seed=42, 2000-permutation sequence")
    Xs = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    ys = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-8)
    cache, increments = {}, np.zeros(6)
    reg = LinearRegression()
    for order in permutations:
        prev = 0.0
        for i in range(6):
            idx = tuple(sorted(order[:i + 1]))
            if idx not in cache:
                cache[idx] = reg.fit(Xs[:, idx], ys).score(Xs[:, idx], ys)
            r2 = cache[idx]
            increments[order[i]] += max(0, r2 - prev)
            prev = r2
    total = increments.sum()
    pct = increments / total * 100 if total != 0 else np.zeros(6)
    return pct, increments / LMG_COUNT


def stable_ranks(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        return np.full(len(values), np.nan)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def positive_lofo_percent(delta):
    positive = np.maximum(np.asarray(delta, float), 0)
    denominator = float(positive.sum())
    return (positive / denominator * 100 if denominator > 0 else
            np.full(len(positive), np.nan)), denominator


def importance(models, permutations):
    rows, qc = [], []
    sequence_sha = hashlib.sha256(permutations.astype('<i8').tobytes()).hexdigest()
    for (init, w, v), m in models.items():
        lpct, lraw = lmg(m['X'], m['y'], permutations)
        names = m['names']
        jdf, jsum, jeig = _johnson_relative_weights(pd.DataFrame(m['X'], columns=names), m['y'], names)
        # Original numerical weights are unchanged; ranking alone uses an explicit tie rule.
        jpct = jdf.johnson_pct.to_numpy()
        delta, reduced_r2, reduced_rmse = [], [], []
        for i in range(6):
            # Never residualize again after deletion; retain full-model scaled columns.
            design = sm.add_constant(np.delete(m['Xz'], i, axis=1), has_constant="add")
            reduced = sm.OLS(m['y'], design).fit()
            reduced_r2.append(float(reduced.rsquared))
            reduced_rmse.append(float(np.sqrt(np.mean((m['y'] - reduced.predict(design))**2))))
            delta.append(m['R2'] - reduced.rsquared)
        fpct, fden = positive_lofo_percent(delta)
        for method, pct, raw_values in [("LMG", lpct, lraw),
                                       ("Johnson", jpct, jdf.johnson_raw.to_numpy()),
                                       ("LOFO", fpct, delta)]:
            ranks = stable_ranks(pct)
            for i, name in enumerate(names):
                rows.append({"init_date": init, "window": w, "version": v,
                             "method": method, "position": i + 1, "predictor": name,
                             "importance_pct": float(pct[i]), "raw_value": float(raw_values[i]),
                             "rank": float(ranks[i]),
                             "R2_without_factor": reduced_r2[i] if method == 'LOFO' else np.nan,
                             "RMSE_without_factor": reduced_rmse[i] if method == 'LOFO' else np.nan})
            total = float(np.sum(pct))
            status = "UNDEFINED" if not np.isfinite(pct).all() else ("PASS" if abs(total - 100) < 1e-7 else "FAIL")
            qc.append({"init_date": init, "window": w, "version": v, "method": method,
                       "pct_sum": total, "normalization_status": status,
                       "johnson_sum_raw": jsum, "johnson_sum_minus_R2": jsum - m['R2'],
                       "johnson_min_eigenvalue": jeig, "johnson_R2_status": 'PASS' if abs(jsum - m['R2']) <= 1e-5 else 'WARNING',
                       "lofo_positive_denominator": fden, "lmg_seed": LMG_SEED,
                       "lmg_permutation_count": LMG_COUNT, "lmg_permutation_sha256": sequence_sha,
                       "lmg_algorithm": LMG_ALGORITHM, "rank_rule": RANK_RULE})
        print(f"[MLR] {init} | {w} | {v}: n=51 R²={m['R2']:.8f} rank={m['rank']} Johnson sum−R²={jsum-m['R2']:.2e}")
    return pd.DataFrame(rows), pd.DataFrame(qc)


def assemble_results(groups, models, imp, model_qc, residual_audit):
    summary, coefficients, fitted, design_rows, equivalence = [], [], [], [], []
    for (init, w, v), m in models.items():
        common = {"init_date": init, "window": w, "version": v}
        summary.append({**common, "n": 51, "R2": m['R2'], "Adj_R2": m['Adj_R2'],
                        "RMSE": m['RMSE'], "predictor_list": ';'.join(m['names']),
                        "design_rank": m['rank'], "condition_number": m['condition_number'],
                        "singular_values": json.dumps(m['singular_values'].tolist()),
                        "standardized_intercept": float(m['model'].params[0]),
                        "raw_equivalent_intercept": m['raw_intercept']})
        for i, name in enumerate(m['names']):
            coefficients.append({**common, "position": i + 1, "predictor": name,
                                 "standardized_coef": float(m['model'].params[i + 1]),
                                 "coef": float(m['raw_slopes'][i]), "mean": m['scaler'].mean_[i],
                                 "scale_ddof0": m['scaler'].scale_[i]})
        coefficients.append({**common, "position": 0, "predictor": 'const',
                             "standardized_coef": float(m['model'].params[0]),
                             "coef": m['raw_intercept'], "mean": np.nan, "scale_ddof0": np.nan})
        for i, member in enumerate(groups[(init, w)].member):
            fitted.append({**common, "member": member, "y_tmax": m['y'][i],
                           "y_fitted": m['fitted'][i], "residual": m['residual'][i]})
            for j, name in enumerate(m['names']):
                design_rows.append({**common, "member": member, "position": j + 1,
                                    "predictor": name, "value": m['X'][i, j],
                                    "standardized_value": m['Xz'][i, j], "y_tmax": m['y'][i]})
    for init, w in groups:
        r, a = models[(init, w, 'residualized')], models[(init, w, 'raw')]
        qr, _ = np.linalg.qr(r['design'], mode='reduced')
        qa, _ = np.linalg.qr(a['design'], mode='reduced')
        diff = {"init_date": init, "window": w, "R2_raw": a['R2'], "R2_resid": r['R2'],
                "R2_difference": a['R2'] - r['R2'],
                "max_abs_fitted_difference": float(np.max(np.abs(a['fitted'] - r['fitted']))),
                "max_abs_residual_difference": float(np.max(np.abs(a['residual'] - r['residual']))),
                "projection_max_abs_difference": float(np.max(np.abs(qa @ qa.T - qr @ qr.T))),
                "design_rank_raw": a['rank'], "design_rank_resid": r['rank'],
                "response_identical": bool(np.array_equal(a['y'], r['y'])), "tolerance": ATOL}
        diff['status'] = 'PASS' if diff['response_identical'] and max(abs(diff['R2_difference']), diff['max_abs_fitted_difference'], diff['max_abs_residual_difference'], diff['projection_max_abs_difference']) <= ATOL else 'FAIL'
        equivalence.append(diff)
    eq = pd.DataFrame(equivalence)
    comparison, cross = [], []
    for init, w in groups:
        eql = eq.loc[(eq.init_date == init) & (eq.window == w)].iloc[0].to_dict()
        for method in METHODS:
            sub = imp[(imp.init_date == init) & (imp.window == w) & (imp.method == method)]
            r = sub[sub.version == 'residualized'].sort_values('position')
            a = sub[sub.version == 'raw'].sort_values('position')
            for _, rr in r.iterrows():
                aa = a[a.position == rr.position].iloc[0]
                # Figure order always uses residualized LMG, never raw LMG.
                lmg_r = imp[(imp.init_date == init) & (imp.window == w) & (imp.method == 'LMG') & (imp.version == 'residualized')]
                display = lmg_r.sort_values(['importance_pct', 'position'], ascending=[False, True], kind='stable').position.tolist()
                comparison.append({**eql, "method": method, "position": rr.position,
                                   "predictor_raw": aa.predictor, "predictor_residualized": rr.predictor,
                                   "display_order": display.index(rr.position) + 1,
                                   "importance_raw_pct": aa.importance_pct, "importance_resid_pct": rr.importance_pct,
                                   "difference_pp": aa.importance_pct - rr.importance_pct,
                                   "rank_raw": aa['rank'], "rank_resid": rr['rank'],
                                   "lofo_delta_R2_raw": aa.raw_value if method == 'LOFO' else np.nan,
                                   "lofo_delta_R2_resid": rr.raw_value if method == 'LOFO' else np.nan})
            ar = a.sort_values(['importance_pct', 'position'], ascending=[False, True], kind='stable')
            rr = r.sort_values(['importance_pct', 'position'], ascending=[False, True], kind='stable')
            cross.append({**eql, "method": method,
                          "raw_first": ar.predictor.iloc[0] if ar.importance_pct.notna().all() else None,
                          "resid_first": rr.predictor.iloc[0] if rr.importance_pct.notna().all() else None,
                          "raw_top3": ';'.join(ar.predictor.iloc[:3]) if ar.importance_pct.notna().all() else None,
                          "resid_top3": ';'.join(rr.predictor.iloc[:3]) if rr.importance_pct.notna().all() else None,
                          "position_order_changed": ar.position.tolist() != rr.position.tolist(),
                          "raw_pct_sum": a.importance_pct.sum(min_count=6),
                          "resid_pct_sum": r.importance_pct.sum(min_count=6)})
    return {"summary": pd.DataFrame(summary), "coefficients": pd.DataFrame(coefficients),
            "fitted": pd.DataFrame(fitted), "design": pd.DataFrame(design_rows),
            "importance": imp, "mapping": predictor_mapping(), "model_qc": model_qc,
            "equivalence": eq, "residualization": residual_audit,
            "comparison": pd.DataFrame(comparison), "cross_init": pd.DataFrame(cross)}


def compare_history(results, history, atol=HISTORY_ATOL):
    """Every available historical field is checked, never substituted for new fits."""
    checks = []
    specs = {
        'summary': ('summary', [], {'R2': 'R2', 'Adj_R2': 'Adj_R2', 'RMSE': 'RMSE', 'n': 'n', 'predictor_list': 'predictor_list'}),
        'coefficients': ('coefficients', ['predictor'], {'standardized_coef': 'standardized_coef', 'coef': 'coef'}),
        'fitted': ('fitted', ['member'], {'y_tmax': 'y_tmax', 'y_fitted': 'y_fitted', 'residual': 'residual'}),
        'lmg': ('importance', ['predictor'], {'LMG_percent': 'importance_pct'}),
        'lofo': ('importance', ['predictor'], {'Delta_R2': 'raw_value', 'positive_delta_R2_share_percent': 'importance_pct'}),
        'methods': ('importance', ['predictor'], {'johnson_raw': 'raw_value', 'johnson_pct': 'importance_pct'}),
    }
    for init in INITS:
        for w, stage in STAGES.items():
            for key, (newkey, keys, fields) in specs.items():
                old = history.get(key)
                base = {'init_date': init, 'window': w, 'source': key}
                if old is None:
                    checks.append({**base, 'field': '*', 'status': 'NOT_AVAILABLE', 'detail': 'historical CSV absent'})
                    continue
                try:
                    _require(old, ['init_date', 'stage'], 'history ' + key)
                    old = old[(old.init_date.astype(str) == init) & (old.stage == stage)].copy()
                    if key == 'methods':
                        old = old.rename(columns={'factor': 'predictor'})
                    new = results[newkey]
                    new = new[(new.init_date == init) & (new.window == w) & (new.version == 'residualized')].copy()
                    if key in ('lmg', 'lofo', 'methods'):
                        new = new[new.method == {'lmg': 'LMG', 'lofo': 'LOFO', 'methods': 'Johnson'}[key]]
                    _require(old, keys, 'history ' + key)
                    if keys:
                        if old.duplicated(keys).any() or set(map(tuple, old[keys].to_numpy())) != set(map(tuple, new[keys].to_numpy())):
                            raise ValueError('historical row keys missing/duplicated/mismatched')
                        old = old.sort_values(keys).reset_index(drop=True)
                        new = new.sort_values(keys).reset_index(drop=True)
                    if len(old) != len(new) or len(new) == 0:
                        raise ValueError(f'row counts historical={len(old)} new={len(new)}')
                    for oldfield, newfield in fields.items():
                        if oldfield not in old:
                            checks.append({**base, 'field': oldfield, 'status': 'NOT_AVAILABLE', 'detail': 'historical field absent'})
                            continue
                        a, b = old[oldfield], new[newfield]
                        if oldfield == 'predictor_list':
                            passed = a.tolist() == b.tolist()
                            difference = np.nan
                        else:
                            av, bv = a.to_numpy(float), b.to_numpy(float)
                            # Historical const standardized_coef was NA; other slopes must be finite.
                            mask = ~(a.isna().to_numpy() & b.isna().to_numpy())
                            if key == 'coefficients' and oldfield == 'standardized_coef':
                                mask = old.predictor.to_numpy() != 'const'
                            passed = bool(np.allclose(av[mask], bv[mask], atol=atol, rtol=0, equal_nan=False))
                            difference = float(np.max(np.abs(av[mask] - bv[mask]))) if mask.any() else 0.0
                        checks.append({**base, 'field': oldfield, 'status': 'PASS' if passed else 'FAIL',
                                       'max_abs_difference': difference, 'tolerance': atol,
                                       'detail': '' if passed else 'Check CSV provenance, model/scaler/residualization and library numerical versions; no historical values substituted'})
                except (ValueError, KeyError, TypeError) as exc:
                    checks.append({**base, 'field': '*', 'status': 'FAIL', 'detail': str(exc)})
            try:
                status = check_predictor_history(history)
                checks.append({'init_date': init, 'window': w, 'source': 'predictors', 'field': 'order', 'status': status})
            except ValueError as exc:
                checks.append({'init_date': init, 'window': w, 'source': 'predictors', 'field': 'order', 'status': 'FAIL', 'detail': str(exc)})
    return pd.DataFrame(checks)


def qc_status(results, history_qc):
    eq = results['equivalence']
    q = results['model_qc']
    fail = ((eq.status != 'PASS').any() or (q.normalization_status != 'PASS').any()
            or (history_qc.status == 'FAIL').any())
    history_status = 'FAIL' if (history_qc.status == 'FAIL').any() else (
        'PASS' if (history_qc.status == 'PASS').all() else 'INCOMPLETE')
    for i, row in results['cross_init'].iterrows():
        h = history_qc[(history_qc.init_date == row.init_date) & (history_qc.window == row.window)]
        results['cross_init'].loc[i, 'history_status'] = 'FAIL' if (h.status == 'FAIL').any() else ('PASS' if (h.status == 'PASS').all() else 'INCOMPLETE')
    return {'computational_status': 'FAIL' if fail else 'PASS', 'history_status': history_status,
            'johnson_R2_warnings': int((q.johnson_R2_status == 'WARNING').sum()),
            'real_scientific_validation': 'requires genuine HPC CSV inputs; synthetic tests are not paper results'}


def output_guard(input_root, output_root):
    inp, out = Path(input_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if out == inp or out in inp.parents:
        raise ValueError('OUTPUT_ROOT cannot equal or contain INPUT_ROOT')
    for historical in ('tables', 'figures', 'qc', 'paper_figures_standalone'):
        protected = inp / historical
        if out == protected or protected in out.parents:
            raise ValueError(f'OUTPUT_ROOT is inside historical {protected}')
    marker = out / 'qc/comparison_manifest.json'
    if out.exists() and any(out.iterdir()):
        if not marker.is_file():
            raise ValueError(f'Refusing to write into an existing unowned directory: {out}')
        owned = json.loads(marker.read_text())
        if owned.get('schema_version') != 1 or owned.get('source_commit') != SOURCE_COMMIT:
            raise ValueError(f'Output ownership manifest does not identify this workflow: {out}')
    # A child symlink must never redirect comparison writes to historical results.
    for folder in ('tables', 'figures', 'qc'):
        child = out / folder
        if child.is_symlink() or (child.exists() and out not in child.resolve().parents):
            raise ValueError(f'Unsafe output subdirectory: {child}')
        if child.is_dir() and any(p.is_symlink() for p in child.iterdir()):
            raise ValueError(f'Output directory contains symlink targets: {child}')
    return out


def save_results(results, history_qc, input_root, output_root, provenance, permutations):
    out = output_guard(input_root, output_root)
    for folder in ('tables', 'figures', 'qc'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    status = qc_status(results, history_qc)
    for key, name in TABLE_FILES.items():
        results[key].to_csv(out / 'tables' / name, index=False)
    history_qc.to_csv(out / 'qc/table_mlr_historical_reproduction.csv', index=False)
    pd.DataFrame(permutations, columns=[f'position_{i}' for i in range(1, 7)]).to_csv(
        out / 'qc/lmg_permutations_seed42_2000.csv', index=False)
    manifest = {**status, 'schema_version': 1, 'source_commit': SOURCE_COMMIT,
                'input_root': str(Path(input_root).resolve()), 'output_root': str(out),
                'windows_bjt': WINDOWS, 'inits': INITS, 'versions': VERSIONS,
                'lmg_seed': LMG_SEED, 'lmg_count': LMG_COUNT, 'lmg_algorithm': LMG_ALGORITHM,
                'rank_rule': RANK_RULE, 'equivalence_atol': ATOL, 'history_atol': HISTORY_ATOL,
                'provenance': provenance,
                'versions_used': {'python': platform.python_version(), 'numpy': np.__version__,
                                  'pandas': pd.__version__, 'scipy': scipy.__version__,
                                  'sklearn': sklearn.__version__, 'statsmodels': statsmodels.__version__},
                'table_sha256': {name: _digest(out / 'tables' / name) for name in TABLE_FILES.values()}}
    (out / 'qc/comparison_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if status['computational_status'] != 'PASS':
        warnings.warn('Results saved for audit, but QC failed/importance undefined. Figure export is blocked; inspect QC CSVs.', stacklevel=2)
    return manifest


# Exact pure function copied from historical Cell 9; no upstream execution.
def _johnson_relative_weights(X_df, y_series, factor_names):
    if len(X_df) != 51:
        raise RuntimeError(f"Johnson relative weights require n=51, got n={len(X_df)}")
    X = X_df[factor_names].astype(float).copy()
    y = pd.Series(y_series, index=X.index).astype(float)
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    yz = (y - y.mean()) / (y.std(ddof=1) + 1e-12)
    Rxx = np.corrcoef(Xz.values, rowvar=False)
    rxy = np.array([np.corrcoef(Xz[col].values, yz.values)[0, 1] for col in factor_names], dtype=float)
    eigval, eigvec = np.linalg.eigh(Rxx)
    min_eig = float(np.nanmin(eigval))
    eigval_clip = np.clip(eigval, 1e-12, None)
    delta = eigvec @ np.diag(np.sqrt(eigval_clip)) @ eigvec.T
    beta = np.linalg.pinv(delta) @ rxy
    raw_weights = (delta ** 2) @ (beta ** 2)
    raw_weights = np.where(raw_weights < 1e-12, 0.0, raw_weights)
    total = float(np.sum(raw_weights))
    pct = raw_weights / total * 100.0 if total > 0 else np.zeros_like(raw_weights)
    rows = []
    ranks = {factor: rank + 1 for rank, factor in enumerate([factor_names[i] for i in np.argsort(-pct)])}
    for i, factor in enumerate(factor_names):
        rows.append({
            "factor": factor,
            "display_label": JOHNSON_DISPLAY_LABELS.get(factor, factor),
            "johnson_raw": float(raw_weights[i]),
            "johnson_pct": float(pct[i]),
            "johnson_rank": int(ranks[factor]),
        })
    return pd.DataFrame(rows), total, min_eig
