'Evaluate construction-event and geographic holdouts, calibration, residuals and feature attribution.'

import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from splits import (calibration_split, conformal_halfwidth, conformal_interval,
                    coverage, feature_matrix, load)

sys.stdout.reconfigure(encoding="utf-8")

CAL_MAX, ALPHA = 1500, 0.1
RNG = np.random.default_rng(0)


def models():
    return {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "LightGBM": lambda: LGBMRegressor(n_estimators=800, learning_rate=0.05,
                                          num_leaves=63, random_state=0, verbose=-1),
        "XGBoost": lambda: XGBRegressor(n_estimators=800, learning_rate=0.05,
                                        max_depth=8, subsample=0.9,
                                        colsample_bytree=0.9, random_state=0,
                                        tree_method="hist"),
        "CatBoost": lambda: CatBoostRegressor(iterations=800, learning_rate=0.05,
                                              depth=8, random_seed=0, verbose=0),
    }


def fit_predict(name, X, y, fit, te, med=None):
    m = models()[name]()
    Xf = X.iloc[fit]
    med = Xf.median() if med is None else med
    if name == "RandomForest":
        Xf = Xf.fillna(med)
    m.fit(Xf, y.iloc[fit])
    Xt = X.iloc[te].fillna(med) if name == "RandomForest" else X.iloc[te]
    return m, m.predict(Xt), med


# ---------------------------------------------------------------- (a) + (b)
def unit_and_state(df, X, y):
    rows = []
    year = df["VISIT_DATE"].dt.year.values

    # (a) construction-event disjoint, forward in time
    units = df["unit_id"].unique().copy()
    RNG.shuffle(units)
    for k, part in enumerate(np.array_split(units, 3)):
        te_mask = np.isin(df["unit_id"].values, part)
        tr = np.where(~te_mask & (year <= 2013))[0]
        te = np.where(te_mask & (year > 2013))[0]
        if len(tr) < 100 or len(te) < 50:
            continue
        for name in models():
            _, p, _ = fit_predict(name, X, y, tr, te)
            rows.append({"design": "construction-event disjoint", "model": name,
                         "fold": k, "R2": r2_score(y.iloc[te], p),
                         "MAE": mean_absolute_error(y.iloc[te], p), "n_test": len(te)})
            print(f"  event fold{k} {name}: R2={rows[-1]['R2']:.4f}",flush=True)

    # (b) leave-one-state-out
    counts = df.groupby("STATE_CODE").size().sort_values(ascending=False)
    for st in counts.head(12).index:
        te = np.where(df["STATE_CODE"].values == st)[0]
        tr = np.where(df["STATE_CODE"].values != st)[0]
        if len(te) < 100:
            continue
        for name in models():
            _, p, _ = fit_predict(name, X, y, tr, te)
            rows.append({"design": "leave-one-state-out", "model": name,
                         "fold": int(st), "R2": r2_score(y.iloc[te], p),
                         "MAE": mean_absolute_error(y.iloc[te], p), "n_test": len(te)})
            print(f"  jurisdiction{st} {name}: R2={rows[-1]['R2']:.4f}",flush=True)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- (c-e)
def calibration_residual_type(df, X, y):
    from splits import splits_temporal
    tr, te = splits_temporal(df, cutoff=2013, n_splits=3, seed=0)[0]
    cal, fit = calibration_split(tr, seed=0)

    cal_rows, res_rows, type_rows = [], [], []
    for name in models():
        m, p_te, med = fit_predict(name, X, y, fit, te)
        Xc = X.iloc[cal].fillna(med) if name == "RandomForest" else X.iloc[cal]
        resid = np.abs(y.iloc[cal].values - m.predict(Xc))
        yt = y.iloc[te].values

        # (c) calibration across nominal levels
        for lvl in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            qh = conformal_halfwidth(resid, alpha=1 - lvl)
            lo, hi = conformal_interval(p_te, qh)
            cal_rows.append({"model": name, "nominal": lvl,
                             "empirical": coverage(yt, lo, hi),
                             "width": float(np.mean(hi - lo))})

        # (d) residuals
        res_rows.append(pd.DataFrame({"model": name, "y_true": yt, "y_pred": p_te,
                                      "residual": yt - p_te,
                                      "age": df.iloc[te]["age_years"].values}))

        # (e) by pavement type, inferred from layer composition
        sub = df.iloc[te]
        rigid = (sub.get("thk_PC", pd.Series(np.nan, index=sub.index)).fillna(0) > 0).values
        for lab, msk in [("No recorded PCC layer", ~rigid), ("Recorded PCC layer", rigid)]:
            if msk.sum() >= 30:
                type_rows.append({"model": name, "pavement_type": lab,
                                  "R2": r2_score(yt[msk], p_te[msk]),
                                  "MAE": mean_absolute_error(yt[msk], p_te[msk]),
                                  "n": int(msk.sum())})
    return (pd.DataFrame(cal_rows), pd.concat(res_rows, ignore_index=True),
            pd.DataFrame(type_rows))


# ---------------------------------------------------------------------- (f)
def importance(df, X, y):
    from splits import splits_temporal
    tr, te = splits_temporal(df, cutoff=2013, n_splits=3, seed=0)[0]
    m = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=8,
                          random_seed=0, verbose=0)
    m.fit(X.iloc[tr], y.iloc[tr])

    pi = permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=10,
                                random_state=0, scoring="r2", n_jobs=1)
    perm = (pd.DataFrame({"feature": X.columns, "perm_importance": pi.importances_mean,
                          "perm_sd": pi.importances_std})
            .sort_values("perm_importance", ascending=False))

    try:
        import shap
        sv = shap.TreeExplainer(m).shap_values(X.iloc[te])
        perm = perm.merge(pd.DataFrame(
            {"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}),
            on="feature", how="left")
    except Exception as e:
        print("SHAP unavailable:", str(e)[:70])
    return perm


if __name__ == "__main__":
    df = load()
    X, y = feature_matrix(df)
    print(f"rows {len(df):,}  features {X.shape[1]}", flush=True)

    t = time.time()
    a = unit_and_state(df, X, y)
    a.to_csv("panel/review_designs.csv", index=False)
    print(f"\n=== additional partitioning designs ({time.time()-t:.0f}s) ===")
    print(a.groupby(["design", "model"])[["R2", "n_test"]].mean().round(4).to_string())

    cal, res, typ = calibration_residual_type(df, X, y)
    cal.to_csv("panel/calibration_curve.csv", index=False)
    res.to_csv("panel/residuals.csv", index=False)
    typ.to_csv("panel/by_pavement_type.csv", index=False)
    print("\n=== calibration (empirical vs nominal) ===")
    print(cal.pivot(index="nominal", columns="model", values="empirical").round(3).to_string())
    print("\n=== by pavement type ===")
    print(typ.pivot(index="model", columns="pavement_type", values="R2").round(4).to_string())

    imp = importance(df, X, y)
    imp.to_csv("panel/feature_importance.csv", index=False)
    print("\n=== top 15 features (CatBoost, permutation) ===")
    print(imp.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
