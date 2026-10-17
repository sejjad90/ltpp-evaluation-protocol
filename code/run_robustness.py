"""Evaluate a repeated section-disjoint later-survey grid.

Three section folds, three seeds and three cut-off years give matched
configurations across learners. Per-fold metrics are written to the path
selected by --out, normally robustness_trees.csv or robustness_tabpfn.csv.
Paired inference is performed separately by run_cluster_stats.py.
"""

import argparse
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from splits import (calibration_split, conformal_halfwidth, conformal_interval,
                    coverage, feature_matrix, load, splits_temporal)

SEEDS = [0, 1, 2]
CUTOFFS = [2010, 2013, 2016]
N_SPLITS = 3
CAL_MAX = 1500
ALPHA = 0.1


def tree_models():
    return {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            random_state=0, verbose=-1),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=8, subsample=0.9,
            colsample_bytree=0.9, random_state=0, tree_method="hist"),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=8, random_seed=0, verbose=0),
    }


def build_model(name):
    if name == "TabPFN-3":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(n_estimators=8, auto_scale_n_estimators=False,
                               random_state=0, ignore_pretraining_limits=True)
    return tree_models()[name]()


def evaluate(name, X, y, fit, cal, te):
    m = build_model(name)
    Xf = X.iloc[fit]
    med = Xf.median()
    if name == "RandomForest":
        Xf = Xf.fillna(med)
    m.fit(Xf, y.iloc[fit])

    def pred(idx):
        Xi = X.iloc[idx]
        if name == "RandomForest":
            Xi = Xi.fillna(med)
        return m.predict(Xi)

    qhat = conformal_halfwidth(np.abs(y.iloc[cal].values - pred(cal)))
    p = pred(te)
    yt = y.iloc[te].values
    lo, hi = conformal_interval(p, qhat)
    return {
        "R2": r2_score(yt, p),
        "MAE": mean_absolute_error(yt, p),
        "coverage_90": coverage(yt, lo, hi),
        "interval_width": float(np.mean(hi - lo)),
        "n_test": len(te),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tree", choices=["tree", "tabpfn", "all"])
    ap.add_argument("--out", default="panel/robustness_folds.csv")
    args = ap.parse_args()

    names = list(tree_models()) if args.models == "tree" else (
        ["TabPFN-3"] if args.models == "tabpfn" else list(tree_models()) + ["TabPFN-3"])

    df = load()
    X, y = feature_matrix(df)
    print(f"rows {len(df):,}  sections {df.section_id.nunique():,}  models {names}", flush=True)

    recs = []
    for seed in SEEDS:
        for cut in CUTOFFS:
            folds = splits_temporal(df, cutoff=cut, n_splits=N_SPLITS, seed=seed)
            for fi, (tr, te) in enumerate(folds):
                cal, fit = calibration_split(tr, seed=seed)
                for name in names:
                    t0 = time.time()
                    r = evaluate(name, X, y, fit, cal, te)
                    r.update(model=name, seed=seed, cutoff=cut, fold=fi,
                             secs=round(time.time() - t0, 1))
                    recs.append(r)
                    print(f"  seed{seed} cut{cut} f{fi} {name:13s} "
                          f"R2={r['R2']:.4f} cov={r['coverage_90']:.3f} "
                          f"n={r['n_test']} ({r['secs']}s)", flush=True)
                pd.DataFrame(recs).to_csv(args.out, index=False)

    d = pd.DataFrame(recs)
    print("\n=== mean over %d configurations ===" % (len(SEEDS) * len(CUTOFFS) * N_SPLITS))
    print(d.groupby("model")[["R2", "coverage_90", "n_test"]]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
