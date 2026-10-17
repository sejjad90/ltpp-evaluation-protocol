'Measure CPU fit and batch prediction costs and single-fold weather/layer subgroups.'

import sys
import time
import tracemalloc
import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from splits import calibration_split, feature_matrix, load, splits_temporal

sys.stdout.reconfigure(encoding="utf-8")

PRECIP_WET_MM = 508.0        # LTPP wet/dry boundary, 20 in
CUTOFF = 2013


def builders():
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
        "TabPFN-3": None,        # constructed lazily, import is slow
    }


def make(name):
    if name == "TabPFN-3":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(n_estimators=8, auto_scale_n_estimators=False,
                               random_state=0, ignore_pretraining_limits=True)
    return builders()[name]()


def climate_zone(row):
    wet = row["t12_precip"] >= PRECIP_WET_MM
    freeze = (row["t12_freeze_thaw"] or 0) >= 1
    return ("Wet" if wet else "Dry") + ("-freeze" if freeze else "-no-freeze")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trees-only',action='store_true')
    args=ap.parse_args()
    df = load()
    X, y = feature_matrix(df)
    tr, te = splits_temporal(df, cutoff=CUTOFF, n_splits=3, seed=0)[0]
    cal, fit = calibration_split(tr, seed=0)
    print(f"cost fold: train {len(fit):,}  calib {len(cal):,}  test {len(te):,}",
          flush=True)

    cost, preds = [], {}
    for name in builders():
        if args.trees_only and name == 'TabPFN-3': continue
        Xf, Xt = X.iloc[fit], X.iloc[te]
        if name == "RandomForest":
            med = Xf.median(); Xf, Xt = Xf.fillna(med), Xt.fillna(med)
        m = make(name)
        tracemalloc.start()
        t0 = time.perf_counter(); m.fit(Xf, y.iloc[fit]); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict(Xt); t_pred = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        preds[name] = p
        cost.append({"model": name, "fit_s": t_fit, "predict_s": t_pred,
                     "total_s": t_fit + t_pred,
                     "peak_mem_MB": peak / 1e6,  # Python-traced allocation only
                     "ms_per_prediction": 1000 * t_pred / len(te),  # amortised batch ms/row
                     "R2": r2_score(y.iloc[te], p)})
        cost[-1].update(n_fit=len(fit),n_cal=len(cal),n_test=len(te))
        print(f"  {name:13s} fit {t_fit:8.1f}s  predict {t_pred:8.1f}s  "
              f"Python-traced peak {peak/1e6:7.1f} MB  R2={cost[-1]['R2']:.4f}", flush=True)

    c = pd.DataFrame(cost)
    c["relative_to_LightGBM"] = c.total_s / c.loc[c.model == "LightGBM", "total_s"].values[0]
    c.to_csv("panel/computational_cost.csv", index=False)
    print("\n=== computational cost (one forward-in-time fold, CPU) ===")
    print(c.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ---------------------------------------------------------- heterogeneity
    sub = df.iloc[te].copy()
    sub["zone"] = sub.apply(climate_zone, axis=1)
    sub["family"] = np.where(
        sub.get("thk_PC", pd.Series(0.0, index=sub.index)).fillna(0) > 0,
        "Recorded PCC layer", "No recorded PCC layer")
    yt = y.iloc[te].values

    rows = []
    for name, p in preds.items():
        for key, lab in [("zone", "climatic regime"), ("family", "pavement family")]:
            for g, idx in sub.groupby(key).groups.items():
                msk = sub.index.isin(idx)
                if msk.sum() < 30:
                    continue
                rows.append({"model": name, "stratum_type": lab, "stratum": g,
                             "n": int(msk.sum()),
                             "R2": r2_score(yt[msk], p[msk]),
                             "MAE": mean_absolute_error(yt[msk], p[msk])})
    h = pd.DataFrame(rows)
    h.to_csv("panel/heterogeneity.csv", index=False)
    print("\n=== forward-in-time R2 by stratum ===")
    print(h.pivot_table(index=["stratum_type", "stratum"], columns="model",
                        values="R2").round(4).to_string())
    print("\nstratum sizes:")
    print(h[h.model == "CatBoost"][["stratum_type", "stratum", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
