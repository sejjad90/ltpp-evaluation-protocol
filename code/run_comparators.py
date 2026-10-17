'Compare four tree learners under row-random, section-disjoint and conditional later-survey protocols, with shared calibration holdouts.'

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from splits import (PROTOCOLS, calibration_split, conformal_halfwidth,
                    conformal_interval, coverage, feature_matrix, load)

CAL_FRAC = 0.2
ALPHA = 0.1


def models():
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "LightGBM": LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            random_state=0, verbose=-1),
        "XGBoost": XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, random_state=0,
            tree_method="hist"),
        "CatBoost": CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=8,
            random_seed=0, verbose=0),
    }


def main():
    df = load()
    X, y = feature_matrix(df)
    X = X.fillna(np.nan)
    print(f"rows {len(df):,}  sections {df.section_id.nunique():,}  features {X.shape[1]}\n")

    rows, fold_rows = [], []
    for pname, splitter in PROTOCOLS.items():
        folds = splitter(df)
        for mname, _ in models().items():
            r2s, maes, rmses, covs, widths = [], [], [], [], []
            for tr, te in folds:
                cal, fit = calibration_split(tr, seed=0)

                m = models()[mname]
                Xf = X.iloc[fit]
                if mname in ("RandomForest",):
                    Xf = Xf.fillna(Xf.median())
                m.fit(Xf, y.iloc[fit])

                def pred(idx):
                    Xi = X.iloc[idx]
                    if mname in ("RandomForest",):
                        Xi = Xi.fillna(X.iloc[fit].median())
                    return m.predict(Xi)

                qhat = conformal_halfwidth(np.abs(y.iloc[cal].values - pred(cal)))
                p = pred(te)
                yt = y.iloc[te].values
                lo, hi = conformal_interval(p, qhat)
                r2s.append(r2_score(yt, p))
                maes.append(mean_absolute_error(yt, p))
                rmses.append(np.sqrt(mean_squared_error(yt, p)))
                covs.append(coverage(yt, lo, hi))
                widths.append(float(np.mean(hi - lo)))
                fold_rows.append(dict(model=mname, protocol=pname, fold=len(r2s)-1,
                    n_fit=len(fit), n_cal=len(cal), n_test=len(te),
                    R2=r2s[-1], MAE=maes[-1], RMSE=rmses[-1],
                    coverage_90=covs[-1], interval_width=widths[-1]))
                pd.DataFrame(fold_rows).to_csv('panel/principal_tree_folds.csv',index=False)

            rows.append({
                "protocol": pname, "model": mname,
                "R2": np.mean(r2s), "R2_sd": np.std(r2s),
                "MAE": np.mean(maes), "RMSE": np.mean(rmses),
                "coverage_90": np.mean(covs), "interval_width": np.mean(widths),
                "MAE_sd":np.std(maes), "RMSE_sd":np.std(rmses),
                "coverage_90_sd":np.std(covs), "interval_width_sd":np.std(widths),
            })
            print(f"  {pname:36s} {mname:14s} R2={np.mean(r2s):.4f}  "
                  f"MAE={np.mean(maes):.4f}  cov={np.mean(covs):.3f}")

    res = pd.DataFrame(rows)
    res.to_csv("panel/comparator_results.csv", index=False)
    print("\n" + res.pivot(index="model", columns="protocol", values="R2")
          .to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
