'Compare predictions with and without first-survey MRI using the shared principal fitting/calibration splits. The ablation measures predictive dependence and does not isolate persistence skill.'

import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from splits import PROTOCOLS, calibration_split, feature_matrix, load

sys.stdout.reconfigure(encoding="utf-8")

rows = []
df = load()
X_full, y = feature_matrix(df)
X_drop = X_full.drop(columns=[c for c in ["initial_MRI"] if c in X_full.columns])

for label, X in [("with initial_MRI", X_full), ("without initial_MRI", X_drop)]:
    for pname, splitter in PROTOCOLS.items():
        r2s, maes = [], []
        for tr, te in splitter(df):
            # Same fitting/calibration split as the principal CatBoost comparison,
            # so with-initial-MRI results match Table 3.
            _cal, fit = calibration_split(tr, seed=0)
            m = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=8,
                                  random_seed=0, verbose=0)
            m.fit(X.iloc[fit], y.iloc[fit])
            p = m.predict(X.iloc[te])
            r2s.append(r2_score(y.iloc[te], p))
            maes.append(mean_absolute_error(y.iloc[te], p))
        rows.append({"features": label, "protocol": pname,
                     "R2": np.mean(r2s), "MAE": np.mean(maes)})
        print(f"  {label:22s} {pname:36s} R2={np.mean(r2s):.4f}", flush=True)

res = pd.DataFrame(rows)
res.to_csv("panel/ablation_initial_mri.csv", index=False)
piv = res.pivot(index="protocol", columns="features", values="R2")
piv["drop_from_covariate"] = piv["with initial_MRI"] - piv["without initial_MRI"]
print("\n=== CatBoost R2 ===")
print(piv.round(4).to_string())
