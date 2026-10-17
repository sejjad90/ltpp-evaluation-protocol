'Evaluate tree-model configuration sensitivity and nested tuning.'

import itertools
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from splits import feature_matrix, load, splits_random, splits_temporal

sys.stdout.reconfigure(encoding="utf-8")

CUTOFF, NFOLD, SEED = 2013, 3, 0

GRIDS = {
    "RandomForest": [{"n_estimators": n, "min_samples_leaf": l}
                     for n, l in itertools.product([200, 400], [1, 2, 5])],
    "LightGBM": [{"n_estimators": n, "num_leaves": v, "learning_rate": r}
                 for n, v, r in itertools.product([400, 800], [31, 63], [0.05])]
                + [{"n_estimators": 800, "num_leaves": 127, "learning_rate": 0.03}],
    "XGBoost": [{"n_estimators": n, "max_depth": d, "learning_rate": 0.05}
                for n, d in itertools.product([400, 800], [6, 8, 10])],
    "CatBoost": [{"iterations": n, "depth": d, "learning_rate": 0.05}
                 for n, d in itertools.product([400, 800], [6, 8, 10])],
}


def build(name, cfg):
    if name == "RandomForest":
        return RandomForestRegressor(n_jobs=-1, random_state=0, **cfg)
    if name == "LightGBM":
        return LGBMRegressor(random_state=0, verbose=-1, **cfg)
    if name == "XGBoost":
        return XGBRegressor(subsample=0.9, colsample_bytree=0.9, random_state=0,
                            tree_method="hist", **cfg)
    return CatBoostRegressor(random_seed=0, verbose=0, **cfg)


def score(name, cfg, X, y, tr, te):
    Xf, Xt = X.iloc[tr], X.iloc[te]
    if name == "RandomForest":
        med = Xf.median(); Xf, Xt = Xf.fillna(med), Xt.fillna(med)
    m = build(name, cfg)
    m.fit(Xf, y.iloc[tr])
    return r2_score(y.iloc[te], m.predict(Xt))


def main():
    df = load()
    X, y = feature_matrix(df)
    sections = df["section_id"].values
    print(f"rows {len(df):,}  features {X.shape[1]}", flush=True)

    # ---------------------------------------------------------- Part A
    rows = []
    protocols = {
        "random (leaky)": splits_random(df, n_splits=NFOLD, seed=SEED),
        "section-disjoint + forward-in-time":
            splits_temporal(df, cutoff=CUTOFF, n_splits=NFOLD, seed=SEED),
    }
    for pname, folds in protocols.items():
        for name, grid in GRIDS.items():
            for ci, cfg in enumerate(grid):
                s = [score(name, cfg, X, y, tr, te) for tr, te in folds]
                rows.append({"protocol": pname, "model": name, "config_id": ci,
                             "config": str(cfg), "R2": float(np.mean(s))})
            g = [r["R2"] for r in rows if r["protocol"] == pname and r["model"] == name]
            print(f"  {pname:36s} {name:13s} R2 {min(g):.4f}-{max(g):.4f} "
                  f"(spread {max(g)-min(g):.4f}, {len(g)} configs)", flush=True)
    sens = pd.DataFrame(rows)
    sens.to_csv("panel/tuning_sensitivity.csv", index=False)

    summ = sens.groupby(["protocol", "model"]).R2.agg(["min", "max", "mean"])
    summ["spread"] = summ["max"] - summ["min"]
    print("\n=== Part A: configuration spread vs protocol effect ===")
    print(summ.round(4).to_string())
    prot = sens.groupby(["protocol", "model"]).R2.max().unstack(0)
    prot["protocol_effect"] = prot.iloc[:, 0] - prot.iloc[:, 1]
    print("\nbest-config R2 under each protocol, and the protocol effect:")
    print(prot.round(4).to_string())

    # ---------------------------------------------------------- Part B
    print("\n=== Part B: nested section-disjoint tuning inside training data ===",
          flush=True)
    out = []
    for k, (tr, te) in enumerate(splits_temporal(df, cutoff=CUTOFF,
                                                 n_splits=NFOLD, seed=SEED)):
        inner = list(GroupKFold(n_splits=3).split(X.iloc[tr], y.iloc[tr],
                                                  groups=sections[tr]))
        for name, grid in GRIDS.items():
            best, best_s = None, -np.inf
            for cfg in grid:
                s = np.mean([score(name, cfg, X.iloc[tr].reset_index(drop=True)
                                   if False else X, y,
                                   tr[a], tr[b]) for a, b in inner])
                if s > best_s:
                    best, best_s = cfg, s
            r2 = score(name, best, X, y, tr, te)
            out.append({"fold": k, "model": name, "chosen": str(best),
                        "inner_R2": best_s, "test_R2": r2})
            print(f"  fold{k} {name:13s} test R2={r2:.4f}  chosen={best}", flush=True)
    tuned = pd.DataFrame(out)
    tuned.to_csv("panel/tuning_nested.csv", index=False)
    print("\n=== nested-tuned forward-in-time performance ===")
    print(tuned.groupby("model").test_R2.agg(["mean", "std"]).round(4)
          .sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main()
