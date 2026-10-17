'Compare three training regimes on fixed section-disjoint later-survey test rows.'

import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from splits import feature_matrix, load

sys.stdout.reconfigure(encoding="utf-8")

SEEDS, CUTOFF, NFOLD = [0, 1, 2], 2013, 3
B = 10000


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


def fit_score(name, X, y, tr, te):
    m = models()[name]()
    Xf, Xt = X.iloc[tr], X.iloc[te]
    if name == "RandomForest":
        med = Xf.median(); Xf, Xt = Xf.fillna(med), Xt.fillna(med)
    m.fit(Xf, y.iloc[tr])
    return r2_score(y.iloc[te], m.predict(Xt))


def main():
    df = load()
    X, y = feature_matrix(df)
    year = df["VISIT_DATE"].dt.year.values
    sec = df["section_id"].values
    print(f"rows {len(df):,}  features {X.shape[1]}  cut-off {CUTOFF}", flush=True)

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        uniq = np.asarray(pd.unique(df["section_id"]), dtype=object)
        rng.shuffle(uniq)
        for k, part in enumerate(np.array_split(uniq, NFOLD)):
            is_test_sec = np.isin(sec, part)
            te = np.where(is_test_sec & (year > CUTOFF))[0]
            if len(te) < 100:
                continue
            tr_leaky = np.setdiff1d(np.arange(len(df)), te)
            tr_sd = np.where(~is_test_sec)[0]
            tr_fw = np.where(~is_test_sec & (year <= CUTOFF))[0]
            for name in models():
                r = {"seed": seed, "fold": k, "model": name, "n_test": len(te),
                     "leaky": fit_score(name, X, y, tr_leaky, te),
                     "sect_disj": fit_score(name, X, y, tr_sd, te),
                     "forward": fit_score(name, X, y, tr_fw, te)}
                r["d_repeat"] = r["leaky"] - r["sect_disj"]     # same-section info
                r["d_shift"] = r["sect_disj"] - r["forward"]    # temporal shift
                rows.append(r)
                print(f"  seed{seed} f{k} {name:13s} leaky={r['leaky']:.4f} "
                      f"sd={r['sect_disj']:.4f} fwd={r['forward']:.4f} "
                      f"| repeat={r['d_repeat']:+.4f} shift={r['d_shift']:+.4f}",
                      flush=True)

    d = pd.DataFrame(rows)
    d.to_csv("panel/decomposition_folds.csv", index=False)

    rng = np.random.default_rng(0)
    out = []
    for name, g in d.groupby("model"):
        for comp in ["d_repeat", "d_shift"]:
            v = g[comp].values
            bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(B)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append({"model": name, "component": comp, "mean": v.mean(),
                        "ci_low": lo, "ci_high": hi, "n_folds": len(v)})
    s = pd.DataFrame(out)
    s.to_csv("panel/decomposition_summary.csv", index=False)
    print("\n=== decomposition of the total decline (fixed test set) ===")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nfirst training-regime contrast as a share of the total descriptive decline:")
    for name, g in d.groupby("model"):
        tot = g.d_repeat.mean() + g.d_shift.mean()
        print(f"  {name:13s} {100*g.d_repeat.mean()/tot:5.1f}%  "
              f"(repeat {g.d_repeat.mean():+.4f}, shift {g.d_shift.mean():+.4f})")


if __name__ == "__main__":
    main()
