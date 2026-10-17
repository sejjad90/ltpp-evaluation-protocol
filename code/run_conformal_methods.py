'Compare residual-based interval calibration strategies on a shared later-survey test fold.'

import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from splits import (ALPHA, calibration_split, conformal_halfwidth,
                    conformal_interval, coverage, feature_matrix, load,
                    splits_temporal)

sys.stdout.reconfigure(encoding="utf-8")

SEEDS, CUTOFFS, NFOLD = [0, 1, 2], [2010, 2013, 2016], 3
LAMBDA = 0.25          # recency decay, per year
RECENT_YEARS = 5       # window for temporal recalibration


def weighted_quantile(resid, weights, alpha=ALPHA):
    """Smallest r with weighted mass >= 1-alpha, including the test point mass."""
    r = np.asarray(resid, float); w = np.asarray(weights, float)
    o = np.argsort(r); r, w = r[o], w[o]
    total = w.sum() + w.max()          # point mass for the unseen test residual
    c = np.cumsum(w) / total
    idx = np.searchsorted(c, 1 - alpha)
    return float(r[min(idx, len(r) - 1)])


def main():
    df = load()
    X, y = feature_matrix(df)
    year = df["VISIT_DATE"].dt.year.values
    rigid = (df.get("thk_PC", pd.Series(0.0, index=df.index)).fillna(0) > 0).values
    print(f"rows {len(df):,}  features {X.shape[1]}", flush=True)

    rows = []
    for seed in SEEDS:
        for cut in CUTOFFS:
            for k, (tr, te) in enumerate(splits_temporal(df, cutoff=cut,
                                                         n_splits=NFOLD, seed=seed)):
                cal, fit = calibration_split(tr, seed=seed)
                m = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=8,
                                      random_seed=0, verbose=0)
                m.fit(X.iloc[fit], y.iloc[fit])
                res_cal = np.abs(y.iloc[cal].values - m.predict(X.iloc[cal]))
                pred = m.predict(X.iloc[te])
                yt = y.iloc[te].values
                cal_year = year[cal]

                out = {}
                # 1. ordinary split conformal
                q = conformal_halfwidth(res_cal)
                lo, hi = conformal_interval(pred, q)
                out["split"] = (coverage(yt, lo, hi), float(np.mean(hi - lo)))

                # 2. recency-weighted
                w = np.exp(-LAMBDA * (cal_year.max() - cal_year))
                q = weighted_quantile(res_cal, w)
                lo, hi = conformal_interval(pred, q)
                out["weighted"] = (coverage(yt, lo, hi), float(np.mean(hi - lo)))

                # 3. temporal recalibration on the most recent years
                msk = cal_year >= (cal_year.max() - RECENT_YEARS + 1)
                if msk.sum() >= 100:
                    q = conformal_halfwidth(res_cal[msk])
                    lo, hi = conformal_interval(pred, q)
                    out["recent"] = (coverage(yt, lo, hi), float(np.mean(hi - lo)))

                # 4. Mondrian by pavement type
                covs, wids = [], []
                for grp in [True, False]:
                    cm, tm = rigid[cal] == grp, rigid[te] == grp
                    if cm.sum() >= 50 and tm.sum() >= 20:
                        q = conformal_halfwidth(res_cal[cm])
                        lo, hi = conformal_interval(pred[tm], q)
                        covs.append(coverage(yt[tm], lo, hi) * tm.sum())
                        wids.append(float(np.mean(hi - lo)) * tm.sum())
                if covs:
                    n = (rigid[te] == True).sum() + (rigid[te] == False).sum()
                    out["mondrian"] = (sum(covs) / n, sum(wids) / n)

                for meth, (cv, wd) in out.items():
                    rows.append({"seed": seed, "cutoff": cut, "fold": k,
                                 "method": meth, "coverage_90": cv,
                                 "width": wd, "n_test": len(te)})
                print(f"  seed{seed} cut{cut} f{k}: " +
                      "  ".join(f"{m}={v[0]:.3f}" for m, v in out.items()), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv("panel/conformal_methods.csv", index=False)
    s = d.groupby("method")[["coverage_90", "width"]].agg(["mean", "std"]).round(4)
    print("\n=== conformal methods under forward-in-time evaluation (nominal 0.90) ===")
    print(s.to_string())


if __name__ == "__main__":
    main()
