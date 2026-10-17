'Evaluate TabPFN-3 on fixed-test training regimes and internal geographic holdouts.'

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from splits import feature_matrix, load

sys.stdout.reconfigure(encoding="utf-8")

SEEDS, CUTOFF, NFOLD = [0, 1, 2], 2013, 3


def make():
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(n_estimators=8, auto_scale_n_estimators=False,
                           random_state=0, ignore_pretraining_limits=True)


def fit_score(X, y, tr, te):
    m = make()
    m.fit(X.iloc[tr], y.iloc[tr])
    p = m.predict(X.iloc[te])
    return r2_score(y.iloc[te], p), mean_absolute_error(y.iloc[te], p)


def main():
    df = load()
    X, y = feature_matrix(df)
    year = df["VISIT_DATE"].dt.year.values
    sec = df["section_id"].values
    print(f"rows {len(df):,}  features {X.shape[1]}", flush=True)

    # ---------------- decomposition, same folds as run_decomposition.py ------
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
            t0 = time.time()
            r = {"seed": seed, "fold": k, "model": "TabPFN-3", "n_test": len(te),
                 "leaky": fit_score(X, y, tr_leaky, te)[0],
                 "sect_disj": fit_score(X, y, tr_sd, te)[0],
                 "forward": fit_score(X, y, tr_fw, te)[0]}
            r["d_repeat"] = r["leaky"] - r["sect_disj"]
            r["d_shift"] = r["sect_disj"] - r["forward"]
            rows.append(r)
            print(f"  decomp seed{seed} f{k}: leaky={r['leaky']:.4f} "
                  f"sd={r['sect_disj']:.4f} fwd={r['forward']:.4f} "
                  f"repeat={r['d_repeat']:+.4f} shift={r['d_shift']:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv("panel/decomposition_tabpfn.csv", index=False)

    # ---------------- additional designs, same as run_review_experiments -----
    out = []
    units = df["unit_id"].unique().copy()
    np.random.default_rng(0).shuffle(units)
    for k, part in enumerate(np.array_split(units, 3)):
        te_mask = np.isin(df["unit_id"].values, part)
        tr = np.where(~te_mask & (year <= CUTOFF))[0]
        te = np.where(te_mask & (year > CUTOFF))[0]
        if len(tr) < 100 or len(te) < 50:
            continue
        t0 = time.time()
        r2, mae = fit_score(X, y, tr, te)
        out.append({"design": "construction-event disjoint", "model": "TabPFN-3",
                    "fold": k, "R2": r2, "MAE": mae, "n_test": len(te)})
        print(f"  event-disjoint f{k}: R2={r2:.4f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(out).to_csv("panel/review_designs_tabpfn.csv", index=False)

    counts = df.groupby("STATE_CODE").size().sort_values(ascending=False)
    for st in counts.head(12).index:
        te = np.where(df["STATE_CODE"].values == st)[0]
        tr = np.where(df["STATE_CODE"].values != st)[0]
        if len(te) < 100:
            continue
        t0 = time.time()
        r2, mae = fit_score(X, y, tr, te)
        out.append({"design": "leave-one-state-out", "model": "TabPFN-3",
                    "fold": int(st), "R2": r2, "MAE": mae, "n_test": len(te)})
        print(f"  LOSO state{st}: R2={r2:.4f} n={len(te)} ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(out).to_csv("panel/review_designs_tabpfn.csv", index=False)

    d = pd.DataFrame(rows)
    tot = d.d_repeat.mean() + d.d_shift.mean()
    print(f"\nTabPFN-3 decomposition: repeat {d.d_repeat.mean():+.4f} "
          f"({100*d.d_repeat.mean()/tot:.0f}%)  shift {d.d_shift.mean():+.4f}")
    o = pd.DataFrame(out)
    print(o.groupby("design").R2.agg(["mean", "std", "size"]).round(4).to_string())


if __name__ == "__main__":
    main()
