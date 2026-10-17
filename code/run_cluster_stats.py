'Compute block-bootstrap intervals, blocked sign-flip tests and multiplicity adjustments.'

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

B = 10000
REFERENCE = "TabPFN-3"
ORDER = ["RandomForest", "LightGBM", "XGBoost", "CatBoost", "TabPFN-3"]


def load_grid():
    d = pd.concat([pd.read_csv("panel/robustness_trees.csv"),
                   pd.read_csv("panel/robustness_tabpfn.csv")], ignore_index=True)
    d["fold_id"] = (d.seed.astype(str) + "_" + d.cutoff.astype(str) + "_"
                    + d.fold.astype(str))
    d["block"] = d.seed.astype(str) + "_" + d.cutoff.astype(str)
    keep = set(d[d.model == REFERENCE].fold_id)
    return d[d.fold_id.isin(keep)].copy()


def rank_biserial(diff):
    d = np.asarray(diff, float); d = d[d != 0]
    if not len(d):
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cluster_bootstrap(diff, blocks, rng, B=B):
    ub = np.unique(blocks)
    out = np.empty(B)
    for i in range(B):
        pick = rng.choice(ub, len(ub), replace=True)
        out[i] = np.concatenate([diff[blocks == b] for b in pick]).mean()
    return np.percentile(out, [2.5, 97.5])


def blocked_permutation(diff, blocks, rng, B=B):
    """Sign-flip whole blocks; two-sided p for the mean paired difference.

    With n blocks the null has 2**n sign assignments. For n <= 20 these are
    enumerated exhaustively, which makes the test exact and reproducible; above
    that the null is sampled. An earlier version always sampled, which produced
    p-values that did not lie on the attainable 2/2**n grid.
    """
    import itertools

    ub = np.unique(blocks)
    n = len(ub)
    parts = [diff[blocks == b] for b in ub]
    obs = abs(np.concatenate(parts).mean())

    if n <= 20:                                   # exact enumeration
        ge = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            m = np.concatenate([p * s for p, s in zip(parts, signs)]).mean()
            ge += abs(m) >= obs - 1e-12
        return ge / 2 ** n

    ge = 0                                        # Monte Carlo fallback
    for _ in range(B):
        signs = rng.choice([-1.0, 1.0], n)
        m = np.concatenate([p * s for p, s in zip(parts, signs)]).mean()
        ge += abs(m) >= obs - 1e-12
    return (ge + 1) / (B + 1)


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (n - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def benjamini_hochberg(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = o[i]
        prev = min(prev, p[idx] * n / (i + 1))
        adj[idx] = min(1.0, prev)
    return adj


def main():
    d = load_grid()
    nblk = d.block.nunique()
    print(f"folds {d.fold_id.nunique()}  blocks {nblk}  "
          f"folds per block {d.fold_id.nunique() // nblk}\n")

    rng = np.random.default_rng(0)
    rows = []
    for metric in ["R2", "coverage_90"]:
        piv = d.pivot(index="fold_id", columns="model", values=metric)
        blk = d.drop_duplicates("fold_id").set_index("fold_id").loc[piv.index, "block"].values
        for m in [x for x in ORDER if x != REFERENCE]:
            diff = (piv[m] - piv[REFERENCE]).values
            lo, hi = cluster_bootstrap(diff, blk, rng)
            rows.append({
                "metric": metric, "comparison": f"{m} vs {REFERENCE}",
                "mean_diff": diff.mean(),
                "median_diff": float(np.median(diff)),
                "ci_low": lo, "ci_high": hi,
                "p_perm_blocked": blocked_permutation(diff, blk, rng),
                "p_wilcoxon_naive": stats.wilcoxon(piv[m], piv[REFERENCE]).pvalue,
                "rank_biserial": rank_biserial(diff),
                "wins": int((diff > 0).sum()), "n": len(diff)})

    res = pd.DataFrame(rows)
    for metric, g in res.groupby("metric"):
        res.loc[g.index, "p_holm"] = holm(g.p_perm_blocked.values)
        res.loc[g.index, "p_bh"] = benjamini_hochberg(g.p_perm_blocked.values)
    res.to_csv("panel/cluster_stats.csv", index=False)

    print("=== cluster-aware paired comparisons against " + REFERENCE + " ===")
    cols = ["metric", "comparison", "mean_diff", "ci_low", "ci_high",
            "p_perm_blocked", "p_holm", "p_bh", "rank_biserial", "wins", "n"]
    print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== effect of ignoring clustering (naive vs blocked p) ===")
    print(res[["metric", "comparison", "p_wilcoxon_naive", "p_perm_blocked",
               "p_holm"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sig = res[res.p_holm < 0.05]
    print(f"\nsurviving Holm adjustment at 0.05: {len(sig)} of {len(res)}")
    for _, r in sig.iterrows():
        print(f"  {r.metric:12s} {r.comparison:28s} "
              f"diff={r.mean_diff:+.4f} p_holm={r.p_holm:.4f}")


if __name__ == "__main__":
    main()
