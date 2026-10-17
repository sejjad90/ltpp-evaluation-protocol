'SUPERSEDED exploratory fold-wise inference. Its independence assumptions are unsupported; use run_cluster_stats.py for current comparisons.'

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ORDER = ["RandomForest", "LightGBM", "XGBoost", "CatBoost", "TabPFN-3"]
B = 10000
RNG = np.random.default_rng(0)


def folds():
    d = pd.concat([pd.read_csv("panel/robustness_trees.csv"),
                   pd.read_csv("panel/robustness_tabpfn.csv")], ignore_index=True)
    d["fold_id"] = (d.seed.astype(str) + "_" + d.cutoff.astype(str) + "_"
                    + d.fold.astype(str))
    keep = set(d[d.model == "TabPFN-3"].fold_id)
    return d[d.fold_id.isin(keep)]


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test."""
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return (rp - rm) / r.sum()


def cliffs_delta(x, y):
    """Cliff's delta: P(X>Y) - P(X<Y), computed on all pairs."""
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def magnitude(d):
    a = abs(d)
    return ("negligible" if a < 0.147 else "small" if a < 0.33
            else "medium" if a < 0.474 else "large")


def boot_ci(v, stat=np.mean, alpha=0.05):
    v = np.asarray(v)
    bs = np.array([stat(RNG.choice(v, len(v), replace=True)) for _ in range(B)])
    return np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def main():
    d = folds()
    rows = []
    for metric in ["R2", "coverage_90"]:
        p = d.pivot(index="fold_id", columns="model", values=metric)
        for m in [x for x in ORDER if x != "TabPFN-3"]:
            a, b = p[m].values, p["TabPFN-3"].values
            w = stats.wilcoxon(a, b)
            rb, cd = rank_biserial(a, b), cliffs_delta(a, b)
            lo, hi = boot_ci(a - b)
            rows.append({
                "metric": metric, "comparison": f"{m} vs TabPFN-3",
                "mean_diff": a.mean() - b.mean(),
                "ci_low": lo, "ci_high": hi,
                "wilcoxon_p": w.pvalue,
                "rank_biserial": rb, "cliffs_delta": cd,
                "magnitude": magnitude(cd),
                "wins": int((a > b).sum()), "n": len(a)})
    eff = pd.DataFrame(rows)
    eff.to_csv("panel/effect_sizes.csv", index=False)
    print("=== effect sizes (matched folds) ===")
    print(eff.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # per-model bootstrap CI on mean R2 and coverage
    ci = []
    for m in ORDER:
        g = d[d.model == m]
        for metric in ["R2", "coverage_90"]:
            lo, hi = boot_ci(g[metric].values)
            ci.append({"model": m, "metric": metric, "mean": g[metric].mean(),
                       "ci_low": lo, "ci_high": hi})
    ci = pd.DataFrame(ci)
    ci.to_csv("panel/bootstrap_ci.csv", index=False)
    print("\n=== bootstrap 95% CI over 27 matched folds ===")
    print(ci.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
