"""Publication figures for the LTPP evaluation-protocol study.

All figures are drawn from the measured result CSVs. Nothing is illustrative.
No titles are embedded in the figures; captions carry the description.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import os

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 0,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.facecolor": "white",
})

# colour-blind-safe (Okabe-Ito), consistent across every figure
C = {"RandomForest": "#0072B2", "LightGBM": "#009E73", "XGBoost": "#E69F00",
     "CatBoost": "#CC79A7", "TabPFN-3": "#D55E00"}
ORDER = ["RandomForest", "LightGBM", "XGBoost", "CatBoost", "TabPFN-3"]
PROTO = ["random (leaky)", "section-disjoint", "section-disjoint + forward-in-time"]
PLAB = ["Random\n(leaky)", "Section-\ndisjoint", "Section-disjoint\n+ forward-in-time"]


def load_all():
    a = pd.read_csv("panel/comparator_results.csv")
    b = pd.read_csv("panel/tabpfn3_results.csv") if Path('panel/tabpfn3_results.csv').exists() and os.environ.get('LTPP_TREES_ONLY')!='1' else a.iloc[:0].copy()
    return pd.concat([a, b], ignore_index=True)


def load_folds():
    t = pd.read_csv("panel/robustness_trees.csv")
    b = pd.read_csv("panel/robustness_tabpfn.csv") if Path('panel/robustness_tabpfn.csv').exists() and os.environ.get('LTPP_TREES_ONLY')!='1' else t.iloc[:0].copy()
    d = pd.concat([t, b], ignore_index=True)
    d["fold_id"] = d.seed.astype(str) + "_" + d.cutoff.astype(str) + "_" + d.fold.astype(str)
    done = set(b.seed.astype(str) + "_" + b.cutoff.astype(str) + "_" + b.fold.astype(str))
    return d[d.fold_id.isin(done)].copy() if len(b) else d


def fig_protocol_r2(d):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w, x = 0.16, np.arange(len(PROTO))
    for i, m in enumerate(ORDER):
        v = [d[(d.model == m) & (d.protocol == p)].R2.values[0] for p in PROTO]
        ax.bar(x + (i - 2) * w, v, w, label=m, color=C[m], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(PLAB)
    ax.set_ylabel("Coefficient of determination, $R^2$")
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    fig.savefig("figures/fig2_protocol_r2.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_rank_inversion(d):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ranks = {}
    for p in PROTO:
        s = d[d.protocol == p].set_index("model").R2
        ranks[p] = s.rank(ascending=False)
    for m in ORDER:
        ax.plot(range(len(PROTO)), [ranks[p][m] for p in PROTO], "o-",
                color=C[m], label=m, linewidth=2, markersize=7)
    ax.set_xticks(range(len(PROTO))); ax.set_xticklabels(PLAB)
    ax.set_ylabel("Rank by $R^2$ (1 = best)")
    ax.set_yticks([1, 2, 3, 4, 5]); ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig("figures/fig3_rank_inversion.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_coverage(d):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w, x = 0.16, np.arange(len(PROTO))
    for i, m in enumerate(ORDER):
        v = [d[(d.model == m) & (d.protocol == p)].coverage_90.values[0] for p in PROTO]
        ax.bar(x + (i - 2) * w, v, w, label=m, color=C[m], edgecolor="black", linewidth=0.4)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.2)
    ax.text(2.42, 0.905, "nominal 90%", fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(PLAB)
    ax.set_ylabel("Empirical coverage of 90% intervals")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, -0.36))
    fig.savefig("figures/fig4_coverage.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_paired(f):
    p = f.pivot(index="fold_id", columns="model", values="R2")
    diff = (p["CatBoost"] - p["TabPFN-3"]).values
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(diff, bins=12, color="#0072B2", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(diff.mean(), color="#D55E00", linewidth=2)
    cs = pd.read_csv('panel/cluster_stats.csv')
    blocked_p = cs[(cs.metric=='R2') & (cs.comparison=='CatBoost vs TabPFN-3')].p_perm_blocked.iloc[0]
    ax.text(0.97, 0.95, f"mean $\\Delta R^2$ = {diff.mean():+.4f}\n"
                        f"CatBoost better in {int((diff>0).sum())}/{len(diff)} folds\n"
                        f"Blocked permutation $p$ = {blocked_p:.4f} (unadjusted)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))
    ax.set_xlabel("Paired difference in $R^2$ (CatBoost $-$ TabPFN-3)")
    ax.set_ylabel("Number of folds")
    fig.savefig("figures/fig5_paired_difference.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_cutoff(f):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in ORDER:
        g = f[f.model == m].groupby("cutoff").R2
        mu, sd = g.mean(), g.std()
        ax.errorbar(mu.index, mu.values, yerr=sd.values, marker="o", capsize=3,
                    color=C[m], label=m, linewidth=1.6, markersize=6)
    ax.set_xlabel("Temporal cut-off year")
    ax.set_ylabel("Coefficient of determination, $R^2$")
    ax.set_xticks(sorted(f.cutoff.unique()))
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.savefig("figures/fig6_cutoff_sensitivity.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    d, f = load_all(), load_folds()
    ORDER = [m for m in ORDER if m in set(d.model)]
    fig_protocol_r2(d); fig_rank_inversion(d); fig_coverage(d)
    if 'TabPFN-3' in set(f.model):fig_paired(f)
    else:print('Paired CatBoost/TabPFN histogram omitted: no TabPFN data.')
    fig_cutoff(f)
    print("figures written:", sorted(__import__("os").listdir("figures")))
