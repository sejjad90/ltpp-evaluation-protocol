'Plot training-regime contrasts, interval calibration strategies and feature attribution.'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11, "axes.titlesize": 0,
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.35, "figure.facecolor": "white",
})

C = {"RandomForest": "#0072B2", "LightGBM": "#009E73", "XGBoost": "#E69F00",
     "CatBoost": "#CC79A7", "TabPFN-3": "#D55E00"}
ORDER = ["RandomForest", "LightGBM", "XGBoost", "CatBoost", "TabPFN-3"]


def fig7_importance():
    """Permutation importance beside mean absolute SHAP."""
    d = pd.read_csv("panel/feature_importance.csv").head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2), sharey=True)
    d['feature']=d.feature.replace({'initial_MRI':'initial MRI'})
    axes[0].barh(d.feature, d.perm_importance, xerr=d.perm_sd, color="#0072B2",
                 edgecolor="black", linewidth=0.4, error_kw={"linewidth": 0.8})
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Permutation importance\n(decrease in $R^2$, log scale)")
    axes[1].barh(d.feature, d.mean_abs_shap, color="#CC79A7",
                 edgecolor="black", linewidth=0.4)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Mean absolute SHAP value\n(m/km, log scale)")
    axes[0].tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig7_importance.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig11_decomposition():
    s = pd.read_csv("panel/decomposition_summary_all.csv")
    models = [m for m in ORDER if m in set(s.model)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x, w = np.arange(len(models)), 0.36
    for i, (comp, lab, col) in enumerate([
            ("d_repeat", "remove other visits of test sections", "#0072B2"),
            ("d_shift", "remove post-cut-off training rows", "#E69F00")]):
        v = [s[(s.model == m) & (s.component == comp)]["mean"].values[0] for m in models]
        lo = [v[j] - s[(s.model == m) & (s.component == comp)].ci_low.values[0]
              for j, m in enumerate(models)]
        hi = [s[(s.model == m) & (s.component == comp)].ci_high.values[0] - v[j]
              for j, m in enumerate(models)]
        ax.bar(x + (i - 0.5) * w, v, w, yerr=[lo, hi], capsize=3, label=lab,
               color=col, edgecolor="black", linewidth=0.4,
               error_kw={"linewidth": 0.9})
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylabel("Training-regime contrast in $R^2$")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.savefig("figures/fig11_decomposition.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig12_conformal_methods():
    d = pd.read_csv("panel/conformal_methods.csv")
    lab = {"split": "split\n(baseline)", "weighted": "recency-\nweighted",
           "recent": "temporal\nrecalibration", "mondrian": "Mondrian\n(by family)"}
    order = ["split", "weighted", "recent", "mondrian"]
    g = d.groupby("method")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    m = [g.coverage_90.mean()[k] for k in order]
    e = [g.coverage_90.std()[k] for k in order]
    axes[0].bar(range(4), m, 0.6, yerr=e, capsize=3, color="#0072B2",
                edgecolor="black", linewidth=0.4, error_kw={"linewidth": 0.9})
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.2)
    axes[0].text(3.42, 0.905, "nominal 90%", fontsize=9, ha="right", va="bottom")
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels([lab[k] for k in order], fontsize=9)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0.4, 1.0)
    w = [g.width.mean()[k] for k in order]
    we = [g.width.std()[k] for k in order]
    axes[1].bar(range(4), w, 0.6, yerr=we, capsize=3, color="#CC79A7",
                edgecolor="black", linewidth=0.4, error_kw={"linewidth": 0.9})
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels([lab[k] for k in order], fontsize=9)
    axes[1].set_ylabel("Mean interval width (m/km)")
    fig.tight_layout()
    fig.savefig("figures/fig12_conformal_methods.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import os
    fig7_importance(); fig11_decomposition(); fig12_conformal_methods()
    print("written:", sorted(f for f in os.listdir("figures")
                             if f.startswith(("fig7", "fig11", "fig12"))))
