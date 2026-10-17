"""Figures added at review: calibration, importance, residuals, ablation."""

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


def fig7_calibration():
    d = pd.read_csv("panel/calibration_curve.csv")
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", linewidth=1.2, label="ideal")
    for m, g in d.groupby("model"):
        g = g.sort_values("nominal")
        ax.plot(g.nominal, g.empirical, "o-", color=C[m], label=m,
                linewidth=1.8, markersize=6)
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0.1, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig("figures/fig9_calibration.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig8_importance():
    d = pd.read_csv("panel/feature_importance.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.barh(d.feature, d.perm_importance, xerr=d.perm_sd, color="#0072B2",
            edgecolor="black", linewidth=0.4, error_kw={"linewidth": 0.8})
    ax.set_xscale("log")
    ax.set_xlabel("Permutation importance (decrease in $R^2$, log scale)")
    ax.tick_params(axis="y", labelsize=9)
    fig.savefig("figures/fig7_importance_perm_only.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig9_residuals():
    d = pd.read_csv("panel/residuals.csv")
    d = d[d.model == "CatBoost"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].scatter(d.y_pred, d.residual, s=6, alpha=0.25, color="#0072B2",
                    edgecolors="none")
    axes[0].axhline(0, color="black", linewidth=1.0)
    axes[0].set_xlabel("Predicted MRI (m/km)")
    axes[0].set_ylabel("Residual (m/km)")
    b = pd.cut(d.age, bins=[0, 5, 10, 15, 20, 25, 40])
    g = d.groupby(b, observed=True).residual
    mid = [iv.mid for iv in g.mean().index]
    axes[1].errorbar(mid, g.mean().values, yerr=g.std().values, marker="o",
                     capsize=3, color="#CC79A7", linewidth=1.6)
    axes[1].axhline(0, color="black", linewidth=1.0)
    axes[1].set_xlabel("Pavement age at survey (years)")
    axes[1].set_ylabel("Mean residual (m/km)")
    fig.tight_layout()
    fig.savefig("figures/fig10_residuals.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig10_ablation():
    d = pd.read_csv("panel/ablation_initial_mri.csv")
    order = ["random (leaky)", "section-disjoint",
             "section-disjoint + forward-in-time"]
    lab = ["Random\n(leaky)", "Section-\ndisjoint", "Section-disjoint\n+ forward-in-time"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    w, x = 0.35, np.arange(3)
    for i, (f, col) in enumerate([("with initial_MRI", "#0072B2"),
                                  ("without initial_MRI", "#E69F00")]):
        v = [d[(d.features == f) & (d.protocol == p)].R2.values[0] for p in order]
        ax.bar(x + (i - 0.5) * w, v, w, label=f.replace('initial_MRI','initial MRI'), color=col, edgecolor="black",
               linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel("Coefficient of determination, $R^2$")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.savefig("figures/fig8_ablation.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig7_calibration(); fig9_residuals(); fig10_ablation()   # importance now in make_figures3
    import os
    print("written:", sorted(f for f in os.listdir("figures") if f.startswith(("fig8", "fig9", "fig10"))))
