"""Figure 1 - study workflow, drawn to publication typography.

Counts shown are read from the assembled panel so the diagram cannot drift out
of step with the data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "Times New Roman", "figure.facecolor": "white"})

from splits import load as load_modelled

if Path('panel/ltpp_panel.parquet').exists():
    df=pd.read_parquet('panel/ltpp_panel.parquet');_m=load_modelled()
    n_rows,n_sec,n_unit=len(df),df.section_id.nunique(),df.unit_id.nunique()
    model_rows,model_sec=len(_m),_m.section_id.nunique()
else:
    s=pd.read_csv('panel/panel_summary.csv').set_index('key').value
    n_rows,n_sec,n_unit=[int(s[k]) for k in ['n_panel','n_sections','n_units']]
    model_rows,model_sec=[int(s[k]) for k in ['n_model','n_model_sec']]

BLUE, GREY, EDGE = "#CFE2F3", "#E8E8E8", "#3C3C3C"

fig, ax = plt.subplots(figsize=(7.0, 8.6))
ax.set_xlim(0, 10); ax.set_ylim(-0.35, 15.4); ax.axis("off")


def box(x, y, w, h, text, fc, fs=10.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                fc=fc, ec=EDGE, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.45)


def arrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=EDGE, linewidth=1.1))


box(0.5, 13.4, 9.0, 1.6,
    "LTPP Standard Data Release 39 (FHWA)\n"
    "20 modules: structure and materials, climate, traffic, performance", BLUE, 10.5)
arrow(5, 13.35, 12.65)

box(0.5, 11.0, 9.0, 1.6,
    "Panel assembly keyed on STATE_CODE + SHRP_ID + CONSTRUCTION_NO\n"
    f"{n_rows:,} section-visits | {n_sec:,} sections | {n_unit:,} construction events", BLUE, 10.5)
arrow(5, 10.95, 10.25)

box(0.5, 8.6, 9.0, 1.6,
    "Feature construction\nage, layers, annual traffic exposure, trailing-window climate\n"
    "(nearest virtual weather station, distance retained)", BLUE, 10.5)
arrow(5, 8.55, 7.85)

box(0.5, 6.6, 9.0, 1.2,
    f"Modelled set: {model_rows:,} section-visits across {model_sec:,} sections\n"
    "(first visit of each construction event excluded: it defines initial roughness)",
    BLUE, 10.0)
arrow(5, 6.55, 5.55)

ax.text(5, 5.25, "Evaluation protocols", ha="center", fontsize=10.5, weight="bold")
box(0.4, 3.5, 2.9, 1.5, "Random\nK-fold\n(leakage\nbaseline)", GREY, 9.8)
box(3.55, 3.5, 2.9, 1.5, "Section-\ndisjoint\nK-fold", GREY, 9.8)
box(6.7, 3.5, 2.9, 1.5, "Section-disjoint\n+ forward-\nin-time", GREY, 9.8)
arrow(5, 3.45, 2.75)

box(0.5, 1.4, 9.0, 1.3,
    ("Four tree learners" if os.environ.get('LTPP_TREES_ONLY')=='1' else "Five learners: trees and TabPFN-3")+"\n"
    "Split-conformal 90% prediction intervals on every model", BLUE, 10.2)
arrow(5, 1.35, 0.75)

box(0.5, 0.0, 9.0, 0.7,
    "Paired comparison over 27 matched folds (3 seeds x 3 cut-offs x 3 folds)",
    GREY, 10.0)

fig.savefig("figures/fig1_workflow.png", dpi=300, bbox_inches="tight", facecolor="white")
print(f"fig1 written: rows={n_rows:,} sections={n_sec:,} units={n_unit:,} modelled={model_rows:,}")
