'Format measured result summaries, CSV panel diagnostics, design settings and bibliographic/environment metadata.'

from pathlib import Path
import json

import numpy as np
import pandas as pd

from splits import CAL_FRAC, CAL_MAX, feature_matrix, load

PROTO = ["random (leaky)", "section-disjoint", "section-disjoint + forward-in-time"]
ORDER = ["RandomForest", "LightGBM", "XGBoost", "CatBoost", "TabPFN-3"]
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}

# values from the superseded, look-ahead-contaminated panel, retained so the
# effect of the correction can be quoted (Section 4.1)



def F(v, n=4):
    return f"{v:.{n}f}"


def _holm_all8(cs):
    """Largest Holm-adjusted p among the comparisons that survive, had a single
    family of all tests been used instead of one family per metric."""
    import numpy as _np
    order = _np.argsort(cs.p_perm_blocked.values)
    p = cs.p_perm_blocked.values[order]
    n = len(p)
    adj, run = _np.empty(n), 0.0
    for i in range(n):
        run = max(run, (n - i) * p[i])
        adj[i] = min(1.0, run)
    restored = _np.empty(n)
    restored[order] = adj
    keep = restored[restored < 0.05]
    return float(keep.max()) if len(keep) else float("nan")


def build():
    available = Path('panel/ltpp_panel.parquet').exists()
    mod = load() if available else None
    X = feature_matrix(mod)[0] if available else None
    summary = pd.read_csv('panel/panel_summary.csv',dtype={'value':str}).set_index('key').value
    raw_summary = pd.read_csv('panel/raw_summary.csv').set_index('key').value
    def S(key): return float(summary[key])
    legacy = Path('results/legacy') if Path('results/legacy').exists() else Path('panel/legacy')
    historical = pd.read_csv(legacy/'all_results.csv')
    historical = historical[historical.model=='CatBoost'].set_index('protocol')
    old_effects = pd.read_csv(legacy/'effect_sizes.csv')
    old_effects = old_effects[(old_effects.comparison=='CatBoost vs TabPFN-3') & (old_effects.metric=='R2')].iloc[0]
    pre_annual = pd.read_csv(legacy/'pre_annual_comparator_results.csv')
    pre_annual_cat = pre_annual[(pre_annual.model=='CatBoost') & (pre_annual.protocol==PROTO[2])].R2.iloc[0]

    main = pd.concat([pd.read_csv("panel/comparator_results.csv"),
                      pd.read_csv("panel/tabpfn3_results.csv")], ignore_index=True)
    grid = pd.concat([pd.read_csv("panel/robustness_trees.csv"),
                      pd.read_csv("panel/robustness_tabpfn.csv")], ignore_index=True)
    grid["fold_id"] = (grid.seed.astype(str) + "_" + grid.cutoff.astype(str)
                       + "_" + grid.fold.astype(str))
    grid["block"] = grid.seed.astype(str) + "_" + grid.cutoff.astype(str)
    keep = set(grid[grid.model == "TabPFN-3"].fold_id)
    grid = grid[grid.fold_id.isin(keep)]

    decs = pd.read_csv("panel/decomposition_summary_all.csv")
    cs = pd.read_csv("panel/cluster_stats.csv")
    cm = pd.read_csv("panel/conformal_methods.csv")
    cost = pd.read_csv("panel/computational_cost.csv")
    het = pd.read_csv("panel/heterogeneity.csv")
    des = pd.read_csv("panel/review_designs_all.csv")
    cal = pd.read_csv("panel/calibration_curve.csv")
    imp = pd.read_csv("panel/feature_importance.csv")
    abl = pd.read_csv("panel/ablation_initial_mri.csv")
    incn = pd.read_csv("panel/inclusion_numeric.csv")
    incs = pd.read_csv("panel/inclusion_structural.csv")
    tsens = pd.read_csv("panel/tuning_sensitivity.csv")
    tnest = pd.read_csv("panel/tuning_nested.csv")
    env = json.loads((Path("panel/environment.json") if Path("panel/environment.json").exists() else Path("results/environment.json")).read_text("utf-8"))
    refs = json.loads((Path("docs/references.json") if Path("docs/references.json").exists() else Path("refs/references.json")).read_text("utf-8"))

    def g(model, proto, col):
        return main[(main.model == model) & (main.protocol == proto)][col].values[0]

    def c(metric, comp, col):
        r = cs[(cs.metric == metric) & (cs.comparison == comp)]
        return r[col].values[0]

    def ab(feat, proto):
        return abl[(abl.features == feat) & (abl.protocol == proto)].R2.values[0]

    r2 = main.pivot(index="model", columns="protocol", values="R2")
    cov = main.pivot(index="model", columns="protocol", values="coverage_90")
    rank_t = r2[PROTO[2]].rank(ascending=False)
    grank = grid.groupby("model").R2.mean().rank(ascending=False)
    gmean = grid.groupby("model").R2.mean()

    # month share before October, for the look-ahead exposure statement
    pre_oct = S("pre_oct_pct")

    components = decs.pivot(index="model", columns="component", values="mean")
    dshare = 100 * components.d_repeat / components.sum(axis=1)
    loso = des[des.design == "leave-one-state-out"]
    loso_tab = loso[loso.model == "TabPFN-3"]
    worst_state = int(loso_tab.loc[loso_tab.R2.idxmin(), "fold"])

    paired_states = loso.pivot(index="fold", columns="model", values="R2")

    cmm = cm.groupby("method")[["coverage_90", "width"]].mean()

    # residual behaviour, by quintile of predicted roughness (Section 4.9)
    res = pd.read_csv("panel/residuals.csv")
    res = res[res.model == "CatBoost"]
    rq = res.groupby(pd.qcut(res.y_pred, 5, duplicates="drop"),
                     observed=True).residual.mean()
    zone = het[(het.model == "CatBoost") & (het.stratum_type == "climatic regime")]
    fam = het[(het.model == "CatBoost") & (het.stratum_type == "pavement family")]

    tfw = tsens[tsens.protocol == PROTO[2]]
    spread = tfw.groupby("model").R2.agg(lambda v: v.max() - v.min())
    best_by_proto = tsens.groupby(["protocol", "model"]).R2.max().unstack(0)
    prot_eff = (best_by_proto[PROTO[0]] - best_by_proto[PROTO[2]]).mean()
    tn = tnest.groupby("model").test_R2.mean().sort_values(ascending=False)

    tree_tot = cost[cost.model.isin(["CatBoost","LightGBM","XGBoost"])].total_s
    tab = cost[cost.model == "TabPFN-3"].iloc[0]
    cat = cost[cost.model == "CatBoost"].iloc[0]

    N = {
        # ---- panel -------------------------------------------------------
        "n_panel": f"{int(S("n_panel")):,}", "n_sections": f"{int(S("n_sections")):,}",
        "n_units": f"{int(S("n_units")):,}",
        "n_model": f"{int(S("n_model")):,}", "n_model_sec": f"{int(S("n_model_sec")):,}",
        "n_features": str(int(S("n_features"))),
        "date_min": summary["date_min"],
        "date_max": summary["date_max"],
        "clim_dist": F(S("clim_dist"), 1),
        "clim_onsite": f"{S('clim_onsite'):.0f}",
        "pre_oct_pct": f"{pre_oct:.0f}",
        "n_excl_first": f"{int(S("n_excl_first")):,}",
        "n_states": str(int(S("n_states"))),
        "inc_state_gap": F(S("inc_state_gap"),3),
        "inc_age_in": F(incn[incn.variable == "age_years"].included_mean.values[0], 1),
        "inc_age_ex": F(incn[incn.variable == "age_years"].excluded_mean.values[0], 1),
        "inc_age_sd": F(incn[incn.variable == "age_years"].std_diff.values[0], 2),

        # ---- main protocol comparison ------------------------------------
        "tab_rand_r2": F(g("TabPFN-3", PROTO[0], "R2")),
        "tab_temp_r2": F(g("TabPFN-3", PROTO[2], "R2")),
        "tab_rand_mae": F(g("TabPFN-3", PROTO[0], "MAE"), 3),
        "tab_temp_mae": F(g("TabPFN-3", PROTO[2], "MAE"), 3),
        "tab_drop": F(g("TabPFN-3", PROTO[0], "R2") - g("TabPFN-3", PROTO[2], "R2"), 3),
        "tab_temp_cov": F(g("TabPFN-3", PROTO[2], "coverage_90"), 3),
        "tab_temp_rank": ORDINAL[int(rank_t["TabPFN-3"])],
        "tab_grid_rank": ORDINAL[int(grank["TabPFN-3"])],
        "tab_grid_r2": F(gmean["TabPFN-3"]),
        "cat_grid_r2": F(gmean["CatBoost"]),
        "xgb_grid_r2": F(gmean["XGBoost"]),
        "xgb_grid_sd": F(grid[grid.model == "XGBoost"].R2.std(), 3),
        "cat_rand_r2": F(g("CatBoost", PROTO[0], "R2")),
        "cat_temp_r2": F(g("CatBoost", PROTO[2], "R2")),
        "rf_temp_cov": F(g("RandomForest", PROTO[2], "coverage_90"), 3),
        "rf_temp_rank": ORDINAL[int(rank_t["RandomForest"])],
        "qin_gap": F(abs(g("TabPFN-3", PROTO[0], "R2") - 0.947), 3),
        "drop_min": F((r2[PROTO[0]] - r2[PROTO[1]]).min(), 3),
        "drop_max": F((r2[PROTO[0]] - r2[PROTO[1]]).max(), 3),
        "cov_rand_min": F(cov[PROTO[0]].min(), 3), "cov_rand_max": F(cov[PROTO[0]].max(), 3),
        "cov_temp_min": F(cov[PROTO[2]].min(), 3), "cov_temp_max": F(cov[PROTO[2]].max(), 3),
        "old_cat_temp": F(historical.loc[PROTO[2],"R2"]), "old_cat_rand": F(historical.loc[PROTO[0],"R2"]),

        # ---- grid and cluster-aware statistics ---------------------------
        "n_folds": str(grid.fold_id.nunique()),
        "n_blocks": str(grid.block.nunique()),
        "n_seeds": str(grid.seed.nunique()), "n_cutoffs": str(grid.cutoff.nunique()),
        "n_splits": str(grid.fold.nunique()),
        "grid_ntest": f"{grid.n_test.mean():.0f}",
        "grid_r2_min": F(gmean.min()), "grid_r2_max": F(gmean.max()),
        "grid_cat_r2": F(gmean["CatBoost"]), "grid_tab_r2": F(gmean["TabPFN-3"]),
        "boot_B": "10,000",
        "cs_r2_cat_diff": F(c("R2", "CatBoost vs TabPFN-3", "mean_diff")),
        "cs_r2_cat_lo": F(c("R2", "CatBoost vs TabPFN-3", "ci_low")),
        "cs_r2_cat_hi": F(c("R2", "CatBoost vs TabPFN-3", "ci_high")),
        "cs_r2_cat_p": F(c("R2", "CatBoost vs TabPFN-3", "p_perm_blocked"), 4),
        "cs_cov_cat_diff": F(c("coverage_90", "CatBoost vs TabPFN-3", "mean_diff")),
        "cs_cov_cat_lo": F(c("coverage_90", "CatBoost vs TabPFN-3", "ci_low")),
        "cs_cov_cat_hi": F(c("coverage_90", "CatBoost vs TabPFN-3", "ci_high")),
        "cs_cov_cat_wins": str(int(c("coverage_90", "CatBoost vs TabPFN-3", "wins"))),
        "cs_cov_cat_holm": F(c("coverage_90", "CatBoost vs TabPFN-3", "p_holm"), 4),
        "cs_cov_rf_diff": F(c("coverage_90", "RandomForest vs TabPFN-3", "mean_diff")),
        "cs_cov_rf_wins": str(int(c("coverage_90", "RandomForest vs TabPFN-3", "wins"))),
        "cs_cov_rf_holm": F(c("coverage_90", "RandomForest vs TabPFN-3", "p_holm"), 4),
        "cs_n_tests": str(len(cs)), "cs_n_sig": str(int((cs.p_holm < 0.05).sum())),
        "cs_holm8_max": F(_holm_all8(cs), 4),
        "cs_perm_min": F(cs.p_perm_blocked.min(), 4),
        "cs_perm_assignments": str(2 ** grid.block.nunique()),
        "cs_perm_min_count": str(int(round(cs.p_perm_blocked.min() * 2 ** grid.block.nunique()))),
        "win_r2": str(int(c("R2", "CatBoost vs TabPFN-3", "wins"))),
        "win_cov": str(int(c("coverage_90", "CatBoost vs TabPFN-3", "wins"))),
        "d_r2": F(c("R2", "CatBoost vs TabPFN-3", "mean_diff")),
        "d_cov": F(c("coverage_90", "CatBoost vs TabPFN-3", "mean_diff")),
        "p_r2": F(c("R2", "CatBoost vs TabPFN-3", "p_holm"), 4),
        "p_cov": F(c("coverage_90", "CatBoost vs TabPFN-3", "p_holm"), 4),

        # ---- decomposition ------------------------------------------------
        "dec_folds": str(int(decs.n_folds.iloc[0])),
        "dec_ntest": f"{pd.read_csv('panel/decomposition_tabpfn.csv').n_test.mean():.0f}",
        "dec_learners": str(decs.model.nunique()),
        "dec_share_min": f"{dshare.min():.0f}",
        "dec_share_max": f"{dshare.max():.0f}",
        "dec_tab_rep": F(components.loc['TabPFN-3', 'd_repeat']),
        "dec_tab_shift": F(components.loc['TabPFN-3', 'd_shift']),
        "dec_tab_share": F(dshare['TabPFN-3'], 0),
        "dec_tab_cat_ratio": F(components.loc['TabPFN-3', 'd_repeat'] / components.loc['CatBoost', 'd_repeat'], 1),
        "dec_cat_rep": F(decs[(decs.model == "CatBoost") & (decs.component == "d_repeat")]["mean"].values[0]),
        "dec_cat_rep_lo": F(decs[(decs.model == "CatBoost") & (decs.component == "d_repeat")].ci_low.values[0]),
        "dec_cat_rep_hi": F(decs[(decs.model == "CatBoost") & (decs.component == "d_repeat")].ci_high.values[0]),
        "dec_cat_shift": F(decs[(decs.model == "CatBoost") & (decs.component == "d_shift")]["mean"].values[0]),
        "dec_cat_shift_lo": F(decs[(decs.model == "CatBoost") & (decs.component == "d_shift")].ci_low.values[0]),
        "dec_cat_shift_hi": F(decs[(decs.model == "CatBoost") & (decs.component == "d_shift")].ci_high.values[0]),

        # ---- additional designs ------------------------------------------
        "ce_cat": F(des[(des.design == "construction-event disjoint")
                        & (des.model == "CatBoost")].R2.mean()),
        "loso_cat": F(des[(des.design == "leave-one-state-out")
                          & (des.model == "CatBoost")].R2.mean()),
        "loso_n": str(des[(des.design == "leave-one-state-out")
                          & (des.model == "CatBoost")].fold.nunique()),
        "ce_tab": F(des[(des.design == 'construction-event disjoint') & (des.model == 'TabPFN-3')].R2.mean()),
        "loso_tab": F(loso_tab.R2.mean()),
        "loso_tab_sd": F(loso_tab.R2.std()),
        "loso_cat_sd": F(loso[loso.model == 'CatBoost'].R2.std()),
        "loso_tab_min": F(loso_tab.R2.min()),
        "loso_tree_min": F(loso[loso.model != 'TabPFN-3'].R2.min()),
        "loso_cat_wins": str(int((paired_states.CatBoost > paired_states['TabPFN-3']).sum())),
        "loso_tab_no_worst": F(paired_states.drop(worst_state)['TabPFN-3'].mean()),
        "loso_cat_no_worst": F(paired_states.drop(worst_state).CatBoost.mean()),
        "loso_tab_mae": F(loso_tab.MAE.mean(), 3),
        "loso_cat_mae": F(loso[loso.model == 'CatBoost'].MAE.mean(), 3),
        "loso_worst_state": str(worst_state),
        "loso_worst_rows": str(int(S(f"state_{worst_state}_rows"))),
        "loso_worst_sections": str(int(S(f"state_{worst_state}_sections"))),
        "loso_worst_rigid_pct": F(S(f"state_{worst_state}_rigid_pct"), 0),
        "loso_worst_span": F(S(f"state_{worst_state}_span"), 1),
        "loso_worst_temp": F(S(f"state_{worst_state}_temp"), 1),
        "loso_worst_precip": F(S(f"state_{worst_state}_precip"), 0),
        "loso_worst_freeze": F(S(f"state_{worst_state}_freeze"), 0),

        # ---- conformal -----------------------------------------------------
        "cal_frac_pct": str(int(CAL_FRAC * 100)), "cal_max": f"{CAL_MAX:,}",
        "cal_50": F(cal[(cal.model == "CatBoost") & (cal.nominal == 0.50)].empirical.values[0], 3),
        "cal_80": F(cal[(cal.model == "CatBoost") & (cal.nominal == 0.80)].empirical.values[0], 3),
        "cal_99": F(cal[(cal.model == "CatBoost") & (cal.nominal == 0.99)].empirical.values[0], 3),
        "cm_split": F(cmm.loc["split", "coverage_90"], 3),
        "cm_weighted": F(cmm.loc["weighted", "coverage_90"], 3),
        "cm_recent": F(cmm.loc["recent", "coverage_90"], 3),
        "cm_mondrian": F(cmm.loc["mondrian", "coverage_90"], 3),
        "cm_gain": F(cmm.loc["recent", "coverage_90"] - cmm.loc["split", "coverage_90"], 3),
        "cm_width_gain": f"{100*(cmm.loc['recent','width']/cmm.loc['split','width']-1):.0f}",

        # ---- importance and ablation ---------------------------------------
        "imp_ratio": str(int(round(imp.perm_importance.iloc[0] / imp.perm_importance.iloc[1]))),
        "shap_ratio": str(int(round(imp.mean_abs_shap.sort_values(ascending=False).iloc[0] / imp.mean_abs_shap.sort_values(ascending=False).iloc[1]))),
        "abl_rand_with": F(ab("with initial_MRI", PROTO[0])),
        "abl_rand_without": F(ab("without initial_MRI", PROTO[0])),
        "abl_sd_with": F(ab("with initial_MRI", PROTO[1])),
        "abl_sd_without": F(ab("without initial_MRI", PROTO[1])),
        "abl_temp_with": F(ab("with initial_MRI", PROTO[2])),
        "abl_temp_without": F(ab("without initial_MRI", PROTO[2])),
        "abl_rand_drop": F(ab("with initial_MRI", PROTO[0]) - ab("without initial_MRI", PROTO[0]), 3),
        "abl_temp_drop": F(ab("with initial_MRI", PROTO[2]) - ab("without initial_MRI", PROTO[2]), 3),

        "res_mean": F(res.residual.mean(), 3),
        "res_top_quintile": F(rq.values[-1], 3),
        "res_min_quintile": F(rq.min(), 3),
        "res_max_quintile": F(rq.max(), 3),
        "res_vs_width": F(abs(res.residual.mean())
                          / cmm.loc["split", "width"] * 100, 0),

        # ---- heterogeneity ---------------------------------------------------
        "zone_best": F(zone.R2.max()), "zone_worst": F(zone.R2.min()),
        "type_flex": F(fam[fam.stratum.isin(["flexible (AC)","No recorded PCC layer"])].R2.values[0]),
        "type_rigid": F(fam[fam.stratum.isin(["rigid (PCC)","Recorded PCC layer"])].R2.values[0]),

        # ---- tuning ----------------------------------------------------------
        "tune_spread_min": F(spread.min(), 3), "tune_spread_max": F(spread.max(), 3),
        "tune_protocol_effect": F(prot_eff, 3),
        "tuned_best": tn.index[0], "tuned_best_r2": F(tn.iloc[0]),

        # ---- cost -------------------------------------------------------------
        "cost_ntrain": f"{int(tab.n_fit):,}", "cost_ntest": f"{int(tab.n_test):,}",
        "cost_tree_min": F(tree_tot.min(), 1), "cost_tree_max": F(tree_tot.max(), 1),
        "cost_tab_total": F(tab.total_s, 1),
        "cost_tab_rel": F(tab.total_s / cat.total_s, 0),
        "cost_tab_ms": F(tab.ms_per_prediction, 1),
        "cost_cat_ms": F(cat.ms_per_prediction, 2),

        # ---- cut-off sensitivity ----------------------------------------------
        "cut_min": str(sorted(grid.cutoff.unique())[0]),
        "xgb_2010": F(grid[(grid.model == "XGBoost") & (grid.cutoff == sorted(grid.cutoff.unique())[0])].R2.mean(), 3),
        "xgb_2013": F(grid[(grid.model == "XGBoost") & (grid.cutoff == sorted(grid.cutoff.unique())[1])].R2.mean(), 3),
        "xgb_2016": F(grid[(grid.model == "XGBoost") & (grid.cutoff == sorted(grid.cutoff.unique())[2])].R2.mean(), 3),

        # ---- protocol specification --------------------------------------------
        "k_main": "5", "k_grid": "3",
        "seeds_str": ", ".join(str(s) for s in sorted(grid.seed.unique())),
        "cutoffs_str": ", ".join(str(s) for s in sorted(grid.cutoff.unique())),

        # ---- environment ---------------------------------------------------------
        "py_ver": env["python"], "platform": env["platform"],
        "v_numpy": env["numpy"], "v_pandas": env["pandas"],
        "v_sklearn": env["scikit-learn"], "v_lgbm": env["lightgbm"],
        "v_xgb": env["xgboost"], "v_cat": env["catboost"], "v_scipy": env["scipy"],
        "v_torch": env["torch"], "v_tabpfn": env["tabpfn"], "v_shap": env["shap"],
        "cpu_logical": str(env["cpu_logical"]), "ram_gb": str(env.get("ram_gb", "n/a")),
        "n_refs": str(len(refs)),
    }
    N.update({
        'n_excluded':f"{int(S('n_excluded')):,}",
        'n_excl_other':f"{int(S('n_excl_other')):,}",
        'vws_sections':f"{int(raw_summary['vws_sections']):,}",
        'vws_stations':f"{int(raw_summary['vws_stations']):,}",
        'restricted_rows':f"{int(raw_summary['restricted_rows']):,}",
        'restricted_sections':f"{int(raw_summary['restricted_sections']):,}",
        'clim_p90':F(S('clim_p90'),1),'clim_p95':F(S('clim_p95'),1),'clim_max':F(S('clim_max'),1),
        'historic_wins':str(int(old_effects.wins)), 'historic_n':str(int(old_effects.n)),
        'historic_p':F(old_effects.wilcoxon_p),
        'cpu_model':env.get('cpu_model','not recorded'),
        'tuned_all':'; '.join(f"{name}: {g.test_R2.mean():.4f} (SD {g.test_R2.std():.4f})" for name,g in tnest.groupby('model')),
        'principal_cutoff':'2015','principal_seed':'0',
        'loso_negative_count':str(int((loso_tab.R2<0).sum())),
        'loso_worst_label':('Ohio (jurisdiction code 39)' if worst_state==39 else f'jurisdiction code {worst_state}'),
        'zone_omitted_n':str(int(cat.n_test-zone.n.sum())),
        'pre_annual_cat_temp':F(pre_annual_cat),
    })
    return N, main, grid, mod, X


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    N, *_ = build()
    print(f"{len(N)} keys")
    for k in sorted(N):
        print(f"  {k:22s} {N[k]}")
