'Summarise included and excluded panel visits and their observed covariates.'

import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

full = pd.read_parquet("panel/ltpp_panel.parquet")
from splits import load
mod = load()

full["included"] = full.index.isin(mod.index)
inc, exc = full[full.included], full[~full.included]
print(f"assembled {len(full):,} | modelled {len(inc):,} | excluded {len(exc):,} "
      f"({100*len(exc)/len(full):.1f}%)\n")

# reason for exclusion
first = full[full.visit_seq == 0]
print(f"excluded as first visit of a construction event: {len(first):,}")
other = exc[exc.visit_seq > 0]
print(f"excluded for <2 post-baseline visits in event : {len(other):,}")
pd.DataFrame([
    {'reason':'baseline visit','n_visits':len(first)},
    {'reason':'fewer than two post-baseline visits','n_visits':len(other)},
    {'reason':'included','n_visits':len(inc)},
]).to_csv('panel/inclusion_flow.csv',index=False)

NUM = ["MRI", "age_years", "thk_total", "t12_mean_temp", "t12_precip",
       "AADTT_ALL_TRUCKS_TREND", "cum_kesal", "clim_dist_km"]
rows = []
for c in NUM:
    if c not in full.columns:
        continue
    a, b = inc[c].dropna(), exc[c].dropna()
    if len(a) < 30 or len(b) < 30:
        continue
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    # standardised difference (Cohen's d, pooled)
    sp = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    rows.append({"variable": c, "included_n":len(a), "excluded_n":len(b),
                 "included_mean": a.mean(), "excluded_mean": b.mean(),
                 "std_diff": (a.mean()-b.mean())/sp if sp else np.nan,
                 "mannwhitney_p": u.pvalue})
num = pd.DataFrame(rows)
print("=== numerical attributes ===")
print(num.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# categorical / structural composition
cat = []
full["rigid"] = (full.get("thk_PC", pd.Series(np.nan, index=full.index)).fillna(0) > 0)
inc, exc = full[full.included], full[~full.included]      # refresh after new column
for c, lab in [("rigid", "recorded PCC-layer share"),
               ("clim_onsite", "on-site weather station share")]:
    if c in full.columns:
        cat.append({"attribute": lab,
                    "included": float(inc[c].mean()),
                    "excluded": float(exc[c].mean())})
cat.append({"attribute": "distinct LTPP jurisdiction codes", "included": inc.STATE_CODE.nunique(),
            "excluded": exc.STATE_CODE.nunique()})
cat.append({"attribute": "distinct sections", "included": inc.section_id.nunique(),
            "excluded": exc.section_id.nunique()})
cat.append({"attribute": "visits per construction event (median)",
            "included": float(inc.groupby("unit_id").size().median()),
            "excluded": float(exc.groupby("unit_id").size().median())})
cdf = pd.DataFrame(cat)
print("\n=== structural composition ===")
print(cdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

num.to_csv("panel/inclusion_numeric.csv", index=False)
cdf.to_csv("panel/inclusion_structural.csv", index=False)

# geographic representativeness
gi = inc.STATE_CODE.value_counts(normalize=True)
ge = exc.STATE_CODE.value_counts(normalize=True)
comp = pd.concat([gi, ge], axis=1, keys=["included", "excluded"]).fillna(0)
print(f"\nmax absolute difference in State share: {(comp.included-comp.excluded).abs().max():.4f}")
print(f"States appearing only in excluded set : {(comp.included == 0).sum()}")
