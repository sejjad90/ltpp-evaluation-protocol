# Superseded historical artifacts

The four original files (all_results.csv, protocol_comparison.csv,
effect_sizes.csv, bootstrap_ci.csv) describe the earlier panel and unmatched
calibration practices. Their accuracy significance claim is withdrawn.
The two pre_annual_* files preserve the audited principal summaries immediately
before the annual-exposure correction; calibration was already harmonised in
that snapshot. All six files are superseded, not current findings, and must not
be combined with corrected annual-exposure outputs. The individual historical
principal fold predictions/split memberships are not retained. Historical
quotations in the private response can be checked against these files; they are
not recomputed by the current cluster-aware pipeline.

`run_review_stats.py` is an unsupported historical diagnostic and is excluded
from `run_all.py`. Use `cluster_stats.csv` for the current supported inference.
