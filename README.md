# LTPP evaluation-protocol analysis

Analytical sources and measured outputs for *Effect of evaluation protocol on
accuracy, model ranking and uncertainty in pavement roughness prediction*.
This local working tree contains the annual-exposure correction. Its raw-data
tree-only path passed a clean-directory run with extracted SDR 39 modules; the
full clean TabPFN path was not tested. Remote publication remains an author
action. See `results/legacy/README.md`
for withdrawn historical summaries; do not use them as current findings.

## Layout

| Path | Contents |
|---|---|
| `code/` | Panel assembly, experiments, statistical summaries and figure scripts |
| `results/` | Shipped measured CSVs and environment/configuration metadata |
| `results/legacy/` | Superseded historical summaries with provenance |
| `docs/` | Bibliographic records and citation keys |
| `panel/` | Working panel and intermediate/result CSVs created during execution |
| `figures/` | Rendered PNGs created during execution |
| `zips/`, `extracted/` | Raw downloads/extracted SDR 39 modules, created separately |

All scripts run with the repository root as working directory. The entry point
sets it explicitly, so it can also be invoked from `code/` or another directory.
Raw SDR 39 modules, the assembled parquet, model weights, document builders and
exact split-membership manifests are not shipped. The private response and
document sources are excluded. Bibliographic metadata verifies identity; it
does not certify that every cited sentence follows from every source.

## Run

Use the pinned Python environment in `requirements.txt` (the recorded run uses
Python 3.14). Install the CPU PyTorch wheel using its official index if your
package index does not provide the `+cpu` build. Raw assembly additionally needs
the 64-bit Microsoft Access ODBC driver used by `pyodbc`; the full raw-data path
is consequently verified on Windows. Rendering shipped CSVs does not open Access.

Render shipped measured summaries without refitting or the assembled parquet:

```powershell
python code/run_all.py --results-only
```

This copies the top-level shipped CSV/JSON records from `results/` into `panel/`,
rebuilds merged summaries and supported statistics, renders figures, and prints
the shared numbers dictionary. Figure 1 and panel-dependent narrative
diagnostics use `panel_summary.csv` and `raw_summary.csv`, emitted by the
included summary script. It does not rebuild a manuscript or response.

Download and extract raw SDR 39 modules:

```powershell
python code/fetch_sdr39.py
```

The shell equivalent is `bash code/fetch_sdr39.sh`; set `PYTHON` if necessary.
Both create repository-relative `zips/` and `extracted/`, then extract modules.
`python code/run_all.py --fetch` invokes the same step before fitting. FHWA
availability, download redirects and Access-driver installation are external
prerequisites; download service availability is not guaranteed by this package.

Refit every stage, or just trees, from raw extracted module directories:

```powershell
python code/run_all.py --trees-only --raw-dir "C:\path\to\extracted"
python code/run_all.py --raw-dir "C:\path\to\extracted"
```

Omit `--raw-dir` when the modules are under this repository's `extracted/`.
Existing working outputs permit resuming; `--force` recomputes them. Shipped
results are staged **only** with `--results-only`, so a new raw-data run does not
silently substitute supplied fitted results. Use an empty `panel/` for a clean
refit. Do not mix outputs from different feature definitions.

TabPFN weights are obtained separately under the
[TabPFN-3 Non-Commercial Licence v1.0](https://huggingface.co/Prior-Labs/tabpfn_3/raw/main/LICENSE)
and its non-commercial/non-production conditions. The MIT grant applies to the
authored analysis code; model weights and TabPFN output use retain the separate
Prior Labs conditions. Weights are excluded by packaging choice, not because the
licence categorically prohibits redistribution. The author must arrange any
required account/licence access. Previously cached weights are used offline
when available. TabPFN stages can take hours on CPU.

## Stages and output granularity

| Stage | Script | Retained granularity |
|---|---|---|
| Assembly | `build_panel.py`, `annual_loading.py` | Section-visit parquet; unique event/survey-year annual-loading provenance |
| Principal comparison | `run_comparators.py`, `run_tabpfn.py` | Protocol/model means and population SDs; corrected reruns retain per-fold metrics/counts |
| Repeated grid | `run_robustness.py` | Model/seed/cut-off/fold scores and test counts |
| Fixed-test regimes | `run_decomposition.py`, `run_tabpfn_designs.py` | Model/seed/fold regime scores; descriptive fold-bootstrap summaries |
| Internal holdouts | `run_review_experiments.py`, `run_tabpfn_designs.py` | Construction-event fold / selected jurisdiction scores |
| Ablation | `run_ablation.py` | Feature-set/protocol averages |
| Interval methods | `run_conformal_methods.py` | CatBoost/seed/cut-off/fold/method coverage and width |
| Cost/subgroups | `run_cost_zones.py` | One shared fold; model costs and visit-weather/layer-group scores |
| Tuning | `run_tuning.py` | Configuration averages and nested outer-fold scores, four tree learners |
| Inclusion | `run_inclusion.py` | Exclusion flow, nonmissing sample sizes, variable and composition summaries |
| Diagnostic provenance | `summarise_panel.py`, `summarise_splits.py` | CSV panel/raw diagnostics and principal/grid aggregate split counts/overlaps |
| Environment | `record_environment.py` | Package versions, constructor defaults, thread policy and current cache hashes |
| Merge/inference | `merge_tabpfn_results.py`, `run_cluster_stats.py` | Combined tables; blocked comparisons and Holm/BH adjustments |
| Figures | `make_fig1.py`, `make_figures*.py` | PNGs from the working measured summaries |

`--trees-only` omits TabPFN fitting, reference comparisons and the paired
CatBoost/TabPFN histogram. Other figures and merged summaries use only trees.
It does not synthesise missing foundation-model results. Legacy
`run_review_stats.py` is excluded from the entry point: its fold-independence
assumptions are not the supported current inference.

## Limits

The outputs are a mixture of summaries, selected per-fold scores, reconstructed
aggregate split counts/overlaps and single-fold diagnostics. Exact split memberships, most predictions and calibration residuals
are not retained; the new principal records supply metrics/counts, not complete
manifests. Some diagnostics require rebuilding the panel to recompute their CSV
source. Seeds and constructor defaults do not retrospectively establish the
historical checkpoint identity or exact per-library thread usage. The corrected
metadata identifies the resolved default regressor checkpoint and its SHA-256;
matching that identity is necessary for an exact checkpoint reproduction.

Later-survey inputs are aligned before each survey, not verified at an earlier
forecast issue date. Fitting/calibration rows can share sections; held-out test
sections differ from that calibration setting. Global unsupervised screening
uses held-out feature distributions. Blocked inference still reuses sections
and years across blocks, and decomposition intervals resample dependent folds.
Weather and PCC-layer groups describe implemented rules, not validated long-term
climate or pure pavement-surface classifications. CPU memory is Python-traced
allocation after construction, and per-row time is amortised batch time. These
are not total RAM or single-request latency measurements.

The old published `v1.0.0` archive is not automatically updated by edits to this
working tree. A corrected release requires the author's separate publication
decision after confidentiality cleanup and verification.
