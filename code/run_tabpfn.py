"""TabPFN-3 on the LTPP roughness panel, with split-conformal intervals.

Uses locally cached weights when available. Initial weight acquisition must
follow the provider's access and licence conditions; these runs use the cache.

Inference is in-context, so cost scales with (train rows x scored rows). The
calibration split is capped as a computational/quantile-precision compromise.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from splits import (PROTOCOLS, calibration_split, conformal_halfwidth,
                    conformal_interval, coverage, feature_matrix, load)

# The licence must have been accepted once on the Prior Labs account; after the
# first successful download the weights are cached locally and load offline, so
# a live token is not required for subsequent runs.
if not os.environ.get("TABPFN_TOKEN"):
    print("TABPFN_TOKEN not set - relying on locally cached weights.", flush=True)

from tabpfn import TabPFNRegressor  # noqa: E402

CAL_MAX = 1500      # rows used to calibrate the conformal interval
ALPHA = 0.1         # target 90% coverage
N_EST = 8           # ensemble members; no ensemble-size comparison is reported
OUT = "panel/tabpfn3_results.csv"


def main():
    df = load()
    X, y = feature_matrix(df)
    print(f"rows {len(df):,}  sections {df.section_id.nunique():,}  features {X.shape[1]}",
          flush=True)

    rows, fold_rows = [], []
    for pname, splitter in PROTOCOLS.items():
        r2s, maes, rmses, covs, widths = [], [], [], [], []
        for k, (tr, te) in enumerate(splitter(df)):
            cal, fit = calibration_split(tr, seed=0)

            t0 = time.time()
            m = TabPFNRegressor(n_estimators=N_EST, auto_scale_n_estimators=False,
                                random_state=0, ignore_pretraining_limits=True)
            m.fit(X.iloc[fit], y.iloc[fit])

            qhat = conformal_halfwidth(np.abs(y.iloc[cal].values - m.predict(X.iloc[cal])))
            p = m.predict(X.iloc[te])
            yt = y.iloc[te].values
            lo, hi = conformal_interval(p, qhat)
            r2s.append(r2_score(yt, p))
            maes.append(mean_absolute_error(yt, p))
            rmses.append(np.sqrt(mean_squared_error(yt, p)))
            covs.append(coverage(yt, lo, hi))
            widths.append(float(np.mean(hi - lo)))
            fold_rows.append(dict(model='TabPFN-3', protocol=pname, fold=k,
                n_fit=len(fit), n_cal=len(cal), n_test=len(te), R2=r2s[-1],
                MAE=maes[-1], RMSE=rmses[-1], coverage_90=covs[-1],
                interval_width=widths[-1]))
            pd.DataFrame(fold_rows).to_csv('panel/principal_tabpfn_folds.csv',index=False)
            print(f"  [{pname}] fold {k+1}: {time.time()-t0:6.0f}s  "
                  f"train={len(fit):,} test={len(te):,}  R2={r2s[-1]:.4f}  "
                  f"cov={covs[-1]:.3f}", flush=True)

        rows.append({"model": "TabPFN-3", "protocol": pname,
                     "R2": np.mean(r2s), "R2_sd": np.std(r2s),
                     "MAE": np.mean(maes), "RMSE": np.mean(rmses),
                     "coverage_90": np.mean(covs),
                     "interval_width": np.mean(widths),
                     "MAE_sd":np.std(maes), "RMSE_sd":np.std(rmses),
                     "coverage_90_sd":np.std(covs), "interval_width_sd":np.std(widths)})
        pd.DataFrame(rows).to_csv(OUT, index=False)   # checkpoint after each protocol
        print(f"  -> {pname}: R2={rows[-1]['R2']:.4f}  cov={rows[-1]['coverage_90']:.3f}\n",
              flush=True)

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
