"""Row-random, section-disjoint and section-disjoint later-survey evaluation.

Later-survey inputs are survey-aligned; earlier issue-date availability is not
established. Exact tasks and calibration limitations are described in Methods.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

TARGET = "MRI"
DROP = {
    "MRI", "VISIT_DATE", "OPEN_DATE", "CN_ASSIGN_DATE", "INV_OPEN",
    "section_id", "unit_id", "SHRP_ID", "VWS_ID", "VWS_NEAREST", "n_runs",
    "GPS_SPS", "RECORD_STATUS",
    "mi",            # month index used only to align the climate window
}

# YEAR denotes the target survey year in this conditional later-survey task.
# Availability at an earlier forecast issue date is not established. Test years
# exceed the fitting range under the combined protocol.


def load(path="panel/ltpp_panel.parquet", min_visits=2):
    df = pd.read_parquet(path)

    # initial_MRI is the first visit's own MRI, so for visit_seq == 0 the feature
    # is the target. Those rows are the reference observation and are dropped
    # rather than predicted.
    df = df[df["visit_seq"] > 0].copy()

    # keep construction events with enough history to be a deterioration curve
    n = df.groupby("unit_id")["MRI"].transform("size")
    df = df[n >= min_visits].copy()

    df = df[df[TARGET].notna() & np.isfinite(df[TARGET])]
    return df


def feature_matrix(df):
    cols = [c for c in df.columns if c not in DROP]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().mean() > 0.02]          # drop all-but-empty columns
    X = X.loc[:, X.nunique(dropna=True) > 1]        # drop constants
    return X, df[TARGET].astype(float)


def splits_random(df, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(df))


def splits_group(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df["section_id"]))


def splits_temporal(df, cutoff=2015, n_splits=5, seed=0):
    """Section-disjoint folds, then a forward-in-time cut within each fold."""
    rng = np.random.default_rng(seed)
    secs = np.asarray(df["section_id"].unique(), dtype=object)
    rng.shuffle(secs)
    folds = np.array_split(secs, n_splits)
    year = df["VISIT_DATE"].dt.year.values
    out = []
    for f in folds:
        test_sec = np.isin(df["section_id"].values, f)
        tr = np.where(~test_sec & (year <= cutoff))[0]
        te = np.where(test_sec & (year > cutoff))[0]
        if len(te) and len(tr):
            out.append((tr, te))
    return out


PROTOCOLS = {
    "random (leaky)": splits_random,
    "section-disjoint": splits_group,
    "section-disjoint + forward-in-time": splits_temporal,
}


# ---------------------------------------------------------------- conformal
# Shared by principal, repeated-grid, ablation and interval experiments.
CAL_MAX = 1500      # computational cap; quantile precision remains sample-dependent
CAL_MIN = 200
CAL_FRAC = 0.2
ALPHA = 0.10        # nominal 90% intervals


def calibration_split(tr, seed=0):
    """Seeded row holdout: fitting/calibration may share sections.

    Test-section exclusion is inherited only under grouped protocols.
    The random protocol can share sections among all subsets. Exchangeability
    of calibration/test residuals is not established by this split.
    """
    rng = np.random.default_rng(seed)
    tr = rng.permutation(np.asarray(tr))
    ncal = min(CAL_MAX, max(CAL_MIN, int(len(tr) * CAL_FRAC)))
    return tr[:ncal], tr[ncal:]


def conformal_halfwidth(residuals, alpha=ALPHA):
    """Linear-interpolated numpy quantile at min(1, ceil((n+1)(1-alpha))/n).

    This is not exact empirical rank selection. The interpolation convention
    is retained for the annual-exposure recomputation.
    """
    r = np.asarray(residuals, dtype=float)
    n = len(r)
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(r, q, method="linear"))


def conformal_interval(pred, qhat, lower_bound=0.0):
    """Symmetric interval, clipped at a physical lower bound.

    Roughness cannot be negative, so the lower limit is clipped at zero. The
    clip is applied before coverage is evaluated, so reported coverage refers to
    the interval an agency would actually use.
    """
    lo = np.maximum(np.asarray(pred) - qhat, lower_bound)
    hi = np.asarray(pred) + qhat
    return lo, hi


def coverage(y_true, lo, hi):
    y = np.asarray(y_true, dtype=float)
    return float(((y >= lo) & (y <= hi)).mean())
