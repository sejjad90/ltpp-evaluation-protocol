"""Merge completed TabPFN experiments without changing measured source files.

The decomposition reproduces run_decomposition.py: 10,000 percentile-bootstrap
replicates of individual fold scores, NOT a seed-block bootstrap. Existing tree
groups are processed first to retain their original seeded Monte Carlo intervals.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd

B = 10000


def main():
    base = Path(__file__).resolve().parent
    root = (base.parent if base.name == 'code' else base) / 'panel'
    tree = pd.read_csv(root / "decomposition_folds.csv")
    tabpath=root/'decomposition_tabpfn.csv'
    tab = pd.read_csv(tabpath) if tabpath.exists() and os.environ.get('LTPP_TREES_ONLY')!='1' else tree.iloc[:0].copy()
    assert set(tree.columns) == set(tab.columns)
    dec = pd.concat([tree, tab], ignore_index=True)
    assert not dec.duplicated(["model", "seed", "fold"]).any()
    reference = tree[tree.model == "CatBoost"].set_index(["seed", "fold"])
    for _, group in dec.groupby("model"):
        pd.testing.assert_series_equal(
            group.set_index(["seed", "fold"]).n_test.sort_index(),
            reference.n_test.sort_index())
    assert np.allclose(dec.d_repeat, dec.leaky - dec.sect_disj)
    assert np.allclose(dec.d_shift, dec.sect_disj - dec.forward)
    rng = np.random.default_rng(0)
    rows = []
    names = sorted(tree.model.unique()) + sorted(tab.model.unique())
    for name in names:
        group = dec[dec.model == name]
        for component in ["d_repeat", "d_shift"]:
            values = group[component].to_numpy()
            draws = np.array([rng.choice(values, len(values), replace=True).mean()
                              for _ in range(B)])
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rows.append(dict(model=name, component=component, mean=values.mean(),
                             ci_low=lo, ci_high=hi, n_folds=len(values)))
    summary = pd.DataFrame(rows)
    old = pd.read_csv(root / "decomposition_summary.csv")
    pd.testing.assert_frame_equal(summary.iloc[:len(old)].reset_index(drop=True), old,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    summary.to_csv(root / "decomposition_summary_all.csv", index=False)

    trees = pd.read_csv(root / "review_designs.csv")
    tabpath=root/'review_designs_tabpfn.csv'
    tabs = pd.read_csv(tabpath) if tabpath.exists() and os.environ.get('LTPP_TREES_ONLY')!='1' else trees.iloc[:0].copy()
    assert set(trees.columns) == set(tabs.columns)
    designs = pd.concat([trees, tabs], ignore_index=True)
    assert not designs.duplicated(["design", "model", "fold"]).any()
    reference = trees[trees.model == "CatBoost"].set_index(["design", "fold"])
    for _, group in designs.groupby("model"):
        pd.testing.assert_series_equal(
            group.set_index(["design", "fold"]).n_test.sort_index(),
            reference.n_test.sort_index())
    designs.to_csv(root / "review_designs_all.csv", index=False)
    print(f"Merged {len(dec)} decomposition folds into {len(summary)} summary rows; "
          f"{len(designs)} additional-design rows. Originals unchanged.")


if __name__ == "__main__":
    main()
