"""Matched-transition, matched-anchor comparisons; not a superiority benchmark."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import cellrank as cr

from scCS.future_fate import solve_discounted_outcomes
from .experiments import unequal_path_chain


def absorption(matrix, masks):
    a = ad.AnnData(np.zeros((matrix.shape[0], 1)))
    a.obs_names = [f"cell_{i}" for i in range(a.n_obs)]
    kernel = cr.kernels.PrecomputedKernel(sparse.csr_matrix(matrix), adata=a)
    estimator = cr.estimators.GPCCA(kernel)
    states = {name: a.obs_names[np.asarray(mask, bool)].tolist() for name, mask in masks.items()}
    estimator.set_terminal_states(states)
    estimator.compute_fate_probabilities(
        solver="gmres", use_petsc=False, n_jobs=1, tol=1e-10, show_progress_bar=False
    )
    return pd.DataFrame(
        np.asarray(estimator.fate_probabilities),
        columns=list(estimator.fate_probabilities.names),
        index=a.obs_names,
    )


def matched_chain():
    matrix, anchors = unequal_path_chain()
    baseline = absorption(matrix, {"A": anchors[:, 0], "B": anchors[:, 1]})
    rows = []
    for h in [4, 16, 64, 256, 4096, 1_000_000]:
        r = solve_discounted_outcomes(
            matrix, anchors, ("A", "B"), effective_horizon=h, solver="direct"
        )
        reach = r.probability[0].sum()
        rows.append(
            {
                "h": h,
                "DFR": reach,
                "CFA_A": r.probability[0, 0] / reach,
                "CellRank_A": baseline.loc["cell_0", "A"],
                "analytic_CFA_A": (h / (h + 1)) / (h / (h + 1) + (h / (h + 1)) ** 3),
            }
        )
    return pd.DataFrame(rows), baseline
