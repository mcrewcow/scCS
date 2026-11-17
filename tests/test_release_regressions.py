"""Regression invariants for fixes introduced in 0.8.1rc1.

Run: python -m pytest -q tests/test_release_regressions.py
"""

import numpy as np
import pandas as pd
import anndata as ad
import pytest
from scipy import sparse
from scCS import SingleScorer
from scCS.future_fate import solve_discounted_outcomes


def fitted():
    obs = pd.DataFrame(
        {
            "cluster": ["root", "root", "A", "A", "B", "B"],
            "pt": [0.1, 0.2, 0.6, 0.9, 0.6, 0.9],
        },
        index=["r0", "r1", "a0", "a1", "b0", "b1"],
    )
    a = ad.AnnData(X=np.zeros((6, 2)), obs=obs)
    P = sparse.csr_matrix(
        [
            [0, 0.5, 0.4, 0, 0.1, 0],
            [0, 0, 0.8, 0, 0.2, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )
    s = SingleScorer(a, root="root", branches=["A", "B"], obs_key="cluster")
    s.build_embedding(ordering_metric="pt", verbose=False)
    s.fit_future_fate(P, min_anchor_cells=1, verbose=False)
    return s, P


def test_score_export_aligns_cell_ids():
    s, P = fitted()
    out = s.adata[::-1].copy()
    s.result.write_to_adata(out)
    expected = pd.Series(s.result.future_fate_reach, index=s.result.cell_ids).reindex(
        out.obs_names
    )
    np.testing.assert_allclose(out.obs.sccs_future_fate_reach, expected)


def test_future_mode_survives_serialization(tmp_path):
    s, P = fitted()
    path = tmp_path / "s.pkl"
    s.save(str(path))
    restored = SingleScorer.load(str(path), s.adata)
    assert restored.scoring_mode == s.scoring_mode


def test_constant_ordering_has_no_flux():
    s, P = fitted()
    s.fit_future_fate(
        P, progression_values=np.ones(6), min_anchor_cells=1, verbose=False
    )
    np.testing.assert_allclose(s.result.signed_ordering_flux, 0, atol=1e-14)


def test_cell_permutation_preserves_tied_ordering_flux():
    s, P = fitted()
    first = s.result
    order = np.array([0, 1, 4, 5, 2, 3])
    a = s.adata[order].copy()
    t = SingleScorer(a, root="root", branches=["A", "B"], obs_key="cluster")
    t.build_embedding(ordering_metric="pt", verbose=False)
    t.fit_future_fate(P[order][:, order], min_anchor_cells=1, verbose=False)
    indices = pd.Index(t.result.cell_ids).get_indexer(first.cell_ids)
    np.testing.assert_allclose(
        first.signed_ordering_flux, t.result.signed_ordering_flux[indices], atol=1e-14
    )


def test_small_reach_does_not_reallocate_outcome_mass():
    s, P = fitted()
    P = P.tolil()
    for i in (0, 1):
        P.rows[i] = [i, 3]
        P.data[i] = [0.999999, 0.000001]
    s.fit_future_fate(P.tocsr(), min_reach=0.01, min_anchor_cells=1, verbose=False)
    np.testing.assert_allclose(
        s.result.future_fate_contribution, s.result.selected_probability, atol=1e-14
    )


def test_zero_minimum_reach_has_explicit_validation():
    s, P = fitted()
    with pytest.raises(ValueError, match="min_reach"):
        s.fit_future_fate(sparse.eye(6), min_reach=0, min_anchor_cells=1, verbose=False)


def test_public_solver_rejects_negative_transition():
    P = np.array([[0, 1.2, -0.2], [0, 1, 0], [0, 0, 1.0]])
    anchors = np.array([[0, 0], [1, 0], [0, 1]], bool)
    with pytest.raises(ValueError, match="negative"):
        solve_discounted_outcomes(P, anchors, ["A", "B"])


def test_fractional_horizon_not_silently_recorded_as_integer():
    P = np.array([[0, 1.0, 0], [0, 1, 0], [0, 0, 1.0]])
    anchors = np.array([[0, 0], [1, 0], [0, 1]], bool)
    with pytest.raises(ValueError, match="integer"):
        solve_discounted_outcomes(P, anchors, ["A", "B"], effective_horizon=1.5)


def test_condition_star_panels_share_color_scale():
    from scCS import PairScorer
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, P = fitted()
    blocks = []
    matrices = []
    for condition, weight in [("control", 0.1), ("treated", 0.9)]:
        for rep in range(2):
            a = base.adata.copy()
            a.obs["condition"] = condition
            a.obs["replicate"] = str(rep)
            a.obs_names = [f"{condition}_{rep}_{x}" for x in a.obs_names]
            blocks.append(a)
            t = P.tolil()
            for i in (0, 1):
                t.rows[i] = [3, 5]
                t.data[i] = [1 - weight, weight]
            matrices.append(t.tocsr())
    a = ad.concat(blocks)
    s = PairScorer(
        a,
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
        replicate_obs_key="replicate",
    )
    s.build_embedding(ordering_metric="pt", verbose=False)
    s.fit(
        sparse.block_diag(matrices),
        scoring_mode="future_fate",
        future_fate_options={"min_anchor_cells": 1},
        verbose=False,
    )
    s.score_all_conditions(min_cells=1, verbose=False)
    fig = s.plot_star_grid(color_by="future_fate_affinity:B", population="root")
    axes = [a for a in fig.axes if a.get_title() in s.conditions]
    clims = [a.collections[0].get_clim() for a in axes]
    plt.close(fig)
    np.testing.assert_allclose(clims[0], clims[1])
