from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from scCS.future_fate import (
    canonicalize_transition_matrix,
    score_future_fate,
    solve_discounted_outcomes,
)
from scCS.furcation import Furcation
from scCS.geometry import SimplexStarGeometry
from scCS.ordering import FurcationOrderingResult, OrderingDiagnostics
from scCS.scoring_embedding import ScoringEmbeddingResult


def _embedding() -> tuple[Furcation, ScoringEmbeddingResult]:
    furcation = Furcation(obs_key="cell_type", root="Root", terminals=["A", "B"])
    geometry = SimplexStarGeometry(["A", "B"])
    root_mask = np.array([True, True, False, False])
    terminal_mask = ~root_mask
    terminal_names = np.array(["", "", "A", "B"], dtype=object)
    root_progress = np.array([0.0, 1.0, 0.0, 0.0])
    diagnostics = OrderingDiagnostics(
        root_lower_bound=0.0,
        root_upper_bound=0.3,
        root_clipped_low_fraction=0.5,
        root_clipped_high_fraction=0.5,
        root_progress_min=0.0,
        root_progress_max=1.0,
        root_progress_mean=0.5,
        root_progress_sd=0.5,
        root_unique_values=2,
        root_unique_fraction=1.0,
        root_largest_tie_fraction=0.5,
    )
    ordering = FurcationOrderingResult(
        root_progress=root_progress,
        terminal_names=terminal_names,
        root_mask=root_mask,
        terminal_mask=terminal_mask,
        diagnostics=diagnostics,
    )
    coordinates = np.empty((4, geometry.dimension))
    coordinates[root_mask] = geometry.root_coordinates(root_progress[root_mask])
    coordinates[terminal_mask] = geometry.terminal_coordinates(terminal_names[terminal_mask])
    embedding = ScoringEmbeddingResult(
        coordinates=coordinates,
        selected_indices=np.array([0, 1, 2, 3]),
        selected_cell_ids=np.array(["r0", "r1", "a", "b"]),
        selected_labels=np.array(["Root", "Root", "A", "B"]),
        selected_ordering_values=np.array([0.0, 0.3, 1.0, 1.0]),
        root_mask=root_mask,
        terminal_mask=terminal_mask,
        terminal_names=terminal_names,
        geometry=geometry,
        ordering=ordering,
        arm_scale=1.0,
        ordering_key="pseudotime",
    )
    return furcation, embedding


def test_canonicalize_tolerates_sparse_row_sum_roundoff():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.5, 0.5000001, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    result = canonicalize_transition_matrix(matrix)
    assert result.n_zero_rows == 1
    assert result.input_max_abs_row_sum_error > 0
    assert np.allclose(np.asarray(result.matrix.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    assert result.matrix[2, 2] == 1.0


def test_discounted_chain_has_expected_geometric_probability():
    # state 0 -> state 1 -> anchor A; anchor B is unreachable from state 0.
    transition = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    anchors = np.zeros((4, 2), dtype=bool)
    anchors[2, 0] = True
    anchors[3, 1] = True
    result = solve_discounted_outcomes(
        transition,
        anchors,
        ("A", "B"),
        effective_horizon=4,
        solver="direct",
    )
    gamma = 4 / 5
    assert np.isclose(result.probability[1, 0], gamma)
    assert np.isclose(result.probability[0, 0], gamma**2)
    assert np.isclose(result.unresolved_probability[0], 1 - gamma**2)
    assert result.probability[0, 1] == 0


def test_iterative_and_direct_solvers_agree():
    transition = sparse.csr_matrix(
        np.array(
            [
                [0.2, 0.4, 0.2, 0.2],
                [0.0, 0.2, 0.4, 0.4],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    anchors = np.zeros((4, 2), dtype=bool)
    anchors[2, 0] = True
    anchors[3, 1] = True
    direct = solve_discounted_outcomes(
        transition, anchors, ("A", "B"), effective_horizon=32, solver="direct"
    )
    iterative = solve_discounted_outcomes(
        transition,
        anchors,
        ("A", "B"),
        effective_horizon=32,
        solver="iterative",
        tolerance=1e-12,
    )
    assert iterative.converged
    assert np.allclose(iterative.probability, direct.probability, atol=1e-10)
    assert np.allclose(iterative.unresolved_probability, direct.unresolved_probability, atol=1e-10)


def test_full_graph_leave_and_return_is_preserved():
    furcation, embedding = _embedding()
    # r0 -> outside -> r1 -> A/B. The outside state is not selected, but it is
    # transient and may return because the solver uses the full source graph.
    transition = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    result = score_future_fate(
        furcation,
        embedding,
        transition,
        effective_horizon=9,
        anchor_quantile=0.5,
        min_anchor_cells=1,
        progression_scale="minmax",
    )
    gamma = 9 / 10
    assert np.isclose(result.future_fate_reach[0], gamma**3)
    assert np.allclose(result.future_fate_affinity[0], [0.5, 0.5])
    assert np.isclose(result.unresolved_probability[0], 1 - gamma**3)
    assert result.one_step_outside_probability[0] == 1.0
    np.testing.assert_array_equal(result.signed_progression, result.progression_velocity)
    np.testing.assert_array_equal(
        result.selected_path_coverage, result.projection.transition_coverage
    )
    np.testing.assert_array_equal(
        result.conditional_fate_affinity, result.future_fate_affinity
    )
    np.testing.assert_array_equal(
        result.discounted_fate_reach, result.future_fate_reach
    )
    np.testing.assert_array_equal(
        result.resolved_commitment, result.reach_supported_specificity
    )
    np.testing.assert_array_equal(
        result.signed_ordering_flux, result.signed_progression
    )
    np.testing.assert_array_equal(
        result.unresolved_future_probability, result.unresolved_probability
    )
    assert result.scoring_mode == "future_fate"


def test_explicit_competing_outcome_remains_separate():
    furcation, embedding = _embedding()
    transition = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    result = score_future_fate(
        furcation,
        embedding,
        transition,
        effective_horizon=9,
        anchor_quantile=0.5,
        min_anchor_cells=1,
        competing_outcomes={"External": [4]},
    )
    gamma = 9 / 10
    assert result.future_fate_reach[0] == 0.0
    assert np.isclose(result.competing_reach[0], gamma)
    assert np.isclose(result.unresolved_probability[0], 1 - gamma)
    assert result.outcome_names == ("A", "B", "External")


class _FakeAnnData:
    def __init__(self, obs):
        self.obs = obs
        self.obs_names = obs.index
        self.n_obs = len(obs)
        self.obsm = {}
        self.uns = {}

    def copy(self):
        import copy

        return copy.deepcopy(self)


def _fake_adata():
    import pandas as pd

    obs = pd.DataFrame(
        {
            "cell_type": ["Root", "Root", "A", "B", "Outside", "Root", "A", "B"],
            "velocity_pseudotime": [0.0, 0.3, 1.0, 1.0, 0.2, 0.1, 0.9, 0.9],
            "condition": [
                "control",
                "treated",
                "control",
                "control",
                "control",
                "treated",
                "treated",
                "treated",
            ],
            "replicate": ["c1", "t1", "c1", "c1", "c1", "t2", "t2", "t2"],
        },
        index=[f"cell{i}" for i in range(8)],
    )
    return _FakeAnnData(obs)


def _fake_transition():
    return sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )


def test_single_scorer_future_fate_public_workflow():
    from scCS.single import SingleScorer

    scorer = SingleScorer(_fake_adata(), root="Root", branches=["A", "B"], obs_key="cell_type")
    scorer.build_embedding(
        ordering_metric="pseudotime",
        write_to_adata=False,
        cache_subset=False,
        verbose=False,
    )
    scorer.fit(
        transition_matrix=_fake_transition(),
        scoring_mode="future_fate",
        future_fate_options={
            "effective_horizon": 16,
            "anchor_quantile": 0.5,
            "min_anchor_cells": 1,
            "verbose": False,
        },
    )
    result = scorer.score(write_to_adata=False, verbose=False)
    assert scorer.scoring_mode == "future_fate"
    assert result.scoring_mode == "future_fate"
    assert result.future_fate_affinity.shape == (7, 2)
    assert np.allclose(result.future_fate_affinity.sum(axis=1), 1.0)
    assert scorer.projected_velocity is None

    import matplotlib.pyplot as plt

    for color_by in (
        "future_fate_reach",
        "future_fate_specificity",
        "reach_supported_specificity",
        "signed_progression",
        "future_fate_affinity:A",
    ):
        figure = scorer.plot_star(result, color_by=color_by)
        plt.close(figure)


def test_pair_scorer_accepts_future_fate_metrics():
    from scCS.pairwise import PairScorer

    scorer = PairScorer(
        _fake_adata(),
        root="Root",
        branches=["A", "B"],
        obs_key="cell_type",
        condition_obs_key="condition",
        replicate_obs_key="replicate",
    )
    scorer.build_embedding(
        ordering_metric="pseudotime",
        write_to_adata=False,
        cache_subset=False,
        verbose=False,
    )
    scorer.fit(
        transition_matrix=_fake_transition(),
        scoring_mode="future_fate",
        future_fate_options={
            "effective_horizon": 16,
            "anchor_quantile": 0.5,
            "min_anchor_cells": 1,
            "verbose": False,
        },
    )
    with pytest.warns(RuntimeWarning):
        results = scorer.score_all_conditions(
            min_cells=1, min_replicates=1, write_to_adata=False, verbose=False
        )
    assert set(results) == {"control", "treated"}
    assert all(result.directional_affinity.shape[1] == 2 for result in results.values())
    assert all(np.isfinite(result.specific_commitment).all() for result in results.values())


def test_future_fate_write_to_h5ad_serializes_anchor_diagnostics(tmp_path):
    anndata = pytest.importorskip("anndata")
    furcation, embedding = _embedding()
    transition = sparse.eye(4, format="csr")
    result = score_future_fate(
        furcation,
        embedding,
        transition,
        effective_horizon=8,
        anchor_quantile=0.5,
        min_anchor_cells=1,
    )
    adata = anndata.AnnData(
        X=np.zeros((4, 1), dtype=float),
        obs={"cell_type": ["Root", "Root", "A", "B"]},
    )
    result.write_to_adata(adata)
    output = tmp_path / "future_fate.h5ad"
    adata.write_h5ad(output)
    restored = anndata.read_h5ad(output)
    diagnostics = restored.uns["sccs_v08"]["future_fate"]["anchor_diagnostics"]
    assert set(diagnostics) == {"A", "B"}
    assert diagnostics["A"]["n_anchors"] == 1
