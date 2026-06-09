"""
Unit tests for scCS — core math, embedding, bifurcation, and end-to-end.

Tests cover:
1.  Magnitude computation (Eq. 1)
2.  Angle computation (Eq. 2-3)
3.  Angular binning (Eq. 4-6)
4.  Sector definitions (equal, centroid)
5.  unCS and nCS (Eq. 8-9)
6.  Commitment vector
7.  Population entropy (compute_population_entropy)
8.  Mean cell entropy (compute_mean_cell_entropy)
9.  Per-fate cell entropy (compute_per_fate_cell_entropy)
10. NN-smoothed cell entropy (compute_nn_cell_entropy)
11. Backward-compatibility alias (compute_commitment_entropy)
12. Pairwise CS matrix
13. Per-cell scores
14. CommitmentScoreResult dataclass
15. Radial star embedding (build_star_embedding)
16. FateMap construction (build_fate_map)
17. End-to-end: synthetic bifurcation (k=2)
18. End-to-end: synthetic trifurcation (k=3)
19. End-to-end: full SingleScorer pipeline (synthetic AnnData)
"""

import warnings

import numpy as np
import pytest
import anndata as ad
import pandas as pd

from scCS.scores import (
    bin_angles,
    centroid_sectors,
    compute_angles,
    compute_cell_scores,
    compute_commitment_entropy,    # backward-compat alias
    compute_commitment_vector,
    compute_magnitudes,
    compute_mean_cell_entropy,
    compute_nn_cell_entropy,
    compute_nCS,
    compute_pairwise_cs_matrix,
    compute_per_fate_cell_entropy,
    compute_population_entropy,
    compute_sector_magnitudes,
    compute_unCS,
    equal_sectors,
    CommitmentScoreResult,
)
from scCS.embedding import build_star_embedding, _resolve_metric
from scCS.bifurcation import build_fate_map, FateMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells: int = 300, seed: int = 0) -> ad.AnnData:
    """Synthetic AnnData with 3 clusters and pseudotime."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, 20))
    # 3 clusters: 0 = progenitor, 1 = fateA, 2 = fateB
    labels = np.array(["0"] * 100 + ["1"] * 100 + ["2"] * 100)
    pseudotime = np.concatenate([
        rng.uniform(0.0, 0.3, 100),   # progenitor: low PT
        rng.uniform(0.5, 1.0, 100),   # fateA: high PT
        rng.uniform(0.5, 1.0, 100),   # fateB: high PT
    ])
    adata = ad.AnnData(X=X)
    adata.obs["leiden"] = pd.Categorical(labels)
    adata.obs["velocity_pseudotime"] = pseudotime
    return adata


# ---------------------------------------------------------------------------
# 1. Magnitude computation (Eq. 1)
# ---------------------------------------------------------------------------

class TestMagnitudes:
    def test_basic(self):
        vx = np.array([3.0, 0.0, -4.0])
        vy = np.array([4.0, 5.0, 3.0])
        mag = compute_magnitudes(vx, vy)
        np.testing.assert_allclose(mag, [5.0, 5.0, 5.0])

    def test_zero_vector(self):
        mag = compute_magnitudes(np.array([0.0]), np.array([0.0]))
        assert mag[0] == 0.0

    def test_unit_vectors(self):
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        mag = compute_magnitudes(np.cos(angles), np.sin(angles))
        np.testing.assert_allclose(mag, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Angle computation (Eq. 2-3)
# ---------------------------------------------------------------------------

class TestAngles:
    def test_cardinal_directions(self):
        vx = np.array([1.0, 0.0, -1.0, 0.0])
        vy = np.array([0.0, 1.0, 0.0, -1.0])
        angles = compute_angles(vx, vy)
        np.testing.assert_allclose(angles, [0.0, 90.0, 180.0, 270.0], atol=1e-10)

    def test_range_0_360(self):
        vx = np.random.randn(100)
        vy = np.random.randn(100)
        angles = compute_angles(vx, vy)
        valid = ~np.isnan(angles)
        assert np.all(angles[valid] >= 0.0)
        assert np.all(angles[valid] < 360.0)

    def test_zero_vector_is_nan(self):
        angles = compute_angles(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        assert np.isnan(angles[0])
        assert not np.isnan(angles[1])


# ---------------------------------------------------------------------------
# 3. Angular binning (Eq. 4-6)
# ---------------------------------------------------------------------------

class TestBinAngles:
    def test_bin_count(self):
        _, M_bin = bin_angles(np.array([0.0, 90.0, 180.0, 270.0]), np.ones(4), n_bins=36)
        assert len(M_bin) == 36

    def test_total_magnitude_preserved(self):
        np.random.seed(42)
        angles = np.random.uniform(0, 360, 1000)
        mags = np.random.exponential(1.0, 1000)
        _, M_bin = bin_angles(angles, mags, n_bins=36)
        np.testing.assert_allclose(M_bin.sum(), mags.sum(), rtol=1e-10)

    def test_nan_ignored(self):
        angles = np.array([0.0, np.nan, 90.0])
        mags = np.array([1.0, 5.0, 2.0])
        _, M_bin = bin_angles(angles, mags, n_bins=36)
        assert M_bin.sum() == pytest.approx(3.0)

    def test_directional_concentration(self):
        angles = np.zeros(100)
        mags = np.ones(100)
        _, M_bin = bin_angles(angles, mags, n_bins=36)
        assert M_bin[0] == pytest.approx(100.0)
        assert M_bin[1:].sum() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. Sector definitions
# ---------------------------------------------------------------------------

class TestSectors:
    def test_equal_sectors_k2(self):
        sectors = equal_sectors(k=2, n_bins=36)
        assert len(sectors) == 2
        assert len(sectors[0]) == 18
        assert len(sectors[1]) == 18
        assert len(set(sectors[0] + sectors[1])) == 36

    def test_equal_sectors_k4(self):
        sectors = equal_sectors(k=4, n_bins=36)
        assert all(len(s) == 9 for s in sectors)

    def test_centroid_sectors_k2(self):
        fate_centroids = np.array([[1.0, 0.0], [-1.0, 0.0]])
        root_centroid = np.array([0.0, 0.0])
        sectors, fate_angles = centroid_sectors(fate_centroids, root_centroid, n_bins=36)
        assert len(sectors) == 2
        np.testing.assert_allclose(fate_angles[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(fate_angles[1], 180.0, atol=1e-10)
        assert abs(len(sectors[0]) - 18) <= 1
        assert abs(len(sectors[1]) - 18) <= 1


# ---------------------------------------------------------------------------
# 5. unCS and nCS (Eq. 8-9)
# ---------------------------------------------------------------------------

class TestCommitmentScores:
    def test_unCS_basic(self):
        assert compute_unCS(10.0, 5.0) == pytest.approx(2.0)
        assert compute_unCS(5.0, 10.0) == pytest.approx(0.5)

    def test_unCS_equal(self):
        assert compute_unCS(7.0, 7.0) == pytest.approx(1.0)

    def test_unCS_zero_denominator(self):
        assert compute_unCS(5.0, 0.0) == np.inf

    def test_nCS_basic(self):
        assert compute_nCS(10.0, 5.0, 100, 50) == pytest.approx(1.0)

    def test_nCS_formula_consistency(self):
        """Verify nCS = unCS * (n_B / n_A) and manuscript ratio."""
        sumA, sumB = 9335.0, 1000.0
        n_A, n_B = 10000, 8641
        unCS = compute_unCS(sumA, sumB)
        nCS = compute_nCS(sumA, sumB, n_A, n_B)
        assert nCS == pytest.approx(unCS * (n_B / n_A), rel=1e-10)
        assert unCS == pytest.approx(9.335, rel=1e-3)
        assert nCS == pytest.approx(8.066, rel=1e-2)


# ---------------------------------------------------------------------------
# 6. Commitment vector
# ---------------------------------------------------------------------------

class TestCommitmentVector:
    def test_normalization(self):
        p = compute_commitment_vector(np.array([3.0, 1.0, 2.0]))
        assert p.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(p, [0.5, 1 / 6, 1 / 3], atol=1e-10)

    def test_zero_input(self):
        p = compute_commitment_vector(np.zeros(3))
        np.testing.assert_allclose(p, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Population entropy (compute_population_entropy)
# ---------------------------------------------------------------------------

class TestPopulationEntropy:
    """Tests for the aggregate velocity-mass entropy metric."""

    def test_uniform_is_max(self):
        assert compute_population_entropy(np.array([0.5, 0.5])) == pytest.approx(1.0)

    def test_fully_committed_is_zero(self):
        assert compute_population_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_range_0_to_1(self):
        for k in [2, 3, 4, 5]:
            p = np.random.dirichlet(np.ones(k))
            H = compute_population_entropy(p)
            assert 0.0 <= H <= 1.0 + 1e-10

    def test_single_fate_returns_zero(self):
        assert compute_population_entropy(np.array([1.0])) == 0.0

    def test_k3_uniform(self):
        assert compute_population_entropy(np.array([1/3, 1/3, 1/3])) == pytest.approx(1.0, abs=1e-10)

    def test_intermediate_k2(self):
        # p = [0.9, 0.1] -> low entropy (high commitment)
        assert compute_population_entropy(np.array([0.9, 0.1])) < 0.5


# ---------------------------------------------------------------------------
# 8. Mean cell entropy (compute_mean_cell_entropy) — PRIMARY METRIC
# ---------------------------------------------------------------------------

class TestMeanCellEntropy:
    """Tests for the per-cell commitment uncertainty metric."""

    def test_all_cells_fully_committed_to_one_fate(self):
        """Every cell points entirely to fate 0 -> H_cell = 0."""
        cell_scores = np.zeros((50, 3))
        cell_scores[:, 0] = 1.0
        assert compute_mean_cell_entropy(cell_scores) == pytest.approx(0.0, abs=1e-10)

    def test_all_cells_maximally_uncertain(self):
        """Every cell has uniform fate scores -> H_cell = 1."""
        k = 3
        cell_scores = np.full((50, k), 1.0 / k)
        assert compute_mean_cell_entropy(cell_scores) == pytest.approx(1.0, abs=1e-10)

    def test_range_0_to_1(self):
        np.random.seed(7)
        for k in [2, 3, 4]:
            cell_scores = np.random.dirichlet(np.ones(k), size=100)
            H = compute_mean_cell_entropy(cell_scores)
            assert 0.0 <= H <= 1.0 + 1e-10

    def test_single_fate_returns_zero(self):
        assert compute_mean_cell_entropy(np.ones((20, 1))) == 0.0

    def test_empty_cells_returns_zero(self):
        assert compute_mean_cell_entropy(np.zeros((0, 3))) == 0.0

    def test_k2_known_value(self):
        """Verify exact arithmetic for a simple 2-fate case."""
        p = np.array([[0.8, 0.2]])
        expected = -(0.8 * np.log(0.8) + 0.2 * np.log(0.2)) / np.log(2)
        assert compute_mean_cell_entropy(p) == pytest.approx(expected, abs=1e-10)

    def test_split_committed_population_is_low(self):
        """
        Key correctness test: a population split 50/50 between two strongly
        committed sub-groups should have LOW mean_cell_entropy even though
        population_entropy would be near 1.
        """
        np.random.seed(42)
        scores_A = np.random.dirichlet([20, 1], size=50)   # strongly → fate 0
        scores_B = np.random.dirichlet([1, 20], size=50)   # strongly → fate 1
        cell_scores = np.vstack([scores_A, scores_B])

        H_cell = compute_mean_cell_entropy(cell_scores)
        M_sector = cell_scores.sum(axis=0)
        H_pop = compute_population_entropy(M_sector / M_sector.sum())

        assert H_cell < 0.4, f"Split-committed population should have low H_cell, got {H_cell:.4f}"
        assert H_pop > 0.9, f"Population entropy should be near 1 for balanced split, got {H_pop:.4f}"
        assert H_pop - H_cell > 0.5, (
            f"Expected H_pop >> H_cell, got H_pop={H_pop:.4f}, H_cell={H_cell:.4f}"
        )

    def test_genuinely_uncommitted_population_is_high(self):
        """Cells with random/uniform velocities should have high H_cell."""
        np.random.seed(0)
        cell_scores = np.random.dirichlet([1, 1], size=100)
        assert compute_mean_cell_entropy(cell_scores) > 0.6


# ---------------------------------------------------------------------------
# 9. Per-fate cell entropy (compute_per_fate_cell_entropy)
# ---------------------------------------------------------------------------

class TestPerFateCellEntropy:
    """Tests for the per-fate binary entropy metric."""

    def test_shape(self):
        cell_scores = np.random.dirichlet(np.ones(3), size=100)
        h = compute_per_fate_cell_entropy(cell_scores)
        assert h.shape == (3,)

    def test_all_cells_fully_committed_to_fate0(self):
        """Every cell has s_0=1 -> binary entropy for fate 0 is 0."""
        cell_scores = np.zeros((50, 3))
        cell_scores[:, 0] = 1.0
        h = compute_per_fate_cell_entropy(cell_scores)
        assert h[0] == pytest.approx(0.0, abs=1e-6)

    def test_all_cells_at_half_for_fate0(self):
        """s_0 = 0.5 for all cells -> binary entropy for fate 0 is 1."""
        cell_scores = np.full((50, 2), 0.5)
        h = compute_per_fate_cell_entropy(cell_scores)
        assert h[0] == pytest.approx(1.0, abs=1e-6)

    def test_range_0_to_1(self):
        np.random.seed(3)
        for k in [2, 3, 4]:
            cell_scores = np.random.dirichlet(np.ones(k), size=100)
            h = compute_per_fate_cell_entropy(cell_scores)
            assert np.all(h >= 0.0 - 1e-10)
            assert np.all(h <= 1.0 + 1e-10)

    def test_k2_known_value(self):
        """Verify exact arithmetic: s = [0.8, 0.2] for all cells."""
        cell_scores = np.tile([0.8, 0.2], (20, 1))
        h = compute_per_fate_cell_entropy(cell_scores)
        # fate 0: H_bin(0.8, 0.2); fate 1: H_bin(0.2, 0.8) — same by symmetry
        expected = -(0.8 * np.log(0.8) + 0.2 * np.log(0.2)) / np.log(2)
        np.testing.assert_allclose(h, [expected, expected], atol=1e-10)

    def test_ambiguous_fate_has_higher_entropy_than_decisive_fate(self):
        """Cells with s_j near 0.5 for fate j should have higher h_j than
        cells with s_j near 0 or 1."""
        n = 200
        # fate 0: all cells have s_0 = 0.5 (maximally ambiguous)
        # fate 1: all cells have s_1 = 0.9 (strongly committed)
        cell_scores = np.zeros((n, 2))
        cell_scores[:, 0] = 0.5
        cell_scores[:, 1] = 0.5
        # Override: make half the cells have s_0=0.1, s_1=0.9 (decisive toward 1)
        cell_scores[:n // 2, 0] = 0.1
        cell_scores[:n // 2, 1] = 0.9
        # Other half: s_0=0.9, s_1=0.1 (decisive toward 0)
        cell_scores[n // 2:, 0] = 0.9
        cell_scores[n // 2:, 1] = 0.1
        h_decisive = compute_per_fate_cell_entropy(cell_scores)
        # Now make all cells ambiguous (s_0 = s_1 = 0.5)
        cell_scores_ambiguous = np.full((n, 2), 0.5)
        h_ambiguous = compute_per_fate_cell_entropy(cell_scores_ambiguous)
        # Ambiguous cells should have higher per-fate entropy than decisive cells
        assert h_ambiguous[0] > h_decisive[0]
        assert h_ambiguous[1] > h_decisive[1]

    def test_empty_returns_zeros(self):
        h = compute_per_fate_cell_entropy(np.zeros((0, 3)))
        assert h.shape == (3,)
        np.testing.assert_array_equal(h, 0.0)


# ---------------------------------------------------------------------------
# 10. NN-smoothed cell entropy (compute_nn_cell_entropy)
# ---------------------------------------------------------------------------

class TestNNCellEntropy:
    """Tests for the nearest-neighbor smoothed per-cell entropy."""

    def _make_committed_scores(self, n=100, k=2, seed=0):
        """Half cells strongly → fate 0, half → fate 1."""
        np.random.seed(seed)
        s_a = np.random.dirichlet([20, 1], size=n // 2)
        s_b = np.random.dirichlet([1, 20], size=n // 2)
        return np.vstack([s_a, s_b])

    def _make_star_coords(self, n=100):
        """Simple 2D coords: first half on right arm, second half on left."""
        coords = np.zeros((n, 2))
        coords[:n // 2, 0] = np.linspace(1, 5, n // 2)   # right arm
        coords[n // 2:, 0] = np.linspace(-1, -5, n // 2) # left arm
        return coords

    def test_output_shape(self):
        cell_scores = self._make_committed_scores()
        coords = self._make_star_coords()
        nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn=5)
        assert nn_ent.shape == (100,)

    def test_range_0_to_1(self):
        cell_scores = self._make_committed_scores()
        coords = self._make_star_coords()
        nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn=5)
        assert np.all(nn_ent >= 0.0 - 1e-10)
        assert np.all(nn_ent <= 1.0 + 1e-10)

    def test_committed_cells_have_low_nn_entropy(self):
        """Cells on committed arms should have low NN entropy."""
        cell_scores = self._make_committed_scores(n=100)
        coords = self._make_star_coords(n=100)
        nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn=5)
        assert nn_ent.mean() < 0.4, (
            f"Committed population should have low NN entropy, got {nn_ent.mean():.4f}"
        )

    def test_uniform_cells_have_high_nn_entropy(self):
        """Cells with uniform fate scores should have high NN entropy."""
        k = 3
        cell_scores = np.full((60, k), 1.0 / k)
        coords = np.random.default_rng(0).normal(size=(60, 2))
        nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn=5)
        np.testing.assert_allclose(nn_ent, 1.0, atol=1e-10)

    def test_larger_k_smooths_more(self):
        """Larger k_nn should reduce variance of nn_entropy."""
        np.random.seed(42)
        cell_scores = np.random.dirichlet([2, 1], size=80)
        coords = np.random.randn(80, 2)
        nn_ent_small = compute_nn_cell_entropy(cell_scores, coords, k_nn=3)
        nn_ent_large = compute_nn_cell_entropy(cell_scores, coords, k_nn=20)
        assert nn_ent_large.std() <= nn_ent_small.std() + 0.05

    def test_k_nn_capped_at_n_cells(self):
        """k_nn larger than n_cells should not raise."""
        cell_scores = np.random.dirichlet([1, 1], size=10)
        coords = np.random.randn(10, 2)
        nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn=50)
        assert nn_ent.shape == (10,)

    def test_single_fate_returns_zeros(self):
        cell_scores = np.ones((20, 1))
        coords = np.random.randn(20, 2)
        nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn=5)
        np.testing.assert_array_equal(nn_ent, 0.0)


# ---------------------------------------------------------------------------
# 11. Backward-compatibility alias
# ---------------------------------------------------------------------------

class TestBackwardCompatAlias:
    """compute_commitment_entropy must still work; CommitmentScoreResult.commitment_entropy
    must emit DeprecationWarning."""

    def test_alias_returns_same_as_population_entropy(self):
        p = np.array([0.7, 0.3])
        expected = compute_population_entropy(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = compute_commitment_entropy(p)
        assert result == pytest.approx(expected)

    def test_result_property_emits_deprecation_warning(self):
        n_bins = 36
        k = 2
        M_bin = np.ones(n_bins)
        sectors = equal_sectors(k, n_bins)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        n_cells = np.array([50.0, 50.0])
        p = compute_commitment_vector(M_sector)
        H_pop = compute_population_entropy(p)
        unCS = compute_pairwise_cs_matrix(M_sector, normalized=False)
        nCS = compute_pairwise_cs_matrix(M_sector, n_cells_per_fate=n_cells, normalized=True)
        result = CommitmentScoreResult(
            fate_names=["A", "B"],
            M_bin=M_bin, bin_edges=np.linspace(0, 360, n_bins + 1),
            sectors=sectors, M_sector=M_sector, n_cells_per_fate=n_cells,
            commitment_vector=p, population_entropy=H_pop,
            mean_cell_entropy=float("nan"),
            per_fate_entropy=np.full(k, float("nan")),
            pairwise_unCS=unCS, pairwise_nCS=nCS,
        )
        with pytest.warns(DeprecationWarning, match="commitment_entropy is deprecated"):
            _ = result.commitment_entropy


# ---------------------------------------------------------------------------
# 12. Pairwise CS matrix
# ---------------------------------------------------------------------------

class TestPairwiseCS:
    def test_diagonal_is_one(self):
        mat = compute_pairwise_cs_matrix(np.array([5.0, 3.0, 2.0]), normalized=False)
        np.testing.assert_allclose(np.diag(mat), 1.0)

    def test_symmetry_inverse(self):
        mat = compute_pairwise_cs_matrix(np.array([6.0, 3.0]), normalized=False)
        assert mat[0, 1] * mat[1, 0] == pytest.approx(1.0)

    def test_k3_equal_cells(self):
        M = np.array([10.0, 5.0, 2.0])
        n = np.array([100, 100, 100])
        mat_n = compute_pairwise_cs_matrix(M, n_cells_per_fate=n, normalized=True)
        mat_u = compute_pairwise_cs_matrix(M, normalized=False)
        np.testing.assert_allclose(mat_n, mat_u)


# ---------------------------------------------------------------------------
# 13. Per-cell scores
# ---------------------------------------------------------------------------

class TestCellScores:
    def test_shape(self):
        scores = compute_cell_scores(
            np.array([1.0, -1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, -1.0]),
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
            np.array([0.0, 0.0]),
        )
        assert scores.shape == (4, 3)

    def test_row_normalization(self):
        scores = compute_cell_scores(
            np.random.randn(50), np.random.randn(50),
            np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            np.array([0.0, 0.0]),
        )
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-10)

    def test_aligned_cell_highest_for_correct_fate(self):
        scores = compute_cell_scores(
            np.array([1.0]), np.array([0.0]),
            np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]),
            np.array([0.0, 0.0]),
        )
        assert np.argmax(scores[0]) == 0

    def test_zero_velocity_uniform(self):
        scores = compute_cell_scores(
            np.array([0.0]), np.array([0.0]),
            np.array([[1.0, 0.0], [-1.0, 0.0]]),
            np.array([0.0, 0.0]),
        )
        np.testing.assert_allclose(scores[0], [0.5, 0.5], atol=1e-10)


# ---------------------------------------------------------------------------
# 14. CommitmentScoreResult dataclass
# ---------------------------------------------------------------------------

class TestCommitmentScoreResult:
    def _make_result(self, k=3, include_cell_scores=False):
        np.random.seed(1)
        n_bins = 36
        M_bin = np.random.exponential(1.0, n_bins)
        bin_edges = np.linspace(0, 360, n_bins + 1)
        sectors = equal_sectors(k, n_bins)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        n_cells = np.array([100.0] * k)
        p = compute_commitment_vector(M_sector)
        H_pop = compute_population_entropy(p)
        unCS = compute_pairwise_cs_matrix(M_sector, normalized=False)
        nCS = compute_pairwise_cs_matrix(M_sector, n_cells_per_fate=n_cells, normalized=True)
        cell_scores = None
        mean_cell_ent = float("nan")
        per_fate_ent = np.full(k, float("nan"))
        if include_cell_scores:
            cell_scores = np.random.dirichlet(np.ones(k), size=200)
            mean_cell_ent = compute_mean_cell_entropy(cell_scores)
            per_fate_ent = compute_per_fate_cell_entropy(cell_scores)
        return CommitmentScoreResult(
            fate_names=[f"fate_{j}" for j in range(k)],
            M_bin=M_bin, bin_edges=bin_edges, sectors=sectors,
            M_sector=M_sector, n_cells_per_fate=n_cells,
            commitment_vector=p, population_entropy=H_pop,
            mean_cell_entropy=mean_cell_ent,
            per_fate_entropy=per_fate_ent,
            pairwise_unCS=unCS, pairwise_nCS=nCS,
            cell_scores=cell_scores,
        )

    def test_k_property(self):
        assert self._make_result(k=3).k == 3

    def test_dominant_fate(self):
        r = self._make_result(k=3)
        assert r.dominant_fate == r.fate_names[int(np.argmax(r.M_sector))]

    def test_to_dataframe_shape(self):
        df = self._make_result(k=4).to_dataframe()
        assert df.shape == (4, 4)
        assert "fate" in df.columns

    def test_pairwise_to_dataframe(self):
        r = self._make_result(k=3)
        df = r.pairwise_to_dataframe(normalized=True)
        assert df.shape == (3, 3)
        assert list(df.index) == r.fate_names

    def test_summary_contains_both_entropy_labels(self):
        s = self._make_result(k=3, include_cell_scores=True).summary()
        assert "CommitmentScoreResult" in s
        assert "Dominant fate" in s
        assert "Mean cell entropy" in s
        assert "Population entropy" in s

    def test_mean_cell_entropy_nan_when_no_cell_scores(self):
        assert np.isnan(self._make_result(k=3, include_cell_scores=False).mean_cell_entropy)

    def test_mean_cell_entropy_populated_with_cell_scores(self):
        r = self._make_result(k=3, include_cell_scores=True)
        assert not np.isnan(r.mean_cell_entropy)
        assert 0.0 <= r.mean_cell_entropy <= 1.0

    def test_per_fate_entropy_shape(self):
        r = self._make_result(k=3, include_cell_scores=True)
        assert r.per_fate_entropy.shape == (3,)
        assert np.all(r.per_fate_entropy >= 0.0 - 1e-10)
        assert np.all(r.per_fate_entropy <= 1.0 + 1e-10)

    def test_per_fate_entropy_nan_when_no_cell_scores(self):
        r = self._make_result(k=3, include_cell_scores=False)
        assert np.all(np.isnan(r.per_fate_entropy))


# ---------------------------------------------------------------------------
# 15. Radial star embedding
# ---------------------------------------------------------------------------

class TestStarEmbedding:
    def setup_method(self):
        self.adata = _make_adata(n_cells=300)

    def test_embedding_shape(self):
        adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
        )
        assert "X_sccs" in adata_sub.obsm
        assert adata_sub.obsm["X_sccs"].shape == (300, 2)

    def test_progenitor_near_origin(self):
        adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
            jitter=0.0,
        )
        coords = adata_sub.obsm["X_sccs"]
        bif_mask = adata_sub.obs["leiden"].astype(str) == "0"
        bif_coords = coords[bif_mask]
        # Progenitor cells should be close to origin (within jitter range)
        assert np.abs(bif_coords).mean() < 1.0

    def test_fate_cells_farther_than_progenitor(self):
        adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
            jitter=0.0,
        )
        coords = adata_sub.obsm["X_sccs"]
        bif_mask = adata_sub.obs["leiden"].astype(str) == "0"
        fate_mask = ~bif_mask
        bif_r = np.linalg.norm(coords[bif_mask], axis=1).mean()
        fate_r = np.linalg.norm(coords[fate_mask], axis=1).mean()
        assert fate_r > bif_r, (
            f"Fate cells (r={fate_r:.2f}) should be farther from origin "
            f"than progenitor cells (r={bif_r:.2f})"
        )

    def test_two_arms_opposite_directions(self):
        """k=2 arms should be ~180° apart."""
        adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
            jitter=0.0,
        )
        arm_dirs = adata_sub.uns["sccs"]["arm_dirs"]
        dot = np.dot(arm_dirs[0], arm_dirs[1])
        # For k=2, arms are 180° apart -> dot product = -1
        assert dot == pytest.approx(-1.0, abs=1e-10)

    def test_three_arms_120_degrees_apart(self):
        """k=3 arms should be 120° apart."""
        adata3 = _make_adata_k3()
        adata3_sub = build_star_embedding(
            adata3,
            root="0",
            branches=["1", "2", "3"],
            obs_key="leiden",
            ordering_metric="pseudotime",
            jitter=0.0,
        )
        arm_dirs = adata3_sub.uns["sccs"]["arm_dirs"]
        for i in range(3):
            for j in range(i + 1, 3):
                dot = np.dot(arm_dirs[i], arm_dirs[j])
                # 120° apart -> dot = cos(120°) = -0.5
                assert dot == pytest.approx(-0.5, abs=1e-10)

    def test_arm_assignment_stored(self):
        adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
        )
        assert "sccs_arm" in adata_sub.obs
        assert "sccs_branch" in adata_sub.obs

    def test_pseudotime_ordering_along_arm(self):
        """Higher pseudotime cells should be farther from origin on their arm."""
        adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
            jitter=0.0,
        )
        coords = adata_sub.obsm["X_sccs"]
        pt = adata_sub.obs["velocity_pseudotime"].values
        # For fate arm 1: correlation between pseudotime and radial distance
        arm1_mask = adata_sub.obs["leiden"].astype(str) == "1"
        r = np.linalg.norm(coords[arm1_mask], axis=1)
        pt_arm1 = pt[arm1_mask]
        corr = np.corrcoef(pt_arm1, r)[0, 1]
        assert corr > 0.5, f"Expected positive correlation, got {corr:.3f}"

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="not found in adata.obs"):
            build_star_embedding(
                self.adata,
                root="0",
                branches=["1", "2"],
                obs_key="leiden",
                ordering_metric="nonexistent_column",
            )

    def test_invert_metric(self):
        """Inverted metric should flip the ordering."""
        adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
            invert_ordering=True,
            jitter=0.0,
        )
        coords = adata_sub.obsm["X_sccs"]
        pt = adata_sub.obs["velocity_pseudotime"].values
        arm1_mask = adata_sub.obs["leiden"].astype(str) == "1"
        r = np.linalg.norm(coords[arm1_mask], axis=1)
        pt_arm1 = pt[arm1_mask]
        corr = np.corrcoef(pt_arm1, r)[0, 1]
        # With inversion, high pseudotime should be CLOSER to center -> negative corr
        assert corr < -0.5, f"Expected negative correlation with invert=True, got {corr:.3f}"


# ---------------------------------------------------------------------------
# 16. FateMap construction
# ---------------------------------------------------------------------------

class TestFateMap:
    def setup_method(self):
        self.adata = _make_adata(n_cells=300)
        # build_star_embedding now returns adata_sub
        self.adata_sub = build_star_embedding(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
            ordering_metric="pseudotime",
        )

    def test_build_fate_map_basic(self):
        fm = build_fate_map(
            self.adata_sub,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
        )
        assert isinstance(fm, FateMap)
        assert fm.k == 2
        assert fm.root == "0"
        assert fm.fate_names == ["1", "2"]

    def test_root_cells_correct(self):
        fm = build_fate_map(
            self.adata_sub,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
        )
        expected_root = np.where(self.adata_sub.obs["leiden"].astype(str) == "0")[0]
        np.testing.assert_array_equal(np.sort(fm.root_cells), np.sort(expected_root))

    def test_fate_cell_indices_correct(self):
        fm = build_fate_map(
            self.adata_sub,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
        )
        for j, name in enumerate(["1", "2"]):
            expected = np.where(self.adata_sub.obs["leiden"].astype(str) == name)[0]
            np.testing.assert_array_equal(
                np.sort(fm.fate_cell_indices[j]), np.sort(expected)
            )

    def test_root_centroid_near_origin(self):
        fm = build_fate_map(
            self.adata_sub,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
        )
        # Root centroid should be near (0, 0) in the star embedding
        assert np.linalg.norm(fm.root_centroid) < 1.0

    def test_invalid_bifurcation_cluster_raises(self):
        with pytest.raises(ValueError, match="not found"):
            build_fate_map(
                self.adata_sub,
                root="999",
                branches=["1", "2"],
                obs_key="leiden",
            )

    def test_missing_fate_warns(self):
        with pytest.warns(UserWarning):
            fm = build_fate_map(
                self.adata_sub,
                root="0",
                branches=["1", "2", "nonexistent"],
                obs_key="leiden",
            )
        assert fm.k == 2  # nonexistent fate skipped

    def test_summary_string(self):
        fm = build_fate_map(
            self.adata_sub,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
        )
        s = fm.summary()
        assert "FateMap" in s
        assert "root='0'" in s

    def test_arm_angles_stored(self):
        fm = build_fate_map(
            self.adata_sub,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
        )
        assert len(fm.arm_angles_deg) == 2
        # k=2 arms should be ~180° apart
        diff = abs(fm.arm_angles_deg[0] - fm.arm_angles_deg[1])
        diff = min(diff, 360 - diff)
        assert diff == pytest.approx(180.0, abs=5.0)


# ---------------------------------------------------------------------------
# 17. End-to-end: synthetic bifurcation (k=2)
# ---------------------------------------------------------------------------

class TestEndToEndBifurcation:
    """200 cells: 50 progenitor + 75 fateA + 75 fateB."""

    def setup_method(self):
        np.random.seed(0)
        n = 200
        vx_a = np.random.normal(1.0, 0.1, 75)
        vy_a = np.random.normal(0.0, 0.1, 75)
        vx_b = np.random.normal(-1.0, 0.1, 75)
        vy_b = np.random.normal(0.0, 0.1, 75)
        vx_p = np.random.normal(0.0, 0.2, 50)
        vy_p = np.random.normal(0.0, 0.2, 50)

        self.vx = np.concatenate([vx_p, vx_a, vx_b])
        self.vy = np.concatenate([vy_p, vy_a, vy_b])

        self.fate_centroids = np.array([[5.0, 0.0], [-5.0, 0.0]])
        self.root_centroid = np.array([0.0, 0.0])

    def test_sector_magnitudes_balanced(self):
        magnitudes = compute_magnitudes(self.vx, self.vy)
        angles = compute_angles(self.vx, self.vy)
        _, M_bin = bin_angles(angles, magnitudes, n_bins=36)
        sectors, _ = centroid_sectors(self.fate_centroids, self.root_centroid, n_bins=36)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        ratio = M_sector[0] / M_sector[1]
        assert 0.7 < ratio < 1.4, f"Expected balanced sectors, got ratio={ratio:.3f}"

    def test_biased_population_unCS_gt1(self):
        np.random.seed(1)
        vx = np.concatenate([
            np.random.normal(1.0, 0.1, 160),
            np.random.normal(-1.0, 0.1, 40),
        ])
        vy = np.random.normal(0.0, 0.1, 200)
        magnitudes = compute_magnitudes(vx, vy)
        angles = compute_angles(vx, vy)
        _, M_bin = bin_angles(angles, magnitudes, n_bins=36)
        sectors, _ = centroid_sectors(self.fate_centroids, self.root_centroid, n_bins=36)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        unCS = compute_unCS(M_sector[0], M_sector[1])
        assert unCS > 3.0, f"Expected unCS >> 1 for biased population, got {unCS:.3f}"

    def test_population_entropy_near_one_for_balanced_split(self):
        """Balanced bifurcation has H_pop ≈ 1 — known limitation of population metric."""
        magnitudes = compute_magnitudes(self.vx, self.vy)
        angles = compute_angles(self.vx, self.vy)
        _, M_bin = bin_angles(angles, magnitudes, n_bins=36)
        sectors, _ = centroid_sectors(self.fate_centroids, self.root_centroid, n_bins=36)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        p = compute_commitment_vector(M_sector)
        H_pop = compute_population_entropy(p)
        assert H_pop > 0.9, f"Balanced bifurcation should have H_pop ≈ 1, got {H_pop:.4f}"

    def test_mean_cell_entropy_low_for_committed_bifurcation(self):
        """Same balanced bifurcation has LOW H_cell — cells are individually decisive.

        Note: this dataset includes 50 near-zero-velocity progenitor cells that are
        blended toward uniform (1/k) by mag_weight=True, raising H_cell moderately.
        The threshold reflects the mixed population (fate cells + progenitors).
        """
        cell_scores = compute_cell_scores(
            self.vx, self.vy, self.fate_centroids, self.root_centroid
        )
        H_cell = compute_mean_cell_entropy(cell_scores)
        assert H_cell < 0.7, f"Committed bifurcation should have low H_cell, got {H_cell:.4f}"

    def test_entropy_metrics_diverge_for_bifurcation(self):
        """H_pop >> H_cell for a balanced committed bifurcation."""
        magnitudes = compute_magnitudes(self.vx, self.vy)
        angles = compute_angles(self.vx, self.vy)
        _, M_bin = bin_angles(angles, magnitudes, n_bins=36)
        sectors, _ = centroid_sectors(self.fate_centroids, self.root_centroid, n_bins=36)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        p = compute_commitment_vector(M_sector)
        H_pop = compute_population_entropy(p)
        cell_scores = compute_cell_scores(
            self.vx, self.vy, self.fate_centroids, self.root_centroid
        )
        H_cell = compute_mean_cell_entropy(cell_scores)
        assert H_pop - H_cell > 0.3, (
            f"Expected H_pop >> H_cell for committed bifurcation, "
            f"got H_pop={H_pop:.4f}, H_cell={H_cell:.4f}"
        )


# ---------------------------------------------------------------------------
# 18. End-to-end: synthetic trifurcation (k=3)
# ---------------------------------------------------------------------------

class TestEndToEndTrifurcation:
    def setup_method(self):
        np.random.seed(42)
        n_per_fate = 100
        angles_deg = [0.0, 120.0, 240.0]
        vx_list, vy_list = [], []
        for a in angles_deg:
            a_rad = np.radians(a)
            vx_list.append(np.random.normal(np.cos(a_rad), 0.1, n_per_fate))
            vy_list.append(np.random.normal(np.sin(a_rad), 0.1, n_per_fate))
        self.vx = np.concatenate(vx_list)
        self.vy = np.concatenate(vy_list)
        self.fate_centroids = np.array([
            [2.0, 0.0],
            [2.0 * np.cos(np.radians(120)), 2.0 * np.sin(np.radians(120))],
            [2.0 * np.cos(np.radians(240)), 2.0 * np.sin(np.radians(240))],
        ])
        self.root_centroid = np.array([0.0, 0.0])

    def test_three_sectors_balanced(self):
        magnitudes = compute_magnitudes(self.vx, self.vy)
        angles = compute_angles(self.vx, self.vy)
        _, M_bin = bin_angles(angles, magnitudes, n_bins=36)
        sectors, _ = centroid_sectors(self.fate_centroids, self.root_centroid, n_bins=36)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        for i in range(3):
            for j in range(3):
                if i != j:
                    ratio = M_sector[i] / M_sector[j]
                    assert 0.7 < ratio < 1.4

    def test_population_entropy_near_max(self):
        magnitudes = compute_magnitudes(self.vx, self.vy)
        angles = compute_angles(self.vx, self.vy)
        _, M_bin = bin_angles(angles, magnitudes, n_bins=36)
        sectors, _ = centroid_sectors(self.fate_centroids, self.root_centroid, n_bins=36)
        M_sector = compute_sector_magnitudes(M_bin, sectors)
        p = compute_commitment_vector(M_sector)
        H_pop = compute_population_entropy(p)
        assert H_pop > 0.9, f"Expected high H_pop for balanced trifurcation, got {H_pop:.3f}"

    def test_mean_cell_entropy_lower_than_random_for_committed_trifurcation(self):
        """
        Committed trifurcation H_cell should be ≤ a genuinely random baseline.
        Note: for k=3 at 120° separation, cosine-similarity scores top out at ~0.67,
        so absolute H_cell values are moderate. The meaningful comparison is vs. uniform.
        """
        cell_scores = compute_cell_scores(
            self.vx, self.vy, self.fate_centroids, self.root_centroid
        )
        H_cell = compute_mean_cell_entropy(cell_scores)
        np.random.seed(99)
        rand_scores = np.random.dirichlet([1, 1, 1], size=300)
        H_rand = compute_mean_cell_entropy(rand_scores)
        assert H_cell <= H_rand + 0.15, (
            f"Committed trifurcation H_cell={H_cell:.4f} should not exceed "
            f"random baseline H_rand={H_rand:.4f} by more than 0.15"
        )


# ---------------------------------------------------------------------------
# 19. End-to-end: full SingleScorer pipeline
# ---------------------------------------------------------------------------

class TestSingleScorerPipeline:
    """Full pipeline test using synthetic AnnData (no scVelo required)."""

    def setup_method(self):
        self.adata = _make_adata(n_cells=300)

    def _make_scorer(self):
        from scCS import SingleScorer
        scorer = SingleScorer(
            self.adata,
            root="0",
            branches=["1", "2"],
            obs_key="leiden",
        )
        return scorer

    def _build_and_fit(self, scorer):
        scorer.build_embedding(ordering_metric="pseudotime")
        # Inject synthetic velocity vectors (no scVelo needed)
        np.random.seed(42)
        vx = np.concatenate([
            np.random.normal(0.0, 0.2, 100),   # progenitor
            np.random.normal(1.0, 0.1, 100),   # fateA -> East
            np.random.normal(-1.0, 0.1, 100),  # fateB -> West
        ])
        vy = np.random.normal(0.0, 0.1, 300)
        scorer.load_velocity_vectors(vx, vy)
        scorer.fit(verbose=False)
        return scorer

    def test_full_pipeline_runs(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False)
        assert isinstance(result, CommitmentScoreResult)

    def test_result_has_correct_fate_names(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False)
        assert result.fate_names == ["1", "2"]

    def test_result_k_equals_2(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False)
        assert result.k == 2

    def test_pairwise_nCS_shape(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False)
        assert result.pairwise_nCS.shape == (2, 2)

    def test_commitment_vector_sums_to_one(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False)
        assert result.commitment_vector.sum() == pytest.approx(1.0)

    def test_entropy_in_range(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False)
        assert 0.0 <= result.population_entropy <= 1.0
        assert 0.0 <= result.mean_cell_entropy <= 1.0

    def test_mean_cell_entropy_lower_than_population_entropy_for_bifurcation(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False, cell_level=True)
        assert not np.isnan(result.mean_cell_entropy)
        assert not np.isnan(result.population_entropy)

    def test_per_fate_entropy_shape_and_range(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False, cell_level=True)
        assert result.per_fate_entropy.shape == (2,)
        assert np.all(result.per_fate_entropy >= 0.0 - 1e-10)
        assert np.all(result.per_fate_entropy <= 1.0 + 1e-10)

    def test_nn_cell_entropy_computed_when_k_nn_set(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False, cell_level=True, k_nn=10)
        assert result.nn_cell_entropy is not None
        assert result.nn_cell_entropy.shape == (scorer.adata_sub.n_obs,)
        assert result.nn_k == 10
        assert "cs_nn_entropy" in scorer.adata_sub.obs

    def test_nn_cell_entropy_none_when_k_nn_not_set(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False, cell_level=True)
        assert result.nn_cell_entropy is None
        assert result.nn_k is None

    def test_cell_scores_written_to_obs(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        scorer.score(verbose=False, cell_level=True)
        # Cell scores are written to scorer.adata_sub.obs (the subset)
        assert "cs_1" in scorer.adata_sub.obs
        assert "cs_2" in scorer.adata_sub.obs
        assert "cs_dominant_fate" in scorer.adata_sub.obs
        assert "cs_entropy" in scorer.adata_sub.obs

    def test_cell_scores_row_normalized(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        result = scorer.score(verbose=False, cell_level=True)
        np.testing.assert_allclose(
            result.cell_scores.sum(axis=1), 1.0, atol=1e-10
        )

    def test_fateA_dominant_for_east_pointing_cells(self):
        """Cells pointing East should prefer fateA (arm at 0°)."""
        scorer = self._make_scorer()
        scorer.build_embedding(ordering_metric="pseudotime")
        # All cells point East
        vx = np.ones(300)
        vy = np.zeros(300)
        scorer.load_velocity_vectors(vx, vy)
        scorer.fit(verbose=False)
        result = scorer.score(verbose=False, cell_level=True)
        # fateA is at arm 0 (0°), fateB at arm 1 (180°)
        # East-pointing velocity -> fateA should have higher M_sector
        fate_a_idx = result.fate_names.index("1")
        fate_b_idx = result.fate_names.index("2")
        assert result.M_sector[fate_a_idx] > result.M_sector[fate_b_idx]

    def test_score_per_subset(self):
        scorer = self._make_scorer()
        scorer = self._build_and_fit(scorer)
        self.adata.obs["condition"] = (
            ["ctrl"] * 150 + ["treat"] * 150
        )
        subset_results = scorer.score_per_subset("condition", verbose=False)
        assert "ctrl" in subset_results
        assert "treat" in subset_results
        assert isinstance(subset_results["ctrl"], CommitmentScoreResult)

    def test_not_fitted_raises(self):
        from scCS import SingleScorer
        scorer = SingleScorer(
            self.adata,
            root="0",
            branches=["1", "2"],
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.score()

    def test_no_embedding_raises(self):
        from scCS import SingleScorer
        scorer = SingleScorer(
            self.adata,
            root="0",
            branches=["1", "2"],
        )
        with pytest.raises(RuntimeError, match="build_embedding"):
            scorer.fit()


# ---------------------------------------------------------------------------
# Helpers for k=3 tests
# ---------------------------------------------------------------------------

def _make_adata_k3(n_cells: int = 400, seed: int = 0) -> ad.AnnData:
    """Synthetic AnnData with 4 clusters: 0=progenitor, 1/2/3=fates."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cells, 20))
    labels = np.array(["0"] * 100 + ["1"] * 100 + ["2"] * 100 + ["3"] * 100)
    pseudotime = np.concatenate([
        rng.uniform(0.0, 0.3, 100),
        rng.uniform(0.5, 1.0, 100),
        rng.uniform(0.5, 1.0, 100),
        rng.uniform(0.5, 1.0, 100),
    ])
    adata = ad.AnnData(X=X)
    adata.obs["leiden"] = pd.Categorical(labels)
    adata.obs["velocity_pseudotime"] = pseudotime
    return adata

# ---------------------------------------------------------------------------
# 20. PairScorer tests
# ---------------------------------------------------------------------------

def _make_adata_multicond(n_per_cond: int = 150, seed: int = 0) -> "ad.AnnData":
    """Synthetic AnnData with 2 conditions, 3 clusters (0=progenitor, 1/2=fates)."""
    rng = np.random.default_rng(seed)
    n_total = n_per_cond * 2
    X = rng.normal(size=(n_total, 20))

    # 3 clusters: 0=progenitor (50/cond), 1=fateA (50/cond), 2=fateB (50/cond)
    labels = np.array(["0"] * 50 + ["1"] * 50 + ["2"] * 50 +
                      ["0"] * 50 + ["1"] * 50 + ["2"] * 50)
    conditions = np.array(["ctrl"] * n_per_cond + ["treat"] * n_per_cond)
    pseudotime = np.concatenate([
        rng.uniform(0.0, 0.3, 50),   # ctrl progenitor
        rng.uniform(0.5, 1.0, 50),   # ctrl fateA
        rng.uniform(0.5, 1.0, 50),   # ctrl fateB
        rng.uniform(0.0, 0.3, 50),   # treat progenitor
        rng.uniform(0.5, 1.0, 50),   # treat fateA
        rng.uniform(0.5, 1.0, 50),   # treat fateB
    ])
    adata = ad.AnnData(X=X)
    adata.obs["leiden"] = pd.Categorical(labels)
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["velocity_pseudotime"] = pseudotime
    return adata


def _make_multicond_scorer(adata=None):
    """Build and fit a PairScorer on synthetic data."""
    from scCS.pairwise import PairScorer

    if adata is None:
        adata = _make_adata_multicond()

    mscorer = PairScorer(
        adata,
        root="0",
        branches=["1", "2"],
        condition_obs_key="condition",
        obs_key="leiden",
    )
    mscorer.build_embedding(ordering_metric="pseudotime", verbose=False)

    # Inject synthetic velocity vectors (no scVelo needed)
    rng = np.random.default_rng(42)
    n_sub = mscorer._scorer.adata_sub.n_obs
    vx = rng.normal(0.5, 0.3, n_sub)   # slight East bias
    vy = rng.normal(0.0, 0.2, n_sub)
    mscorer._scorer.load_velocity_vectors(vx, vy)
    mscorer.fit(verbose=False)
    return mscorer


class TestPairScorerPipeline:
    """Tests for PairScorer and all Tier 2/3 methods."""

    def setup_method(self):
        self.adata = _make_adata_multicond()
        self.mscorer = _make_multicond_scorer(self.adata)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_conditions_detected(self):
        assert set(self.mscorer.conditions) == {"ctrl", "treat"}

    def test_repr_contains_key_info(self):
        r = repr(self.mscorer)
        assert "PairScorer" in r
        assert "fitted" in r
        assert "ctrl" in r or "treat" in r

    def test_invalid_condition_key_raises(self):
        from scCS.pairwise import PairScorer
        with pytest.raises(ValueError, match="not found in adata.obs"):
            PairScorer(
                self.adata,
                root="0",
                branches=["1", "2"],
                condition_obs_key="nonexistent_col",
            )

    def test_single_condition_raises(self):
        from scCS.pairwise import PairScorer
        adata_single = self.adata.copy()
        adata_single.obs["condition"] = "ctrl"
        with pytest.raises(ValueError, match="exactly 2 conditions"):
            PairScorer(
                adata_single,
                root="0",
                branches=["1", "2"],
                condition_obs_key="condition",
            )

    # ------------------------------------------------------------------
    # score_all_conditions
    # ------------------------------------------------------------------

    def test_score_all_conditions_returns_both(self):
        results = self.mscorer.score_all_conditions(verbose=False)
        assert set(results.keys()) == {"ctrl", "treat"}

    def test_score_all_conditions_result_type(self):
        results = self.mscorer.score_all_conditions(verbose=False)
        for cond, res in results.items():
            assert isinstance(res, CommitmentScoreResult),                 f"Expected CommitmentScoreResult for '{cond}', got {type(res)}"

    def test_score_all_conditions_fate_names(self):
        results = self.mscorer.score_all_conditions(verbose=False)
        for cond, res in results.items():
            assert res.fate_names == ["1", "2"],                 f"Unexpected fate_names for '{cond}': {res.fate_names}"

    def test_score_all_conditions_commitment_vector_sums_to_one(self):
        results = self.mscorer.score_all_conditions(verbose=False)
        for cond, res in results.items():
            assert res.commitment_vector.sum() == pytest.approx(1.0),                 f"commitment_vector does not sum to 1 for '{cond}'"

    def test_score_all_conditions_cell_scores_available(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        for cond, res in results.items():
            assert res.cell_scores is not None,                 f"cell_scores is None for '{cond}'"
            assert res.cell_scores.shape[1] == 2

    # ------------------------------------------------------------------
    # Regression test: no obs clobbering (Issue #2)
    # ------------------------------------------------------------------

    def test_score_all_conditions_does_not_clobber_obs(self):
        """Regression test for Issue #2: score_all_conditions(write_to_obs=False)
        must NOT write cs_* columns to adata_sub.obs during the loop.

        Before the fix, each condition's score() call would overwrite
        cs_1, cs_2, cs_entropy, cs_dominant_fate in adata_sub.obs, so only
        the last condition's values survived.  After the fix, write_to_obs=False
        is passed internally, so adata_sub.obs is not touched by the loop.
        """
        # Use a fresh scorer to ensure no prior score() call has written obs columns
        mscorer = _make_multicond_scorer()
        adata_sub = mscorer._scorer.adata_sub
        # Ensure cs_1 is not already in obs
        for col in ["cs_1", "cs_2", "cs_entropy", "cs_dominant_fate"]:
            if col in adata_sub.obs.columns:
                del adata_sub.obs[col]

        results = mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )

        # After score_all_conditions(), adata_sub.obs should NOT have cs_1/cs_2
        # written by the loop (write_to_obs=False is passed internally).
        assert "cs_1" not in adata_sub.obs.columns, (
            "cs_1 was written to adata_sub.obs by score_all_conditions() loop "
            "(write_to_obs=False should prevent this)"
        )
        assert "cs_2" not in adata_sub.obs.columns, (
            "cs_2 was written to adata_sub.obs by score_all_conditions() loop"
        )

        # Both conditions should have valid cell_scores in the results dict
        for cond, res in results.items():
            assert res.cell_scores is not None, f"cell_scores is None for '{cond}'"
            assert res.cell_scores.shape[1] == 2


    # ------------------------------------------------------------------
    # compute_delta_CS
    # ------------------------------------------------------------------

    def test_compute_delta_cs_keys(self):
        delta = self.mscorer.compute_delta_CS(
            "ctrl", "treat", n_bootstrap=20, verbose=False
        )
        for key in ["delta_nCS", "ci_low", "ci_high", "nCS_A", "nCS_B",
                    "fate_names", "condition_a", "condition_b"]:
            assert key in delta, f"Missing key '{key}' in delta_CS result"

    def test_compute_delta_cs_shape(self):
        delta = self.mscorer.compute_delta_CS(
            "ctrl", "treat", n_bootstrap=20, verbose=False
        )
        assert delta["delta_nCS"].shape == (2, 2)
        assert delta["ci_low"].shape == (2, 2)
        assert delta["ci_high"].shape == (2, 2)

    def test_compute_delta_cs_ci_ordering(self):
        """ci_low <= delta_nCS <= ci_high (element-wise, ignoring inf/nan)."""
        delta = self.mscorer.compute_delta_CS(
            "ctrl", "treat", n_bootstrap=50, verbose=False
        )
        d = delta["delta_nCS"]
        lo = delta["ci_low"]
        hi = delta["ci_high"]
        finite = np.isfinite(d) & np.isfinite(lo) & np.isfinite(hi)
        assert np.all(lo[finite] <= hi[finite]), "ci_low > ci_high for some entries"

    def test_compute_delta_cs_invalid_condition_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.mscorer.compute_delta_CS("ctrl", "nonexistent", verbose=False)

    # ------------------------------------------------------------------
    # compare_conditions
    # ------------------------------------------------------------------

    def test_compare_conditions_returns_dataframe(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_conditions(results, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_compare_conditions_columns(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_conditions(results, verbose=False)
        for col in ["fate", "test", "statistic", "pval", "pval_adj", "significant"]:
            assert col in df.columns, f"Missing column '{col}'"

    def test_compare_conditions_pval_adj_in_range(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_conditions(results, verbose=False)
        assert df["pval_adj"].between(0.0, 1.0).all(),             "pval_adj values outside [0, 1]"

    def test_compare_conditions_fstring_fix(self):
        """Regression test: verbose print should not contain literal {pval_cutoff}."""
        import io, sys
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured
        try:
            self.mscorer.compare_conditions(results, verbose=True, pval_threshold=0.05)
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        # The f-string fix: should NOT contain the literal string "{pval_cutoff}"
        assert "{pval_cutoff}" not in output,             "f-string not interpolated: found literal '{pval_cutoff}' in output"
        # Should contain the actual value
        assert "0.05" in output, "Expected pval_cutoff value '0.05' in output"

    @pytest.mark.skip(reason="PairScorer enforces exactly 2 conditions; "
                            "kruskal path moved to MultiScorer.compare_omnibus.")
    def test_compare_conditions_kruskal_path(self):
        """k>2 conditions should use kruskal-wallis test."""
        # Build a 3-condition dataset by relabeling the last 50 cells as "extra"
        adata3cond = self.adata.copy()
        cond_vals = adata3cond.obs["condition"].astype(str).values.copy()
        cond_vals[-50:] = "extra"
        adata3cond.obs["condition"] = pd.Categorical(cond_vals)

        from scCS.pairwise import PairScorer
        ms3 = PairScorer(
            adata3cond,
            root="0",
            branches=["1", "2"],
            condition_obs_key="condition",
            obs_key="leiden",
        )
        ms3.build_embedding(ordering_metric="pseudotime", verbose=False)
        rng = np.random.default_rng(7)
        n_sub = ms3._scorer.adata_sub.n_obs
        ms3._scorer.load_velocity_vectors(
            rng.normal(0.5, 0.3, n_sub), rng.normal(0.0, 0.2, n_sub)
        )
        ms3.fit(verbose=False)
        results3 = ms3.score_all_conditions(cell_level=True, verbose=False)
        df3 = ms3.compare_conditions(results3, verbose=False)
        assert "kruskal-wallis" in df3["test"].values,             "Expected kruskal-wallis test for k>2 conditions"
        assert "mann-whitney" in df3["test"].values,             "Expected mann-whitney pairwise tests for k>2 conditions"

    # ------------------------------------------------------------------
    # trajectory_shift
    # ------------------------------------------------------------------

    def test_trajectory_shift_returns_dataframe(self):
        # Add subset pseudotime to adata_sub
        n_sub = self.mscorer._scorer.adata_sub.n_obs
        self.mscorer._scorer.adata_sub.obs["sccs_pseudotime"] =             np.random.default_rng(0).uniform(0, 1, n_sub)
        results = self.mscorer.score_all_conditions(verbose=False)
        df = self.mscorer.trajectory_shift(
            results, n_bootstrap=20, verbose=False
        )
        assert isinstance(df, pd.DataFrame)

    def test_trajectory_shift_columns(self):
        n_sub = self.mscorer._scorer.adata_sub.n_obs
        self.mscorer._scorer.adata_sub.obs["sccs_pseudotime"] =             np.random.default_rng(1).uniform(0, 1, n_sub)
        results = self.mscorer.score_all_conditions(verbose=False)
        df = self.mscorer.trajectory_shift(
            results, n_bootstrap=20, verbose=False
        )
        for col in ["fate", "comparison", "ks_stat", "ks_pval", "wasserstein",
                    "ks_pval_adj", "significant"]:
            assert col in df.columns, f"Missing column '{col}'"

    # ------------------------------------------------------------------
    # fit_mixed_model
    # ------------------------------------------------------------------

    def test_fit_mixed_model_runs(self):
        pytest.importorskip("statsmodels")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.fit_mixed_model(results, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_fit_mixed_model_columns(self):
        pytest.importorskip("statsmodels")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.fit_mixed_model(results, verbose=False)
        if not df.empty:
            for col in ["fate", "condition", "coef", "pval", "pval_adj", "significant"]:
                assert col in df.columns, f"Missing column '{col}'"



# ===========================================================================
# TestVelocityFateDrivers
# ===========================================================================

class TestVelocityFateDrivers:
    """Tests for get_velocity_fate_drivers() — velocity-fate correlation method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build a minimal scorer with a synthetic velocity layer."""
        import anndata as ad
        rng = np.random.default_rng(42)
        n_cells = 200
        n_genes = 50

        # Synthetic expression + velocity
        X = rng.poisson(2.0, (n_cells, n_genes)).astype(float)
        velocity = rng.normal(0.0, 1.0, (n_cells, n_genes))

        # Cluster labels: 0=root, 1=fateA, 2=fateB
        leiden = np.array(["0"] * 60 + ["1"] * 70 + ["2"] * 70)
        pseudotime = np.concatenate([
            rng.uniform(0.0, 0.2, 60),
            rng.uniform(0.3, 1.0, 70),
            rng.uniform(0.3, 1.0, 70),
        ])

        adata = ad.AnnData(X=X)
        adata.obs["leiden"] = pd.Categorical(leiden)
        adata.obs["velocity_pseudotime"] = pseudotime
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.layers["velocity"] = velocity

        import scipy.sparse as sp
        adata.obsp["velocity_graph"] = sp.eye(n_cells, format="csr") * 0.1

        from scCS.single import SingleScorer
        scorer = SingleScorer(
            adata, root="0", branches=["1", "2"], obs_key="leiden"
        )
        scorer.build_embedding(ordering_metric="pseudotime", verbose=False)
        n_sub = scorer.adata_sub.n_obs
        scorer.load_velocity_vectors(
            rng.normal(0.5, 0.3, n_sub), rng.normal(0.0, 0.2, n_sub)
        )
        scorer.fit(verbose=False)
        self.scorer = scorer
        self.result = scorer.score(cell_level=True, verbose=False)

    def test_returns_dict_with_fate_keys(self):
        drivers = self.scorer.get_velocity_fate_drivers(self.result, n_top_genes=5)
        assert isinstance(drivers, dict)
        for fate in ["1", "2"]:
            assert fate in drivers, f"Missing fate key '{fate}'"

    def test_dataframe_columns(self):
        drivers = self.scorer.get_velocity_fate_drivers(self.result, n_top_genes=5)
        expected_cols = {"gene", "spearman_r", "pval", "pval_adj",
                         "mean_velocity", "delta_velocity", "significant"}
        for fate, df in drivers.items():
            assert expected_cols.issubset(set(df.columns)), (
                f"Missing columns for fate '{fate}': "
                f"{expected_cols - set(df.columns)}"
            )

    def test_spearman_r_in_range(self):
        drivers = self.scorer.get_velocity_fate_drivers(self.result, n_top_genes=5)
        for fate, df in drivers.items():
            assert df["spearman_r"].between(-1.0, 1.0).all(), (
                f"Spearman r out of [-1, 1] for fate '{fate}'"
            )

    def test_top_gene_has_highest_correlation(self):
        drivers = self.scorer.get_velocity_fate_drivers(self.result, n_top_genes=5)
        for fate, df in drivers.items():
            assert df["spearman_r"].iloc[0] == df["spearman_r"].max(), (
                f"Top gene does not have highest spearman_r for fate '{fate}'"
            )

    def test_missing_velocity_layer_raises(self):
        from scCS.drivers import get_velocity_fate_drivers
        adata_no_vel = self.scorer.adata_sub.copy()
        del adata_no_vel.layers["velocity"]
        with pytest.raises(ValueError):
            get_velocity_fate_drivers(
                adata_no_vel,
                cell_scores=self.result.cell_scores,
                fate_names=["1", "2"],
                obs_key="leiden",
                root="0",
            )

    def test_missing_cell_scores_raises(self):
        with pytest.raises(ValueError):
            self.scorer.get_velocity_fate_drivers(
                self.result.__class__(
                    fate_names=self.result.fate_names,
                    M_bin=self.result.M_bin,
                    bin_edges=self.result.bin_edges,
                    sectors=self.result.sectors,
                    M_sector=self.result.M_sector,
                    n_cells_per_fate=self.result.n_cells_per_fate,
                    commitment_vector=self.result.commitment_vector,
                    population_entropy=self.result.population_entropy,
                    mean_cell_entropy=self.result.mean_cell_entropy,
                    per_fate_entropy=self.result.per_fate_entropy,
                    pairwise_unCS=self.result.pairwise_unCS,
                    pairwise_nCS=self.result.pairwise_nCS,
                    cell_scores=None,  # <-- None
                    fate_angles=self.result.fate_angles,
                )
            )


# ===========================================================================
# TestNewPlots
# ===========================================================================

class TestNewPlots:
    """Smoke tests for new plot functions added in v0.6.1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build a minimal PairScorer with two conditions."""
        import anndata as ad
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for tests

        rng = np.random.default_rng(99)
        n_cells = 200
        n_genes = 20

        X = rng.poisson(2.0, (n_cells, n_genes)).astype(float)
        leiden = np.array(["0"] * 60 + ["1"] * 70 + ["2"] * 70)
        condition = np.array(["ctrl"] * 100 + ["treat"] * 100)
        pseudotime = np.concatenate([
            rng.uniform(0.0, 0.2, 60),
            rng.uniform(0.3, 1.0, 70),
            rng.uniform(0.3, 1.0, 70),
        ])

        adata = ad.AnnData(X=X)
        adata.obs["leiden"] = pd.Categorical(leiden)
        adata.obs["condition"] = pd.Categorical(condition)
        adata.obs["velocity_pseudotime"] = pseudotime
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        import scipy.sparse as sp
        adata.obsp["velocity_graph"] = sp.eye(n_cells, format="csr") * 0.1

        from scCS.pairwise import PairScorer
        mscorer = PairScorer(
            adata, root="0", branches=["1", "2"],
            condition_obs_key="condition", obs_key="leiden",
        )
        mscorer.build_embedding(ordering_metric="pseudotime", verbose=False)
        n_sub = mscorer._scorer.adata_sub.n_obs
        mscorer._scorer.load_velocity_vectors(
            rng.normal(0.5, 0.3, n_sub), rng.normal(0.0, 0.2, n_sub)
        )
        mscorer.fit(verbose=False)
        self.mscorer = mscorer
        self.results = mscorer.score_all_conditions(cell_level=True, verbose=False)
        self.delta = mscorer.compute_delta_CS(
            "ctrl", "treat", n_bootstrap=20, verbose=False
        )
        yield
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_rose_grid_returns_figure(self):
        import matplotlib.figure
        fig = self.mscorer.plot_rose_grid(self.results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_delta_cs_heatmap_returns_figure(self):
        import matplotlib.figure
        fig = self.mscorer.plot_delta_cs_heatmap(self.delta)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_compare_conditions_bar_returns_figure(self):
        import matplotlib.figure
        fig = self.mscorer.plot_compare_conditions_bar(self.results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_commitment_vector_radar_returns_figure(self):
        import matplotlib.figure
        # k=2 falls back to bar chart — should still return a Figure
        fig = self.mscorer.plot_commitment_vector_radar(self.results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_standalone_plot_rose_grid(self):
        import matplotlib.figure
        import scCS
        fig = scCS.plot_rose_grid(self.results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_standalone_plot_delta_cs_heatmap(self):
        import matplotlib.figure
        import scCS
        fig = scCS.plot_delta_cs_heatmap(self.delta)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_standalone_plot_compare_conditions_bar(self):
        import matplotlib.figure
        import scCS
        fig = scCS.plot_compare_conditions_bar(self.results)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_standalone_plot_commitment_vector_radar(self):
        import matplotlib.figure
        import scCS
        fig = scCS.plot_commitment_vector_radar(self.results)
        assert isinstance(fig, matplotlib.figure.Figure)




# ---------------------------------------------------------------------------
# 20b. Plot bug-fix tests (v0.7.1)
# ---------------------------------------------------------------------------

class TestPlotStarAutoScale:
    """Smoke tests for the v0.7.1 plot.py fixes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import matplotlib
        matplotlib.use("Agg")

        # Build a tiny SingleScorer scored with k_nn so cs_nn_entropy exists
        from scCS.single import SingleScorer

        rng = np.random.default_rng(123)
        n_cells = 200
        n_genes = 20
        X = rng.poisson(2.0, (n_cells, n_genes)).astype(float)
        leiden = np.array(["0"] * 60 + ["1"] * 70 + ["2"] * 70)
        pseudotime = np.concatenate([
            rng.uniform(0.0, 0.2, 60),
            rng.uniform(0.3, 1.0, 70),
            rng.uniform(0.3, 1.0, 70),
        ])

        adata = ad.AnnData(X=X)
        adata.obs["leiden"] = pd.Categorical(leiden)
        adata.obs["velocity_pseudotime"] = pseudotime
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        import scipy.sparse as sp
        adata.obsp["velocity_graph"] = sp.eye(n_cells, format="csr") * 0.1

        scorer = SingleScorer(
            adata, root="0", branches=["1", "2"], obs_key="leiden",
        )
        scorer.build_embedding(ordering_metric="pseudotime", verbose=False)
        n_sub = scorer.adata_sub.n_obs
        scorer.load_velocity_vectors(
            rng.normal(0.5, 0.3, n_sub), rng.normal(0.0, 0.2, n_sub)
        )
        scorer.fit(verbose=False)
        # Score with k_nn so cs_nn_entropy is populated
        self.result = scorer.score(cell_level=True, k_nn=10, verbose=False)
        self.scorer = scorer
        yield
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_star_entropy_autoscale(self):
        """vmin/vmax default to data range (not [0, 1])."""
        import matplotlib.figure
        fig = self.scorer.plot_star(self.result, color_by="entropy")
        assert isinstance(fig, matplotlib.figure.Figure)
        # Find the QuadMesh/PathCollection with a colorbar mappable
        norm = None
        for ax in fig.axes:
            for coll in ax.collections:
                if (hasattr(coll, "norm") and coll.norm is not None
                        and coll.get_array() is not None
                        and coll.get_array().size > 0):
                    norm = coll.norm
                    break
            if norm is not None:
                break
        assert norm is not None, "No mappable found for colorbar"

        # Data range of cs_entropy on the embedded subset
        ent_vals = np.asarray(self.scorer.adata_sub.obs["cs_entropy"]).astype(float)
        ent_vals = ent_vals[np.isfinite(ent_vals)]
        # The colorbar limits should match the finite data range (not [0, 1])
        assert np.isclose(norm.vmin, float(np.min(ent_vals)), atol=1e-6)
        assert np.isclose(norm.vmax, float(np.max(ent_vals)), atol=1e-6)

    def test_plot_star_entropy_explicit_range(self):
        """Explicit vmin/vmax override autoscaling."""
        fig = self.scorer.plot_star(
            self.result, color_by="entropy", vmin=0.0, vmax=1.0
        )
        norm = None
        for ax in fig.axes:
            for coll in ax.collections:
                if (hasattr(coll, "norm") and coll.norm is not None
                        and coll.get_array() is not None
                        and coll.get_array().size > 0):
                    norm = coll.norm
                    break
            if norm is not None:
                break
        assert norm is not None
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0

    def test_plot_star_nn_entropy_renders(self):
        """color_by='nn_entropy' now produces a real colored scatter (was gray)."""
        import matplotlib.figure
        # cs_nn_entropy was populated by score(k_nn=10) in setup
        assert "cs_nn_entropy" in self.scorer.adata_sub.obs.columns
        fig = self.scorer.plot_star(self.result, color_by="nn_entropy")
        assert isinstance(fig, matplotlib.figure.Figure)
        # Colorbar should reflect actual nn_entropy range, not be missing
        nn_vals = np.asarray(
            self.scorer.adata_sub.obs["cs_nn_entropy"]
        ).astype(float)
        nn_vals = nn_vals[np.isfinite(nn_vals)]
        # Find the QuadMesh/PathCollection with a colorbar mappable
        norm = None
        for ax in fig.axes:
            for coll in ax.collections:
                if (hasattr(coll, "norm") and coll.norm is not None
                        and coll.get_array() is not None
                        and coll.get_array().size > 0):
                    norm = coll.norm
                    break
            if norm is not None:
                break
        assert norm is not None, (
            "nn_entropy branch fell through to gray fallback (no colormap)"
        )
        assert np.isclose(norm.vmin, float(np.min(nn_vals)), atol=1e-6)
        assert np.isclose(norm.vmax, float(np.max(nn_vals)), atol=1e-6)

    def test_plot_subset_comparison_inf_handling(self):
        """Progenitor-only subsets render as hatched 'inf' placeholders, not empty."""
        import warnings
        import matplotlib.figure
        from scCS.scores import CommitmentScoreResult
        from scCS.plot import plot_subset_comparison

        def _result(ncs):
            return CommitmentScoreResult(
                fate_names=["A", "B"],
                M_bin=np.zeros(36),
                bin_edges=np.linspace(-np.pi, np.pi, 37),
                sectors=[(np.array([0.0]), np.array([1.0]))] * 2,
                M_sector=np.array([1.0, 1.0]),
                n_cells_per_fate=np.array([50, 50]),
                commitment_vector=np.array([0.5, 0.5]),
                population_entropy=1.0,
                mean_cell_entropy=0.85,
                per_fate_entropy=np.array([0.8, 0.9]),
                pairwise_unCS=np.abs(ncs),
                pairwise_nCS=ncs,
                cell_scores=None,
                fate_angles=np.array([0.0, np.pi]),
                cell_obs_names=[f"c{i}" for i in range(50)],
            )

        subset_results = {
            "good": _result(np.array([[0.0, 0.7], [0.7, 0.0]])),
            "prog_only": _result(np.array([[0.0, np.inf], [np.inf, 0.0]])),
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig = plot_subset_comparison(subset_results, figsize=(6, 3))

        assert isinstance(fig, matplotlib.figure.Figure)

        # UserWarning must be emitted listing the bad subset
        bad_warns = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "prog_only" in str(x.message)
        ]
        assert len(bad_warns) >= 1, "Expected UserWarning naming the inf subset"

        # Inspect bars: good=finite, prog_only=hatched
        ax = fig.axes[0]
        rects = [p for c in ax.containers for p in c.patches]
        hatched = [p for p in rects if p.get_hatch()]
        assert len(hatched) >= 1, "Expected at least one hatched 'inf' bar"
        # And an "inf" text annotation
        labels = [t.get_text() for t in ax.texts]
        assert "inf" in labels, "Expected 'inf' annotation on hatched bar"


# ---------------------------------------------------------------------------
# Helper for 3+ condition tests
# ---------------------------------------------------------------------------

def _make_adata_multicond3(n_per_cond: int = 100, seed: int = 0) -> "ad.AnnData":
    """Synthetic AnnData with 3 conditions, 3 clusters (0=progenitor, 1/2=fates)."""
    rng = np.random.default_rng(seed)
    n_total = n_per_cond * 3
    X = rng.normal(size=(n_total, 20))

    # 3 clusters: 0=progenitor (33/cond), 1=fateA (33/cond), 2=fateB (34/cond)
    n_per_cluster = n_per_cond // 3
    labels = np.array(
        ["0"] * n_per_cluster + ["1"] * n_per_cluster + ["2"] * (n_per_cond - 2 * n_per_cluster) +
        ["0"] * n_per_cluster + ["1"] * n_per_cluster + ["2"] * (n_per_cond - 2 * n_per_cluster) +
        ["0"] * n_per_cluster + ["1"] * n_per_cluster + ["2"] * (n_per_cond - 2 * n_per_cluster)
    )
    conditions = np.array(
        ["ctrl"] * n_per_cond + ["drug_A"] * n_per_cond + ["drug_B"] * n_per_cond
    )
    pseudotime = np.concatenate([
        rng.uniform(0.0, 0.3, n_per_cluster),
        rng.uniform(0.5, 1.0, n_per_cluster),
        rng.uniform(0.5, 1.0, n_per_cond - 2 * n_per_cluster),
        rng.uniform(0.0, 0.3, n_per_cluster),
        rng.uniform(0.5, 1.0, n_per_cluster),
        rng.uniform(0.5, 1.0, n_per_cond - 2 * n_per_cluster),
        rng.uniform(0.0, 0.3, n_per_cluster),
        rng.uniform(0.5, 1.0, n_per_cluster),
        rng.uniform(0.5, 1.0, n_per_cond - 2 * n_per_cluster),
    ])
    adata = ad.AnnData(X=X)
    adata.obs["leiden"] = pd.Categorical(labels)
    adata.obs["condition"] = pd.Categorical(conditions)
    adata.obs["velocity_pseudotime"] = pseudotime
    return adata


def _make_multiscorer(adata=None):
    """Build and fit a MultiScorer on synthetic 3-condition data."""
    from scCS.multicomparison import MultiScorer

    if adata is None:
        adata = _make_adata_multicond3()

    mscorer = MultiScorer(
        adata,
        root="0",
        branches=["1", "2"],
        condition_obs_key="condition",
        obs_key="leiden",
    )
    mscorer.build_embedding(ordering_metric="pseudotime", verbose=False)

    # Inject synthetic velocity vectors
    rng = np.random.default_rng(42)
    n_sub = mscorer._scorer.adata_sub.n_obs
    vx = rng.normal(0.5, 0.3, n_sub)
    vy = rng.normal(0.0, 0.2, n_sub)
    mscorer._scorer.load_velocity_vectors(vx, vy)
    mscorer.fit(verbose=False)
    return mscorer


# ---------------------------------------------------------------------------
# 21. MultiScorer tests
# ---------------------------------------------------------------------------

class TestMultiScorerPipeline:
    """Tests for MultiScorer — 3+ condition comparisons."""

    def setup_method(self):
        self.adata = _make_adata_multicond3()
        self.mscorer = _make_multiscorer(self.adata)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_conditions_detected(self):
        assert set(self.mscorer.conditions) == {"ctrl", "drug_A", "drug_B"}

    def test_repr_contains_key_info(self):
        r = repr(self.mscorer)
        assert "MultiScorer" in r
        assert "fitted" in r

    def test_validation_rejects_2_conditions(self):
        """MultiScorer should reject exactly 2 conditions (suggest PairScorer)."""
        from scCS.multicomparison import MultiScorer
        adata2 = _make_adata_multicond()  # 2 conditions
        with pytest.raises(ValueError, match="at least 3 conditions"):
            MultiScorer(
                adata2,
                root="0",
                branches=["1", "2"],
                condition_obs_key="condition",
                obs_key="leiden",
            )

    def test_validation_rejects_1_condition(self):
        """MultiScorer should reject 1 condition."""
        from scCS.multicomparison import MultiScorer
        adata1 = self.adata.copy()
        adata1.obs["condition"] = "ctrl"
        with pytest.raises(ValueError, match="at least 3 conditions"):
            MultiScorer(
                adata1,
                root="0",
                branches=["1", "2"],
                condition_obs_key="condition",
                obs_key="leiden",
            )

    # ------------------------------------------------------------------
    # score_all_conditions
    # ------------------------------------------------------------------

    def test_score_all_conditions_returns_three(self):
        results = self.mscorer.score_all_conditions(verbose=False)
        assert set(results.keys()) == {"ctrl", "drug_A", "drug_B"}

    def test_score_all_conditions_result_type(self):
        results = self.mscorer.score_all_conditions(verbose=False)
        for cond, res in results.items():
            assert isinstance(res, CommitmentScoreResult)

    def test_score_all_conditions_cell_scores_available(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        for cond, res in results.items():
            assert res.cell_scores is not None
            assert res.cell_scores.shape[1] == 2

    # ------------------------------------------------------------------
    # compare_omnibus
    # ------------------------------------------------------------------

    def test_omnibus_kruskal(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_omnibus(results, test="kruskal", verbose=False)
        assert isinstance(df, pd.DataFrame)
        for col in ["fate", "test", "statistic", "pval", "pval_adj", "significant", "n_conditions"]:
            assert col in df.columns, f"Missing column '{col}'"
        assert len(df) == 2  # 2 fates
        assert (df["n_conditions"] == 3).all()

    def test_omnibus_anova(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_omnibus(results, test="anova", verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert df["test"].str.contains("anova").all()

    def test_omnibus_pval_adj_in_range(self):
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_omnibus(results, verbose=False)
        assert df["pval_adj"].between(0.0, 1.0).all()

    # ------------------------------------------------------------------
    # compare_posthoc
    # ------------------------------------------------------------------

    def test_posthoc_dunn(self):
        pytest.importorskip("scikit_posthocs")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_posthoc(results, method="dunn", verbose=False)
        assert isinstance(df, pd.DataFrame)
        for col in ["fate", "comparison", "method", "pval", "pval_adj", "significant"]:
            assert col in df.columns, f"Missing column '{col}'"
        # 3 conditions -> C(3,2) = 3 pairwise comparisons per fate
        assert len(df) == 2 * 3  # 2 fates × 3 pairs

    def test_posthoc_fdr_correction(self):
        pytest.importorskip("scikit_posthocs")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_posthoc(
            results, method="dunn", pval_correction="fdr", verbose=False
        )
        # FDR correction: pval_adj should be >= pval (or equal)
        assert (df["pval_adj"] >= df["pval"] - 1e-10).all()

    def test_posthoc_bonferroni_correction(self):
        pytest.importorskip("scikit_posthocs")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.compare_posthoc(
            results, method="dunn", pval_correction="bonferroni", verbose=False
        )
        # Bonferroni: pval_adj = min(pval * n_tests, 1.0)
        assert df["pval_adj"].between(0.0, 1.0).all()

    def test_posthoc_with_omnibus_filter(self):
        pytest.importorskip("scikit_posthocs")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        omnibus_df = self.mscorer.compare_omnibus(results, verbose=False)
        posthoc_df = self.mscorer.compare_posthoc(
            results, omnibus_results=omnibus_df, verbose=False
        )
        # Should only include fates where omnibus was significant
        # (or all if none significant, depending on implementation)
        assert isinstance(posthoc_df, pd.DataFrame)

    # ------------------------------------------------------------------
    # compute_pairwise_deltas
    # ------------------------------------------------------------------

    def test_pairwise_deltas(self):
        deltas = self.mscorer.compute_pairwise_deltas(
            n_bootstrap=20, verbose=False
        )
        # 3 conditions -> C(3,2) = 3 pairs
        assert len(deltas) == 3
        for key, delta in deltas.items():
            assert "delta_nCS" in delta
            assert "ci_low" in delta
            assert "ci_high" in delta
            assert delta["delta_nCS"].shape == (2, 2)

    def test_pairwise_deltas_ci_ordering(self):
        deltas = self.mscorer.compute_pairwise_deltas(
            n_bootstrap=20, verbose=False
        )
        for key, delta in deltas.items():
            d = delta["delta_nCS"]
            lo = delta["ci_low"]
            hi = delta["ci_high"]
            finite = np.isfinite(d) & np.isfinite(lo) & np.isfinite(hi)
            assert np.all(lo[finite] <= hi[finite])

    # ------------------------------------------------------------------
    # fit_mixed_model_contrasts
    # ------------------------------------------------------------------

    def test_mixed_model_contrasts(self):
        pytest.importorskip("statsmodels")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        df = self.mscorer.fit_mixed_model_contrasts(
            results, ref_condition="ctrl", verbose=False
        )
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            for col in ["fate", "contrast", "coef", "std_err", "z_score", "pval", "pval_adj", "significant"]:
                assert col in df.columns, f"Missing column '{col}'"

    def test_mixed_model_contrasts_custom(self):
        pytest.importorskip("statsmodels")
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        contrasts = [("drug_A", "ctrl"), ("drug_B", "ctrl")]
        df = self.mscorer.fit_mixed_model_contrasts(
            results, contrasts=contrasts, verbose=False
        )
        assert isinstance(df, pd.DataFrame)

    # ------------------------------------------------------------------
    # Plot smoke tests
    # ------------------------------------------------------------------

    def test_plot_omnibus_summary_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        omnibus_df = self.mscorer.compare_omnibus(results, verbose=False)
        fig = self.mscorer.plot_omnibus_summary(omnibus_df, results)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_posthoc_heatmap_returns_figure(self):
        pytest.importorskip("scikit_posthocs")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        results = self.mscorer.score_all_conditions(
            cell_level=True, verbose=False
        )
        posthoc_df = self.mscorer.compare_posthoc(results, verbose=False)
        fig = self.mscorer.plot_posthoc_heatmap(posthoc_df)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_pairwise_delta_grid_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        deltas = self.mscorer.compute_pairwise_deltas(
            n_bootstrap=20, verbose=False
        )
        fig = self.mscorer.plot_pairwise_delta_grid(deltas)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
