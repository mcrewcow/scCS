"""Tests for v0.7.4 fixes.

Covers:

- ``build_star_embedding`` arm rescale uses **fate cells only** for
  s_min/s_max (was using fate+bif in v0.7.3). Bifurcation cells no
  longer drag the lower edge of the pseudotime scale upward and push
  the closest fate cell off the origin.
- In ``arm_norm="per_arm"`` mode, every arm's closest fate cell
  touches the origin (since each arm has its own (s_min, s_max) from
  its own fate cells).
- In ``arm_norm="global"`` mode, **at least one** arm's closest fate
  cell touches the origin (the arm containing the fate cell with the
  global-minimum ordering metric), and the longest-range arm touches
  ``arm_scale``.
- Bifurcation cells are still clamped to a tight inner cluster
  (controlled by the v0.7.0 origin-jitter logic, unrelated to the rescale fix).
- The scvelo HVG fallback uses ``flavor="cell_ranger"`` on tutorials —
  smoke-tested against synthetic data that contains zero-mean genes
  (which produce ``-inf`` log-dispersions on the seurat flavor and
  trigger pandas 2.2+ ``pd.cut`` ValueError).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

from scCS.embedding import build_star_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_offset_adata(seed: int = 0) -> ad.AnnData:
    """Toy AnnData where the bifurcation occupies a narrow pseudotime band
    that is offset from 0, so that fate cells have ordering metric values
    strictly greater than bif cells. This is the v0.7.3 regression case:
    if rescale uses bif+fate, the smallest fate cell is pushed off origin.

    Pseudotime layout:
      - Root (bifurcation): 30 cells in [0.00, 0.05]   <-- starts at 0
      - Fate A: 40 cells in [0.20, 1.00]               <-- starts at 0.20
      - Fate B: 30 cells in [0.30, 0.70]               <-- starts at 0.30
      - Fate C: 20 cells in [0.40, 0.55]               <-- starts at 0.40
      - Fate D: 30 cells in [0.25, 0.80]               <-- starts at 0.25
      - Other:  20 cells in [0.00, 1.00]               <-- excluded
    """
    rng = np.random.default_rng(seed)
    n_root, n_a, n_b, n_c, n_d, n_other = 30, 40, 30, 20, 30, 20
    n = n_root + n_a + n_b + n_c + n_d + n_other

    labels = (
        ["Root"] * n_root
        + ["A"] * n_a
        + ["B"] * n_b
        + ["C"] * n_c
        + ["D"] * n_d
        + ["Other"] * n_other
    )
    pt = np.concatenate([
        rng.uniform(0.00, 0.05, n_root),
        rng.uniform(0.20, 1.00, n_a),
        rng.uniform(0.30, 0.70, n_b),
        rng.uniform(0.40, 0.55, n_c),
        rng.uniform(0.25, 0.80, n_d),
        rng.uniform(0.00, 1.00, n_other),
    ])
    X = sp.csr_matrix(rng.random((n, 10)))
    obs = pd.DataFrame({
        "clusters": pd.Categorical(labels),
        "velocity_pseudotime": pt,
    })
    obs.index = [f"cell_{i}" for i in range(n)]
    return ad.AnnData(X=X, obs=obs)


# ---------------------------------------------------------------------------
# Fate-only rescale: per_arm mode
# ---------------------------------------------------------------------------

class TestPerArmFateOnly:
    """In per_arm mode each arm has its own (s_min, s_max) computed from
    its own fate cells only. The lowest-pseudotime fate cell on each arm
    should land at radius ~= 0 after rescale."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.adata = _make_offset_adata(seed=0)

    def test_per_arm_every_fate_touches_origin(self):
        adata_sub = build_star_embedding(
            self.adata,
            root="Root",
            branches=["A", "B", "C", "D"],
            obs_key="clusters",
            ordering_metric="velocity_pseudotime",
            arm_norm="per_arm",
            arm_scale=10.0,
            jitter=0.0,  # no noise so the min cell sits exactly at 0
            seed=42,
        )
        coords = adata_sub.obsm["X_sccs"]
        labels = adata_sub.obs["clusters"].astype(str).values
        for fate in ["A", "B", "C", "D"]:
            m = labels == fate
            r = np.linalg.norm(coords[m], axis=1)
            assert r.min() < 1e-6, (
                f"per_arm mode: fate {fate} closest cell radius "
                f"{r.min():.6f} should be ~0 after fate-only rescale "
                f"(v0.7.4 bug fix)"
            )
            assert r.max() == pytest.approx(10.0, rel=0.05), (
                f"per_arm mode: fate {fate} farthest cell should reach "
                f"~arm_scale=10.0, got {r.max():.3f}"
            )

    def test_per_arm_bif_cells_stay_near_origin(self):
        """Bifurcation cells (Root) should remain in the inner cluster
        with radius < ~1 (controlled by the separate origin-jitter logic,
        not by the rescale fix)."""
        adata_sub = build_star_embedding(
            self.adata,
            root="Root",
            branches=["A", "B", "C", "D"],
            obs_key="clusters",
            ordering_metric="velocity_pseudotime",
            arm_norm="per_arm",
            arm_scale=10.0,
            jitter=0.0,
            seed=42,
        )
        coords = adata_sub.obsm["X_sccs"]
        labels = adata_sub.obs["clusters"].astype(str).values
        root_r = np.linalg.norm(coords[labels == "Root"], axis=1)
        assert root_r.max() < 2.0, (
            f"Bifurcation cells should stay near origin; got max radius "
            f"{root_r.max():.3f}"
        )


# ---------------------------------------------------------------------------
# Fate-only rescale: global mode
# ---------------------------------------------------------------------------

class TestGlobalFateOnly:
    """In global mode there is one shared (s_min, s_max) across all arms,
    computed from **fate cells only** (excluding bif cells). The arm
    containing the global-min fate cell should reach radius ~0; the arm
    containing the global-max fate cell should reach ~arm_scale; other
    arms span proportionally."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.adata = _make_offset_adata(seed=0)

    def test_global_min_fate_cell_touches_origin(self):
        """Fate cell with the lowest pseudotime should sit at radius ~0,
        because the rescale (s - s_min)/(s_max - s_min) is computed
        across fate cells only (v0.7.4 fix). In the offset fixture, the
        min fate-cell pseudotime is ~0.20 (in fate A)."""
        adata_sub = build_star_embedding(
            self.adata,
            root="Root",
            branches=["A", "B", "C", "D"],
            obs_key="clusters",
            ordering_metric="velocity_pseudotime",
            arm_norm="global",
            arm_scale=10.0,
            jitter=0.0,
            seed=42,
        )
        coords = adata_sub.obsm["X_sccs"]
        labels = adata_sub.obs["clusters"].astype(str).values
        # The arm with the global-min fate cell must touch origin.
        min_radii = {}
        for fate in ["A", "B", "C", "D"]:
            m = labels == fate
            min_radii[fate] = float(np.linalg.norm(coords[m], axis=1).min())
        # At least ONE arm should touch origin in global mode.
        assert any(r < 1e-6 for r in min_radii.values()), (
            f"global mode: no arm reached origin. Min radii by fate: "
            f"{min_radii}. Expected the arm with the global-min fate "
            f"cell (likely A starting at pt=0.20) to sit at radius=0."
        )

    def test_global_max_fate_reaches_arm_scale(self):
        """Longest-range arm should reach radius ~= arm_scale."""
        adata_sub = build_star_embedding(
            self.adata,
            root="Root",
            branches=["A", "B", "C", "D"],
            obs_key="clusters",
            ordering_metric="velocity_pseudotime",
            arm_norm="global",
            arm_scale=10.0,
            jitter=0.0,
            seed=42,
        )
        coords = adata_sub.obsm["X_sccs"]
        labels = adata_sub.obs["clusters"].astype(str).values
        max_radii = {
            fate: float(np.linalg.norm(coords[labels == fate], axis=1).max())
            for fate in ["A", "B", "C", "D"]
        }
        # Arm A (max pt ~1.00) is the longest-range fate.
        assert max_radii["A"] == pytest.approx(10.0, rel=0.01), (
            f"global mode: longest arm A should reach arm_scale=10.0, "
            f"got {max_radii['A']:.3f}. All max radii: {max_radii}"
        )

    def test_global_bif_cells_excluded_from_rescale(self):
        """Regression test: the v0.7.3 bug used fate+bif for s_min,
        which moves s_min to bif's lower bound (e.g. 0.00) and pushes
        every fate cell off the origin. After the v0.7.4 fix, s_min is
        the fate-cells-only minimum (~0.20 in this fixture), so fate
        A's smallest cell sits at origin and fate B's smallest cell is
        at radius (0.30 - 0.20) / (1.00 - 0.20) * 10 ≈ 1.25 — NOT at
        the old (0.30 - 0.00) / (1.00 - 0.00) * 10 = 3.00."""
        adata_sub = build_star_embedding(
            self.adata,
            root="Root",
            branches=["A", "B", "C", "D"],
            obs_key="clusters",
            ordering_metric="velocity_pseudotime",
            arm_norm="global",
            arm_scale=10.0,
            jitter=0.0,
            seed=42,
        )
        coords = adata_sub.obsm["X_sccs"]
        labels = adata_sub.obs["clusters"].astype(str).values
        # Fate B's min pseudotime is ~0.30, fate A's min ~0.20.
        # With fate-only rescale: B_min radius ≈ (0.30 - 0.20)/(1.0 - 0.20) * 10 = 1.25
        # With buggy fate+bif rescale: B_min radius ≈ (0.30 - 0.0)/(1.0 - 0.0) * 10 = 3.0
        # Allow ±0.5 tolerance for the random seed.
        b_min = float(np.linalg.norm(coords[labels == "B"], axis=1).min())
        assert b_min < 2.0, (
            f"global mode: fate B min radius {b_min:.3f} too large; "
            f"v0.7.3 bug (using fate+bif for rescale) would give ~3.0, "
            f"v0.7.4 fix gives ~1.25."
        )


# ---------------------------------------------------------------------------
# scvelo HVG fallback robustness
# ---------------------------------------------------------------------------

class TestScveloHvgFallbackCellRanger:
    """The tutorials' HVG fallback uses ``flavor="cell_ranger"``. This
    test ensures the cell_ranger flavor does not crash on synthetic data
    that triggers the seurat-flavor pandas 2.2+ ``pd.cut`` ValueError
    (zero-mean genes producing -inf log-dispersions)."""

    def test_cell_ranger_handles_zero_mean_genes(self):
        """Synthetic adata with a few all-zero genes (mean=0 → log-dispersion=-inf)
        is the failure mode for flavor="seurat" on pandas>=2.2. cell_ranger
        must succeed."""
        scanpy = pytest.importorskip("scanpy")
        import scanpy as sc
        rng = np.random.default_rng(0)
        n_cells = 200
        n_genes = 300
        # 90% normal-expression genes + 10% all-zero (triggers -inf)
        n_normal = int(n_genes * 0.9)
        n_zero = n_genes - n_normal
        X = np.concatenate(
            [
                rng.lognormal(0.5, 1.0, size=(n_cells, n_normal)),
                np.zeros((n_cells, n_zero)),
            ],
            axis=1,
        ).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(n_genes)]
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # This should NOT raise — cell_ranger uses explicit -inf/+inf bin edges.
        sc.pp.highly_variable_genes(
            adata, n_top_genes=50, subset=True, flavor="cell_ranger"
        )
        assert adata.n_vars == 50, (
            f"cell_ranger HVG fallback should select n_top_genes=50, "
            f"got {adata.n_vars}"
        )
