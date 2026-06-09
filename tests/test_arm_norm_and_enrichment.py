"""Tests for v0.7.3 features.

Covers:
- ``build_star_embedding`` ``arm_norm`` parameter (global / per_arm / invalid).
- ``run_enrichment_per_fate`` ``fate_names`` inference + mismatch warning.
- ``plot_omnibus_summary`` auto-scaled vmin/vmax (and explicit override).
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import scipy.sparse as sp

from scCS.embedding import build_star_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_star_adata(seed: int = 0) -> ad.AnnData:
    """Toy AnnData with 4 fates of varying pseudotime ranges.

    Layout:
      - root cluster 'Root': 30 cells, pseudotime in [0.00, 0.05]
      - fate 'A': 40 cells, pseudotime in [0.05, 1.00]  (long arm)
      - fate 'B': 30 cells, pseudotime in [0.05, 0.50]  (medium arm)
      - fate 'C': 20 cells, pseudotime in [0.05, 0.20]  (short arm)
      - fate 'D': 30 cells, pseudotime in [0.05, 0.80]  (long-ish arm)
      - 20 'other' cells (excluded from subset)
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
        rng.uniform(0.05, 1.00, n_a),
        rng.uniform(0.05, 0.50, n_b),
        rng.uniform(0.05, 0.20, n_c),
        rng.uniform(0.05, 0.80, n_d),
        rng.uniform(0.00, 1.00, n_other),
    ])
    X = sp.csr_matrix(rng.random((n, 10)))
    obs = pd.DataFrame({
        "clusters": pd.Categorical(labels),
        "velocity_pseudotime": pt,
    })
    obs.index = [f"cell_{i}" for i in range(n)]
    adata = ad.AnnData(X=X, obs=obs)
    return adata


# ---------------------------------------------------------------------------
# build_star_embedding arm_norm
# ---------------------------------------------------------------------------

class TestArmNorm:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.adata = _make_star_adata(seed=0)
        self.root = "Root"
        self.branches = ["A", "B", "C", "D"]
        self.obs_key = "clusters"

    def test_arm_norm_global_produces_variable_arm_lengths(self):
        """Global mode: arms whose cells span shorter pseudotime ranges
        should reach shorter max radii than the longest arm."""
        adata_sub = build_star_embedding(
            self.adata,
            root=self.root,
            branches=self.branches,
            obs_key=self.obs_key,
            ordering_metric="velocity_pseudotime",
            arm_norm="global",
            arm_scale=10.0,
            jitter=0.0,  # disable noise so radii are exact
            seed=42,
        )
        assert adata_sub.uns["sccs"]["arm_norm"] == "global"

        coords = adata_sub.obsm["X_sccs"]
        labels = adata_sub.obs["clusters"].astype(str).values
        max_r = {}
        for fate in self.branches:
            m = labels == fate
            max_r[fate] = float(np.linalg.norm(coords[m], axis=1).max())

        # A has the largest pseudotime range -> reaches close to arm_scale.
        assert max_r["A"] == pytest.approx(10.0, rel=0.01)
        # C has the shortest pseudotime range -> radius much smaller.
        assert max_r["C"] < max_r["A"]
        assert max_r["B"] < max_r["A"]
        assert max_r["D"] < max_r["A"]
        # Order roughly tracks the pseudotime range upper bounds.
        assert max_r["C"] < max_r["B"] < max_r["D"] < max_r["A"]

    def test_arm_norm_per_arm_clamps_all_to_arm_scale(self):
        """Per-arm mode: every arm's max radius should be ~arm_scale."""
        adata_sub = build_star_embedding(
            self.adata,
            root=self.root,
            branches=self.branches,
            obs_key=self.obs_key,
            ordering_metric="velocity_pseudotime",
            arm_norm="per_arm",
            arm_scale=10.0,
            jitter=0.0,
            seed=42,
        )
        assert adata_sub.uns["sccs"]["arm_norm"] == "per_arm"

        coords = adata_sub.obsm["X_sccs"]
        labels = adata_sub.obs["clusters"].astype(str).values
        for fate in self.branches:
            m = labels == fate
            max_r = float(np.linalg.norm(coords[m], axis=1).max())
            assert max_r == pytest.approx(10.0, rel=0.05), (
                f"per_arm mode: fate {fate} max radius {max_r:.3f} "
                f"should be near arm_scale=10.0"
            )

    def test_arm_norm_invalid_raises(self):
        with pytest.raises(ValueError, match="arm_norm must be"):
            build_star_embedding(
                self.adata,
                root=self.root,
                branches=self.branches,
                obs_key=self.obs_key,
                ordering_metric="velocity_pseudotime",
                arm_norm="bogus",
            )

    def test_arm_norm_default_is_global(self):
        """v0.7.3 default — global mode without explicit kwarg."""
        adata_sub = build_star_embedding(
            self.adata,
            root=self.root,
            branches=self.branches,
            obs_key=self.obs_key,
            ordering_metric="velocity_pseudotime",
            arm_scale=10.0,
            jitter=0.0,
            seed=42,
        )
        assert adata_sub.uns["sccs"]["arm_norm"] == "global"


# ---------------------------------------------------------------------------
# run_enrichment_per_fate fate_names inference
# ---------------------------------------------------------------------------

class TestEnrichmentFateNamesInference:
    def _fake_deg(self):
        """Tiny fake DEG dict — too few genes to actually run gseapy,
        but enough to test the signature / inference logic."""
        return {
            "Alpha": pd.DataFrame({
                "gene": ["Gcg"],
                "logfoldchange": [2.0],
                "pval": [0.01],
                "pval_adj": [0.05],
                "significant": [True],
            }),
            "Beta": pd.DataFrame({
                "gene": ["Ins1"],
                "logfoldchange": [2.0],
                "pval": [0.001],
                "pval_adj": [0.01],
                "significant": [True],
            }),
        }

    def test_fate_names_inferred_from_deg_keys(self):
        """No fate_names passed — should iterate deg_drivers.keys()."""
        gseapy = pytest.importorskip("gseapy")  # noqa: F841
        from scCS.enrichment import run_enrichment_per_fate

        result = run_enrichment_per_fate(
            self._fake_deg(),
            organism="mouse",
            plot=False,
            pval_threshold=0.05,
            logfc_threshold=0.25,
        )
        # Both fates should be processed; values may be empty dicts because
        # of the too-few-genes guard, but the keys must be present.
        assert set(result.keys()) == {"Alpha", "Beta"}

    def test_fate_names_mismatch_emits_warning_and_intersects(self):
        """Mismatched fate_names: warn + use only intersection."""
        gseapy = pytest.importorskip("gseapy")  # noqa: F841
        from scCS.enrichment import run_enrichment_per_fate

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_enrichment_per_fate(
                self._fake_deg(),
                fate_names=["Alpha", "Gamma"],  # Gamma not in deg_drivers
                organism="mouse",
                plot=False,
            )
            mismatch = [str(wi.message) for wi in w
                        if issubclass(wi.category, UserWarning)
                        and "do not match" in str(wi.message).lower()]
        assert len(mismatch) == 1
        # Only Alpha (intersection) should appear in result keys.
        assert "Alpha" in result
        assert "Gamma" not in result
        assert "Beta" not in result  # excluded by user-provided fate_names

    def test_fate_names_explicit_match_no_warning(self):
        """fate_names exactly matches deg_drivers.keys() — silent."""
        gseapy = pytest.importorskip("gseapy")  # noqa: F841
        from scCS.enrichment import run_enrichment_per_fate

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_enrichment_per_fate(
                self._fake_deg(),
                fate_names=["Alpha", "Beta"],
                organism="mouse",
                plot=False,
            )
            mismatch = [wi for wi in w
                        if issubclass(wi.category, UserWarning)
                        and "do not match" in str(wi.message).lower()]
        assert mismatch == []
        assert set(result.keys()) == {"Alpha", "Beta"}


# ---------------------------------------------------------------------------
# plot_omnibus_summary auto-scaling
# ---------------------------------------------------------------------------

class TestPlotOmnibusSummaryAutoscale:
    """End-to-end test: build a tiny MultiScorer with the helper adata,
    score it, run a fake omnibus_df, and call plot_omnibus_summary to
    check the heatmap colorbar limits are auto-derived from the data
    range (not hard-pinned to [0, 1]) when vmin/vmax are not given."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        # Build a slightly bigger adata so MultiScorer has data to work with.
        from scCS.multicomparison import MultiScorer

        adata = _make_star_adata(seed=1)
        # Split adata into 3 conditions (MultiScorer requires >= 3).
        n = adata.n_obs
        third = n // 3
        cond = np.array(["cond1"] * third + ["cond2"] * third + ["cond3"] * (n - 2*third))
        adata.obs["condition"] = pd.Categorical(cond)

        self.mscorer = MultiScorer(
            adata,
            root="Root",
            branches=["A", "B", "C", "D"],
            obs_key="clusters",
            condition_obs_key="condition",
        )
        self.mscorer.build_embedding(
            ordering_metric="velocity_pseudotime",
            arm_scale=10.0,
            jitter=0.0,
            seed=42,
            verbose=False,
        )
        self.mscorer.fit(verbose=False)
        # Score per-condition with cell_level=True so cell_scores is populated.
        self.results = self.mscorer.score_all_conditions(
            cell_level=True, n_bootstrap=0, verbose=False,
        )

        # Build a minimal omnibus_df.
        self.omnibus_df = pd.DataFrame({
            "fate": list(self.mscorer._scorer._fate_map.fate_names),
            "pval_adj": [0.001, 0.05, 0.5, 0.2],
        })

    def test_auto_vmin_vmax_within_data_range(self):
        """Default (no vmin/vmax) — colormap clim should sit inside
        the actual mean-affinity range, not at [0, 1]."""
        fig = self.mscorer.plot_omnibus_summary(
            omnibus_df=self.omnibus_df,
            results=self.results,
        )
        ax = fig.axes[0]
        qm = ax.collections[0]
        clim = qm.get_clim()
        # Mean affinities in real data are typically in [0.05, 0.6].
        # The defaults from before v0.7.3 would give clim=(0.0, 1.0).
        assert clim[1] < 1.0, (
            f"vmax should be auto-derived from data, got {clim[1]}"
        )
        assert clim[0] >= 0.0
        assert clim[1] > clim[0]
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_explicit_vmin_vmax_honored(self):
        fig = self.mscorer.plot_omnibus_summary(
            omnibus_df=self.omnibus_df,
            results=self.results,
            vmin=0.0, vmax=1.0,
        )
        ax = fig.axes[0]
        qm = ax.collections[0]
        clim = qm.get_clim()
        assert clim[0] == pytest.approx(0.0)
        assert clim[1] == pytest.approx(1.0)
        import matplotlib.pyplot as plt
        plt.close(fig)
