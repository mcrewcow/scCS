"""
_base.py — Shared base class for all scCS scorers.

Extracts common initialization, embedding, and validation logic
from SingleScorer, PairScorer, and MultiScorer into a single
abstract base class to avoid code duplication.
"""

from __future__ import annotations

from abc import ABC
from typing import List, Literal, Optional, Union

import numpy as np

from .bifurcation import FateMap, build_fate_map
from .embedding import (
    build_star_embedding,
    project_velocity_star,
    compute_local_pseudotime,
    scale_metric_01,
)

SectorMode = Literal["centroid", "equal"]


class _BaseScorer(ABC):
    """Abstract base class for scCS commitment scorers.

    Encapsulates shared initialization parameters, embedding construction,
    pseudotime recomputation, and common validation logic used by
    SingleScorer, PairScorer, and MultiScorer.

    Parameters
    ----------
    adata : AnnData
        Single-cell dataset.
    root : str
        Label of the progenitor/root cluster in adata.obs[obs_key].
    branches : list of str
        Labels of the k terminal fate clusters.
    obs_key : str
        Column in adata.obs with cluster labels.  Default: 'leiden'.
    n_angle_bins : int
        Number of angular bins for commitment scoring.  Default: 36.
    sector_method : {\'centroid\', \'equal\'}
        How to define angular sectors.
    copy : bool
        Work on a copy of adata.
    """

    def __init__(
        self,
        adata,
        root: str,
        branches: List[str],
        obs_key: str = "leiden",
        n_angle_bins: int = 36,
        sector_method: SectorMode = "centroid",
        copy: bool = False,
    ):
        self.adata = adata.copy() if copy else adata
        self.root = str(root)
        self.branches = list(branches)
        self.obs_key = obs_key
        self.n_angle_bins = n_angle_bins
        self.sector_method = sector_method

        self._fate_map: Optional[FateMap] = None
        self._vx: Optional[np.ndarray] = None
        self._vy: Optional[np.ndarray] = None
        self._embedding_built = False
        self._fitted = False
        self._needs_refit = False
        self.adata_sub = None

    # ------------------------------------------------------------------
    # Embedding construction
    # ------------------------------------------------------------------

    def build_embedding(
        self,
        ordering_metric: Union[str, np.ndarray] = "pseudotime",
        invert_ordering: bool = False,
        scale_ordering: bool = False,
        arm_scale: float = 10.0,
        jitter: float = 0.3,
        seed: int = 42,
        arm_norm: str = "global",
        verbose: bool = True,
    ):
        """Construct the radial star embedding (X_sccs).

        Places the bifurcation cluster at the origin and arranges each
        terminal fate on its own radial arm.  Cells are ordered along
        each arm by the differentiation metric.

        Parameters
        ----------
        ordering_metric : str or np.ndarray
            Metric used to order cells along each arm.
        invert_ordering : bool
            Set True if your metric is inverted (high = less differentiated).
        scale_ordering : bool
            If True, min-max scale the metric to [0, 1] before embedding.
        arm_scale : float
            Maximum radial distance (arm length).
        jitter : float
            Perpendicular noise to avoid overplotting.
        seed : int
        arm_norm : {"global", "per_arm"}, default "global"
            Ordering-metric normalization across arms.  See
            ``scCS.embedding.build_star_embedding`` for the full
            description.  ``"global"`` preserves relative pseudotime
            ranges across arms (recommended); ``"per_arm"`` restores
            pre-v0.7.3 behavior.
        verbose : bool

        Returns
        -------
        self
        """
        if verbose:
            print(
                f"[scCS] Building star embedding: "
                f"root=\'{self.root}\', "
                f"k={len(self.branches)} fates, "
                f"metric=\'{ordering_metric}\'"
            )

        metric = ordering_metric
        if scale_ordering and isinstance(metric, np.ndarray):
            metric = scale_metric_01(metric)
            if verbose:
                print("[scCS] Metric scaled to [0, 1].")

        self.adata_sub = build_star_embedding(
            self.adata,
            root=self.root,
            branches=self.branches,
            obs_key=self.obs_key,
            ordering_metric=metric,
            invert_ordering=invert_ordering,
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
        arm_norm=arm_norm,
        )
        self._embedding_built = True

        if verbose:
            print(
                f"[scCS] Star embedding stored in scorer.adata_sub.obsm[\'X_sccs\']. "
                f"({self.adata_sub.n_obs} cells)"
            )

        return self

    # ------------------------------------------------------------------
    # Velocity projection
    # ------------------------------------------------------------------

    def project_velocity(self, verbose: bool = True):
        """Project RNA velocity vectors into the scCS star embedding.

        Call after build_embedding().  Uses the full adata's velocity_graph
        (intact graph, correct dimensions) and slices to the subset cells.

        Returns
        -------
        self
        """
        self._check_embedding()
        self._vx, self._vy = project_velocity_star(
            self.adata_sub,
            adata_full=self.adata,
            verbose=verbose,
        )
        return self

    def load_velocity_vectors(self, vx: np.ndarray, vy: np.ndarray):
        """Directly supply pre-computed velocity vectors in scCS space.

        Parameters
        ----------
        vx, vy : np.ndarray, shape (n_cells,)

        Returns
        -------
        self
        """
        self._vx = np.asarray(vx, dtype=float)
        self._vy = np.asarray(vy, dtype=float)
        return self

    # ------------------------------------------------------------------
    # Subset pseudotime recomputation
    # ------------------------------------------------------------------

    def compute_local_pseudotime(
        self,
        scale_01: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """Recompute velocity pseudotime on the subset's induced subgraph.

        After calling this, rebuild the embedding with the corrected pseudotime::

            scorer.build_embedding(ordering_metric=\'pseudotime\')
            scorer.compute_local_pseudotime(scale_01=True)
            scorer.refit_pseudotime()
            scorer.fit()

        Parameters
        ----------
        scale_01 : bool
            If True (default), min-max scale the recomputed pseudotime to [0, 1].
        verbose : bool

        Returns
        -------
        pt_sub : np.ndarray, shape (n_sub_cells,)
        """
        self._check_embedding()
        return compute_local_pseudotime(
            self.adata_sub,
            adata_full=self.adata,
            scale_01=scale_01,
            verbose=verbose,
        )

    def refit_pseudotime(
        self,
        scale_01: bool = True,
        arm_scale: float = 10.0,
        jitter: float = 0.3,
        seed: int = 42,
        arm_norm: str = "global",
        verbose: bool = True,
    ):
        """Rebuild the star embedding using subset-local pseudotime.

        Convenience wrapper that:
        1. Recomputes pseudotime on the subset's induced velocity subgraph.
        2. Optionally scales it to [0, 1] (recommended).
        3. Rebuilds the star embedding using the corrected pseudotime.

        Parameters
        ----------
        scale_01 : bool
            Scale subset pseudotime to [0, 1] before rebuilding.  Default True.
        arm_scale : float
        jitter : float
        seed : int
        arm_norm : {"global", "per_arm"}, default "global"
            Ordering-metric normalization across arms; passed through to
            ``build_star_embedding``.
        verbose : bool

        Returns
        -------
        self
        """
        self._check_embedding()

        # Step 1: recompute pseudotime on the subset subgraph
        pt_sub = compute_local_pseudotime(
            self.adata_sub,
            adata_full=self.adata,
            scale_01=scale_01,
            verbose=verbose,
        )

        # Step 2: map local pseudotime back to full-adata indices
        parent_idx = self.adata_sub.uns.get("sccs", {}).get("parent_indices", None)
        pt_full = np.full(self.adata.n_obs, np.nan)
        if parent_idx is not None:
            pt_full[parent_idx] = pt_sub
        else:
            sub_names = list(self.adata_sub.obs_names)
            full_names = list(self.adata.obs_names)
            name_to_full = {n: i for i, n in enumerate(full_names)}
            for sub_i, name in enumerate(sub_names):
                if name in name_to_full:
                    pt_full[name_to_full[name]] = pt_sub[sub_i]

        # Fill non-subset cells with median
        nan_mask = np.isnan(pt_full)
        if nan_mask.any():
            pt_full[nan_mask] = np.nanmedian(pt_full)

        # Step 3: rebuild embedding with the corrected metric
        if verbose:
            print("[scCS] Rebuilding star embedding with subset-local pseudotime...")

        self.adata_sub = build_star_embedding(
            self.adata,
            root=self.root,
            branches=self.branches,
            obs_key=self.obs_key,
            ordering_metric=pt_full,
            invert_ordering=False,
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
        arm_norm=arm_norm,
        )
        self._embedding_built = True
        self._fitted = False
        self._needs_refit = True
        self._vx = None
        self._vy = None

        if verbose:
            print(
                "[scCS] Embedding rebuilt. Call fit() again to update the FateMap "
                "and velocity projection."
            )

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fate_map(self) -> Optional[FateMap]:
        return self._fate_map

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def embedding(self) -> Optional[np.ndarray]:
        """The X_sccs star embedding coordinates, shape (n_subset_cells, 2)."""
        if self.adata_sub is not None and "X_sccs" in self.adata_sub.obsm:
            return np.array(self.adata_sub.obsm["X_sccs"])
        return None

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_embedding(self):
        if not self._embedding_built:
            raise RuntimeError(
                "Star embedding not built. Call build_embedding() first."
            )

    def _check_fitted(self):
        if not self._fitted:
            if self._needs_refit:
                raise RuntimeError(
                    "Embedding was rebuilt. Call fit() again to update the "
                    "FateMap and velocity projection before scoring."
                )
            raise RuntimeError(
                "Scorer is not fitted. Call fit() first."
            )
        if self._vx is None:
            raise RuntimeError(
                "Velocity vectors not loaded. Call project_velocity() or "
                "load_velocity_vectors() after build_embedding()."
            )
