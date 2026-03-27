"""
trajectory.py — CommitmentScorer: main user-facing API for scCS.

Orchestrates:
1. RNA velocity computation (optional, via scVelo)
2. Radial star embedding construction (embedding.py)
   → subsets adata to bifurcation + terminal fate cells only
3. FateMap construction from user-supplied cluster labels (bifurcation.py)
4. Commitment score computation (scores.py)
5. Driver gene analysis (drivers.py)
6. Pathway enrichment (enrichment.py)
7. Plotting (plot.py)

The key design change from commitscores:
  - The bifurcation point is a single user-supplied cluster label
    (e.g., leiden cluster '17'), not auto-detected.
  - The embedding is a radial star layout, not FA2/UMAP.
  - Cells are ordered along each arm by a differentiation metric.
  - Only bifurcation + terminal fate cells are included in the embedding.

Quick start
-----------
>>> import scCS
>>> scorer = scCS.CommitmentScorer(
...     adata,
...     bifurcation_cluster='17',
...     terminal_cell_types=['Monocyte', 'DC', 'Neutrophil'],
...     cluster_key='leiden',
... )
>>> scorer.build_embedding(differentiation_metric='pseudotime')
>>> scorer.fit()
>>> result = scorer.score()
>>> print(result.summary())
>>> scorer.plot_star(result)
>>> # Driver genes
>>> vel_drivers = scorer.get_velocity_drivers()
>>> deg_drivers = scorer.get_deg_drivers()
>>> # Pathway enrichment (mouse)
>>> enrichment = scorer.get_enrichment(deg_drivers)
"""

from __future__ import annotations

import warnings
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from .bifurcation import FateMap, build_fate_map
from .embedding import (
    build_star_embedding,
    project_velocity_star,
    run_velocity_pipeline,
    recompute_subset_pseudotime,
    scale_metric_01,
)
from .scores import (
    CommitmentScoreResult,
    bin_angles,
    bootstrap_cs,
    centroid_sectors,
    compute_cell_scores,
    compute_commitment_entropy,       # backward-compat alias
    compute_mean_cell_entropy,
    compute_nn_cell_entropy,
    compute_per_fate_cell_entropy,
    compute_population_entropy,
    compute_commitment_vector,
    compute_magnitudes,
    compute_angles,
    compute_pairwise_cs_matrix,
    compute_sector_magnitudes,
    equal_sectors,
)
from .drivers import get_velocity_drivers, get_deg_drivers
from .enrichment import run_enrichment_per_fate


SectorMode = Literal["centroid", "equal"]


class CommitmentScorer:
    """RNA velocity commitment scorer with radial star embedding.

    Computes commitment scores for a k-furcation defined by a single
    user-supplied bifurcation cluster and k terminal fate clusters.

    The embedding places the bifurcation cluster at the origin and
    arranges each fate on its own radial arm, with cells ordered by
    differentiation level (pseudotime, CytoTRACE2, or custom score).

    Parameters
    ----------
    adata : AnnData
        Single-cell dataset.
    bifurcation_cluster : str
        Label of the progenitor/root cluster in adata.obs[cluster_key].
        Example: '17'  (leiden cluster 17)
    terminal_cell_types : list of str
        Labels of the k terminal fate clusters.
        Example: ['Monocyte', 'DC', 'Neutrophil']
    cluster_key : str
        Column in adata.obs with cluster labels.  Default: 'leiden'.
    n_bins : int
        Number of angular bins for commitment scoring.  Default: 36 (10° each).
    sector_mode : {'centroid', 'equal'}
        How to define angular sectors:
        - 'centroid': anchor sectors to the direction from origin to each
          fate centroid in the star embedding (recommended).
        - 'equal': divide [0°, 360°] into k equal sectors.
    copy : bool
        Work on a copy of adata.

    Examples
    --------
    # k=2 bifurcation
    scorer = CommitmentScorer(
        adata,
        bifurcation_cluster='17',
        terminal_cell_types=['homeostatic', 'activated'],
        cluster_key='leiden',
    )
    scorer.build_embedding(differentiation_metric='pseudotime')
    scorer.fit()
    result = scorer.score()
    scorer.plot_star(result)

    # k=3 with CytoTRACE2
    scorer = CommitmentScorer(
        adata,
        bifurcation_cluster='5',
        terminal_cell_types=['FateA', 'FateB', 'FateC'],
        cluster_key='cell_type',
    )
    scorer.build_embedding(differentiation_metric='cytotrace')
    scorer.fit()
    result = scorer.score()
    scorer.plot_star(result)
    """

    def __init__(
        self,
        adata,
        bifurcation_cluster: str,
        terminal_cell_types: List[str],
        cluster_key: str = "leiden",
        n_bins: int = 36,
        sector_mode: SectorMode = "centroid",
        copy: bool = False,
    ):
        self.adata = adata.copy() if copy else adata
        self.bifurcation_cluster = str(bifurcation_cluster)
        self.terminal_cell_types = list(terminal_cell_types)
        self.cluster_key = cluster_key
        self.n_bins = n_bins
        self.sector_mode = sector_mode

        self._fate_map: Optional[FateMap] = None
        self._vx: Optional[np.ndarray] = None
        self._vy: Optional[np.ndarray] = None
        self._embedding_built = False
        self._fitted = False
        # adata_sub: subset containing only bifurcation + terminal fate cells
        self.adata_sub = None

    # ------------------------------------------------------------------
    # Step 1 (optional): RNA velocity
    # ------------------------------------------------------------------

    def compute_velocity(
        self,
        mode: str = "dynamical",
        n_top_genes: int = 2000,
        n_pcs: int = 30,
        n_neighbors: int = 30,
        min_shared_counts: int = 20,
        verbose: bool = True,
    ) -> "CommitmentScorer":
        """Run the full scVelo RNA velocity pipeline.

        Call this if adata does not yet have velocity vectors.
        Requires 'spliced' and 'unspliced' layers.

        Parameters
        ----------
        mode : {'dynamical', 'stochastic', 'steady_state'}
        n_top_genes, n_pcs, n_neighbors, min_shared_counts : int
        verbose : bool

        Returns
        -------
        self
        """
        run_velocity_pipeline(
            self.adata,
            mode=mode,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            min_shared_counts=min_shared_counts,
            verbose=verbose,
        )
        return self

    # ------------------------------------------------------------------
    # Step 2: Build the radial star embedding
    # ------------------------------------------------------------------

    def build_embedding(
        self,
        differentiation_metric: Union[str, np.ndarray] = "pseudotime",
        invert_metric: bool = False,
        scale_metric: bool = False,
        arm_scale: float = 10.0,
        jitter: float = 0.3,
        seed: int = 42,
        verbose: bool = True,
    ) -> "CommitmentScorer":
        """Construct the radial star embedding (X_sccs).

        Places the bifurcation cluster at the origin and arranges each
        terminal fate on its own radial arm.  Cells are ordered along
        each arm by the differentiation metric.

        Parameters
        ----------
        differentiation_metric : str or np.ndarray
            Metric used to order cells along each arm:
            - 'pseudotime'  : scVelo velocity_pseudotime (computed if absent)
            - 'cytotrace'   : CytoTRACE2 score (adata.obs['cytotrace2_score'])
            - any str       : any numeric column in adata.obs
            - np.ndarray    : per-cell scores, shape (n_cells,)
            Higher value = more differentiated = farther from center.
        invert_metric : bool
            Set True if your metric is inverted (high = less differentiated).
            Note: CytoTRACE2 inversion is handled automatically.
        scale_metric : bool
            If True, min-max scale the metric to [0, 1] before embedding.
            Useful when the metric has a compressed range within the subset
            (e.g., full-adata pseudotime).  For pseudotime, prefer calling
            ``rebuild_embedding_with_subset_pseudotime()`` instead, which
            recomputes pseudotime on the subset subgraph before scaling.
            Default: False.
        arm_scale : float
            Maximum radial distance (arm length).
        jitter : float
            Perpendicular noise to avoid overplotting.
        seed : int
        verbose : bool

        Returns
        -------
        self
        """
        if verbose:
            print(
                f"[scCS] Building star embedding: "
                f"bifurcation='{self.bifurcation_cluster}', "
                f"k={len(self.terminal_cell_types)} fates, "
                f"metric='{differentiation_metric}'"
            )

        metric = differentiation_metric
        if scale_metric and isinstance(metric, np.ndarray):
            metric = scale_metric_01(metric)
            if verbose:
                print("[scCS] Metric scaled to [0, 1].")

        self.adata_sub = build_star_embedding(
            self.adata,
            bifurcation_cluster=self.bifurcation_cluster,
            terminal_cell_types=self.terminal_cell_types,
            cluster_key=self.cluster_key,
            differentiation_metric=metric,
            invert_metric=invert_metric,
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
        )
        self._embedding_built = True

        if verbose:
            print(
                f"[scCS] Star embedding stored in scorer.adata_sub.obsm['X_sccs']. "
                f"({self.adata_sub.n_obs} cells)"
            )

        return self

    # ------------------------------------------------------------------
    # Step 3: Project velocity into the star embedding
    # ------------------------------------------------------------------

    def project_velocity(self, verbose: bool = True) -> "CommitmentScorer":
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
            adata_full=self.adata,   # pass full adata so graph dims are correct
            verbose=verbose,
        )
        return self

    def load_velocity_vectors(
        self, vx: np.ndarray, vy: np.ndarray
    ) -> "CommitmentScorer":
        """Directly supply pre-computed velocity vectors in scCS space.

        Use this when you have computed velocity externally or want to
        use a custom displacement field.

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
    # Step 4: Fit (build FateMap)
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "CommitmentScorer":
        """Build the FateMap from the user-supplied cluster labels.

        Must be called after build_embedding().

        This step:
        1. Validates that bifurcation_cluster and terminal_cell_types
           exist in adata.obs[cluster_key].
        2. Computes fate centroids in the X_sccs embedding.
        3. Extracts velocity vectors if not already loaded.

        Returns
        -------
        self
        """
        self._check_embedding()

        self._fate_map = build_fate_map(
            self.adata_sub,
            bifurcation_cluster=self.bifurcation_cluster,
            terminal_cell_types=self.terminal_cell_types,
            cluster_key=self.cluster_key,
            verbose=verbose,
        )

        # Extract velocity vectors if not already loaded
        if self._vx is None or self._vy is None:
            try:
                self.project_velocity(verbose=verbose)
            except Exception as e:
                warnings.warn(
                    f"Could not project velocity ({e}). "
                    "Call project_velocity() or load_velocity_vectors() manually.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if verbose:
            print(self._fate_map.summary())

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Step 5: Score computation
    # ------------------------------------------------------------------

    def score(
        self,
        cell_mask: Optional[np.ndarray] = None,
        compute_cell_level: bool = True,
        k_nn: Optional[int] = None,
        n_bootstrap: int = 0,
        bootstrap_ci: float = 0.95,
        bootstrap_seed: int = 42,
        verbose: bool = True,
    ) -> CommitmentScoreResult:
        """Compute commitment scores for the full population or a subset.

        Parameters
        ----------
        cell_mask : np.ndarray of bool, shape (n_sub_cells,), optional
            Boolean mask over ``adata_sub`` cells (NOT the full adata).
            If provided, only cells where mask=True contribute to the
            population-level score (M_bin, M_sector, unCS, nCS).
            Per-cell scores are still computed for all cells.
        compute_cell_level : bool
            Whether to compute per-cell fate affinity scores.
            When True, ``result.mean_cell_entropy``, ``result.per_fate_entropy``,
            and (if k_nn is set) ``result.nn_cell_entropy`` are populated.
            When False, these fields are NaN / None.
        k_nn : int, optional
            If set, compute NN-smoothed per-cell entropy using this many
            nearest neighbors in the scCS embedding (X_sccs).
            Result stored in ``result.nn_cell_entropy`` and
            ``adata_sub.obs['cs_nn_entropy']``.
            Use ``scorer.plot_nn_entropy_elbow()`` to choose a good value.
        n_bootstrap : int
            Number of bootstrap replicates for CS confidence intervals.
            0 (default) disables bootstrapping.  Recommended: 500.
        bootstrap_ci : float
            Confidence interval level for bootstrap.  Default 0.95 (95% CI).
        bootstrap_seed : int
            Random seed for bootstrap resampling.
        verbose : bool

        Returns
        -------
        CommitmentScoreResult
        """
        self._check_fitted()

        vx = self._vx.copy()
        vy = self._vy.copy()

        # Apply cell mask for population-level scoring
        if cell_mask is not None:
            vx_pop = vx[cell_mask]
            vy_pop = vy[cell_mask]
        else:
            vx_pop = vx
            vy_pop = vy

        # 1. Magnitudes and angles
        magnitudes = compute_magnitudes(vx_pop, vy_pop)
        angles = compute_angles(vx_pop, vy_pop)

        # 2. Angular binning
        bin_edges, M_bin = bin_angles(angles, magnitudes, n_bins=self.n_bins)

        # 3. Sector definition
        fate_map = self._fate_map
        if self.sector_mode == "centroid":
            sectors, fate_angles = centroid_sectors(
                fate_map.fate_centroids,
                fate_map.root_centroid,
                n_bins=self.n_bins,
            )
        else:
            sectors = equal_sectors(fate_map.k, n_bins=self.n_bins)
            fate_angles = fate_map.arm_angles_deg

        # 4. Sector magnitudes
        M_sector = compute_sector_magnitudes(M_bin, sectors)

        # 5. Cell counts per fate
        if cell_mask is not None:
            n_cells_per_fate = np.array([
                int(cell_mask[idx].sum()) for idx in fate_map.fate_cell_indices
            ], dtype=float)
        else:
            n_cells_per_fate = np.array([
                len(idx) for idx in fate_map.fate_cell_indices
            ], dtype=float)

        # 6. Commitment vector and population-level entropy
        commitment_vector = compute_commitment_vector(M_sector)
        pop_entropy = compute_population_entropy(commitment_vector)

        # 7. Pairwise CS matrices
        pairwise_unCS = compute_pairwise_cs_matrix(M_sector, normalized=False)
        pairwise_nCS = compute_pairwise_cs_matrix(
            M_sector, n_cells_per_fate=n_cells_per_fate, normalized=True
        )

        # 8. Per-cell scores, per-fate entropy, NN entropy
        cell_scores = None
        mean_cell_ent = float("nan")
        per_fate_ent = np.full(fate_map.k, float("nan"))
        nn_ent = None

        if compute_cell_level:
            cell_scores = compute_cell_scores(
                vx, vy,
                fate_map.fate_centroids,
                fate_map.root_centroid,
            )

            # Global mean per-cell entropy
            mean_cell_ent = compute_mean_cell_entropy(cell_scores)

            # Per-fate binary cell entropy — shape (k,)
            per_fate_ent = compute_per_fate_cell_entropy(cell_scores)

            # Write per-cell fate scores and raw entropy to adata_sub.obs
            for j, name in enumerate(fate_map.fate_names):
                self.adata_sub.obs[f"cs_{name}"] = cell_scores[:, j]
            self.adata_sub.obs["cs_dominant_fate"] = [
                fate_map.fate_names[int(np.argmax(cell_scores[i]))]
                for i in range(len(cell_scores))
            ]
            k_fates = cell_scores.shape[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_s = np.where(cell_scores > 0, np.log(cell_scores), 0.0)
            per_cell_H = -np.sum(cell_scores * log_s, axis=1) / np.log(k_fates)
            self.adata_sub.obs["cs_entropy"] = per_cell_H

            # NN-smoothed entropy
            if k_nn is not None and k_nn > 0:
                coords = np.array(self.adata_sub.obsm["X_sccs"])
                nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn)
                self.adata_sub.obs["cs_nn_entropy"] = nn_ent

        # 9. Bootstrap CI (optional)
        boot_ci = None
        if n_bootstrap > 0:
            if verbose:
                print(f"[scCS] Computing bootstrap CI (n={n_bootstrap})...")
            boot_ci = bootstrap_cs(
                vx_pop, vy_pop,
                sectors=sectors,
                n_cells_per_fate=n_cells_per_fate,
                n_bins=self.n_bins,
                n_bootstrap=n_bootstrap,
                ci=bootstrap_ci,
                seed=bootstrap_seed,
                normalized=True,
            )

        result = CommitmentScoreResult(
            fate_names=fate_map.fate_names,
            M_bin=M_bin,
            bin_edges=bin_edges,
            sectors=sectors,
            M_sector=M_sector,
            n_cells_per_fate=n_cells_per_fate,
            commitment_vector=commitment_vector,
            population_entropy=pop_entropy,
            mean_cell_entropy=mean_cell_ent,
            per_fate_entropy=per_fate_ent,
            pairwise_unCS=pairwise_unCS,
            pairwise_nCS=pairwise_nCS,
            cell_scores=cell_scores,
            fate_angles=fate_angles,
            cell_obs_names=np.array(self.adata_sub.obs_names),
            nn_cell_entropy=nn_ent,
            nn_k=k_nn,
            bootstrap_ci=boot_ci,
        )

        if verbose:
            print(result.summary())

        return result

    def score_per_subset(
        self,
        subset_key: str,
        compute_cell_level: bool = False,
        n_bootstrap: int = 0,
        verbose: bool = False,
    ) -> dict:
        """Compute commitment scores separately for each value of subset_key.

        Useful for comparing commitment across conditions, time points,
        or trajectory directions.

        .. note::
            The ``cell_mask`` is applied to ``adata_sub`` (the embedding subset),
            not the full adata.  Only cells present in the embedding are scored.

        Parameters
        ----------
        subset_key : str
            Column in adata_sub.obs to split by.
        compute_cell_level : bool
        n_bootstrap : int
            Bootstrap replicates for CI.  0 = disabled.
        verbose : bool

        Returns
        -------
        dict mapping subset_value -> CommitmentScoreResult
        """
        self._check_fitted()
        results = {}
        # Prefer adata_sub.obs; fall back to adata.obs aligned by obs_names.
        # Users typically add metadata to the full adata, not adata_sub.
        if subset_key in self.adata_sub.obs.columns:
            subset_col = self.adata_sub.obs[subset_key]
        elif subset_key in self.adata.obs.columns:
            subset_col = self.adata.obs.loc[self.adata_sub.obs_names, subset_key]
        else:
            raise KeyError(
                f"'{subset_key}' not found in adata_sub.obs or adata.obs columns."
            )
        for val in subset_col.unique():
            mask = (subset_col == val).values
            if mask.sum() < 10:
                warnings.warn(
                    f"Subset '{val}' has only {mask.sum()} cells in adata_sub. Skipping.",
                    stacklevel=2,
                )
                continue
            results[val] = self.score(
                cell_mask=mask,
                compute_cell_level=compute_cell_level,
                n_bootstrap=n_bootstrap,
                verbose=verbose,
            )
            if verbose:
                print(f"\n--- Subset: {val} ---")
                print(results[val].summary())
        return results

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
    # Subset pseudotime recomputation
    # ------------------------------------------------------------------

    def recompute_subset_pseudotime(
        self,
        scale_01: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """Recompute velocity pseudotime on the subset's induced subgraph.

        When ``build_embedding(differentiation_metric='pseudotime')`` is used,
        pseudotime is resolved on the full adata before subsetting.  This can
        compress the pseudotime range within the subset, leaving cells poorly
        distributed along the arms (e.g., all clustered near the origin).

        This method extracts the velocity_graph submatrix for the subset cells,
        recomputes pseudotime locally, and optionally scales it to [0, 1].
        The result is stored in ``adata_sub.obs['velocity_pseudotime_sub']``.

        After calling this, rebuild the embedding with the corrected pseudotime::

            scorer.build_embedding(differentiation_metric='pseudotime')
            scorer.recompute_subset_pseudotime(scale_01=True)
            scorer.rebuild_embedding_with_subset_pseudotime()
            scorer.fit()

        Parameters
        ----------
        scale_01 : bool
            If True (default), min-max scale the recomputed pseudotime to [0, 1]
            within the subset.  Recommended: ensures cells span the full arm
            length regardless of where the subset sits in the global range.
            If False, raw pseudotime values are kept (useful for cross-condition
            comparisons where absolute pseudotime ordering matters).
        verbose : bool

        Returns
        -------
        pt_sub : np.ndarray, shape (n_sub_cells,)
            Subset-local pseudotime, also stored in
            ``adata_sub.obs['velocity_pseudotime_sub']``.
        """
        self._check_embedding()
        return recompute_subset_pseudotime(
            self.adata_sub,
            adata_full=self.adata,
            scale_01=scale_01,
            verbose=verbose,
        )

    def rebuild_embedding_with_subset_pseudotime(
        self,
        scale_01: bool = True,
        arm_scale: float = 10.0,
        jitter: float = 0.3,
        seed: int = 42,
        verbose: bool = True,
    ) -> "CommitmentScorer":
        """Rebuild the star embedding using subset-local pseudotime.

        Convenience wrapper that:
        1. Recomputes pseudotime on the subset's induced velocity subgraph.
        2. Optionally scales it to [0, 1] (recommended).
        3. Rebuilds the star embedding using the corrected pseudotime.

        This corrects the arm-coverage problem that arises when the full-adata
        pseudotime is compressed within the subset (cells cluster near the
        origin instead of spanning the arm).

        Parameters
        ----------
        scale_01 : bool
            Scale subset pseudotime to [0, 1] before rebuilding.  Default True.
        arm_scale : float
            Maximum radial distance (arm length).
        jitter : float
            Perpendicular noise to avoid overplotting.
        seed : int
        verbose : bool

        Returns
        -------
        self
        """
        self._check_embedding()

        # Step 1: recompute pseudotime on the subset subgraph
        pt_sub = recompute_subset_pseudotime(
            self.adata_sub,
            adata_full=self.adata,
            scale_01=scale_01,
            verbose=verbose,
        )

        # Step 2: map subset pseudotime back to full-adata indices so that
        # build_star_embedding can slice it correctly via keep_mask
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

        # Fill non-subset cells with median (they will be excluded by keep_mask anyway)
        nan_mask = np.isnan(pt_full)
        if nan_mask.any():
            pt_full[nan_mask] = np.nanmedian(pt_full)

        # Step 3: rebuild embedding with the corrected metric
        if verbose:
            print("[scCS] Rebuilding star embedding with subset-local pseudotime...")

        self.adata_sub = build_star_embedding(
            self.adata,
            bifurcation_cluster=self.bifurcation_cluster,
            terminal_cell_types=self.terminal_cell_types,
            cluster_key=self.cluster_key,
            differentiation_metric=pt_full,   # pass as pre-resolved array
            invert_metric=False,               # already oriented correctly
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
        )
        self._embedding_built = True
        self._fitted = False   # must re-fit after rebuilding
        self._vx = None
        self._vy = None

        if verbose:
            print(
                "[scCS] Embedding rebuilt. Call fit() again to update the FateMap "
                "and velocity projection."
            )

        return self

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
            raise RuntimeError(
                "CommitmentScorer is not fitted. Call fit() first."
            )
        if self._vx is None:
            raise RuntimeError(
                "Velocity vectors not loaded. Call project_velocity() or "
                "load_velocity_vectors() after build_embedding()."
            )

    # ------------------------------------------------------------------
    # Label transfer to full adata
    # ------------------------------------------------------------------

    def transfer_labels(
        self,
        adata,
        result: CommitmentScoreResult,
        prefix: str = "cs_",
    ) -> None:
        """Write per-cell commitment scores back to the full adata.

        After scoring, per-cell fate affinities, dominant fate, and entropy
        are stored in ``adata_sub.obs``.  This method transfers those columns
        to the full adata so they can be used in downstream analyses (e.g.,
        UMAP coloring, integration with other tools).

        Cells not in the embedding subset receive NaN for numeric columns
        and 'unassigned' for categorical columns.

        Parameters
        ----------
        adata : AnnData
            The full dataset (same object passed to CommitmentScorer.__init__).
        result : CommitmentScoreResult
            Output of scorer.score(compute_cell_level=True).
        prefix : str
            Column prefix.  Default: 'cs_'.

        Columns written to adata.obs
        ----------------------------
        ``{prefix}{fate_name}``     : per-cell fate affinity (float, NaN outside subset)
        ``{prefix}dominant_fate``   : dominant fate label (str, 'unassigned' outside subset)
        ``{prefix}entropy``         : per-cell commitment entropy (float, NaN outside subset)
        ``{prefix}nn_entropy``      : NN-smoothed entropy if computed (float, NaN outside subset)
        ``{prefix}pseudotime_sub``  : subset-local pseudotime if computed (float, NaN outside subset)
        """
        self._check_fitted()
        if result.cell_scores is None:
            raise ValueError(
                "result.cell_scores is None. "
                "Run scorer.score(compute_cell_level=True) first."
            )

        n_full = adata.n_obs
        full_names = list(adata.obs_names)
        name_to_full_idx = {n: i for i, n in enumerate(full_names)}

        sub_names = list(self.adata_sub.obs_names)
        sub_to_full = np.array([
            name_to_full_idx.get(n, -1) for n in sub_names
        ])
        valid = sub_to_full >= 0

        # Per-fate affinity
        for j, fate_name in enumerate(result.fate_names):
            col = f"{prefix}{fate_name}"
            arr = np.full(n_full, np.nan)
            arr[sub_to_full[valid]] = result.cell_scores[valid, j]
            adata.obs[col] = arr

        # Dominant fate
        dom_col = f"{prefix}dominant_fate"
        dom_arr = np.full(n_full, "unassigned", dtype=object)
        sub_dom = self.adata_sub.obs.get("cs_dominant_fate", None)
        if sub_dom is not None:
            dom_arr[sub_to_full[valid]] = sub_dom.values[valid]
        adata.obs[dom_col] = dom_arr

        # Per-cell entropy
        ent_col = f"{prefix}entropy"
        ent_arr = np.full(n_full, np.nan)
        sub_ent = self.adata_sub.obs.get("cs_entropy", None)
        if sub_ent is not None:
            ent_arr[sub_to_full[valid]] = sub_ent.values[valid]
        adata.obs[ent_col] = ent_arr

        # NN-smoothed entropy
        if result.nn_cell_entropy is not None:
            nn_col = f"{prefix}nn_entropy"
            nn_arr = np.full(n_full, np.nan)
            nn_arr[sub_to_full[valid]] = result.nn_cell_entropy[valid]
            adata.obs[nn_col] = nn_arr

        # Subset-local pseudotime
        if "velocity_pseudotime_sub" in self.adata_sub.obs:
            pt_col = f"{prefix}pseudotime_sub"
            pt_arr = np.full(n_full, np.nan)
            pt_arr[sub_to_full[valid]] = self.adata_sub.obs["velocity_pseudotime_sub"].values[valid]
            adata.obs[pt_col] = pt_arr

        print(
            f"[scCS] Labels transferred to adata.obs for {valid.sum()} / {n_full} cells. "
            f"Columns: {[f'{prefix}{f}' for f in result.fate_names] + [dom_col, ent_col]}"
        )

    # ------------------------------------------------------------------
    # Plotting shortcuts
    # ------------------------------------------------------------------

    def plot_star(self, result: CommitmentScoreResult, **kwargs):
        """Radial star embedding plot — primary visualization."""
        from .plot import plot_star_embedding
        return plot_star_embedding(self.adata_sub, result, **kwargs)

    def plot_rose(self, result: CommitmentScoreResult, **kwargs):
        """Rose/polar plot of cumulative magnitudes per angular bin."""
        from .plot import plot_rose
        return plot_rose(result, **kwargs)

    def plot_pairwise_cs(self, result: CommitmentScoreResult, **kwargs):
        """Heatmap of pairwise normalized commitment scores."""
        from .plot import plot_pairwise_cs
        return plot_pairwise_cs(result, **kwargs)

    def plot_commitment_bar(self, result: CommitmentScoreResult, **kwargs):
        """Bar chart of unCS vs nCS per fate pair."""
        from .plot import plot_commitment_bar
        return plot_commitment_bar(result, **kwargs)

    def plot_commitment_heatmap(self, result: CommitmentScoreResult, **kwargs):
        """Per-cell fate affinity heatmap."""
        from .plot import plot_commitment_heatmap
        return plot_commitment_heatmap(result, **kwargs)

    def plot_subset_comparison(self, subset_results: dict, **kwargs):
        """Compare commitment scores across subsets."""
        from .plot import plot_subset_comparison
        return plot_subset_comparison(subset_results, **kwargs)

    def plot_nn_entropy_elbow(self, **kwargs):
        """Elbow plots for choosing k_nn for NN-smoothed entropy.

        Sweeps k_nn values and plots mean NN entropy (all cells) and per fate.
        Call after fit().  Requires no prior score() call.

        Parameters
        ----------
        k_nn_range : list or range, optional
            k_nn values to sweep.  Default: range(5, 51, 5).
        **kwargs
            Passed to :func:`scCS.plot.plot_nn_entropy_elbow`.

        Returns
        -------
        fig : matplotlib Figure
        """
        from .plot import plot_nn_entropy_elbow
        return plot_nn_entropy_elbow(self, **kwargs)

    # ------------------------------------------------------------------
    # Driver genes
    # ------------------------------------------------------------------

    def get_velocity_drivers(
        self,
        n_top: int = 50,
    ) -> dict:
        """Rank genes by mean scVelo velocity in each fate arm.

        Requires the 'velocity' layer in adata_sub (from scVelo pipeline).
        High positive velocity = gene is being upregulated along that fate.

        Parameters
        ----------
        n_top : int
            Number of top driver genes to print per fate.

        Returns
        -------
        dict : fate_name -> DataFrame[gene, mean_velocity, rank]
        """
        self._check_fitted()
        return get_velocity_drivers(
            self.adata_sub,
            fate_names=self._fate_map.fate_names,
            cluster_key=self.cluster_key,
            bifurcation_cluster=self.bifurcation_cluster,
            n_top=n_top,
        )

    def get_deg_drivers(
        self,
        n_top: int = 50,
        pval_cutoff: float = 0.05,
        logfc_cutoff: float = 0.25,
    ) -> dict:
        """Find DEGs for each fate arm vs the bifurcation cluster (Wilcoxon).

        Parameters
        ----------
        n_top : int
            Number of top significant DEGs to print per fate.
        pval_cutoff : float
            Adjusted p-value threshold for significance.
        logfc_cutoff : float
            Minimum absolute log fold-change for significance.

        Returns
        -------
        dict : fate_name -> DataFrame[gene, logfoldchange, pval, pval_adj, significant]
        """
        self._check_fitted()
        return get_deg_drivers(
            self.adata_sub,
            fate_names=self._fate_map.fate_names,
            cluster_key=self.cluster_key,
            bifurcation_cluster=self.bifurcation_cluster,
            n_top=n_top,
            pval_cutoff=pval_cutoff,
            logfc_cutoff=logfc_cutoff,
        )

    def get_enrichment(
        self,
        deg_drivers: dict,
        gene_sets: Optional[List[str]] = None,
        organism: str = "mouse",
        pval_cutoff: float = 0.05,
        logfc_cutoff: float = 0.25,
        plot: bool = True,
        n_top_terms: int = 15,
    ) -> dict:
        """Run pathway enrichment on DEG driver genes per fate arm.

        Runs Enrichr ORA on up- and down-regulated DEGs separately for
        each fate arm.  Default gene sets: KEGG_2019_Mouse,
        GO_Biological_Process_2021, Reactome_2022.

        Parameters
        ----------
        deg_drivers : dict
            Output of get_deg_drivers().
        gene_sets : list of str, optional
            Enrichr gene set libraries.  Defaults to mouse KEGG + GO BP + Reactome.
        organism : str
            'mouse' or 'human'.
        pval_cutoff : float
        logfc_cutoff : float
        plot : bool
            If True, generate dot plots per fate per direction.
        n_top_terms : int
            Number of top enriched terms to plot.

        Returns
        -------
        dict : fate_name -> {'up': DataFrame, 'down': DataFrame}
        """
        self._check_fitted()
        return run_enrichment_per_fate(
            deg_drivers=deg_drivers,
            fate_names=self._fate_map.fate_names,
            gene_sets=gene_sets,
            organism=organism,
            pval_cutoff=pval_cutoff,
            logfc_cutoff=logfc_cutoff,
            plot=plot,
            n_top_terms=n_top_terms,
        )
