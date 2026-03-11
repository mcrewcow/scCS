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
)
from .scores import (
    CommitmentScoreResult,
    bin_angles,
    centroid_sectors,
    compute_cell_scores,
    compute_commitment_entropy,
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

        self.adata_sub = build_star_embedding(
            self.adata,
            bifurcation_cluster=self.bifurcation_cluster,
            terminal_cell_types=self.terminal_cell_types,
            cluster_key=self.cluster_key,
            differentiation_metric=differentiation_metric,
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
        verbose: bool = True,
    ) -> CommitmentScoreResult:
        """Compute commitment scores for the full population or a subset.

        Parameters
        ----------
        cell_mask : np.ndarray of bool, shape (n_cells,), optional
            If provided, only cells where mask=True contribute to the
            population-level score (M_bin, M_sector, unCS, nCS).
            Per-cell scores are still computed for all cells.
        compute_cell_level : bool
            Whether to compute per-cell fate affinity scores.
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

        # 6. Commitment vector and entropy
        commitment_vector = compute_commitment_vector(M_sector)
        entropy = compute_commitment_entropy(commitment_vector)

        # 7. Pairwise CS matrices
        pairwise_unCS = compute_pairwise_cs_matrix(M_sector, normalized=False)
        pairwise_nCS = compute_pairwise_cs_matrix(
            M_sector, n_cells_per_fate=n_cells_per_fate, normalized=True
        )

        # 8. Per-cell scores
        cell_scores = None
        if compute_cell_level:
            cell_scores = compute_cell_scores(
                vx, vy,
                fate_map.fate_centroids,
                fate_map.root_centroid,
            )
            # Write back to adata_sub.obs
            for j, name in enumerate(fate_map.fate_names):
                self.adata_sub.obs[f"cs_{name}"] = cell_scores[:, j]
            self.adata_sub.obs["cs_dominant_fate"] = [
                fate_map.fate_names[int(np.argmax(cell_scores[i]))]
                for i in range(len(cell_scores))
            ]
            self.adata_sub.obs["cs_entropy"] = [
                compute_commitment_entropy(cell_scores[i])
                for i in range(len(cell_scores))
            ]

        result = CommitmentScoreResult(
            fate_names=fate_map.fate_names,
            M_bin=M_bin,
            bin_edges=bin_edges,
            sectors=sectors,
            M_sector=M_sector,
            n_cells_per_fate=n_cells_per_fate,
            commitment_vector=commitment_vector,
            commitment_entropy=entropy,
            pairwise_unCS=pairwise_unCS,
            pairwise_nCS=pairwise_nCS,
            cell_scores=cell_scores,
            fate_angles=fate_angles,
            cell_obs_names=np.array(self.adata_sub.obs_names),
        )

        if verbose:
            print(result.summary())

        return result

    def score_per_subset(
        self,
        subset_key: str,
        compute_cell_level: bool = False,
        verbose: bool = False,
    ) -> dict:
        """Compute commitment scores separately for each value of subset_key.

        Useful for comparing commitment across conditions, time points,
        or trajectory directions.

        Parameters
        ----------
        subset_key : str
            Column in adata.obs to split by.
        compute_cell_level : bool
        verbose : bool

        Returns
        -------
        dict mapping subset_value -> CommitmentScoreResult
        """
        self._check_fitted()
        results = {}
        for val in self.adata.obs[subset_key].unique():
            mask = (self.adata.obs[subset_key] == val).values
            if mask.sum() < 10:
                warnings.warn(
                    f"Subset '{val}' has only {mask.sum()} cells. Skipping.",
                    stacklevel=2,
                )
                continue
            results[val] = self.score(
                cell_mask=mask,
                compute_cell_level=compute_cell_level,
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
