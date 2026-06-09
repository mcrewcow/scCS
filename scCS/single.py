"""
single.py — SingleScorer: single-condition commitment score analysis for scCS.

Orchestrates:
1. RNA velocity computation (optional, via scVelo)
2. Radial star embedding construction (embedding.py)
3. FateMap construction from user-supplied cluster labels (bifurcation.py)
4. Commitment score computation (scores.py)
5. Driver gene analysis (drivers.py)
6. Pathway enrichment (enrichment.py)
7. Plotting (plot.py)

Quick start
-----------
>>> import scCS
>>> scorer = scCS.SingleScorer(
...     adata,
...     root='17',
...     branches=['Monocyte', 'DC', 'Neutrophil'],
...     obs_key='leiden',
... )
>>> scorer.build_embedding(ordering_metric='pseudotime')
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

from ._base import _BaseScorer, SectorMode
from .bifurcation import FateMap, build_fate_map
from .embedding import run_velocity_pipeline
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
from .drivers import get_velocity_drivers, get_deg_drivers, get_velocity_fate_drivers
from .enrichment import run_enrichment_per_fate


class SingleScorer(_BaseScorer):
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
    root : str
        Label of the progenitor/root cluster in adata.obs[obs_key].
        Example: '17'  (leiden cluster 17)
    branches : list of str
        Labels of the k terminal fate clusters.
        Example: ['Monocyte', 'DC', 'Neutrophil']
    obs_key : str
        Column in adata.obs with cluster labels.  Default: 'leiden'.
    n_angle_bins : int
        Number of angular bins for commitment scoring.  Default: 36 (10° each).
    sector_method : {'centroid', 'equal'}
        How to define angular sectors:
        - 'centroid': anchor sectors to the direction from origin to each
          fate centroid in the star embedding (recommended).
        - 'equal': divide [0°, 360°] into k equal sectors.
    copy : bool
        Work on a copy of adata.

    Examples
    --------
    # k=2 bifurcation
    scorer = SingleScorer(
        adata,
        root='17',
        branches=['homeostatic', 'activated'],
        obs_key='leiden',
    )
    scorer.build_embedding(ordering_metric='pseudotime')
    scorer.fit()
    result = scorer.score()
    scorer.plot_star(result)

    # k=3 with CytoTRACE2
    scorer = SingleScorer(
        adata,
        root='5',
        branches=['FateA', 'FateB', 'FateC'],
        obs_key='cell_type',
    )
    scorer.build_embedding(ordering_metric='cytotrace')
    scorer.fit()
    result = scorer.score()
    scorer.plot_star(result)
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
        super().__init__(
            adata=adata,
            root=root,
            branches=branches,
            obs_key=obs_key,
            n_angle_bins=n_angle_bins,
            sector_method=sector_method,
            copy=copy,
        )

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
    ) -> "SingleScorer":
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
    # Step 4: Fit (build FateMap)
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "SingleScorer":
        """Build the FateMap from the user-supplied cluster labels.

        Must be called after build_embedding().

        This step:
        1. Validates that root and branches
           exist in adata.obs[obs_key].
        2. Computes fate centroids in the X_sccs embedding.
        3. Extracts velocity vectors if not already loaded.

        Returns
        -------
        self
        """
        self._check_embedding()
        self._needs_refit = False  # clear after rebuild

        self._fate_map = build_fate_map(
            self.adata_sub,
            root=self.root,
            branches=self.branches,
            obs_key=self.obs_key,
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
        cell_level: bool = True,
        k_nn: Optional[int] = None,
        n_bootstrap: int = 0,
        bootstrap_ci: float = 0.95,
        bootstrap_seed: int = 42,
        verbose: bool = True,
        write_to_obs: bool = True,
    ) -> CommitmentScoreResult:
        """Compute commitment scores for the full population or a subset.

        Parameters
        ----------
        cell_mask : np.ndarray of bool, shape (n_sub_cells,), optional
            Boolean mask over ``adata_sub`` cells (NOT the full adata).
            If provided, only cells where mask=True contribute to the
            population-level score (M_bin, M_sector, unCS, nCS).
            Per-cell scores are still computed for all cells.
        cell_level : bool
            Whether to compute per-cell fate affinity scores.
        k_nn : int, optional
            If set, compute NN-smoothed per-cell entropy using this many
            nearest neighbors in the scCS embedding (X_sccs).
        n_bootstrap : int
            Number of bootstrap replicates for CS confidence intervals.
            0 (default) disables bootstrapping.  Recommended: 500.
        bootstrap_ci : float
            Confidence interval level for bootstrap.  Default 0.95 (95% CI).
        bootstrap_seed : int
            Random seed for bootstrap resampling.
        verbose : bool
        write_to_obs : bool
            If True (default), write per-cell scores to ``adata_sub.obs``.

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
        bin_edges, M_bin = bin_angles(angles, magnitudes, n_bins=self.n_angle_bins)

        # 3. Sector definition
        fate_map = self._fate_map
        if self.sector_method == "centroid":
            sectors, fate_angles = centroid_sectors(
                fate_map.fate_centroids,
                fate_map.root_centroid,
                n_bins=self.n_angle_bins,
            )
        else:
            sectors = equal_sectors(fate_map.k, n_bins=self.n_angle_bins)
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

        if cell_level:
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
            k_fates = cell_scores.shape[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_s = np.where(cell_scores > 0, np.log(cell_scores), 0.0)
            per_cell_H = -np.sum(cell_scores * log_s, axis=1) / np.log(k_fates)

            if write_to_obs:
                for j, name in enumerate(fate_map.fate_names):
                    self.adata_sub.obs[f"cs_{name}"] = cell_scores[:, j]
                self.adata_sub.obs["cs_dominant_fate"] = [
                    fate_map.fate_names[int(np.argmax(cell_scores[i]))]
                    for i in range(len(cell_scores))
                ]
                self.adata_sub.obs["cs_entropy"] = per_cell_H

            # NN-smoothed entropy
            if k_nn is not None and k_nn > 0:
                coords = np.array(self.adata_sub.obsm["X_sccs"])
                nn_ent = compute_nn_cell_entropy(cell_scores, coords, k_nn)
                if write_to_obs:
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
                n_bins=self.n_angle_bins,
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
        split_by: str,
        cell_level: bool = False,
        n_bootstrap: int = 0,
        verbose: bool = False,
    ) -> dict:
        """Compute commitment scores separately for each value of split_by.

        Useful for comparing commitment across conditions, time points,
        or trajectory directions.

        Parameters
        ----------
        split_by : str
            Column in adata_sub.obs to split by.
        cell_level : bool
        n_bootstrap : int
            Bootstrap replicates for CI.  0 = disabled.
        verbose : bool

        Returns
        -------
        dict mapping subset_value -> CommitmentScoreResult
        """
        self._check_fitted()
        results = {}
        if split_by in self.adata_sub.obs.columns:
            subset_col = self.adata_sub.obs[split_by]
        elif split_by in self.adata.obs.columns:
            subset_col = self.adata.obs.loc[self.adata_sub.obs_names, split_by]
        else:
            raise KeyError(
                f"'{split_by}' not found in adata_sub.obs or adata.obs columns."
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
                cell_level=cell_level,
                n_bootstrap=n_bootstrap,
                verbose=False,
                write_to_obs=False,
            )
            # Warn when all off-diagonal nCS are inf (progenitor-only subset)
            ncs = results[val].pairwise_nCS
            k = ncs.shape[0]
            off_diag = [ncs[i, j] for i in range(k) for j in range(k) if i != j]
            if off_diag and all(np.isinf(v) for v in off_diag):
                warnings.warn(
                    f"Subset '{val}': no cells from any fate arm — "
                    f"pairwise nCS is undefined (inf). "
                    f"This is expected for progenitor-only subsets.",
                    stacklevel=2,
                )
            if verbose:
                print(f"\n--- Subset: {val} ---")
                print(results[val].summary())
        return results

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
        to the full adata so they can be used in downstream analyses.

        Cells not in the embedding subset receive NaN for numeric columns
        and 'unassigned' for categorical columns.

        Parameters
        ----------
        adata : AnnData
            The full dataset (same object passed to SingleScorer.__init__).
        result : CommitmentScoreResult
            Output of scorer.score(cell_level=True).
        prefix : str
            Column prefix.  Default: 'cs_'.
        """
        self._check_fitted()
        if result.cell_scores is None:
            raise ValueError(
                "result.cell_scores is None. "
                "Run scorer.score(cell_level=True) first."
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
        if "sccs_pseudotime" in self.adata_sub.obs:
            pt_col = f"{prefix}sccs_pseudotime"
            pt_arr = np.full(n_full, np.nan)
            pt_arr[sub_to_full[valid]] = self.adata_sub.obs["sccs_pseudotime"].values[valid]
            adata.obs[pt_col] = pt_arr

        print(
            f"[scCS] Labels transferred to adata.obs for {valid.sum()} / {n_full} cells. "
            f"Columns: {[f'{prefix}{f}' for f in result.fate_names] + [dom_col, ent_col]}"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize scorer state to a pickle file.

        Saves the embedding, FateMap, velocity vectors, and configuration.
        The full ``adata`` is NOT saved (too large); pass it again to
        :meth:`load`.

        Parameters
        ----------
        path : str
            Destination file path (e.g., ``'scorer.pkl'``).
        """
        import pickle
        import pathlib

        state = {k: getattr(self, k) for k in [
            "adata_sub", "_fate_map", "_vx", "_vy",
            "root", "branches", "obs_key",
            "n_angle_bins", "sector_method", "_embedding_built", "_fitted",
            "_needs_refit",
        ]}
        pathlib.Path(path).write_bytes(pickle.dumps(state))
        print(f"[scCS] SingleScorer state saved to '{path}'.")

    @classmethod
    def load(cls, path: str, adata) -> "SingleScorer":
        """Load a scorer from a pickle file.

        Parameters
        ----------
        path : str
            Path to a file saved by :meth:`save`.
        adata : AnnData
            The full dataset (same object originally passed to
            ``SingleScorer.__init__``).  Not stored in the pickle.

        Returns
        -------
        SingleScorer
        """
        import pickle
        import pathlib

        state = pickle.loads(pathlib.Path(path).read_bytes())
        scorer = cls.__new__(cls)
        scorer.adata = adata
        for k, v in state.items():
            setattr(scorer, k, v)
        # Ensure _needs_refit exists for pickles from older versions
        if not hasattr(scorer, "_needs_refit"):
            scorer._needs_refit = False
        print(f"[scCS] SingleScorer state loaded from '{path}'.")
        return scorer

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
        n_top_genes: int = 50,
    ) -> dict:
        """Rank genes by mean scVelo velocity in each fate arm.

        Parameters
        ----------
        n_top_genes : int
            Number of top driver genes to print per fate.

        Returns
        -------
        dict : fate_name -> DataFrame[gene, mean_velocity, rank]
        """
        self._check_fitted()
        return get_velocity_drivers(
            self.adata_sub,
            fate_names=self._fate_map.fate_names,
            obs_key=self.obs_key,
            root=self.root,
            n_top_genes=n_top_genes,
        )

    def get_deg_drivers(
        self,
        n_top_genes: int = 50,
        pval_threshold: float = 0.05,
        logfc_threshold: float = 0.25,
    ) -> dict:
        """Find DEGs for each fate arm vs the bifurcation cluster (Wilcoxon).

        Parameters
        ----------
        n_top_genes : int
        pval_threshold : float
        logfc_threshold : float

        Returns
        -------
        dict : fate_name -> DataFrame[gene, logfoldchange, pval, pval_adj, significant]
        """
        self._check_fitted()
        return get_deg_drivers(
            self.adata_sub,
            fate_names=self._fate_map.fate_names,
            obs_key=self.obs_key,
            root=self.root,
            n_top_genes=n_top_genes,
            pval_threshold=pval_threshold,
            logfc_threshold=logfc_threshold,
        )

    def get_velocity_fate_drivers(
        self,
        result: CommitmentScoreResult,
        n_top_genes: int = 50,
        pval_threshold: float = 0.05,
    ) -> dict:
        """Identify driver genes by correlating gene velocity with fate affinity.

        Parameters
        ----------
        result : CommitmentScoreResult
            Output of scorer.score(cell_level=True).
        n_top_genes : int
        pval_threshold : float

        Returns
        -------
        dict : fate_name -> DataFrame[gene, spearman_r, pval, pval_adj,
                                      mean_velocity, delta_velocity, significant]
        """
        self._check_fitted()
        if result.cell_scores is None:
            raise ValueError(
                "result.cell_scores is None. "
                "Run scorer.score(cell_level=True) first."
            )
        return get_velocity_fate_drivers(
            self.adata_sub,
            cell_scores=result.cell_scores,
            fate_names=self._fate_map.fate_names,
            obs_key=self.obs_key,
            root=self.root,
            n_top_genes=n_top_genes,
            pval_threshold=pval_threshold,
        )

    def get_enrichment(
        self,
        deg_drivers: dict,
        gene_sets: Optional[List[str]] = None,
        organism: str = "mouse",
        pval_threshold: float = 0.05,
        logfc_threshold: float = 0.25,
        plot: bool = True,
        n_top_pathways: int = 15,
    ) -> dict:
        """Run pathway enrichment on DEG driver genes per fate arm.

        Parameters
        ----------
        deg_drivers : dict
            Output of get_deg_drivers().
        gene_sets : list of str, optional
        organism : str
        pval_threshold : float
        logfc_threshold : float
        plot : bool
        n_top_pathways : int

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
            pval_threshold=pval_threshold,
            logfc_threshold=logfc_threshold,
            plot=plot,
            n_top_pathways=n_top_pathways,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._fitted:
            status = "fitted"
        elif self._needs_refit:
            status = "needs refit (embedding rebuilt)"
        elif self._embedding_built:
            status = "embedding built"
        else:
            status = "uninitialised"
        return (
            f"SingleScorer("
            f"root='{self.root}', "
            f"branches={self.branches}, "
            f"k={len(self.branches)}, "
            f"status='{status}')"
        )
