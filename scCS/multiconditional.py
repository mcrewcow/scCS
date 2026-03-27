"""
multiconditional.py — Multi-condition commitment score analysis for scCS.

Extends the single-condition CommitmentScorer to handle 2 or more experimental
conditions (e.g., treatment vs. control, multiple time points, genotypes).

Key design principle: shared embedding
---------------------------------------
All conditions are embedded in a SINGLE shared star layout built on the pooled
data.  This is critical — if each condition had its own embedding, the arm
angles would differ and CS values would not be comparable across conditions.

Architecture
------------
MultiConditionScorer
    Wraps CommitmentScorer.  Pools all conditions for embedding, then scores
    each condition separately using cell masks on the shared embedding.

Tier 1 — Core multi-condition API
    score_all_conditions()          : dict[condition -> CommitmentScoreResult]
    score_per_condition()           : alias with pseudotime-aware options

Tier 2 — Statistical comparison
    compare_conditions()            : permutation test (k=2) or Kruskal-Wallis
                                      + pairwise Mann-Whitney (k>2) on per-cell
                                      fate affinity scores
    compute_delta_CS()              : ΔCS = nCS_A − nCS_B with bootstrap CI
    plot_condition_comparison()     : violin/box plots of per-cell affinities
                                      split by condition, one panel per fate

Tier 3 — Advanced
    fit_mixed_model()               : linear mixed-effects model on per-cell
                                      fate affinity scores (condition fixed,
                                      sample_id random) via statsmodels MixedLM
    trajectory_shift()              : KS test + Wasserstein distance on
                                      pseudotime distributions per fate arm
                                      across conditions
    plot_trajectory_shift()         : visualization of pseudotime distributions
                                      per condition per fate arm

Usage
-----
>>> mscorer = scCS.MultiConditionScorer(
...     adata,
...     bifurcation_cluster='17',
...     terminal_cell_types=['homeostatic', 'activated'],
...     condition_key='treatment',
...     cluster_key='leiden',
... )
>>> mscorer.build_embedding(differentiation_metric='pseudotime')
>>> mscorer.fit()
>>> results = mscorer.score_all_conditions()
>>> delta = mscorer.compute_delta_CS('control', 'treated')
>>> stats = mscorer.compare_conditions(results)
>>> mscorer.plot_condition_comparison(results)
>>> shift = mscorer.trajectory_shift(results)
>>> mscorer.plot_trajectory_shift(shift)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.figure
import numpy as np
import pandas as pd

from .trajectory import CommitmentScorer
from .scores import (
    CommitmentScoreResult,
    bootstrap_cs,
    compute_cell_scores,
    compute_magnitudes,
    compute_angles,
    bin_angles,
    compute_sector_magnitudes,
    compute_pairwise_cs_matrix,
    centroid_sectors,
    equal_sectors,
    compute_commitment_vector,
    compute_population_entropy,
    compute_mean_cell_entropy,
    compute_per_fate_cell_entropy,
    compute_nn_cell_entropy,
)
from .plot import _fate_colors, PROGENITOR_COLOR


# ---------------------------------------------------------------------------
# MultiConditionScorer
# ---------------------------------------------------------------------------

class MultiConditionScorer:
    """RNA velocity commitment scorer for multi-condition experiments.

    Builds a SHARED star embedding on the pooled data from all conditions,
    then scores each condition separately.  This ensures arm geometry is
    identical across conditions, making CS values directly comparable.

    Parameters
    ----------
    adata : AnnData
        Full single-cell dataset containing all conditions.
    bifurcation_cluster : str
        Label of the progenitor/root cluster in adata.obs[cluster_key].
    terminal_cell_types : list of str
        Labels of the k terminal fate clusters.
    condition_key : str
        Column in adata.obs with condition labels (e.g., 'treatment').
    cluster_key : str
        Column in adata.obs with cluster labels.  Default: 'leiden'.
    n_bins : int
        Number of angular bins.  Default: 36.
    sector_mode : {'centroid', 'equal'}
        Sector definition strategy.
    copy : bool
        Work on a copy of adata.

    Examples
    --------
    >>> mscorer = MultiConditionScorer(
    ...     adata,
    ...     bifurcation_cluster='17',
    ...     terminal_cell_types=['homeostatic', 'activated'],
    ...     condition_key='treatment',
    ...     cluster_key='leiden',
    ... )
    >>> mscorer.build_embedding(differentiation_metric='pseudotime')
    >>> mscorer.fit()
    >>> results = mscorer.score_all_conditions()
    >>> delta = mscorer.compute_delta_CS('control', 'treated')
    >>> stats = mscorer.compare_conditions(results)
    """

    def __init__(
        self,
        adata,
        bifurcation_cluster: str,
        terminal_cell_types: List[str],
        condition_key: str,
        cluster_key: str = "leiden",
        n_bins: int = 36,
        sector_mode: Literal["centroid", "equal"] = "centroid",
        copy: bool = False,
    ):
        self.adata = adata.copy() if copy else adata
        self.bifurcation_cluster = str(bifurcation_cluster)
        self.terminal_cell_types = list(terminal_cell_types)
        self.condition_key = condition_key
        self.cluster_key = cluster_key
        self.n_bins = n_bins
        self.sector_mode = sector_mode

        # Validate condition key
        if condition_key not in adata.obs:
            raise ValueError(
                f"condition_key='{condition_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        self.conditions = sorted(adata.obs[condition_key].astype(str).unique().tolist())
        if len(self.conditions) < 2:
            raise ValueError(
                f"condition_key='{condition_key}' has only {len(self.conditions)} "
                "unique value(s).  Need at least 2 conditions."
            )

        # Internal CommitmentScorer built on pooled data
        self._scorer: Optional[CommitmentScorer] = None
        self._fitted = False

        print(
            f"[scCS] MultiConditionScorer initialized.\n"
            f"       Conditions ({len(self.conditions)}): {self.conditions}\n"
            f"       Bifurcation: '{bifurcation_cluster}', "
            f"Fates: {terminal_cell_types}"
        )

    # ------------------------------------------------------------------
    # Step 1: Build shared embedding (delegates to CommitmentScorer)
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
    ) -> "MultiConditionScorer":
        """Build the shared star embedding on pooled data from all conditions.

        The embedding is built on ALL cells (all conditions pooled), ensuring
        that arm geometry is identical across conditions.

        Parameters
        ----------
        differentiation_metric : str or np.ndarray
            See CommitmentScorer.build_embedding().
        invert_metric : bool
        scale_metric : bool
            Min-max scale the metric to [0, 1] before embedding.
        arm_scale : float
        jitter : float
        seed : int
        verbose : bool

        Returns
        -------
        self
        """
        if verbose:
            print(
                f"[scCS] Building SHARED embedding on pooled data "
                f"({self.adata.n_obs} cells, {len(self.conditions)} conditions)..."
            )

        self._scorer = CommitmentScorer(
            self.adata,
            bifurcation_cluster=self.bifurcation_cluster,
            terminal_cell_types=self.terminal_cell_types,
            cluster_key=self.cluster_key,
            n_bins=self.n_bins,
            sector_mode=self.sector_mode,
            copy=False,
        )
        self._scorer.build_embedding(
            differentiation_metric=differentiation_metric,
            invert_metric=invert_metric,
            scale_metric=scale_metric,
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
            verbose=verbose,
        )
        return self

    def rebuild_embedding_with_subset_pseudotime(
        self,
        scale_01: bool = True,
        arm_scale: float = 10.0,
        jitter: float = 0.3,
        seed: int = 42,
        verbose: bool = True,
    ) -> "MultiConditionScorer":
        """Rebuild the shared embedding using subset-local pseudotime.

        See CommitmentScorer.rebuild_embedding_with_subset_pseudotime().
        """
        self._check_embedding()
        self._scorer.rebuild_embedding_with_subset_pseudotime(
            scale_01=scale_01,
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
            verbose=verbose,
        )
        self._fitted = False
        return self

    # ------------------------------------------------------------------
    # Step 2: Fit (delegates to CommitmentScorer)
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "MultiConditionScorer":
        """Fit the shared FateMap and project velocity.

        Must be called after build_embedding().

        Returns
        -------
        self
        """
        self._check_embedding()
        self._scorer.fit(verbose=verbose)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Tier 1: Score all conditions
    # ------------------------------------------------------------------

    def score_all_conditions(
        self,
        compute_cell_level: bool = True,
        k_nn: Optional[int] = None,
        n_bootstrap: int = 0,
        bootstrap_ci: float = 0.95,
        verbose: bool = True,
    ) -> Dict[str, CommitmentScoreResult]:
        """Compute commitment scores separately for each condition.

        Uses the shared embedding and FateMap.  Each condition's cells are
        masked from the shared adata_sub, so arm geometry is identical.

        Parameters
        ----------
        compute_cell_level : bool
            Compute per-cell fate affinity scores.
        k_nn : int, optional
            NN-smoothed entropy neighbors.
        n_bootstrap : int
            Bootstrap replicates for CI.  0 = disabled.
        bootstrap_ci : float
            CI level for bootstrap.
        verbose : bool

        Returns
        -------
        dict : condition_label -> CommitmentScoreResult
        """
        self._check_fitted()
        results: Dict[str, CommitmentScoreResult] = {}

        for cond in self.conditions:
            # Mask over adata_sub (shared embedding subset)
            mask = (
                self._scorer.adata_sub.obs[self.condition_key].astype(str) == cond
            ).values

            n_cond = mask.sum()
            if n_cond < 10:
                warnings.warn(
                    f"Condition '{cond}' has only {n_cond} cells in the embedding. "
                    "Skipping.",
                    stacklevel=2,
                )
                continue

            if verbose:
                print(f"\n[scCS] Scoring condition: '{cond}' ({n_cond} cells)...")

            results[cond] = self._scorer.score(
                cell_mask=mask,
                compute_cell_level=compute_cell_level,
                k_nn=k_nn,
                n_bootstrap=n_bootstrap,
                bootstrap_ci=bootstrap_ci,
                verbose=verbose,
            )

        return results

    def score_per_condition(
        self,
        compute_cell_level: bool = True,
        scale_01_pseudotime: bool = False,
        n_bootstrap: int = 0,
        verbose: bool = True,
    ) -> Dict[str, CommitmentScoreResult]:
        """Alias for score_all_conditions() with pseudotime-aware options.

        Parameters
        ----------
        compute_cell_level : bool
        scale_01_pseudotime : bool
            If True, pseudotime is NOT scaled to [0, 1] per condition
            (preserves absolute ordering for cross-condition comparison).
            This is the default for multi-condition analysis.
            Note: this parameter is informational — actual scaling is
            controlled during build_embedding().
        n_bootstrap : int
        verbose : bool

        Returns
        -------
        dict : condition_label -> CommitmentScoreResult
        """
        if scale_01_pseudotime:
            warnings.warn(
                "scale_01_pseudotime=True is informational only. "
                "Pseudotime scaling is set during build_embedding(). "
                "For cross-condition comparison, use scale_01=False in "
                "rebuild_embedding_with_subset_pseudotime().",
                UserWarning,
                stacklevel=2,
            )
        return self.score_all_conditions(
            compute_cell_level=compute_cell_level,
            n_bootstrap=n_bootstrap,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Tier 2: Statistical comparison
    # ------------------------------------------------------------------

    def compute_delta_CS(
        self,
        condition_a: str,
        condition_b: str,
        n_bootstrap: int = 500,
        ci: float = 0.95,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict:
        """Compute ΔCS = nCS_A − nCS_B with bootstrap confidence intervals.

        For each pair of fates (i, j), computes the difference in normalized
        commitment score between condition A and condition B, with a bootstrap
        CI obtained by resampling cells within each condition.

        Parameters
        ----------
        condition_a, condition_b : str
            Condition labels (must be in self.conditions).
        n_bootstrap : int
            Number of bootstrap replicates.  Default 500.
        ci : float
            Confidence interval level.  Default 0.95.
        seed : int
        verbose : bool

        Returns
        -------
        dict with keys:
            'delta_nCS'  : np.ndarray (k, k) — nCS_A − nCS_B
            'ci_low'     : np.ndarray (k, k) — lower CI bound on delta
            'ci_high'    : np.ndarray (k, k) — upper CI bound on delta
            'nCS_A'      : np.ndarray (k, k) — nCS for condition A
            'nCS_B'      : np.ndarray (k, k) — nCS for condition B
            'fate_names' : list of str
            'condition_a': str
            'condition_b': str
            'n_bootstrap': int
            'ci_level'   : float
        """
        self._check_fitted()
        for cond in [condition_a, condition_b]:
            if cond not in self.conditions:
                raise ValueError(
                    f"Condition '{cond}' not found. "
                    f"Available: {self.conditions}"
                )

        fate_map = self._scorer._fate_map
        vx = self._scorer._vx
        vy = self._scorer._vy

        # Masks over adata_sub
        mask_a = (
            self._scorer.adata_sub.obs[self.condition_key].astype(str) == condition_a
        ).values
        mask_b = (
            self._scorer.adata_sub.obs[self.condition_key].astype(str) == condition_b
        ).values

        # Sector definition (shared)
        if self.sector_mode == "centroid":
            sectors, _ = centroid_sectors(
                fate_map.fate_centroids,
                fate_map.root_centroid,
                n_bins=self.n_bins,
            )
        else:
            sectors = equal_sectors(fate_map.k, n_bins=self.n_bins)

        n_cells_a = np.array([
            int(mask_a[idx].sum()) for idx in fate_map.fate_cell_indices
        ], dtype=float)
        n_cells_b = np.array([
            int(mask_b[idx].sum()) for idx in fate_map.fate_cell_indices
        ], dtype=float)

        # Point estimates
        def _score_mask(mask, n_cells_per_fate):
            vx_m, vy_m = vx[mask], vy[mask]
            mag = compute_magnitudes(vx_m, vy_m)
            ang = compute_angles(vx_m, vy_m)
            _, M_bin = bin_angles(ang, mag, n_bins=self.n_bins)
            M_sec = compute_sector_magnitudes(M_bin, sectors)
            return compute_pairwise_cs_matrix(
                M_sec, n_cells_per_fate=n_cells_per_fate, normalized=True
            )

        nCS_A = _score_mask(mask_a, n_cells_a)
        nCS_B = _score_mask(mask_b, n_cells_b)
        delta = nCS_A - nCS_B

        # Bootstrap
        rng = np.random.default_rng(seed)
        alpha = (1.0 - ci) / 2.0
        k = fate_map.k
        boot_deltas = np.zeros((n_bootstrap, k, k))

        idx_a = np.where(mask_a)[0]
        idx_b = np.where(mask_b)[0]

        for b in range(n_bootstrap):
            # Resample within each condition
            boot_a = rng.choice(idx_a, size=len(idx_a), replace=True)
            boot_b = rng.choice(idx_b, size=len(idx_b), replace=True)

            mask_ba = np.zeros(len(vx), dtype=bool)
            mask_ba[boot_a] = True
            mask_bb = np.zeros(len(vx), dtype=bool)
            mask_bb[boot_b] = True

            nCS_ba = _score_mask(mask_ba, n_cells_a)
            nCS_bb = _score_mask(mask_bb, n_cells_b)
            boot_deltas[b] = nCS_ba - nCS_bb

        boot_deltas = np.where(np.isinf(boot_deltas), np.nan, boot_deltas)

        result = {
            "delta_nCS":   delta,
            "ci_low":      np.nanpercentile(boot_deltas, alpha * 100, axis=0),
            "ci_high":     np.nanpercentile(boot_deltas, (1 - alpha) * 100, axis=0),
            "nCS_A":       nCS_A,
            "nCS_B":       nCS_B,
            "fate_names":  fate_map.fate_names,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "n_bootstrap": n_bootstrap,
            "ci_level":    ci,
        }

        if verbose:
            ci_pct = int(ci * 100)
            print(f"\n=== ΔCS: '{condition_a}' − '{condition_b}' ===")
            df_delta = pd.DataFrame(
                delta, index=fate_map.fate_names, columns=fate_map.fate_names
            )
            print("  ΔnCS (point estimate):")
            print(df_delta.round(3).to_string())
            print(f"\n  {ci_pct}% CI low:")
            print(
                pd.DataFrame(
                    result["ci_low"],
                    index=fate_map.fate_names, columns=fate_map.fate_names,
                ).round(3).to_string()
            )
            print(f"\n  {ci_pct}% CI high:")
            print(
                pd.DataFrame(
                    result["ci_high"],
                    index=fate_map.fate_names, columns=fate_map.fate_names,
                ).round(3).to_string()
            )

        return result

    def compare_conditions(
        self,
        results: Dict[str, CommitmentScoreResult],
        test: Literal["permutation", "kruskal"] = "auto",
        n_permutations: int = 1000,
        pval_cutoff: float = 0.05,
        seed: int = 42,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Statistical comparison of per-cell fate affinity scores across conditions.

        For k=2 conditions: permutation test (shuffle condition labels, recompute
        mean per-cell affinity difference, get empirical null distribution).

        For k>2 conditions: Kruskal-Wallis test across all conditions, followed
        by pairwise Mann-Whitney U tests with Bonferroni correction.

        Both tests operate on per-cell fate affinity scores (``cell_scores``),
        which are more statistically powerful than comparing scalar CS values.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions() with compute_cell_level=True.
        test : {'auto', 'permutation', 'kruskal'}
            Statistical test to use.  'auto' selects permutation for k=2
            and kruskal for k>2.
        n_permutations : int
            Number of permutations for the permutation test.  Default 1000.
        pval_cutoff : float
            Significance threshold.  Default 0.05.
        seed : int
        verbose : bool

        Returns
        -------
        pd.DataFrame with columns:
            fate, test, statistic, pval, pval_adj, significant
            [+ comparison column for pairwise tests]
        """
        self._check_fitted()

        # Validate that cell_scores are available
        for cond, res in results.items():
            if res.cell_scores is None:
                raise ValueError(
                    f"cell_scores not available for condition '{cond}'. "
                    "Re-run score_all_conditions(compute_cell_level=True)."
                )

        fate_names = list(results.values())[0].fate_names
        k_cond = len(results)

        if test == "auto":
            test = "permutation" if k_cond == 2 else "kruskal"

        rows = []

        if test == "permutation" and k_cond == 2:
            cond_a, cond_b = list(results.keys())
            scores_a = results[cond_a].cell_scores  # (n_a, k_fates)
            scores_b = results[cond_b].cell_scores  # (n_b, k_fates)

            rng = np.random.default_rng(seed)

            for j, fate in enumerate(fate_names):
                a_vals = scores_a[:, j]
                b_vals = scores_b[:, j]
                obs_diff = np.mean(a_vals) - np.mean(b_vals)

                # Permutation null
                pooled = np.concatenate([a_vals, b_vals])
                n_a = len(a_vals)
                null_diffs = np.zeros(n_permutations)
                for p in range(n_permutations):
                    perm = rng.permutation(pooled)
                    null_diffs[p] = perm[:n_a].mean() - perm[n_a:].mean()

                pval = (np.abs(null_diffs) >= np.abs(obs_diff)).mean()
                rows.append({
                    "fate": fate,
                    "comparison": f"{cond_a} vs {cond_b}",
                    "test": "permutation",
                    "statistic": float(obs_diff),
                    "pval": float(pval),
                    "mean_A": float(a_vals.mean()),
                    "mean_B": float(b_vals.mean()),
                })

            df = pd.DataFrame(rows)
            # Bonferroni correction across fates
            df["pval_adj"] = np.minimum(df["pval"] * len(fate_names), 1.0)
            df["significant"] = df["pval_adj"] < pval_cutoff

        else:
            # Kruskal-Wallis + pairwise Mann-Whitney
            from scipy.stats import kruskal, mannwhitneyu
            from itertools import combinations

            cond_list = list(results.keys())
            all_scores = {c: results[c].cell_scores for c in cond_list}

            for j, fate in enumerate(fate_names):
                groups = [all_scores[c][:, j] for c in cond_list]

                # Kruskal-Wallis
                try:
                    stat_kw, pval_kw = kruskal(*groups)
                except Exception:
                    stat_kw, pval_kw = np.nan, np.nan

                rows.append({
                    "fate": fate,
                    "comparison": "all",
                    "test": "kruskal-wallis",
                    "statistic": float(stat_kw),
                    "pval": float(pval_kw),
                    "mean_A": np.nan,
                    "mean_B": np.nan,
                })

                # Pairwise Mann-Whitney
                for ca, cb in combinations(cond_list, 2):
                    try:
                        stat_mw, pval_mw = mannwhitneyu(
                            all_scores[ca][:, j],
                            all_scores[cb][:, j],
                            alternative="two-sided",
                        )
                    except Exception:
                        stat_mw, pval_mw = np.nan, np.nan

                    rows.append({
                        "fate": fate,
                        "comparison": f"{ca} vs {cb}",
                        "test": "mann-whitney",
                        "statistic": float(stat_mw),
                        "pval": float(pval_mw),
                        "mean_A": float(all_scores[ca][:, j].mean()),
                        "mean_B": float(all_scores[cb][:, j].mean()),
                    })

            df = pd.DataFrame(rows)
            # Bonferroni correction within each test type
            for test_type in df["test"].unique():
                mask = df["test"] == test_type
                df.loc[mask, "pval_adj"] = np.minimum(
                    df.loc[mask, "pval"] * mask.sum(), 1.0
                )
            df["significant"] = df["pval_adj"] < pval_cutoff

        if verbose:
            print("\n=== Condition comparison ===")
            sig = df[df["significant"]]
            print(f"  Test: {test}  |  Significant results: {len(sig)} / {len(df)}")
            if len(sig) > 0:
                print(sig[["fate", "comparison", "test", "statistic", "pval_adj"]].to_string(index=False))
            else:
                print("  No significant differences at pval_adj < {pval_cutoff}.")

        return df

    def plot_condition_comparison(
        self,
        results: Dict[str, CommitmentScoreResult],
        plot_type: Literal["violin", "box", "strip"] = "violin",
        color_map: Optional[Dict[str, str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Violin/box plots of per-cell fate affinity scores by condition.

        One panel per fate, showing the distribution of per-cell affinity
        scores split by condition.  This is more informative than comparing
        scalar CS values because it shows the full distribution.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions(compute_cell_level=True).
        plot_type : {'violin', 'box', 'strip'}
        color_map : dict, optional
            condition_label -> hex color.
        figsize : tuple, optional
        title : str, optional
        save_path : str, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fate_names = list(results.values())[0].fate_names
        k_fates = len(fate_names)
        conditions = list(results.keys())

        # Build long-form DataFrame
        rows = []
        for cond, res in results.items():
            if res.cell_scores is None:
                continue
            for j, fate in enumerate(fate_names):
                for val in res.cell_scores[:, j]:
                    rows.append({"condition": cond, "fate": fate, "affinity": val})
        df = pd.DataFrame(rows)

        # Colors for conditions
        if color_map is None:
            from .plot import FATE_PALETTE
            color_map = {c: FATE_PALETTE[i % len(FATE_PALETTE)]
                         for i, c in enumerate(conditions)}

        ncols = min(k_fates, 3)
        nrows = int(np.ceil(k_fates / ncols))
        if figsize is None:
            figsize = (ncols * 4.0, nrows * 3.5)

        sns.set_theme(style="ticks")
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for j, fate in enumerate(fate_names):
            row, col = divmod(j, ncols)
            ax = axes[row][col]
            sub = df[df["fate"] == fate]

            palette = [color_map.get(c, "#888888") for c in conditions]

            if plot_type == "violin":
                sns.violinplot(
                    data=sub, x="condition", y="affinity",
                    palette=palette, ax=ax, inner="box",
                    order=conditions, cut=0,
                )
            elif plot_type == "box":
                sns.boxplot(
                    data=sub, x="condition", y="affinity",
                    palette=palette, ax=ax, order=conditions,
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
                )
            else:  # strip
                sns.stripplot(
                    data=sub, x="condition", y="affinity",
                    palette=palette, ax=ax, order=conditions,
                    size=2, alpha=0.4, jitter=True,
                )

            ax.set_title(fate, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Fate affinity" if col == 0 else "")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=20)
            sns.despine(ax=ax)

        for j in range(k_fates, nrows * ncols):
            row, col = divmod(j, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            title or f"Per-cell fate affinity by condition ({plot_type})",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Tier 3: Advanced — mixed-effects model
    # ------------------------------------------------------------------

    def fit_mixed_model(
        self,
        results: Dict[str, CommitmentScoreResult],
        sample_key: Optional[str] = None,
        reference_condition: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Linear mixed-effects model on per-cell fate affinity scores.

        Models per-cell fate affinity as a function of condition (fixed effect)
        with optional sample/replicate as a random effect.  This is the
        statistically correct approach when you have multiple biological
        replicates per condition.

        Model (per fate j):
            affinity_ij ~ condition_i + (1 | sample_id_i)

        Uses statsmodels MixedLM.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions(compute_cell_level=True).
        sample_key : str, optional
            Column in adata_sub.obs with sample/replicate IDs.
            If None, each cell is treated as its own replicate (no random
            effect — equivalent to a simple linear model).
        reference_condition : str, optional
            Reference condition for the fixed effect.  Defaults to the
            first condition alphabetically.
        verbose : bool

        Returns
        -------
        pd.DataFrame with columns:
            fate, condition, coef, std_err, z_score, pval, pval_adj,
            ci_low, ci_high, significant
        """
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            raise ImportError(
                "statsmodels is required for mixed-effects modeling. "
                "pip install statsmodels"
            )

        fate_names = list(results.values())[0].fate_names
        conditions = list(results.keys())

        if reference_condition is None:
            reference_condition = sorted(conditions)[0]
        if reference_condition not in conditions:
            raise ValueError(
                f"reference_condition='{reference_condition}' not in conditions: "
                f"{conditions}"
            )

        # Build long-form DataFrame with cell-level data
        rows = []
        for cond, res in results.items():
            if res.cell_scores is None:
                continue
            obs_names = res.cell_obs_names if res.cell_obs_names is not None else []
            for i, obs_name in enumerate(obs_names):
                row = {"condition": cond, "obs_name": obs_name}
                for j, fate in enumerate(fate_names):
                    row[f"affinity_{fate}"] = res.cell_scores[i, j]
                if sample_key is not None and sample_key in self._scorer.adata_sub.obs:
                    row["sample_id"] = str(
                        self._scorer.adata_sub.obs.loc[obs_name, sample_key]
                        if obs_name in self._scorer.adata_sub.obs_names
                        else "unknown"
                    )
                else:
                    row["sample_id"] = obs_name  # each cell = own group (no random effect)
                rows.append(row)

        df_long = pd.DataFrame(rows)

        # Set reference condition
        df_long["condition"] = pd.Categorical(
            df_long["condition"],
            categories=[reference_condition]
            + [c for c in conditions if c != reference_condition],
        )

        all_rows = []
        for fate in fate_names:
            col = f"affinity_{fate}"
            if col not in df_long.columns:
                continue

            try:
                if sample_key is not None:
                    # Mixed model with random intercept per sample
                    model = smf.mixedlm(
                        f"{col} ~ C(condition)",
                        data=df_long,
                        groups=df_long["sample_id"],
                    )
                    fit = model.fit(reml=True, method="lbfgs")
                else:
                    # OLS (no random effect)
                    model = smf.ols(f"{col} ~ C(condition)", data=df_long)
                    fit = model.fit()

                for param_name in fit.params.index:
                    if "condition" not in param_name:
                        continue
                    # Extract condition label from parameter name
                    cond_label = (
                        param_name
                        .replace(f"C(condition)[T.", "")
                        .replace("]", "")
                        .strip()
                    )
                    ci = fit.conf_int().loc[param_name]
                    all_rows.append({
                        "fate": fate,
                        "condition": cond_label,
                        "reference": reference_condition,
                        "coef": float(fit.params[param_name]),
                        "std_err": float(fit.bse[param_name]),
                        "z_score": float(fit.tvalues[param_name]),
                        "pval": float(fit.pvalues[param_name]),
                        "ci_low": float(ci.iloc[0]),
                        "ci_high": float(ci.iloc[1]),
                    })

            except Exception as e:
                warnings.warn(
                    f"Mixed model failed for fate '{fate}': {e}",
                    stacklevel=2,
                )

        if not all_rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_rows)
        # Bonferroni correction
        result_df["pval_adj"] = np.minimum(result_df["pval"] * len(result_df), 1.0)
        result_df["significant"] = result_df["pval_adj"] < 0.05
        result_df = result_df.sort_values(["fate", "pval_adj"]).reset_index(drop=True)

        if verbose:
            print("\n=== Mixed-effects model results ===")
            sig = result_df[result_df["significant"]]
            print(f"  Significant effects: {len(sig)} / {len(result_df)}")
            if len(sig) > 0:
                print(
                    sig[["fate", "condition", "coef", "std_err", "pval_adj"]]
                    .to_string(index=False)
                )

        return result_df

    # ------------------------------------------------------------------
    # Tier 3: Advanced — trajectory shift analysis
    # ------------------------------------------------------------------

    def trajectory_shift(
        self,
        results: Dict[str, CommitmentScoreResult],
        pseudotime_col: str = "velocity_pseudotime_sub",
        n_bootstrap: int = 500,
        seed: int = 42,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Test whether pseudotime distributions differ across conditions per fate arm.

        For each fate arm and each pair of conditions, computes:
        - Kolmogorov-Smirnov (KS) statistic and p-value
        - Wasserstein distance (Earth Mover's Distance)
        - Bootstrap CI on the Wasserstein distance

        A significant KS test or large Wasserstein distance indicates that
        cells commit earlier or later under one condition vs. another.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions().
        pseudotime_col : str
            Column in adata_sub.obs with pseudotime values.
            Defaults to 'velocity_pseudotime_sub' (subset-local pseudotime).
            Falls back to 'velocity_pseudotime' if not found.
        n_bootstrap : int
            Bootstrap replicates for Wasserstein CI.  Default 500.
        seed : int
        verbose : bool

        Returns
        -------
        pd.DataFrame with columns:
            fate, comparison, ks_stat, ks_pval, wasserstein,
            wasserstein_ci_low, wasserstein_ci_high,
            mean_pt_A, mean_pt_B, delta_mean_pt, significant
        """
        from scipy.stats import ks_2samp
        try:
            from scipy.stats import wasserstein_distance
        except ImportError:
            from scipy.stats import energy_distance as wasserstein_distance

        self._check_fitted()

        fate_names = self._scorer._fate_map.fate_names
        conditions = list(results.keys())

        # Resolve pseudotime column
        if pseudotime_col not in self._scorer.adata_sub.obs:
            fallback = "velocity_pseudotime"
            if fallback in self._scorer.adata_sub.obs:
                warnings.warn(
                    f"'{pseudotime_col}' not found. Using '{fallback}'. "
                    "Run recompute_subset_pseudotime() for better results.",
                    stacklevel=2,
                )
                pseudotime_col = fallback
            else:
                raise ValueError(
                    f"Neither '{pseudotime_col}' nor 'velocity_pseudotime' found "
                    "in adata_sub.obs. Run recompute_subset_pseudotime() first."
                )

        pt_all = np.array(
            self._scorer.adata_sub.obs[pseudotime_col], dtype=float
        )
        cluster_labels = self._scorer.adata_sub.obs[self.cluster_key].astype(str).values
        cond_labels = self._scorer.adata_sub.obs[self.condition_key].astype(str).values

        rng = np.random.default_rng(seed)
        from itertools import combinations

        rows = []
        for fate in fate_names:
            fate_mask = cluster_labels == str(fate)

            for ca, cb in combinations(conditions, 2):
                mask_a = fate_mask & (cond_labels == ca)
                mask_b = fate_mask & (cond_labels == cb)

                pt_a = pt_all[mask_a]
                pt_b = pt_all[mask_b]

                if len(pt_a) < 5 or len(pt_b) < 5:
                    warnings.warn(
                        f"Too few cells for fate '{fate}', "
                        f"'{ca}' (n={len(pt_a)}) or '{cb}' (n={len(pt_b)}). "
                        "Skipping.",
                        stacklevel=2,
                    )
                    continue

                # KS test
                ks_stat, ks_pval = ks_2samp(pt_a, pt_b)

                # Wasserstein distance
                w_obs = wasserstein_distance(pt_a, pt_b)

                # Bootstrap CI on Wasserstein
                w_boot = np.zeros(n_bootstrap)
                for b in range(n_bootstrap):
                    ba = rng.choice(pt_a, size=len(pt_a), replace=True)
                    bb = rng.choice(pt_b, size=len(pt_b), replace=True)
                    w_boot[b] = wasserstein_distance(ba, bb)

                rows.append({
                    "fate": fate,
                    "comparison": f"{ca} vs {cb}",
                    "condition_a": ca,
                    "condition_b": cb,
                    "ks_stat": float(ks_stat),
                    "ks_pval": float(ks_pval),
                    "wasserstein": float(w_obs),
                    "wasserstein_ci_low": float(np.percentile(w_boot, 2.5)),
                    "wasserstein_ci_high": float(np.percentile(w_boot, 97.5)),
                    "mean_pt_A": float(pt_a.mean()),
                    "mean_pt_B": float(pt_b.mean()),
                    "delta_mean_pt": float(pt_a.mean() - pt_b.mean()),
                    "n_cells_A": int(len(pt_a)),
                    "n_cells_B": int(len(pt_b)),
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Bonferroni correction on KS p-values
        df["ks_pval_adj"] = np.minimum(df["ks_pval"] * len(df), 1.0)
        df["significant"] = df["ks_pval_adj"] < 0.05
        df = df.sort_values(["fate", "ks_pval_adj"]).reset_index(drop=True)

        if verbose:
            print("\n=== Trajectory shift analysis ===")
            sig = df[df["significant"]]
            print(f"  Significant shifts: {len(sig)} / {len(df)}")
            if len(sig) > 0:
                print(
                    sig[[
                        "fate", "comparison", "ks_stat", "ks_pval_adj",
                        "wasserstein", "delta_mean_pt",
                    ]].to_string(index=False)
                )

        return df

    def plot_trajectory_shift(
        self,
        shift_df: pd.DataFrame,
        pseudotime_col: str = "velocity_pseudotime_sub",
        color_map: Optional[Dict[str, str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Visualize pseudotime distributions per condition per fate arm.

        Produces a grid of KDE plots: one row per fate arm, one column per
        pairwise comparison.  Overlaid KDEs show how pseudotime distributions
        shift between conditions.  Wasserstein distance and KS p-value are
        annotated on each panel.

        Parameters
        ----------
        shift_df : pd.DataFrame
            Output of trajectory_shift().
        pseudotime_col : str
            Column in adata_sub.obs with pseudotime values.
        color_map : dict, optional
            condition_label -> hex color.
        figsize : tuple, optional
        title : str, optional
        save_path : str, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        self._check_fitted()

        if pseudotime_col not in self._scorer.adata_sub.obs:
            pseudotime_col = "velocity_pseudotime"

        pt_all = np.array(
            self._scorer.adata_sub.obs[pseudotime_col], dtype=float
        )
        cluster_labels = self._scorer.adata_sub.obs[self.cluster_key].astype(str).values
        cond_labels = self._scorer.adata_sub.obs[self.condition_key].astype(str).values

        fate_names = shift_df["fate"].unique().tolist()
        comparisons = shift_df["comparison"].unique().tolist()

        if color_map is None:
            from .plot import FATE_PALETTE
            all_conds = self.conditions
            color_map = {c: FATE_PALETTE[i % len(FATE_PALETTE)]
                         for i, c in enumerate(all_conds)}

        n_fates = len(fate_names)
        n_comps = len(comparisons)
        if figsize is None:
            figsize = (n_comps * 4.0, n_fates * 3.0)

        sns.set_theme(style="ticks")
        fig, axes = plt.subplots(
            n_fates, n_comps,
            figsize=figsize,
            squeeze=False,
        )

        for fi, fate in enumerate(fate_names):
            fate_mask = cluster_labels == str(fate)
            for ci, comp in enumerate(comparisons):
                ax = axes[fi][ci]
                row = shift_df[
                    (shift_df["fate"] == fate) & (shift_df["comparison"] == comp)
                ]
                if row.empty:
                    ax.set_visible(False)
                    continue

                ca = row["condition_a"].values[0]
                cb = row["condition_b"].values[0]

                pt_a = pt_all[fate_mask & (cond_labels == ca)]
                pt_b = pt_all[fate_mask & (cond_labels == cb)]

                sns.kdeplot(
                    pt_a, ax=ax, color=color_map.get(ca, "#0072B2"),
                    fill=True, alpha=0.35, label=ca, linewidth=1.5,
                )
                sns.kdeplot(
                    pt_b, ax=ax, color=color_map.get(cb, "#D55E00"),
                    fill=True, alpha=0.35, label=cb, linewidth=1.5,
                )

                ks_p = row["ks_pval_adj"].values[0]
                w = row["wasserstein"].values[0]
                sig_str = "*" if ks_p < 0.05 else "ns"
                ax.set_title(
                    f"{fate}\nW={w:.3f}  KS p={ks_p:.3f} {sig_str}",
                    fontsize=9,
                )
                ax.set_xlabel("Pseudotime", fontsize=8)
                ax.set_ylabel("Density" if ci == 0 else "", fontsize=8)
                if fi == 0 and ci == 0:
                    ax.legend(fontsize=7, frameon=False)
                sns.despine(ax=ax)

        fig.suptitle(
            title or "Trajectory shift: pseudotime distributions by condition",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Convenience: transfer labels to full adata
    # ------------------------------------------------------------------

    def transfer_labels(
        self,
        results: Dict[str, CommitmentScoreResult],
        prefix: str = "cs_",
    ) -> None:
        """Transfer per-cell commitment scores to the full adata for all conditions.

        Calls CommitmentScorer.transfer_labels() for each condition's result,
        writing condition-specific columns to adata.obs.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions(compute_cell_level=True).
        prefix : str
            Column prefix.  Default: 'cs_'.
        """
        self._check_fitted()
        for cond, res in results.items():
            cond_prefix = f"{prefix}{cond}_"
            self._scorer.transfer_labels(self.adata, res, prefix=cond_prefix)

    # ------------------------------------------------------------------
    # Plotting shortcuts (delegate to scorer)
    # ------------------------------------------------------------------

    def plot_star(self, result: CommitmentScoreResult, **kwargs):
        """Radial star embedding plot."""
        self._check_fitted()
        return self._scorer.plot_star(result, **kwargs)

    def plot_condition_star(
        self,
        results: Dict[str, CommitmentScoreResult],
        color_map: Optional[Dict[str, str]] = None,
        figsize_per_panel: Tuple[float, float] = (6, 6),
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Side-by-side star embedding plots, one per condition.

        All panels share the same arm geometry and color scale.

        Parameters
        ----------
        results : dict
        color_map : dict, optional
            fate_name -> hex color (for fate coloring, not condition).
        figsize_per_panel : tuple
        save_path : str, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt
        from .plot import plot_star_embedding

        self._check_fitted()
        conditions = list(results.keys())
        n = len(conditions)
        fig, axes = plt.subplots(
            1, n,
            figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
            squeeze=False,
        )

        for i, cond in enumerate(conditions):
            plot_star_embedding(
                self._scorer.adata_sub,
                results[cond],
                color_by="fate",
                color_map=color_map,
                ax=axes[0][i],
                title=cond,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scorer(self) -> Optional[CommitmentScorer]:
        """The underlying shared CommitmentScorer."""
        return self._scorer

    @property
    def adata_sub(self):
        """The shared embedding subset AnnData."""
        if self._scorer is not None:
            return self._scorer.adata_sub
        return None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_embedding(self):
        if self._scorer is None or not self._scorer._embedding_built:
            raise RuntimeError(
                "Shared embedding not built. Call build_embedding() first."
            )

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "MultiConditionScorer not fitted. Call fit() first."
            )
