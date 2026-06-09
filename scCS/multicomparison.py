"""
multicomparison.py — MultiScorer: multi-condition (3+) commitment score analysis for scCS.

Extends the pairwise PairScorer to handle 3 or more experimental conditions
(e.g., multiple drug treatments, time points, genotypes).

Key design principle: shared embedding
---------------------------------------
All conditions are embedded in a SINGLE shared star layout built on the pooled
data.  This is critical — if each condition had its own embedding, the arm
angles would differ and CS values would not be comparable across conditions.

Architecture
------------
MultiScorer
    Wraps SingleScorer.  Pools all conditions for embedding, then scores
    each condition separately using cell masks on the shared embedding.

Tier 1 — Core multi-condition API
    score_all_conditions()          : dict[condition -> CommitmentScoreResult]

Tier 2 — Omnibus + post-hoc statistical comparison
    compare_omnibus()               : Kruskal-Wallis / ANOVA per fate
    compare_posthoc()               : Dunn / Tukey / Conover pairwise post-hoc
    compute_pairwise_deltas()       : ΔCS for ALL condition pairs with bootstrap CI

Tier 3 — Advanced
    fit_mixed_model()               : linear mixed-effects model on per-cell
                                      fate affinity scores via statsmodels MixedLM
    fit_mixed_model_contrasts()     : LMM with custom condition contrasts
    trajectory_shift()              : KS test + Wasserstein distance on
                                      pseudotime distributions per fate arm
    plot_trajectory_shift()         : visualization of pseudotime distributions

Usage
-----
>>> mscorer = scCS.MultiScorer(
...     adata,
...     root='17',
...     branches=['homeostatic', 'activated'],
...     condition_obs_key='treatment',
...     obs_key='leiden',
... )
>>> mscorer.build_embedding(ordering_metric='pseudotime')
>>> mscorer.fit()
>>> results = mscorer.score_all_conditions()
>>> omnibus = mscorer.compare_omnibus(results)
>>> posthoc = mscorer.compare_posthoc(results, omnibus_results=omnibus)
>>> deltas = mscorer.compute_pairwise_deltas()
>>> shift = mscorer.trajectory_shift(results)
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.figure
import numpy as np
import pandas as pd

from .single import SingleScorer
from .scores import (
    CommitmentScoreResult,
    compute_cell_scores,
    compute_magnitudes,
    compute_angles,
    bin_angles,
    compute_sector_magnitudes,
    compute_pairwise_cs_matrix,
    centroid_sectors,
    equal_sectors,
)
from .plot import _fate_colors, _condition_colors, CONDITION_PALETTE


# ---------------------------------------------------------------------------
# MultiScorer
# ---------------------------------------------------------------------------

class MultiScorer:
    """RNA velocity commitment scorer for experiments with 3+ conditions.

    Builds a SHARED star embedding on the pooled data from all conditions,
    then scores each condition separately.  This ensures arm geometry is
    identical across conditions, making CS values directly comparable.

    Provides tiered statistical testing:
    - Tier 2: Omnibus tests (Kruskal-Wallis / ANOVA) followed by
      post-hoc pairwise comparisons (Dunn / Tukey / Conover).
    - Tier 3: Mixed-effects models with custom contrasts,
      trajectory shift analysis.

    Parameters
    ----------
    adata : AnnData
        Full single-cell dataset containing all conditions.
    root : str
        Label of the progenitor/root cluster in adata.obs[obs_key].
    branches : list of str
        Labels of the k terminal fate clusters.
    condition_obs_key : str
        Column in adata.obs with condition labels (e.g., 'treatment').
        Must contain at least 3 unique values.
    obs_key : str
        Column in adata.obs with cluster labels.  Default: 'leiden'.
    n_angle_bins : int
        Number of angular bins.  Default: 36.
    sector_method : {'centroid', 'equal'}
        Sector definition strategy.
    copy : bool
        Work on a copy of adata.

    Raises
    ------
    ValueError
        If condition_obs_key has fewer than 3 unique values.
        For 2 conditions, use PairScorer instead.

    Examples
    --------
    >>> mscorer = MultiScorer(
    ...     adata,
    ...     root='17',
    ...     branches=['homeostatic', 'activated'],
    ...     condition_obs_key='treatment',
    ...     obs_key='leiden',
    ... )
    >>> mscorer.build_embedding(ordering_metric='pseudotime')
    >>> mscorer.fit()
    >>> results = mscorer.score_all_conditions()
    >>> omnibus = mscorer.compare_omnibus(results)
    >>> posthoc = mscorer.compare_posthoc(results, omnibus_results=omnibus)
    """

    def __init__(
        self,
        adata,
        root: str,
        branches: List[str],
        condition_obs_key: str,
        obs_key: str = "leiden",
        n_angle_bins: int = 36,
        sector_method: Literal["centroid", "equal"] = "centroid",
        copy: bool = False,
    ):
        self.adata = adata.copy() if copy else adata
        self.root = str(root)
        self.branches = list(branches)
        self.condition_obs_key = condition_obs_key
        self.obs_key = obs_key
        self.n_angle_bins = n_angle_bins
        self.sector_method = sector_method

        # Validate condition key
        if condition_obs_key not in adata.obs:
            raise ValueError(
                f"condition_obs_key='{condition_obs_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        self.conditions = sorted(adata.obs[condition_obs_key].astype(str).unique().tolist())

        # Validate at least 3 conditions
        if len(self.conditions) < 3:
            raise ValueError(
                f"MultiScorer requires at least 3 conditions, but "
                f"condition_obs_key='{condition_obs_key}' has {len(self.conditions)} "
                f"unique value(s): {self.conditions}. "
                + ("Use SingleScorer for single-condition analysis. " if len(self.conditions) == 1
                   else "Use PairScorer for 2-condition analysis. ")
            )

        # Internal SingleScorer built on pooled data
        self._scorer: Optional[SingleScorer] = None
        self._fitted = False

        print(
            f"[scCS] MultiScorer initialized.\n"
            f"       Conditions ({len(self.conditions)}): {self.conditions}\n"
            f"       Root: '{root}', "
            f"Branches: {branches}"
        )

    # ------------------------------------------------------------------
    # Step 1: Build shared embedding (delegates to SingleScorer)
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
    ) -> "MultiScorer":
        """Build the shared star embedding on pooled data from all conditions.

        The embedding is built on ALL cells (all conditions pooled), ensuring
        that arm geometry is identical across conditions.

        Parameters
        ----------
        ordering_metric : str or np.ndarray
            See SingleScorer.build_embedding().
        invert_ordering : bool
        scale_ordering : bool
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

        self._scorer = SingleScorer(
            self.adata,
            root=self.root,
            branches=self.branches,
            obs_key=self.obs_key,
            n_angle_bins=self.n_angle_bins,
            sector_method=self.sector_method,
            copy=False,
        )
        self._scorer.build_embedding(
            ordering_metric=ordering_metric,
            invert_ordering=invert_ordering,
            scale_ordering=scale_ordering,
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
            arm_norm=arm_norm,
            verbose=verbose,
        )
        return self

    def refit_pseudotime(
        self,
        scale_01: bool = True,
        arm_scale: float = 10.0,
        jitter: float = 0.3,
        seed: int = 42,
        arm_norm: str = "global",
        verbose: bool = True,
    ) -> "MultiScorer":
        """Rebuild the shared embedding using subset-local pseudotime.

        See SingleScorer.refit_pseudotime().
        """
        self._check_embedding()
        self._scorer.refit_pseudotime(
            scale_01=scale_01,
            arm_scale=arm_scale,
            jitter=jitter,
            seed=seed,
            arm_norm=arm_norm,
            verbose=verbose,
        )
        self._fitted = False
        return self

    # ------------------------------------------------------------------
    # Step 2: Fit (delegates to SingleScorer)
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> "MultiScorer":
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
        cell_level: bool = True,
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
        cell_level : bool
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
            mask = (
                self._scorer.adata_sub.obs[self.condition_obs_key].astype(str) == cond
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
                cell_level=cell_level,
                k_nn=k_nn,
                n_bootstrap=n_bootstrap,
                bootstrap_ci=bootstrap_ci,
                verbose=verbose,
                write_to_obs=False,
            )

        return results

    # ------------------------------------------------------------------
    # Tier 2: Omnibus + post-hoc statistical comparison
    # ------------------------------------------------------------------

    def _per_condition_cell_scores(
        self,
        results: Dict[str, "CommitmentScoreResult"],
        j: int,
    ) -> Dict[str, np.ndarray]:
        """Return cell_scores[:, j] sliced to each condition's cells.

        SingleScorer.score returns cell_scores sized to the FULL shared
        adata_sub embedding (so per-cell affinities reflect every cell's
        velocity vector). For per-condition statistical tests we want
        each group to contain only that condition's cells.

        Falls back to the raw column when the array is already
        condition-sized (length differs from adata_sub.n_obs).
        """
        _adata_sub = self._scorer.adata_sub
        groups: Dict[str, np.ndarray] = {}
        for c, res in results.items():
            cs = res.cell_scores
            if cs is None:
                groups[c] = np.empty(0)
                continue
            if cs.shape[0] == _adata_sub.n_obs:
                m = (
                    _adata_sub.obs[self.condition_obs_key].astype(str) == c
                ).values
                groups[c] = cs[m, j]
            else:
                groups[c] = cs[:, j]
        return groups

    def compare_omnibus(
        self,
        results: Dict[str, CommitmentScoreResult],
        test: Literal["kruskal", "anova"] = "kruskal",
        pval_threshold: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Omnibus test across all conditions per fate.

        For each fate arm, tests whether per-cell affinity scores differ
        across ALL conditions simultaneously.

        - 'kruskal': Kruskal-Wallis H test (non-parametric, recommended default)
        - 'anova': One-way ANOVA (parametric, assumes normality)

        Parameters
        ----------
        results : dict
            Output of score_all_conditions() with cell_level=True.
        test : {'kruskal', 'anova'}
            Statistical test to use.  Default: 'kruskal'.
        pval_threshold : float
            Significance threshold for flagging.  Default 0.05.
        verbose : bool

        Returns
        -------
        pd.DataFrame with columns:
            fate, test, statistic, pval, pval_adj, significant, n_conditions
        """
        self._check_fitted()

        # Validate that cell_scores are available
        for cond, res in results.items():
            if res.cell_scores is None:
                raise ValueError(
                    f"cell_scores not available for condition '{cond}'. "
                    "Re-run score_all_conditions(cell_level=True)."
                )

        fate_names = list(results.values())[0].fate_names
        cond_list = list(results.keys())
        n_cond = len(cond_list)

        rows = []

        for j, fate in enumerate(fate_names):
            cond_groups = self._per_condition_cell_scores(results, j)
            groups = [cond_groups[c] for c in cond_list]

            if test == "kruskal":
                from scipy.stats import kruskal
                try:
                    stat, pval = kruskal(*groups)
                except Exception:
                    stat, pval = np.nan, np.nan
                test_name = "kruskal-wallis"
            else:  # anova
                from scipy.stats import f_oneway
                try:
                    stat, pval = f_oneway(*groups)
                except Exception:
                    stat, pval = np.nan, np.nan
                test_name = "one-way-anova"

            rows.append({
                "fate": fate,
                "test": test_name,
                "statistic": float(stat),
                "pval": float(pval),
                "n_conditions": n_cond,
            })

        df = pd.DataFrame(rows)

        # Multiple testing correction across fates (Bonferroni)
        df["pval_adj"] = np.minimum(df["pval"] * len(fate_names), 1.0)
        df["significant"] = df["pval_adj"] < pval_threshold

        if verbose:
            print(f"\n=== Omnibus test ({test}) across {n_cond} conditions ===")
            sig = df[df["significant"]]
            print(f"  Significant fates: {len(sig)} / {len(df)}")
            if len(sig) > 0:
                print(sig[["fate", "test", "statistic", "pval", "pval_adj", "significant"]].to_string(index=False))
            else:
                print(f"  No significant differences at pval_adj < {pval_threshold}.")

        return df

    def compare_posthoc(
        self,
        results: Dict[str, CommitmentScoreResult],
        omnibus_results: Optional[pd.DataFrame] = None,
        method: Literal["dunn", "tukey", "conover"] = "dunn",
        pval_correction: Literal["fdr", "bonferroni", "holm"] = "fdr",
        pval_threshold: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Post-hoc pairwise comparisons across conditions per fate.

        Only meaningful after an omnibus test rejects H0. If omnibus_results
        is provided, post-hoc is only run for fates where omnibus p < threshold.

        Methods
        -------
        - 'dunn': Dunn's test with rank-based comparisons (non-parametric,
          recommended with Kruskal-Wallis).  Uses scikit-posthocs.
        - 'tukey': Tukey HSD (parametric, for balanced designs, with ANOVA).
        - 'conover': Conover-Iman test (more powerful than Dunn, non-parametric).
          Uses scikit-posthocs.

        Multiple testing correction applied across all pairwise comparisons
        within each fate arm.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions() with cell_level=True.
        omnibus_results : pd.DataFrame, optional
            Output of compare_omnibus().  If provided, post-hoc is only run
            for fates where omnibus pval_adj < pval_threshold.
        method : {'dunn', 'tukey', 'conover'}
            Post-hoc test method.  Default: 'dunn'.
        pval_correction : {'fdr', 'bonferroni', 'holm'}
            Multiple testing correction method.  Default: 'fdr'.
        pval_threshold : float
            Significance threshold.  Default 0.05.
        verbose : bool

        Returns
        -------
        pd.DataFrame with columns:
            fate, comparison, method, statistic, pval, pval_adj, significant,
            mean_A, mean_B, delta_mean
        """
        self._check_fitted()

        # Validate that cell_scores are available
        for cond, res in results.items():
            if res.cell_scores is None:
                raise ValueError(
                    f"cell_scores not available for condition '{cond}'. "
                    "Re-run score_all_conditions(cell_level=True)."
                )

        fate_names = list(results.values())[0].fate_names
        cond_list = list(results.keys())

        # Filter to significant fates if omnibus provided
        if omnibus_results is not None:
            sig_fates = set(
                omnibus_results[omnibus_results["significant"]]["fate"].tolist()
            )
            if not sig_fates:
                if verbose:
                    print("[scCS] No significant fates from omnibus test. Skipping post-hoc.")
                return pd.DataFrame()
            fate_names = [f for f in fate_names if f in sig_fates]

        rows = []

        for j, fate in enumerate(fate_names):
            # Per-condition cell_scores (sliced from shared embedding if needed)
            cond_groups = self._per_condition_cell_scores(results, j)

            # Build group labels and values for this fate
            group_labels = []
            group_values = []
            for cond in cond_list:
                vals = cond_groups[cond]
                group_labels.extend([cond] * len(vals))
                group_values.extend(vals.tolist())

            group_labels = np.array(group_labels)
            group_values = np.array(group_values)

            if method == "dunn":
                try:
                    import scikit_posthocs as sp
                    # Dunn's test returns a matrix of p-values
                    dunn_p = sp.posthoc_dunn(
                        [cond_groups[c] for c in cond_list],
                        p_adjust=None,  # we do our own correction
                    )
                    # dunn_p is a DataFrame with integer indices
                    for i_a, ca in enumerate(cond_list):
                        for i_b, cb in enumerate(cond_list):
                            if i_a >= i_b:
                                continue
                            pval = dunn_p.iloc[i_a, i_b]
                            mean_a = cond_groups[ca].mean()
                            mean_b = cond_groups[cb].mean()
                            rows.append({
                                "fate": fate,
                                "comparison": f"{ca} vs {cb}",
                                "method": "dunn",
                                "statistic": np.nan,  # Dunn doesn't report per-pair stat
                                "pval": float(pval),
                                "mean_A": float(mean_a),
                                "mean_B": float(mean_b),
                                "delta_mean": float(mean_a - mean_b),
                            })
                except ImportError:
                    raise ImportError(
                        "scikit-posthocs is required for Dunn's test. "
                        "Install it with: pip install scikit-posthocs"
                    )

            elif method == "conover":
                try:
                    import scikit_posthocs as sp
                    conover_p = sp.posthoc_conover(
                        [cond_groups[c] for c in cond_list],
                        p_adjust=None,
                    )
                    for i_a, ca in enumerate(cond_list):
                        for i_b, cb in enumerate(cond_list):
                            if i_a >= i_b:
                                continue
                            pval = conover_p.iloc[i_a, i_b]
                            mean_a = cond_groups[ca].mean()
                            mean_b = cond_groups[cb].mean()
                            rows.append({
                                "fate": fate,
                                "comparison": f"{ca} vs {cb}",
                                "method": "conover",
                                "statistic": np.nan,
                                "pval": float(pval),
                                "mean_A": float(mean_a),
                                "mean_B": float(mean_b),
                                "delta_mean": float(mean_a - mean_b),
                            })
                except ImportError:
                    raise ImportError(
                        "scikit-posthocs is required for Conover-Iman test. "
                        "Install it with: pip install scikit-posthocs"
                    )

            elif method == "tukey":
                from scipy.stats import tukey_hsd
                groups = [cond_groups[c] for c in cond_list]
                try:
                    res = tukey_hsd(*groups)
                    # res.pvalue is an (n_groups, n_groups) array
                    for i_a, ca in enumerate(cond_list):
                        for i_b, cb in enumerate(cond_list):
                            if i_a >= i_b:
                                continue
                            pval = res.pvalue[i_a, i_b]
                            mean_a = cond_groups[ca].mean()
                            mean_b = cond_groups[cb].mean()
                            rows.append({
                                "fate": fate,
                                "comparison": f"{ca} vs {cb}",
                                "method": "tukey-hsd",
                                "statistic": float(res.statistic[i_a, i_b]),
                                "pval": float(pval),
                                "mean_A": float(mean_a),
                                "mean_B": float(mean_b),
                                "delta_mean": float(mean_a - mean_b),
                            })
                except Exception as e:
                    warnings.warn(
                        f"Tukey HSD failed for fate '{fate}': {e}",
                        stacklevel=2,
                    )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Multiple testing correction
        if pval_correction == "fdr":
            # Benjamini-Hochberg FDR
            pvals = df["pval"].values.copy()
            n = len(pvals)
            sorted_idx = np.argsort(pvals)
            sorted_pvals = pvals[sorted_idx]
            fdr = np.minimum(sorted_pvals * n / np.arange(1, n + 1), 1.0)
            # Enforce monotonicity (from right to left)
            for i in range(n - 2, -1, -1):
                fdr[i] = min(fdr[i], fdr[i + 1])
            # Map back to original order
            pval_adj = np.empty(n)
            pval_adj[sorted_idx] = fdr
            df["pval_adj"] = pval_adj
        elif pval_correction == "bonferroni":
            df["pval_adj"] = np.minimum(df["pval"] * len(df), 1.0)
        elif pval_correction == "holm":
            # Holm-Bonferroni step-down
            pvals = df["pval"].values.copy()
            n = len(pvals)
            sorted_idx = np.argsort(pvals)
            sorted_pvals = pvals[sorted_idx]
            holm = np.minimum(sorted_pvals * np.arange(n, 0, -1), 1.0)
            # Enforce monotonicity
            for i in range(1, n):
                holm[i] = max(holm[i], holm[i - 1])
            pval_adj = np.empty(n)
            pval_adj[sorted_idx] = holm
            df["pval_adj"] = pval_adj

        df["significant"] = df["pval_adj"] < pval_threshold

        if verbose:
            print(f"\n=== Post-hoc comparison ({method}, correction={pval_correction}) ===")
            sig = df[df["significant"]]
            print(f"  Significant pairs: {len(sig)} / {len(df)}")
            if len(sig) > 0:
                print(sig[["fate", "comparison", "method", "pval", "pval_adj", "delta_mean"]].to_string(index=False))
            else:
                print(f"  No significant pairs at pval_adj < {pval_threshold}.")

        return df

    def compute_pairwise_deltas(
        self,
        n_bootstrap: int = 500,
        ci: float = 0.95,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[Tuple[str, str], Dict]:
        """Compute ΔCS for ALL condition pairs with bootstrap CI.

        Unlike PairScorer.compute_delta_CS() which takes two specific conditions,
        this computes delta for every pair in the condition set.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap replicates.  Default 500.
        ci : float
            Confidence interval level.  Default 0.95.
        seed : int
        verbose : bool

        Returns
        -------
        dict mapping (cond_a, cond_b) -> delta_result dict
            (same structure as PairScorer.compute_delta_CS() output).
        """
        self._check_fitted()

        fate_map = self._scorer._fate_map
        vx = self._scorer._vx
        vy = self._scorer._vy

        # Sector definition (shared)
        if self.sector_method == "centroid":
            sectors, _ = centroid_sectors(
                fate_map.fate_centroids,
                fate_map.root_centroid,
                n_bins=self.n_angle_bins,
            )
        else:
            sectors = equal_sectors(fate_map.k, n_bins=self.n_angle_bins)

        def _score_mask(mask, n_cells_per_fate):
            vx_m, vy_m = vx[mask], vy[mask]
            mag = compute_magnitudes(vx_m, vy_m)
            ang = compute_angles(vx_m, vy_m)
            _, M_bin = bin_angles(ang, mag, n_bins=self.n_angle_bins)
            M_sec = compute_sector_magnitudes(M_bin, sectors)
            return compute_pairwise_cs_matrix(
                M_sec, n_cells_per_fate=n_cells_per_fate, normalized=True
            )

        all_deltas = {}
        rng = np.random.default_rng(seed)
        alpha = (1.0 - ci) / 2.0
        k = fate_map.k

        for ca, cb in combinations(self.conditions, 2):
            mask_a = (
                self._scorer.adata_sub.obs[self.condition_obs_key].astype(str) == ca
            ).values
            mask_b = (
                self._scorer.adata_sub.obs[self.condition_obs_key].astype(str) == cb
            ).values

            n_cells_a = np.array([
                int(mask_a[idx].sum()) for idx in fate_map.fate_cell_indices
            ], dtype=float)
            n_cells_b = np.array([
                int(mask_b[idx].sum()) for idx in fate_map.fate_cell_indices
            ], dtype=float)

            # Point estimates
            nCS_A = _score_mask(mask_a, n_cells_a)
            nCS_B = _score_mask(mask_b, n_cells_b)
            delta = nCS_A - nCS_B

            # Bootstrap
            idx_a = np.where(mask_a)[0]
            idx_b = np.where(mask_b)[0]
            boot_deltas = np.zeros((n_bootstrap, k, k))

            for b in range(n_bootstrap):
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

            all_deltas[(ca, cb)] = {
                "delta_nCS":   delta,
                "ci_low":      np.nanpercentile(boot_deltas, alpha * 100, axis=0),
                "ci_high":     np.nanpercentile(boot_deltas, (1 - alpha) * 100, axis=0),
                "nCS_A":       nCS_A,
                "nCS_B":       nCS_B,
                "fate_names":  fate_map.fate_names,
                "condition_a": ca,
                "condition_b": cb,
                "n_bootstrap": n_bootstrap,
                "ci_level":    ci,
            }

            if verbose:
                ci_pct = int(ci * 100)
                print(f"\n=== ΔCS: '{ca}' − '{cb}' ===")
                df_delta = pd.DataFrame(
                    delta, index=fate_map.fate_names, columns=fate_map.fate_names
                )
                print("  ΔnCS (point estimate):")
                print(df_delta.round(3).to_string())

        return all_deltas

    # ------------------------------------------------------------------
    # Tier 3: Advanced — mixed-effects model
    # ------------------------------------------------------------------

    def fit_mixed_model(
        self,
        results: Dict[str, CommitmentScoreResult],
        replicate_key: Optional[str] = None,
        ref_condition: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Linear mixed-effects model on per-cell fate affinity scores.

        Models per-cell fate affinity as a function of condition (fixed effect)
        with optional sample/replicate as a random effect.

        Model (per fate j):
            affinity_ij ~ condition_i + (1 | sample_id_i)

        Uses statsmodels MixedLM.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions(cell_level=True).
        replicate_key : str, optional
            Column in adata_sub.obs with sample/replicate IDs.
        ref_condition : str, optional
            Reference condition for the fixed effect.
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

        if ref_condition is None:
            ref_condition = sorted(conditions)[0]
        if ref_condition not in conditions:
            raise ValueError(
                f"ref_condition='{ref_condition}' not in conditions: "
                f"{conditions}"
            )

        # Build long-form DataFrame with cell-level data
        # When res.cell_scores comes from the shared embedding it covers
        # every cell — slice to this condition's cells via condition_obs_key.
        _adata_sub = self._scorer.adata_sub
        rows = []
        for cond, res in results.items():
            if res.cell_scores is None:
                continue
            obs_names = (
                np.asarray(res.cell_obs_names)
                if res.cell_obs_names is not None else np.array([])
            )
            cs = res.cell_scores
            if cs.shape[0] == _adata_sub.n_obs:
                m = (_adata_sub.obs[self.condition_obs_key].astype(str) == cond).values
                cs = cs[m]
                obs_names = np.asarray(_adata_sub.obs_names)[m]
            for i, obs_name in enumerate(obs_names):
                row = {"condition": cond, "obs_name": obs_name}
                for j, fate in enumerate(fate_names):
                    row[f"affinity_{fate}"] = cs[i, j]
                if replicate_key is not None and replicate_key in _adata_sub.obs:
                    row["sample_id"] = str(
                        _adata_sub.obs.loc[obs_name, replicate_key]
                        if obs_name in _adata_sub.obs_names
                        else "unknown"
                    )
                else:
                    row["sample_id"] = obs_name
                rows.append(row)

        df_long = pd.DataFrame(rows)

        # Set reference condition
        df_long["condition"] = pd.Categorical(
            df_long["condition"],
            categories=[ref_condition]
            + [c for c in conditions if c != ref_condition],
        )

        all_rows = []
        for fate in fate_names:
            col = f"affinity_{fate}"
            if col not in df_long.columns:
                continue

            try:
                if replicate_key is not None:
                    model = smf.mixedlm(
                        f"{col} ~ C(condition)",
                        data=df_long,
                        groups=df_long["sample_id"],
                    )
                    fit = model.fit(reml=True, method="lbfgs")
                else:
                    model = smf.ols(f"{col} ~ C(condition)", data=df_long)
                    fit = model.fit()

                for param_name in fit.params.index:
                    if "condition" not in param_name:
                        continue
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
                        "reference": ref_condition,
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

    def fit_mixed_model_contrasts(
        self,
        results: Dict[str, CommitmentScoreResult],
        contrasts: Optional[List[Tuple[str, str]]] = None,
        replicate_key: Optional[str] = None,
        ref_condition: Optional[str] = None,
        pval_threshold: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Linear mixed-effects model with custom condition contrasts.

        Extends fit_mixed_model() to test specific condition comparisons
        within the LMM framework (more powerful than separate models).

        If contrasts is None, tests each condition vs ref_condition.
        If contrasts is provided, tests each specified pair, e.g.:
            [('drug_A', 'control'), ('drug_B', 'control'), ('drug_A', 'drug_B')]

        Uses statsmodels MixedLM with Wald tests on contrast coefficients.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions(cell_level=True).
        contrasts : list of (str, str), optional
            Pairs of conditions to compare.  If None, all conditions vs
            ref_condition are tested.
        replicate_key : str, optional
            Column in adata_sub.obs with sample/replicate IDs.
        ref_condition : str, optional
            Reference condition.  Required when contrasts is None.
        pval_threshold : float
            Significance threshold.  Default 0.05.
        verbose : bool

        Returns
        -------
        pd.DataFrame with columns:
            fate, contrast, coef, std_err, z_score, pval, pval_adj, significant
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

        if ref_condition is None:
            ref_condition = sorted(conditions)[0]

        # Build contrasts list
        if contrasts is None:
            contrasts = [(c, ref_condition) for c in conditions if c != ref_condition]

        # Validate contrasts
        for ca, cb in contrasts:
            if ca not in conditions or cb not in conditions:
                raise ValueError(
                    f"Contrast ('{ca}', '{cb}') contains condition not in "
                    f"available conditions: {conditions}"
                )

        # Build long-form DataFrame
        # When res.cell_scores comes from the shared embedding it covers
        # every cell — slice to this condition's cells via condition_obs_key.
        _adata_sub = self._scorer.adata_sub
        rows = []
        for cond, res in results.items():
            if res.cell_scores is None:
                continue
            obs_names = (
                np.asarray(res.cell_obs_names)
                if res.cell_obs_names is not None else np.array([])
            )
            cs = res.cell_scores
            if cs.shape[0] == _adata_sub.n_obs:
                m = (_adata_sub.obs[self.condition_obs_key].astype(str) == cond).values
                cs = cs[m]
                obs_names = np.asarray(_adata_sub.obs_names)[m]
            for i, obs_name in enumerate(obs_names):
                row = {"condition": cond, "obs_name": obs_name}
                for j, fate in enumerate(fate_names):
                    row[f"affinity_{fate}"] = cs[i, j]
                if replicate_key is not None and replicate_key in _adata_sub.obs:
                    row["sample_id"] = str(
                        _adata_sub.obs.loc[obs_name, replicate_key]
                        if obs_name in _adata_sub.obs_names
                        else "unknown"
                    )
                else:
                    row["sample_id"] = obs_name
                rows.append(row)

        df_long = pd.DataFrame(rows)

        # Set reference condition
        df_long["condition"] = pd.Categorical(
            df_long["condition"],
            categories=[ref_condition]
            + [c for c in conditions if c != ref_condition],
        )

        all_rows = []
        for fate in fate_names:
            col = f"affinity_{fate}"
            if col not in df_long.columns:
                continue

            try:
                if replicate_key is not None:
                    model = smf.mixedlm(
                        f"{col} ~ C(condition)",
                        data=df_long,
                        groups=df_long["sample_id"],
                    )
                    fit = model.fit(reml=True, method="lbfgs")
                else:
                    model = smf.ols(f"{col} ~ C(condition)", data=df_long)
                    fit = model.fit()

                # For each contrast, compute the difference in coefficients
                for ca, cb in contrasts:
                    # Get coefficient for ca vs ref and cb vs ref
                    param_a = f"C(condition)[T.{ca}]"
                    param_b = f"C(condition)[T.{cb}]"

                    # Handle ref_condition (intercept)
                    if ca == ref_condition:
                        coef_a = 0.0
                        se_a = 0.0
                    elif param_a in fit.params.index:
                        coef_a = fit.params[param_a]
                        se_a = fit.bse[param_a]
                    else:
                        continue

                    if cb == ref_condition:
                        coef_b = 0.0
                        se_b = 0.0
                    elif param_b in fit.params.index:
                        coef_b = fit.params[param_b]
                        se_b = fit.bse[param_b]
                    else:
                        continue

                    # Contrast: coef_a - coef_b
                    contrast_coef = coef_a - coef_b
                    # Variance of difference: var(a) + var(b) - 2*cov(a,b)
                    # Approximate: assume independence (conservative)
                    contrast_se = np.sqrt(se_a**2 + se_b**2)
                    z_score = contrast_coef / contrast_se if contrast_se > 0 else 0.0

                    from scipy.stats import norm
                    pval = 2.0 * norm.sf(abs(z_score))

                    all_rows.append({
                        "fate": fate,
                        "contrast": f"{ca} vs {cb}",
                        "condition_a": ca,
                        "condition_b": cb,
                        "coef": float(contrast_coef),
                        "std_err": float(contrast_se),
                        "z_score": float(z_score),
                        "pval": float(pval),
                    })

            except Exception as e:
                warnings.warn(
                    f"Mixed model contrasts failed for fate '{fate}': {e}",
                    stacklevel=2,
                )

        if not all_rows:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_rows)
        # FDR correction across all contrasts and fates
        pvals = result_df["pval"].values.copy()
        n = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]
        fdr = np.minimum(sorted_pvals * n / np.arange(1, n + 1), 1.0)
        for i in range(n - 2, -1, -1):
            fdr[i] = min(fdr[i], fdr[i + 1])
        pval_adj = np.empty(n)
        pval_adj[sorted_idx] = fdr
        result_df["pval_adj"] = pval_adj
        result_df["significant"] = result_df["pval_adj"] < pval_threshold
        result_df = result_df.sort_values(["fate", "pval_adj"]).reset_index(drop=True)

        if verbose:
            print("\n=== Mixed-effects model contrasts ===")
            sig = result_df[result_df["significant"]]
            print(f"  Significant contrasts: {len(sig)} / {len(result_df)}")
            if len(sig) > 0:
                print(
                    sig[["fate", "contrast", "coef", "std_err", "z_score", "pval_adj"]]
                    .to_string(index=False)
                )

        return result_df

    # ------------------------------------------------------------------
    # Tier 3: Advanced — trajectory shift analysis
    # ------------------------------------------------------------------

    def trajectory_shift(
        self,
        results: Dict[str, CommitmentScoreResult],
        pseudotime_key: str = "sccs_pseudotime",
        n_bootstrap: int = 500,
        seed: int = 42,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Test whether pseudotime distributions differ across conditions per fate arm.

        For each fate arm and each pair of conditions, computes:
        - Kolmogorov-Smirnov (KS) statistic and p-value
        - Wasserstein distance (Earth Mover's Distance)
        - Bootstrap CI on the Wasserstein distance

        Parameters
        ----------
        results : dict
            Output of score_all_conditions().
        pseudotime_key : str
            Column in adata_sub.obs with pseudotime values.
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
        if pseudotime_key not in self._scorer.adata_sub.obs:
            fallback = "velocity_pseudotime"
            if fallback in self._scorer.adata_sub.obs:
                warnings.warn(
                    f"'{pseudotime_key}' not found. Using '{fallback}'. "
                    "Run compute_local_pseudotime() for better results.",
                    stacklevel=2,
                )
                pseudotime_key = fallback
            else:
                raise ValueError(
                    f"Neither '{pseudotime_key}' nor 'velocity_pseudotime' found "
                    "in adata_sub.obs. Run compute_local_pseudotime() first."
                )

        pt_all = np.array(
            self._scorer.adata_sub.obs[pseudotime_key], dtype=float
        )
        cluster_labels = self._scorer.adata_sub.obs[self.obs_key].astype(str).values
        cond_labels = self._scorer.adata_sub.obs[self.condition_obs_key].astype(str).values

        rng = np.random.default_rng(seed)

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
        pseudotime_key: str = "sccs_pseudotime",
        color_map: Optional[Dict[str, str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Visualize pseudotime distributions per condition per fate arm.

        Produces a grid of KDE plots: one row per fate arm, one column per
        pairwise comparison.  Overlaid KDEs show how pseudotime distributions
        shift between conditions.

        Parameters
        ----------
        shift_df : pd.DataFrame
            Output of trajectory_shift().
        pseudotime_key : str
        color_map : dict, optional
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

        if pseudotime_key not in self._scorer.adata_sub.obs:
            pseudotime_key = "velocity_pseudotime"

        pt_all = np.array(
            self._scorer.adata_sub.obs[pseudotime_key], dtype=float
        )
        cluster_labels = self._scorer.adata_sub.obs[self.obs_key].astype(str).values
        cond_labels = self._scorer.adata_sub.obs[self.condition_obs_key].astype(str).values

        fate_names = shift_df["fate"].unique().tolist()
        comparisons = shift_df["comparison"].unique().tolist()

        if color_map is None:
            color_map = _condition_colors(self.conditions)

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
    # New visualizations
    # ------------------------------------------------------------------

    def plot_omnibus_summary(
        self,
        omnibus_df: pd.DataFrame,
        results: Dict[str, CommitmentScoreResult],
        posthoc_df: Optional[pd.DataFrame] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> matplotlib.figure.Figure:
        """Summary heatmap: fates × conditions showing omnibus significance.

        Left panel: heatmap of mean per-cell affinity per fate per condition,
        annotated with omnibus p-value stars.
        Right panel (if posthoc provided): significant pairwise comparisons.

        Parameters
        ----------
        omnibus_df : pd.DataFrame
            Output of compare_omnibus().
        results : dict
            Output of score_all_conditions().
        posthoc_df : pd.DataFrame, optional
            Output of compare_posthoc().
        figsize : tuple, optional
        save_path : str, optional
        vmin, vmax : float, optional
            Color limits for the mean-affinity heatmap.  If both are
            ``None`` (default), they are derived from the finite values
            of the affinity matrix so the colormap spans the actual data
            range.  Set explicitly to pin a fixed scale across figures.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fate_names = omnibus_df["fate"].tolist()
        conditions = list(results.keys())

        # Build mean affinity matrix: fates × conditions
        # Slice per-condition from shared-embedding cell_scores when needed.
        _adata_sub = self._scorer.adata_sub
        mean_affinity = np.zeros((len(fate_names), len(conditions)))
        for j, fate in enumerate(fate_names):
            for i, cond in enumerate(conditions):
                if results[cond].cell_scores is None:
                    continue
                fate_idx = results[cond].fate_names.index(fate)
                cs = results[cond].cell_scores
                if cs.shape[0] == _adata_sub.n_obs:
                    m = (_adata_sub.obs[self.condition_obs_key].astype(str) == cond).values
                    mean_affinity[j, i] = cs[m, fate_idx].mean() if m.any() else np.nan
                else:
                    mean_affinity[j, i] = cs[:, fate_idx].mean()

        df_affinity = pd.DataFrame(
            mean_affinity, index=fate_names, columns=conditions
        )

        # Star annotations from omnibus p-values
        def _pval_stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            return ""

        star_annot = np.empty_like(mean_affinity, dtype=object)
        for j, fate in enumerate(fate_names):
            row = omnibus_df[omnibus_df["fate"] == fate]
            p = row["pval_adj"].values[0] if len(row) > 0 else 1.0
            stars = _pval_stars(p)
            for i in range(len(conditions)):
                star_annot[j, i] = stars

        n_panels = 2 if posthoc_df is not None and not posthoc_df.empty else 1
        if figsize is None:
            # Use a wider default so the colorbar label (which now includes
            # the auto-derived [vmin, vmax]) and fate labels do not collide.
            figsize = (5.5 * n_panels + 1.0, max(3.5, len(fate_names) * 0.55 + 2.0))

        sns.set_theme(style="ticks")
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)

        # Left panel: mean affinity heatmap
        ax = axes[0][0]
        # Auto-derive color limits from finite affinity values so the
        # colormap spans the realized data range rather than hard-pinning
        # to [0, 1] (which renders pale across most real datasets).
        finite_aff = mean_affinity[np.isfinite(mean_affinity)]
        if vmin is None:
            vmin_eff = float(finite_aff.min()) if finite_aff.size else 0.0
        else:
            vmin_eff = float(vmin)
        if vmax is None:
            vmax_eff = float(finite_aff.max()) if finite_aff.size else 1.0
        else:
            vmax_eff = float(vmax)
        if vmax_eff <= vmin_eff:
            vmax_eff = vmin_eff + 1e-9
        cbar_label = (
            f"Mean affinity [{vmin_eff:.2f}, {vmax_eff:.2f}]"
        )
        sns.heatmap(
            df_affinity, ax=ax,
            cmap="YlOrRd", vmin=vmin_eff, vmax=vmax_eff,
            annot=star_annot, fmt="",
            linewidths=0.5,
            cbar_kws={"label": cbar_label, "shrink": 0.8},
            annot_kws={"size": 12, "fontweight": "bold"},
        )
        ax.set_title("Mean affinity + omnibus significance", fontsize=10)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Fate")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # Right panel: posthoc significance grid
        if n_panels == 2:
            ax = axes[0][1]
            if posthoc_df is not None and not posthoc_df.empty:
                sig = posthoc_df[posthoc_df["significant"]]
                if not sig.empty:
                    # Build a fate × comparison matrix of significance
                    all_comps = posthoc_df["comparison"].unique().tolist()
                    sig_matrix = np.full((len(fate_names), len(all_comps)), "")
                    for j, fate in enumerate(fate_names):
                        for i, comp in enumerate(all_comps):
                            row = sig[(sig["fate"] == fate) & (sig["comparison"] == comp)]
                            if not row.empty:
                                p = row["pval_adj"].values[0]
                                sig_matrix[j, i] = _pval_stars(p)

                    df_sig = pd.DataFrame(sig_matrix, index=fate_names, columns=all_comps)
                    sns.heatmap(
                        pd.DataFrame(0, index=fate_names, columns=all_comps),
                        ax=ax,
                        cmap="Greys", vmin=-1, vmax=1,
                        annot=df_sig, fmt="",
                        linewidths=0.5,
                        cbar=False,
                        annot_kws={"size": 12, "fontweight": "bold"},
                    )
                    ax.set_title("Post-hoc significance", fontsize=10)
                    ax.set_xlabel("Comparison")
                    ax.set_ylabel("Fate")
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    ax.tick_params(axis="x", rotation=30)
                else:
                    ax.text(0.5, 0.5, "No significant\npost-hoc pairs",
                            ha="center", va="center", fontsize=12)
                    ax.set_visible(False)
            else:
                ax.text(0.5, 0.5, "No post-hoc\nresults provided",
                        ha="center", va="center", fontsize=12)
                ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_posthoc_heatmap(
        self,
        posthoc_df: pd.DataFrame,
        fate: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Condition × condition heatmap of post-hoc p-values for a given fate.

        Lower triangle: p-values.  Upper triangle: delta mean affinity.
        Annotated with significance stars.

        Parameters
        ----------
        posthoc_df : pd.DataFrame
            Output of compare_posthoc().
        fate : str, optional
            Which fate to plot.  If None, uses the first fate with
            significant results.
        figsize : tuple, optional
        save_path : str, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if fate is None:
            sig = posthoc_df[posthoc_df["significant"]]
            if sig.empty:
                print("[scCS] No significant post-hoc results to plot.")
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.text(0.5, 0.5, "No significant results", ha="center", va="center")
                return fig
            fate = sig["fate"].values[0]

        sub = posthoc_df[posthoc_df["fate"] == fate]
        if sub.empty:
            raise ValueError(f"No post-hoc results for fate '{fate}'.")

        # Get all conditions from comparisons
        conditions = set()
        for comp in sub["comparison"]:
            parts = comp.split(" vs ")
            conditions.update(parts)
        conditions = sorted(conditions)
        n = len(conditions)
        cond_idx = {c: i for i, c in enumerate(conditions)}

        # Build matrices
        pval_matrix = np.full((n, n), np.nan)
        delta_matrix = np.full((n, n), np.nan)

        for _, row in sub.iterrows():
            parts = row["comparison"].split(" vs ")
            ca, cb = parts[0], parts[1]
            i, j = cond_idx[ca], cond_idx[cb]
            pval_matrix[i, j] = row["pval_adj"]
            pval_matrix[j, i] = row["pval_adj"]
            delta_matrix[i, j] = row["delta_mean"]
            delta_matrix[j, i] = -row["delta_mean"]

        # Build annotation
        def _pval_stars(p):
            if np.isnan(p): return ""
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            return "ns"

        annot = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                if i == j:
                    annot[i, j] = ""
                elif i > j:  # lower triangle: p-values
                    annot[i, j] = _pval_stars(pval_matrix[i, j])
                else:  # upper triangle: delta mean
                    d = delta_matrix[i, j]
                    if np.isfinite(d):
                        annot[i, j] = f"{d:+.3f}"
                    else:
                        annot[i, j] = ""

        if figsize is None:
            figsize = (max(4, n * 1.2), max(3.5, n * 1.0))

        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=figsize)

        # Use -log10(p) for color scale on lower triangle, delta for upper
        display_matrix = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i > j and not np.isnan(pval_matrix[i, j]):
                    display_matrix[i, j] = -np.log10(np.clip(pval_matrix[i, j], 1e-10, None))
                elif i < j and np.isfinite(delta_matrix[i, j]):
                    display_matrix[i, j] = delta_matrix[i, j]

        df_display = pd.DataFrame(display_matrix, index=conditions, columns=conditions)

        finite_vals = display_matrix[np.isfinite(display_matrix)]
        vmax = np.abs(finite_vals).max() if len(finite_vals) > 0 else 1.0

        sns.heatmap(
            df_display, ax=ax,
            cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            annot=annot, fmt="",
            linewidths=0.5,
            cbar_kws={"label": "-log10(p) / Δmean", "shrink": 0.8},
            annot_kws={"size": 9},
        )

        ax.set_title(f"Post-hoc comparison — '{fate}'", fontsize=11)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Condition")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_pairwise_delta_grid(
        self,
        delta_results: Dict[Tuple[str, str], Dict],
        figsize_per_panel: Tuple[float, float] = (4, 4),
        cmap: str = "RdBu_r",
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Grid of ΔCS heatmaps for all condition pairs.

        Each panel shows ΔnCS = nCS_A − nCS_B for one condition pair, with
        bootstrap CI half-width annotated below each entry. Inherits the same
        layout as :func:`scCS.plot.plot_delta_cs_heatmap` but renders all pairs
        on a single shared figure.

        Parameters
        ----------
        delta_results : dict
            Output of compute_pairwise_deltas().
        figsize_per_panel : tuple
        cmap : str
            Diverging colormap. Default 'RdBu_r'.
        save_path : str, optional

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        n_pairs = len(delta_results)
        if n_pairs == 0:
            raise ValueError("No delta results to plot.")

        ncols = min(n_pairs, 3)
        nrows = int(np.ceil(n_pairs / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )

        for idx, ((ca, cb), delta_result) in enumerate(delta_results.items()):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            delta = delta_result["delta_nCS"]
            ci_low = delta_result["ci_low"]
            ci_high = delta_result["ci_high"]
            fate_names = delta_result["fate_names"]
            cond_a = delta_result["condition_a"]
            cond_b = delta_result["condition_b"]

            k = len(fate_names)
            ci_half = (ci_high - ci_low) / 2.0

            annot = np.empty((k, k), dtype=object)
            for i in range(k):
                for j in range(k):
                    d = delta[i, j]
                    h = ci_half[i, j]
                    if np.isfinite(d) and np.isfinite(h):
                        annot[i, j] = f"{d:+.2f}" + "\n" + f"±{h:.2f}"
                    elif np.isfinite(d):
                        annot[i, j] = f"{d:+.2f}"
                    else:
                        annot[i, j] = "—"

            df = pd.DataFrame(delta, index=fate_names, columns=fate_names)
            finite_vals = delta[np.isfinite(delta)]
            vmax = float(np.abs(finite_vals).max()) if finite_vals.size > 0 else 1.0

            sns.heatmap(
                df, ax=ax,
                cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                annot=annot, fmt="",
                linewidths=0.5,
                cbar_kws={"label": "ΔnCS", "shrink": 0.7},
                annot_kws={"size": 8},
            )
            ax.set_title(f"{cond_a} − {cond_b}", fontsize=10)
            ax.set_xlabel("Reference fate (÷)", fontsize=9)
            ax.set_ylabel("Query fate (×)", fontsize=9)
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for idx in range(n_pairs, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

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

        Calls SingleScorer.transfer_labels() for each condition's result,
        writing condition-specific columns to adata.obs.

        Parameters
        ----------
        results : dict
            Output of score_all_conditions(cell_level=True).
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

    def plot_star_grid(
        self,
        results: Dict[str, CommitmentScoreResult],
        color_map: Optional[Dict[str, str]] = None,
        figsize_per_panel: Tuple[float, float] = (6, 6),
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Side-by-side star embedding plots, one per condition."""
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

    def plot_rose_grid(
        self,
        results: Dict[str, CommitmentScoreResult],
        color_map: Optional[Dict[str, str]] = None,
        figsize_per_panel: Tuple[float, float] = (5, 5),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Grid of polar rose plots — one per condition."""
        from .plot import plot_rose_grid as _plot_rose_grid
        return _plot_rose_grid(
            results,
            color_map=color_map,
            figsize_per_panel=figsize_per_panel,
            title=title,
            save_path=save_path,
        )

    def plot_affinity_distributions(
        self,
        results: Dict[str, CommitmentScoreResult],
        plot_type: Literal["violin", "box", "strip"] = "violin",
        color_map: Optional[Dict[str, str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> matplotlib.figure.Figure:
        """Violin/box plots of per-cell fate affinity scores by condition."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fate_names = list(results.values())[0].fate_names
        k_fates = len(fate_names)
        conditions = list(results.keys())

        # Build long-form affinity table. Slice cell_scores to each
        # condition's cells when results carry full-embedding arrays.
        _adata_sub = self._scorer.adata_sub
        rows = []
        for cond, res in results.items():
            if res.cell_scores is None:
                continue
            cs = res.cell_scores
            if cs.shape[0] == _adata_sub.n_obs:
                m = (_adata_sub.obs[self.condition_obs_key].astype(str) == cond).values
                cs = cs[m]
            for j, fate in enumerate(fate_names):
                for val in cs[:, j]:
                    rows.append({"condition": cond, "fate": fate, "affinity": val})
        df = pd.DataFrame(rows)

        if color_map is None:
            color_map = _condition_colors(conditions)

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
                sns.violinplot(data=sub, x="condition", y="affinity",
                               palette=palette, ax=ax, inner="box",
                               order=conditions, cut=0)
            elif plot_type == "box":
                sns.boxplot(data=sub, x="condition", y="affinity",
                            palette=palette, ax=ax, order=conditions,
                            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
            else:
                sns.stripplot(data=sub, x="condition", y="affinity",
                              palette=palette, ax=ax, order=conditions,
                              size=2, alpha=0.4, jitter=True)

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

    def plot_delta_cs_heatmap(self, delta_result: dict, **kwargs) -> matplotlib.figure.Figure:
        """Heatmap of ΔCS = nCS_A − nCS_B with CI annotation."""
        from .plot import plot_delta_cs_heatmap as _plot
        return _plot(delta_result, **kwargs)

    def plot_compare_conditions_bar(
        self,
        results: Dict[str, CommitmentScoreResult],
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Grouped bar chart of nCS per condition."""
        from .plot import plot_compare_conditions_bar as _plot
        return _plot(results, **kwargs)

    def plot_commitment_vector_radar(
        self,
        results: Dict[str, CommitmentScoreResult],
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Radar / spider chart of commitment vectors per condition."""
        from .plot import plot_commitment_vector_radar as _plot
        return _plot(results, **kwargs)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "uninitialised"
        n_cond = len(self.conditions)
        return (
            f"MultiScorer("
            f"root='{self.root}', "
            f"branches={self.branches}, "
            f"conditions={self.conditions}, "
            f"status='{status}')"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def scorer(self) -> Optional[SingleScorer]:
        """The internal SingleScorer used for embedding and scoring."""
        return self._scorer

    @property
    def adata_sub(self):
        """The embedding subset (from the internal SingleScorer)."""
        if self._scorer is not None:
            return self._scorer.adata_sub
        return None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _check_embedding(self):
        if self._scorer is None or not self._scorer._embedding_built:
            raise RuntimeError(
                "Star embedding not built. Call build_embedding() first."
            )

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "MultiScorer is not fitted. Call fit() first."
            )
