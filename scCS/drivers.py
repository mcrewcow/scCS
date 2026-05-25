"""
drivers.py — Driver gene identification for scCS fate arms.

Two complementary strategies:

1. Velocity-based drivers
   For each fate arm, rank genes by their mean scVelo velocity in arm cells.
   High positive velocity = gene is being actively upregulated along that fate.
   Requires the 'velocity' layer (from scVelo pipeline).

2. DEG-based drivers
   For each fate arm, run a Wilcoxon rank-sum test comparing arm cells vs
   the bifurcation (progenitor) cluster.  Returns logFC and adjusted p-value
   per gene, with a significance flag.

Both functions operate on adata_sub (the subset returned by build_star_embedding),
which contains only bifurcation + terminal fate cells.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Velocity-based driver genes
# ---------------------------------------------------------------------------

def get_velocity_drivers(
    adata_sub,
    fate_names: List[str],
    obs_key: str,
    root: str,
    n_top_genes: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Rank genes by mean scVelo velocity in each fate arm's cells.

    Parameters
    ----------
    adata_sub : AnnData
        Subset containing only bifurcation + terminal fate cells.
        Must have the 'velocity' layer (from scVelo).
    fate_names : list of str
        Terminal fate cluster labels.
    obs_key : str
        Column in adata_sub.obs with cluster labels.
    root : str
        Label of the progenitor cluster (used for context only).
    n_top_genes : int
        Number of top driver genes to print per fate.

    Returns
    -------
    dict : fate_name -> DataFrame with columns [gene, mean_velocity, rank]
        Sorted by mean_velocity descending (most upregulated first).
    """
    if "velocity" not in adata_sub.layers:
        raise ValueError(
            "'velocity' layer not found in adata_sub. "
            "Run the scVelo pipeline first (scorer.compute_velocity() or "
            "scvelo.tl.velocity())."
        )

    import scipy.sparse as sp

    V_genes = adata_sub.layers["velocity"]
    if sp.issparse(V_genes):
        V_genes = V_genes.toarray()
    V_genes = np.asarray(V_genes, dtype=float)  # (n_cells, n_genes)

    genes = adata_sub.var_names
    obs_labels = adata_sub.obs[obs_key].astype(str).values
    results: Dict[str, pd.DataFrame] = {}

    # Compute progenitor mean velocity once (used as baseline for delta)
    bif_mask = obs_labels == str(root)
    if bif_mask.sum() > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_vel_progenitor = np.nanmean(V_genes[bif_mask, :], axis=0)
    else:
        mean_vel_progenitor = np.zeros(V_genes.shape[1])

    for name in fate_names:
        mask = obs_labels == str(name)
        if mask.sum() == 0:
            warnings.warn(
                f"No cells found for fate '{name}' in adata_sub. Skipping.",
                stacklevel=2,
            )
            continue

        V_fate = V_genes[mask, :]  # (n_fate_cells, n_genes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_vel_fate = np.nanmean(V_fate, axis=0)

        # Delta velocity: fate mean minus progenitor mean.
        # This removes genes that are constitutively active in the progenitor,
        # highlighting genes specifically upregulated along this fate arm.
        delta_vel = mean_vel_fate - mean_vel_progenitor

        df = pd.DataFrame({
            "gene": genes,
            "mean_velocity": mean_vel_fate,
            "progenitor_velocity": mean_vel_progenitor,
            "delta_velocity": delta_vel,
        }).dropna(subset=["mean_velocity"])

        # Sort by delta_velocity (fate-specific upregulation)
        df = df.sort_values("delta_velocity", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        results[name] = df

        print(f"\n── Velocity drivers: {name} (top {n_top_genes}, sorted by delta_velocity) ──")
        print(
            df.head(n_top_genes)[["rank", "gene", "delta_velocity", "mean_velocity"]]
            .to_string(index=False)
        )

    return results


# ---------------------------------------------------------------------------
# 2. DEG-based driver genes
# ---------------------------------------------------------------------------

def get_deg_drivers(
    adata_sub,
    fate_names: List[str],
    obs_key: str,
    root: str,
    n_top_genes: int = 50,
    pval_threshold: float = 0.05,
    logfc_threshold: float = 0.25,
) -> Dict[str, pd.DataFrame]:
    """Find DEGs for each fate arm vs the bifurcation cluster (Wilcoxon).

    For each fate arm, compares arm cells against progenitor (bifurcation)
    cells using a Wilcoxon rank-sum test via scanpy.

    Parameters
    ----------
    adata_sub : AnnData
        Subset containing only bifurcation + terminal fate cells.
    fate_names : list of str
        Terminal fate cluster labels.
    obs_key : str
        Column in adata_sub.obs with cluster labels.
    root : str
        Label of the progenitor cluster (reference group).
    n_top_genes : int
        Number of top significant DEGs to print per fate.
    pval_threshold : float
        Adjusted p-value threshold for significance.
    logfc_threshold : float
        Minimum absolute log fold-change for significance.

    Returns
    -------
    dict : fate_name -> DataFrame with columns:
        [gene, logfoldchange, pval, pval_adj, significant]
        Sorted by logfoldchange descending.
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for DEG analysis. pip install scanpy")

    obs_labels = adata_sub.obs[obs_key].astype(str).values
    results: Dict[str, pd.DataFrame] = {}

    for name in fate_names:
        fate_mask = obs_labels == str(name)
        bif_mask = obs_labels == str(root)
        sub_mask = fate_mask | bif_mask

        n_fate = fate_mask.sum()
        n_bif = bif_mask.sum()

        if n_fate < 5:
            warnings.warn(
                f"Fate '{name}' has only {n_fate} cells. "
                "Skipping DEG analysis (need ≥5).",
                stacklevel=2,
            )
            continue
        if n_bif < 5:
            warnings.warn(
                f"Root cluster '{root}' has only {n_bif} cells. "
                "Skipping DEG analysis (need ≥5).",
                stacklevel=2,
            )
            continue

        # Subset to fate + progenitor only for this pairwise comparison
        adata_pair = adata_sub[sub_mask].copy()
        adata_pair.obs["_deg_group"] = [
            name if l == str(name) else "progenitor"
            for l in adata_pair.obs[obs_key].astype(str)
        ]

        try:
            sc.tl.rank_genes_groups(
                adata_pair,
                groupby="_deg_group",
                groups=[name],
                reference="progenitor",
                method="wilcoxon",
                key_added="rank_genes",
                pts=True,
            )
        except Exception as e:
            warnings.warn(
                f"rank_genes_groups failed for fate '{name}': {e}",
                stacklevel=2,
            )
            continue

        rg = adata_pair.uns["rank_genes"]
        df = pd.DataFrame({
            "gene": rg["names"][name],
            "logfoldchange": rg["logfoldchanges"][name],
            "pval": rg["pvals"][name],
            "pval_adj": rg["pvals_adj"][name],
        })
        df["significant"] = (
            (df["pval_adj"] < pval_threshold)
            & (df["logfoldchange"].abs() > logfc_threshold)
        )

        # Extract percent-expressed per group (available when pts=True)
        try:
            pts_dict = rg.get("pts", {})
            pts_rest_dict = rg.get("pts_rest", {})
            gene_order = list(rg["names"][name])
            if name in pts_dict:
                pts_vals = pts_dict[name]
                # pts may be a dict keyed by gene name or an array in gene order
                if hasattr(pts_vals, "__getitem__") and not isinstance(pts_vals, np.ndarray):
                    df["pct_fate"] = [float(pts_vals.get(g, np.nan)) for g in gene_order]
                else:
                    df["pct_fate"] = np.asarray(pts_vals, dtype=float)
            else:
                df["pct_fate"] = np.nan
            if "progenitor" in pts_rest_dict:
                pts_rest_vals = pts_rest_dict["progenitor"]
                if hasattr(pts_rest_vals, "__getitem__") and not isinstance(pts_rest_vals, np.ndarray):
                    df["pct_progenitor"] = [float(pts_rest_vals.get(g, np.nan)) for g in gene_order]
                else:
                    df["pct_progenitor"] = np.asarray(pts_rest_vals, dtype=float)
            else:
                df["pct_progenitor"] = np.nan
        except Exception:
            df["pct_fate"] = np.nan
            df["pct_progenitor"] = np.nan

        df = df.sort_values("logfoldchange", ascending=False).reset_index(drop=True)
        results[name] = df

        n_sig = df["significant"].sum()
        n_up = ((df["logfoldchange"] > logfc_threshold) & df["significant"]).sum()
        n_dn = ((df["logfoldchange"] < -logfc_threshold) & df["significant"]).sum()

        print(f"\n── DEG drivers: {name} vs progenitor ──")
        print(f"   Significant: {n_sig}  (up: {n_up}, down: {n_dn})")
        sig_df = df[df["significant"]].head(n_top_genes)
        if len(sig_df) > 0:
            cols = ["gene", "logfoldchange", "pval_adj"]
            if "pct_fate" in sig_df.columns and not sig_df["pct_fate"].isna().all():
                cols += ["pct_fate", "pct_progenitor"]
            print(sig_df[cols].to_string(index=False))
        else:
            print("   (no significant DEGs at current thresholds)")

    return results


# ---------------------------------------------------------------------------
# 3. Velocity-fate correlation drivers (CellRank-style)
# ---------------------------------------------------------------------------

def get_velocity_fate_drivers(
    adata_sub,
    cell_scores: np.ndarray,
    fate_names: List[str],
    obs_key: str,
    root: str,
    n_top_genes: int = 50,
    pval_threshold: float = 0.05,
    min_cells: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Identify driver genes by correlating gene velocity with fate affinity.

    For each fate arm, computes the Spearman correlation between each gene's
    velocity (from the 'velocity' layer) and the cell's fate affinity score
    (from cell_scores[:, j]).  Genes with high positive Spearman correlation
    are being upregulated specifically as cells commit to that fate — a
    stronger signal than mean velocity alone, because it filters out genes
    that are fast everywhere.

    Algorithm
    ---------
    1. For each fate j, extract velocity matrix V (n_cells × n_genes).
    2. Extract fate affinity vector a (n_cells,) = cell_scores[:, j].
    3. Compute Spearman correlation between a and each gene's velocity column.
    4. Compute FDR-corrected p-values (Benjamini-Hochberg via statsmodels).
    5. Return DataFrame sorted by spearman_r descending.

    Parameters
    ----------
    adata_sub : AnnData
        Subset containing only bifurcation + terminal fate cells.
        Must have the 'velocity' layer (from scVelo).
    cell_scores : np.ndarray, shape (n_cells, k)
        Per-cell fate affinity scores from CommitmentScoreResult.cell_scores.
    fate_names : list of str
        Terminal fate cluster labels (length k).
    obs_key : str
        Column in adata_sub.obs with cluster labels.
    root : str
        Label of the progenitor cluster.
    n_top_genes : int
        Number of top driver genes to print per fate.
    pval_threshold : float
        FDR-adjusted p-value threshold for significance.
    min_cells : int
        Minimum number of cells required to compute correlations.

    Returns
    -------
    dict : fate_name -> DataFrame with columns:
        [gene, spearman_r, pval, pval_adj, mean_velocity, delta_velocity,
         significant]
        Sorted by spearman_r descending.
    """
    if "velocity" not in adata_sub.layers:
        raise ValueError(
            "'velocity' layer not found in adata_sub. "
            "Run the scVelo pipeline first (scorer.compute_velocity() or "
            "scvelo.tl.velocity())."
        )
    if cell_scores is None:
        raise ValueError(
            "cell_scores is None. "
            "Run scorer.score(cell_level=True) first."
        )
    if cell_scores.shape[0] != adata_sub.n_obs:
        raise ValueError(
            f"cell_scores has {cell_scores.shape[0]} rows but "
            f"adata_sub has {adata_sub.n_obs} cells."
        )
    if cell_scores.shape[1] != len(fate_names):
        raise ValueError(
            f"cell_scores has {cell_scores.shape[1]} columns but "
            f"fate_names has {len(fate_names)} entries."
        )

    import scipy.sparse as sp
    from scipy.stats import spearmanr

    try:
        from statsmodels.stats.multitest import multipletests
        _has_statsmodels = True
    except ImportError:
        _has_statsmodels = False
        warnings.warn(
            "statsmodels not found. p-values will not be FDR-corrected. "
            "pip install statsmodels",
            stacklevel=2,
        )

    V_genes = adata_sub.layers["velocity"]
    if sp.issparse(V_genes):
        V_genes = V_genes.toarray()
    V_genes = np.asarray(V_genes, dtype=float)  # (n_cells, n_genes)

    genes = np.array(adata_sub.var_names)
    obs_labels = adata_sub.obs[obs_key].astype(str).values

    # Progenitor mean velocity for delta computation
    bif_mask = obs_labels == str(root)
    if bif_mask.sum() > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_vel_progenitor = np.nanmean(V_genes[bif_mask, :], axis=0)
    else:
        mean_vel_progenitor = np.zeros(V_genes.shape[1])

    results: Dict[str, pd.DataFrame] = {}

    for j, name in enumerate(fate_names):
        affinity = cell_scores[:, j]  # (n_cells,)

        # Filter to cells with non-NaN affinity
        valid = ~np.isnan(affinity)
        if valid.sum() < min_cells:
            warnings.warn(
                f"Only {valid.sum()} valid cells for fate '{name}'. "
                f"Need ≥{min_cells}. Skipping.",
                stacklevel=2,
            )
            continue

        V_sub = V_genes[valid, :]       # (n_valid, n_genes)
        a_sub = affinity[valid]          # (n_valid,)

        n_genes = V_sub.shape[1]
        rho = np.zeros(n_genes)
        pvals = np.ones(n_genes)

        # Compute Spearman r for each gene
        for g in range(n_genes):
            v_g = V_sub[:, g]
            # Skip genes with zero variance in velocity
            if np.nanstd(v_g) < 1e-10:
                continue
            try:
                r, p = spearmanr(a_sub, v_g, nan_policy="omit")
                rho[g] = float(r) if np.isfinite(r) else 0.0
                pvals[g] = float(p) if np.isfinite(p) else 1.0
            except Exception:
                pass

        # FDR correction
        if _has_statsmodels:
            _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")
        else:
            pvals_adj = pvals.copy()

        # Mean velocity in fate arm cells
        fate_mask = obs_labels == str(name)
        if fate_mask.sum() > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_vel_fate = np.nanmean(V_genes[fate_mask, :], axis=0)
        else:
            mean_vel_fate = np.nanmean(V_genes[valid, :], axis=0)

        delta_vel = mean_vel_fate - mean_vel_progenitor

        df = pd.DataFrame({
            "gene": genes,
            "spearman_r": rho,
            "pval": pvals,
            "pval_adj": pvals_adj,
            "mean_velocity": mean_vel_fate,
            "delta_velocity": delta_vel,
        })
        df["significant"] = df["pval_adj"] < pval_threshold
        df = df.sort_values("spearman_r", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        results[name] = df

        n_sig = df["significant"].sum()
        print(f"\n── Velocity-fate drivers: {name} (top {n_top_genes}, sorted by Spearman r) ──")
        print(f"   Significant (FDR < {pval_threshold}): {n_sig} / {len(df)}")
        top = df[df["significant"]].head(n_top_genes)
        if len(top) == 0:
            top = df.head(n_top_genes)
        print(
            top[["rank", "gene", "spearman_r", "pval_adj", "mean_velocity"]]
            .to_string(index=False)
        )

    return results
