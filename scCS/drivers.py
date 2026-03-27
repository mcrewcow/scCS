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
    cluster_key: str,
    bifurcation_cluster: str,
    n_top: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Rank genes by mean scVelo velocity in each fate arm's cells.

    Parameters
    ----------
    adata_sub : AnnData
        Subset containing only bifurcation + terminal fate cells.
        Must have the 'velocity' layer (from scVelo).
    fate_names : list of str
        Terminal fate cluster labels.
    cluster_key : str
        Column in adata_sub.obs with cluster labels.
    bifurcation_cluster : str
        Label of the progenitor cluster (used for context only).
    n_top : int
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
    obs_labels = adata_sub.obs[cluster_key].astype(str).values
    results: Dict[str, pd.DataFrame] = {}

    # Compute progenitor mean velocity once (used as baseline for delta)
    bif_mask = obs_labels == str(bifurcation_cluster)
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

        print(f"\n── Velocity drivers: {name} (top {n_top}, sorted by delta_velocity) ──")
        print(
            df.head(n_top)[["rank", "gene", "delta_velocity", "mean_velocity"]]
            .to_string(index=False)
        )

    return results


# ---------------------------------------------------------------------------
# 2. DEG-based driver genes
# ---------------------------------------------------------------------------

def get_deg_drivers(
    adata_sub,
    fate_names: List[str],
    cluster_key: str,
    bifurcation_cluster: str,
    n_top: int = 50,
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
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
    cluster_key : str
        Column in adata_sub.obs with cluster labels.
    bifurcation_cluster : str
        Label of the progenitor cluster (reference group).
    n_top : int
        Number of top significant DEGs to print per fate.
    pval_cutoff : float
        Adjusted p-value threshold for significance.
    logfc_cutoff : float
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

    obs_labels = adata_sub.obs[cluster_key].astype(str).values
    results: Dict[str, pd.DataFrame] = {}

    for name in fate_names:
        fate_mask = obs_labels == str(name)
        bif_mask = obs_labels == str(bifurcation_cluster)
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
                f"Bifurcation cluster '{bifurcation_cluster}' has only {n_bif} cells. "
                "Skipping DEG analysis (need ≥5).",
                stacklevel=2,
            )
            continue

        # Subset to fate + progenitor only for this pairwise comparison
        adata_pair = adata_sub[sub_mask].copy()
        adata_pair.obs["_deg_group"] = [
            name if l == str(name) else "progenitor"
            for l in adata_pair.obs[cluster_key].astype(str)
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
            (df["pval_adj"] < pval_cutoff)
            & (df["logfoldchange"].abs() > logfc_cutoff)
        )
        df = df.sort_values("logfoldchange", ascending=False).reset_index(drop=True)
        results[name] = df

        n_sig = df["significant"].sum()
        n_up = ((df["logfoldchange"] > logfc_cutoff) & df["significant"]).sum()
        n_dn = ((df["logfoldchange"] < -logfc_cutoff) & df["significant"]).sum()

        print(f"\n── DEG drivers: {name} vs progenitor ──")
        print(f"   Significant: {n_sig}  (up: {n_up}, down: {n_dn})")
        sig_df = df[df["significant"]].head(n_top)
        if len(sig_df) > 0:
            print(
                sig_df[["gene", "logfoldchange", "pval_adj"]]
                .to_string(index=False)
            )
        else:
            print("   (no significant DEGs at current thresholds)")

    return results
