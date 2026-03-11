"""
enrichment.py — Pathway enrichment analysis for scCS fate arms.

Runs Enrichr ORA (over-representation analysis) on DEG driver genes
for each fate arm, separately for up- and down-regulated genes.

Default gene sets (mouse):
  - KEGG_2019_Mouse
  - GO_Biological_Process_2021
  - Reactome_2022

Requires gseapy >= 1.0.  Install with: pip install gseapy

Results are returned as DataFrames and optionally visualized as dot plots
(dot size = gene ratio, color = -log10 adjusted p-value).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Default gene sets per organism
_DEFAULT_GENE_SETS = {
    "mouse": [
        "KEGG_2019_Mouse",
        "GO_Biological_Process_2021",
        "Reactome_2022",
    ],
    "human": [
        "KEGG_2021_Human",
        "GO_Biological_Process_2021",
        "Reactome_2022",
    ],
}


def run_enrichment_per_fate(
    deg_drivers: Dict[str, pd.DataFrame],
    fate_names: List[str],
    gene_sets: Optional[List[str]] = None,
    organism: str = "mouse",
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
    plot: bool = True,
    n_top_terms: int = 15,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run Enrichr ORA on DEG driver genes for each fate arm.

    Runs separately for up-regulated and down-regulated genes.
    Requires gseapy >= 1.0.

    Parameters
    ----------
    deg_drivers : dict
        Output of get_deg_drivers().
        fate_name -> DataFrame[gene, logfoldchange, pval, pval_adj, significant]
    fate_names : list of str
        Terminal fate cluster labels (determines iteration order).
    gene_sets : list of str, optional
        Enrichr gene set library names.  Defaults to KEGG + GO BP + Reactome
        for the specified organism.
    organism : str
        'mouse' or 'human'.  Used for default gene sets and Enrichr organism.
    pval_cutoff : float
        Adjusted p-value threshold for reporting enriched terms.
    logfc_cutoff : float
        Minimum absolute logFC used to split up/down gene lists.
    plot : bool
        If True, generate dot plots per fate per direction.
    n_top_terms : int
        Number of top enriched terms to show in dot plots.

    Returns
    -------
    dict : fate_name -> {'up': DataFrame, 'down': DataFrame}
        Each DataFrame has columns:
        [Gene_set, Term, Overlap, P-value, Adjusted P-value, Genes]
        Sorted by Adjusted P-value ascending.
        Empty DataFrame if no significant terms found.
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError(
            "gseapy is required for pathway enrichment. "
            "Install with: pip install gseapy"
        )

    if gene_sets is None:
        org_key = organism.lower()
        if org_key not in _DEFAULT_GENE_SETS:
            raise ValueError(
                f"Unknown organism '{organism}'. "
                f"Supported: {list(_DEFAULT_GENE_SETS.keys())}"
            )
        gene_sets = _DEFAULT_GENE_SETS[org_key]

    enrichment_results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for name in fate_names:
        if name not in deg_drivers:
            warnings.warn(
                f"Fate '{name}' not found in deg_drivers. Skipping enrichment.",
                stacklevel=2,
            )
            continue

        df = deg_drivers[name]
        sig = df[df["significant"]]

        up_genes = sig[sig["logfoldchange"] > logfc_cutoff]["gene"].tolist()
        down_genes = sig[sig["logfoldchange"] < -logfc_cutoff]["gene"].tolist()

        print(f"\n{'='*60}")
        print(f"  Pathway enrichment: {name}")
        print(f"  Gene sets: {gene_sets}")
        print(f"  Up-regulated genes  : {len(up_genes)}")
        print(f"  Down-regulated genes: {len(down_genes)}")
        print(f"{'='*60}")

        fate_results: Dict[str, pd.DataFrame] = {}

        for direction, gene_list in [("up", up_genes), ("down", down_genes)]:
            if len(gene_list) < 5:
                print(
                    f"  [{direction}] Too few genes ({len(gene_list)}), "
                    "skipping enrichment (need ≥5)."
                )
                fate_results[direction] = pd.DataFrame()
                continue

            try:
                enr = gp.enrichr(
                    gene_list=gene_list,
                    gene_sets=gene_sets,
                    organism=organism,
                    outdir=None,
                    cutoff=pval_cutoff,
                )
                res = enr.results.copy()
                res = res[res["Adjusted P-value"] < pval_cutoff].copy()
                res = res.sort_values("Adjusted P-value").reset_index(drop=True)
                fate_results[direction] = res

                n_sig = len(res)
                print(f"\n  [{direction}] Significant terms: {n_sig}")
                if n_sig > 0:
                    print(
                        res[["Gene_set", "Term", "Overlap", "Adjusted P-value"]]
                        .head(10)
                        .to_string(index=False)
                    )

            except Exception as e:
                warnings.warn(
                    f"Enrichr failed for fate '{name}' [{direction}]: {e}",
                    stacklevel=2,
                )
                fate_results[direction] = pd.DataFrame()

        enrichment_results[name] = fate_results

        if plot:
            _plot_enrichment_dotplot(name, fate_results, n_top_terms=n_top_terms)

    return enrichment_results


# ---------------------------------------------------------------------------
# Internal: dot plot
# ---------------------------------------------------------------------------

def _plot_enrichment_dotplot(
    fate_name: str,
    fate_results: Dict[str, pd.DataFrame],
    n_top_terms: int = 15,
    figsize_per_panel: tuple = (10, 5),
) -> None:
    """Draw dot plots for up- and down-regulated enrichment results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        warnings.warn("matplotlib/seaborn not available. Skipping dot plot.", stacklevel=2)
        return

    for direction in ["up", "down"]:
        res = fate_results.get(direction, pd.DataFrame())
        if res is None or res.empty:
            continue

        plot_df = res.head(n_top_terms).copy()
        plot_df["-log10(padj)"] = -np.log10(
            plot_df["Adjusted P-value"].clip(1e-300)
        )

        def _parse_ratio(s: str) -> float:
            try:
                a, b = str(s).split("/")
                return int(a) / int(b)
            except Exception:
                return 0.0

        plot_df["gene_ratio"] = plot_df["Overlap"].apply(_parse_ratio)

        # Clean up gene set name for label prefix
        plot_df = plot_df.sort_values(["Gene_set", "Adjusted P-value"])
        plot_df["label"] = (
            plot_df["Gene_set"]
            .str.replace(r"_2019_Mouse|_2021|_2022|_2021_Human", "", regex=True)
            + ": "
            + plot_df["Term"].str[:55]
        )

        fig, ax = plt.subplots(figsize=figsize_per_panel)

        sc_ = ax.scatter(
            plot_df["gene_ratio"],
            range(len(plot_df)),
            c=plot_df["-log10(padj)"],
            s=plot_df["gene_ratio"] * 2000,
            cmap="RdYlBu_r",
            vmin=0,
            edgecolors="grey",
            linewidths=0.4,
            zorder=3,
        )
        plt.colorbar(sc_, ax=ax, label="-log10(adj. p-value)", shrink=0.6)

        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["label"], fontsize=8)
        ax.set_xlabel("Gene ratio (overlap / gene set size)")
        ax.set_title(
            f"Pathway enrichment: {fate_name}  [{direction}-regulated]\n"
            f"(KEGG + GO BP + Reactome, {plot_df['Gene_set'].str.contains('Mouse').any() and 'mouse' or 'human'})",
            fontsize=11,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_enrichment_tables(
    enrichment_results: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str = ".",
    prefix: str = "enrichment",
) -> List[str]:
    """Save enrichment result DataFrames to CSV files.

    Parameters
    ----------
    enrichment_results : dict
        Output of run_enrichment_per_fate().
    output_dir : str
        Directory to save files.
    prefix : str
        Filename prefix.

    Returns
    -------
    list of str : paths of saved files.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for fate_name, fate_results in enrichment_results.items():
        safe_name = fate_name.replace(" ", "_").replace("/", "_")
        for direction, df in fate_results.items():
            if df is None or df.empty:
                continue
            fname = os.path.join(output_dir, f"{prefix}_{safe_name}_{direction}.csv")
            df.to_csv(fname, index=False)
            saved.append(fname)
            print(f"Saved: {fname}")

    return saved
