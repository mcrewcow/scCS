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


def _resolve_gene_sets(
    gene_sets: List[str],
    organism: str,
) -> List[str]:
    """Resolve gene set names, falling back to fuzzy matching if a name is stale.

    Enrichr gene set names include year suffixes (e.g., ``KEGG_2021_Human``)
    that change as the database is updated.  This helper:

    1. Returns the names as-is if they appear valid (no network check).
    2. If ``gseapy.get_library_name()`` is available, checks whether each
       name exists in the current Enrichr library list.  If a name is not
       found, strips the year suffix and looks for a fuzzy match.
    3. Warns the user if a substitution was made.

    Parameters
    ----------
    gene_sets : list of str
    organism : str

    Returns
    -------
    resolved : list of str
    """
    try:
        import gseapy as gp
        available = gp.get_library_name(organism=organism)
        available_set = set(available)
    except Exception:
        # Can't reach Enrichr or gseapy not installed — return as-is
        return gene_sets

    resolved = []
    for gs in gene_sets:
        if gs in available_set:
            resolved.append(gs)
        else:
            # Strip year suffix and try fuzzy match
            import re as _re
            base = _re.sub(r"_\d{4}(_\w+)?$", "", gs)
            candidates = [a for a in available if a.startswith(base)]
            if candidates:
                best = sorted(candidates)[-1]  # pick most recent year
                warnings.warn(
                    f"Gene set '{gs}' not found in Enrichr library list. "
                    f"Using '{best}' instead. "
                    "Update _DEFAULT_GENE_SETS in enrichment.py to silence this.",
                    UserWarning,
                    stacklevel=3,
                )
                resolved.append(best)
            else:
                warnings.warn(
                    f"Gene set '{gs}' not found in Enrichr library list and no "
                    f"fuzzy match found for base '{base}'. Keeping original name.",
                    UserWarning,
                    stacklevel=3,
                )
                resolved.append(gs)

    return resolved


def run_enrichment_per_fate(
    deg_drivers: Dict[str, pd.DataFrame],
    fate_names: Optional[List[str]] = None,
    gene_sets: Optional[List[str]] = None,
    organism: str = "mouse",
    pval_threshold: float = 0.05,
    logfc_threshold: float = 0.25,
    plot: bool = True,
    n_top_pathways: int = 15,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run Enrichr ORA on DEG driver genes for each fate arm.

    Runs separately for up-regulated and down-regulated genes.
    Requires gseapy >= 1.0.

    Parameters
    ----------
    deg_drivers : dict
        Output of get_deg_drivers().
        fate_name -> DataFrame[gene, logfoldchange, pval, pval_adj, significant]
    fate_names : list of str, optional
        Terminal fate cluster labels (determines iteration order).  If
        omitted (default ``None``), the fate names are inferred from
        ``deg_drivers.keys()`` in their natural insertion order.  If
        provided but missing entries that appear in ``deg_drivers``, a
        warning is emitted and only the intersection is used.
    gene_sets : list of str, optional
        Enrichr gene set library names.  Defaults to KEGG + GO BP + Reactome
        for the specified organism.
    organism : str
        'mouse' or 'human'.  Used for default gene sets and Enrichr organism.
    pval_threshold : float
        Adjusted p-value threshold for reporting enriched terms.
    logfc_threshold : float
        Minimum absolute logFC used to split up/down gene lists.
    plot : bool
        If True, generate dot plots per fate per direction.
    n_top_pathways : int
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

    # Resolve gene set names — substitutes stale year-suffixed names if needed
    gene_sets = _resolve_gene_sets(gene_sets, organism)

    # Default / validate fate_names against deg_drivers keys
    deg_keys = list(deg_drivers.keys())
    if fate_names is None:
        fate_names = deg_keys
    else:
        provided = list(fate_names)
        missing = [n for n in provided if n not in deg_drivers]
        extra_in_dict = [n for n in deg_keys if n not in provided]
        if missing or extra_in_dict:
            warnings.warn(
                "run_enrichment_per_fate: fate_names and deg_drivers.keys() "
                f"do not match.  Missing from deg_drivers: {missing}.  "
                f"In deg_drivers but not in fate_names: {extra_in_dict}.  "
                "Using only fates present in both.",
                UserWarning,
                stacklevel=2,
            )
        fate_names = [n for n in provided if n in deg_drivers]

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

        up_genes = sig[sig["logfoldchange"] > logfc_threshold]["gene"].tolist()
        down_genes = sig[sig["logfoldchange"] < -logfc_threshold]["gene"].tolist()

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
                    cutoff=pval_threshold,
                )
                res = enr.results.copy()
                res = res[res["Adjusted P-value"] < pval_threshold].copy()
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
            _plot_enrichment_dotplot(name, fate_results, n_top_pathways=n_top_pathways)

    return enrichment_results


# ---------------------------------------------------------------------------
# Internal: dot plot
# ---------------------------------------------------------------------------

def _plot_enrichment_dotplot(
    fate_name: str,
    fate_results: Dict[str, pd.DataFrame],
    n_top_pathways: int = 15,
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

        plot_df = res.head(n_top_pathways).copy()
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
