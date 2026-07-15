"""Reproducible pathway and regulon enrichment for scCS v0.8.

Local gene-set mappings and GMT files are the preferred reproducible input.
Remote Enrichr libraries remain optional and are clearly recorded as a remote,
version-unstable source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .drivers import CommitmentGeneAssociationResult

GeneSetInput = Union[Mapping[str, Sequence[str]], str, Path, Sequence[str]]


@dataclass(frozen=True)
class CommitmentEnrichmentResult:
    """Enrichment tables and provenance for one scCS gene analysis."""

    tables: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, Any]

    def top(self, target: str, n: int = 20, *, significant_only: bool = True) -> pd.DataFrame:
        if target not in self.tables:
            raise KeyError(f"Unknown target {target!r}; available: {list(self.tables)}")
        table = self.tables[target]
        if significant_only:
            table = table[table["significant"]]
        return table.head(int(n)).copy()

    def export(self, output_dir: Union[str, Path], prefix: str = "sccs_enrichment") -> list[str]:
        import json

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for target, table in self.tables.items():
            safe = _safe_name(target)
            path = output / f"{prefix}_{safe}.csv"
            table.to_csv(path, index=False)
            paths.append(str(path))
        metadata_path = output / f"{prefix}_metadata.json"
        metadata_path.write_text(json.dumps(dict(self.metadata), indent=2, default=str))
        paths.append(str(metadata_path))
        return paths


def _safe_name(value: object) -> str:
    text = str(value).strip().replace(" ", "_").replace("/", "_")
    return "".join(character for character in text if character.isalnum() or character in "_-.")


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    p = np.clip(values[valid], 0.0, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    inverse = np.empty(n, dtype=int)
    inverse[order] = np.arange(n)
    adjusted[valid] = q[inverse]
    return adjusted


def load_gmt(path: Union[str, Path]) -> Dict[str, set[str]]:
    """Load a GMT file into a term-to-gene mapping."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    gene_sets: Dict[str, set[str]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                warnings.warn(
                    f"Ignoring malformed GMT line {line_number} in {source}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            term = fields[0].strip()
            genes = {gene.strip() for gene in fields[2:] if gene.strip()}
            if term and genes:
                gene_sets[term] = genes
    if not gene_sets:
        raise ValueError(f"No usable gene sets were found in {source}.")
    return gene_sets


def _normalise_gene_sets(gene_sets: GeneSetInput):
    if isinstance(gene_sets, Mapping):
        mapping = {
            str(term): {str(gene) for gene in genes if str(gene)}
            for term, genes in gene_sets.items()
        }
        mapping = {term: genes for term, genes in mapping.items() if genes}
        if not mapping:
            raise ValueError("gene_sets mapping contains no non-empty sets.")
        return mapping, {"source_type": "local_mapping", "source": "in_memory"}
    if isinstance(gene_sets, (str, Path)):
        path = Path(gene_sets)
        if path.exists():
            return load_gmt(path), {"source_type": "local_gmt", "source": str(path.resolve())}
        return None, {"source_type": "remote_enrichr", "libraries": [str(gene_sets)]}
    libraries = [str(value) for value in gene_sets]
    if not libraries:
        raise ValueError("gene_sets sequence is empty.")
    return None, {"source_type": "remote_enrichr", "libraries": libraries}


def _association_tables(
    gene_results: Union[CommitmentGeneAssociationResult, Mapping[str, pd.DataFrame]],
) -> tuple[Mapping[str, pd.DataFrame], Mapping[str, Any]]:
    if isinstance(gene_results, CommitmentGeneAssociationResult):
        return gene_results.tables, gene_results.metadata
    return gene_results, {}


def _effect_column(table: pd.DataFrame) -> str:
    for column in ("effect", "logfoldchange", "score"):
        if column in table:
            return column
    raise ValueError("Gene table must contain 'effect', 'logfoldchange', or 'score'.")


def _pvalue_column(table: pd.DataFrame) -> Optional[str]:
    for column in ("pvalue_adj", "pval_adj", "pvals_adj"):
        if column in table:
            return column
    return None


def _query_genes(
    table: pd.DataFrame,
    *,
    direction: str,
    effect_threshold: float,
    fdr_threshold: float,
    significant_only: bool,
    max_genes: Optional[int],
) -> list[str]:
    effect_column = _effect_column(table)
    selected = table.copy()
    if significant_only:
        if "significant" in selected:
            selected = selected[selected["significant"].astype(bool)]
        else:
            p_column = _pvalue_column(selected)
            if p_column is None:
                raise ValueError("significant_only=True but no significance column is available.")
            selected = selected[selected[p_column] < fdr_threshold]
    if direction == "positive":
        selected = selected[selected[effect_column] >= effect_threshold]
        selected = selected.sort_values(effect_column, ascending=False)
    elif direction == "negative":
        selected = selected[selected[effect_column] <= -effect_threshold]
        selected = selected.sort_values(effect_column, ascending=True)
    elif direction == "both":
        selected = selected[selected[effect_column].abs() >= effect_threshold]
        selected = selected.assign(_absolute=selected[effect_column].abs()).sort_values(
            "_absolute", ascending=False
        )
    else:
        raise ValueError("direction must be 'positive', 'negative', or 'both'.")
    if max_genes is not None:
        selected = selected.head(int(max_genes))
    return pd.unique(selected["gene"].astype(str)).tolist()


def _local_ora(
    query: Sequence[str],
    gene_sets: Mapping[str, set[str]],
    background: set[str],
    *,
    min_set_size: int,
    max_set_size: Optional[int],
    fdr_threshold: float,
) -> pd.DataFrame:
    universe = set(background)
    query_set = set(query) & universe
    rows = []
    for term, raw_genes in gene_sets.items():
        set_genes = set(raw_genes) & universe
        if len(set_genes) < min_set_size:
            continue
        if max_set_size is not None and len(set_genes) > max_set_size:
            continue
        overlap = query_set & set_genes
        a = len(overlap)
        b = len(query_set - set_genes)
        c = len(set_genes - query_set)
        d = len(universe - query_set - set_genes)
        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        expected = (len(query_set) * len(set_genes) / len(universe)) if universe else np.nan
        rows.append(
            {
                "term": str(term),
                "overlap": a,
                "query_size": len(query_set),
                "gene_set_size": len(set_genes),
                "background_size": len(universe),
                "expected_overlap": expected,
                "odds_ratio": float(odds_ratio),
                "pvalue": float(pvalue),
                "genes": ";".join(sorted(overlap)),
                "gene_ratio": a / len(query_set) if query_set else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "term",
                "overlap",
                "query_size",
                "gene_set_size",
                "background_size",
                "expected_overlap",
                "odds_ratio",
                "pvalue",
                "pvalue_adj",
                "genes",
                "gene_ratio",
                "significant",
            ]
        )
    table["pvalue_adj"] = _bh_adjust(table["pvalue"].to_numpy(dtype=float))
    table["significant"] = table["pvalue_adj"] < fdr_threshold
    return table.sort_values(["pvalue_adj", "odds_ratio"], ascending=[True, False]).reset_index(
        drop=True
    )


def _remote_enrichr(
    query: Sequence[str],
    libraries: Sequence[str],
    *,
    organism: str,
    fdr_threshold: float,
) -> pd.DataFrame:
    try:
        import gseapy as gp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Remote Enrichr analysis requires gseapy. Install scCS-py[enrichment]."
        ) from exc
    warnings.warn(
        "Remote Enrichr libraries can change over time. For publication-grade "
        "reproducibility, use a local GMT file or in-memory gene-set mapping.",
        RuntimeWarning,
        stacklevel=3,
    )
    result = gp.enrichr(
        gene_list=list(query),
        gene_sets=list(libraries),
        organism=organism,
        outdir=None,
        cutoff=1.0,
    ).results.copy()
    if result.empty:
        return result
    rename = {
        "Gene_set": "gene_set",
        "Term": "term",
        "Overlap": "overlap_text",
        "P-value": "pvalue",
        "Adjusted P-value": "pvalue_adj",
        "Odds Ratio": "odds_ratio",
        "Combined Score": "combined_score",
        "Genes": "genes",
    }
    result = result.rename(columns={key: value for key, value in rename.items() if key in result})
    if "overlap_text" in result:
        split = result["overlap_text"].astype(str).str.split("/", expand=True)
        result["overlap"] = pd.to_numeric(split[0], errors="coerce")
        result["gene_set_size"] = pd.to_numeric(split[1], errors="coerce")
        result["gene_ratio"] = result["overlap"] / len(set(query))
    result["significant"] = result["pvalue_adj"] < fdr_threshold
    return result.sort_values("pvalue_adj").reset_index(drop=True)


def run_commitment_enrichment(
    gene_results: Union[CommitmentGeneAssociationResult, Mapping[str, pd.DataFrame]],
    *,
    gene_sets: GeneSetInput,
    background: Optional[Sequence[str]] = None,
    direction: str = "positive",
    effect_threshold: float = 0.0,
    fdr_threshold: float = 0.05,
    significant_only: bool = True,
    max_genes: Optional[int] = 500,
    min_query_genes: int = 5,
    min_set_size: int = 5,
    max_set_size: Optional[int] = 1000,
    organism: str = "mouse",
    verbose: bool = True,
) -> CommitmentEnrichmentResult:
    """Run local ORA or optional remote Enrichr on scCS gene tables.

    The default background is the union of genes tested in the supplied
    association tables, which is usually more appropriate than the whole
    genome.  Pass an explicit assay-specific background for final analyses.
    """
    tables, association_metadata = _association_tables(gene_results)
    local_gene_sets, source_metadata = _normalise_gene_sets(gene_sets)
    if background is None:
        background_set = {
            str(gene)
            for table in tables.values()
            if "gene" in table
            for gene in table["gene"].astype(str)
        }
        background_source = "union_of_tested_genes"
    else:
        background_set = {str(gene) for gene in background}
        background_source = "user_supplied"
    if not background_set:
        raise ValueError("Enrichment background is empty.")

    output: Dict[str, pd.DataFrame] = {}
    query_sizes: Dict[str, int] = {}
    for target, table in tables.items():
        query = _query_genes(
            table,
            direction=direction,
            effect_threshold=effect_threshold,
            fdr_threshold=fdr_threshold,
            significant_only=significant_only,
            max_genes=max_genes,
        )
        query = [gene for gene in query if gene in background_set]
        query_sizes[str(target)] = len(query)
        if len(query) < min_query_genes:
            warnings.warn(
                f"Skipping enrichment for {target!r}: only {len(query)} eligible genes; "
                f"need at least {min_query_genes}.",
                RuntimeWarning,
                stacklevel=2,
            )
            output[str(target)] = pd.DataFrame()
            continue
        if local_gene_sets is not None:
            table_result = _local_ora(
                query,
                local_gene_sets,
                background_set,
                min_set_size=min_set_size,
                max_set_size=max_set_size,
                fdr_threshold=fdr_threshold,
            )
            table_result.insert(0, "target", str(target))
            table_result.insert(1, "direction", direction)
        else:
            table_result = _remote_enrichr(
                query,
                source_metadata["libraries"],
                organism=organism,
                fdr_threshold=fdr_threshold,
            )
            if not table_result.empty:
                table_result.insert(0, "target", str(target))
                table_result.insert(1, "direction", direction)
        output[str(target)] = table_result
        if verbose:
            significant = (
                int(table_result["significant"].sum())
                if not table_result.empty and "significant" in table_result
                else 0
            )
            print(
                f"[scCS] Enrichment {target}: query_genes={len(query)}, "
                f"significant_terms={significant}"
            )

    metadata: Dict[str, Any] = {
        "scientific_scope": "overrepresentation_of_candidate_commitment_associated_genes",
        "direction": direction,
        "effect_threshold": float(effect_threshold),
        "fdr_threshold": float(fdr_threshold),
        "significant_only": bool(significant_only),
        "max_genes": max_genes,
        "background_source": background_source,
        "background_size": len(background_set),
        "query_sizes": query_sizes,
        "gene_set_source": source_metadata,
        "organism": organism,
        "association_metadata": dict(association_metadata),
    }
    return CommitmentEnrichmentResult(tables=output, metadata=metadata)


def plot_enrichment_dotplot(
    result: CommitmentEnrichmentResult,
    target: str,
    *,
    n_terms: int = 15,
    significant_only: bool = True,
    ax=None,
):
    """Plot enriched terms for one target using matplotlib only."""
    import matplotlib.pyplot as plt

    if target not in result.tables:
        raise KeyError(f"Unknown target {target!r}; available: {list(result.tables)}")
    table = result.tables[target]
    if significant_only and not table.empty and "significant" in table:
        table = table[table["significant"]]
    table = table.head(int(n_terms)).copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, max(3.0, 0.34 * max(1, len(table)) + 1.5)))
    if table.empty:
        ax.text(0.5, 0.5, "No enriched terms", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax.figure
    table = table.iloc[::-1]
    x = -np.log10(table["pvalue_adj"].clip(lower=1e-300))
    size = 40.0 + 260.0 * table["gene_ratio"].fillna(0.0).to_numpy(dtype=float)
    scatter = ax.scatter(x, np.arange(len(table)), s=size, c=table["odds_ratio"], linewidths=0.4)
    ax.set_yticks(np.arange(len(table)))
    ax.set_yticklabels(table["term"].astype(str))
    ax.set_xlabel("−log10 adjusted p-value")
    ax.set_title(f"scCS commitment enrichment: {target}")
    ax.grid(axis="x", alpha=0.25)
    ax.figure.colorbar(scatter, ax=ax, label="Odds ratio")
    return ax.figure


def export_enrichment_tables(
    enrichment_results: Union[CommitmentEnrichmentResult, Mapping[str, pd.DataFrame]],
    output_dir: Union[str, Path] = ".",
    prefix: str = "enrichment",
) -> list[str]:
    """Export enrichment tables; retained as a convenient public helper."""
    if isinstance(enrichment_results, CommitmentEnrichmentResult):
        return enrichment_results.export(output_dir, prefix=prefix)
    return CommitmentEnrichmentResult(tables=enrichment_results, metadata={}).export(
        output_dir, prefix=prefix
    )


# The established name remains as a concise alias, but v0.8 accepts generic
# commitment-gene tables rather than only terminal-vs-root DEGs.
run_enrichment_per_fate = run_commitment_enrichment


__all__ = [
    "CommitmentEnrichmentResult",
    "load_gmt",
    "run_commitment_enrichment",
    "run_enrichment_per_fate",
    "plot_enrichment_dotplot",
    "export_enrichment_tables",
]
