"""Candidate gene analyses for scCS v0.8.

The functions in this module identify *associations* with supervised commitment
outputs.  They do not infer causal lineage drivers.  Two analysis units are
supported:

``cell_exploratory``
    Partial Spearman association across cells.  This is useful for discovery
    and visualization, but cells are not treated as biological replicates.

``replicate``
    Partial Spearman association across replicate-level means.  This is the
    preferred inferential mode when enough independent biological replicates
    are available.

All association functions are chunked and sparse-safe.  Expression, spliced,
unspliced, or velocity layers can be supplied without densifying the complete
cell-by-gene matrix at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, t as student_t

from .pipeline import FurcationScoreResult


_FATE_OUTCOMES = {"directional_affinity", "commitment_contribution"}
_SCALAR_OUTCOMES = {
    "commitment_strength",
    "directional_entropy",
    "commitment_entropy",
    "directional_specificity",
    "specific_commitment",
    "progression_velocity",
}
_OUTCOME_ALIASES = {
    "affinity": "directional_affinity",
    "direction": "directional_affinity",
    "contribution": "commitment_contribution",
    "strength": "commitment_strength",
    "entropy": "directional_entropy",
    "directional_uncertainty": "directional_entropy",
    "commitment_uncertainty": "commitment_entropy",
    "specificity": "directional_specificity",
    "specific": "specific_commitment",
    "progression": "progression_velocity",
    # Mode-specific public names.  The underlying arrays remain shared so that
    # existing downstream tables and code stay backward compatible.
    "future_fate_affinity": "directional_affinity",
    "conditional_fate_affinity": "directional_affinity",
    "future_fate_contribution": "commitment_contribution",
    "future_fate_reach": "commitment_strength",
    "discounted_fate_reach": "commitment_strength",
    "future_fate_entropy": "directional_entropy",
    "future_fate_specificity": "directional_specificity",
    "reach_supported_specificity": "specific_commitment",
    "resolved_commitment": "specific_commitment",
    "signed_progression": "progression_velocity",
    "signed_ordering_flux": "progression_velocity",
}


@dataclass(frozen=True)
class CommitmentGeneAssociationResult:
    """Tables and provenance for one commitment–gene association analysis."""

    tables: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, Any]

    def top(
        self,
        target: str,
        n: int = 20,
        *,
        significant_only: bool = False,
        positive_only: bool = False,
    ) -> pd.DataFrame:
        """Return the top genes for one target table."""
        if target not in self.tables:
            raise KeyError(f"Unknown target {target!r}; available: {list(self.tables)}")
        table = self.tables[target]
        selected = table
        if significant_only:
            selected = selected[selected["significant"]]
        if positive_only:
            selected = selected[selected["effect"] > 0]
        return selected.head(int(n)).copy()

    def export(
        self,
        output_dir: Union[str, Path],
        prefix: str = "sccs_gene_association",
    ) -> list[str]:
        """Export association tables and metadata to CSV/JSON files."""
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
    """Benjamini–Hochberg adjustment without a statsmodels dependency."""
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


def _resolve_outcome_name(outcome: str) -> str:
    resolved = _OUTCOME_ALIASES.get(str(outcome), str(outcome))
    valid = _FATE_OUTCOMES | _SCALAR_OUTCOMES
    if resolved not in valid:
        raise ValueError(f"Unknown outcome {outcome!r}; supported: {sorted(valid)}")
    return resolved


def _resolve_population_mask(
    result: FurcationScoreResult,
    population: Union[str, Sequence[bool], np.ndarray],
) -> np.ndarray:
    if isinstance(population, str):
        if population == "root":
            return result.root_mask.copy()
        if population == "all":
            return np.ones(len(result.cell_ids), dtype=bool)
        raise ValueError("population must be 'root', 'all', or a Boolean mask.")
    mask = np.asarray(population)
    if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(result.cell_ids):
        raise ValueError("Custom population mask must be Boolean and aligned to result.cell_ids.")
    if not mask.any():
        raise ValueError("Custom population mask selects no cells.")
    return mask.copy()


def _resolve_outcomes(
    result: FurcationScoreResult,
    outcome: str,
    fate_names: Optional[Sequence[str]],
) -> Dict[str, np.ndarray]:
    name = _resolve_outcome_name(outcome)
    if name in _FATE_OUTCOMES:
        selected_fates = list(result.fate_names if fate_names is None else fate_names)
        missing = [fate for fate in selected_fates if fate not in result.fate_names]
        if missing:
            raise ValueError(f"Unknown fate names: {missing}; available: {list(result.fate_names)}")
        values = getattr(result, name)
        return {
            str(fate): np.asarray(values[:, result.fate_names.index(str(fate))], dtype=float)
            for fate in selected_fates
        }
    return {name: np.asarray(getattr(result, name), dtype=float)}


def _resolve_matrix(adata, *, layer: Optional[str], use_raw: bool):
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is not available.")
        if layer is not None:
            raise ValueError("layer and use_raw cannot both be specified.")
        return adata.raw.X, np.asarray(adata.raw.var_names).astype(str), "raw.X"
    if layer is None:
        return adata.X, np.asarray(adata.var_names).astype(str), "X"
    if layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} is missing from adata.layers.")
    return adata.layers[layer], np.asarray(adata.var_names).astype(str), f"layer:{layer}"


def _selected_adata_indices(adata, result: FurcationScoreResult) -> np.ndarray:
    indexer = pd.Index(np.asarray(adata.obs_names).astype(str)).get_indexer(result.cell_ids)
    if np.any(indexer < 0):
        missing = np.asarray(result.cell_ids)[indexer < 0][:5].tolist()
        raise ValueError(f"Result cell IDs are absent from AnnData, e.g. {missing}")
    return indexer.astype(int)


def _rank_numeric(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return rankdata(array, method="average")


def _build_cell_covariates(
    obs: pd.DataFrame,
    keys: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(obs), dtype=float)]
    names = ["intercept"]
    for key in keys:
        if key not in obs:
            raise KeyError(f"Covariate {key!r} is missing from adata.obs.")
        series = obs[key]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Numeric covariate {key!r} contains non-finite values.")
            if np.nanstd(values) <= np.finfo(float).eps:
                warnings.warn(f"Covariate {key!r} is constant and was omitted.", stacklevel=3)
                continue
            ranked = _rank_numeric(values)
            ranked = (ranked - ranked.mean()) / ranked.std(ddof=0)
            columns.append(ranked)
            names.append(key)
        else:
            categorical = series.astype(str)
            dummies = pd.get_dummies(categorical, prefix=key, drop_first=True, dtype=float)
            for dummy_name in dummies.columns:
                columns.append(dummies[dummy_name].to_numpy(dtype=float))
                names.append(str(dummy_name))
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        _, independent = np.unique(design, axis=1, return_index=True)
        keep = np.sort(independent)
        design = design[:, keep]
        names = [names[index] for index in keep]
    return design, names


def _aggregate_replicates(
    obs: pd.DataFrame,
    replicate_key: str,
    condition_key: Optional[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    if replicate_key not in obs:
        raise KeyError(f"replicate_key {replicate_key!r} is missing from adata.obs.")
    replicate = obs[replicate_key].astype(str)
    if replicate.isna().any() or (replicate == "nan").any():
        raise ValueError("replicate_key contains missing values.")
    if condition_key is not None:
        if condition_key not in obs:
            raise KeyError(f"condition_key {condition_key!r} is missing from adata.obs.")
        condition = obs[condition_key].astype(str)
        unit = condition + "::" + replicate
    else:
        condition = pd.Series("single_condition", index=obs.index, dtype=str)
        unit = replicate
    unit_values = unit.to_numpy(dtype=str)
    levels = pd.Index(pd.unique(unit_values))
    group_index = levels.get_indexer(unit_values)
    metadata_rows = []
    for level_index, level in enumerate(levels):
        mask = group_index == level_index
        conditions = pd.unique(condition.to_numpy(dtype=str)[mask])
        if len(conditions) != 1:
            raise ValueError(f"Replicate unit {level!r} spans multiple conditions.")
        metadata_rows.append(
            {
                "replicate_unit": str(level),
                "condition": str(conditions[0]),
                "n_cells": int(mask.sum()),
            }
        )
    return group_index.astype(int), pd.DataFrame(metadata_rows)


def _aggregate_vector(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.bincount(groups, weights=np.asarray(values, dtype=float), minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups)
    return sums / counts


def _aggregate_matrix(values, groups: np.ndarray, n_groups: int) -> np.ndarray:
    dense = values.toarray() if sparse.issparse(values) else np.asarray(values, dtype=float)
    output = np.empty((n_groups, dense.shape[1]), dtype=float)
    for group in range(n_groups):
        output[group] = np.mean(dense[groups == group], axis=0)
    return output


def _partial_rank_association(
    features: np.ndarray,
    outcome: np.ndarray,
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features must be two-dimensional.")
    y = _rank_numeric(outcome)
    ranked_features = rankdata(features, axis=0, method="average")

    pseudo_inverse = np.linalg.pinv(design)
    y_residual = y - design @ (pseudo_inverse @ y)
    x_residual = ranked_features - design @ (pseudo_inverse @ ranked_features)

    y_norm = float(np.sqrt(np.sum(y_residual**2)))
    x_norm = np.sqrt(np.sum(x_residual**2, axis=0))
    denominator = x_norm * y_norm
    effects = np.zeros(features.shape[1], dtype=float)
    valid = denominator > np.finfo(float).eps
    effects[valid] = (y_residual @ x_residual[:, valid]) / denominator[valid]
    effects = np.clip(effects, -1.0, 1.0)

    degrees = len(outcome) - np.linalg.matrix_rank(design) - 1
    pvalues = np.ones(features.shape[1], dtype=float)
    if degrees > 0:
        safe = valid & (np.abs(effects) < 1.0 - 1e-15)
        statistic = effects[safe] * np.sqrt(degrees / np.maximum(1e-15, 1.0 - effects[safe] ** 2))
        pvalues[safe] = 2.0 * student_t.sf(np.abs(statistic), df=degrees)
        pvalues[valid & ~safe] = 0.0
    return effects, pvalues


def _matrix_chunk(matrix, row_indices: np.ndarray, start: int, stop: int) -> np.ndarray:
    chunk = matrix[row_indices, start:stop]
    if sparse.issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk, dtype=float)


def get_commitment_associated_genes(
    adata,
    result: FurcationScoreResult,
    *,
    outcome: str = "commitment_contribution",
    fate_names: Optional[Sequence[str]] = None,
    population: Union[str, Sequence[bool], np.ndarray] = "root",
    layer: Optional[str] = None,
    use_raw: bool = False,
    inference_unit: str = "cell_exploratory",
    replicate_key: Optional[str] = None,
    condition_key: Optional[str] = None,
    pseudotime_key: Union[str, None] = "auto",
    covariates: Optional[Sequence[str]] = None,
    min_cells: int = 30,
    min_replicates: int = 6,
    min_cells_per_replicate: int = 5,
    min_feature_cells: int = 5,
    min_feature_fraction: float = 0.01,
    chunk_size: int = 512,
    fdr_threshold: float = 0.05,
    effect_threshold: float = 0.0,
    n_top_genes: int = 20,
    verbose: bool = True,
) -> CommitmentGeneAssociationResult:
    """Associate genes with scCS commitment outcomes.

    Parameters
    ----------
    adata
        AnnData containing the cells used to create ``result``.
    result
        Fitted :class:`scCS.FurcationScoreResult`.
    outcome
        One of ``directional_affinity``, ``commitment_contribution``,
        ``commitment_strength``, ``directional_entropy``,
        ``commitment_entropy``, ``directional_specificity``,
        ``specific_commitment``, or ``progression_velocity``.
    fate_names
        Fates to analyze for fate-specific outcomes.  The default analyzes all
        manually supplied terminal populations.
    population
        ``"root"`` (default), ``"all"``, or a Boolean mask aligned to
        ``result.cell_ids``.
    layer, use_raw
        Source gene matrix.  ``layer="velocity"`` tests gene velocity;
        ``layer=None`` uses ``adata.X``.  Input values are not normalized by
        this function.
    inference_unit
        ``"cell_exploratory"`` or ``"replicate"``.  Cell-level p-values are
        exploratory because cells are not independent biological replicates.
    replicate_key, condition_key
        Required/optional grouping columns for replicate-level association.
    min_cells_per_replicate
        Replicate units with fewer selected population cells are excluded before
        aggregation. The default is five cells.
    pseudotime_key
        ``"auto"`` uses the ordering key from the fitted scCS embedding.
        Set to ``None`` to omit pseudotime adjustment.
    covariates
        Additional numeric or categorical ``adata.obs`` columns.

    Returns
    -------
    CommitmentGeneAssociationResult
        One ranked table per fate or scalar outcome.
    """
    if inference_unit not in {"cell_exploratory", "replicate"}:
        raise ValueError("inference_unit must be 'cell_exploratory' or 'replicate'.")
    if min_cells < 3 or min_replicates < 3:
        raise ValueError("min_cells and min_replicates must be at least 3.")
    if min_cells_per_replicate < 1:
        raise ValueError("min_cells_per_replicate must be at least 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if not 0.0 <= min_feature_fraction <= 1.0:
        raise ValueError("min_feature_fraction must lie in [0, 1].")

    matrix, genes, matrix_source = _resolve_matrix(adata, layer=layer, use_raw=use_raw)
    selected_adata = _selected_adata_indices(adata, result)
    population_mask = _resolve_population_mask(result, population)
    outcomes = _resolve_outcomes(result, outcome, fate_names)

    selected_rows = selected_adata[population_mask]
    selected_obs = adata.obs.iloc[selected_rows].copy()
    if len(selected_rows) < min_cells:
        raise ValueError(
            f"Selected population contains {len(selected_rows)} cells; need {min_cells}."
        )
    population_outcomes = {
        target: np.asarray(values, dtype=float)[population_mask]
        for target, values in outcomes.items()
    }
    initial_selected_cells = int(len(selected_rows))
    excluded_replicate_units: list[dict[str, Any]] = []

    covariate_keys = list(covariates or [])
    resolved_pseudotime: Optional[str]
    if pseudotime_key == "auto":
        resolved_pseudotime = result.embedding.ordering_key
    else:
        resolved_pseudotime = pseudotime_key
    if resolved_pseudotime is not None and resolved_pseudotime not in covariate_keys:
        covariate_keys.insert(0, resolved_pseudotime)

    if inference_unit == "cell_exploratory":
        design, design_names = _build_cell_covariates(selected_obs, covariate_keys)
        analysis_rows = selected_rows
        unit_count = len(analysis_rows)
        replicate_metadata = None
    else:
        if replicate_key is None:
            raise ValueError("replicate_key is required for replicate-level association.")

        # Filter low-cell replicate units before any expression or outcome
        # aggregation.  This prevents unstable one- or two-cell means from
        # being treated as equally reliable biological units.
        if condition_key is not None:
            if condition_key not in selected_obs:
                raise KeyError(f"condition_key {condition_key!r} is missing from adata.obs.")
            unit_labels = (
                selected_obs[condition_key].astype(str)
                + "::"
                + selected_obs[replicate_key].astype(str)
            )
        else:
            unit_labels = selected_obs[replicate_key].astype(str)
        unit_counts = unit_labels.value_counts(sort=False)
        excluded = unit_counts[unit_counts < int(min_cells_per_replicate)]
        excluded_replicate_units = [
            {"replicate_unit": str(name), "n_cells": int(count)} for name, count in excluded.items()
        ]
        keep_cells = unit_labels.isin(
            unit_counts[unit_counts >= int(min_cells_per_replicate)].index
        ).to_numpy()
        selected_rows = selected_rows[keep_cells]
        selected_obs = selected_obs.iloc[np.flatnonzero(keep_cells)].copy()
        population_outcomes = {
            target: values[keep_cells] for target, values in population_outcomes.items()
        }
        if len(selected_rows) < min_cells:
            raise ValueError(
                "After min_cells_per_replicate filtering, the selected population "
                f"contains {len(selected_rows)} cells; need {min_cells}."
            )

        groups, replicate_metadata = _aggregate_replicates(
            selected_obs,
            replicate_key=replicate_key,
            condition_key=condition_key,
        )
        unit_count = len(replicate_metadata)
        if unit_count < min_replicates:
            raise ValueError(
                f"Only {unit_count} independent replicate units are available; "
                f"need at least {min_replicates}."
            )
        if unit_count < 10:
            warnings.warn(
                f"Only {unit_count} replicate units are available. Gene-level association "
                "may be underpowered; interpret effect sizes and confidence cautiously.",
                RuntimeWarning,
                stacklevel=2,
            )
        replicate_obs = replicate_metadata.copy()
        # Aggregate numeric cell-level covariates; condition remains categorical.
        for key in covariate_keys:
            if key not in selected_obs:
                raise KeyError(f"Covariate {key!r} is missing from adata.obs.")
            if pd.api.types.is_numeric_dtype(selected_obs[key]):
                replicate_obs[key] = _aggregate_vector(
                    selected_obs[key].to_numpy(dtype=float), groups, unit_count
                )
            else:
                values = selected_obs[key].astype(str).to_numpy()
                aggregated = []
                for group in range(unit_count):
                    unique = pd.unique(values[groups == group])
                    if len(unique) != 1:
                        raise ValueError(
                            f"Categorical covariate {key!r} varies within replicate unit "
                            f"{replicate_metadata.iloc[group]['replicate_unit']!r}."
                        )
                    aggregated.append(str(unique[0]))
                replicate_obs[key] = aggregated
        if condition_key is not None and condition_key not in covariate_keys:
            replicate_obs[condition_key] = replicate_metadata["condition"].astype(str)
            covariate_keys.append(condition_key)
        design, design_names = _build_cell_covariates(replicate_obs, covariate_keys)
        analysis_rows = selected_rows

    tested_counts = np.zeros(len(genes), dtype=int)
    feature_means = np.full(len(genes), np.nan, dtype=float)
    feature_sds = np.full(len(genes), np.nan, dtype=float)
    target_effects = {target: np.full(len(genes), np.nan, dtype=float) for target in outcomes}
    target_pvalues = {target: np.full(len(genes), np.nan, dtype=float) for target in outcomes}

    if inference_unit == "cell_exploratory":
        outcome_values = population_outcomes
    else:
        outcome_values = {
            target: _aggregate_vector(values, groups, unit_count)
            for target, values in population_outcomes.items()
        }

    for start in range(0, len(genes), chunk_size):
        stop = min(start + chunk_size, len(genes))
        chunk = _matrix_chunk(matrix, analysis_rows, start, stop)
        if not np.all(np.isfinite(chunk)):
            finite_columns = np.all(np.isfinite(chunk), axis=0)
        else:
            finite_columns = np.ones(chunk.shape[1], dtype=bool)
        detected = np.sum(np.abs(chunk) > np.finfo(float).eps, axis=0)
        minimum_detected = max(
            int(min_feature_cells),
            int(np.ceil(min_feature_fraction * len(chunk))),
        )
        variable = np.nanstd(chunk, axis=0) > np.finfo(float).eps
        eligible = finite_columns & variable & (detected >= minimum_detected)

        tested_counts[start:stop] = detected
        feature_means[start:stop] = np.nanmean(chunk, axis=0)
        feature_sds[start:stop] = np.nanstd(chunk, axis=0)
        if not eligible.any():
            continue

        if inference_unit == "replicate":
            analysis_chunk = _aggregate_matrix(chunk[:, eligible], groups, unit_count)
        else:
            analysis_chunk = chunk[:, eligible]

        for target, target_values in outcome_values.items():
            valid_units = np.isfinite(target_values)
            if valid_units.sum() <= np.linalg.matrix_rank(design[valid_units]) + 2:
                continue
            effects, pvalues = _partial_rank_association(
                analysis_chunk[valid_units],
                target_values[valid_units],
                design[valid_units],
            )
            eligible_indices = np.flatnonzero(eligible) + start
            target_effects[target][eligible_indices] = effects
            target_pvalues[target][eligible_indices] = pvalues

    tables: Dict[str, pd.DataFrame] = {}
    for target in outcomes:
        effects = target_effects[target]
        pvalues = target_pvalues[target]
        p_adjusted = _bh_adjust(pvalues)
        table = pd.DataFrame(
            {
                "gene": genes,
                "effect": effects,
                "pvalue": pvalues,
                "pvalue_adj": p_adjusted,
                "feature_mean": feature_means,
                "feature_sd": feature_sds,
                "n_detected_cells": tested_counts,
                "n_units": unit_count,
            }
        )
        table = table[np.isfinite(table["effect"])].copy()
        table["significant"] = (table["pvalue_adj"] < float(fdr_threshold)) & (
            table["effect"].abs() >= float(effect_threshold)
        )
        table["absolute_effect"] = table["effect"].abs()
        table = table.sort_values(["effect", "pvalue_adj"], ascending=[False, True]).reset_index(
            drop=True
        )
        table["rank"] = np.arange(1, len(table) + 1)
        tables[target] = table

        if verbose:
            print(
                f"\n[scCS] Candidate commitment-associated genes: {target}\n"
                f"        inference_unit={inference_unit}; matrix={matrix_source}; "
                f"n_units={unit_count}; tested_genes={len(table)}"
            )
            shown = table[table["significant"]].head(n_top_genes)
            if shown.empty:
                shown = table.head(n_top_genes)
            print(
                shown[["rank", "gene", "effect", "pvalue_adj", "significant"]].to_string(
                    index=False
                )
            )

    metadata: Dict[str, Any] = {
        "scientific_scope": "candidate_commitment_association_not_causal_driver_inference",
        "outcome": _resolve_outcome_name(outcome),
        "targets": list(tables),
        "population": population if isinstance(population, str) else "custom_mask",
        "matrix_source": matrix_source,
        "inference_unit": inference_unit,
        "association_method": "partial_spearman_rank_correlation",
        "pseudotime_key": resolved_pseudotime,
        "covariates": covariate_keys,
        "design_columns": design_names,
        "replicate_key": replicate_key,
        "condition_key": condition_key,
        "n_cells_initial": initial_selected_cells,
        "n_cells": int(len(selected_rows)),
        "n_units": int(unit_count),
        "min_cells_per_replicate": (
            int(min_cells_per_replicate) if inference_unit == "replicate" else None
        ),
        "n_excluded_replicate_units": len(excluded_replicate_units),
        "excluded_replicate_units": excluded_replicate_units,
        "fdr_threshold": float(fdr_threshold),
        "effect_threshold": float(effect_threshold),
        "min_feature_cells": int(min_feature_cells),
        "min_feature_fraction": float(min_feature_fraction),
    }
    if replicate_metadata is not None:
        metadata["replicate_units"] = replicate_metadata.to_dict(orient="records")
    return CommitmentGeneAssociationResult(tables=tables, metadata=metadata)


def get_fate_markers(
    adata,
    result: FurcationScoreResult,
    *,
    layer: Optional[str] = None,
    use_raw: bool = False,
    method: str = "wilcoxon",
    n_genes: Optional[int] = None,
    fdr_threshold: float = 0.05,
    logfc_threshold: float = 0.25,
    min_cells: int = 5,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Find annotation markers for each terminal population versus the root.

    This is a cell-identity marker analysis, not a commitment association and
    not a causal driver analysis.  For multi-sample experiments, use an
    external pseudobulk workflow for formal differential-expression inference.
    """
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scanpy is required for get_fate_markers().") from exc

    selected = _selected_adata_indices(adata, result)
    subset = adata[selected].copy()
    subset.obs["_sccs_population"] = np.where(
        result.root_mask,
        result.furcation.root_name,
        result.embedding.terminal_names.astype(str),
    )
    if layer is not None and layer not in subset.layers:
        raise KeyError(f"Layer {layer!r} is missing from adata.layers.")

    output: Dict[str, pd.DataFrame] = {}
    root_name = result.furcation.root_name
    for fate in result.fate_names:
        pair_mask = subset.obs["_sccs_population"].isin([root_name, fate]).to_numpy()
        pair = subset[pair_mask].copy()
        counts = pair.obs["_sccs_population"].value_counts()
        if counts.get(root_name, 0) < min_cells or counts.get(fate, 0) < min_cells:
            warnings.warn(
                f"Skipping fate {fate!r}: need at least {min_cells} root and terminal cells.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        sc.tl.rank_genes_groups(
            pair,
            groupby="_sccs_population",
            groups=[fate],
            reference=root_name,
            method=method,
            layer=layer,
            use_raw=use_raw,
            n_genes=n_genes,
            pts=True,
            key_added="_sccs_fate_markers",
        )
        table = sc.get.rank_genes_groups_df(pair, group=fate, key="_sccs_fate_markers")
        table = table.rename(
            columns={
                "names": "gene",
                "scores": "score",
                "logfoldchanges": "logfoldchange",
                "pvals": "pvalue",
                "pvals_adj": "pvalue_adj",
                "pct_nz_group": "pct_terminal",
                "pct_nz_reference": "pct_root",
            }
        )
        table["significant"] = (table["pvalue_adj"] < fdr_threshold) & (
            table["logfoldchange"].abs() >= logfc_threshold
        )
        table["analysis_type"] = "terminal_annotation_marker_vs_root"
        table = table.sort_values(["logfoldchange", "pvalue_adj"], ascending=[False, True])
        table = table.reset_index(drop=True)
        table["rank"] = np.arange(1, len(table) + 1)
        output[str(fate)] = table
        if verbose:
            print(
                f"\n[scCS] Fate markers: {fate} vs {root_name}; "
                f"significant={int(table['significant'].sum())}"
            )
            shown = table[table["significant"]].head(20)
            if shown.empty:
                shown = table.head(20)
            columns = [
                column
                for column in ["rank", "gene", "logfoldchange", "pvalue_adj"]
                if column in shown
            ]
            print(shown[columns].to_string(index=False))
    return output


# The pre-v0.8 name remains a concise alias; the documentation uses
# ``get_fate_markers`` to avoid causal language.
get_deg_drivers = get_fate_markers


__all__ = [
    "CommitmentGeneAssociationResult",
    "get_commitment_associated_genes",
    "get_fate_markers",
    "get_deg_drivers",
]
