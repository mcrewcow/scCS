"""Shared replicate-aware condition engine for scCS v0.8."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence
import warnings

import numpy as np
import pandas as pd

from .furcation import Furcation, LabelSpec, TerminalSpec
from .inference import percentile_interval
from .population import PopulationCommitmentSummary, summarize_commitment
from .single import SingleScorer
from .transitions import get_scvelo_transition_matrix


_METRIC_ALIASES = {
    "mean_commitment": "mean_commitment_contribution",
    "future_fate_contribution": "mean_commitment_contribution",
    "future_fate_affinity": "directional_affinity",
    "conditional_fate_affinity": "directional_affinity",
    "future_fate_reach": "commitment_strength",
    "discounted_fate_reach": "commitment_strength",
    "future_fate_entropy": "directional_entropy",
    "future_fate_specificity": "directional_specificity",
    "reach_supported_specificity": "specific_commitment",
    "resolved_commitment": "specific_commitment",
    "signed_progression": "progression_velocity",
    "signed_ordering_flux": "progression_velocity",
    "selected_path_coverage": "transition_coverage",
}
_FATE_METRICS = {
    "mean_commitment_contribution",
    "directional_affinity",
    "commitment_composition",
}
_SCALAR_METRICS = {
    "commitment_strength",
    "directional_entropy",
    "commitment_entropy",
    "directional_specificity",
    "nearest_fate_angle_degrees",
    "specific_commitment",
    "progression_velocity",
    "transition_coverage",
}
_PAIR_METRICS = {"pairwise_log_commitment_ratio"}


def _safe_name(value: object) -> str:
    return str(value).replace("::", "__")


def _canonical_metric(metric: str) -> str:
    """Return the canonical public metric name.

    Compatibility and future-fate names map onto the shared internal arrays.
    Public documentation uses mode-specific names, while established result
    tables remain backward compatible.
    """
    value = str(metric)
    return _METRIC_ALIASES.get(value, value)


def _metric_label(metric: str, *, fate: Optional[str] = None) -> str:
    """Return a concise human-facing label for one requested outcome.

    Future-fate aliases retain the terminology used in the public DFFP
    documentation, while established instantaneous names remain available for
    backward compatibility.
    """
    requested = str(metric)
    public_labels = {
        "future_fate_contribution": "Future-fate contribution",
        "future_fate_affinity": "Conditional Fate Affinity (CFA)",
        "conditional_fate_affinity": "Conditional Fate Affinity (CFA)",
        "future_fate_reach": "Discounted Fate Reach (DFR)",
        "discounted_fate_reach": "Discounted Fate Reach (DFR)",
        "future_fate_entropy": "Future-fate entropy",
        "future_fate_specificity": "Future-Fate Specificity (FFS)",
        "reach_supported_specificity": "Resolved Commitment (RC)",
        "resolved_commitment": "Resolved Commitment (RC)",
        "signed_progression": "Signed Ordering Flux (SOF)",
        "signed_ordering_flux": "Signed Ordering Flux (SOF)",
        "selected_path_coverage": "Selected-path coverage",
    }
    if requested in public_labels:
        label = public_labels[requested]
    else:
        canonical = _canonical_metric(requested)
        labels = {
            "mean_commitment_contribution": "Mean commitment contribution",
            "directional_affinity": "Directional affinity",
            "commitment_composition": "Commitment composition",
            "commitment_strength": "Commitment strength",
            "directional_entropy": "Directional entropy",
            "commitment_entropy": "Commitment entropy",
            "directional_specificity": "Directional specificity",
            "nearest_fate_angle_degrees": "Nearest fate angle (degrees)",
            "specific_commitment": "Specific commitment",
            "progression_velocity": "Progression velocity",
            "transition_coverage": "Transition coverage",
            "pairwise_log_commitment_ratio": "Pairwise log commitment ratio",
        }
        label = labels.get(canonical, canonical.replace("_", " ").capitalize())
    if fate is not None:
        label += f" toward {fate}"
    return label


def _outcome_column(
    metric: str,
    *,
    fate: Optional[str] = None,
    fate_pair: Optional[tuple[str, str]] = None,
) -> str:
    metric = _canonical_metric(metric)
    if metric in _FATE_METRICS:
        if fate is None:
            raise ValueError(f"metric={metric!r} requires a fate.")
        return f"{metric}::{_safe_name(fate)}"
    if metric in _SCALAR_METRICS:
        return f"mean_{metric}"
    if metric in _PAIR_METRICS:
        if fate_pair is None or len(fate_pair) != 2:
            raise ValueError(f"metric={metric!r} requires fate_pair=(fate_a, fate_b).")
        return (
            f"pairwise_log_commitment_ratio::{_safe_name(fate_pair[0])}::{_safe_name(fate_pair[1])}"
        )
    raise ValueError(
        f"Unknown metric {metric!r}. Valid metrics are "
        f"{sorted(_FATE_METRICS | _SCALAR_METRICS | _PAIR_METRICS)}."
    )


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def _matrix_row_sums(matrix) -> np.ndarray:
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover
        sparse = None
    if sparse is not None and sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    return np.asarray(matrix, dtype=float).sum(axis=1)


def _block_transition_matrix(matrix, groups: np.ndarray):
    """Remove transitions between different condition/replicate groups."""
    labels = np.asarray(groups).astype(str)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Transition matrix shape must match the number of AnnData cells.")
    try:
        from scipy import sparse
    except ImportError:  # pragma: no cover - scipy is a core dependency
        sparse = None

    if sparse is not None and sparse.issparse(matrix):
        coo = matrix.tocoo(copy=True)
        keep = labels[coo.row] == labels[coo.col]
        blocked = sparse.csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=coo.shape,
        )
        blocked.eliminate_zeros()
        return blocked

    values = np.asarray(matrix, dtype=float).copy()
    values[labels[:, None] != labels[None, :]] = 0.0
    return values


@dataclass(frozen=True)
class ConditionCommitmentResult:
    """Condition-specific commitment values for one explicit population."""

    condition: str
    fate_names: tuple[str, ...]
    cell_ids: np.ndarray
    replicate_ids: Optional[np.ndarray]
    directional_affinity: np.ndarray
    commitment_strength: np.ndarray
    commitment_contribution: np.ndarray
    directional_entropy: np.ndarray
    commitment_entropy: np.ndarray
    directional_specificity: np.ndarray
    nearest_fate_angle_degrees: np.ndarray
    specific_commitment: np.ndarray
    progression_velocity: np.ndarray
    transition_coverage: np.ndarray
    status: np.ndarray
    population_summary: PopulationCommitmentSummary
    replicate_summary: pd.DataFrame
    population: str

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_replicates(self) -> int:
        if self.replicate_ids is None:
            return 0
        return int(len(np.unique(self.replicate_ids)))


class ConditionScorer:
    """Shared pooled-geometry engine for condition comparisons.

    The scorer builds and fits one :class:`SingleScorer` on all conditions
    pooled together.  Directional geometry and the magnitude scale are shared.
    Formal inference uses biological-replicate summaries; cell-level outputs
    are descriptive and may only enter the within-replicate level of a
    hierarchical bootstrap.
    """

    minimum_conditions: int = 2
    maximum_conditions: Optional[int] = None

    def __init__(
        self,
        adata,
        root: Optional[LabelSpec] = None,
        branches: Optional[TerminalSpec] = None,
        condition_obs_key: str = "condition",
        obs_key: str = "leiden",
        replicate_obs_key: Optional[str] = None,
        copy: bool = False,
        *,
        furcation: Optional[Furcation] = None,
        replicate_key: Optional[str] = None,
        condition_order: Optional[Sequence[object]] = None,
        design: str = "independent",
    ) -> None:
        design = str(design).lower()
        if design != "independent":
            raise NotImplementedError(
                "scCS v0.8 currently supports independent-group condition designs only. "
                "Paired/repeated-measure inference will be added separately."
            )

        if replicate_key is not None:
            if replicate_obs_key is not None and replicate_obs_key != replicate_key:
                raise ValueError("replicate_obs_key and replicate_key disagree.")
            replicate_obs_key = replicate_key

        if condition_obs_key not in adata.obs:
            raise ValueError(f"condition_obs_key={condition_obs_key!r} is missing from adata.obs.")
        if replicate_obs_key is not None and replicate_obs_key not in adata.obs:
            raise ValueError(f"replicate_obs_key={replicate_obs_key!r} is missing from adata.obs.")

        self.adata = adata.copy() if copy else adata
        self.condition_obs_key = condition_obs_key
        self.replicate_obs_key = replicate_obs_key
        self.design = design
        self._scorer = SingleScorer(
            self.adata,
            root=root,
            branches=branches,
            obs_key=obs_key,
            copy=False,
            furcation=furcation,
        )

        # Conditions are defined only on cells participating in the supervised
        # furcation. Missing values on unrelated cells must not create a
        # spurious ``"nan"`` condition. Conversely, every selected furcation
        # cell must have complete condition/replicate metadata.
        labels = self.adata.obs[self._scorer.obs_key].astype(str).to_numpy()
        validation = self._scorer.furcation.validate_labels(labels)
        selected_index = self.adata.obs.index[validation.selected_mask]
        selected_conditions = self.adata.obs.loc[selected_index, condition_obs_key]
        if selected_conditions.isna().any():
            n_missing = int(selected_conditions.isna().sum())
            raise ValueError(
                f"{condition_obs_key!r} is missing for {n_missing} selected furcation cells."
            )
        observed = [str(value) for value in pd.unique(selected_conditions)]
        if condition_order is None:
            if isinstance(selected_conditions.dtype, pd.CategoricalDtype):
                category_order = [str(value) for value in selected_conditions.cat.categories]
                conditions = [value for value in category_order if value in set(observed)]
                self.condition_order_source = "categorical_categories"
            else:
                # Preserve first appearance.  Alphabetical sorting can silently
                # scramble dose, time, or severity designs.  Users should pass
                # condition_order explicitly whenever the intended order is not
                # already encoded in the data.
                conditions = observed
                self.condition_order_source = "first_appearance"
        else:
            conditions = [str(value) for value in condition_order]
            if len(conditions) != len(set(conditions)):
                raise ValueError("condition_order contains duplicates.")
            if set(conditions) != set(observed):
                raise ValueError(
                    "condition_order must contain every condition observed among "
                    "selected furcation cells exactly once."
                )
            self.condition_order_source = "explicit"

        if replicate_obs_key is not None:
            selected_replicates = self.adata.obs.loc[selected_index, replicate_obs_key]
            if selected_replicates.isna().any():
                n_missing = int(selected_replicates.isna().sum())
                raise ValueError(
                    f"{replicate_obs_key!r} is missing for {n_missing} selected furcation cells."
                )

        if len(conditions) < self.minimum_conditions:
            raise ValueError(
                f"{type(self).__name__} requires at least {self.minimum_conditions} "
                f"conditions among selected furcation cells; found {conditions}."
            )
        if self.maximum_conditions is not None and len(conditions) > self.maximum_conditions:
            raise ValueError(
                f"{type(self).__name__} requires at most {self.maximum_conditions} "
                f"conditions among selected furcation cells; found {conditions}."
            )

        self.conditions = conditions
        self.furcation = self._scorer.furcation
        self.root = self._scorer.root
        self.branches = list(self._scorer.branches)
        self.obs_key = self._scorer.obs_key
        self._condition_results: Optional[Dict[str, ConditionCommitmentResult]] = None
        self.transition_scope = "pooled"
        self._transition_scope_coverage: Optional[np.ndarray] = None
        self._transition_scope_removed_mass: Optional[np.ndarray] = None

    def preflight(
        self,
        *,
        ordering_metric="pseudotime",
        check_velocity: bool = True,
        raise_on_error: bool = False,
    ):
        """Run pooled furcation, condition, and replicate diagnostics."""
        from .preflight import condition_preflight

        report = condition_preflight(
            self,
            ordering_metric=ordering_metric,
            check_velocity=check_velocity,
        )
        if raise_on_error:
            report.raise_for_errors()
        return report

    # ------------------------------------------------------------------
    # Familiar pooled workflow delegates
    # ------------------------------------------------------------------

    def compute_velocity(self, *args, **kwargs) -> "ConditionScorer":
        self._scorer.compute_velocity(*args, **kwargs)
        self._condition_results = None
        return self

    def build_embedding(self, *args, **kwargs) -> "ConditionScorer":
        self._scorer.build_embedding(*args, **kwargs)
        self._condition_results = None
        return self

    def project_velocity(self, *args, **kwargs) -> "ConditionScorer":
        self._scorer.project_velocity(*args, **kwargs)
        self._condition_results = None
        return self

    def load_velocity_vectors(self, *args, **kwargs) -> "ConditionScorer":
        self._scorer.load_velocity_vectors(*args, **kwargs)
        self._condition_results = None
        return self

    def fit(
        self,
        transition_matrix=None,
        *,
        transition_scope: str = "pooled",
        **kwargs,
    ) -> "ConditionScorer":
        """Fit one pooled geometry with an explicit transition-graph scope.

        Parameters
        ----------
        transition_scope
            ``"pooled"`` keeps the supplied/full transition graph.
            ``"condition"`` removes edges connecting different conditions.
            ``"replicate"`` removes edges connecting different biological
            replicates (condition-qualified to avoid accidental donor-name
            collisions).  Blocked rows are renormalized later by the induced
            furcation projector.

        Notes
        -----
        When projected velocity vectors were loaded directly, graph blocking
        is not applicable and ``transition_scope`` must remain ``"pooled"``.
        """
        scope = str(transition_scope)
        if scope not in {"pooled", "condition", "replicate"}:
            raise ValueError("transition_scope must be 'pooled', 'condition', or 'replicate'.")
        if scope != "pooled" and self._scorer._projection_result is not None:
            raise ValueError(
                "transition_scope blocking cannot be applied after projected "
                "velocity vectors have been loaded. Build the embedding and fit "
                "from a transition matrix instead."
            )

        matrix = transition_matrix
        self._transition_scope_coverage = None
        self._transition_scope_removed_mass = None
        if scope != "pooled":
            if matrix is None:
                matrix = get_scvelo_transition_matrix(self.adata)
            original_mass = _matrix_row_sums(matrix)
            if scope == "condition":
                groups = self.adata.obs[self.condition_obs_key].astype(str).to_numpy()
            else:
                if self.replicate_obs_key is None:
                    raise ValueError("transition_scope='replicate' requires replicate_obs_key.")
                groups = (
                    self.adata.obs[self.condition_obs_key].astype(str)
                    + "::"
                    + self.adata.obs[self.replicate_obs_key].astype(str)
                ).to_numpy()
            matrix = _block_transition_matrix(matrix, groups)
            retained_mass = _matrix_row_sums(matrix)
            coverage = np.ones_like(original_mass, dtype=float)
            positive = original_mass > 0
            coverage[positive] = retained_mass[positive] / original_mass[positive]
            coverage[~positive] = 0.0
            self._transition_scope_coverage = coverage
            self._transition_scope_removed_mass = np.maximum(original_mass - retained_mass, 0.0)

        self._scorer.fit(transition_matrix=matrix, **kwargs)
        self.transition_scope = scope
        self._condition_results = None
        return self

    @property
    def is_fitted(self) -> bool:
        return self._scorer.is_fitted

    @property
    def result(self):
        return self._scorer.result

    # ------------------------------------------------------------------
    # Condition and replicate summaries
    # ------------------------------------------------------------------

    def _selected_metadata(self) -> pd.DataFrame:
        result = self._require_fitted_result()
        metadata = self.adata.obs.loc[result.cell_ids].copy()
        if metadata[self.condition_obs_key].isna().any():
            raise ValueError(
                f"{self.condition_obs_key!r} contains missing values among "
                "selected furcation cells."
            )
        metadata[self.condition_obs_key] = metadata[self.condition_obs_key].astype(str)
        if self.replicate_obs_key is not None:
            if metadata[self.replicate_obs_key].isna().any():
                raise ValueError(
                    f"{self.replicate_obs_key!r} contains missing values among "
                    "selected furcation cells."
                )
            raw = metadata[self.replicate_obs_key].astype(str)
            metadata["__sccs_replicate__"] = (
                metadata[self.condition_obs_key].astype(str) + "::" + raw
            )
            metadata["__sccs_replicate_label__"] = raw
        return metadata

    def _population_mask(self, population: str | np.ndarray) -> tuple[np.ndarray, str]:
        result = self._require_fitted_result()
        if isinstance(population, str):
            if population == "root":
                return result.root_mask.copy(), "root"
            if population == "all":
                return np.ones(len(result.cell_ids), dtype=bool), "all"
            raise ValueError("population must be 'root', 'all', or a Boolean mask.")
        mask = np.asarray(population)
        if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(result.cell_ids):
            raise ValueError(
                "A custom population mask must be Boolean and aligned to selected cells."
            )
        if not mask.any():
            raise ValueError("The custom population mask is empty.")
        return mask.copy(), "custom"

    def score_all_conditions(
        self,
        *,
        population: str | np.ndarray = "root",
        min_cells: int = 5,
        min_replicates: int = 2,
        write_to_adata: bool = True,
        verbose: bool = True,
    ) -> Dict[str, ConditionCommitmentResult]:
        """Create condition-specific descriptive and replicate-level summaries."""
        if min_cells < 1 or min_replicates < 1:
            raise ValueError("min_cells and min_replicates must be positive.")
        pooled = self._require_fitted_result()
        if write_to_adata:
            pooled.write_to_adata(self.adata)

        metadata = self._selected_metadata()
        population_mask, population_name = self._population_mask(population)
        condition_values = metadata[self.condition_obs_key].to_numpy(dtype=str)
        results: Dict[str, ConditionCommitmentResult] = {}

        eligibility_errors: list[str] = []
        for condition in self.conditions:
            mask = population_mask & (condition_values == condition)
            n_cells = int(mask.sum())
            if n_cells < min_cells:
                eligibility_errors.append(
                    f"{condition!r} has {n_cells} eligible cells; requires at least {min_cells}"
                )
                continue

            replicate_ids: Optional[np.ndarray]
            if self.replicate_obs_key is None:
                replicate_ids = None
                replicate_table = pd.DataFrame()
            else:
                replicate_ids = metadata.loc[mask, "__sccs_replicate__"].to_numpy(str)
                replicate_table = self._build_replicate_summary(
                    condition,
                    mask,
                    metadata,
                )
                n_replicates = len(replicate_table)
                if n_replicates < min_replicates:
                    eligibility_errors.append(
                        f"{condition!r} has {n_replicates} biological replicates; "
                        f"requires at least {min_replicates}"
                    )
                    continue
                if n_replicates < 4:
                    warnings.warn(
                        f"Condition {condition!r} contains only {n_replicates} "
                        "biological replicates. Formal inference is available but has "
                        "limited resolution; at least 4 and preferably 5-6 replicates "
                        "per condition are recommended.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            summary = summarize_commitment(pooled.commitment_contribution[mask])
            results[condition] = ConditionCommitmentResult(
                condition=condition,
                fate_names=pooled.fate_names,
                cell_ids=pooled.cell_ids[mask].copy(),
                replicate_ids=None if replicate_ids is None else replicate_ids.copy(),
                directional_affinity=pooled.directional_affinity[mask].copy(),
                commitment_strength=pooled.commitment_strength[mask].copy(),
                commitment_contribution=pooled.commitment_contribution[mask].copy(),
                directional_entropy=pooled.directional_entropy[mask].copy(),
                commitment_entropy=pooled.commitment_entropy[mask].copy(),
                directional_specificity=pooled.directional_specificity[mask].copy(),
                nearest_fate_angle_degrees=pooled.nearest_fate_angle_degrees[mask].copy(),
                specific_commitment=pooled.specific_commitment[mask].copy(),
                progression_velocity=pooled.progression_velocity[mask].copy(),
                transition_coverage=pooled.projection.transition_coverage[mask].copy(),
                status=pooled.status[mask].copy(),
                population_summary=summary,
                replicate_summary=replicate_table,
                population=population_name,
            )
            if verbose:
                replicate_text = (
                    "no replicate key"
                    if replicate_ids is None
                    else f"{len(replicate_table)} replicates"
                )
                print(f"[scCS] {condition!r}: {n_cells} {population_name} cells; {replicate_text}.")

        if eligibility_errors:
            raise ValueError(
                "Condition eligibility failed; partial designs are not scored: "
                + "; ".join(eligibility_errors)
            )
        if set(results) != set(self.conditions):
            raise RuntimeError(
                "Internal error: condition scoring did not return every configured condition."
            )

        self._condition_results = results
        return results

    def _build_replicate_summary(
        self,
        condition: str,
        condition_mask: np.ndarray,
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        pooled = self._require_fitted_result()
        replicate_values = metadata["__sccs_replicate__"].to_numpy(str)
        raw_labels = metadata["__sccs_replicate_label__"].to_numpy(str)
        rows = []
        for replicate_id in pd.unique(replicate_values[condition_mask]):
            mask = condition_mask & (replicate_values == replicate_id)
            index = np.flatnonzero(mask)
            row = {
                "condition": condition,
                "replicate_id": replicate_id,
                "replicate_label": str(raw_labels[index[0]]),
                "n_cells": int(len(index)),
                "n_valid_projection": int(np.sum(pooled.projection.velocity_defined[index])),
                "mean_commitment_strength": _finite_mean(pooled.commitment_strength[index]),
                "mean_directional_entropy": _finite_mean(pooled.directional_entropy[index]),
                "mean_commitment_entropy": _finite_mean(pooled.commitment_entropy[index]),
                "mean_directional_specificity": _finite_mean(pooled.directional_specificity[index]),
                "mean_nearest_fate_angle_degrees": _finite_mean(
                    pooled.nearest_fate_angle_degrees[index]
                ),
                "mean_specific_commitment": _finite_mean(pooled.specific_commitment[index]),
                "mean_progression_velocity": _finite_mean(pooled.progression_velocity[index]),
                "mean_transition_coverage": _finite_mean(
                    pooled.projection.transition_coverage[index]
                ),
            }
            summary = summarize_commitment(pooled.commitment_contribution[index])
            for fate_index, fate in enumerate(pooled.fate_names):
                row[_outcome_column("mean_commitment_contribution", fate=fate)] = float(
                    pooled.commitment_contribution[index, fate_index].mean()
                )
                row[_outcome_column("directional_affinity", fate=fate)] = float(
                    pooled.directional_affinity[index, fate_index].mean()
                )
                row[_outcome_column("commitment_composition", fate=fate)] = (
                    float(summary.commitment_composition[fate_index])
                    if summary.composition_defined
                    else float("nan")
                )
            for fate_a, fate_b in combinations(pooled.fate_names, 2):
                a = pooled.fate_names.index(fate_a)
                b = pooled.fate_names.index(fate_b)
                value = float(summary.pairwise_log_commitment_ratio[a, b])
                row[
                    _outcome_column(
                        "pairwise_log_commitment_ratio",
                        fate_pair=(fate_a, fate_b),
                    )
                ] = value
                # Pairwise log ratios are antisymmetric.  Store the reverse
                # orientation as well so public fate_pair input is genuinely
                # ordered rather than depending on the internal fate list.
                row[
                    _outcome_column(
                        "pairwise_log_commitment_ratio",
                        fate_pair=(fate_b, fate_a),
                    )
                ] = -value
            rows.append(row)
        return pd.DataFrame(rows).sort_values("replicate_id").reset_index(drop=True)

    def replicate_table(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
    ) -> pd.DataFrame:
        """Concatenate replicate summaries across conditions."""
        chosen = self._resolve_results(results)
        frames = [result.replicate_summary for result in chosen.values()]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            raise ValueError(
                "No replicate summaries are available. Supply replicate_obs_key and "
                "call score_all_conditions()."
            )
        return pd.concat(frames, ignore_index=True)

    def _outcome_specs(
        self,
        metric: str,
        *,
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
    ) -> list[tuple[Optional[str], Optional[tuple[str, str]], str]]:
        metric = _canonical_metric(metric)
        fate_names = tuple(self._require_fitted_result().fate_names)
        if metric in _FATE_METRICS:
            fates = fate_names if fate is None else (str(fate),)
            invalid = [name for name in fates if name not in fate_names]
            if invalid:
                raise ValueError(f"Unknown fate(s): {invalid}; valid fates are {fate_names}.")
            return [(name, None, _outcome_column(metric, fate=name)) for name in fates]
        if metric in _SCALAR_METRICS:
            if fate is not None or fate_pair is not None:
                raise ValueError(f"metric={metric!r} does not use fate arguments.")
            return [(None, None, _outcome_column(metric))]
        if metric in _PAIR_METRICS:
            pairs = list(combinations(fate_names, 2)) if fate_pair is None else [fate_pair]
            for pair in pairs:
                if (
                    len(pair) != 2
                    or pair[0] == pair[1]
                    or pair[0] not in fate_names
                    or pair[1] not in fate_names
                ):
                    raise ValueError(f"Invalid fate pair {pair!r} for fates {fate_names}.")
            return [
                (None, tuple(pair), _outcome_column(metric, fate_pair=tuple(pair)))
                for pair in pairs
            ]
        _outcome_column(metric)
        raise AssertionError("unreachable")

    def hierarchical_bootstrap(
        self,
        *,
        condition_a: str,
        condition_b: str,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        resample_cells_within_replicate: bool = True,
        random_state: int = 0,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
    ) -> pd.DataFrame:
        """Hierarchical bootstrap with replicate-first resampling."""
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be positive.")
        chosen = self._resolve_results(results)
        metric_public = str(metric)
        metric = _canonical_metric(metric)
        condition_a = str(condition_a)
        condition_b = str(condition_b)
        if condition_a not in chosen or condition_b not in chosen:
            raise ValueError("Both requested conditions must have scored results.")
        if self.replicate_obs_key is None:
            raise ValueError("hierarchical_bootstrap requires replicate_obs_key.")

        specs = self._outcome_specs(metric, fate=fate, fate_pair=fate_pair)
        rng = np.random.default_rng(random_state)
        rows = []
        for fate_name, pair, column in specs:
            point_table = self.replicate_table(chosen)
            point_a = point_table.loc[point_table["condition"] == condition_a, column]
            point_b = point_table.loc[point_table["condition"] == condition_b, column]
            estimate = float(point_b.mean() - point_a.mean())
            samples = np.empty(n_bootstrap, dtype=float)
            for iteration in range(n_bootstrap):
                estimate_a = self._bootstrap_condition_mean(
                    chosen[condition_a],
                    metric,
                    fate=fate_name,
                    fate_pair=pair,
                    rng=rng,
                    resample_cells=resample_cells_within_replicate,
                )
                estimate_b = self._bootstrap_condition_mean(
                    chosen[condition_b],
                    metric,
                    fate=fate_name,
                    fate_pair=pair,
                    rng=rng,
                    resample_cells=resample_cells_within_replicate,
                )
                samples[iteration] = estimate_b - estimate_a
            interval = percentile_interval(
                samples,
                confidence_level=confidence_level,
                estimate=estimate,
            )
            rows.append(
                {
                    "metric": metric,
                    "metric_public": metric_public,
                    "metric_label": _metric_label(metric_public, fate=fate_name),
                    "fate": fate_name,
                    "fate_a": None if pair is None else pair[0],
                    "fate_b": None if pair is None else pair[1],
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "effect_b_minus_a": estimate,
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    "confidence_level": interval.confidence_level,
                    "n_bootstrap": interval.n_bootstrap,
                    "resample_cells_within_replicate": bool(resample_cells_within_replicate),
                }
            )
        return pd.DataFrame(rows)

    def _bootstrap_condition_mean(
        self,
        result: ConditionCommitmentResult,
        metric: str,
        *,
        fate: Optional[str],
        fate_pair: Optional[tuple[str, str]],
        rng: np.random.Generator,
        resample_cells: bool,
    ) -> float:
        if result.replicate_ids is None:
            raise ValueError("Replicate IDs are required for hierarchical bootstrap.")
        unique = np.unique(result.replicate_ids)
        if len(unique) < 2:
            raise ValueError("At least two replicates per condition are required.")
        selected_replicates = rng.choice(unique, size=len(unique), replace=True)
        replicate_values = []
        for replicate_id in selected_replicates:
            indices = np.flatnonzero(result.replicate_ids == replicate_id)
            if resample_cells:
                indices = rng.choice(indices, size=len(indices), replace=True)
            replicate_values.append(
                self._cell_outcome(result, indices, metric, fate=fate, fate_pair=fate_pair)
            )
        return float(np.mean(replicate_values))

    @staticmethod
    def _cell_outcome(
        result: ConditionCommitmentResult,
        indices: np.ndarray,
        metric: str,
        *,
        fate: Optional[str],
        fate_pair: Optional[tuple[str, str]],
    ) -> float:
        metric = _canonical_metric(metric)
        if len(indices) == 0:
            raise ValueError("Cannot calculate an outcome from zero cells.")
        if metric in _FATE_METRICS:
            if fate is None:
                raise ValueError(f"metric={metric!r} requires a fate.")
            fate_index = result.fate_names.index(fate)
            if metric == "mean_commitment_contribution":
                return float(result.commitment_contribution[indices, fate_index].mean())
            if metric == "directional_affinity":
                return float(result.directional_affinity[indices, fate_index].mean())
            summary = summarize_commitment(result.commitment_contribution[indices])
            if not summary.composition_defined:
                return float("nan")
            return float(summary.commitment_composition[fate_index])
        if metric == "commitment_strength":
            return _finite_mean(result.commitment_strength[indices])
        if metric == "directional_entropy":
            return _finite_mean(result.directional_entropy[indices])
        if metric == "commitment_entropy":
            return _finite_mean(result.commitment_entropy[indices])
        if metric == "directional_specificity":
            return _finite_mean(result.directional_specificity[indices])
        if metric == "nearest_fate_angle_degrees":
            return _finite_mean(result.nearest_fate_angle_degrees[indices])
        if metric == "specific_commitment":
            return _finite_mean(result.specific_commitment[indices])
        if metric == "progression_velocity":
            return _finite_mean(result.progression_velocity[indices])
        if metric == "transition_coverage":
            return _finite_mean(result.transition_coverage[indices])
        if metric == "pairwise_log_commitment_ratio":
            if fate_pair is None:
                raise ValueError("pairwise_log_commitment_ratio requires fate_pair.")
            summary = summarize_commitment(result.commitment_contribution[indices])
            a = result.fate_names.index(fate_pair[0])
            b = result.fate_names.index(fate_pair[1])
            return float(summary.pairwise_log_commitment_ratio[a, b])
        raise ValueError(f"Unknown metric {metric!r}.")

    def transition_scope_summary(
        self,
        *,
        population: str = "root",
    ) -> pd.DataFrame:
        """Summarize transition mass retained by condition/replicate blocking.

        This diagnostic is distinct from furcation transition coverage.  It
        quantifies edges removed solely because they cross the requested
        ``transition_scope``.
        """
        result = self._require_fitted_result()
        selected = result.embedding.selected_indices
        if population == "root":
            population_mask = result.root_mask
        elif population == "all":
            population_mask = np.ones(len(selected), dtype=bool)
        else:
            raise ValueError("population must be 'root' or 'all'.")

        metadata = self.adata.obs.iloc[selected].copy()
        metadata["condition"] = metadata[self.condition_obs_key].astype(str).to_numpy()
        if self.replicate_obs_key is not None:
            metadata["replicate"] = (
                metadata[self.condition_obs_key].astype(str)
                + "::"
                + metadata[self.replicate_obs_key].astype(str)
            ).to_numpy()
        else:
            metadata["replicate"] = "not_available"

        if self._transition_scope_coverage is None:
            scope_coverage = np.ones(len(selected), dtype=float)
            removed_mass = np.zeros(len(selected), dtype=float)
        else:
            scope_coverage = self._transition_scope_coverage[selected]
            removed_mass = self._transition_scope_removed_mass[selected]

        table = pd.DataFrame(
            {
                "condition": metadata["condition"].to_numpy(),
                "replicate_id": metadata["replicate"].to_numpy(),
                "scope_coverage": scope_coverage,
                "scope_removed_mass": removed_mass,
                "furcation_transition_coverage": result.projection.transition_coverage,
            },
            index=result.cell_ids,
        )
        table = table.loc[population_mask]
        return table.groupby(["condition", "replicate_id"], as_index=False).agg(
            n_cells=("scope_coverage", "size"),
            mean_scope_coverage=("scope_coverage", "mean"),
            q05_scope_coverage=("scope_coverage", lambda x: float(np.quantile(x, 0.05))),
            mean_scope_removed_mass=("scope_removed_mass", "mean"),
            mean_furcation_transition_coverage=("furcation_transition_coverage", "mean"),
        )

    @staticmethod
    def _cell_metric_vector(
        result: ConditionCommitmentResult,
        metric: str,
        *,
        fate: Optional[str] = None,
    ) -> np.ndarray:
        """Return one cell-level vector for descriptive trend plots."""
        metric = _canonical_metric(metric)
        if metric == "mean_commitment_contribution":
            if fate is None:
                raise ValueError("mean_commitment_contribution requires fate.")
            return result.commitment_contribution[:, result.fate_names.index(fate)]
        if metric == "directional_affinity":
            if fate is None:
                raise ValueError("directional_affinity requires fate.")
            return result.directional_affinity[:, result.fate_names.index(fate)]
        if metric == "commitment_strength":
            return result.commitment_strength
        if metric == "directional_entropy":
            return result.directional_entropy
        if metric == "commitment_entropy":
            return result.commitment_entropy
        if metric == "directional_specificity":
            return result.directional_specificity
        if metric == "nearest_fate_angle_degrees":
            return result.nearest_fate_angle_degrees
        if metric == "specific_commitment":
            return result.specific_commitment
        if metric == "progression_velocity":
            return result.progression_velocity
        if metric == "transition_coverage":
            return result.transition_coverage
        raise ValueError(f"metric={metric!r} is not a cell-level trend outcome.")

    def plot_replicate_outcomes(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
        show_ci: bool = True,
        ax=None,
    ):
        """Plot every biological replicate with condition mean and uncertainty.

        Points are the formal units of inference.  Boxes/violins are avoided so
        that small replicate counts remain visible rather than appearing as a
        large cell-level sample.
        """
        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        specs = self._outcome_specs(metric, fate=fate, fate_pair=fate_pair)
        if len(specs) != 1:
            raise ValueError("Select exactly one fate or fate pair for this plot.")
        _, _, column = specs[0]
        table = self.replicate_table(chosen)

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 4.8))
        for position, condition in enumerate(self.conditions):
            values = table.loc[table["condition"] == condition, column].dropna().to_numpy(float)
            if len(values) == 0:
                continue
            offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                np.full(len(values), position, dtype=float) + offsets,
                values,
                zorder=3,
                label=condition,
            )
            mean = float(values.mean())
            ax.scatter([position], [mean], marker="D", s=55, zorder=4)
            if show_ci and len(values) > 1:
                sem = float(values.std(ddof=1) / np.sqrt(len(values)))
                ax.errorbar(
                    [position],
                    [mean],
                    yerr=[[1.96 * sem], [1.96 * sem]],
                    capsize=4,
                    linewidth=1.5,
                    zorder=2,
                )
        ax.set_xticks(np.arange(len(self.conditions)))
        ax.set_xticklabels(self.conditions)
        selected_fate = fate
        if selected_fate is None and fate_pair is not None:
            selected_fate = f"{fate_pair[0]} / {fate_pair[1]}"
        ax.set_ylabel(_metric_label(metric, fate=selected_fate))
        ax.set_title("scCS replicate-level outcomes")
        return ax.figure

    def plot_commitment_heatmap(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "mean_commitment_contribution",
        level: str = "condition",
        reference: Optional[str] = None,
        annotate: bool = True,
        ax=None,
    ):
        """Plot condition- or replicate-level fate commitment as a heatmap."""
        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        if metric not in _FATE_METRICS:
            raise ValueError(
                "plot_commitment_heatmap requires a fate-specific metric: "
                "mean_commitment_contribution, directional_affinity, or "
                "commitment_composition."
            )
        if level not in {"condition", "replicate"}:
            raise ValueError("level must be 'condition' or 'replicate'.")

        table = self.replicate_table(chosen)
        columns = [_outcome_column(metric, fate=fate) for fate in self.branches]
        if level == "condition":
            matrix_frame = (
                table.groupby("condition", sort=False)[columns].mean().reindex(self.conditions)
            )
        else:
            matrix_frame = table.set_index("replicate_id")[columns]

        if reference is not None:
            reference = str(reference)
            condition_means = table.groupby("condition")[columns].mean()
            if reference not in condition_means.index:
                raise ValueError(f"Unknown reference condition {reference!r}.")
            matrix_frame = matrix_frame - condition_means.loc[reference].to_numpy()

        import matplotlib.pyplot as plt

        if ax is None:
            height = max(3.2, 0.34 * len(matrix_frame) + 1.5)
            _, ax = plt.subplots(figsize=(7.2, height))
        image = ax.imshow(matrix_frame.to_numpy(dtype=float), aspect="auto")
        ax.set_xticks(np.arange(len(self.branches)))
        ax.set_xticklabels(self.branches, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(matrix_frame)))
        ax.set_yticklabels(matrix_frame.index.astype(str))
        title = _metric_label(metric)
        if reference is not None:
            title += f" change from {reference}"
        ax.set_title(title)
        ax.figure.colorbar(image, ax=ax, label=_metric_label(metric))
        if annotate and matrix_frame.size <= 120:
            values = matrix_frame.to_numpy(dtype=float)
            for row in range(values.shape[0]):
                for column_index in range(values.shape[1]):
                    value = values[row, column_index]
                    if np.isfinite(value):
                        ax.text(column_index, row, f"{value:.3f}", ha="center", va="center")
        return ax.figure

    def plot_pseudotime_trends(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        pseudotime_key: Optional[str] = None,
        metric: str = "specific_commitment",
        fate: Optional[str] = None,
        n_bins: int = 8,
        show_replicates: bool = True,
        scale_pseudotime: bool = True,
        ax=None,
    ):
        """Plot replicate-aware commitment trends over root pseudotime."""
        if n_bins < 3:
            raise ValueError("n_bins must be at least 3.")
        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        if pseudotime_key is None:
            candidate = self._scorer._ordering_argument
            if not isinstance(candidate, str):
                raise ValueError("pseudotime_key is required when build_embedding used an array.")
            pseudotime_key = candidate
        if pseudotime_key not in self.adata.obs:
            raise KeyError(f"{pseudotime_key!r} is missing from adata.obs.")

        all_pt = np.concatenate(
            [
                self.adata.obs.loc[result.cell_ids, pseudotime_key].to_numpy(dtype=float)
                for result in chosen.values()
            ]
        )
        finite_pt = all_pt[np.isfinite(all_pt)]
        if len(finite_pt) == 0:
            raise ValueError("No finite pseudotime values are available.")
        pt_min = float(np.min(finite_pt))
        pt_max = float(np.max(finite_pt))
        if scale_pseudotime:
            if pt_max <= pt_min:
                raise ValueError("Pseudotime is constant and cannot be scaled.")
            finite_pt = (finite_pt - pt_min) / (pt_max - pt_min)
        edges = np.unique(np.quantile(finite_pt, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 4:
            raise ValueError("Pseudotime has too few distinct values for trend binning.")
        centers = 0.5 * (edges[:-1] + edges[1:])

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 5.0))
        for condition in self.conditions:
            result = chosen[condition]
            if result.replicate_ids is None:
                raise ValueError("plot_pseudotime_trends requires replicate_obs_key.")
            pt = self.adata.obs.loc[result.cell_ids, pseudotime_key].to_numpy(dtype=float)
            if scale_pseudotime:
                pt = (pt - pt_min) / (pt_max - pt_min)
            outcome = self._cell_metric_vector(result, metric, fate=fate)
            replicate_curves = []
            for replicate_id in np.unique(result.replicate_ids):
                replicate_mask = result.replicate_ids == replicate_id
                bins = np.digitize(pt[replicate_mask], edges[1:-1], right=False)
                curve = np.full(len(centers), np.nan, dtype=float)
                for bin_index in range(len(centers)):
                    values = outcome[replicate_mask][bins == bin_index]
                    values = values[np.isfinite(values)]
                    if len(values):
                        curve[bin_index] = values.mean()
                replicate_curves.append(curve)
                if show_replicates:
                    ax.plot(centers, curve, linewidth=1.0, alpha=0.22)
            mean_curve = np.nanmean(np.asarray(replicate_curves), axis=0)
            ax.plot(centers, mean_curve, marker="o", linewidth=2.4, label=condition)
        ax.set_xlabel("Root pseudotime (scaled 0–1)" if scale_pseudotime else pseudotime_key)
        ax.set_ylabel(_metric_label(metric, fate=fate))
        ax.set_title("Replicate-aware commitment trends")
        ax.legend(frameon=False)
        return ax.figure

    def plot_transition_coverage(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        ax=None,
    ):
        """Plot replicate-level projection coverage by condition.

        When projected velocities were supplied directly, transition coverage
        is defined as 1 by construction. The plot remains available for a
        complete QC report but is explicitly labeled as non-informative.
        """
        figure = self.plot_replicate_outcomes(
            results=results,
            metric="transition_coverage",
            ax=ax,
        )
        pooled = self._require_fitted_result()
        supplied = (
            pooled.projection.min_coverage == 0.0
            and np.allclose(pooled.projection.transition_coverage, 1.0)
            and np.allclose(pooled.projection.external_transition_mass, 0.0)
        )
        if supplied:
            warnings.warn(
                "Projected velocities were supplied directly; transition "
                "coverage is fixed at 1 and is not a projection QC metric.",
                RuntimeWarning,
                stacklevel=2,
            )
            axis = figure.axes[0]
            axis.clear()
            axis.axis("off")
            axis.text(
                0.5,
                0.58,
                "Transition coverage is fixed at 1",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                transform=axis.transAxes,
            )
            axis.text(
                0.5,
                0.43,
                "Projected velocities were supplied directly;\n"
                "coverage is not available as a projection QC measure.",
                ha="center",
                va="center",
                fontsize=10,
                transform=axis.transAxes,
            )
        return figure

    def plot_status_composition(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        ax=None,
    ):
        """Plot descriptive cell-status proportions for each condition."""
        chosen = self._resolve_results(results)
        statuses = sorted({str(status) for result in chosen.values() for status in result.status})
        proportions = np.zeros((len(self.conditions), len(statuses)), dtype=float)
        for row, condition in enumerate(self.conditions):
            values = np.asarray(chosen[condition].status).astype(str)
            for column, status in enumerate(statuses):
                proportions[row, column] = np.mean(values == status)

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 4.8))
        bottom = np.zeros(len(self.conditions), dtype=float)
        x = np.arange(len(self.conditions))
        for column, status in enumerate(statuses):
            ax.bar(x, proportions[:, column], bottom=bottom, label=status)
            bottom += proportions[:, column]
        ax.set_xticks(x)
        ax.set_xticklabels(self.conditions)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Cell proportion")
        ax.set_title("scCS commitment-status composition")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        return ax.figure

    def _condition_display_mask(
        self,
        condition: object,
        *,
        population: str = "all",
    ) -> np.ndarray:
        """Return a pooled-result mask for one condition and population."""
        pooled = self._require_fitted_result()
        condition = str(condition)
        if condition not in self.conditions:
            raise ValueError(f"Unknown condition {condition!r}; expected one of {self.conditions}.")
        metadata = self._selected_metadata()
        mask = metadata[self.condition_obs_key].to_numpy(dtype=str) == condition
        if population == "root":
            mask &= pooled.root_mask
        elif population == "terminal":
            mask &= pooled.terminal_mask
        elif population != "all":
            raise ValueError("population must be 'root', 'terminal', or 'all'.")
        if not np.any(mask):
            raise ValueError(
                f"Condition {condition!r} contains no selected cells for population={population!r}."
            )
        return mask

    def plot_star_grid(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        color_by: str = "specific_commitment",
        population: str = "all",
        ncols: int = 3,
        figsize_per_panel: tuple[float, float] = (6.2, 5.3),
        **plot_star_kwargs,
    ):
        """Plot the pooled v0.8 star separately for every condition.

        Geometry and score scaling remain pooled.  Each panel simply masks the
        pooled result to cells from one condition, which makes condition panels
        directly comparable without refitting a separate star.
        """
        self._resolve_results(results)
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")

        import matplotlib.pyplot as plt

        ncols = min(ncols, len(self.conditions))
        nrows = int(np.ceil(len(self.conditions) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )
        pooled = self._require_fitted_result()
        for condition, axis in zip(self.conditions, axes.ravel()):
            mask = self._condition_display_mask(condition, population=population)
            self._scorer.plot_star(
                pooled,
                color_by=color_by,
                cell_mask=mask,
                title=str(condition),
                ax=axis,
                **plot_star_kwargs,
            )
        for axis in axes.ravel()[len(self.conditions) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"scCS pooled geometry by condition ({population})",
            y=1.01,
        )
        return fig

    def plot_gene_expression_star_grid(
        self,
        gene: str,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        population: str = "all",
        layer: Optional[str] = None,
        use_raw: bool = False,
        gene_symbols: Optional[str] = None,
        log1p: bool = False,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        percentile_range: Optional[tuple[float, float]] = (1.0, 99.0),
        shared_scale: bool = True,
        sort_cells: bool = True,
        ncols: int = 3,
        figsize_per_panel: tuple[float, float] = (6.2, 5.3),
        **plot_star_kwargs,
    ):
        """Plot one gene on the pooled star separately for every condition.

        The fitted geometry, ordering, and expression source are shared. By
        default, all condition panels use one robust color scale so expression
        levels remain directly comparable. This is a descriptive visualization;
        it does not perform differential-expression inference.
        """
        self._resolve_results(results)
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")
        prohibited = {
            "color_values",
            "color_label",
            "cell_mask",
            "title",
            "ax",
            "cmap",
            "vmin",
            "vmax",
            "sort_by_color",
        }
        overlap = prohibited.intersection(plot_star_kwargs)
        if overlap:
            raise ValueError(
                "plot_star_kwargs contains arguments controlled by "
                f"plot_gene_expression_star_grid: {sorted(overlap)}."
            )

        import matplotlib.pyplot as plt

        pooled = self._require_fitted_result()
        values, source_label = self._scorer._gene_expression_values(
            pooled,
            str(gene),
            layer=layer,
            use_raw=use_raw,
            gene_symbols=gene_symbols,
            log1p=log1p,
        )
        condition_masks = [
            self._condition_display_mask(condition, population=population)
            for condition in self.conditions
        ]
        if shared_scale:
            union_mask = np.logical_or.reduce(condition_masks)
            shared_limits = self._scorer._continuous_color_limits(
                values,
                union_mask,
                vmin=vmin,
                vmax=vmax,
                percentile_range=percentile_range,
            )
            limits = [shared_limits] * len(self.conditions)
        else:
            limits = [
                self._scorer._continuous_color_limits(
                    values,
                    mask,
                    vmin=vmin,
                    vmax=vmax,
                    percentile_range=percentile_range,
                )
                for mask in condition_masks
            ]

        ncols = min(ncols, len(self.conditions))
        nrows = int(np.ceil(len(self.conditions) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )
        for condition, mask, limits_pair, axis in zip(
            self.conditions,
            condition_masks,
            limits,
            axes.ravel(),
        ):
            lower, upper = limits_pair
            self._scorer.plot_star(
                pooled,
                color_values=values,
                color_label=f"{gene} expression ({source_label})",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                sort_by_color=sort_cells,
                cell_mask=mask,
                title=str(condition),
                ax=axis,
                **plot_star_kwargs,
            )
        for axis in axes.ravel()[len(self.conditions) :]:
            axis.set_visible(False)
        fig.suptitle(
            f"{gene} expression by condition ({population})",
            y=1.01,
        )
        return fig

    def plot_rose_grid(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        population: str = "root",
        mode: str = "auto",
        n_bins: int = 24,
        ncols: int = 3,
        figsize_per_panel: tuple[float, float] = (6.0, 5.8),
        title: Optional[str] = None,
    ):
        """Plot condition-specific fate-directed velocity-mass rose plots."""
        self._resolve_results(results)
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")

        import matplotlib.pyplot as plt

        ncols = min(ncols, len(self.conditions))
        nrows = int(np.ceil(len(self.conditions) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            subplot_kw={"projection": "polar"},
            squeeze=False,
        )
        pooled = self._require_fitted_result()
        for condition, axis in zip(self.conditions, axes.ravel()):
            mask = self._condition_display_mask(condition, population=population)
            self._scorer.plot_rose(
                pooled,
                mask=mask,
                mode=mode,
                n_bins=n_bins,
                title=str(condition),
                ax=axis,
            )
        for axis in axes.ravel()[len(self.conditions) :]:
            axis.set_visible(False)
        fig.suptitle(
            title or f"Fate-directed branch-velocity mass by condition ({population})",
            y=1.02,
        )
        return fig

    def plot_compare_conditions_bar(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "mean_commitment_contribution",
        show_sem: bool = True,
        title: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        ax=None,
    ):
        """Compare replicate-level fate outcomes with grouped bars.

        Bars are means of biological-replicate summaries.  Error bars are the
        standard error across biological replicates, not across cells.
        """
        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        if metric not in _FATE_METRICS:
            raise ValueError(
                "metric must be mean_commitment_contribution, "
                "directional_affinity, or commitment_composition."
            )
        table = self.replicate_table(chosen)
        columns = [_outcome_column(metric, fate=fate) for fate in self.branches]
        means = table.groupby("condition", sort=False)[columns].mean().reindex(self.conditions)
        sem = table.groupby("condition", sort=False)[columns].sem().reindex(self.conditions)

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8.0, 5.0) if figsize is None else figsize)
        x = np.arange(len(self.conditions))
        width = 0.82 / len(self.branches)
        for fate_index, fate in enumerate(self.branches):
            offset = (fate_index - (len(self.branches) - 1) / 2.0) * width
            error = sem.iloc[:, fate_index].to_numpy(float) if show_sem else None
            ax.bar(
                x + offset,
                means.iloc[:, fate_index].to_numpy(float),
                width * 0.92,
                yerr=error,
                capsize=3 if show_sem else 0,
                label=fate,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(self.conditions)
        ax.set_ylabel(_metric_label(metric))
        ax.set_title(title or "Replicate-level scCS condition comparison")
        if metric in {"directional_affinity", "commitment_composition"}:
            ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        return ax.figure

    def plot_commitment_vector_radar(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "commitment_composition",
        title: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        ax=None,
    ):
        """Plot replicate-mean fate vectors on a polar radar axis.

        If a rectangular Matplotlib placeholder axis is supplied, scCS
        replaces it with a polar axis at the same subplot position.
        """
        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        if metric not in _FATE_METRICS:
            raise ValueError(
                "metric must be mean_commitment_contribution, "
                "directional_affinity, or commitment_composition."
            )
        table = self.replicate_table(chosen)
        columns = [_outcome_column(metric, fate=fate) for fate in self.branches]
        means = table.groupby("condition", sort=False)[columns].mean().reindex(self.conditions)

        import matplotlib.pyplot as plt

        if ax is None:
            figure = plt.figure(figsize=(7.0, 6.4) if figsize is None else figsize)
            ax = figure.add_subplot(111, projection="polar")
        elif getattr(ax, "name", "") != "polar":
            # Tutorial users commonly place the radar beside rectangular
            # panels created by ``plt.subplots``. Replace that placeholder in
            # the same figure rather than failing after a long analysis.
            figure = ax.figure
            subplot_spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
            bounds = ax.get_position().bounds
            ax.remove()
            if subplot_spec is not None:
                ax = figure.add_subplot(subplot_spec, projection="polar")
            else:
                ax = figure.add_axes(bounds, projection="polar")
        angles = np.linspace(0.0, 2.0 * np.pi, len(self.branches), endpoint=False)
        closed_angles = np.r_[angles, angles[0]]
        for condition in self.conditions:
            values = means.loc[condition].to_numpy(float)
            closed = np.r_[values, values[0]]
            ax.plot(closed_angles, closed, marker="o", linewidth=2.0, label=condition)
            ax.fill(closed_angles, closed, alpha=0.08)
        ax.set_xticks(angles)
        ax.set_xticklabels(self.branches)
        ax.set_title(title or _metric_label(metric), pad=18)
        ax.legend(frameon=False, bbox_to_anchor=(1.25, 1.10))
        return ax.figure

    def plot_trajectory_shift(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        ax=None,
    ):
        """Plot replicate-level progression velocity by condition.

        This is the v0.8 visualization corresponding to the familiar
        ``trajectory_shift`` name.  It does not test cell-level pseudotime
        distributions.
        """
        return self.plot_replicate_outcomes(
            results=results,
            metric="progression_velocity",
            ax=ax,
        )

    def _require_fitted_result(self):
        if not self._scorer.is_fitted or self._scorer.result is None:
            raise RuntimeError("Call build_embedding() and fit() before condition scoring.")
        return self._scorer.result

    def _resolve_results(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]],
    ) -> Mapping[str, ConditionCommitmentResult]:
        chosen = self._condition_results if results is None else results
        if chosen is None or len(chosen) == 0:
            raise RuntimeError("Call score_all_conditions() first.")
        return chosen
