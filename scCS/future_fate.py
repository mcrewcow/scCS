"""Discounted Future-Fate Propagation (DFFP) on an RNA-velocity graph.

This module implements the graph-based scoring mode selected during the scCS
v0.8 methodological benchmarks.  The scientific calculation remains in the
original velocity graph.  The supervised star is used for standardized display
and does not define the future-fate probabilities.

The main quantities are deliberately separate:

``future_fate_affinity``
    Conditional probability across the user-supplied supervised fates.
``future_fate_reach``
    Discounted probability of reaching any supervised fate anchor.
``future_fate_specificity``
    One minus normalized Shannon entropy of future-fate affinity.
``reach_supported_specificity``
    Reach multiplied by specificity.
``signed_progression``
    Expected change in the supplied ordering after conditioning on transitions
    retained inside the selected furcation.

External competing outcomes are optional and must be supplied explicitly.
Unmodelled futures remain in ``unresolved_probability`` rather than being
silently converted into an automatically inferred terminal fate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ._version import VERSION
from .affinity import CommitmentAffinityResult, normalized_entropy
from .furcation import Furcation
from .population import PopulationCommitmentSummary, summarize_commitment
from .projection import ProjectionResult
from .scoring_embedding import ScoringEmbeddingResult

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class TransitionNormalization:
    """Diagnostics from conversion to a float64 row-stochastic CSR matrix."""

    matrix: sparse.csr_matrix
    input_row_sum_min: float
    input_row_sum_max: float
    input_max_abs_row_sum_error: float
    output_max_abs_row_sum_error: float
    n_zero_rows: int
    max_negative_entry_magnitude: float


def canonicalize_transition_matrix(transition_matrix) -> TransitionNormalization:
    """Return a non-negative, float64, row-stochastic transition matrix.

    Sparse float32 accumulation may leave nominally normalized matrices a few
    ulps above or below one.  Materially negative entries still raise.  Empty
    rows receive self-loops so every row represents a valid Markov state.
    """
    matrix = sparse.csr_matrix(transition_matrix, dtype=float).copy()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square.")
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise ValueError("transition_matrix contains non-finite values.")

    negative = matrix.data[matrix.data < 0]
    max_negative = float(np.max(np.abs(negative))) if negative.size else 0.0
    if max_negative > 1e-10:
        raise ValueError(
            "transition_matrix contains materially negative entries; "
            f"maximum magnitude={max_negative:.3e}."
        )
    if matrix.data.size:
        matrix.data = np.clip(matrix.data, 0.0, None)
        matrix.eliminate_zeros()

    original_row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    input_min = float(np.min(original_row_sum))
    input_max = float(np.max(original_row_sum))
    input_error = float(np.max(np.abs(original_row_sum - 1.0)))
    zero_rows = original_row_sum <= _EPS
    n_zero_rows = int(np.sum(zero_rows))
    if n_zero_rows:
        matrix = matrix.tolil(copy=True)
        for index in np.flatnonzero(zero_rows):
            matrix.rows[index] = [int(index)]
            matrix.data[index] = [1.0]
        matrix = matrix.tocsr()

    row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    inverse = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > _EPS)
    matrix = (sparse.diags(inverse) @ matrix).tocsr()
    matrix.eliminate_zeros()

    # Remove residual sparse accumulation error one final time.
    row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    correction = np.divide(1.0, row_sum, out=np.ones_like(row_sum), where=row_sum > _EPS)
    matrix = (sparse.diags(correction) @ matrix).tocsr()
    output_row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    output_error = float(np.max(np.abs(output_row_sum - 1.0)))
    if output_error > 1e-12:
        raise ValueError("Could not produce a row-stochastic transition matrix.")

    return TransitionNormalization(
        matrix=matrix,
        input_row_sum_min=input_min,
        input_row_sum_max=input_max,
        input_max_abs_row_sum_error=input_error,
        output_max_abs_row_sum_error=output_error,
        n_zero_rows=n_zero_rows,
        max_negative_entry_magnitude=max_negative,
    )


def _stable_fractional_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    if len(order) <= 1:
        ranks[:] = 0.0
    else:
        ranks[order] = np.arange(len(order), dtype=float) / (len(order) - 1.0)
    return ranks


def _scale_progression(values: Sequence[float], mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("progression_values must be a finite non-empty 1D array.")
    mode = str(mode)
    if mode == "rank":
        return _stable_fractional_rank(array)
    if mode == "minmax":
        low = float(np.min(array))
        high = float(np.max(array))
        if high <= low:
            return np.zeros_like(array)
        return (array - low) / (high - low)
    if mode == "none":
        return array.copy()
    raise ValueError("progression_scale must be 'rank', 'minmax', or 'none'.")


def _selected_transition_blocks(
    full_transition: sparse.csr_matrix,
    selected_indices: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    retained = full_transition[selected_indices][:, selected_indices].tocsr()
    retained.data = np.clip(retained.data, 0.0, None)
    retained.eliminate_zeros()
    retained_mass = np.asarray(retained.sum(axis=1)).ravel().astype(float)
    retained_mass = np.clip(retained_mass, 0.0, 1.0)
    outside_mass = np.clip(1.0 - retained_mass, 0.0, 1.0)

    conditioned = retained.copy()
    defined = retained_mass > _EPS
    inverse = np.zeros_like(retained_mass)
    inverse[defined] = 1.0 / retained_mass[defined]
    conditioned = (sparse.diags(inverse) @ conditioned).tocsr()
    if np.any(~defined):
        conditioned = conditioned.tolil(copy=True)
        for index in np.flatnonzero(~defined):
            conditioned.rows[index] = [int(index)]
            conditioned.data[index] = [1.0]
        conditioned = conditioned.tocsr()
    if not np.allclose(np.asarray(conditioned.sum(axis=1)).ravel(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("Could not condition selected-path transitions.")
    return retained, conditioned, retained_mass, outside_mass


def expected_ordering_change(
    transition_matrix,
    ordering: Sequence[float],
) -> np.ndarray:
    """Return expected signed one-step change in a supplied ordering."""
    matrix = sparse.csr_matrix(transition_matrix, dtype=float)
    values = np.asarray(ordering, dtype=float)
    if matrix.shape != (len(values), len(values)):
        raise ValueError("ordering length must match transition_matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError("ordering must be finite.")
    row_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if not np.allclose(row_sum, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("transition_matrix rows must sum to one.")
    return np.asarray(matrix @ values).ravel() - values


def choose_endpoint_anchors(
    embedding: ScoringEmbeddingResult,
    *,
    quantile: float = 0.90,
    min_cells: int = 10,
    ordering_values: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Choose late anchors independently within each annotated terminal fate."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("anchor_quantile must lie strictly between 0 and 1.")
    if min_cells < 1:
        raise ValueError("min_anchor_cells must be positive.")
    values = (
        embedding.selected_ordering_values
        if ordering_values is None
        else np.asarray(ordering_values, dtype=float)
    )
    values = np.asarray(values, dtype=float)
    if values.shape != (embedding.n_selected,) or not np.all(np.isfinite(values)):
        raise ValueError("anchor ordering must be finite and aligned to selected cells.")

    anchors = np.zeros((embedding.n_selected, len(embedding.geometry.fate_names)), dtype=bool)
    terminal_names = np.asarray(embedding.terminal_names).astype(str)
    for fate_index, fate in enumerate(embedding.geometry.fate_names):
        candidates = np.flatnonzero(embedding.terminal_mask & (terminal_names == fate))
        if candidates.size == 0:
            raise ValueError(f"No selected cells are assigned to terminal fate {fate!r}.")
        threshold = float(np.quantile(values[candidates], quantile))
        chosen = candidates[values[candidates] >= threshold]
        target = min(int(min_cells), int(candidates.size))
        if chosen.size < target:
            order = np.argsort(values[candidates], kind="mergesort")
            chosen = candidates[order[-target:]]
        anchors[chosen, fate_index] = True
    return anchors


def _coerce_anchor_mask(value, n_states: int, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == bool:
        if array.ndim != 1 or len(array) != n_states:
            raise ValueError(f"Boolean anchors for {name!r} must have length {n_states}.")
        mask = array.copy()
    else:
        indices = np.asarray(value, dtype=int)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"Anchor indices for {name!r} must be a non-empty 1D sequence.")
        if np.any(indices < 0) or np.any(indices >= n_states):
            raise IndexError(f"Anchor indices for {name!r} fall outside the transition matrix.")
        mask = np.zeros(n_states, dtype=bool)
        mask[indices] = True
    if not np.any(mask):
        raise ValueError(f"Competing outcome {name!r} has no anchor cells.")
    return mask


def build_outcome_anchors(
    n_states: int,
    selected_indices: Sequence[int],
    selected_anchors: np.ndarray,
    fate_names: Sequence[str],
    *,
    competing_outcomes: Optional[Mapping[str, Sequence[int] | np.ndarray]] = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build full-state anchor columns for selected and optional competing outcomes."""
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_anchors = np.asarray(selected_anchors, dtype=bool)
    if selected_anchors.shape != (len(selected_indices), len(fate_names)):
        raise ValueError("selected_anchors shape does not match selected cells and fates.")
    anchors = np.zeros((n_states, len(fate_names)), dtype=bool)
    anchors[selected_indices] = selected_anchors
    names = [str(name) for name in fate_names]

    extra_columns = []
    if competing_outcomes:
        occupied = anchors.any(axis=1)
        for raw_name, value in competing_outcomes.items():
            name = str(raw_name)
            if name in names:
                raise ValueError(f"Competing outcome name {name!r} duplicates a supervised fate.")
            mask = _coerce_anchor_mask(value, n_states, name=name)
            overlap = occupied & mask
            if np.any(overlap):
                count = int(overlap.sum())
                raise ValueError(
                    f"Competing outcome {name!r} overlaps {count} existing anchor cells."
                )
            occupied |= mask
            names.append(name)
            extra_columns.append(mask)
    if extra_columns:
        anchors = np.column_stack([anchors, *extra_columns])
    return tuple(names), anchors


def _make_absorbing(transition: sparse.csr_matrix, anchors: np.ndarray) -> sparse.csr_matrix:
    absorbing = np.asarray(anchors, dtype=bool).any(axis=1)
    output = transition.tolil(copy=True)
    for index in np.flatnonzero(absorbing):
        output.rows[index] = [int(index)]
        output.data[index] = [1.0]
    return output.tocsr()


@dataclass(frozen=True)
class DiscountedOutcomeSolution:
    """Discounted multi-outcome hitting probabilities and solver diagnostics."""

    outcome_names: tuple[str, ...]
    probability: np.ndarray
    unresolved_probability: np.ndarray
    anchors: np.ndarray
    effective_horizon: int
    gamma: float
    solver: str
    iterations: int
    converged: bool
    residual: float


def solve_discounted_outcomes(
    transition_matrix,
    anchors: np.ndarray,
    outcome_names: Sequence[str],
    *,
    effective_horizon: int = 64,
    solver: str = "auto",
    direct_max_states: int = 50_000,
    tolerance: float = 1e-10,
    max_iter: int = 20_000,
) -> DiscountedOutcomeSolution:
    """Solve geometrically discounted hitting probabilities.

    Before each transition the process stops with probability ``1-gamma``,
    where ``gamma = h / (h + 1)`` and ``h`` is ``effective_horizon``.  The
    unresolved probability is the chance of stopping before any supplied
    outcome anchor is reached.

    ``solver='auto'`` uses sparse LU on modest graphs and fixed-point iteration
    on larger graphs.  The iterative solver uses only sparse matrix products
    and therefore avoids dense state-by-state objects.
    """
    if int(effective_horizon) <= 0:
        raise ValueError("effective_horizon must be a positive integer.")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")

    transition = sparse.csr_matrix(transition_matrix, dtype=float)
    anchors = np.asarray(anchors, dtype=bool)
    if transition.shape[0] != transition.shape[1]:
        raise ValueError("transition_matrix must be square.")
    if anchors.ndim != 2 or anchors.shape[0] != transition.shape[0]:
        raise ValueError("anchors must be state-by-outcome and match transition_matrix.")
    if anchors.shape[1] != len(outcome_names):
        raise ValueError("outcome_names and anchor columns do not match.")
    if np.any(anchors.sum(axis=1) > 1):
        raise ValueError("A state cannot anchor more than one outcome.")
    if not np.all(anchors.any(axis=0)):
        missing = [
            str(name) for name, present in zip(outcome_names, anchors.any(axis=0)) if not present
        ]
        raise ValueError(f"Outcomes without anchors: {missing}.")

    row_sum = np.asarray(transition.sum(axis=1)).ravel().astype(float)
    if not np.allclose(row_sum, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("transition_matrix rows must sum to one.")

    absorbing_transition = _make_absorbing(transition, anchors)
    absorbing = anchors.any(axis=1)
    transient_indices = np.flatnonzero(~absorbing)
    gamma = float(effective_horizon / (effective_horizon + 1.0))
    requested = str(solver)
    if requested not in {"auto", "direct", "iterative"}:
        raise ValueError("solver must be 'auto', 'direct', or 'iterative'.")
    chosen = requested
    if chosen == "auto":
        chosen = "direct" if len(transient_indices) <= int(direct_max_states) else "iterative"

    values = anchors.astype(float)
    iterations = 0
    converged = True
    residual = 0.0

    if transient_indices.size and chosen == "direct":
        q = absorbing_transition[transient_indices][:, transient_indices].tocsr()
        rhs = gamma * np.asarray(absorbing_transition[transient_indices] @ anchors.astype(float))
        system = sparse.eye(len(transient_indices), format="csc") - gamma * q.tocsc()
        solved = np.asarray(splu(system).solve(rhs), dtype=float)
        values[transient_indices] = np.clip(solved, 0.0, 1.0)
        residual_array = np.asarray(system @ solved - rhs)
        residual = float(np.max(np.abs(residual_array))) if residual_array.size else 0.0
        iterations = 1
    elif transient_indices.size:
        # Full-state fixed-point iteration keeps absorbing boundary values exact.
        current = anchors.astype(float)
        converged = False
        for iteration in range(1, int(max_iter) + 1):
            updated = gamma * np.asarray(absorbing_transition @ current, dtype=float)
            updated[absorbing] = anchors[absorbing].astype(float)
            delta = float(np.max(np.abs(updated - current)))
            current = updated
            if delta <= tolerance:
                converged = True
                iterations = iteration
                break
        if not converged:
            iterations = int(max_iter)
            warnings.warn(
                "Discounted future-fate fixed-point solver did not reach the requested "
                f"tolerance after {max_iter} iterations; residual={delta:.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        values = np.clip(current, 0.0, 1.0)
        check = values.copy()
        updated = gamma * np.asarray(absorbing_transition @ check, dtype=float)
        updated[absorbing] = anchors[absorbing].astype(float)
        residual = float(np.max(np.abs(updated - check)))

    unresolved = np.clip(1.0 - values.sum(axis=1), 0.0, 1.0)
    conservation = np.max(np.abs(values.sum(axis=1) + unresolved - 1.0))
    if conservation > 1e-9:
        raise RuntimeError(f"Discounted probability conservation failed: {conservation:.3e}.")

    return DiscountedOutcomeSolution(
        outcome_names=tuple(str(name) for name in outcome_names),
        probability=values,
        unresolved_probability=unresolved,
        anchors=anchors,
        effective_horizon=int(effective_horizon),
        gamma=gamma,
        solver=chosen,
        iterations=int(iterations),
        converged=bool(converged),
        residual=float(residual),
    )


def _conditional_affinity(
    selected_probability: np.ndarray,
    min_reach: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reach = np.asarray(selected_probability, dtype=float).sum(axis=1)
    defined = reach >= float(min_reach)
    k = selected_probability.shape[1]
    affinity = np.full_like(selected_probability, 1.0 / k)
    affinity[defined] = selected_probability[defined] / reach[defined, None]
    return reach, affinity, defined


def _future_status(
    reach: np.ndarray,
    specificity: np.ndarray,
    defined: np.ndarray,
    *,
    committed_reach_threshold: float,
    committed_specificity_threshold: float,
) -> np.ndarray:
    status = np.full(len(reach), "unresolved_future", dtype=object)
    status[defined] = "future_ambiguous"
    low_reach = defined & (reach < committed_reach_threshold)
    status[low_reach] = "low_future_reach"
    committed = (
        defined
        & (reach >= committed_reach_threshold)
        & (specificity >= committed_specificity_threshold)
    )
    status[committed] = "future_fate_committed"
    return status


@dataclass(frozen=True)
class FutureFateScoreResult:
    """Cell-level discounted future-fate result for one supervised furcation."""

    furcation: Furcation
    embedding: ScoringEmbeddingResult
    projection: ProjectionResult
    commitment: CommitmentAffinityResult
    selected_probability: np.ndarray
    competing_probability: np.ndarray
    unresolved_probability: np.ndarray
    selected_reach: np.ndarray
    competing_reach: np.ndarray
    progression_velocity: np.ndarray
    supported_progression_flux: np.ndarray
    one_step_outside_probability: np.ndarray
    outcome_names: tuple[str, ...]
    selected_anchor_mask: np.ndarray
    outcome_anchor_mask: np.ndarray
    effective_horizon: int
    solver: str
    solver_iterations: int
    solver_converged: bool
    solver_residual: float
    min_reach: float
    anchor_quantile: float
    min_anchor_cells: int
    progression_scale: str
    status: np.ndarray
    dominant_fate: np.ndarray
    root_population_summary: PopulationCommitmentSummary
    transition_normalization: TransitionNormalization
    anchor_diagnostics: tuple[dict[str, Any], ...]
    scoring_mode: str = "future_fate"
    magnitude_scaler: Any = None
    magnitude_fit_population: str = "not_applicable"

    @property
    def cell_ids(self) -> np.ndarray:
        return self.embedding.selected_cell_ids

    @property
    def fate_names(self) -> tuple[str, ...]:
        return self.embedding.geometry.fate_names

    @property
    def k(self) -> int:
        return len(self.fate_names)

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_root(self) -> int:
        return int(np.sum(self.root_mask))

    @property
    def n_terminal(self) -> int:
        return int(np.sum(self.terminal_mask))

    @property
    def root_mask(self) -> np.ndarray:
        return self.embedding.root_mask

    @property
    def terminal_mask(self) -> np.ndarray:
        return self.embedding.terminal_mask

    @property
    def directional_affinity(self) -> np.ndarray:
        """Compatibility alias for :attr:`future_fate_affinity`."""
        return self.commitment.directional_affinity

    @property
    def future_fate_affinity(self) -> np.ndarray:
        return self.commitment.directional_affinity

    @property
    def conditional_fate_affinity(self) -> np.ndarray:
        """Conditional Fate Affinity (CFA).

        This documentation-level name emphasizes that the distribution is
        normalized across supervised fates after conditioning on selected-fate
        reach. It is an exact read-only alias of
        :attr:`future_fate_affinity`.
        """
        return self.future_fate_affinity

    @property
    def commitment_strength(self) -> np.ndarray:
        """Compatibility alias for :attr:`future_fate_reach`."""
        return self.selected_reach

    @property
    def future_fate_reach(self) -> np.ndarray:
        return self.selected_reach

    @property
    def discounted_fate_reach(self) -> np.ndarray:
        """Discounted Fate Reach (DFR), an alias of future-fate reach."""
        return self.future_fate_reach

    @property
    def signed_progression(self) -> np.ndarray:
        """Signed expected change in the supplied progression coordinate.

        This is the public semantic alias for :attr:`progression_velocity`.
        Positive values indicate movement toward larger ordering values; negative
        values preserve retrograde or loop-returning dynamics.
        """
        return self.progression_velocity

    @property
    def signed_ordering_flux(self) -> np.ndarray:
        """Signed Ordering Flux (SOF), an alias of signed progression."""
        return self.signed_progression

    @property
    def selected_path_coverage(self) -> np.ndarray:
        """One-step transition mass retained within the supervised cell set."""
        return self.projection.transition_coverage

    @property
    def commitment_contribution(self) -> np.ndarray:
        """Discounted selected-fate probability allocated across fates."""
        return self.commitment.commitment_contribution

    @property
    def future_fate_contribution(self) -> np.ndarray:
        return self.commitment.commitment_contribution

    @property
    def directional_entropy(self) -> np.ndarray:
        return self.commitment.directional_entropy

    @property
    def future_fate_entropy(self) -> np.ndarray:
        return self.commitment.directional_entropy

    @property
    def commitment_entropy(self) -> np.ndarray:
        return self.commitment.commitment_entropy

    @property
    def aligned_directional_entropy(self) -> float:
        return 0.0

    @property
    def directional_specificity(self) -> np.ndarray:
        return self.commitment.directional_specificity

    @property
    def future_fate_specificity(self) -> np.ndarray:
        return self.commitment.directional_specificity

    @property
    def specific_commitment(self) -> np.ndarray:
        return self.commitment.specific_commitment

    @property
    def reach_supported_specificity(self) -> np.ndarray:
        return self.commitment.specific_commitment

    @property
    def resolved_commitment(self) -> np.ndarray:
        """Resolved Commitment (RC), reach multiplied by fate specificity."""
        return self.reach_supported_specificity

    @property
    def unresolved_future_probability(self) -> np.ndarray:
        """Unresolved Future Probability (UFP)."""
        return self.unresolved_probability

    @property
    def nearest_fate_angle_degrees(self) -> np.ndarray:
        return np.full(self.n_cells, np.nan, dtype=float)

    @property
    def fate_cosine_similarity(self) -> np.ndarray:
        return np.full((self.n_cells, self.k), np.nan, dtype=float)

    @property
    def projected_velocity(self):
        raise RuntimeError(
            "future_fate mode does not project a scientific velocity vector into the star. "
            "Use signed progression and future-fate probabilities instead."
        )

    @property
    def branch_velocity(self):
        raise RuntimeError(
            "future_fate mode has no branch-velocity vector. The star is display-only."
        )

    def summarize(self, mask: Optional[np.ndarray] = None) -> PopulationCommitmentSummary:
        if mask is None:
            mask = self.root_mask
        values = np.asarray(mask)
        if values.dtype != bool or values.ndim != 1 or len(values) != self.n_cells:
            raise ValueError("mask must be Boolean and aligned to selected cells.")
        if not values.any():
            raise ValueError("Cannot summarize an empty population.")
        return summarize_commitment(self.commitment_contribution[values])

    def summary(self) -> str:
        root = self.root_mask
        composition = self.root_population_summary.commitment_composition
        if self.root_population_summary.composition_defined:
            composition_text = ", ".join(
                f"{name}={value:.3f}" for name, value in zip(self.fate_names, composition)
            )
        else:
            composition_text = "undefined (zero selected-fate reach)"
        gamma = self.effective_horizon / (self.effective_horizon + 1)
        coverage = float(np.mean(self.projection.velocity_defined[root]))
        reach = float(np.mean(self.selected_reach[root]))
        entropy = float(np.mean(self.directional_entropy[root]))
        specificity = float(np.mean(self.directional_specificity[root]))
        supported = float(np.mean(self.specific_commitment[root]))
        unresolved = float(np.mean(self.unresolved_probability[root]))
        progression = float(np.mean(self.progression_velocity[root]))
        return (
            "scCS FutureFateScoreResult\n"
            f"  Furcation: {self.furcation.root_name} -> {list(self.fate_names)}\n"
            f"  Effective horizon: {self.effective_horizon} (gamma={gamma:.6f})\n"
            f"  Cells: {self.n_cells} selected; {self.n_root} root\n"
            f"  Root affinity coverage: {coverage:.3f}\n"
            f"  Root mean future-fate reach: {reach:.3f}\n"
            f"  Root mean future-fate entropy: {entropy:.3f}\n"
            f"  Root mean future-fate specificity: {specificity:.3f}\n"
            f"  Root mean reach-supported specificity: {supported:.3f}\n"
            f"  Root mean unresolved probability: {unresolved:.3f}\n"
            f"  Root mean signed progression: {progression:.3f}\n"
            f"  Root future-fate composition: {composition_text}\n"
            f"  Solver: {self.solver}; iterations={self.solver_iterations}; "
            f"residual={self.solver_residual:.3e}"
        )

    def anchor_diagnostics_frame(self):
        """Return per-fate anchor diagnostics as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([dict(row) for row in self.anchor_diagnostics])

    def write_to_adata(self, adata, *, metadata_key: str = "sccs_v08") -> None:
        """Write full-length future-fate outputs and reproducibility metadata."""
        self.embedding.write_to_adata(adata, metadata_key=metadata_key)
        n_obs = adata.n_obs
        selected = self.embedding.selected_indices

        def full_vector(values, *, fill=np.nan, dtype=float):
            output = np.full(n_obs, fill, dtype=dtype)
            output[selected] = values
            return output

        def full_matrix(values, width):
            output = np.full((n_obs, width), np.nan, dtype=float)
            output[selected] = values
            return output

        adata.obsm["sccs_future_fate_probability"] = full_matrix(self.selected_probability, self.k)
        adata.obsm["sccs_future_fate_affinity"] = full_matrix(self.future_fate_affinity, self.k)
        adata.obsm["sccs_future_fate_contribution"] = full_matrix(
            self.future_fate_contribution, self.k
        )
        if self.competing_probability.shape[1]:
            adata.obsm["sccs_competing_outcome_probability"] = full_matrix(
                self.competing_probability, self.competing_probability.shape[1]
            )
        adata.obs["sccs_future_fate_reach"] = full_vector(self.selected_reach)
        adata.obs["sccs_competing_fate_reach"] = full_vector(self.competing_reach)
        adata.obs["sccs_unresolved_probability"] = full_vector(self.unresolved_probability)
        adata.obs["sccs_future_fate_entropy"] = full_vector(self.future_fate_entropy)
        adata.obs["sccs_future_fate_specificity"] = full_vector(self.future_fate_specificity)
        adata.obs["sccs_reach_supported_specificity"] = full_vector(
            self.reach_supported_specificity
        )
        adata.obs["sccs_signed_progression"] = full_vector(self.progression_velocity)
        adata.obs["sccs_supported_progression_flux"] = full_vector(self.supported_progression_flux)
        adata.obs["sccs_selected_path_coverage"] = full_vector(self.projection.transition_coverage)
        adata.obs["sccs_future_fate_status"] = full_vector(self.status, fill="", dtype=object)
        adata.obs["sccs_future_dominant_fate"] = full_vector(
            self.dominant_fate, fill="", dtype=object
        )

        metadata = dict(adata.uns.get(metadata_key, {}))
        metadata["version"] = VERSION
        metadata["scoring_mode"] = "future_fate"
        metadata["future_fate"] = {
            "effective_horizon": int(self.effective_horizon),
            "solver": self.solver,
            "solver_iterations": int(self.solver_iterations),
            "solver_converged": bool(self.solver_converged),
            "solver_residual": float(self.solver_residual),
            "min_reach": float(self.min_reach),
            "anchor_quantile": float(self.anchor_quantile),
            "min_anchor_cells": int(self.min_anchor_cells),
            "progression_scale": self.progression_scale,
            "fate_names": list(self.fate_names),
            "outcome_names": list(self.outcome_names),
            "anchor_diagnostics": {
                str(row["fate"]): {key: value for key, value in row.items() if key != "fate"}
                for row in self.anchor_diagnostics
            },
            "transition_normalization": {
                key: value
                for key, value in asdict(self.transition_normalization).items()
                if key != "matrix"
            },
        }
        adata.uns[metadata_key] = metadata


def _anchor_diagnostics(
    full_transition: sparse.csr_matrix,
    embedding: ScoringEmbeddingResult,
    selected_anchors: np.ndarray,
    progression: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    selected_indices = embedding.selected_indices
    selected_full = np.zeros(full_transition.shape[0], dtype=bool)
    selected_full[selected_indices] = True
    root_full = np.zeros_like(selected_full)
    root_full[selected_indices[embedding.root_mask]] = True
    terminal_names = np.asarray(embedding.terminal_names).astype(str)
    rows: list[dict[str, Any]] = []
    for fate_index, fate in enumerate(embedding.geometry.fate_names):
        local_anchor = selected_anchors[:, fate_index]
        full_anchor = selected_indices[local_anchor]
        same_full = np.zeros_like(selected_full)
        same_full[selected_indices[embedding.terminal_mask & (terminal_names == fate)]] = True
        other_full = selected_full & ~root_full & ~same_full
        block = full_transition[full_anchor]
        rows.append(
            {
                "fate": fate,
                "n_anchors": int(local_anchor.sum()),
                "mean_transition_to_same_fate_population": float(
                    np.mean(np.asarray(block[:, same_full].sum(axis=1)).ravel())
                ),
                "mean_transition_to_root": float(
                    np.mean(np.asarray(block[:, root_full].sum(axis=1)).ravel())
                ),
                "mean_transition_to_other_selected_fates": float(
                    np.mean(np.asarray(block[:, other_full].sum(axis=1)).ravel())
                ),
                "mean_transition_outside_selected_path": float(
                    np.mean(np.asarray(block[:, ~selected_full].sum(axis=1)).ravel())
                ),
                "mean_signed_progression": float(np.mean(progression[local_anchor])),
                "forward_fraction": float(np.mean(progression[local_anchor] > 0)),
            }
        )
    return tuple(rows)


def score_future_fate(
    furcation: Furcation,
    embedding: ScoringEmbeddingResult,
    transition_matrix,
    *,
    effective_horizon: int = 64,
    anchor_quantile: float = 0.90,
    min_anchor_cells: int = 10,
    competing_outcomes: Optional[Mapping[str, Sequence[int] | np.ndarray]] = None,
    min_reach: float = 1e-6,
    progression_values: Optional[Sequence[float]] = None,
    progression_scale: str = "rank",
    solver: str = "auto",
    direct_max_states: int = 50_000,
    tolerance: float = 1e-10,
    max_iter: int = 20_000,
    committed_reach_threshold: float = 0.25,
    committed_specificity_threshold: float = 0.25,
) -> FutureFateScoreResult:
    """Score discounted future-fate identity and signed progression.

    The transition graph must cover the full AnnData object.  Selected fate
    anchors are chosen from the late ordering tail of each annotated terminal
    population.  Optional competing outcomes must be supplied explicitly as
    full-length Boolean masks or full-state index sequences.
    """
    if not 0 <= min_reach <= 1:
        raise ValueError("min_reach must lie in [0, 1].")
    normalization = canonicalize_transition_matrix(transition_matrix)
    full_transition = normalization.matrix
    if full_transition.shape[0] <= int(np.max(embedding.selected_indices)):
        raise ValueError("transition_matrix does not cover all selected furcation cells.")

    selected_indices = np.asarray(embedding.selected_indices, dtype=int)
    retained, conditioned, retained_mass, outside_mass = _selected_transition_blocks(
        full_transition, selected_indices
    )

    if progression_values is None:
        selected_progression_values = embedding.selected_ordering_values
    else:
        supplied_progression = np.asarray(progression_values, dtype=float)
        if supplied_progression.shape == (embedding.n_selected,):
            selected_progression_values = supplied_progression
        elif supplied_progression.shape == (full_transition.shape[0],):
            selected_progression_values = supplied_progression[selected_indices]
        else:
            raise ValueError(
                "progression_values must be aligned to selected cells or the full transition graph."
            )
    if not np.all(np.isfinite(selected_progression_values)):
        raise ValueError("progression_values contains non-finite selected-cell values.")
    progression_coordinate = _scale_progression(selected_progression_values, progression_scale)
    progression = expected_ordering_change(conditioned, progression_coordinate)
    supported_progression = np.asarray(retained @ progression_coordinate).ravel() - (
        retained_mass * progression_coordinate
    )

    selected_anchors = choose_endpoint_anchors(
        embedding,
        quantile=anchor_quantile,
        min_cells=min_anchor_cells,
        ordering_values=selected_progression_values,
    )
    outcome_names, outcome_anchors = build_outcome_anchors(
        full_transition.shape[0],
        selected_indices,
        selected_anchors,
        embedding.geometry.fate_names,
        competing_outcomes=competing_outcomes,
    )
    solution = solve_discounted_outcomes(
        full_transition,
        outcome_anchors,
        outcome_names,
        effective_horizon=effective_horizon,
        solver=solver,
        direct_max_states=direct_max_states,
        tolerance=tolerance,
        max_iter=max_iter,
    )

    k = len(embedding.geometry.fate_names)
    selected_probability_full = solution.probability[:, :k]
    competing_probability_full = solution.probability[:, k:]
    selected_probability = selected_probability_full[selected_indices]
    selected_reach, affinity, defined = _conditional_affinity(selected_probability, min_reach)
    competing_reach_full = (
        competing_probability_full.sum(axis=1)
        if competing_probability_full.shape[1]
        else np.zeros(full_transition.shape[0], dtype=float)
    )
    competing_reach = competing_reach_full[selected_indices]

    entropy = normalized_entropy(affinity)
    specificity = np.clip(1.0 - entropy, 0.0, 1.0)
    uniform = np.full_like(affinity, 1.0 / k)
    commitment_affinity = (1.0 - selected_reach[:, None]) * uniform + (
        selected_reach[:, None] * affinity
    )
    commitment_entropy = normalized_entropy(commitment_affinity)
    contribution = selected_reach[:, None] * affinity
    specific_commitment = selected_reach * specificity
    commitment = CommitmentAffinityResult(
        directional_affinity=affinity,
        velocity_magnitude=selected_reach.copy(),
        commitment_strength=selected_reach.copy(),
        commitment_affinity=commitment_affinity,
        commitment_contribution=contribution,
        directional_entropy=entropy,
        commitment_entropy=commitment_entropy,
        directional_specificity=specificity,
        specific_commitment=specific_commitment,
        velocity_defined=defined,
    )

    projection = ProjectionResult(
        velocity=np.full((embedding.n_selected, embedding.dimension), np.nan, dtype=float),
        retained_transition_mass=retained_mass,
        external_transition_mass=outside_mass,
        transition_coverage=retained_mass,
        velocity_defined=defined,
        selected_indices=selected_indices.copy(),
        renormalized=False,
        min_coverage=float(min_reach),
    )
    status = _future_status(
        selected_reach,
        specificity,
        defined,
        committed_reach_threshold=committed_reach_threshold,
        committed_specificity_threshold=committed_specificity_threshold,
    )
    dominant_index = np.argmax(affinity, axis=1)
    dominant_fate = np.asarray(embedding.geometry.fate_names, dtype=object)[dominant_index]
    dominant_fate[status != "future_fate_committed"] = ""
    root_summary = summarize_commitment(contribution[embedding.root_mask])
    diagnostics = _anchor_diagnostics(
        full_transition,
        embedding,
        selected_anchors,
        progression,
    )

    return FutureFateScoreResult(
        furcation=furcation,
        embedding=embedding,
        projection=projection,
        commitment=commitment,
        selected_probability=selected_probability,
        competing_probability=competing_probability_full[selected_indices],
        unresolved_probability=solution.unresolved_probability[selected_indices],
        selected_reach=selected_reach,
        competing_reach=competing_reach,
        progression_velocity=progression,
        supported_progression_flux=supported_progression,
        one_step_outside_probability=outside_mass,
        outcome_names=solution.outcome_names,
        selected_anchor_mask=selected_anchors,
        outcome_anchor_mask=solution.anchors,
        effective_horizon=int(effective_horizon),
        solver=solution.solver,
        solver_iterations=solution.iterations,
        solver_converged=solution.converged,
        solver_residual=solution.residual,
        min_reach=float(min_reach),
        anchor_quantile=float(anchor_quantile),
        min_anchor_cells=int(min_anchor_cells),
        progression_scale=str(progression_scale),
        status=status,
        dominant_fate=dominant_fate,
        root_population_summary=root_summary,
        transition_normalization=normalization,
        anchor_diagnostics=diagnostics,
    )
