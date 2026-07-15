"""Direct transition-based velocity projection for scCS v0.8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class RootProjectionGeometryDiagnostics:
    """Root-cell checks that the transition projection preserves fate direction.

    For root cells, scientific coordinates have no component in the terminal
    simplex subspace.  Their projected fate-directed velocity must therefore
    equal the retained transition mass to each terminal population multiplied
    by the corresponding ideal terminal direction.  This diagnostic
    reconstructs that quantity directly from the transition matrix and
    compares it with the stored scCS branch velocity and directional affinity.
    """

    root_local_indices: np.ndarray
    terminal_transition_mass: np.ndarray
    reconstructed_branch_velocity: np.ndarray
    stored_branch_velocity: np.ndarray
    reconstructed_directional_affinity: np.ndarray
    stored_directional_affinity: np.ndarray
    defined_mask: np.ndarray
    informative_mask: np.ndarray
    decisive_mask: np.ndarray
    max_abs_branch_error: float
    branch_rmse: float
    max_abs_affinity_error: float
    median_direction_cosine: float
    dominant_fate_agreement: float

    @property
    def n_root_cells(self) -> int:
        return int(len(self.root_local_indices))

    @property
    def n_informative_cells(self) -> int:
        return int(np.sum(self.informative_mask))

    @property
    def n_decisive_cells(self) -> int:
        return int(np.sum(self.decisive_mask))

    def summary(self) -> dict[str, float | int]:
        """Return scalar diagnostics suitable for a notebook table."""
        return {
            "n_root_cells": self.n_root_cells,
            "n_informative_terminal_transition_cells": self.n_informative_cells,
            "n_decisive_terminal_transition_cells": self.n_decisive_cells,
            "max_abs_branch_reconstruction_error": self.max_abs_branch_error,
            "branch_reconstruction_rmse": self.branch_rmse,
            "max_abs_affinity_reconstruction_error": self.max_abs_affinity_error,
            "median_reconstructed_vs_stored_direction_cosine": self.median_direction_cosine,
            "dominant_terminal_mass_vs_affinity_agreement": self.dominant_fate_agreement,
        }


@dataclass(frozen=True)
class RootProgressionDirectionDiagnostics:
    """Compare root progression implied by ordering, projection, and display QC.

    The scientific incoming arm is oriented so that increasing root progress
    moves cells toward the central furcation.  For root cells, the direct
    transition-weighted change in a progression coordinate that assigns
    terminal cells progress 1 must therefore match the stored scientific
    progression velocity up to ``arm_scale``.  The optional scVelo-centered
    display projection is reported separately because its normalized and
    baseline-centered embedding formula can reverse arrows on a nearly
    collinear star even when the direct transition displacement is forward.
    """

    root_local_indices: np.ndarray
    root_progress: np.ndarray
    expected_progress_change: np.ndarray
    scientific_progression_velocity: np.ndarray
    transition_display_centerward_velocity: np.ndarray
    scvelo_display_centerward_velocity: np.ndarray
    selected_root_local_index: int
    selected_root_progress: float
    selected_root_radius: float
    max_abs_progression_identity_error: float
    forward_expected_progress_fraction: float
    forward_scientific_fraction: float
    forward_transition_display_fraction: float
    forward_scvelo_display_fraction: float

    def summary(self) -> dict[str, float | int]:
        """Return scalar diagnostics suitable for display in a notebook."""
        return {
            "n_root_cells": int(len(self.root_local_indices)),
            "selected_root_local_index": int(self.selected_root_local_index),
            "selected_root_progress": float(self.selected_root_progress),
            "selected_root_radius": float(self.selected_root_radius),
            "max_abs_expected_progress_vs_scientific_error": float(
                self.max_abs_progression_identity_error
            ),
            "forward_expected_progress_fraction": float(self.forward_expected_progress_fraction),
            "forward_scientific_progression_fraction": float(self.forward_scientific_fraction),
            "forward_transition_display_fraction": float(self.forward_transition_display_fraction),
            "forward_scvelo_centered_display_fraction": float(self.forward_scvelo_display_fraction),
            "mean_expected_progress_change": float(np.nanmean(self.expected_progress_change)),
            "mean_scientific_progression_velocity": float(
                np.nanmean(self.scientific_progression_velocity)
            ),
            "mean_transition_display_centerward_velocity": float(
                np.nanmean(self.transition_display_centerward_velocity)
            ),
            "mean_scvelo_centered_display_velocity": float(
                np.nanmean(self.scvelo_display_centerward_velocity)
            ),
        }


@dataclass(frozen=True)
class ProjectionResult:
    """Projected displacement and transition-coverage diagnostics."""

    velocity: np.ndarray
    retained_transition_mass: np.ndarray
    external_transition_mass: np.ndarray
    transition_coverage: np.ndarray
    velocity_defined: np.ndarray
    selected_indices: np.ndarray
    renormalized: bool
    min_coverage: float


def _row_sums(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    return np.asarray(matrix, dtype=float).sum(axis=1)


def project_transition_velocity(
    transition_matrix,
    selected_coordinates: np.ndarray,
    *,
    selected_indices: Optional[Sequence[int]] = None,
    renormalize_retained: bool = True,
    min_coverage: float = 0.05,
) -> ProjectionResult:
    """Project directed transitions into a selected scCS scoring geometry.

    Parameters
    ----------
    transition_matrix
        Square dense or scipy-sparse directed transition matrix.  It may cover
        the full dataset or only the selected furcation cells.
    selected_coordinates
        Scientific coordinates for selected furcation cells, shape
        ``(n_selected, dimension)``.
    selected_indices
        Indices of selected cells in the full transition matrix.  Omit when
        the transition matrix already contains exactly the selected cells.
    renormalize_retained
        If true, normalize transition mass retained within the furcation before
        computing expected displacement.
    min_coverage
        Minimum fraction of outgoing transition mass retained within the
        selected furcation.  Lower-coverage velocities are returned as NaN and
        marked undefined.

    Notes
    -----
    Excluded cells are never assigned artificial coordinates.  Their outgoing
    transition mass is reported as external mass instead of pulling projected
    vectors toward an arbitrary origin.
    """
    coordinates = np.asarray(selected_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0:
        raise ValueError("selected_coordinates must be a non-empty two-dimensional array.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("selected_coordinates contains non-finite values.")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must lie in [0, 1].")

    matrix = transition_matrix
    if not sparse.issparse(matrix):
        matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square.")
    if sparse.issparse(matrix):
        if matrix.data.size and (not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0)):
            raise ValueError("transition_matrix entries must be finite and non-negative.")
    else:
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("transition_matrix entries must be finite and non-negative.")

    n_selected = coordinates.shape[0]
    if selected_indices is None:
        if matrix.shape[0] != n_selected:
            raise ValueError(
                "selected_indices is required when transition_matrix includes "
                "cells outside selected_coordinates."
            )
        indices = np.arange(n_selected, dtype=int)
    else:
        indices = np.asarray(selected_indices, dtype=int)
        if indices.ndim != 1 or len(indices) != n_selected:
            raise ValueError(
                "selected_indices must be one-dimensional and match selected_coordinates."
            )
        if len(np.unique(indices)) != len(indices):
            raise ValueError("selected_indices contains duplicates.")
        if np.any(indices < 0) or np.any(indices >= matrix.shape[0]):
            raise IndexError("selected_indices is outside transition_matrix bounds.")

    total_mass = _row_sums(matrix)[indices]
    if sparse.issparse(matrix):
        retained = matrix.tocsr()[indices][:, indices]
    else:
        retained = matrix[np.ix_(indices, indices)]
    retained_mass = _row_sums(retained)
    external_mass = np.maximum(total_mass - retained_mass, 0.0)

    coverage = np.zeros(n_selected, dtype=float)
    has_outgoing = total_mass > 0
    coverage[has_outgoing] = retained_mass[has_outgoing] / total_mass[has_outgoing]

    defined = has_outgoing & (retained_mass > 0) & (coverage >= min_coverage)

    if sparse.issparse(retained):
        weights = retained.astype(float).tocsr(copy=True)
        if renormalize_retained:
            inverse = np.zeros_like(retained_mass)
            positive = retained_mass > 0
            inverse[positive] = 1.0 / retained_mass[positive]
            weights = sparse.diags(inverse) @ weights
        expected_destination = np.asarray(weights @ coordinates)
        effective_row_sum = _row_sums(weights)
    else:
        weights = np.asarray(retained, dtype=float).copy()
        if renormalize_retained:
            positive = retained_mass > 0
            weights[positive] /= retained_mass[positive, None]
        expected_destination = weights @ coordinates
        effective_row_sum = weights.sum(axis=1)

    velocity = expected_destination - effective_row_sum[:, None] * coordinates
    velocity = np.asarray(velocity, dtype=float)
    velocity[~defined] = np.nan

    return ProjectionResult(
        velocity=velocity,
        retained_transition_mass=retained_mass,
        external_transition_mass=external_mass,
        transition_coverage=coverage,
        velocity_defined=defined,
        selected_indices=indices,
        renormalized=bool(renormalize_retained),
        min_coverage=float(min_coverage),
    )
