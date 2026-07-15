"""Integrated v0.8 scoring pipeline for one supervised furcation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np

from ._version import VERSION
from .affinity import (
    CommitmentAffinityResult,
    MagnitudeScaler,
    aligned_directional_entropy,
    combine_direction_and_strength,
    cosine_softmax_affinity,
)
from .furcation import Furcation
from .ordering import FurcationOrderingScaler
from .population import PopulationCommitmentSummary, summarize_commitment
from .projection import ProjectionResult, project_transition_velocity
from .scoring_embedding import ScoringEmbeddingResult, build_scoring_embedding
from .transitions import get_scvelo_transition_matrix

MagnitudeFitPopulation = Literal["root", "all"]


@dataclass(frozen=True)
class FurcationScoreResult:
    """Cell- and population-level commitment result for one furcation.

    The result deliberately separates fate direction, fate-directed magnitude,
    and magnitude-weighted commitment.  ``directional_affinity`` is not a
    calibrated future-fate probability; it is a supervised allocation across
    the manually defined terminal populations.
    """

    furcation: Furcation
    embedding: ScoringEmbeddingResult
    projection: ProjectionResult
    projected_velocity: np.ndarray
    progression_velocity: np.ndarray
    branch_velocity: np.ndarray
    aligned_probability: float
    commitment: CommitmentAffinityResult
    status: np.ndarray
    dominant_fate: np.ndarray
    root_population_summary: PopulationCommitmentSummary
    magnitude_scaler: Any
    magnitude_fit_population: str

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
        """Number of cells in the manually selected furcation."""
        return len(self.cell_ids)

    @property
    def n_root(self) -> int:
        """Number of cells in the manually annotated root population."""
        return int(np.sum(self.root_mask))

    @property
    def n_terminal(self) -> int:
        """Number of cells in all manually annotated terminal populations."""
        return int(np.sum(self.terminal_mask))

    @property
    def root_mask(self) -> np.ndarray:
        return self.embedding.root_mask

    @property
    def terminal_mask(self) -> np.ndarray:
        return self.embedding.terminal_mask

    @property
    def directional_affinity(self) -> np.ndarray:
        return self.commitment.directional_affinity

    @property
    def commitment_strength(self) -> np.ndarray:
        return self.commitment.commitment_strength

    @property
    def commitment_contribution(self) -> np.ndarray:
        return self.commitment.commitment_contribution

    @property
    def directional_entropy(self) -> np.ndarray:
        """Raw normalized Shannon entropy of cosine-softmax fate affinity."""
        return self.commitment.directional_entropy

    @property
    def commitment_entropy(self) -> np.ndarray:
        """Entropy after weak velocities are blended toward uniform affinity."""
        return self.commitment.commitment_entropy

    @property
    def aligned_directional_entropy(self) -> float:
        """Attainable entropy floor for a perfectly aligned fate vector."""
        return aligned_directional_entropy(self.k, self.aligned_probability)

    @property
    def directional_specificity(self) -> np.ndarray:
        """Support-adjusted specificity: uniform=0 and perfect alignment=1."""
        return self.commitment.directional_specificity

    @property
    def fate_cosine_similarity(self) -> np.ndarray:
        """Cosine similarity of branch velocity to every ideal fate axis.

        Rows with undefined or zero fate-directed velocity are returned as
        ``NaN`` because an angle is not defined without a direction.
        """
        vectors = np.asarray(self.branch_velocity, dtype=float)
        directions = np.asarray(self.embedding.geometry.terminal_directions, dtype=float)
        output = np.full((len(vectors), len(directions)), np.nan, dtype=float)
        finite = np.all(np.isfinite(vectors), axis=1)
        magnitudes = np.linalg.norm(np.nan_to_num(vectors, nan=0.0), axis=1)
        defined = finite & (magnitudes > np.finfo(float).eps)
        if np.any(defined):
            unit_vectors = vectors[defined] / magnitudes[defined, None]
            unit_directions = directions / np.linalg.norm(directions, axis=1)[:, None]
            output[defined] = np.clip(unit_vectors @ unit_directions.T, -1.0, 1.0)
        return output

    @property
    def nearest_fate_angle_degrees(self) -> np.ndarray:
        """Smallest angular deviation from any ideal fate axis, in degrees."""
        cosine = self.fate_cosine_similarity
        values = np.full(len(cosine), np.nan, dtype=float)
        defined = np.any(np.isfinite(cosine), axis=1)
        if np.any(defined):
            values[defined] = np.degrees(
                np.arccos(np.clip(np.nanmax(cosine[defined], axis=1), -1.0, 1.0))
            )
        return values

    @property
    def specific_commitment(self) -> np.ndarray:
        return self.commitment.specific_commitment

    def summarize(self, mask: Optional[np.ndarray] = None) -> PopulationCommitmentSummary:
        """Summarize commitment over an explicit selected-cell mask.

        If no mask is supplied, the manually annotated root population is
        summarized.  The mask is always aligned to ``result.cell_ids``.
        """
        if mask is None:
            mask = self.root_mask
        values = np.asarray(mask)
        if values.dtype != bool or values.ndim != 1 or len(values) != len(self.cell_ids):
            raise ValueError("mask must be a Boolean array aligned to selected furcation cells.")
        if not values.any():
            raise ValueError("Cannot summarize an empty population.")
        return summarize_commitment(self.commitment.commitment_contribution[values])

    def summary(self) -> str:
        """Return a concise human-readable summary."""
        root = self.root_mask
        valid = self.projection.velocity_defined
        valid_root = root & valid
        statuses, counts = np.unique(self.status[root], return_counts=True)
        status_text = ", ".join(f"{name}={count}" for name, count in zip(statuses, counts))
        if valid_root.any():
            mean_strength = float(np.mean(self.commitment_strength[valid_root]))
            mean_entropy = float(np.mean(self.directional_entropy[valid_root]))
            mean_commitment_entropy = float(np.mean(self.commitment_entropy[valid_root]))
            mean_specificity = float(np.mean(self.directional_specificity[valid_root]))
            mean_specific_commitment = float(np.mean(self.specific_commitment[valid_root]))
            median_coverage = float(np.median(self.projection.transition_coverage[valid_root]))
        else:
            mean_strength = float("nan")
            mean_entropy = float("nan")
            mean_commitment_entropy = float("nan")
            mean_specificity = float("nan")
            mean_specific_commitment = float("nan")
            median_coverage = float("nan")

        composition = self.root_population_summary.commitment_composition
        if self.root_population_summary.composition_defined:
            composition_text = ", ".join(
                f"{fate}={value:.3f}" for fate, value in zip(self.fate_names, composition)
            )
        else:
            composition_text = "undefined (zero commitment mass)"

        return (
            "scCS FurcationScoreResult\n"
            f"  Furcation: {self.furcation.root_name} -> {list(self.fate_names)}\n"
            f"  Cells: {len(self.cell_ids)} selected; {int(root.sum())} root\n"
            f"  Valid projected velocities: {int(valid.sum())}/{len(valid)}\n"
            f"  Root median transition coverage: {median_coverage:.3f}\n"
            f"  Root mean commitment strength: {mean_strength:.3f}\n"
            f"  Root mean directional entropy: {mean_entropy:.3f}\n"
            f"  Aligned directional entropy floor: {self.aligned_directional_entropy:.3f}\n"
            f"  Root mean commitment entropy: {mean_commitment_entropy:.3f}\n"
            f"  Root mean directional specificity: {mean_specificity:.3f}\n"
            f"  Root mean specific commitment: {mean_specific_commitment:.3f}\n"
            f"  Root commitment composition: {composition_text}\n"
            f"  Root status counts: {status_text}"
        )

    def write_to_adata(
        self,
        adata,
        *,
        metadata_key: str = "sccs",
    ) -> None:
        """Write v0.8 outputs to full-length AnnData containers."""
        n_obs = adata.n_obs
        selected = self.embedding.selected_indices
        k = len(self.fate_names)
        dimension = self.embedding.dimension

        def full_matrix(values: np.ndarray, width: int) -> np.ndarray:
            result = np.full((n_obs, width), np.nan, dtype=float)
            result[selected] = values
            return result

        def full_vector(values: np.ndarray, *, dtype=float, fill=np.nan):
            result = np.full(n_obs, fill, dtype=dtype)
            result[selected] = values
            return result

        self.embedding.write_to_adata(adata, metadata_key=metadata_key)
        adata.obsm["sccs_projected_velocity"] = full_matrix(self.projected_velocity, dimension)
        adata.obsm["sccs_branch_velocity"] = full_matrix(self.branch_velocity, dimension)
        adata.obsm["sccs_directional_affinity"] = full_matrix(
            self.commitment.directional_affinity, k
        )
        adata.obsm["sccs_commitment_affinity"] = full_matrix(self.commitment.commitment_affinity, k)
        adata.obsm["sccs_commitment_contribution"] = full_matrix(
            self.commitment.commitment_contribution, k
        )
        adata.obsm["sccs_fate_cosine_similarity"] = full_matrix(self.fate_cosine_similarity, k)
        adata.obs["sccs_progression_velocity"] = full_vector(self.progression_velocity)
        adata.obs["sccs_branch_magnitude"] = full_vector(self.commitment.velocity_magnitude)
        adata.obs["sccs_commitment_strength"] = full_vector(self.commitment.commitment_strength)
        adata.obs["sccs_directional_entropy"] = full_vector(self.commitment.directional_entropy)
        adata.obs["sccs_commitment_entropy"] = full_vector(self.commitment.commitment_entropy)
        adata.obs["sccs_nearest_fate_angle_degrees"] = full_vector(self.nearest_fate_angle_degrees)
        adata.obs["sccs_directional_specificity"] = full_vector(
            self.commitment.directional_specificity
        )
        adata.obs["sccs_specific_commitment"] = full_vector(self.commitment.specific_commitment)
        adata.obs["sccs_transition_coverage"] = full_vector(self.projection.transition_coverage)
        adata.obs["sccs_external_transition_mass"] = full_vector(
            self.projection.external_transition_mass
        )
        adata.obs["sccs_status"] = full_vector(
            self.status.astype(object), dtype=object, fill="outside_furcation"
        )
        adata.obs["sccs_dominant_fate"] = full_vector(
            self.dominant_fate.astype(object), dtype=object, fill=""
        )

        metadata = dict(adata.uns.get(metadata_key, {}))
        metadata["schema_version"] = "0.8"
        metadata["package_version"] = VERSION
        metadata["scientific_scope"] = "supervised_annotated_furcation"
        metadata["geometry"] = {
            "type": "incoming_root_regular_simplex_fixed_terminal_vertices",
            "dimension": int(self.embedding.dimension),
            "arm_scale": float(self.embedding.arm_scale),
            "fate_names": list(self.fate_names),
        }
        metadata["projection"] = {
            "renormalized_retained": bool(self.projection.renormalized),
            "min_coverage": float(self.projection.min_coverage),
            "mean_transition_coverage": float(np.nanmean(self.projection.transition_coverage)),
            "undefined_fraction": float(np.mean(~self.projection.velocity_defined)),
        }
        metadata["affinity_model"] = "calibrated_cosine_softmax"
        metadata["aligned_probability"] = float(self.aligned_probability)
        metadata["entropy"] = {
            "directional_entropy": "normalized_shannon_of_directional_affinity",
            "commitment_entropy": "normalized_shannon_of_strength_blended_affinity",
            "aligned_directional_entropy_floor": float(self.aligned_directional_entropy),
            "directional_specificity": (
                "support_adjusted_(1-H)/(1-H_aligned); uniform=0; aligned=1"
            ),
        }
        metadata["magnitude_fit_population"] = self.magnitude_fit_population
        metadata["root_population_summary"] = {
            "total_commitment_mass": (self.root_population_summary.total_commitment_mass.tolist()),
            "mean_commitment_contribution": (
                self.root_population_summary.mean_commitment_contribution.tolist()
            ),
            "commitment_composition": (
                self.root_population_summary.commitment_composition.tolist()
            ),
            "pairwise_log_commitment_ratio": (
                self.root_population_summary.pairwise_log_commitment_ratio.tolist()
            ),
            "population_balance_entropy": (self.root_population_summary.population_balance_entropy),
            "n_cells": self.root_population_summary.n_cells,
            "total_mass": self.root_population_summary.total_mass,
            "composition_defined": (self.root_population_summary.composition_defined),
        }
        scaler_metadata = {}
        for key, value in vars(self.magnitude_scaler).items():
            if isinstance(value, np.ndarray):
                scaler_metadata[key] = value.tolist()
            elif isinstance(value, (str, int, float, bool)) or value is None:
                scaler_metadata[key] = value
        metadata["magnitude_scaler"] = scaler_metadata
        adata.uns[metadata_key] = metadata


def _score_status(
    projection: ProjectionResult,
    progression: np.ndarray,
    branch_magnitude: np.ndarray,
    strength: np.ndarray,
    specificity: np.ndarray,
    *,
    committed_strength_threshold: float,
    committed_specificity_threshold: float,
) -> np.ndarray:
    status = np.full(len(branch_magnitude), "uncommitted", dtype=object)
    total_outgoing = projection.retained_transition_mass + projection.external_transition_mass
    no_outgoing = total_outgoing <= np.finfo(float).eps
    low_coverage = (~projection.velocity_defined) & (~no_outgoing)
    status[no_outgoing] = "no_projected_velocity"
    status[low_coverage] = "low_transition_coverage"

    valid = projection.velocity_defined
    branch_zero = valid & (branch_magnitude <= np.finfo(float).eps)
    progressing = branch_zero & (np.abs(progression) > np.finfo(float).eps)
    status[branch_zero] = "no_fate_velocity"
    status[progressing] = "progressing_uncommitted"

    has_branch = valid & ~branch_zero
    weak = has_branch & (strength < committed_strength_threshold)
    ambiguous = has_branch & ~weak & (specificity < committed_specificity_threshold)
    committed = has_branch & ~weak & ~ambiguous
    status[weak] = "weak_fate_velocity"
    status[ambiguous] = "directionally_ambiguous"
    status[committed] = "fate_committed"
    return status


def score_projected_furcation(
    furcation: Furcation,
    embedding: ScoringEmbeddingResult,
    projection: ProjectionResult,
    *,
    aligned_probability: float = 0.90,
    magnitude_scaler: Optional[Any] = None,
    magnitude_fit_population: MagnitudeFitPopulation = "root",
    committed_strength_threshold: float = 0.25,
    committed_specificity_threshold: float = 0.25,
) -> FurcationScoreResult:
    """Score a pre-built furcation embedding and projected velocity field."""
    if magnitude_fit_population not in {"root", "all"}:
        raise ValueError("magnitude_fit_population must be 'root' or 'all'.")
    if len(projection.velocity) != embedding.n_selected:
        raise ValueError("projection and embedding contain different cell counts.")

    safe_velocity = np.zeros_like(projection.velocity)
    safe_velocity[projection.velocity_defined] = projection.velocity[projection.velocity_defined]
    progression, safe_branch = embedding.geometry.decompose_velocity(safe_velocity)

    # Reuse the immutable projection payload instead of keeping a second
    # full-size copy. This is especially important for million-cell runs.
    projected_velocity = projection.velocity
    branch_velocity = safe_branch.copy()
    branch_velocity[~projection.velocity_defined] = np.nan
    progression = progression.astype(float)
    progression[~projection.velocity_defined] = np.nan

    directional_affinity = cosine_softmax_affinity(
        safe_branch,
        embedding.geometry.terminal_directions,
        aligned_probability=aligned_probability,
    )

    branch_magnitude = np.linalg.norm(safe_branch, axis=1)
    scaler = magnitude_scaler or MagnitudeScaler(
        scale_quantile=0.75,
        power=1.0,
    )

    fit_mask = projection.velocity_defined.copy()
    if magnitude_fit_population == "root":
        fit_mask &= embedding.root_mask
    if not fit_mask.any():
        raise ValueError("No valid projected velocities are available to fit magnitude strength.")
    scaler.fit(branch_magnitude[fit_mask])
    strength = np.zeros(len(branch_magnitude), dtype=float)
    strength[projection.velocity_defined] = scaler.transform(
        branch_magnitude[projection.velocity_defined]
    )

    commitment = combine_direction_and_strength(
        directional_affinity,
        safe_branch,
        strength,
        aligned_probability=aligned_probability,
    )

    status = _score_status(
        projection,
        np.nan_to_num(progression, nan=0.0),
        branch_magnitude,
        commitment.commitment_strength,
        commitment.directional_specificity,
        committed_strength_threshold=committed_strength_threshold,
        committed_specificity_threshold=committed_specificity_threshold,
    )

    dominant_indices = np.argmax(
        commitment.directional_affinity,
        axis=1,
    )
    dominant_fate = np.asarray(embedding.geometry.fate_names, dtype=object)[dominant_indices]
    dominant_fate[status != "fate_committed"] = ""

    root_summary = summarize_commitment(commitment.commitment_contribution[embedding.root_mask])

    return FurcationScoreResult(
        furcation=furcation,
        embedding=embedding,
        projection=projection,
        projected_velocity=projected_velocity,
        progression_velocity=progression,
        branch_velocity=branch_velocity,
        aligned_probability=float(aligned_probability),
        commitment=commitment,
        status=status,
        dominant_fate=dominant_fate,
        root_population_summary=root_summary,
        magnitude_scaler=scaler,
        magnitude_fit_population=magnitude_fit_population,
    )


def score_furcation(
    adata,
    furcation: Furcation,
    *,
    ordering: Union[str, Sequence[float], np.ndarray],
    transition_matrix=None,
    ordering_scaler: Optional[FurcationOrderingScaler] = None,
    arm_scale: float = 1.0,
    aligned_probability: float = 0.90,
    magnitude_scaler: Optional[Any] = None,
    magnitude_fit_population: MagnitudeFitPopulation = "root",
    renormalize_retained: bool = True,
    min_transition_coverage: float = 0.05,
    committed_strength_threshold: float = 0.25,
    committed_specificity_threshold: float = 0.25,
    write_to_adata: bool = False,
) -> FurcationScoreResult:
    """Run the integrated v0.8 engine on one manual furcation."""
    embedding = build_scoring_embedding(
        adata,
        furcation,
        ordering=ordering,
        ordering_scaler=ordering_scaler,
        arm_scale=arm_scale,
        write_to_adata=False,
    )

    if transition_matrix is None:
        transition_matrix = get_scvelo_transition_matrix(adata)

    projection = project_transition_velocity(
        transition_matrix,
        embedding.coordinates,
        selected_indices=embedding.selected_indices,
        renormalize_retained=renormalize_retained,
        min_coverage=min_transition_coverage,
    )

    result = score_projected_furcation(
        furcation,
        embedding,
        projection,
        aligned_probability=aligned_probability,
        magnitude_scaler=magnitude_scaler,
        magnitude_fit_population=magnitude_fit_population,
        committed_strength_threshold=committed_strength_threshold,
        committed_specificity_threshold=committed_specificity_threshold,
    )

    if write_to_adata:
        result.write_to_adata(adata)

    return result
