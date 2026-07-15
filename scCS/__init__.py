"""scCS — supervised single-cell commitment scoring at annotated furcations.

v0.8 is a breaking scientific redesign.  The lightweight mathematical core is
imported eagerly; dependency-heavy scorer and plotting modules are imported
lazily when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

from ._version import VERSION as __version__

__author__ = "Emil Kriukov"

from .furcation import Furcation, FurcationValidation
from .geometry import SimplexStarGeometry, regular_simplex_directions
from .ordering import (
    FurcationOrderingResult,
    FurcationOrderingScaler,
    OrderingDiagnostics,
)
from .scoring_embedding import ScoringEmbeddingResult, build_scoring_embedding
from .affinity import (
    CommitmentAffinityResult,
    MagnitudeScaler,
    aligned_directional_entropy,
    calibrated_softmax_beta,
    combine_direction_and_strength,
    cosine_softmax_affinity,
    normalized_entropy,
    support_adjusted_directional_specificity,
)
from .population import PopulationCommitmentSummary, summarize_commitment
from .projection import (
    ProjectionResult,
    RootProjectionGeometryDiagnostics,
    RootProgressionDirectionDiagnostics,
    project_transition_velocity,
)
from .transitions import get_scvelo_transition_matrix
from .future_fate import (
    DiscountedOutcomeSolution,
    FutureFateScoreResult,
    TransitionNormalization,
    build_outcome_anchors,
    canonicalize_transition_matrix,
    choose_endpoint_anchors,
    expected_ordering_change,
    score_future_fate,
    solve_discounted_outcomes,
)
from .pipeline import FurcationScoreResult, score_furcation
from .condition import ConditionCommitmentResult, ConditionScorer
from .inference import BootstrapInterval, PermutationTestResult
from .preflight import PreflightDiagnostic, PreflightReport

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "SingleScorer": (".single", "SingleScorer"),
    "PairScorer": (".pairwise", "PairScorer"),
    "MultiScorer": (".multicomparison", "MultiScorer"),
    "CommitmentGeneAssociationResult": (".drivers", "CommitmentGeneAssociationResult"),
    "get_commitment_associated_genes": (".drivers", "get_commitment_associated_genes"),
    "get_fate_markers": (".drivers", "get_fate_markers"),
    "get_deg_drivers": (".drivers", "get_deg_drivers"),
    "CommitmentEnrichmentResult": (".enrichment", "CommitmentEnrichmentResult"),
    "load_gmt": (".enrichment", "load_gmt"),
    "run_commitment_enrichment": (".enrichment", "run_commitment_enrichment"),
    "run_enrichment_per_fate": (".enrichment", "run_enrichment_per_fate"),
    "plot_enrichment_dotplot": (".enrichment", "plot_enrichment_dotplot"),
    "export_enrichment_tables": (".enrichment", "export_enrichment_tables"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attribute_name = _LAZY_IMPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "Furcation",
    "FurcationValidation",
    "SimplexStarGeometry",
    "regular_simplex_directions",
    "OrderingDiagnostics",
    "FurcationOrderingResult",
    "FurcationOrderingScaler",
    "ScoringEmbeddingResult",
    "build_scoring_embedding",
    "CommitmentAffinityResult",
    "MagnitudeScaler",
    "aligned_directional_entropy",
    "calibrated_softmax_beta",
    "cosine_softmax_affinity",
    "combine_direction_and_strength",
    "normalized_entropy",
    "support_adjusted_directional_specificity",
    "PopulationCommitmentSummary",
    "summarize_commitment",
    "ProjectionResult",
    "RootProjectionGeometryDiagnostics",
    "RootProgressionDirectionDiagnostics",
    "project_transition_velocity",
    "get_scvelo_transition_matrix",
    "TransitionNormalization",
    "DiscountedOutcomeSolution",
    "FutureFateScoreResult",
    "canonicalize_transition_matrix",
    "choose_endpoint_anchors",
    "build_outcome_anchors",
    "solve_discounted_outcomes",
    "expected_ordering_change",
    "score_future_fate",
    "FurcationScoreResult",
    "score_furcation",
    "ConditionCommitmentResult",
    "ConditionScorer",
    "BootstrapInterval",
    "PermutationTestResult",
    "PreflightDiagnostic",
    "PreflightReport",
    "SingleScorer",
    "PairScorer",
    "MultiScorer",
    "CommitmentGeneAssociationResult",
    "get_commitment_associated_genes",
    "get_fate_markers",
    "get_deg_drivers",
    "CommitmentEnrichmentResult",
    "load_gmt",
    "run_commitment_enrichment",
    "run_enrichment_per_fate",
    "plot_enrichment_dotplot",
    "export_enrichment_tables",
]
