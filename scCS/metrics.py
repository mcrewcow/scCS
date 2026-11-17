"""One source of metric aliases, scientific labels, and mode applicability."""

from __future__ import annotations

ALIASES = {
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
    "unresolved_future_probability": "unresolved_probability",
    "competing_fate_reach": "competing_reach",
}
FUTURE_ONLY = set(ALIASES) - {"mean_commitment"}
FUTURE_ONLY |= {"unresolved_probability", "competing_reach"}
FATE_METRICS = {
    "mean_commitment_contribution",
    "directional_affinity",
    "commitment_composition",
}
SCALAR_METRICS = {
    "commitment_strength",
    "directional_entropy",
    "commitment_entropy",
    "directional_specificity",
    "nearest_fate_angle_degrees",
    "specific_commitment",
    "progression_velocity",
    "transition_coverage",
    "unresolved_probability",
    "competing_reach",
    "affinity_defined_fraction",
}
LABELS = {
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
    "unresolved_probability": "Unresolved Future Probability (UFP)",
    "competing_reach": "Competing-outcome reach",
    "affinity_defined_fraction": "Fraction with defined affinity",
}
FUTURE_LABELS = {
    "mean_commitment_contribution": "Discounted fate probability",
    "directional_affinity": "Conditional Fate Affinity (CFA)",
    "commitment_strength": "Discounted Fate Reach (DFR)",
    "directional_entropy": "Future-fate entropy",
    "commitment_entropy": "Reach-adjusted future entropy",
    "directional_specificity": "Future-Fate Specificity (FFS)",
    "specific_commitment": "Resolved Commitment (RC)",
    "progression_velocity": "Signed Ordering Flux (SOF)",
    "transition_coverage": "Selected-path coverage",
}


def canonical_metric(metric: str) -> str:
    return ALIASES.get(str(metric), str(metric))


def resolve_metric(metric: str, scoring_mode: str) -> str:
    requested = str(metric)
    if scoring_mode != "future_fate" and requested in FUTURE_ONLY:
        raise ValueError(f"Metric {requested!r} requires scoring_mode='future_fate'.")
    canonical = canonical_metric(requested)
    if scoring_mode == "future_fate" and canonical == "nearest_fate_angle_degrees":
        raise ValueError(
            "Fate angles are only defined for scoring_mode='instantaneous'."
        )
    return canonical


def metric_label(metric: str, *, fate=None, scoring_mode=None) -> str:
    canonical = canonical_metric(metric)
    future = scoring_mode == "future_fate" or (
        scoring_mode is None and metric in FUTURE_ONLY
    )
    labels = {**LABELS, **FUTURE_LABELS} if future else LABELS
    label = labels.get(canonical, canonical.replace("_", " ").capitalize())
    return label if fate is None else f"{label} toward {fate}"


# Display syntax resolves through the same scientific alias registry.
STAR_PREFIXES = {
    "affinity": "directional_affinity",
    "contribution": "mean_commitment_contribution",
    "commitment_contribution": "mean_commitment_contribution",
}


def star_color_metric(color_by, scoring_mode):
    """Resolve a star color name while preserving its optional fate suffix."""
    prefix, separator, fate = str(color_by).partition(":")
    metric = resolve_metric(STAR_PREFIXES.get(prefix, prefix), scoring_mode)
    return metric + (separator + fate if separator else "")
