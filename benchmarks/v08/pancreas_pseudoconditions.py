"""Reproducible pseudo-condition utilities for the pancreas validation benchmark.

These helpers are benchmark support code, not part of the public scCS scoring
API.  They create balanced pseudo-replicates and inject controlled effects into
already projected velocity vectors while preserving progression along the root
axis unless explicitly stated otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _as_label_set(value) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _normalized(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite one-dimensional vector.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError(f"{name} must have nonzero magnitude.")
    return vector / norm


def _validate_velocity_inputs(
    velocity,
    root_mask,
    condition_labels,
    replicate_labels,
    root_direction,
):
    values = np.asarray(velocity, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("velocity must be a finite cell-by-dimension array.")
    mask = np.asarray(root_mask)
    if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(values):
        raise ValueError("root_mask must be Boolean and aligned to velocity rows.")
    conditions = np.asarray(condition_labels, dtype=str)
    replicates = np.asarray(replicate_labels, dtype=str)
    if conditions.shape != (len(values),) or replicates.shape != (len(values),):
        raise ValueError("condition_labels and replicate_labels must align to cells.")
    root_axis = _normalized(root_direction, name="root_direction")
    if len(root_axis) != values.shape[1]:
        raise ValueError("root_direction dimensionality does not match velocity.")
    return values, mask, conditions, replicates, root_axis


def assign_stratified_pseudoreplicates(
    adata,
    *,
    annotation_key: str,
    root,
    terminals: Sequence[str],
    pseudotime_key: str,
    conditions: Sequence[str] = ("control", "treated"),
    replicates_per_condition: int = 4,
    n_root_bins: int = 4,
    condition_key: str = "sccs_condition",
    replicate_key: str = "sccs_replicate",
    random_state: int = 0,
) -> pd.DataFrame:
    """Assign every cell to balanced pseudo-conditions within biological strata.

    Root cells are stratified by pseudotime quantiles. Terminal cells are
    stratified by annotation, and unrelated cells are retained in annotation-
    specific background strata. Within each stratum, cells are shuffled and
    allocated cyclically across every condition/replicate combination, so
    observed stratum counts differ by at most one.
    """
    if annotation_key not in adata.obs or pseudotime_key not in adata.obs:
        raise KeyError("annotation_key and pseudotime_key must exist in adata.obs.")
    conditions = tuple(str(value) for value in conditions)
    if len(conditions) < 2 or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must contain at least two unique labels.")
    if int(replicates_per_condition) < 1:
        raise ValueError("replicates_per_condition must be positive.")
    if int(n_root_bins) < 1:
        raise ValueError("n_root_bins must be positive.")

    annotations = adata.obs[annotation_key].astype(str)
    pseudotime = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce")
    if pseudotime.isna().any():
        raise ValueError(f"{pseudotime_key!r} contains missing or nonnumeric values.")

    root_labels = _as_label_set(root)
    terminal_labels = {str(value) for value in terminals}
    strata = pd.Series(index=adata.obs_names, dtype=object)
    root_mask = annotations.isin(root_labels)
    if root_mask.any():
        root_values = pseudotime.loc[root_mask]
        n_unique = int(root_values.nunique())
        bins = min(int(n_root_bins), n_unique, int(root_mask.sum()))
        if bins <= 1:
            root_bin = pd.Series("0", index=root_values.index)
        else:
            ranked = root_values.rank(method="first")
            root_bin = pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(str)
        strata.loc[root_mask] = "root::" + root_bin

    terminal_mask = annotations.isin(terminal_labels)
    strata.loc[terminal_mask] = "terminal::" + annotations.loc[terminal_mask]
    background = ~(root_mask | terminal_mask)
    strata.loc[background] = "background::" + annotations.loc[background]

    combinations = [
        (condition, f"{condition}_{replicate + 1}")
        for condition in conditions
        for replicate in range(int(replicates_per_condition))
    ]
    rng = np.random.default_rng(random_state)
    assigned_condition = pd.Series(index=adata.obs_names, dtype=object)
    assigned_replicate = pd.Series(index=adata.obs_names, dtype=object)
    for _, indices in strata.groupby(strata, sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=object)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            condition, replicate = combinations[position % len(combinations)]
            assigned_condition.loc[index] = condition
            assigned_replicate.loc[index] = replicate

    adata.obs[condition_key] = pd.Categorical(assigned_condition, categories=conditions)
    replicate_order = [replicate for _, replicate in combinations]
    adata.obs[replicate_key] = pd.Categorical(
        assigned_replicate,
        categories=replicate_order,
    )
    return pd.DataFrame(
        {
            "stratum": strata.astype(str),
            condition_key: adata.obs[condition_key].astype(str),
            replicate_key: adata.obs[replicate_key].astype(str),
        },
        index=adata.obs_names,
    )


def _replicate_random_effects(
    conditions: np.ndarray,
    replicates: np.ndarray,
    *,
    standard_deviation: float,
    rng: np.random.Generator,
) -> dict[tuple[str, str], float]:
    if standard_deviation < 0 or not np.isfinite(standard_deviation):
        raise ValueError("Random-effect standard deviation must be finite and nonnegative.")
    keys = sorted(set(zip(conditions.tolist(), replicates.tolist())))
    return {
        key: float(rng.normal(0.0, standard_deviation)) if standard_deviation else 0.0
        for key in keys
    }


def _slerp_direction(source: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    dot = float(np.clip(source @ target, -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle < 1e-12:
        return source.copy()
    if np.pi - angle < 1e-8:
        # Choose a deterministic orthogonal direction for the antipodal case.
        basis = np.zeros_like(source)
        basis[int(np.argmin(np.abs(source)))] = 1.0
        orthogonal = basis - (basis @ source) * source
        orthogonal /= np.linalg.norm(orthogonal)
        return np.cos(np.pi * fraction) * source + np.sin(np.pi * fraction) * orthogonal
    denominator = np.sin(angle)
    output = (
        np.sin((1.0 - fraction) * angle) / denominator * source
        + np.sin(fraction * angle) / denominator * target
    )
    return output / np.linalg.norm(output)


def inject_root_directional_effect(
    velocity,
    *,
    root_mask,
    condition_labels,
    replicate_labels,
    root_direction,
    target_direction,
    effect_by_condition: Mapping[str, float],
    replicate_effect_sd: float = 0.0,
    replicate_magnitude_log_sd: float = 0.0,
    random_state: int = 0,
) -> np.ndarray:
    """Rotate root fate components toward a target by an angular fraction.

    The progression component parallel to ``root_direction`` and the original
    fate-component magnitude are preserved apart from the optional replicate-
    level log-magnitude perturbation. Non-root vectors are unchanged.
    """
    values, mask, conditions, replicates, root_axis = _validate_velocity_inputs(
        velocity,
        root_mask,
        condition_labels,
        replicate_labels,
        root_direction,
    )
    target = np.asarray(target_direction, dtype=float)
    if target.ndim != 1 or len(target) != values.shape[1] or not np.all(np.isfinite(target)):
        raise ValueError("target_direction must be finite and match velocity dimensions.")
    target = target - float(target @ root_axis) * root_axis
    target = _normalized(target, name="target_direction orthogonal component")

    rng = np.random.default_rng(random_state)
    angular_random = _replicate_random_effects(
        conditions,
        replicates,
        standard_deviation=replicate_effect_sd,
        rng=rng,
    )
    magnitude_random = _replicate_random_effects(
        conditions,
        replicates,
        standard_deviation=replicate_magnitude_log_sd,
        rng=rng,
    )
    output = values.copy()
    for index in np.flatnonzero(mask):
        condition = conditions[index]
        if condition not in effect_by_condition:
            raise KeyError(f"Missing directional effect for condition {condition!r}.")
        progression = float(values[index] @ root_axis)
        branch = values[index] - progression * root_axis
        magnitude = float(np.linalg.norm(branch))
        if magnitude <= 0:
            continue
        source = branch / magnitude
        key = (condition, replicates[index])
        fraction = float(effect_by_condition[condition]) + angular_random[key]
        direction = _slerp_direction(source, target, fraction)
        magnitude *= float(np.exp(magnitude_random[key]))
        output[index] = progression * root_axis + magnitude * direction
    return output


def inject_root_magnitude_effect(
    velocity,
    *,
    root_mask,
    condition_labels,
    replicate_labels,
    root_direction,
    factor_by_condition: Mapping[str, float],
    replicate_magnitude_log_sd: float = 0.0,
    random_state: int = 0,
) -> np.ndarray:
    """Scale root fate-component magnitude while preserving its direction."""
    values, mask, conditions, replicates, root_axis = _validate_velocity_inputs(
        velocity,
        root_mask,
        condition_labels,
        replicate_labels,
        root_direction,
    )
    rng = np.random.default_rng(random_state)
    magnitude_random = _replicate_random_effects(
        conditions,
        replicates,
        standard_deviation=replicate_magnitude_log_sd,
        rng=rng,
    )
    output = values.copy()
    for index in np.flatnonzero(mask):
        condition = conditions[index]
        if condition not in factor_by_condition:
            raise KeyError(f"Missing magnitude factor for condition {condition!r}.")
        factor = float(factor_by_condition[condition])
        if factor < 0 or not np.isfinite(factor):
            raise ValueError("Magnitude factors must be finite and nonnegative.")
        key = (condition, replicates[index])
        factor *= float(np.exp(magnitude_random[key]))
        progression = float(values[index] @ root_axis)
        branch = values[index] - progression * root_axis
        output[index] = progression * root_axis + factor * branch
    return output


def inject_root_combined_effect(
    velocity,
    *,
    root_mask,
    condition_labels,
    replicate_labels,
    root_direction,
    target_direction,
    directional_effect_by_condition: Mapping[str, float],
    magnitude_factor_by_condition: Mapping[str, float],
    replicate_direction_sd: float = 0.0,
    replicate_magnitude_log_sd: float = 0.0,
    random_state: int = 0,
) -> np.ndarray:
    """Apply controlled directional and magnitude effects to root cells only."""
    directed = inject_root_directional_effect(
        velocity,
        root_mask=root_mask,
        condition_labels=condition_labels,
        replicate_labels=replicate_labels,
        root_direction=root_direction,
        target_direction=target_direction,
        effect_by_condition=directional_effect_by_condition,
        replicate_effect_sd=replicate_direction_sd,
        replicate_magnitude_log_sd=0.0,
        random_state=random_state,
    )
    return inject_root_magnitude_effect(
        directed,
        root_mask=root_mask,
        condition_labels=condition_labels,
        replicate_labels=replicate_labels,
        root_direction=root_direction,
        factor_by_condition=magnitude_factor_by_condition,
        replicate_magnitude_log_sd=replicate_magnitude_log_sd,
        random_state=random_state + 1,
    )
