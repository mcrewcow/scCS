"""Directional fate affinity and commitment strength for scCS v0.8.

The v0.8 scientific model separates two biological quantities:

* directional affinity: where the fate-directed projected velocity points;
* commitment strength: how large the fate-directed projected velocity is.

Directional affinity uses a calibrated cosine softmax on regular-simplex fate
axes.  Commitment strength uses a robust Hill transform fitted once on the
analysis population, normally all root cells pooled across conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return row-wise Shannon entropy normalized by ``log(k)``."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_cells, k) with k >= 2.")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be finite and non-negative.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("probability rows must sum to one.")
    clipped = np.clip(values, 1e-15, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / np.log(values.shape[1])


def calibrated_softmax_beta(k: int, aligned_probability: float) -> float:
    """Return the inverse temperature for a requested aligned affinity.

    For a regular simplex, a vector aligned with one fate has cosine
    similarity ``1`` to that fate and ``-1 / (k - 1)`` to every alternative.
    This function chooses ``beta`` so the aligned fate receives exactly
    ``aligned_probability`` after softmax.
    """
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer greater than or equal to 2.")
    lower = 1.0 / k
    if not lower < aligned_probability < 1.0:
        raise ValueError(f"aligned_probability must lie strictly between {lower} and 1.")
    return float(
        (k - 1.0) / k * np.log((k - 1.0) * aligned_probability / (1.0 - aligned_probability))
    )


def aligned_directional_entropy(k: int, aligned_probability: float = 0.90) -> float:
    """Return the normalized entropy of a perfectly fate-aligned vector.

    The cosine-softmax calibration deliberately assigns finite probability to
    alternative fates.  Therefore the raw normalized Shannon entropy does not
    reach zero at perfect alignment.  This value is the attainable entropy
    floor for the configured regular-simplex geometry and softmax calibration.
    """
    calibrated_softmax_beta(k, aligned_probability)
    alternatives = (1.0 - float(aligned_probability)) / (k - 1)
    distribution = np.full(k, alternatives, dtype=float)
    distribution[0] = float(aligned_probability)
    return float(normalized_entropy(distribution[None, :])[0])


def support_adjusted_directional_specificity(
    directional_entropy: np.ndarray,
    *,
    k: int,
    aligned_probability: float = 0.90,
) -> np.ndarray:
    """Map raw directional entropy to a 0--1 geometry-aware specificity.

    Uniform directional affinity maps to zero.  A vector exactly aligned with
    one ideal fate axis maps to one under the configured cosine-softmax
    calibration.  Values are clipped to ``[0, 1]`` for numerical stability.
    """
    entropy = np.asarray(directional_entropy, dtype=float)
    if not np.all(np.isfinite(entropy)):
        raise ValueError("directional_entropy must be finite.")
    floor = aligned_directional_entropy(k, aligned_probability)
    denominator = 1.0 - floor
    if denominator <= np.finfo(float).eps:
        raise ValueError("The configured entropy support is degenerate.")
    return np.clip((1.0 - entropy) / denominator, 0.0, 1.0)


def _validate_vectors_and_directions(
    vectors: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(vectors, dtype=float)
    d = np.asarray(directions, dtype=float)
    if v.ndim != 2 or d.ndim != 2 or v.shape[1] != d.shape[1]:
        raise ValueError("vectors and directions must be two-dimensional with matching dimensions.")
    if d.shape[0] < 2:
        raise ValueError("At least two fate directions are required.")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(d)):
        raise ValueError("vectors and directions must be finite.")

    direction_norms = np.linalg.norm(d, axis=1)
    if np.any(direction_norms <= 0):
        raise ValueError("Fate directions must be nonzero.")
    d_unit = d / direction_norms[:, None]

    magnitudes = np.linalg.norm(v, axis=1)
    v_unit = np.zeros_like(v)
    defined = magnitudes > np.finfo(float).eps
    v_unit[defined] = v[defined] / magnitudes[defined, None]
    return v_unit, d_unit, defined


def cosine_softmax_affinity(
    branch_velocity: np.ndarray,
    fate_directions: np.ndarray,
    *,
    aligned_probability: float = 0.90,
) -> np.ndarray:
    """Calculate calibrated cosine-softmax fate affinity.

    A zero branch velocity receives a uniform directional affinity.  This is a
    neutral mathematical value only; no-signal status is carried separately in
    :class:`CommitmentAffinityResult` and the scorer result.
    """
    v_unit, d_unit, defined = _validate_vectors_and_directions(branch_velocity, fate_directions)
    k = d_unit.shape[0]
    beta = calibrated_softmax_beta(k, aligned_probability)
    logits = beta * (v_unit @ d_unit.T)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    probabilities[~defined] = 1.0 / k
    return probabilities


@dataclass
class MagnitudeScaler:
    """Robust Hill scaler for fate-directed velocity magnitude.

    The transform is

    ``m**power / (m**power + scale**power)``.

    ``scale`` is estimated from a quantile of positive reference magnitudes.
    The default 75th percentile was selected in the real-pancreas v0.8
    benchmark because it was robust to outliers, avoided hard clipping, and
    remained conservative for weak velocities.
    """

    scale_quantile: float = 0.75
    power: float = 1.0
    scale_: Optional[float] = None
    n_reference_: Optional[int] = None
    n_positive_reference_: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.scale_quantile <= 1.0:
            raise ValueError("scale_quantile must lie in (0, 1].")
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be positive and finite.")

    @property
    def fitted(self) -> bool:
        return self.scale_ is not None

    def fit(self, magnitudes: np.ndarray) -> "MagnitudeScaler":
        values = np.asarray(magnitudes, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("magnitudes must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("magnitudes must be finite and non-negative.")

        positive = values[values > np.finfo(float).eps]
        self.n_reference_ = int(len(values))
        self.n_positive_reference_ = int(len(positive))
        if len(positive) == 0:
            # A no-signal reference population maps every magnitude to zero.
            self.scale_ = np.inf
        else:
            scale = float(np.quantile(positive, self.scale_quantile))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                raise ValueError("Could not estimate a positive magnitude scale.")
            self.scale_ = scale
        return self

    def transform(self, magnitudes: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("MagnitudeScaler must be fit before transform().")
        values = np.asarray(magnitudes, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("magnitudes must be a finite non-negative 1D array.")
        assert self.scale_ is not None
        if np.isinf(self.scale_):
            return np.zeros_like(values)
        numerator = np.power(values, self.power)
        denominator = numerator + self.scale_**self.power
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(values),
            where=denominator > 0,
        )

    def fit_transform(self, magnitudes: np.ndarray) -> np.ndarray:
        return self.fit(magnitudes).transform(magnitudes)


@dataclass(frozen=True)
class CommitmentAffinityResult:
    """Cell-level directional and magnitude-aware commitment outputs."""

    directional_affinity: np.ndarray
    velocity_magnitude: np.ndarray
    commitment_strength: np.ndarray
    commitment_affinity: np.ndarray
    commitment_contribution: np.ndarray
    directional_entropy: np.ndarray
    commitment_entropy: np.ndarray
    directional_specificity: np.ndarray
    specific_commitment: np.ndarray
    velocity_defined: np.ndarray


def combine_direction_and_strength(
    directional_affinity: np.ndarray,
    branch_velocity: np.ndarray,
    strength: np.ndarray,
    *,
    aligned_probability: float = 0.90,
) -> CommitmentAffinityResult:
    """Combine affinity and strength while preserving both native quantities."""
    q = np.asarray(directional_affinity, dtype=float)
    vectors = np.asarray(branch_velocity, dtype=float)
    s = np.asarray(strength, dtype=float)
    if q.ndim != 2 or vectors.ndim != 2 or s.ndim != 1:
        raise ValueError("Invalid array dimensions.")
    if len(q) != len(vectors) or len(q) != len(s):
        raise ValueError("All cell-level arrays must have matching lengths.")
    if np.any((s < 0) | (s > 1)) or not np.all(np.isfinite(s)):
        raise ValueError("strength must be finite and lie in [0, 1].")
    if not np.allclose(q.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("directional_affinity rows must sum to one.")

    k = q.shape[1]
    uniform = np.full_like(q, 1.0 / k)
    commitment_affinity = (1.0 - s[:, None]) * uniform + s[:, None] * q
    contribution = s[:, None] * q
    directional_entropy = normalized_entropy(q)
    commitment_entropy = normalized_entropy(commitment_affinity)
    specificity = support_adjusted_directional_specificity(
        directional_entropy,
        k=k,
        aligned_probability=aligned_probability,
    )
    magnitude = np.linalg.norm(vectors, axis=1)
    defined = magnitude > np.finfo(float).eps

    return CommitmentAffinityResult(
        directional_affinity=q,
        velocity_magnitude=magnitude,
        commitment_strength=s,
        commitment_affinity=commitment_affinity,
        commitment_contribution=contribution,
        directional_entropy=directional_entropy,
        commitment_entropy=commitment_entropy,
        directional_specificity=specificity,
        specific_commitment=s * specificity,
        velocity_defined=defined,
    )
