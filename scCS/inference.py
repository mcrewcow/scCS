"""Replicate-aware statistical utilities for scCS v0.8.

The formal unit of inference is the biological replicate.  Cell-level values
may be resampled within replicate during a hierarchical bootstrap, but cells
are never permuted as if they were independent experimental replicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PermutationTestResult:
    effect: float
    statistic: float
    pvalue: float
    method: str
    n_permutations: int
    n_group_a: int
    n_group_b: int


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_bootstrap: int


def _finite_1d(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def two_group_mean_difference(
    values: Sequence[float],
    labels: Sequence[object],
    group_a: object,
    group_b: object,
) -> float:
    """Return ``mean(group_b) - mean(group_a)``."""
    x = _finite_1d(values, name="values")
    g = np.asarray(labels).astype(str)
    if g.ndim != 1 or len(g) != len(x):
        raise ValueError("labels must be one-dimensional and match values.")
    a = g == str(group_a)
    b = g == str(group_b)
    if not a.any() or not b.any():
        raise ValueError("Both requested groups must contain at least one value.")
    return float(x[b].mean() - x[a].mean())


def two_group_permutation_test(
    values: Sequence[float],
    labels: Sequence[object],
    group_a: object,
    group_b: object,
    *,
    n_permutations: int = 9999,
    max_exact: int = 100_000,
    random_state: int = 0,
) -> PermutationTestResult:
    """Two-sided replicate-label permutation test for a mean difference.

    Exact enumeration is used when the number of unique allocations is no
    greater than ``max_exact``.  Otherwise a Monte Carlo p-value with the
    ``(b + 1) / (B + 1)`` correction is returned.
    """
    x = _finite_1d(values, name="values")
    g = np.asarray(labels).astype(str)
    if g.ndim != 1 or len(g) != len(x):
        raise ValueError("labels must be one-dimensional and match values.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    if max_exact < 1:
        raise ValueError("max_exact must be positive.")

    keep = np.isin(g, [str(group_a), str(group_b)])
    x = x[keep]
    g = g[keep]
    n_a = int(np.sum(g == str(group_a)))
    n_b = int(np.sum(g == str(group_b)))
    if n_a < 2 or n_b < 2:
        raise ValueError("Replicate-level permutation requires at least two replicates per group.")

    observed = two_group_mean_difference(x, g, group_a, group_b)
    absolute_observed = abs(observed)
    n_total = len(x)
    n_allocations = comb(n_total, n_b)
    tolerance = 1e-15

    if n_allocations <= max_exact:
        exceed = 0
        total = 0
        indices = np.arange(n_total)
        for b_indices in combinations(indices, n_b):
            b_mask = np.zeros(n_total, dtype=bool)
            b_mask[list(b_indices)] = True
            statistic = float(x[b_mask].mean() - x[~b_mask].mean())
            exceed += abs(statistic) >= absolute_observed - tolerance
            total += 1
        pvalue = exceed / total
        return PermutationTestResult(
            effect=observed,
            statistic=observed,
            pvalue=float(pvalue),
            method="exact",
            n_permutations=total,
            n_group_a=n_a,
            n_group_b=n_b,
        )

    rng = np.random.default_rng(random_state)
    exceed = 0
    base = np.r_[np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)]
    for _ in range(n_permutations):
        permuted = rng.permutation(base).astype(bool)
        statistic = float(x[permuted].mean() - x[~permuted].mean())
        exceed += abs(statistic) >= absolute_observed - tolerance
    pvalue = (exceed + 1) / (n_permutations + 1)
    return PermutationTestResult(
        effect=observed,
        statistic=observed,
        pvalue=float(pvalue),
        method="monte_carlo",
        n_permutations=int(n_permutations),
        n_group_a=n_a,
        n_group_b=n_b,
    )


def one_way_f_statistic(values: Sequence[float], labels: Sequence[object]) -> float:
    """Return the ordinary one-way ANOVA F statistic on replicate summaries."""
    x = _finite_1d(values, name="values")
    g = np.asarray(labels).astype(str)
    if g.ndim != 1 or len(g) != len(x):
        raise ValueError("labels must be one-dimensional and match values.")
    groups = [x[g == name] for name in np.unique(g)]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        raise ValueError("Each condition must contain at least two replicates.")

    grand = float(x.mean())
    ss_between = sum(len(group) * (float(group.mean()) - grand) ** 2 for group in groups)
    ss_within = sum(float(np.sum((group - group.mean()) ** 2)) for group in groups)
    df_between = len(groups) - 1
    df_within = len(x) - len(groups)
    if df_within <= 0:
        raise ValueError("Not enough replicate degrees of freedom.")
    if ss_within <= np.finfo(float).eps:
        return float("inf") if ss_between > np.finfo(float).eps else 0.0
    return float((ss_between / df_between) / (ss_within / df_within))


def one_way_permutation_test(
    values: Sequence[float],
    labels: Sequence[object],
    *,
    n_permutations: int = 9999,
    random_state: int = 0,
) -> PermutationTestResult:
    """Monte Carlo replicate-label permutation test using a one-way F statistic."""
    x = _finite_1d(values, name="values")
    g = np.asarray(labels).astype(str)
    if g.ndim != 1 or len(g) != len(x):
        raise ValueError("labels must be one-dimensional and match values.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    unique, counts = np.unique(g, return_counts=True)
    if len(unique) < 3:
        raise ValueError("One-way multi-condition testing requires at least three groups.")
    if np.any(counts < 2):
        raise ValueError("Each condition must contain at least two replicates.")

    observed = one_way_f_statistic(x, g)
    rng = np.random.default_rng(random_state)
    exceed = 0
    for _ in range(n_permutations):
        statistic = one_way_f_statistic(x, rng.permutation(g))
        exceed += statistic >= observed - 1e-15
    pvalue = (exceed + 1) / (n_permutations + 1)
    return PermutationTestResult(
        effect=float("nan"),
        statistic=float(observed),
        pvalue=float(pvalue),
        method="monte_carlo",
        n_permutations=int(n_permutations),
        n_group_a=int(counts.min()),
        n_group_b=int(counts.max()),
    )


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    """Holm family-wise error correction."""
    p = np.asarray(list(pvalues), dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite and lie in [0, 1].")
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def percentile_interval(
    samples: Sequence[float],
    *,
    confidence_level: float = 0.95,
    estimate: float | None = None,
) -> BootstrapInterval:
    """Construct a percentile interval from finite bootstrap samples."""
    values = _finite_1d(samples, name="samples")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1).")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = float(np.mean(values)) if estimate is None else float(estimate)
    return BootstrapInterval(
        estimate=point,
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        n_bootstrap=len(values),
    )
