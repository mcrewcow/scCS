"""Soft population commitment summaries for scCS v0.8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .affinity import normalized_entropy


@dataclass(frozen=True)
class PopulationCommitmentSummary:
    total_commitment_mass: np.ndarray
    mean_commitment_contribution: np.ndarray
    commitment_composition: np.ndarray
    pairwise_log_commitment_ratio: np.ndarray
    population_balance_entropy: float
    n_cells: int
    total_mass: float
    composition_defined: bool
    pseudocount: float

    @property
    def mean_commitment(self) -> np.ndarray:
        """Concise alias for :attr:`mean_commitment_contribution`."""
        return self.mean_commitment_contribution


def summarize_commitment(
    commitment_contribution: np.ndarray,
    *,
    pseudocount: float = 1e-12,
) -> PopulationCommitmentSummary:
    """Summarize soft per-cell fate commitment over an explicit population.

    ``total_commitment_mass`` is abundance-sensitive.  ``mean_commitment_contribution`` is
    normalized by the number of summarized cells.  No terminal-population
    count correction is applied.
    """
    values = np.asarray(commitment_contribution, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(
            "commitment_contribution must have shape (n_cells, k), with n_cells > 0 and k >= 2."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("commitment contributions must be finite and non-negative.")
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be positive and finite.")

    total = values.sum(axis=0)
    mean = values.mean(axis=0)
    total_mass = float(total.sum())
    if total_mass > 0:
        composition = total / total_mass
        balance_entropy = float(normalized_entropy(composition[None, :])[0])
        composition_defined = True
    else:
        composition = np.full(values.shape[1], np.nan)
        balance_entropy = np.nan
        composition_defined = False

    log_mean = np.log(mean + pseudocount)
    pairwise = log_mean[:, None] - log_mean[None, :]

    return PopulationCommitmentSummary(
        total_commitment_mass=total,
        mean_commitment_contribution=mean,
        commitment_composition=composition,
        pairwise_log_commitment_ratio=pairwise,
        population_balance_entropy=balance_entropy,
        n_cells=values.shape[0],
        total_mass=total_mass,
        composition_defined=composition_defined,
        pseudocount=float(pseudocount),
    )
