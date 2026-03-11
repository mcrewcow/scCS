"""
scores.py — Core commitment score math engine for scCS.

Implements the generalized k-furcation commitment score framework,
extending the 2-state (homeostatic/activated) formulation from:

    Kriukov et al. (2025) "Single-cell transcriptome of myeloid cells in
    response to transplantation of human retinal neurons reveals reversibility
    of microglial activation"

Mathematical framework
----------------------
Given per-cell RNA velocity vectors (vx_i, vy_i) in the scCS radial embedding:

1.  magnitude_i  = sqrt(vx_i^2 + vy_i^2)                          [Eq. 1]
2.  theta_i      = atan2(vy_i, vx_i) -> [0, 360)                  [Eq. 2-3]
3.  Bin angles into N bins of width 360/N degrees                  [Eq. 4-5]
4.  M_bin(b)     = sum of magnitude_i for all cells in bin b       [Eq. 6]
5.  M_sector(j)  = sum of M_bin(b) for b in sector j              [Eq. 7]
6.  unCS(i,j)    = M_sector(i) / M_sector(j)                      [Eq. 8]
7.  nCS(i,j)     = unCS(i,j) * n_cells(j) / n_cells(i)           [Eq. 9]

Generalization to k fates:
-  CS_vec        = [M_sector(1), ..., M_sector(k)]  (raw)
-  p_vec         = CS_vec / sum(CS_vec)              (normalized)
-  H             = -sum(p_k * log(p_k))              (commitment entropy)
-  cell_scores   = dot(unit_velocity_i, unit_direction_to_fate_j)  (per-cell)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level vector math
# ---------------------------------------------------------------------------

def compute_magnitudes(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Euclidean norm of 2D velocity vectors (Eq. 1).

    Parameters
    ----------
    vx, vy : array-like, shape (n_cells,)
        x and y components of velocity vectors.

    Returns
    -------
    magnitudes : np.ndarray, shape (n_cells,)
        Non-negative magnitudes; NaN inputs yield NaN.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    return np.sqrt(vx ** 2 + vy ** 2)


def compute_angles(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Angle of each velocity vector in [0, 360) degrees (Eq. 2-3).

    Parameters
    ----------
    vx, vy : array-like, shape (n_cells,)

    Returns
    -------
    angles_deg : np.ndarray, shape (n_cells,)
        Angles in degrees, range [0, 360).  NaN for zero-magnitude vectors.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    angles_rad = np.arctan2(vy, vx)
    angles_deg = np.degrees(angles_rad) % 360.0
    zero_mask = (vx == 0) & (vy == 0)
    angles_deg[zero_mask] = np.nan
    return angles_deg


def bin_angles(
    angles_deg: np.ndarray,
    magnitudes: np.ndarray,
    n_bins: int = 36,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize angles and accumulate magnitudes per bin (Eq. 4-6).

    Parameters
    ----------
    angles_deg : np.ndarray, shape (n_cells,)
        Angles in [0, 360).  NaN values are ignored.
    magnitudes : np.ndarray, shape (n_cells,)
        Per-cell magnitudes.  NaN values are ignored.
    n_bins : int
        Number of angular bins.  Default 36 (10° each, as in manuscript).

    Returns
    -------
    bin_edges : np.ndarray, shape (n_bins + 1,)
        Bin edge angles in degrees.
    M_bin : np.ndarray, shape (n_bins,)
        Cumulative magnitude per bin.
    """
    bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
    M_bin = np.zeros(n_bins, dtype=float)

    valid = ~(np.isnan(angles_deg) | np.isnan(magnitudes))
    a = angles_deg[valid]
    m = magnitudes[valid]

    bin_indices = np.searchsorted(bin_edges[1:], a, side="right")
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    np.add.at(M_bin, bin_indices, m)

    return bin_edges, M_bin


# ---------------------------------------------------------------------------
# Sector definition helpers
# ---------------------------------------------------------------------------

def equal_sectors(k: int, n_bins: int = 36) -> List[List[int]]:
    """Divide n_bins into k equal contiguous sectors.

    Parameters
    ----------
    k : int
        Number of fates / sectors.
    n_bins : int
        Total number of angular bins.

    Returns
    -------
    sectors : list of lists
        Each inner list contains the bin indices belonging to that sector.
    """
    if n_bins % k != 0:
        warnings.warn(
            f"n_bins={n_bins} is not evenly divisible by k={k}. "
            "Sectors will have unequal sizes.",
            stacklevel=2,
        )
    bins_per_sector = n_bins / k
    sectors = []
    for j in range(k):
        start = int(round(j * bins_per_sector))
        end = int(round((j + 1) * bins_per_sector))
        sectors.append(list(range(start, end)))
    return sectors


def centroid_sectors(
    fate_centroids: np.ndarray,
    root_centroid: np.ndarray,
    n_bins: int = 36,
) -> Tuple[List[List[int]], np.ndarray]:
    """Define sectors anchored to fate centroid directions from the root.

    Each sector is centered on the angle from the root to the corresponding
    fate centroid.  Sector boundaries are placed at the midpoints between
    adjacent fate angles.

    Parameters
    ----------
    fate_centroids : np.ndarray, shape (k, 2)
        2D embedding coordinates of each fate centroid.
    root_centroid : np.ndarray, shape (2,)
        2D embedding coordinate of the root / progenitor centroid.
    n_bins : int
        Number of angular bins.

    Returns
    -------
    sectors : list of k lists of bin indices
    fate_angles : np.ndarray, shape (k,)
        Central angle (degrees) for each fate.
    """
    fate_centroids = np.asarray(fate_centroids, dtype=float)
    root_centroid = np.asarray(root_centroid, dtype=float)

    deltas = fate_centroids - root_centroid  # (k, 2)
    fate_angles = np.degrees(np.arctan2(deltas[:, 1], deltas[:, 0])) % 360.0

    k = len(fate_angles)
    sort_idx = np.argsort(fate_angles)
    sorted_angles = fate_angles[sort_idx]

    bin_width = 360.0 / n_bins
    bin_centers = np.arange(n_bins) * bin_width + bin_width / 2.0

    sectors: List[List[int]] = [[] for _ in range(k)]
    for b, center in enumerate(bin_centers):
        dists = np.array([
            min(abs(center - fa), 360.0 - abs(center - fa))
            for fa in sorted_angles
        ])
        nearest = int(np.argmin(dists))
        sectors[nearest].append(b)

    inv_sort = np.argsort(sort_idx)
    sectors = [sectors[inv_sort[j]] for j in range(k)]

    return sectors, fate_angles


# ---------------------------------------------------------------------------
# Sector magnitude accumulation
# ---------------------------------------------------------------------------

def compute_sector_magnitudes(
    M_bin: np.ndarray,
    sectors: List[List[int]],
) -> np.ndarray:
    """Sum M_bin values within each sector (Eq. 7).

    Parameters
    ----------
    M_bin : np.ndarray, shape (n_bins,)
    sectors : list of k lists of bin indices

    Returns
    -------
    M_sector : np.ndarray, shape (k,)
    """
    return np.array([M_bin[s].sum() for s in sectors], dtype=float)


# ---------------------------------------------------------------------------
# Commitment score computations
# ---------------------------------------------------------------------------

def compute_unCS(M_sector_i: float, M_sector_j: float) -> float:
    """Unnormalized commitment score of fate i relative to fate j (Eq. 8).

    unCS > 1  =>  population is more committed to fate i than fate j.

    Parameters
    ----------
    M_sector_i, M_sector_j : float
        Cumulative magnitudes for fates i and j.

    Returns
    -------
    float  (inf if M_sector_j == 0)
    """
    if M_sector_j == 0:
        return np.inf
    return float(M_sector_i / M_sector_j)


def compute_nCS(
    M_sector_i: float,
    M_sector_j: float,
    n_cells_i: int,
    n_cells_j: int,
) -> float:
    """Cell-number-normalized commitment score (Eq. 9).

    nCS = (M_sector_i / M_sector_j) * (n_cells_j / n_cells_i)

    Parameters
    ----------
    M_sector_i, M_sector_j : float
    n_cells_i, n_cells_j : int
        Number of cells in each population / trajectory arm.

    Returns
    -------
    float
    """
    if M_sector_j == 0 or n_cells_i == 0:
        return np.inf
    return float((M_sector_i / M_sector_j) * (n_cells_j / n_cells_i))


def compute_commitment_vector(M_sector: np.ndarray) -> np.ndarray:
    """Normalize sector magnitudes to a probability-like commitment vector.

    Parameters
    ----------
    M_sector : np.ndarray, shape (k,)

    Returns
    -------
    p_vec : np.ndarray, shape (k,)
        Sums to 1.  All-zero input returns uniform distribution.
    """
    total = M_sector.sum()
    if total == 0:
        return np.ones(len(M_sector)) / len(M_sector)
    return M_sector / total


def compute_commitment_entropy(p_vec: np.ndarray) -> float:
    """Shannon entropy of the commitment vector, normalized to [0, 1].

    H = 0  =>  fully committed to one fate (all mass in one sector).
    H = 1  =>  maximally uncertain (uniform across all fates).

    Parameters
    ----------
    p_vec : np.ndarray, shape (k,)
        Normalized commitment vector (sums to 1).

    Returns
    -------
    float in [0, 1]
    """
    k = len(p_vec)
    if k <= 1:
        return 0.0
    p = p_vec[p_vec > 0]
    H_raw = -np.sum(p * np.log(p))
    H_max = np.log(k)
    return float(H_raw / H_max)


def compute_pairwise_cs_matrix(
    M_sector: np.ndarray,
    n_cells_per_fate: Optional[np.ndarray] = None,
    normalized: bool = True,
) -> np.ndarray:
    """Compute full k x k pairwise commitment score matrix.

    Entry [i, j] = CS(i relative to j).
    Diagonal is 1.0.

    Parameters
    ----------
    M_sector : np.ndarray, shape (k,)
    n_cells_per_fate : np.ndarray, shape (k,), optional
        If provided and normalized=True, computes nCS; else unCS.
    normalized : bool

    Returns
    -------
    cs_matrix : np.ndarray, shape (k, k)
    """
    k = len(M_sector)
    cs_matrix = np.ones((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            if normalized and n_cells_per_fate is not None:
                cs_matrix[i, j] = compute_nCS(
                    M_sector[i], M_sector[j],
                    int(n_cells_per_fate[i]), int(n_cells_per_fate[j]),
                )
            else:
                cs_matrix[i, j] = compute_unCS(M_sector[i], M_sector[j])
    return cs_matrix


# ---------------------------------------------------------------------------
# Per-cell fate affinity scores
# ---------------------------------------------------------------------------

def compute_cell_scores(
    vx: np.ndarray,
    vy: np.ndarray,
    fate_centroids: np.ndarray,
    root_centroid: np.ndarray,
) -> np.ndarray:
    """Per-cell fate affinity: cosine similarity of velocity to fate direction.

    For each cell i and fate j, computes:
        score(i, j) = dot(unit_v_i, unit_d_j)
    where unit_d_j is the unit vector from root_centroid to fate_centroid_j.

    Scores are then shifted to [0, 1] via (score + 1) / 2 and row-normalized.

    Parameters
    ----------
    vx, vy : np.ndarray, shape (n_cells,)
    fate_centroids : np.ndarray, shape (k, 2)
    root_centroid : np.ndarray, shape (2,)

    Returns
    -------
    cell_scores : np.ndarray, shape (n_cells, k)
        Per-cell affinity for each fate, row-normalized to sum to 1.
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    fate_centroids = np.asarray(fate_centroids, dtype=float)
    root_centroid = np.asarray(root_centroid, dtype=float)

    mag = compute_magnitudes(vx, vy)
    with np.errstate(invalid="ignore", divide="ignore"):
        uvx = np.where(mag > 0, vx / mag, 0.0)
        uvy = np.where(mag > 0, vy / mag, 0.0)

    deltas = fate_centroids - root_centroid  # (k, 2)
    delta_mag = np.linalg.norm(deltas, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_dirs = np.where(delta_mag > 0, deltas / delta_mag, 0.0)  # (k, 2)

    V = np.stack([uvx, uvy], axis=1)  # (n_cells, 2)
    scores = V @ unit_dirs.T          # (n_cells, k)

    scores = (scores + 1.0) / 2.0

    row_sums = scores.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(row_sums > 0, scores / row_sums, 1.0 / scores.shape[1])

    return scores


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CommitmentScoreResult:
    """Container for all commitment score outputs.

    Attributes
    ----------
    fate_names : list of str
    M_bin : np.ndarray, shape (n_bins,)
        Cumulative magnitude per angular bin.
    bin_edges : np.ndarray, shape (n_bins + 1,)
    sectors : list of k lists of bin indices
    M_sector : np.ndarray, shape (k,)
        Cumulative magnitude per fate sector.
    n_cells_per_fate : np.ndarray, shape (k,)
    commitment_vector : np.ndarray, shape (k,)
        Normalized (sums to 1).
    commitment_entropy : float
        Normalized Shannon entropy in [0, 1].
    pairwise_unCS : np.ndarray, shape (k, k)
    pairwise_nCS : np.ndarray, shape (k, k)
    cell_scores : np.ndarray, shape (n_cells, k), optional
    fate_angles : np.ndarray, shape (k,), optional
        Angle (degrees) of each fate axis in the radial embedding.
    dominant_fate : str
        Fate with highest M_sector.
    """
    fate_names: List[str]
    M_bin: np.ndarray
    bin_edges: np.ndarray
    sectors: List[List[int]]
    M_sector: np.ndarray
    n_cells_per_fate: np.ndarray
    commitment_vector: np.ndarray
    commitment_entropy: float
    pairwise_unCS: np.ndarray
    pairwise_nCS: np.ndarray
    cell_scores: Optional[np.ndarray] = None
    fate_angles: Optional[np.ndarray] = None
    cell_obs_names: Optional[np.ndarray] = None  # obs_names of adata_sub rows

    @property
    def k(self) -> int:
        return len(self.fate_names)

    @property
    def dominant_fate(self) -> str:
        return self.fate_names[int(np.argmax(self.M_sector))]

    def to_dataframe(self) -> pd.DataFrame:
        """Summary DataFrame with one row per fate."""
        rows = []
        for j, name in enumerate(self.fate_names):
            rows.append({
                "fate": name,
                "M_sector": self.M_sector[j],
                "n_cells": self.n_cells_per_fate[j],
                "commitment_fraction": self.commitment_vector[j],
            })
        return pd.DataFrame(rows)

    def pairwise_to_dataframe(self, normalized: bool = True) -> pd.DataFrame:
        """Pairwise CS matrix as a labeled DataFrame."""
        mat = self.pairwise_nCS if normalized else self.pairwise_unCS
        return pd.DataFrame(mat, index=self.fate_names, columns=self.fate_names)

    def summary(self) -> str:
        lines = [
            "=== CommitmentScoreResult ===",
            f"  Fates ({len(self.fate_names)}): {', '.join(self.fate_names)}",
            f"  Dominant fate: {self.dominant_fate}",
            f"  Commitment entropy (normalized): {self.commitment_entropy:.4f}",
            "",
            "  Commitment vector (normalized):",
        ]
        for name, p in zip(self.fate_names, self.commitment_vector):
            lines.append(f"    {name}: {p:.4f}")
        lines += ["", "  Pairwise nCS matrix:"]
        df = self.pairwise_to_dataframe(normalized=True)
        lines.append(df.to_string())
        return "\n".join(lines)
