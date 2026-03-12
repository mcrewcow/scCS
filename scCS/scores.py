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
-  H_pop         = -sum(p_k * log(p_k)) / log(k)    (population entropy)
-  H_cell_j      = mean_i[ h_bin(s_ij) ]            (per-fate cell entropy, k values)
-  H_nn_i        = H( mean_{n in NN(i)}(cell_scores[n]) )  (NN-smoothed per-cell entropy)
-  cell_scores   = dot(unit_velocity_i, unit_direction_to_fate_j)  (per-cell)

Entropy notes
-------------
Three complementary entropy metrics are provided:

``compute_population_entropy(p_vec)``  →  float
    Entropy of the aggregate commitment vector (M_sector / sum(M_sector)).
    Single scalar.  Measures how evenly total velocity mass is distributed.
    **Limitation**: high for any balanced split, even if every cell is decisive.

``compute_per_fate_cell_entropy(cell_scores)``  →  ndarray shape (k,)
    For each fate j: binary entropy of each cell's affinity toward j,
    averaged over all cells.  h_j = mean_i[ H_bin(s_ij, 1-s_ij) ].
    Tells you per-fate how individually decisive cells are toward that fate.
    Low h_j = cells are sharply committed (or sharply not committed) to fate j.
    High h_j = cells are ambiguous about fate j.

``compute_nn_cell_entropy(cell_scores, coords, k_nn)``  →  ndarray shape (n_cells,)
    For each cell: average cell_scores over its k_nn nearest neighbors in
    the scCS embedding (X_sccs), then compute full k-way entropy on the
    smoothed scores.  Spatially local smoothing removes single-cell noise.
    Use the elbow plots (plot_nn_entropy_elbow) to choose k_nn.
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


def compute_population_entropy(p_vec: np.ndarray) -> float:
    """Shannon entropy of the aggregate commitment vector, normalized to [0, 1].

    Operates on the *population-level* commitment vector
    ``p_vec = M_sector / sum(M_sector)``, which reflects how total velocity
    mass is distributed across fate sectors.

    H_pop = 0  =>  all velocity mass concentrated in one sector.
    H_pop = 1  =>  velocity mass uniformly spread across all sectors.

    .. warning::
        This metric can be misleading when cells are split between fates.
        A population where 50 % of cells strongly commit to fate A and 50 %
        strongly commit to fate B will yield H_pop ≈ 1 (maximum uncertainty),
        even though every individual cell is decisive.  Use
        :func:`compute_mean_cell_entropy` as the primary commitment metric.

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


# Backward-compatible alias — emits no warning at function level;
# the DeprecationWarning is raised by CommitmentScoreResult.commitment_entropy
compute_commitment_entropy = compute_population_entropy


def compute_mean_cell_entropy(cell_scores: np.ndarray) -> float:
    """Mean per-cell Shannon entropy of fate-affinity scores, normalized to [0, 1].

    For each cell *i*, computes the normalized Shannon entropy of its
    row-normalized fate-affinity vector ``s_i = cell_scores[i, :]``::

        h_i = -sum_j( s_ij * log(s_ij) ) / log(k)

    and returns the mean over all cells::

        H_cell = mean_i( h_i )

    This is the **recommended primary entropy metric** because it measures
    individual cell commitment uncertainty rather than population-level
    velocity-mass balance.

    Interpretation
    --------------
    H_cell ≈ 0  =>  cells are individually decisive (each cell strongly
                    favors one fate).  Occurs in committed populations
                    regardless of whether cells split between fates.
    H_cell ≈ 1  =>  cells are individually undecided (each cell's velocity
                    points equally toward all fates).  Occurs in genuinely
                    uncommitted / progenitor-like populations.

    Contrast with :func:`compute_population_entropy`
    -------------------------------------------------
    A population split 50/50 between two strongly committed sub-groups gives:
    - H_pop ≈ 1.0  (misleadingly high — velocity mass is balanced)
    - H_cell ≈ 0.0 (correctly low — each cell is individually committed)

    Parameters
    ----------
    cell_scores : np.ndarray, shape (n_cells, k)
        Per-cell fate-affinity matrix, row-normalized to sum to 1.
        Typically the output of :func:`compute_cell_scores`.

    Returns
    -------
    float in [0, 1]
        Mean normalized per-cell entropy.  Returns 0.0 for a single cell
        or single fate.
    """
    cell_scores = np.asarray(cell_scores, dtype=float)
    n_cells, k = cell_scores.shape
    if k <= 1 or n_cells == 0:
        return 0.0
    H_max = np.log(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.where(cell_scores > 0, np.log(cell_scores), 0.0)
    per_cell_H = -np.sum(cell_scores * log_s, axis=1) / H_max
    return float(np.mean(per_cell_H))


def compute_per_fate_cell_entropy(cell_scores: np.ndarray) -> np.ndarray:
    """Per-fate mean binary cell entropy of fate-affinity scores.

    For each fate *j*, treats each cell's affinity score ``s_ij`` as a
    binary distribution ``[s_ij, 1 - s_ij]`` and computes the normalized
    binary Shannon entropy, then averages over all cells::

        h_j = mean_i[ H_bin(s_ij) ]
            = mean_i[ -(s_ij * log(s_ij) + (1-s_ij) * log(1-s_ij)) / log(2) ]

    Interpretation
    --------------
    h_j ≈ 0  =>  cells are sharply decisive about fate j (either strongly
                 committed or strongly not committed).
    h_j ≈ 1  =>  cells are ambiguous about fate j (scores cluster near 0.5).

    This is the per-fate analogue of :func:`compute_mean_cell_entropy`.

    Parameters
    ----------
    cell_scores : np.ndarray, shape (n_cells, k)
        Per-cell fate-affinity matrix, row-normalized to sum to 1.
        Typically the output of :func:`compute_cell_scores`.

    Returns
    -------
    per_fate_entropy : np.ndarray, shape (k,)
        Mean binary entropy for each fate.  Returns zeros for k=0 or n=0.
    """
    cell_scores = np.asarray(cell_scores, dtype=float)
    if cell_scores.ndim != 2 or cell_scores.size == 0:
        return np.zeros(cell_scores.shape[1] if cell_scores.ndim == 2 else 0)
    n_cells, k = cell_scores.shape
    if n_cells == 0 or k == 0:
        return np.zeros(k)

    s = np.clip(cell_scores, 1e-15, 1 - 1e-15)   # avoid log(0)
    one_minus_s = 1.0 - s
    # Binary entropy per cell per fate, normalized by log(2)
    h = -(s * np.log(s) + one_minus_s * np.log(one_minus_s)) / np.log(2)
    return h.mean(axis=0)   # shape (k,)


def compute_nn_cell_entropy(
    cell_scores: np.ndarray,
    coords: np.ndarray,
    k_nn: int,
) -> np.ndarray:
    """NN-smoothed per-cell commitment entropy in the scCS embedding.

    For each cell *i*:
    1. Find its ``k_nn`` nearest neighbors in ``coords`` (X_sccs, 2D).
    2. Average ``cell_scores`` over those neighbors (including cell *i* itself).
    3. Compute normalized k-way Shannon entropy on the smoothed scores.

    This removes single-cell velocity noise while preserving local commitment
    structure.  Use :func:`scCS.plot.plot_nn_entropy_elbow` to choose k_nn.

    Parameters
    ----------
    cell_scores : np.ndarray, shape (n_cells, k)
        Per-cell fate-affinity matrix from :func:`compute_cell_scores`.
    coords : np.ndarray, shape (n_cells, 2)
        2D scCS embedding coordinates (``adata_sub.obsm['X_sccs']``).
    k_nn : int
        Number of nearest neighbors to average over (excluding self).
        Self is always included, so the effective window is k_nn + 1 cells.

    Returns
    -------
    nn_entropy : np.ndarray, shape (n_cells,)
        Normalized Shannon entropy of the NN-smoothed fate scores per cell,
        in [0, 1].
    """
    from sklearn.neighbors import NearestNeighbors

    cell_scores = np.asarray(cell_scores, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n_cells, k = cell_scores.shape

    if n_cells == 0 or k <= 1:
        return np.zeros(n_cells)

    # k_nn neighbors + self; cap at n_cells
    n_query = min(k_nn + 1, n_cells)
    nbrs = NearestNeighbors(n_neighbors=n_query, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)   # (n_cells, n_query), first col = self

    # Average cell_scores over neighborhood (including self)
    smoothed = cell_scores[indices].mean(axis=1)   # (n_cells, k)

    # Normalize rows to sum to 1 (averaging may break normalization slightly)
    row_sums = smoothed.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(row_sums > 0, smoothed / row_sums, 1.0 / k)

    # Normalized k-way Shannon entropy per cell
    H_max = np.log(k)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.where(smoothed > 0, np.log(smoothed), 0.0)
    nn_entropy = -np.sum(smoothed * log_s, axis=1) / H_max   # (n_cells,)

    return nn_entropy


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
    population_entropy : float
        Normalized Shannon entropy of the aggregate commitment vector in [0, 1].
        Single scalar.  See :func:`compute_population_entropy`.
    mean_cell_entropy : float
        Mean normalized per-cell Shannon entropy in [0, 1].
        See :func:`compute_mean_cell_entropy`.
        NaN when compute_cell_level=False.
    per_fate_entropy : np.ndarray, shape (k,)
        Mean binary cell entropy for each fate individually.
        per_fate_entropy[j] = mean over cells of H_bin(s_ij, 1-s_ij).
        See :func:`compute_per_fate_cell_entropy`.
        All-NaN array when compute_cell_level=False.
    pairwise_unCS : np.ndarray, shape (k, k)
    pairwise_nCS : np.ndarray, shape (k, k)
    cell_scores : np.ndarray, shape (n_cells, k), optional
    fate_angles : np.ndarray, shape (k,), optional
        Angle (degrees) of each fate axis in the radial embedding.
    nn_cell_entropy : np.ndarray, shape (n_cells,), optional
        NN-smoothed per-cell entropy.  Set when k_nn > 0 in score().
        Also written to adata_sub.obs['cs_nn_entropy'].
    nn_k : int, optional
        The k_nn value used to compute nn_cell_entropy.
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
    population_entropy: float
    mean_cell_entropy: float
    per_fate_entropy: np.ndarray          # shape (k,) — binary entropy per fate
    pairwise_unCS: np.ndarray
    pairwise_nCS: np.ndarray
    cell_scores: Optional[np.ndarray] = None
    fate_angles: Optional[np.ndarray] = None
    cell_obs_names: Optional[np.ndarray] = None
    nn_cell_entropy: Optional[np.ndarray] = None   # shape (n_cells,), set when k_nn>0
    nn_k: Optional[int] = None                     # k_nn used to compute nn_cell_entropy

    # ------------------------------------------------------------------
    # Backward-compatibility alias
    # ------------------------------------------------------------------

    @property
    def commitment_entropy(self) -> float:
        """Alias for ``population_entropy`` (deprecated, use ``mean_cell_entropy``)."""
        warnings.warn(
            "CommitmentScoreResult.commitment_entropy is deprecated. "
            "Use .mean_cell_entropy (recommended) or .population_entropy instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.population_entropy

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
            "",
            "  Entropy metrics:",
            f"    Population entropy:           {self.population_entropy:.4f}"
            + "  [aggregate velocity-mass balance]",
        ]
        if not np.isnan(self.mean_cell_entropy):
            lines.append(
                f"    Mean cell entropy:            {self.mean_cell_entropy:.4f}"
                + "  [per-cell average, k-way]"
            )
            lines.append("    Per-fate cell entropy:")
            for name, h in zip(self.fate_names, self.per_fate_entropy):
                lines.append(f"      {name}: {h:.4f}")
        else:
            lines.append(
                "    Mean cell entropy:            [not computed — set compute_cell_level=True]"
            )
        if self.nn_cell_entropy is not None:
            lines.append(
                f"    NN-smoothed entropy (k={self.nn_k}):  "
                f"mean={self.nn_cell_entropy.mean():.4f}  "
                f"[per-cell, stored in adata_sub.obs['cs_nn_entropy']]"
            )
        lines += [
            "",
            "  Commitment vector (normalized):",
        ]
        for name, p in zip(self.fate_names, self.commitment_vector):
            lines.append(f"    {name}: {p:.4f}")
        lines += ["", "  Pairwise nCS matrix:"]
        df = self.pairwise_to_dataframe(normalized=True)
        lines.append(df.to_string())
        return "\n".join(lines)
