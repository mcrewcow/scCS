"""
bifurcation.py — Cluster-level fate map construction for scCS.

In scCS, the bifurcation point is explicitly defined by the user as a
single cluster (e.g., leiden cluster '17').  There is no automatic
fate detection — the user supplies:

    bifurcation_cluster  : the progenitor/root cluster label
    terminal_cell_types  : list of terminal fate cluster labels

This module builds a standardized FateMap from those labels, computing
centroids in the scCS star embedding space (X_sccs) and collecting
per-fate cell indices.

The FateMap is the single source of truth consumed by CommitmentScorer.score().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# FateMap dataclass
# ---------------------------------------------------------------------------

@dataclass
class FateMap:
    """Standardized description of k cell fates for commitment scoring.

    Attributes
    ----------
    bifurcation_cluster : str
        Label of the progenitor/root cluster supplied by the user.
    fate_names : list of str
        Human-readable labels for each terminal fate (length k).
    fate_centroids : np.ndarray, shape (k, 2)
        Mean 2D position of each fate's cells in the scCS embedding.
    root_centroid : np.ndarray, shape (2,)
        Mean 2D position of the bifurcation cluster cells.
        In the scCS star embedding this is always near (0, 0).
    root_cells : np.ndarray of int
        Indices of bifurcation cluster cells in adata.
    fate_cell_indices : list of np.ndarray
        Per-fate arrays of cell indices.
    arm_angles_deg : np.ndarray, shape (k,)
        Angle (degrees) of each fate's radial arm in the star embedding.
    cluster_key : str
        The obs column used for cluster labels.
    k : int
        Number of fates (read-only property).
    """
    bifurcation_cluster: str
    fate_names: List[str]
    fate_centroids: np.ndarray
    root_centroid: np.ndarray
    root_cells: np.ndarray
    fate_cell_indices: List[np.ndarray]
    arm_angles_deg: np.ndarray
    cluster_key: str

    @property
    def k(self) -> int:
        return len(self.fate_names)

    def summary(self) -> str:
        lines = [
            f"FateMap  (bifurcation_cluster='{self.bifurcation_cluster}', k={self.k})",
            f"  Cluster key : '{self.cluster_key}'",
            f"  Root cells  : {len(self.root_cells)}",
            f"  Root centroid: ({self.root_centroid[0]:.3f}, {self.root_centroid[1]:.3f})",
        ]
        for j, name in enumerate(self.fate_names):
            n = len(self.fate_cell_indices[j])
            c = self.fate_centroids[j]
            a = self.arm_angles_deg[j]
            lines.append(
                f"  Fate {j}: '{name}'  n_cells={n}  "
                f"centroid=({c[0]:.2f}, {c[1]:.2f})  arm_angle={a:.1f}°"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FateMap construction
# ---------------------------------------------------------------------------

def build_fate_map(
    adata,
    bifurcation_cluster: str,
    terminal_cell_types: List[str],
    cluster_key: str = "leiden",
    verbose: bool = True,
) -> FateMap:
    """Build a FateMap from user-supplied cluster labels.

    This is the only fate-detection strategy in scCS.  The user explicitly
    names the bifurcation cluster and all terminal fate clusters.

    Parameters
    ----------
    adata : AnnData
        Must have X_sccs in obsm (built by build_star_embedding).
    bifurcation_cluster : str
        Label of the progenitor cluster in adata.obs[cluster_key].
        Example: '17'  (leiden cluster 17)
    terminal_cell_types : list of str
        Labels of the k terminal fate clusters.
        Example: ['Monocyte', 'DC', 'Neutrophil']
    cluster_key : str
        Column in adata.obs with cluster labels.
    verbose : bool

    Returns
    -------
    FateMap
    """
    if "X_sccs" not in adata.obsm:
        raise ValueError(
            "X_sccs embedding not found in adata.obsm. "
            "Run CommitmentScorer.build_embedding() before build_fate_map()."
        )

    obs_labels = adata.obs[cluster_key].astype(str).values
    embedding = np.array(adata.obsm["X_sccs"])

    # --- Validate bifurcation cluster ---
    bif_mask = obs_labels == str(bifurcation_cluster)
    if bif_mask.sum() == 0:
        available = sorted(set(obs_labels))
        raise ValueError(
            f"Bifurcation cluster '{bifurcation_cluster}' not found in "
            f"adata.obs['{cluster_key}']. "
            f"Available labels: {available}"
        )
    root_cells = np.where(bif_mask)[0]
    root_centroid = embedding[root_cells].mean(axis=0)

    if verbose:
        print(
            f"[scCS] Bifurcation cluster '{bifurcation_cluster}': "
            f"{len(root_cells)} cells, "
            f"centroid=({root_centroid[0]:.2f}, {root_centroid[1]:.2f})"
        )

    # --- Validate and collect terminal fates ---
    fate_names = []
    fate_centroids = []
    fate_cell_indices = []
    skipped = []

    for name in terminal_cell_types:
        mask = obs_labels == str(name)
        n = mask.sum()
        if n == 0:
            warnings.warn(
                f"Terminal fate '{name}' not found in adata.obs['{cluster_key}']. "
                "Skipping.",
                stacklevel=2,
            )
            skipped.append(name)
            continue
        idx = np.where(mask)[0]
        fate_names.append(str(name))
        fate_cell_indices.append(idx)
        fate_centroids.append(embedding[idx].mean(axis=0))

        if verbose:
            c = embedding[idx].mean(axis=0)
            print(f"[scCS]   Fate '{name}': {n} cells, centroid=({c[0]:.2f}, {c[1]:.2f})")

    if len(fate_names) == 0:
        raise ValueError(
            "No valid terminal fate clusters found. "
            f"Skipped: {skipped}"
        )

    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} fate(s) not found in data: {skipped}",
            stacklevel=2,
        )

    fate_centroids = np.array(fate_centroids)

    # --- Retrieve arm angles from embedding metadata ---
    # build_star_embedding stores these in adata.uns['sccs']
    sccs_meta = adata.uns.get("sccs", {})
    stored_fates = sccs_meta.get("fate_names", [])
    stored_angles = sccs_meta.get("arm_angles_deg", None)

    arm_angles_deg = np.zeros(len(fate_names))
    if stored_angles is not None and len(stored_fates) == len(stored_angles):
        fate_to_angle = dict(zip(stored_fates, stored_angles))
        for j, name in enumerate(fate_names):
            if name in fate_to_angle:
                arm_angles_deg[j] = fate_to_angle[name]
            else:
                # Compute from centroid direction
                delta = fate_centroids[j] - root_centroid
                arm_angles_deg[j] = np.degrees(np.arctan2(delta[1], delta[0])) % 360.0
    else:
        # Compute from centroid directions
        for j in range(len(fate_names)):
            delta = fate_centroids[j] - root_centroid
            arm_angles_deg[j] = np.degrees(np.arctan2(delta[1], delta[0])) % 360.0

    fate_map = FateMap(
        bifurcation_cluster=str(bifurcation_cluster),
        fate_names=fate_names,
        fate_centroids=fate_centroids,
        root_centroid=root_centroid,
        root_cells=root_cells,
        fate_cell_indices=fate_cell_indices,
        arm_angles_deg=arm_angles_deg,
        cluster_key=cluster_key,
    )

    if verbose:
        print(f"[scCS] FateMap built: k={fate_map.k} fates")

    return fate_map
