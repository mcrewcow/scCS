"""
embedding.py — Radial star embedding for scCS.

Constructs a custom 2D layout where:
  - The bifurcation cluster (progenitor) sits at the origin (0, 0).
  - Each terminal fate population occupies its own radial arm, evenly
    spaced at 360/k degrees around the origin.
  - Within each arm, cells are ordered along the radial axis by a
    differentiation metric (pseudotime, CytoTRACE2, pathway score, etc.)
    so that less-differentiated cells are close to the center and
    more-differentiated cells are at the periphery.
  - ONLY cells belonging to the bifurcation cluster or a terminal fate
    are included.  All other populations are excluded from the embedding.

The result is stored in adata_sub.obsm['X_sccs'] on the returned subset
AnnData, and looks like a star or sunburst when plotted — one arm per
fate, radiating from the progenitor.

Velocity projection
-------------------
RNA velocity vectors (from scVelo) are projected into this custom 2D
space by computing the transition-probability-weighted displacement of
each cell in the scCS coordinate system.

Differentiation metrics supported
----------------------------------
- 'pseudotime'   : scVelo velocity_pseudotime (default)
- 'cytotrace'    : CytoTRACE2 score (column in adata.obs)
- 'custom'       : any per-cell numeric column in adata.obs
- np.ndarray     : directly supplied per-cell scores (shape n_cells,)

In all cases, higher score = more differentiated = farther from center.
If the metric is inverted (e.g., CytoTRACE2 where high = less
differentiated), pass invert_metric=True.
"""

from __future__ import annotations
import anndata
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import scvelo as scv
    _SCVELO_AVAILABLE = True
except ImportError:
    _SCVELO_AVAILABLE = False

try:
    import scanpy as sc
    _SCANPY_AVAILABLE = True
except ImportError:
    _SCANPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_star_embedding(
    adata,
    bifurcation_cluster: str,
    terminal_cell_types: List[str],
    cluster_key: str = "leiden",
    differentiation_metric: Union[str, np.ndarray] = "pseudotime",
    invert_metric: bool = False,
    arm_scale: float = 10.0,
    jitter: float = 0.3,
    seed: int = 42,
) -> "anndata.AnnData":
    """Build the radial star embedding on a subset of adata.

    Only cells belonging to the bifurcation cluster or a terminal fate
    cluster are included.  All other populations are excluded entirely.

    Parameters
    ----------
    adata : AnnData
        Full dataset.  Will NOT be modified.
    bifurcation_cluster : str
        Label of the progenitor/bifurcation cluster in adata.obs[cluster_key].
        These cells are placed at the origin.
    terminal_cell_types : list of str
        Labels of the k terminal fate populations.  Each gets one radial arm.
    cluster_key : str
        Column in adata.obs with cluster labels.
    differentiation_metric : str or np.ndarray
        How to order cells along each arm:
        - 'pseudotime'  : uses adata.obs['velocity_pseudotime'] (computed if absent)
        - 'cytotrace'   : uses adata.obs['cytotrace2_score'] (must be pre-computed)
        - any str       : uses adata.obs[differentiation_metric] directly
        - np.ndarray    : per-cell scores, shape (n_cells,) for the FULL adata
        Higher value = more differentiated = farther from center.
    invert_metric : bool
        If True, invert the metric so that high values map to the center
        (use for metrics where high = less differentiated, e.g. raw CytoTRACE2).
    arm_scale : float
        Maximum radial distance (length of each arm).
    jitter : float
        Gaussian noise added perpendicular to each arm to avoid overplotting.
    seed : int
        Random seed for jitter.

    Returns
    -------
    adata_sub : AnnData
        Subset containing ONLY bifurcation + terminal fate cells.
        Star embedding stored in adata_sub.obsm['X_sccs'].
        Metadata stored in adata_sub.uns['sccs'].
    """
    import anndata

    rng = np.random.default_rng(seed)
    obs_labels_full = adata.obs[cluster_key].astype(str).values

    # --- 0. Subset to relevant cells only ---
    keep_labels = set([str(bifurcation_cluster)] + [str(f) for f in terminal_cell_types])
    keep_mask = np.array([l in keep_labels for l in obs_labels_full])

    if keep_mask.sum() == 0:
        raise ValueError(
            f"No cells found matching bifurcation_cluster='{bifurcation_cluster}' "
            f"or terminal_cell_types={terminal_cell_types} in "
            f"adata.obs['{cluster_key}']."
        )

    # --- Resolve differentiation metric on the FULL adata BEFORE subsetting ---
    # This is critical for 'pseudotime': scVelo's velocity_pseudotime computation
    # requires the intact neighbor/velocity graph, which breaks after subsetting.
    # We resolve the metric on the full object, then slice to keep_mask.
    metric_for_sub: np.ndarray  # will always be a pre-resolved array after this block

    if isinstance(differentiation_metric, np.ndarray):
        arr = np.asarray(differentiation_metric, dtype=float).ravel()
        if len(arr) != adata.n_obs:
            raise ValueError(
                f"Custom metric array has length {len(arr)}, "
                f"expected {adata.n_obs} (full adata)."
            )
        metric_for_sub = arr[keep_mask]
    else:
        # Resolve on full adata (graph intact), then slice
        scores_full = _resolve_metric(adata, differentiation_metric, invert_metric)
        metric_for_sub = scores_full[keep_mask]

    adata_sub = adata[keep_mask].copy()
    obs_labels = adata_sub.obs[cluster_key].astype(str).values
    n_cells = adata_sub.n_obs

    print(f"[scCS] Subsetting: {keep_mask.sum()} / {adata.n_obs} cells kept")
    print(f"       ({adata.n_obs - keep_mask.sum()} cells from other populations excluded)")
    for lbl in sorted(keep_labels):
        n = (obs_labels == lbl).sum()
        role = "progenitor" if lbl == str(bifurcation_cluster) else "fate"
        print(f"       {lbl}: {n} cells ({role})")

    # --- 1. Use the pre-resolved metric (already sliced to subset) ---
    # metric_for_sub is always a np.ndarray at this point (resolved above).
    # _fill_nan handles any remaining NaNs; inversion was already applied.
    scores = _fill_nan(np.asarray(metric_for_sub, dtype=float).ravel())

    # --- 2. Compute arm directions (evenly spaced angles) ---
    k = len(terminal_cell_types)
    arm_angles_deg = np.linspace(0.0, 360.0, k, endpoint=False)
    arm_angles_rad = np.radians(arm_angles_deg)
    arm_dirs = np.stack([np.cos(arm_angles_rad), np.sin(arm_angles_rad)], axis=1)  # (k, 2)

    # --- 3. Assign each cell to an arm ---
    # Bifurcation cells -> arm index -1 (origin)
    # Terminal fate cells -> their arm index
    arm_assignment = np.full(n_cells, -1, dtype=int)
    for j, fate in enumerate(terminal_cell_types):
        mask = obs_labels == str(fate)
        arm_assignment[mask] = j

    # --- 4. Compute per-arm score ranges for normalization ---
    bif_mask_sub = obs_labels == str(bifurcation_cluster)
    arm_score_ranges = []
    for j, fate in enumerate(terminal_cell_types):
        fate_mask = obs_labels == str(fate)
        combined_mask = fate_mask | bif_mask_sub
        if combined_mask.sum() > 0:
            s = scores[combined_mask]
            arm_score_ranges.append((s.min(), s.max()))
        else:
            arm_score_ranges.append((scores.min(), scores.max()))

    # --- 5. Place cells in 2D ---
    coords = np.zeros((n_cells, 2), dtype=float)

    # Bifurcation cluster: cluster at origin with small jitter
    n_bif = bif_mask_sub.sum()
    if n_bif > 0:
        coords[bif_mask_sub] = rng.normal(0.0, jitter * 0.5, size=(n_bif, 2))

    # Fate cells: place along their assigned arm
    for j in range(k):
        cell_mask = arm_assignment == j
        if cell_mask.sum() == 0:
            continue

        s_min, s_max = arm_score_ranges[j]
        if s_max <= s_min:
            r = np.linspace(0.0, arm_scale, cell_mask.sum())
        else:
            cell_scores_arm = scores[cell_mask]
            r = (cell_scores_arm - s_min) / (s_max - s_min) * arm_scale
            r = np.clip(r, 0.0, arm_scale)

        arm_dir = arm_dirs[j]
        positions = np.outer(r, arm_dir)
        perp_dir = np.array([-arm_dir[1], arm_dir[0]])
        perp_noise = rng.normal(0.0, jitter, size=cell_mask.sum())
        positions += np.outer(perp_noise, perp_dir)
        coords[cell_mask] = positions

    # --- 6. Store in subset adata ---
    adata_sub.obsm["X_sccs"] = coords

    if "sccs" not in adata_sub.uns:
        adata_sub.uns["sccs"] = {}
    adata_sub.uns["sccs"]["arm_angles_deg"] = arm_angles_deg
    adata_sub.uns["sccs"]["arm_dirs"] = arm_dirs
    adata_sub.uns["sccs"]["arm_scale"] = arm_scale
    adata_sub.uns["sccs"]["fate_names"] = [str(f) for f in terminal_cell_types]
    adata_sub.uns["sccs"]["bifurcation_cluster"] = str(bifurcation_cluster)
    adata_sub.uns["sccs"]["cluster_key"] = cluster_key
    # Store integer indices of kept cells in the original adata (for velocity projection)
    adata_sub.uns["sccs"]["parent_indices"] = np.where(keep_mask)[0]

    adata_sub.obs["sccs_arm"] = arm_assignment
    adata_sub.obs["sccs_arm_name"] = [
        str(terminal_cell_types[a]) if a >= 0 else str(bifurcation_cluster)
        for a in arm_assignment
    ]

    print(
        f'\n[scCS] Star embedding built → adata_sub.obsm["X_sccs"] shape: {coords.shape}'
    )
    print(
        f'       Arm angles: '
        + str({str(f): round(float(a), 1)
               for f, a in zip(terminal_cell_types, arm_angles_deg)})
    )

    return adata_sub


def project_velocity_star(
    adata_sub,
    adata_full=None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project RNA velocity into the scCS star embedding space.

    Uses the transition probability matrix from the full (unsubsetted) adata
    to compute the expected displacement of each subset cell in the X_sccs
    coordinate system.

    This is necessary because subsetting breaks the velocity/neighbor graph
    matrices (they retain full-dataset dimensions). We always use the full
    graph and restrict to subset cell indices.

    Parameters
    ----------
    adata_sub : AnnData
        Subset returned by build_star_embedding(). Must have X_sccs in obsm
        and a 'sccs_parent_indices' entry in uns (set automatically).
    adata_full : AnnData, optional
        The original full dataset with intact velocity_graph in uns.
        If None, falls back to using adata_sub directly (only works if
        velocity_graph was computed on the subset).

    Returns
    -------
    vx, vy : np.ndarray, shape (n_sub_cells,)
        Velocity components in the scCS embedding.
        Also stored in adata_sub.obsm['velocity_sccs'].
    """
    if "X_sccs" not in adata_sub.obsm:
        raise ValueError(
            "X_sccs embedding not found. Run build_star_embedding() first."
        )

    coords_sub = np.array(adata_sub.obsm["X_sccs"])  # (n_sub, 2)
    n_sub = adata_sub.n_obs

    # Retrieve the parent indices (positions in full adata) stored during subsetting
    parent_idx = adata_sub.uns.get("sccs", {}).get("parent_indices", None)

    # ── Strategy 1: scVelo velocity_embedding on the full adata ──────────────
    # Run on full adata, then slice to subset rows. This is the most accurate.
    if _SCVELO_AVAILABLE and adata_full is not None and "velocity_graph" in adata_full.uns:
        if verbose:
            print("[scCS] Projecting velocity via scVelo on full adata → slicing to subset...")
        try:
            # Temporarily inject X_sccs into full adata for all cells.
            # Subset cells get their star coords; other cells get zeros (ignored after slicing).
            n_full = adata_full.n_obs
            coords_full = np.zeros((n_full, 2), dtype=float)
            if parent_idx is not None:
                coords_full[parent_idx] = coords_sub
            else:
                # Fallback: match by obs_names
                sub_names = set(adata_sub.obs_names)
                full_names = list(adata_full.obs_names)
                idx_map = [i for i, n in enumerate(full_names) if n in sub_names]
                coords_full[idx_map] = coords_sub

            adata_full.obsm["X_sccs_tmp"] = coords_full
            scv.tl.velocity_embedding(adata_full, basis="sccs_tmp")
            V_full = np.array(adata_full.obsm["velocity_sccs_tmp"])  # (n_full, 2)

            # Slice to subset
            if parent_idx is not None:
                V_sub = V_full[parent_idx]
            else:
                V_sub = V_full[idx_map]

            vx, vy = V_sub[:, 0], V_sub[:, 1]
            adata_sub.obsm["velocity_sccs"] = V_sub

            # Clean up temporary keys
            del adata_full.obsm["X_sccs_tmp"]
            if "velocity_sccs_tmp" in adata_full.obsm:
                del adata_full.obsm["velocity_sccs_tmp"]

            if verbose:
                print(f"[scCS] Velocity projected. Shape: {V_sub.shape}")
            return vx, vy

        except Exception as e:
            warnings.warn(
                f"scVelo velocity_embedding on full adata failed ({e}). "
                "Falling back to graph-based projection.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Strategy 2: graph-based projection using full adata's velocity_graph ──
    # Manually compute T[sub, :][:, sub] × coords_sub - coords_sub
    if adata_full is not None and "velocity_graph" in adata_full.uns:
        if verbose:
            print("[scCS] Using graph-based projection from full velocity_graph...")
        try:
            import scipy.sparse as sp

            T_full = adata_full.uns["velocity_graph"]
            if not sp.issparse(T_full):
                T_full = sp.csr_matrix(T_full)

            if parent_idx is None:
                sub_names = set(adata_sub.obs_names)
                full_names = list(adata_full.obs_names)
                parent_idx = np.array([i for i, n in enumerate(full_names) if n in sub_names])

            # Extract sub × sub block of the transition matrix
            T_sub = T_full[parent_idx, :][:, parent_idx]  # (n_sub, n_sub)

            # Row-normalize
            row_sums = np.array(T_sub.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            T_norm = sp.diags(1.0 / row_sums) @ T_sub

            expected = T_norm @ coords_sub        # (n_sub, 2)
            V_sub = expected - coords_sub

            vx, vy = V_sub[:, 0], V_sub[:, 1]
            adata_sub.obsm["velocity_sccs"] = V_sub

            if verbose:
                print(f"[scCS] Graph-based velocity projected. Shape: {V_sub.shape}")
            return vx, vy

        except Exception as e:
            warnings.warn(
                f"Graph-based projection from full adata failed ({e}). "
                "Falling back to subset-only projection.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── Strategy 3: last resort — use whatever graph is in adata_sub ─────────
    if verbose:
        warnings.warn(
            "No full adata provided and no compatible velocity_graph found. "
            "Using subset-only graph (may have dimension issues). "
            "Pass adata_full=adata to project_velocity() for best results.",
            RuntimeWarning,
            stacklevel=2,
        )
    vx, vy = _graph_velocity_projection(adata_sub, coords_sub, verbose=verbose)
    adata_sub.obsm["velocity_sccs"] = np.stack([vx, vy], axis=1)
    return vx, vy


def run_velocity_pipeline(
    adata,
    mode: str = "dynamical",
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    min_shared_counts: int = 20,
    verbose: bool = True,
) -> None:
    """Run the full scVelo RNA velocity pipeline.

    Requires spliced and unspliced count layers.

    Parameters
    ----------
    adata : AnnData
        Must contain layers 'spliced' and 'unspliced'.
    mode : {'dynamical', 'stochastic', 'steady_state'}
    n_top_genes : int
    n_pcs : int
    n_neighbors : int
    min_shared_counts : int
    verbose : bool
    """
    if not _SCVELO_AVAILABLE:
        raise ImportError("scvelo is required. pip install scvelo")

    missing = [l for l in ["spliced", "unspliced"] if l not in adata.layers]
    if missing:
        raise ValueError(
            f"Missing required layers: {missing}. "
            "These are generated by velocyto, STARsolo, or alevin-fry."
        )

    if verbose:
        print(f"[scCS] Running scVelo pipeline (mode='{mode}')...")

    scv.pp.filter_and_normalize(
        adata, min_shared_counts=min_shared_counts,
        n_top_genes=n_top_genes, log=True,
    )

    if "X_pca" not in adata.obsm and _SCANPY_AVAILABLE:
        sc.tl.pca(adata, n_comps=n_pcs)
    if "neighbors" not in adata.uns and _SCANPY_AVAILABLE:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    if mode == "dynamical":
        try:
            scv.tl.recover_dynamics(adata, n_jobs=-1)
            scv.tl.velocity(adata, mode="dynamical")
        except Exception as e:
            warnings.warn(
                f"Dynamical model failed ({e}). Falling back to stochastic.",
                RuntimeWarning, stacklevel=2,
            )
            scv.tl.velocity(adata, mode="stochastic")
    else:
        scv.tl.velocity(adata, mode=mode)

    scv.tl.velocity_graph(adata)

    try:
        scv.tl.velocity_pseudotime(adata)
    except Exception:
        pass

    if verbose:
        print("[scCS] Velocity pipeline complete.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_metric(
    adata,
    metric: Union[str, np.ndarray],
    invert: bool,
) -> np.ndarray:
    """Resolve differentiation metric to a per-cell float array."""
    n_cells = adata.n_obs

    if isinstance(metric, np.ndarray):
        scores = np.asarray(metric, dtype=float).ravel()
        if len(scores) != n_cells:
            raise ValueError(
                f"Custom metric array has length {len(scores)}, "
                f"expected {n_cells}."
            )

    elif metric == "pseudotime":
        if "velocity_pseudotime" not in adata.obs:
            if _SCVELO_AVAILABLE and "velocity_graph" in adata.uns:
                scv.tl.velocity_pseudotime(adata)
            else:
                warnings.warn(
                    "velocity_pseudotime not found and cannot be computed. "
                    "Falling back to uniform scores (random ordering).",
                    RuntimeWarning, stacklevel=3,
                )
                scores = np.random.default_rng(0).uniform(0, 1, n_cells)
                if invert:
                    scores = 1.0 - scores
                return _fill_nan(scores)
        scores = np.array(adata.obs["velocity_pseudotime"], dtype=float)
        # NOTE: This pseudotime was computed on the full adata.  After subsetting,
        # the caller should invoke recompute_subset_pseudotime() to get a
        # subset-local pseudotime with better arm coverage.

    elif metric == "cytotrace":
        # CytoTRACE2: look for common column names
        candidates = ["cytotrace2_score", "CytoTRACE2_Score", "cytotrace_score",
                      "CytoTRACE2", "cytotrace2"]
        found = None
        for c in candidates:
            if c in adata.obs:
                found = c
                break
        if found is None:
            raise ValueError(
                "CytoTRACE2 score not found in adata.obs. "
                f"Expected one of: {candidates}. "
                "Run CytoTRACE2 first or pass the column name as metric."
            )
        scores = np.array(adata.obs[found], dtype=float)
        # CytoTRACE2: high score = stem-like = LESS differentiated
        # So we invert by default unless user explicitly set invert=False
        # We flip the invert flag here since CytoTRACE2 is naturally inverted
        invert = not invert

    else:
        # Treat as column name in adata.obs
        if metric not in adata.obs:
            raise ValueError(
                f"Column '{metric}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        scores = np.array(adata.obs[metric], dtype=float)

    scores = _fill_nan(scores)

    if invert:
        scores = scores.max() - scores

    return scores


def _fill_nan(scores: np.ndarray) -> np.ndarray:
    """Replace NaN values with the column median."""
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        median = np.nanmedian(scores)
        scores = scores.copy()
        scores[nan_mask] = median
    return scores



def _graph_velocity_projection(
    adata,
    coords: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: project velocity using the velocity graph transition matrix.

    For each cell i, the velocity vector is the weighted average displacement
    to its neighbors, weighted by the transition probability T[i, j]:

        v_i = sum_j T[i,j] * (x_j - x_i)

    Parameters
    ----------
    adata : AnnData
    coords : np.ndarray, shape (n_cells, 2)
    verbose : bool

    Returns
    -------
    vx, vy : np.ndarray, shape (n_cells,)
    """
    import scipy.sparse as sp

    if verbose:
        print("[scCS] Using graph-based velocity projection...")

    # Try velocity_graph first, then connectivities as fallback
    T = None
    for key in ["velocity_graph", "velocity_graph_neg"]:
        if key in adata.uns:
            T_raw = adata.uns[key]
            if sp.issparse(T_raw):
                T = T_raw
            else:
                T = sp.csr_matrix(T_raw)
            break

    if T is None:
        # Last resort: use kNN connectivities
        if "connectivities" in adata.obsp:
            T = adata.obsp["connectivities"]
            if verbose:
                warnings.warn(
                    "velocity_graph not found. Using kNN connectivities as proxy.",
                    RuntimeWarning, stacklevel=2,
                )
        else:
            warnings.warn(
                "No velocity graph or connectivity matrix found. "
                "Returning zero velocity vectors.",
                RuntimeWarning, stacklevel=2,
            )
            return np.zeros(adata.n_obs), np.zeros(adata.n_obs)

    # Row-normalize transition matrix
    T = T.astype(float)
    row_sums = np.array(T.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    T_norm = sp.diags(1.0 / row_sums) @ T

    # Expected position under transition
    expected_coords = T_norm @ coords  # (n_cells, 2)
    V = expected_coords - coords       # displacement = velocity

    return V[:, 0], V[:, 1]


# ---------------------------------------------------------------------------
# Subset-local pseudotime recomputation
# ---------------------------------------------------------------------------

def recompute_subset_pseudotime(
    adata_sub,
    adata_full,
    scale_01: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Recompute velocity pseudotime on the subset's induced subgraph.

    When ``build_star_embedding`` uses ``differentiation_metric='pseudotime'``,
    the pseudotime is resolved on the full adata before subsetting.  This means
    the pseudotime range within the bifurcation+fate subset is compressed and
    non-uniform: cells that span the full differentiation axis in the subset
    may all cluster near 0 or 1 on the arm, leaving large empty stretches.

    This function extracts the velocity_graph submatrix for the subset cells,
    recomputes pseudotime locally, and optionally scales it to [0, 1].  The
    result is stored in ``adata_sub.obs['velocity_pseudotime_sub']`` and
    returned as an array.

    Call this after ``build_embedding()`` and before (or instead of) using
    the full-adata pseudotime for arm ordering.  To rebuild the embedding with
    the corrected pseudotime, pass the returned array as a custom metric::

        scorer.build_embedding(differentiation_metric='pseudotime')
        pt_sub = recompute_subset_pseudotime(scorer.adata_sub, adata)
        scorer.build_embedding(differentiation_metric=pt_sub_full)
        # where pt_sub_full is the subset scores mapped back to full adata indices

    Alternatively, use the convenience method
    ``CommitmentScorer.rebuild_embedding_with_subset_pseudotime()``.

    Parameters
    ----------
    adata_sub : AnnData
        Subset returned by ``build_star_embedding()``.  Must have
        ``uns['sccs']['parent_indices']`` set (done automatically).
    adata_full : AnnData
        Full dataset with intact ``uns['velocity_graph']``.
    scale_01 : bool
        If True (default), min-max scale the recomputed pseudotime to [0, 1]
        within the subset.  This ensures cells span the full arm length
        regardless of where the subset sits in the global pseudotime range.
        If False, the raw pseudotime values are returned (useful when you
        want to compare absolute pseudotime across conditions).
    verbose : bool

    Returns
    -------
    pt_sub : np.ndarray, shape (n_sub_cells,)
        Subset-local pseudotime, stored in
        ``adata_sub.obs['velocity_pseudotime_sub']``.
    """
    if not _SCVELO_AVAILABLE:
        raise ImportError(
            "scvelo is required for pseudotime recomputation. pip install scvelo"
        )
    if "velocity_graph" not in adata_full.uns:
        raise ValueError(
            "velocity_graph not found in adata_full.uns. "
            "Run scvelo.tl.velocity_graph() first."
        )

    import scipy.sparse as sp

    parent_idx = adata_sub.uns.get("sccs", {}).get("parent_indices", None)
    if parent_idx is None:
        # Fall back to obs_names matching
        sub_names = set(adata_sub.obs_names)
        full_names = list(adata_full.obs_names)
        parent_idx = np.array([i for i, n in enumerate(full_names) if n in sub_names])

    if verbose:
        print(
            f"[scCS] Recomputing pseudotime on subset "
            f"({len(parent_idx)} / {adata_full.n_obs} cells)..."
        )

    # Extract the sub × sub block of the velocity graph
    T_full = adata_full.uns["velocity_graph"]
    if not sp.issparse(T_full):
        T_full = sp.csr_matrix(T_full)
    T_sub = T_full[parent_idx, :][:, parent_idx]  # (n_sub, n_sub)

    # Inject the subgraph into a temporary copy of adata_sub for scVelo
    adata_tmp = adata_sub.copy()
    adata_tmp.uns["velocity_graph"] = T_sub

    # scVelo's velocity_pseudotime uses the graph to compute a diffusion-based
    # ordering.  We need neighbors connectivities too; use the subset block.
    if "connectivities" in adata_full.obsp:
        C_full = adata_full.obsp["connectivities"]
        if not sp.issparse(C_full):
            C_full = sp.csr_matrix(C_full)
        C_sub = C_full[parent_idx, :][:, parent_idx]
        adata_tmp.obsp["connectivities"] = C_sub
        adata_tmp.obsp["distances"] = C_sub  # placeholder; scVelo only needs connectivities

    try:
        scv.tl.velocity_pseudotime(adata_tmp)
        pt_sub = np.array(adata_tmp.obs["velocity_pseudotime"], dtype=float)
    except Exception as e:
        warnings.warn(
            f"scvelo.tl.velocity_pseudotime on subset failed ({e}). "
            "Falling back to diffusion pseudotime via scanpy.",
            RuntimeWarning,
            stacklevel=2,
        )
        pt_sub = _fallback_dpt(adata_tmp, verbose=verbose)

    pt_sub = _fill_nan(pt_sub)

    if scale_01:
        pt_min, pt_max = pt_sub.min(), pt_sub.max()
        if pt_max > pt_min:
            pt_sub = (pt_sub - pt_min) / (pt_max - pt_min)
        else:
            pt_sub = np.zeros_like(pt_sub)
        if verbose:
            print("[scCS] Subset pseudotime scaled to [0, 1].")

    adata_sub.obs["velocity_pseudotime_sub"] = pt_sub

    if verbose:
        print(
            f"[scCS] Subset pseudotime stored in "
            f"adata_sub.obs['velocity_pseudotime_sub']. "
            f"Range: [{pt_sub.min():.3f}, {pt_sub.max():.3f}]"
        )

    return pt_sub


def scale_metric_01(scores: np.ndarray) -> np.ndarray:
    """Min-max scale a per-cell metric to [0, 1].

    Useful for normalizing any differentiation metric (pseudotime, CytoTRACE2,
    pathway score, etc.) before passing it to ``build_star_embedding`` so that
    cells span the full arm length uniformly.

    Parameters
    ----------
    scores : np.ndarray, shape (n_cells,)
        Per-cell metric values.  NaN values are preserved.

    Returns
    -------
    scaled : np.ndarray, shape (n_cells,)
        Values in [0, 1].  Returns zeros if all values are identical.
    """
    scores = np.asarray(scores, dtype=float)
    s_min = np.nanmin(scores)
    s_max = np.nanmax(scores)
    if s_max <= s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def _fallback_dpt(adata_tmp, verbose: bool = True) -> np.ndarray:
    """Fallback: diffusion pseudotime via scanpy when scVelo fails."""
    if not _SCANPY_AVAILABLE:
        warnings.warn(
            "scanpy not available for DPT fallback. Returning uniform pseudotime.",
            RuntimeWarning, stacklevel=2,
        )
        return np.linspace(0, 1, adata_tmp.n_obs)
    try:
        import scanpy as sc
        if "connectivities" not in adata_tmp.obsp:
            sc.pp.neighbors(adata_tmp, n_neighbors=15, use_rep="X_sccs")
        # Use the cell with lowest scCS radial distance as root
        coords = np.array(adata_tmp.obsm["X_sccs"])
        radii = np.linalg.norm(coords, axis=1)
        root_idx = int(np.argmin(radii))
        adata_tmp.uns["iroot"] = root_idx
        sc.tl.dpt(adata_tmp)
        pt = np.array(adata_tmp.obs["dpt_pseudotime"], dtype=float)
        if verbose:
            print("[scCS] Used scanpy DPT as pseudotime fallback.")
        return pt
    except Exception as e2:
        warnings.warn(
            f"DPT fallback also failed ({e2}). Returning radial distance as pseudotime.",
            RuntimeWarning, stacklevel=2,
        )
        coords = np.array(adata_tmp.obsm["X_sccs"])
        return np.linalg.norm(coords, axis=1)
