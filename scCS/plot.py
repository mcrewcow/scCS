"""
plot.py — Publication-quality visualizations for scCS.

Primary visualization: plot_star_embedding()
  Radial star layout with one arm per fate, cells colored by cluster,
  pseudotime, fate affinity, or commitment entropy.  Arm axes are drawn
  with fate labels at the tips.

Additional plots:
  plot_rose()                — polar rose of velocity magnitude by direction
  plot_pairwise_cs()         — heatmap of pairwise nCS/unCS matrix
  plot_commitment_bar()      — unCS/nCS bar chart per fate pair
  plot_commitment_heatmap()  — per-cell fate affinity heatmap
  plot_expression_trends()   — CellRank-style gene expression vs pseudotime
  plot_subset_comparison()   — multi-subset CS comparison
  plot_nn_entropy_elbow()    — elbow plot for k_nn selection

Multi-condition plots (PairScorer + MultiScorer):
  plot_delta_cs_heatmap()          — ΔCS heatmap with CI annotation
  plot_compare_conditions_bar()    — grouped bar chart of nCS per condition
  plot_commitment_vector_radar()   — radar chart of commitment vectors
  plot_omnibus_summary()           — fates × conditions heatmap with omnibus significance
  plot_posthoc_heatmap()           — condition × condition post-hoc p-value heatmap
  plot_pairwise_delta_grid()       — grid of ΔCS heatmaps for all condition pairs

Color maps
----------
All plot functions accept an optional ``color_map`` dict mapping fate name
to a hex color string.  Pass this to preserve your original cluster colors
from scanpy/Seurat across all scCS plots.  Progenitor cells always use
PROGENITOR_COLOR (gray) regardless of color_map.

Example::

    # Extract colors from scanpy
    color_map = dict(zip(
        adata.obs['cell_type'].cat.categories,
        adata.uns['cell_type_colors'],
    ))
    scorer.plot_star(result, color_map=color_map)

All plots use seaborn ticks theme.
Figures are returned as matplotlib Figure objects.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

from .scores import CommitmentScoreResult

# Colorblind-friendly palette (Wong 2011)
FATE_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

PROGENITOR_COLOR = "#AAAAAA"  # neutral grey for bifurcation cluster

# Condition colors — distinct from fate colors, also colorblind-safe (Wong 2011 reordered)
CONDITION_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#000000",  # black
    "#5B51D3",  # purple
    "#E6AB02",  # dark gold
    "#A6CEE3",  # light blue
    "#B2DF8A",  # light green
]


def _fate_colors(
    fate_names: List[str],
    color_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return a color dict for fate_names.

    If color_map is provided, use it for any fate names it contains and fall
    back to FATE_PALETTE for the rest.  This lets users pass their original
    scanpy/Seurat cluster colors directly.
    """
    out = {}
    palette_idx = 0
    for name in fate_names:
        if color_map and name in color_map:
            out[name] = color_map[name]
        else:
            out[name] = FATE_PALETTE[palette_idx % len(FATE_PALETTE)]
            palette_idx += 1
    return out


def _condition_colors(
    condition_names: List[str],
    color_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return a color dict for condition_names.

    Draws from CONDITION_PALETTE (distinct from FATE_PALETTE).
    If color_map is provided, use it for any condition names it contains
    and fall back to CONDITION_PALETTE for the rest.
    """
    out = {}
    palette_idx = 0
    for name in condition_names:
        if color_map and name in color_map:
            out[name] = color_map[name]
        else:
            out[name] = CONDITION_PALETTE[palette_idx % len(CONDITION_PALETTE)]
            palette_idx += 1
    return out




def _significance_stars(pval: float) -> str:
    """Return significance stars for a p-value."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "ns"

# ---------------------------------------------------------------------------
# 1. Star embedding — primary visualization
# ---------------------------------------------------------------------------

def plot_star_embedding(
    adata,
    result: CommitmentScoreResult,
    color_by: str = "fate",
    figsize: Tuple[float, float] = (8, 8),
    point_size: float = 8.0,
    alpha: float = 0.75,
    arm_color: str = "#CCCCCC",
    arm_linewidth: float = 1.5,
    arm_linestyle: str = "--",
    show_arm_labels: bool = True,
    show_velocity: bool = False,
    velocity_scale: float = 1.0,
    color_map: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Radial star embedding plot — the primary scCS visualization.

    Draws the X_sccs embedding with:
    - Radial arm axes (dashed lines from origin to each fate tip)
    - Fate labels at the arm tips
    - Cells colored by fate, pseudotime, entropy, or per-fate affinity
    - Optional velocity arrows

    Parameters
    ----------
    adata : AnnData
        Must have X_sccs in obsm.
    result : CommitmentScoreResult
    color_by : str
        What to color cells by:
        - ``"fate"``        — cluster/arm assignment (default)
        - ``"pseudotime"``  — reads ``sccs_pseudotime`` then ``velocity_pseudotime``
        - ``"entropy"``     — per-cell commitment entropy (``cs_entropy``)
        - ``"nn_entropy"``  — NN-smoothed entropy (``cs_nn_entropy``;
          requires ``score(k_nn=...)``)
        - a fate name       — per-cell affinity (``cs_{fate}``;
          requires ``score(cell_level=True)``)
        - any other str     — auto-detected numeric or categorical column in
          ``adata.obs``
    figsize : tuple
    point_size : float
    alpha : float
    arm_color : str
        Color of the radial arm guide lines.
    arm_linewidth : float
    arm_linestyle : str
    show_arm_labels : bool
        Draw fate name labels at arm tips.
    show_velocity : bool
        Overlay velocity arrows (requires velocity_sccs in obsm).
    velocity_scale : float
        Scale factor for velocity arrows.
    title : str, optional
    vmin, vmax : float, optional
        Color-scale limits for numeric ``color_by`` modes. Defaults to the
        finite data range, so structure is always visible regardless of the
        absolute entropy/affinity scale. Pass explicit values to pin limits
        for cross-figure comparison.
    cmap : str, optional
        Matplotlib colormap name. Defaults: ``"RdYlBu_r"`` for entropy,
        ``"viridis"`` for pseudotime/generic numeric, ``"Blues"`` for per-fate
        affinity.
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    sns.set_theme(style="ticks")

    if "X_sccs" not in adata.obsm:
        raise KeyError("X_sccs not found in adata.obsm. Run build_embedding() first.")

    coords = np.array(adata.obsm["X_sccs"])
    sccs_meta = adata.uns.get("sccs", {})
    arm_scale = sccs_meta.get("arm_scale", 10.0)
    arm_dirs = sccs_meta.get("arm_dirs", None)
    fate_names = result.fate_names
    colors = _fate_colors(fate_names, color_map)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- Draw radial arm axes ---
    if arm_dirs is not None:
        for j, (name, arm_dir) in enumerate(zip(fate_names, arm_dirs)):
            tip = arm_dir * arm_scale * 1.15
            ax.plot(
                [0, tip[0]], [0, tip[1]],
                color=arm_color,
                linewidth=arm_linewidth,
                linestyle=arm_linestyle,
                zorder=1,
            )
            if show_arm_labels:
                label_pos = arm_dir * arm_scale * 1.25
                ax.text(
                    label_pos[0], label_pos[1],
                    name,
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color=colors[name],
                    path_effects=[
                        pe.withStroke(linewidth=2, foreground="white")
                    ],
                )

    # Mark origin
    ax.scatter(0, 0, s=80, c="black", marker="+", zorder=5, linewidths=1.5)

    # --- Color cells ---
    _scatter_cells(
        ax, adata, coords, result, color_by,
        fate_names, colors, point_size, alpha,
        vmin=vmin, vmax=vmax, cmap=cmap,
    )

    # --- Velocity arrows ---
    if show_velocity and "velocity_sccs" in adata.obsm:
        V = np.array(adata.obsm["velocity_sccs"])
        # Subsample for readability
        n_arrows = min(300, adata.n_obs)
        idx = np.random.choice(adata.n_obs, n_arrows, replace=False)
        ax.quiver(
            coords[idx, 0], coords[idx, 1],
            V[idx, 0] * velocity_scale, V[idx, 1] * velocity_scale,
            alpha=0.5, color="black", scale=20, width=0.003,
            headwidth=4, headlength=5, zorder=4,
        )

    # --- Formatting ---
    ax.set_aspect("equal")
    ax.set_xlabel("scCS dim 1", fontsize=10)
    ax.set_ylabel("scCS dim 2", fontsize=10)
    ax.set_title(
        title or f"scCS Star Embedding  (bifurcation: cluster '{sccs_meta.get('root', '?')}')",
        fontsize=11,
    )
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_nn_entropy_elbow(
    scorer,
    k_nn_range: Union[List[int], range] = range(5, 51, 5),
    color_map: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (12, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Elbow plots for choosing the optimal number of nearest neighbors (k_nn).

    Sweeps over ``k_nn_range``, computing NN-smoothed cell entropy at each k,
    and produces two side-by-side subplots:

    - **Left**: mean NN entropy across all cells vs k_nn.
    - **Right**: mean NN entropy per fate arm vs k_nn (one line per fate).

    Use these plots to identify the elbow — the k_nn where entropy stabilizes,
    indicating that additional smoothing no longer changes the signal.

    Parameters
    ----------
    scorer : SingleScorer
        A fitted scorer with ``build_embedding()`` and ``fit()`` already called.
        No prior ``score()`` call is needed — cell scores are recomputed
        internally from the velocity vectors.
    k_nn_range : list or range
        k_nn values to sweep.  Default: 5, 10, 15, ..., 50.
    color_map : dict, optional
        Fate name -> hex color.  Falls back to the default FATE_PALETTE.
    figsize : tuple
    title : str, optional
        Overall figure title.  Defaults to "NN Entropy Elbow".
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    >>> scorer.build_embedding(differentiation_metric='pseudotime')
    >>> scorer.fit()
    >>> result = scorer.score(compute_cell_level=True)
    >>> fig = scorer.plot_nn_entropy_elbow()
    """
    from .scores import compute_nn_cell_entropy, compute_cell_scores

    if scorer._fate_map is None or not scorer._fitted:
        raise RuntimeError("scorer must be fitted before plotting elbow.")
    if scorer._vx is None:
        raise RuntimeError("Velocity vectors not loaded. Call fit() or load_velocity_vectors().")

    fate_map = scorer._fate_map
    fate_names = fate_map.fate_names
    k_fates = fate_map.k
    coords = np.array(scorer.adata_sub.obsm["X_sccs"])

    # Compute cell_scores once
    cell_scores = compute_cell_scores(
        scorer._vx, scorer._vy,
        fate_map.fate_centroids,
        fate_map.root_centroid,
    )

    # Fate arm membership for per-fate means
    cluster_labels = scorer.adata_sub.obs[scorer.obs_key].astype(str).values
    fate_masks = {
        name: cluster_labels == name
        for name in fate_names
    }

    k_nn_list = list(k_nn_range)
    mean_all = []
    mean_per_fate = {name: [] for name in fate_names}

    for k in k_nn_list:
        nn_ent = compute_nn_cell_entropy(cell_scores, coords, k)
        mean_all.append(nn_ent.mean())
        for name in fate_names:
            mask = fate_masks[name]
            mean_per_fate[name].append(nn_ent[mask].mean() if mask.any() else float("nan"))

    # Colors
    colors = _fate_colors(fate_names, color_map)

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title or "NN Entropy Elbow", fontsize=13, y=1.01)

    # --- Left: overall mean ---
    ax = axes[0]
    ax.plot(k_nn_list, mean_all, color="#333333", linewidth=2, marker="o",
            markersize=5, label="All cells")
    ax.set_xlabel("k (nearest neighbors)", fontsize=11)
    ax.set_ylabel("Mean NN-smoothed entropy", fontsize=11)
    ax.set_title("Overall", fontsize=11)
    ax.set_xticks(k_nn_list)
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)

    # --- Right: per-fate means ---
    ax = axes[1]
    for name in fate_names:
        ax.plot(k_nn_list, mean_per_fate[name],
                color=colors[name], linewidth=2, marker="o",
                markersize=5, label=name)
    ax.set_xlabel("k (nearest neighbors)", fontsize=11)
    ax.set_ylabel("Mean NN-smoothed entropy", fontsize=11)
    ax.set_title("Per fate", fontsize=11)
    ax.set_xticks(k_nn_list)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 8. Expression trends along commitment axis
# ---------------------------------------------------------------------------

def plot_expression_trends(
    adata,
    result: CommitmentScoreResult,
    genes: List[str],
    fate: Optional[str] = None,
    x_axis: str = "affinity",
    n_bins: int = 10,
    layer: Optional[str] = None,
    smooth: bool = True,
    smooth_frac: float = 0.4,
    color_map: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ncols: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot gene expression trends along a chosen commitment axis.

    Cells are binned along the x-axis and mean expression per bin is plotted
    with a LOWESS smooth.

    Parameters
    ----------
    adata : AnnData
        Must contain the same cells as ``result``.
    result : CommitmentScoreResult
    genes : list of str
        Gene names to plot.  Must be present in ``adata.var_names``.
    fate : str, optional
        Which fate to use as the reference.
        Defaults to the fate with the highest M_sector.
    x_axis : str
        What to use as the x-axis for binning:
        - ``'affinity'``       : per-cell fate affinity score for ``fate``
                                 (0 → 1, from compute_cell_scores).
        - ``'pseudotime'``     : velocity_pseudotime from adata.obs
                                 (or sccs_pseudotime if available via compute_local_pseudotime()).
        - ``'radial_distance'``: Euclidean distance from origin in X_sccs
                                 (arm position, 0 = progenitor, arm_scale = tip).
        Default: ``'affinity'``.
    n_bins : int
        Number of bins along the x-axis.
    layer : str, optional
        AnnData layer to use for expression.  Defaults to ``adata.X``.
    smooth : bool
        Whether to overlay a LOWESS smoothed curve.
    smooth_frac : float
        LOWESS smoothing fraction (0–1).
    color_map : dict, optional
        Fate name → hex color.  Used to color the smoothed line.
    figsize : tuple, optional
    ncols : int
        Number of columns in the subplot grid.
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import scipy.sparse as sp
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        if smooth:
            raise ImportError(
                "statsmodels is required for LOWESS smoothing in plot_expression_trends. "
                "Install it with:  pip install statsmodels  "
                "or pass smooth=False to disable smoothing."
            ) from None
        lowess = None

    if result.cell_scores is None:
        raise ValueError(
            "cell_scores not available. Run scorer.score(compute_cell_level=True)."
        )

    # Resolve fate
    if fate is None:
        fate = result.fate_names[int(np.argmax(result.M_sector))]
    if fate not in result.fate_names:
        raise ValueError(f"fate '{fate}' not in fate_names: {result.fate_names}")
    fate_idx = result.fate_names.index(fate)

    colors = _fate_colors(result.fate_names, color_map)
    fate_color = colors[fate]

    # Validate genes
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise ValueError(f"Genes not found in adata.var_names: {missing}")

    # Align adata to the subset used during scoring.
    # Three supported usage patterns:
    #   (1) ``result.cell_obs_names`` == ``adata.obs_names`` (one-to-one)
    #   (2) ``result.cell_obs_names`` is a SUBSET of ``adata.obs_names``
    #       (passing a larger adata, e.g. full anndata).
    #   (3) ``adata.obs_names`` is a SUBSET of ``result.cell_obs_names``
    #       (passing a condition-masked subset of the scored adata).
    # In case (3), cell_scores is sliced to keep only the rows whose
    # obs_names are present in ``adata`` so x_vals stays aligned.
    cell_score_idx = None  # if set, slice result-derived arrays through this
    if result.cell_obs_names is not None:
        result_names = np.asarray(result.cell_obs_names)
        adata_names = np.asarray(adata.obs_names)
        # Find rows of result that are present in adata
        adata_set = set(map(str, adata_names))
        keep_mask = np.array([str(n) in adata_set for n in result_names])
        n_keep = int(keep_mask.sum())
        if n_keep == 0:
            raise ValueError(
                "None of result.cell_obs_names match adata.obs_names. "
                "Pass the AnnData used during scoring (or a superset of it)."
            )
        if n_keep < len(result_names):
            # case (3): adata is smaller than the scored subset.
            warnings.warn(
                f"plot_expression_trends: only {n_keep}/{len(result_names)} "
                "cells from result are present in adata. Slicing result "
                "arrays to match the passed adata.",
                UserWarning,
                stacklevel=2,
            )
            cell_score_idx = np.where(keep_mask)[0]
            adata_sub = adata[result_names[keep_mask]]
        else:
            # case (1) or (2): all result cells found
            adata_sub = adata[result_names]
    elif result.cell_scores.shape[0] != adata.n_obs:
        raise ValueError(
            f"result.cell_scores has {result.cell_scores.shape[0]} rows but "
            f"adata has {adata.n_obs} cells, and result.cell_obs_names is not "
            f"set (old result object). Re-run scorer.score(compute_cell_level=True)."
        )
    else:
        adata_sub = adata

    # --- Resolve x-axis values ---
    x_axis = x_axis.lower()
    if x_axis == "affinity":
        cs = result.cell_scores
        if cell_score_idx is not None:
            cs = cs[cell_score_idx]
        x_vals = cs[:, fate_idx]
        x_label = f"Fate affinity — {fate}"
    elif x_axis == "pseudotime":
        # Prefer subset-local pseudotime if available
        pt_col = (
            "sccs_pseudotime"
            if "sccs_pseudotime" in adata_sub.obs
            else "velocity_pseudotime"
        )
        if pt_col not in adata_sub.obs:
            raise ValueError(
                f"'{pt_col}' not found in adata.obs. "
                "Run scorer.compute_local_pseudotime() or scvelo.tl.velocity_pseudotime()."
            )
        x_vals = np.array(adata_sub.obs[pt_col], dtype=float)
        x_label = "Pseudotime"
    elif x_axis in ("radial_distance", "radial"):
        if "X_sccs" not in adata_sub.obsm:
            raise ValueError(
                "X_sccs not found in adata.obsm. Run build_embedding() first."
            )
        coords = np.array(adata_sub.obsm["X_sccs"])
        x_vals = np.linalg.norm(coords, axis=1)
        x_label = "Radial distance (scCS)"
    else:
        raise ValueError(
            f"Unknown x_axis='{x_axis}'. "
            "Choose from: 'affinity', 'pseudotime', 'radial_distance'."
        )

    # Extract expression matrix (n_cells_sub × n_genes)
    gene_idx = [adata_sub.var_names.get_loc(g) for g in genes]
    X = adata_sub.layers[layer] if layer is not None else adata_sub.X
    if sp.issparse(X):
        expr = np.asarray(X[:, gene_idx].todense())
    else:
        expr = np.asarray(X[:, gene_idx])

    # Bin cells along x-axis
    valid = ~np.isnan(x_vals)
    x_min, x_max = x_vals[valid].min(), x_vals[valid].max()
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_assign = np.digitize(x_vals, bin_edges[1:-1])  # 0 … n_bins-1

    mean_expr = np.full((n_bins, len(genes)), np.nan)
    for b in range(n_bins):
        mask = (bin_assign == b) & valid
        if mask.sum() > 0:
            mean_expr[b] = expr[mask].mean(axis=0)

    # Layout
    n_genes = len(genes)
    ncols = min(ncols, n_genes)
    nrows = int(np.ceil(n_genes / ncols))
    if figsize is None:
        figsize = (ncols * 3.5, nrows * 3.0)

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for gi, gene in enumerate(genes):
        row, col = divmod(gi, ncols)
        ax = axes[row][col]

        y = mean_expr[:, gi]
        valid_bins = ~np.isnan(y)
        ax.scatter(
            bin_centers[valid_bins], y[valid_bins],
            color=fate_color, s=30, alpha=0.7, zorder=3,
        )

        if smooth and valid_bins.sum() >= 5:
            try:
                sm = lowess(
                    y[valid_bins], bin_centers[valid_bins],
                    frac=smooth_frac, return_sorted=True,
                )
                ax.plot(sm[:, 0], sm[:, 1], color=fate_color, linewidth=2.0)
            except Exception:
                ax.plot(
                    bin_centers[valid_bins], y[valid_bins],
                    color=fate_color, linewidth=1.5,
                )
        else:
            ax.plot(
                bin_centers[valid_bins], y[valid_bins],
                color=fate_color, linewidth=1.5,
            )

        ax.set_title(gene, fontsize=10)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel("Mean expression", fontsize=8)
        sns.despine(ax=ax)

    # Hide unused axes
    for gi in range(n_genes, nrows * ncols):
        row, col = divmod(gi, ncols)
        axes[row][col].set_visible(False)

    plt.suptitle(
        f"Expression trends — '{fate}' arm  (x: {x_axis})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _scatter_cells(
    ax, adata, coords, result, color_by,
    fate_names, colors, point_size, alpha,
    vmin=None, vmax=None, cmap=None,
):
    """Internal: scatter cells with the requested coloring scheme.

    Supports these ``color_by`` modes:

    - ``"fate"`` (default) — categorical, arm / progenitor / unassigned
    - ``"entropy"`` or ``"cs_entropy"`` — per-cell commitment entropy
      (``adata.obs["cs_entropy"]``)
    - ``"nn_entropy"`` or ``"cs_nn_entropy"`` — NN-smoothed entropy
      (``adata.obs["cs_nn_entropy"]``; requires ``score(k_nn=...)``)
    - ``"pseudotime"`` — pseudotime, reading ``sccs_pseudotime`` then
      ``velocity_pseudotime`` (falls back to gray with a warning if neither
      column is present)
    - a fate name in ``fate_names`` — per-cell affinity column
      ``cs_{fate}`` (requires ``score(cell_level=True)``)
    - any other ``adata.obs`` column — auto-detected numeric or categorical

    Numeric color scales auto-scale to the data range by default. Pass
    ``vmin`` / ``vmax`` to pin limits (useful for cross-figure comparison).
    Pass ``cmap`` to override the default colormap for that branch.
    """
    sccs_meta = adata.uns.get("sccs", {})
    bif_cluster = sccs_meta.get("root", None)

    def _auto_lim(values, vmin_in, vmax_in):
        """Compute (vmin, vmax) defaulting to data min/max when not provided."""
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (vmin_in if vmin_in is not None else 0.0,
                    vmax_in if vmax_in is not None else 1.0)
        lo = vmin_in if vmin_in is not None else float(np.nanmin(finite))
        hi = vmax_in if vmax_in is not None else float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    if color_by == "fate":
        # Color by arm assignment (categorical)
        arm_names = adata.obs.get("sccs_branch", None)
        if arm_names is not None:
            arm_names = arm_names.astype(str).values
            # Bifurcation cluster
            if bif_cluster is not None:
                bif_mask = arm_names == str(bif_cluster)
                if bif_mask.sum() > 0:
                    ax.scatter(
                        coords[bif_mask, 0], coords[bif_mask, 1],
                        c=PROGENITOR_COLOR, s=point_size, alpha=alpha,
                        label=f"Progenitor ({bif_cluster})", zorder=2, rasterized=True,
                    )
            # Each fate arm
            for name in fate_names:
                mask = arm_names == str(name)
                if mask.sum() > 0:
                    ax.scatter(
                        coords[mask, 0], coords[mask, 1],
                        c=colors[name], s=point_size, alpha=alpha,
                        label=name, zorder=2, rasterized=True,
                    )
            # Unassigned
            unassigned = arm_names == "unassigned"
            if unassigned.sum() > 0:
                ax.scatter(
                    coords[unassigned, 0], coords[unassigned, 1],
                    c="#DDDDDD", s=point_size * 0.6, alpha=alpha * 0.5,
                    label="other", zorder=1, rasterized=True,
                )
            ax.legend(
                markerscale=2.5, fontsize=8, frameon=False,
                loc="upper right", bbox_to_anchor=(1.0, 1.0),
            )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c="gray",
                       s=point_size, alpha=alpha, rasterized=True)

    elif color_by in ("entropy", "cs_entropy"):
        col = "cs_entropy"
        if col not in adata.obs:
            warnings.warn("cs_entropy not in adata.obs. Run score() first.", stacklevel=3)
            ax.scatter(coords[:, 0], coords[:, 1], c="gray",
                       s=point_size, alpha=alpha, rasterized=True)
            return
        vals = adata.obs[col].values.astype(float)
        lo, hi = _auto_lim(vals, vmin, vmax)
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=vals, cmap=(cmap or "RdYlBu_r"),
            s=point_size, alpha=alpha,
            vmin=lo, vmax=hi, zorder=2, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Commitment entropy", shrink=0.7, pad=0.02)

    elif color_by in ("nn_entropy", "cs_nn_entropy"):
        col = "cs_nn_entropy"
        if col not in adata.obs:
            warnings.warn(
                "cs_nn_entropy not in adata.obs. "
                "Call score(k_nn=...) with a positive k_nn to populate it.",
                stacklevel=3,
            )
            ax.scatter(coords[:, 0], coords[:, 1], c="gray",
                       s=point_size, alpha=alpha, rasterized=True)
            return
        vals = adata.obs[col].values.astype(float)
        lo, hi = _auto_lim(vals, vmin, vmax)
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=vals, cmap=(cmap or "RdYlBu_r"),
            s=point_size, alpha=alpha,
            vmin=lo, vmax=hi, zorder=2, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="NN-smoothed entropy", shrink=0.7, pad=0.02)

    elif color_by == "pseudotime":
        # Try sccs_pseudotime first, then velocity_pseudotime
        col = None
        for candidate in ("sccs_pseudotime", "velocity_pseudotime"):
            if candidate in adata.obs:
                col = candidate
                break
        if col is None:
            warnings.warn(
                "Neither 'sccs_pseudotime' nor 'velocity_pseudotime' found in adata.obs. "
                "Run refit_pseudotime() or compute velocity pseudotime first.",
                stacklevel=3,
            )
            ax.scatter(coords[:, 0], coords[:, 1], c="gray",
                       s=point_size, alpha=alpha, rasterized=True)
            return
        vals = adata.obs[col].values.astype(float)
        lo, hi = _auto_lim(vals, vmin, vmax)
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=vals, cmap=(cmap or "viridis"),
            s=point_size, alpha=alpha,
            vmin=lo, vmax=hi, zorder=2, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=col, shrink=0.7, pad=0.02)

    elif color_by in fate_names:
        # Per-fate affinity
        col = f"cs_{color_by}"
        if col not in adata.obs:
            warnings.warn(
                f"'{col}' not in adata.obs. Run score(cell_level=True) first.",
                stacklevel=3,
            )
            ax.scatter(coords[:, 0], coords[:, 1], c="gray",
                       s=point_size, alpha=alpha, rasterized=True)
            return
        vals = adata.obs[col].values.astype(float)
        lo, hi = _auto_lim(vals, vmin, vmax)
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=vals, cmap=(cmap or "Blues"),
            s=point_size, alpha=alpha,
            vmin=lo, vmax=hi, zorder=2, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label=f"Affinity: {color_by}", shrink=0.7, pad=0.02)

    else:
        # Generic: numeric or categorical column in adata.obs
        if color_by not in adata.obs:
            warnings.warn(
                f"'{color_by}' not found in adata.obs. Coloring by gray.",
                stacklevel=3,
            )
            ax.scatter(coords[:, 0], coords[:, 1], c="gray",
                       s=point_size, alpha=alpha, rasterized=True)
            return

        vals = adata.obs[color_by]
        try:
            vals_float = vals.astype(float).values
            lo, hi = _auto_lim(vals_float, vmin, vmax)
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=vals_float, cmap=(cmap or "viridis"),
                s=point_size, alpha=alpha,
                vmin=lo, vmax=hi, zorder=2, rasterized=True,
            )
            plt.colorbar(sc, ax=ax, label=color_by, shrink=0.7, pad=0.02)
        except (ValueError, TypeError):
            # Categorical
            categories = vals.astype("category").cat.categories
            cat_colors = {c: FATE_PALETTE[i % len(FATE_PALETTE)]
                          for i, c in enumerate(categories)}
            for cat in categories:
                mask = vals.astype(str) == str(cat)
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=cat_colors[cat], s=point_size, alpha=alpha,
                    label=str(cat), zorder=2, rasterized=True,
                )
            ax.legend(markerscale=2.5, fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# 2. Multi-panel star embedding
# ---------------------------------------------------------------------------

def plot_star_panels(
    adata,
    result: CommitmentScoreResult,
    panels: Optional[List[str]] = None,
    figsize_per_panel: Tuple[float, float] = (6, 6),
    point_size: float = 6.0,
    alpha: float = 0.75,
    color_map: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel star embedding: one panel per coloring scheme.

    Default panels: fate assignment, pseudotime, entropy, + one per fate.

    Parameters
    ----------
    adata : AnnData
    result : CommitmentScoreResult
    panels : list of str, optional
        List of color_by values.  Defaults to
        ['fate', 'pseudotime', 'entropy'] + fate_names.
    figsize_per_panel : tuple
    point_size : float
    alpha : float
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    if panels is None:
        panels = ["fate", "pseudotime", "entropy"] + list(result.fate_names)

    n = len(panels)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for idx, (panel, ax) in enumerate(zip(panels, axes.ravel())):
        plot_star_embedding(
            adata, result,
            color_by=panel,
            point_size=point_size,
            alpha=alpha,
            color_map=color_map,
            ax=ax,
            title=panel,
        )

    # Hide unused axes
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3. Rose / polar plot
# ---------------------------------------------------------------------------

def plot_rose(
    result: CommitmentScoreResult,
    title: str = "Cumulative Velocity Magnitude by Direction",
    figsize: Tuple[float, float] = (7, 7),
    show_sectors: bool = True,
    color_map: Optional[Dict[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Polar rose plot of cumulative velocity magnitudes per angular bin.

    Each bin shows the total velocity magnitude pointing in that direction.
    Fate sectors are shaded with distinct colors.

    Parameters
    ----------
    result : CommitmentScoreResult
    title : str
    figsize : tuple
    show_sectors : bool
    ax : matplotlib Axes (polar), optional
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    sns.set_theme(style="ticks")

    n_bins = len(result.M_bin)
    bin_width = 2 * np.pi / n_bins
    bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + bin_width / 2
    colors = _fate_colors(result.fate_names, color_map)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.get_figure()

    bin_colors = ["#cccccc"] * n_bins
    if show_sectors:
        for j, (name, sector_bins) in enumerate(zip(result.fate_names, result.sectors)):
            for b in sector_bins:
                bin_colors[b] = colors[name]

    ax.bar(
        bin_centers,
        result.M_bin,
        width=bin_width * 0.9,
        color=bin_colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    if result.fate_angles is not None:
        for j, (name, angle) in enumerate(zip(result.fate_names, result.fate_angles)):
            angle_rad = np.radians(angle)
            r_max = result.M_bin.max() * 1.15
            ax.annotate(
                name,
                xy=(angle_rad, r_max),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color=colors[name],
            )

    patches = [
        mpatches.Patch(
            color=colors[name],
            label=f"{name} (M={result.M_sector[j]:.1f})"
        )
        for j, name in enumerate(result.fate_names)
    ]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title(title, pad=20, fontsize=12)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig



# ---------------------------------------------------------------------------
# 3b. Per-condition rose grid
# ---------------------------------------------------------------------------

def plot_rose_grid(
    results: Dict[str, "CommitmentScoreResult"],
    color_map: Optional[Dict[str, str]] = None,
    figsize_per_panel: Tuple[float, float] = (5, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grid of polar rose plots — one per condition.

    All panels share the same radial scale (max of all M_bin.max() across
    conditions), making magnitudes directly comparable.  Fate sectors are
    shaded with FATE_PALETTE colors (consistent with single-condition
    plot_rose).

    Parameters
    ----------
    results : dict
        Mapping of condition_label -> CommitmentScoreResult
        (output of PairScorer.score_all_conditions()).
    color_map : dict, optional
        fate_name -> hex color.  Falls back to FATE_PALETTE.
    figsize_per_panel : tuple
        Size of each polar subplot.
    title : str, optional
        Overall figure title.
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    sns.set_theme(style="ticks")

    conditions = list(results.keys())
    n = len(conditions)
    if n == 0:
        raise ValueError("results dict is empty.")

    # Shared radial scale
    r_max = max(res.M_bin.max() for res in results.values())
    r_max = r_max * 1.15 if r_max > 0 else 1.0

    # Use fate_names from first result
    fate_names = list(results.values())[0].fate_names
    colors = _fate_colors(fate_names, color_map)

    fig = plt.figure(figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]))

    for idx, cond in enumerate(conditions):
        res = results[cond]
        ax = fig.add_subplot(1, n, idx + 1, projection="polar")

        n_bins = len(res.M_bin)
        bin_width = 2 * np.pi / n_bins
        bin_centers = np.linspace(0, 2 * np.pi, n_bins, endpoint=False) + bin_width / 2

        bin_colors = ["#cccccc"] * n_bins
        for j, (name, sector_bins) in enumerate(zip(res.fate_names, res.sectors)):
            for b in sector_bins:
                bin_colors[b] = colors[name]

        ax.bar(
            bin_centers,
            res.M_bin,
            width=bin_width * 0.9,
            color=bin_colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Fate arm labels
        if res.fate_angles is not None:
            for j, (name, angle) in enumerate(zip(res.fate_names, res.fate_angles)):
                angle_rad = np.radians(angle)
                ax.annotate(
                    name,
                    xy=(angle_rad, r_max),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=colors[name],
                )

        ax.set_ylim(0, r_max)
        ax.set_title(cond, pad=15, fontsize=11, fontweight="bold")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.grid(True, alpha=0.3)

    # Shared legend (first panel's fate names)
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=colors[name], label=name)
        for name in fate_names
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=min(len(fate_names), 4),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. Pairwise CS heatmap
# ---------------------------------------------------------------------------

def plot_pairwise_cs(
    result: CommitmentScoreResult,
    normalized: bool = True,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of pairwise commitment scores.

    Entry [i, j] = CS(fate_i relative to fate_j).
    Values > 1 indicate stronger commitment to fate_i than fate_j.
    Color scale is log2-transformed for readability.

    Parameters
    ----------
    result : CommitmentScoreResult
    normalized : bool
        Use nCS (True) or unCS (False).
    title : str, optional
    figsize : tuple, optional
    cmap : str
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    mat = result.pairwise_nCS if normalized else result.pairwise_unCS
    df = pd.DataFrame(mat, index=result.fate_names, columns=result.fate_names)

    k = result.k
    if figsize is None:
        figsize = (max(4, k * 1.2), max(3.5, k * 1.0))

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)

    log_mat = np.log2(np.clip(mat, 1e-6, None))
    log_df = pd.DataFrame(log_mat, index=result.fate_names, columns=result.fate_names)

    finite_vals = log_mat[np.isfinite(log_mat)]
    vmax = np.abs(finite_vals).max() if len(finite_vals) > 0 else 5.0

    sns.heatmap(
        log_df, ax=ax,
        cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        annot=df.round(2), fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "log2(CS)", "shrink": 0.8},
    )

    label = "Normalized CS (nCS)" if normalized else "Unnormalized CS (unCS)"
    ax.set_title(title or f"Pairwise Commitment Scores ({label})", fontsize=11)
    ax.set_xlabel("Reference fate (denominator)")
    ax.set_ylabel("Query fate (numerator)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 5. Commitment bar chart
# ---------------------------------------------------------------------------

def plot_commitment_bar(
    result: CommitmentScoreResult,
    ref_fate: Optional[str] = None,
    mode: str = "auto",
    color_map: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of unCS and nCS for all k populations.

    For a k-furcation, produces k subplots — one per reference fate.
    Each subplot shows unCS (solid) and nCS (hatched) for all other k-1
    fates relative to that reference.  This way every population is shown
    as both a query and a reference, and nothing is hidden.

    For k=2 a single subplot is produced (equivalent to the old behaviour).

    Parameters
    ----------
    result : CommitmentScoreResult
    ref_fate : str, optional
        If given, produce only a single subplot using this fate as reference.
        Useful when you want a focused comparison.
    mode : str
        Kept for backward compatibility; ignored.
    color_map : dict, optional
        Mapping of fate name → hex color.
    title : str, optional
        Overall figure title.
    figsize : tuple, optional
        Per-subplot size ``(w, h)``.  Total figure width scales with k.
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    colors = _fate_colors(result.fate_names, color_map)
    sns.set_theme(style="ticks")

    # Decide which reference fates to show
    if ref_fate is not None:
        if ref_fate not in result.fate_names:
            raise ValueError(
                f"ref_fate '{ref_fate}' not in fate_names: "
                f"{result.fate_names}"
            )
        ref_indices = [result.fate_names.index(ref_fate)]
    else:
        ref_indices = list(range(result.k))

    n_panels = len(ref_indices)
    panel_w, panel_h = (figsize if figsize is not None else (4.0, 4.5))
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(panel_w * n_panels, panel_h),
        squeeze=False,
    )

    for col, ref_idx in enumerate(ref_indices):
        ax = axes[0, col]
        ref_name = result.fate_names[ref_idx]

        query_names = [n for i, n in enumerate(result.fate_names) if i != ref_idx]
        query_idx   = [i for i in range(result.k) if i != ref_idx]

        unCS_vals = [result.pairwise_unCS[i, ref_idx] for i in query_idx]
        nCS_vals  = [result.pairwise_nCS[i, ref_idx]  for i in query_idx]

        x = np.arange(len(query_names))
        width = 0.35
        bar_colors = [colors[n] for n in query_names]

        bars_un = ax.bar(
            x - width / 2, unCS_vals, width,
            color=bar_colors, alpha=0.90,
            label="unCS",
            edgecolor="white", linewidth=0.5,
        )
        bars_n = ax.bar(
            x + width / 2, nCS_vals, width,
            color=bar_colors, alpha=0.55,
            hatch="///", edgecolor="white", linewidth=0.5,
            label="nCS",
        )

        ax.axhline(1.0, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.45, label="CS = 1")

        # Value labels — compute y_max first
        all_vals = [v for v in unCS_vals + nCS_vals if np.isfinite(v)]
        y_max = max(all_vals) if all_vals else 2.0
        pad = y_max * 0.03

        for bar in bars_un:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + pad,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
        for bar in bars_n:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + pad,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color="#555555")

        ax.set_xticks(x)
        ax.set_xticklabels(query_names, rotation=15, ha="right")
        ax.set_ylim(0, y_max * 1.20)
        ax.set_ylabel(f"CS  (÷ '{ref_name}')" if col == 0 else "")
        ax.set_title(f"vs  '{ref_name}'", fontsize=10)
        if col == 0:
            ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle(
        title or "Commitment scores — all populations",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 6. Per-cell commitment heatmap
# ---------------------------------------------------------------------------

def plot_commitment_heatmap(
    result: CommitmentScoreResult,
    cell_scores: Optional[np.ndarray] = None,
    max_cells: int = 500,
    title: str = "Per-Cell Fate Affinity",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of per-cell fate affinity scores (cells × fates).

    Parameters
    ----------
    result : CommitmentScoreResult
    cell_scores : np.ndarray, shape (n_cells, k), optional
        If None, uses result.cell_scores.
    max_cells : int
        Subsample to this many cells for readability.
    title : str
    figsize : tuple, optional
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    scores = cell_scores if cell_scores is not None else result.cell_scores
    if scores is None:
        raise ValueError(
            "cell_scores not available. Run scorer.score(compute_cell_level=True)."
        )

    n_cells = scores.shape[0]
    if n_cells > max_cells:
        idx = np.random.choice(n_cells, max_cells, replace=False)
        idx = idx[np.argsort(np.argmax(scores[idx], axis=1))]
        scores_plot = scores[idx]
    else:
        idx = np.argsort(np.argmax(scores, axis=1))
        scores_plot = scores[idx]

    if figsize is None:
        figsize = (max(4, result.k * 1.5), min(8, max(3, n_cells / 80)))

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)

    df = pd.DataFrame(scores_plot, columns=result.fate_names)
    sns.heatmap(
        df, ax=ax,
        cmap="Blues", vmin=0, vmax=1,
        xticklabels=True, yticklabels=False,
        cbar_kws={"label": "Fate affinity", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Fate")
    ax.set_ylabel(f"Cells (n={scores_plot.shape[0]}, sorted by dominant fate)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 7. Multi-subset comparison
# ---------------------------------------------------------------------------

def plot_subset_comparison(
    subset_results: dict,
    ref_fate: Optional[str] = None,
    normalized: bool = True,
    title: str = "Commitment Score by Subset",
    figsize: Tuple[float, float] = (8, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare commitment scores across multiple subsets.

    Subsets whose chosen reference pair yields ``inf`` (e.g. progenitor-only
    subsets with no fate-arm cells, so ``pairwise_nCS`` is undefined) are
    rendered as gray hatched placeholders at zero height with an ``"inf"``
    annotation, instead of silently producing empty bars.

    Parameters
    ----------
    subset_results : dict
        Mapping of subset_name -> CommitmentScoreResult
        (from ``SingleScorer.score_per_subset``).
    ref_fate : str, optional
        Reference fate for the CS column. If None, use the fate with smallest
        sector magnitude (most likely to be present in all subsets).
    normalized : bool
        If True use ``pairwise_nCS``, else ``pairwise_unCS``.
    title : str
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    rows = []
    for subset_name, result in subset_results.items():
        if ref_fate is None:
            ref_idx = int(np.argmin(result.M_sector))
        else:
            ref_idx = result.fate_names.index(ref_fate)

        for j, fate_name in enumerate(result.fate_names):
            if j == ref_idx:
                continue
            cs_val = (result.pairwise_nCS[j, ref_idx] if normalized
                      else result.pairwise_unCS[j, ref_idx])
            rows.append({
                "subset": subset_name,
                "fate": fate_name,
                "CS": float(cs_val),
            })

    df = pd.DataFrame(rows)
    fate_names = df["fate"].unique().tolist()
    colors = _fate_colors(fate_names)

    # Warn about subsets that have all-inf values (progenitor-only)
    bad_subsets = (
        df.groupby("subset")["CS"]
          .apply(lambda s: np.all(~np.isfinite(s)))
    )
    progenitor_only = bad_subsets[bad_subsets].index.tolist()
    if progenitor_only:
        warnings.warn(
            f"Subsets with no fate-arm cells (pairwise_nCS = inf for all pairs): "
            f"{progenitor_only}. Rendered as hatched placeholders at y=0.",
            stacklevel=2,
        )

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)

    subset_names = list(subset_results.keys())
    x = np.arange(len(subset_names))
    width = 0.8 / max(len(fate_names), 1)

    finite_max = 1.0  # for y-limit headroom

    for j, fate in enumerate(fate_names):
        sub_df = df[df["fate"] == fate].set_index("subset")
        vals = []
        is_inf = []
        for s in subset_names:
            if s in sub_df.index:
                v = float(sub_df.loc[s, "CS"])
                if np.isfinite(v):
                    vals.append(v)
                    is_inf.append(False)
                    finite_max = max(finite_max, v)
                else:
                    vals.append(0.0)
                    is_inf.append(True)
            else:
                vals.append(0.0)
                is_inf.append(False)

        offset = (j - len(fate_names) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            label=fate, color=colors[fate], alpha=0.85,
            edgecolor=colors[fate], linewidth=0.5,
        )

        # Post-pass: hatched placeholder for inf bars
        for bar, inf_flag in zip(bars, is_inf):
            if inf_flag:
                bar.set_facecolor("#DDDDDD")
                bar.set_edgecolor("#888888")
                bar.set_hatch("///")
                bar.set_alpha(0.6)
                bar.set_height(0.04 * finite_max)  # small visible stub
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015 * finite_max,
                    "inf",
                    ha="center", va="bottom",
                    fontsize=7, color="#555555", fontstyle="italic",
                )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(subset_names, rotation=15, ha="right")
    ax.set_ylabel("Commitment Score (CS)")
    ax.set_title(title)
    ax.legend(frameon=False)
    # Headroom for the "inf" labels
    ax.set_ylim(top=finite_max * 1.15)
    sns.despine(ax=ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 9. ΔCS heatmap with CI annotation
# ---------------------------------------------------------------------------

def plot_delta_cs_heatmap(
    delta_result: dict,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of ΔCS = nCS_A − nCS_B with CI annotation.

    Entry [i, j] = nCS_A(i÷j) − nCS_B(i÷j).  Positive values (red) mean
    condition A has stronger commitment of fate i relative to fate j.
    Cells are annotated with Δ ± CI_half.

    Parameters
    ----------
    delta_result : dict
        Output of PairScorer.compute_delta_CS().
    title : str, optional
    figsize : tuple, optional
    cmap : str
        Diverging colormap.  Default: 'RdBu_r'.
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    delta = delta_result["delta_nCS"]
    ci_low = delta_result["ci_low"]
    ci_high = delta_result["ci_high"]
    fate_names = delta_result["fate_names"]
    cond_a = delta_result["condition_a"]
    cond_b = delta_result["condition_b"]

    k = len(fate_names)
    if figsize is None:
        figsize = (max(4, k * 1.5), max(3.5, k * 1.3))

    # CI half-width for annotation
    ci_half = (ci_high - ci_low) / 2.0

    # Build annotation matrix
    annot = np.empty((k, k), dtype=object)
    for i in range(k):
        for j in range(k):
            d = delta[i, j]
            h = ci_half[i, j]
            if np.isfinite(d) and np.isfinite(h):
                annot[i, j] = f"{d:+.2f}" + "\n" + f"±{h:.2f}"
            elif np.isfinite(d):
                annot[i, j] = f"{d:+.2f}"
            else:
                annot[i, j] = "—"

    df = pd.DataFrame(delta, index=fate_names, columns=fate_names)

    finite_vals = delta[np.isfinite(delta)]
    vmax = np.abs(finite_vals).max() if len(finite_vals) > 0 else 1.0

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df, ax=ax,
        cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        annot=annot, fmt="",
        linewidths=0.5,
        cbar_kws={"label": "ΔnCS", "shrink": 0.8},
        annot_kws={"size": 9},
    )

    ax.set_title(
        title or f"ΔCS: '{cond_a}' − '{cond_b}'",
        fontsize=11,
    )
    ax.set_xlabel(f"Reference fate (÷)")
    ax.set_ylabel(f"Query fate (×)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 10. Grouped bar chart of nCS per condition
# ---------------------------------------------------------------------------

def plot_compare_conditions_bar(
    results: Dict[str, "CommitmentScoreResult"],
    ref_fate: Optional[str] = None,
    color_map: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart of nCS per condition.

    For each fate pair (query ÷ reference), one group of bars — one bar per
    condition, colored by CONDITION_PALETTE.  A horizontal dashed line at
    CS = 1 marks the neutral point.

    Parameters
    ----------
    results : dict
        Mapping of condition_label -> CommitmentScoreResult
        (output of PairScorer.score_all_conditions()).
    ref_fate : str, optional
        Reference fate for the denominator.  If None, uses the fate with
        the lowest mean M_sector across conditions.
    color_map : dict, optional
        condition_label -> hex color.  Falls back to CONDITION_PALETTE.
    title : str, optional
    figsize : tuple, optional
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    conditions = list(results.keys())
    fate_names = list(results.values())[0].fate_names
    k = len(fate_names)

    # Resolve reference fate
    if ref_fate is None:
        mean_m = np.array([
            np.mean([results[c].M_sector[j] for c in conditions])
            for j in range(k)
        ])
        ref_idx = int(np.argmin(mean_m))
        ref_fate = fate_names[ref_idx]
    else:
        if ref_fate not in fate_names:
            raise ValueError(f"ref_fate '{ref_fate}' not in fate_names: {fate_names}")
        ref_idx = fate_names.index(ref_fate)

    query_names = [n for i, n in enumerate(fate_names) if i != ref_idx]
    query_idx = [i for i in range(k) if i != ref_idx]

    # Condition colors
    cond_colors = _condition_colors(conditions, color_map)

    # Build data
    rows = []
    for cond in conditions:
        res = results[cond]
        for qi, qname in zip(query_idx, query_names):
            rows.append({
                "condition": cond,
                "fate_pair": f"{qname} ÷ {ref_fate}",
                "nCS": res.pairwise_nCS[qi, ref_idx],
            })
    df = pd.DataFrame(rows)

    fate_pairs = df["fate_pair"].unique().tolist()
    n_pairs = len(fate_pairs)
    n_conds = len(conditions)

    if figsize is None:
        figsize = (max(5, n_pairs * n_conds * 0.8), 4.5)

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_pairs)
    width = 0.8 / n_conds

    for ci, cond in enumerate(conditions):
        sub = df[df["condition"] == cond]
        vals = [sub[sub["fate_pair"] == fp]["nCS"].values[0]
                if len(sub[sub["fate_pair"] == fp]) > 0 else np.nan
                for fp in fate_pairs]
        offset = (ci - n_conds / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width * 0.9,
            color=cond_colors[cond], alpha=0.85,
            label=cond, edgecolor="white", linewidth=0.5,
        )
        # Value labels
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.45, label="CS = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(fate_pairs, rotation=15, ha="right")
    ax.set_ylabel("nCS")
    ax.set_title(title or f"nCS by condition  (÷ '{ref_fate}')", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 11. Radar / spider chart of commitment vectors
# ---------------------------------------------------------------------------

def plot_commitment_vector_radar(
    results: Dict[str, "CommitmentScoreResult"],
    color_map: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Radar / spider chart of commitment vectors per condition.

    Each condition is one closed polygon.  Axes = fate names (k spokes).
    Values = commitment_vector (sums to 1).  Conditions colored by
    CONDITION_PALETTE.

    For k < 3, falls back to a grouped bar chart with a warning.

    Parameters
    ----------
    results : dict
        Mapping of condition_label -> CommitmentScoreResult
        (output of PairScorer.score_all_conditions()).
    color_map : dict, optional
        condition_label -> hex color.  Falls back to CONDITION_PALETTE.
    title : str, optional
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.patches as mpatches

    conditions = list(results.keys())
    fate_names = list(results.values())[0].fate_names
    k = len(fate_names)

    cond_colors = _condition_colors(conditions, color_map)

    if k < 3:
        warnings.warn(
            f"k={k} fates — radar chart requires k≥3. Falling back to bar chart.",
            stacklevel=2,
        )
        # Simple bar chart fallback
        sns.set_theme(style="ticks")
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(k)
        width = 0.8 / len(conditions)
        for ci, cond in enumerate(conditions):
            cv = results[cond].commitment_vector
            offset = (ci - len(conditions) / 2 + 0.5) * width
            ax.bar(x + offset, cv, width * 0.9,
                   color=cond_colors[cond], alpha=0.85, label=cond)
        ax.set_xticks(x)
        ax.set_xticklabels(fate_names)
        ax.set_ylabel("Commitment weight")
        ax.set_title(title or "Commitment vectors by condition", fontsize=11)
        ax.legend(frameon=False)
        sns.despine(ax=ax)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    for cond in conditions:
        cv = list(results[cond].commitment_vector)
        cv += cv[:1]  # close
        color = cond_colors[cond]
        ax.plot(angles, cv, color=color, linewidth=2.0, label=cond)
        ax.fill(angles, cv, color=color, alpha=0.15)

    # Spoke labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(fate_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="gray")
    ax.grid(True, alpha=0.3)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1),
        fontsize=9,
        frameon=False,
    )
    ax.set_title(title or "Commitment vectors by condition", pad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 12. Omnibus summary heatmap (MultiScorer)
# ---------------------------------------------------------------------------

def plot_omnibus_summary(
    omnibus_df,
    results: Dict[str, CommitmentScoreResult],
    posthoc_df=None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Summary heatmap: fates × conditions showing omnibus significance.

    Left panel: heatmap of mean per-cell affinity per fate per condition,
    annotated with omnibus p-value stars.
    Right panel (if posthoc_df provided): significant pairwise comparisons
    as a connectivity grid.

    Parameters
    ----------
    omnibus_df : pd.DataFrame
        Output of MultiScorer.compare_omnibus().
        Columns: fate, test, statistic, pval, pval_adj, significant.
    results : dict
        Mapping of condition_label -> CommitmentScoreResult
        (output of MultiScorer.score_all_conditions()).
    posthoc_df : pd.DataFrame, optional
        Output of MultiScorer.compare_posthoc().
        If provided, right panel shows post-hoc significance grid.
    figsize : tuple, optional
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    conditions = list(results.keys())
    fate_names = list(results.values())[0].fate_names
    n_conds = len(conditions)
    n_fates = len(fate_names)

    # Build mean affinity matrix: fates × conditions
    mean_affinity = np.zeros((n_fates, n_conds))
    for ci, cond in enumerate(conditions):
        res = results[cond]
        if res.cell_scores is not None:
            for fi in range(n_fates):
                mean_affinity[fi, ci] = res.cell_scores[:, fi].mean()
        else:
            # Fall back to M_sector proportions
            total_m = res.M_sector.sum()
            if total_m > 0:
                mean_affinity[fi, ci] = res.M_sector[fi] / total_m

    # Build annotation with significance stars from omnibus_df
    annot = np.empty((n_fates, n_conds), dtype=object)
    omnibus_map = {}
    if omnibus_df is not None and len(omnibus_df) > 0:
        for _, row in omnibus_df.iterrows():
            omnibus_map[row["fate"]] = row

    for fi, fate in enumerate(fate_names):
        for ci, cond in enumerate(conditions):
            val = mean_affinity[fi, ci]
            if fate in omnibus_map:
                pval_adj = omnibus_map[fate]["pval_adj"]
                stars = _significance_stars(pval_adj)
                annot[fi, ci] = f"{val:.3f}\n{stars}"
            else:
                annot[fi, ci] = f"{val:.3f}"

    # Determine layout
    has_posthoc = posthoc_df is not None and len(posthoc_df) > 0
    n_panels = 2 if has_posthoc else 1

    if figsize is None:
        panel_w = max(4, n_conds * 1.5)
        panel_h = max(3, n_fates * 0.8)
        figsize = (panel_w * n_panels + 1, panel_h)

    sns.set_theme(style="ticks")
    if has_posthoc:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                         gridspec_kw={"width_ratios": [2, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        ax2 = None

    # --- Left panel: mean affinity heatmap ---
    aff_df = pd.DataFrame(mean_affinity, index=fate_names, columns=conditions)
    sns.heatmap(
        aff_df, ax=ax1,
        cmap="YlOrRd", annot=annot, fmt="",
        linewidths=0.5,
        cbar_kws={"label": "Mean affinity", "shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax1.set_title("Mean affinity + omnibus significance", fontsize=11)
    ax1.set_xlabel("Condition")
    ax1.set_ylabel("Fate")

    # --- Right panel: post-hoc significance grid ---
    if ax2 is not None and posthoc_df is not None:
        # Count significant pairwise comparisons per fate
        sig_counts = {}
        for fate in fate_names:
            fate_posthoc = posthoc_df[posthoc_df["fate"] == fate]
            sig_counts[fate] = fate_posthoc["significant"].sum() if "significant" in fate_posthoc.columns else 0

        # Build a simple bar chart of significant comparison counts per fate
        sig_vals = [sig_counts.get(f, 0) for f in fate_names]
        fate_colors = _fate_colors(fate_names)
        bar_colors = [fate_colors[f] for f in fate_names]

        ax2.barh(fate_names, sig_vals, color=bar_colors, alpha=0.85)
        ax2.set_xlabel("Significant pairwise comparisons")
        ax2.set_title("Post-hoc significance count", fontsize=11)
        ax2.invert_yaxis()
        sns.despine(ax=ax2)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 13. Post-hoc heatmap (MultiScorer)
# ---------------------------------------------------------------------------

def plot_posthoc_heatmap(
    posthoc_df,
    fate: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Condition × condition heatmap of post-hoc p-values for a given fate.

    Lower triangle: p-values (color intensity). Upper triangle: delta mean
    affinity. Annotated with significance stars.

    Parameters
    ----------
    posthoc_df : pd.DataFrame
        Output of MultiScorer.compare_posthoc().
        Columns: fate, comparison, method, statistic, pval, pval_adj,
                 significant, mean_A, mean_B, delta_mean.
    fate : str, optional
        Which fate to plot. If None, uses the first fate in posthoc_df.
    figsize : tuple, optional
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    if posthoc_df is None or len(posthoc_df) == 0:
        raise ValueError("posthoc_df is empty or None.")

    # Resolve fate
    if fate is None:
        fate = posthoc_df["fate"].iloc[0]
    fate_df = posthoc_df[posthoc_df["fate"] == fate]
    if len(fate_df) == 0:
        raise ValueError(f"No post-hoc results for fate '{fate}'.")

    # Extract unique conditions from comparison strings
    all_conds = set()
    for comp in fate_df["comparison"]:
        parts = comp.split(" vs ")
        if len(parts) == 2:
            all_conds.add(parts[0].strip())
            all_conds.add(parts[1].strip())
    conditions = sorted(all_conds)
    n = len(conditions)
    cond_idx = {c: i for i, c in enumerate(conditions)}

    if n < 2:
        raise ValueError(f"Need at least 2 conditions for post-hoc heatmap, got {n}.")

    # Build matrices
    pval_matrix = np.full((n, n), np.nan)
    delta_matrix = np.full((n, n), np.nan)
    sig_matrix = np.full((n, n), False)

    for _, row in fate_df.iterrows():
        comp = row["comparison"]
        parts = comp.split(" vs ")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        ai, bi = cond_idx[a], cond_idx[b]

        pval_matrix[ai, bi] = row["pval_adj"]
        pval_matrix[bi, ai] = row["pval_adj"]

        delta_matrix[ai, bi] = row.get("delta_mean", np.nan)
        delta_matrix[bi, ai] = -row.get("delta_mean", np.nan) if not np.isnan(row.get("delta_mean", np.nan)) else np.nan

        sig_matrix[ai, bi] = row.get("significant", False)
        sig_matrix[bi, ai] = row.get("significant", False)

    # Build annotation: lower = p-value + stars, upper = delta + stars
    annot = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot[i, j] = ""
            elif i > j:
                # Lower triangle: p-value
                p = pval_matrix[i, j]
                if not np.isnan(p):
                    stars = _significance_stars(p)
                    annot[i, j] = f"{p:.3f}\n{stars}"
                else:
                    annot[i, j] = ""
            else:
                # Upper triangle: delta mean
                d = delta_matrix[i, j]
                if not np.isnan(d):
                    stars = _significance_stars(pval_matrix[i, j]) if not np.isnan(pval_matrix[i, j]) else ""
                    annot[i, j] = f"{d:+.3f}\n{stars}"
                else:
                    annot[i, j] = ""

    if figsize is None:
        figsize = (max(5, n * 1.5), max(4, n * 1.3))

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize)

    # Use pval_matrix for color scale (lower triangle)
    # Mask upper triangle for p-value coloring
    pval_display = np.copy(pval_matrix)
    pval_display[np.triu_indices(n)] = np.nan  # mask upper

    # Plot p-value heatmap (lower triangle)
    log_pval = -np.log10(np.clip(pval_display, 1e-10, None))
    log_df = pd.DataFrame(log_pval, index=conditions, columns=conditions)

    sns.heatmap(
        log_df, ax=ax,
        cmap="Reds", annot=annot, fmt="",
        linewidths=0.5,
        cbar_kws={"label": "-log10(adj p-value)", "shrink": 0.8},
        annot_kws={"size": 9},
    )

    ax.set_title(f"Post-hoc comparisons — {fate}", fontsize=11)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Condition")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 14. Pairwise delta grid (MultiScorer)
# ---------------------------------------------------------------------------

def plot_pairwise_delta_grid(
    delta_results: Dict[Tuple[str, str], dict],
    figsize_per_panel: Tuple[float, float] = (4, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grid of ΔCS heatmaps for all condition pairs.

    Each panel shows the ΔnCS heatmap for one condition pair,
    using the same layout as plot_delta_cs_heatmap().

    Parameters
    ----------
    delta_results : dict
        Output of MultiScorer.compute_pairwise_deltas().
        Mapping of (cond_a, cond_b) -> delta_result dict.
    figsize_per_panel : tuple
        Size of each subplot.
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import pandas as pd

    pairs = list(delta_results.keys())
    n_pairs = len(pairs)
    if n_pairs == 0:
        raise ValueError("delta_results is empty.")

    ncols = min(n_pairs, 3)
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for idx, (pair, delta_result) in enumerate(delta_results.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        delta = delta_result["delta_nCS"]
        ci_low = delta_result["ci_low"]
        ci_high = delta_result["ci_high"]
        fate_names = delta_result["fate_names"]
        cond_a = delta_result["condition_a"]
        cond_b = delta_result["condition_b"]

        k = len(fate_names)
        ci_half = (ci_high - ci_low) / 2.0

        # Build annotation
        annot = np.empty((k, k), dtype=object)
        for i in range(k):
            for j in range(k):
                d = delta[i, j]
                h = ci_half[i, j]
                if np.isfinite(d) and np.isfinite(h):
                    annot[i, j] = f"{d:+.2f}\n±{h:.2f}"
                elif np.isfinite(d):
                    annot[i, j] = f"{d:+.2f}"
                else:
                    annot[i, j] = "—"

        df = pd.DataFrame(delta, index=fate_names, columns=fate_names)
        finite_vals = delta[np.isfinite(delta)]
        vmax = np.abs(finite_vals).max() if len(finite_vals) > 0 else 1.0

        sns.heatmap(
            df, ax=ax,
            cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            annot=annot, fmt="",
            linewidths=0.5,
            cbar=False,
            annot_kws={"size": 7},
        )
        ax.set_title(f"ΔCS: {cond_a} − {cond_b}", fontsize=9)
        ax.set_xlabel("Ref fate", fontsize=8)
        ax.set_ylabel("Query fate", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
