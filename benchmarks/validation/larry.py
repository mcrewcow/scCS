"""Clone-held-out LARRY benchmark with a deliberately time-directed graph.

No RNA velocity is claimed: the public CoSpar object has no splicing layers.
Later test-clone cells are excluded before feature fitting and graph building.
Clonal outcome fractions are used only after prediction, and for the training
baseline. This evaluates state-to-fate association, not single-cell determinism.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from .data import fetch
from .experiments import fit_simulation
from .cellrank_comparison import absorption


def exclude_heldout_future(time, clones, test_clone_ids):
    """Keep early states, but exclude every later cell carrying any test barcode."""
    time = np.asarray(time)
    if time.ndim != 1 or len(time) != clones.shape[0] or not len(time):
        raise ValueError("Time labels must match the nonempty cell-by-clone matrix.")
    belongs_test = np.asarray(clones[:, test_clone_ids].sum(1)).ravel() > 0
    return ~((time > time.min()) & belongs_test)


def run(directory, seed=19, n_neighbors=30):
    a = ad.read_h5ad(fetch("larry", directory))
    time = a.obs.time_info.astype(str).astype(int).to_numpy()
    state = a.obs.state_info.astype(str).to_numpy()
    clones = sparse.csr_matrix(a.obsm["X_clone"]).astype(bool).astype(float)
    first, last = time.min(), time.max()
    early = (time == first) & (state == "undiff")
    late = time == last
    neu = late & (state == "Neutrophil")
    mono = late & (state == "Monocyte")
    early_counts = np.asarray(clones[early].sum(0)).ravel()
    neu_counts = np.asarray(clones[neu].sum(0)).ravel()
    mono_counts = np.asarray(clones[mono].sum(0)).ravel()
    late_counts = np.asarray(clones[late].sum(0)).ravel()
    eligible = np.flatnonzero((early_counts > 0) & (neu_counts + mono_counts >= 3))
    rng = np.random.default_rng(seed)
    eligible = rng.permutation(eligible)
    n_test = max(1, int(round(len(eligible) * 0.3)))
    test = np.sort(eligible[:n_test])
    train = np.sort(eligible[n_test:])
    assert set(test).isdisjoint(train) and len(train) > 10
    belongs_test = np.asarray(clones[:, test].sum(1)).ravel() > 0
    keep = exclude_heldout_future(time, clones, test)
    assert not np.any(keep & (time > first) & belongs_test)
    idx = np.flatnonzero(keep)
    # Construct a new object without barcode-derived annotations/predictions.
    obs = pd.DataFrame({"time": time[idx], "source_state": state[idx]}, index=a.obs_names[idx])
    obs["clusters"] = "Transit"
    obs.loc[(obs.time == first) & obs.source_state.eq("undiff"), "clusters"] = "Root"
    obs.loc[(obs.time == last) & obs.source_state.eq("Neutrophil"), "clusters"] = "Neutrophil"
    obs.loc[(obs.time == last) & obs.source_state.eq("Monocyte"), "clusters"] = "Monocyte"
    g = ad.AnnData(a.X[idx].copy(), obs=obs, var=a.var.copy())
    # CoSpar supplies library-normalized, non-log expression values.
    # Transform and select features only after excluding held-out future cells.
    sc.pp.log1p(g)
    sc.pp.highly_variable_genes(g, n_top_genes=2000, flavor="seurat", subset=True)
    sc.pp.pca(g, n_comps=30, zero_center=True, svd_solver="arpack", random_state=seed)
    coordinates = np.asarray(g.obsm["X_pca"])
    g.obsm["X_benchmark"] = coordinates
    rows, cols, values = [], [], []
    points = np.sort(g.obs.time.unique())
    gt = g.obs.time.to_numpy()
    for start, end in zip(points[:-1], points[1:]):
        source = np.flatnonzero(gt == start)
        target = np.flatnonzero(gt == end)
        knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(target)), n_jobs=1)
        distance, neighbors = knn.fit(coordinates[target]).kneighbors(coordinates[source])
        bandwidth = np.median(distance**2, axis=1, keepdims=True) + 1e-12
        weights = np.exp(-(distance**2) / bandwidth)
        weights /= weights.sum(1, keepdims=True)
        rows.extend(np.repeat(source, neighbors.shape[1]))
        cols.extend(target[neighbors].ravel())
        values.extend(weights.ravel())
    endpoints = np.flatnonzero(gt == last)
    rows.extend(endpoints)
    cols.extend(endpoints)
    values.extend(np.ones(len(endpoints)))
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(g.n_obs, g.n_obs))
    g.obsp["T_velocity"] = matrix  # A storage key, not a claim of RNA-velocity estimation.
    other = (gt == last) & ~g.obs.clusters.isin(["Neutrophil", "Monocyte"]).to_numpy()
    scorer, result = fit_simulation(
        g,
        root="Root",
        branches=("Neutrophil", "Monocyte"),
        ordering="time",
        competing={"Other": other},
    )
    masks = {name: result.outcome_anchor_mask[:, j] for j, name in enumerate(result.outcome_names)}
    cr = absorption(matrix, masks)
    j = result.fate_names.index("Neutrophil")
    predictions = pd.DataFrame(index=g.obs_names)
    predictions["scCS"] = np.nan
    predictions.loc[result.cell_ids, "scCS"] = result.conditional_fate_affinity[:, j]
    denominator = cr[["Neutrophil", "Monocyte"]].sum(1)
    predictions["CellRank"] = np.where(
        denominator > 1e-6, cr.Neutrophil.to_numpy() / denominator, np.nan
    )
    predictions["DFR"] = np.nan
    predictions.loc[result.cell_ids, "DFR"] = result.discounted_fate_reach
    train_baseline = np.mean(neu_counts[train] / (neu_counts[train] + mono_counts[train]))
    rows = []
    for clone in test:
        source_cells = np.flatnonzero(early & (np.asarray(clones[:, clone].toarray()).ravel() > 0))
        cell_names = a.obs_names[source_cells]
        p = predictions.loc[cell_names]
        defined = p.scCS.notna()
        rows.append(
            {
                "clone": int(clone),
                "n_early": len(p),
                "n_late_selected": int(neu_counts[clone] + mono_counts[clone]),
                "n_late_total": int(late_counts[clone]),
                "observed_neutrophil_fraction": neu_counts[clone]
                / (neu_counts[clone] + mono_counts[clone]),
                "observed_selected_fraction": (neu_counts[clone] + mono_counts[clone])
                / late_counts[clone],
                "defined_early_fraction": defined.mean(),
                "scCS": p.scCS.mean() if defined.any() else np.nan,
                "CellRank": p.CellRank.mean(),
                "training_clone_baseline": train_baseline,
                "DFR": p.DFR.mean(),
            }
        )
    table = pd.DataFrame(rows)
    metrics = []
    common = table[["scCS", "CellRank"]].notna().all(axis=1)
    for method in ["scCS", "CellRank", "training_clone_baseline"]:
        defined = table[method].notna()
        y = table.loc[common, "observed_neutrophil_fraction"].to_numpy()
        prediction = table.loc[common, method].to_numpy()
        metrics.append(
            {
                "method": method,
                "n_test_clones": len(table),
                "n_scored_clones": int(defined.sum()),
                "n_common_evaluated_clones": int(common.sum()),
                "clone_fraction_MSE": np.mean((prediction - y) ** 2),
                "clone_equal_weight_Brier": np.mean(prediction**2 - 2 * prediction * y + y),
                "Spearman": (
                    stats.spearmanr(prediction, y).statistic
                    if np.std(prediction) > 1e-12
                    else np.nan
                ),
            }
        )
    audit = {
        "source_cells": a.n_obs,
        "graph_cells": g.n_obs,
        "eligible_clones": len(eligible),
        "training_clones": len(train),
        "test_clones": len(test),
        "excluded_future_test_cells": int((~keep).sum()),
        "test_future_cells_remaining_in_graph": int(np.sum(keep & (time > first) & belongs_test)),
        "graph": "log1p-normalized expression, 2000 HVGs, PCA30, next-time-point neighbors",
        "not_RNA_velocity": True,
        "horizon": 64,
        "test_clone_ids": test.tolist(),
        "train_clone_ids": train.tolist(),
        "max_scCS_CellRank_difference": float(np.nanmax(abs(table.scCS - table.CellRank))),
    }
    return table, pd.DataFrame(metrics), audit
