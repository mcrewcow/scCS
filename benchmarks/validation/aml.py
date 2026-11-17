"""Single-patient AML treatment pilot: descriptive state associations only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

import scCS
from scCS.transitions import get_scvelo_transition_matrix
from .data import load_aml


def prepare(directory):
    a = load_aml(directory)
    composition = pd.crosstab(a.obs.cell_type, a.obs["sample"])
    labels = {
        "Blasts and Immature Eryth": "Root",
        "Monocytes": "Monocytic",
        "Early Erythroids": "Erythroid",
        "Mature Erythroids": "Erythroid",
        "Macrophages": "Outside",
    }
    mask = a.obs.cell_type.isin(labels) & a.obs["sample"].isin(["day0", "day2"])
    a = a[mask].copy()
    a.obs["clusters"] = a.obs.cell_type.map(labels).astype("category")
    a.obs["condition"] = a.obs["sample"].astype("category")
    scv.pp.filter_genes(a, min_shared_counts=10)
    scv.pp.normalize_per_cell(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=30, random_state=18)
    sc.pp.neighbors(a, n_neighbors=30, n_pcs=30, random_state=18)
    scv.pp.moments(a, n_neighbors=None, n_pcs=None)
    scv.tl.velocity(a, mode="deterministic")
    scv.tl.velocity_graph(a, n_jobs=1, show_progress_bar=False)
    matrix = get_scvelo_transition_matrix(a)
    # A predefined marker proxy, not inferred physical time or clonal truth.
    immature = ["CD34", "KIT", "GATA2"]
    mature = ["LYZ", "S100A8", "CTSS", "GATA1", "KLF1", "HBB"]

    def score(names):
        names = [g for g in names if g in a.var_names]
        if not names:
            raise ValueError("Required AML marker genes are absent.")
        x = a[:, names].X.toarray()
        z = (x - x.mean(0)) / (x.std(0) + 1e-8)
        return z.mean(1)

    a.obs["marker_maturation"] = score(mature) - score(immature)
    a.obsp["T_velocity"] = matrix
    return a, composition


def compare(a):
    pair = scCS.PairScorer(
        a,
        root="Root",
        branches=["Monocytic", "Erythroid"],
        obs_key="clusters",
        condition_obs_key="condition",
        replicate_obs_key="patient",
    )
    pair.build_embedding(
        ordering_metric="marker_maturation",
        cache_subset=False,
        write_to_adata=False,
        verbose=False,
    )
    rows = []
    results = {}
    for scope in ["pooled", "condition"]:
        pair.fit(
            a.obsp["T_velocity"],
            transition_scope=scope,
            scoring_mode="future_fate",
            future_fate_options={"min_anchor_cells": 3},
            verbose=False,
        )
        summaries = pair.score_all_conditions(
            min_cells=5, min_replicates=1, write_to_adata=False, verbose=False
        )
        pooled = pair.result
        results[scope] = pooled
        for condition, summary in summaries.items():
            locations = pd.Index(pooled.cell_ids).get_indexer(summary.cell_ids)
            assert (locations >= 0).all()
            defined = pooled.affinity_defined[locations]
            row = {
                "scope": scope,
                "condition": condition,
                "n_root_cells": len(locations),
                "biological_replicates": 1,
                "defined_fraction": defined.mean(),
                "DFR": pooled.discounted_fate_reach[locations].mean(),
                "UFP": pooled.unresolved_probability[locations].mean(),
            }
            for j, fate in enumerate(pooled.fate_names):
                row["CFA_" + fate] = (
                    pooled.conditional_fate_affinity[locations[defined], j].mean()
                    if defined.any()
                    else np.nan
                )
            rows.append(row)
    try:
        pair.compare_conditions(metric="discounted_fate_reach", verbose=False)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("Single-patient formal inference must not be accepted.")
    return pd.DataFrame(rows), results, error
