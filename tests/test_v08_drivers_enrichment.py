"""Tests for v0.8 commitment-associated genes and reproducible enrichment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

anndata = pytest.importorskip("anndata")

from scCS import SingleScorer
from scCS.drivers import get_commitment_associated_genes, get_fate_markers
from scCS.enrichment import load_gmt, run_commitment_enrichment


def make_gene_scorer(*, sparse_x: bool = False):
    rng = np.random.default_rng(17)
    n_replicates = 8
    root_per_rep = 6
    n_root = n_replicates * root_per_rep
    n_terminal = 8
    labels = np.array(["root"] * n_root + ["A"] * n_terminal + ["B"] * n_terminal)
    replicate_root = np.repeat([f"r{i}" for i in range(n_replicates)], root_per_rep)
    replicate = np.r_[replicate_root, ["terminal"] * (2 * n_terminal)]
    condition_root = np.repeat(["control"] * 4 + ["treated"] * 4, root_per_rep)
    condition = np.r_[condition_root, ["terminal"] * (2 * n_terminal)]
    pt_root = np.tile(np.linspace(0.05, 0.95, root_per_rep), n_replicates)
    pt = np.r_[pt_root, np.linspace(0.7, 1.0, n_terminal), np.linspace(0.7, 1.0, n_terminal)]

    replicate_effect = np.repeat(np.linspace(0.05, 0.95, n_replicates), root_per_rep)
    t = np.clip(0.75 * replicate_effect + 0.10 * pt_root + rng.normal(0, 0.015, n_root), 0, 1)
    gene_b = t + rng.normal(0, 0.015, n_root)
    gene_pt = pt_root + rng.normal(0, 0.015, n_root)
    gene_noise = rng.normal(0, 1, n_root)
    gene_constant = np.ones(n_root)
    root_matrix = np.column_stack([gene_b, gene_pt, gene_noise, gene_constant, np.zeros(n_root)])

    a_terminal = np.column_stack(
        [
            np.full(n_terminal, 0.1),
            np.full(n_terminal, 0.8),
            rng.normal(0, 1, n_terminal),
            np.ones(n_terminal),
            np.full(n_terminal, 5.0),
        ]
    )
    b_terminal = np.column_stack(
        [
            np.full(n_terminal, 0.9),
            np.full(n_terminal, 0.8),
            rng.normal(0, 1, n_terminal),
            np.ones(n_terminal),
            np.zeros(n_terminal),
        ]
    )
    X = np.vstack([root_matrix, a_terminal, b_terminal])
    if sparse_x:
        X = sparse.csr_matrix(X)
    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical(labels),
            "pt": pt,
            "replicate": replicate,
            "condition": pd.Categorical(condition),
        },
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = anndata.AnnData(X=X, obs=obs)
    adata.var_names = ["GeneB", "GenePT", "Noise", "Constant", "MarkerA"]
    adata.layers["velocity"] = X.copy()

    scorer = SingleScorer(adata, root="root", branches=["A", "B"], obs_key="cluster")
    scorer.build_embedding(ordering_metric="pt", verbose=False)
    geometry = scorer._embedding_result.geometry
    d_a, d_b = geometry.terminal_directions
    velocity = np.zeros((len(labels), geometry.dimension), dtype=float)
    raw = (1 - t[:, None]) * d_a[None, :] + t[:, None] * d_b[None, :]
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    velocity[:n_root] = raw * 0.7
    scorer.load_velocity_vectors(velocity)
    scorer.fit(verbose=False)
    result = scorer.score(write_to_adata=False, verbose=False)
    return scorer, result


def test_cell_exploratory_association_recovers_known_gene():
    scorer, result = make_gene_scorer()
    association = get_commitment_associated_genes(
        scorer.adata,
        result,
        outcome="directional_affinity",
        fate_names=["B"],
        pseudotime_key="pt",
        min_feature_cells=3,
        min_cells=20,
        verbose=False,
    )
    table = association.tables["B"]
    assert table.iloc[0]["gene"] == "GeneB"
    assert table.iloc[0]["effect"] > 0.7
    assert association.metadata["inference_unit"] == "cell_exploratory"
    assert association.metadata["pseudotime_key"] == "pt"


def test_replicate_association_uses_independent_units():
    scorer, result = make_gene_scorer()
    association = scorer.get_commitment_associated_genes(
        outcome="directional_affinity",
        fate_names=["B"],
        inference_unit="replicate",
        replicate_key="replicate",
        condition_key=None,
        pseudotime_key=None,
        min_feature_cells=3,
        min_replicates=6,
        verbose=False,
    )
    table = association.tables["B"]
    assert table.iloc[0]["gene"] == "GeneB"
    assert table.iloc[0]["effect"] > 0.9
    assert association.metadata["n_units"] == 8


def test_sparse_matrix_association_matches_dense_direction():
    scorer, result = make_gene_scorer(sparse_x=True)
    association = get_commitment_associated_genes(
        scorer.adata,
        result,
        outcome="directional_affinity",
        fate_names=["B"],
        pseudotime_key="pt",
        min_feature_cells=3,
        verbose=False,
    )
    assert association.tables["B"].iloc[0]["gene"] == "GeneB"


def test_velocity_layer_is_recorded():
    scorer, result = make_gene_scorer()
    association = get_commitment_associated_genes(
        scorer.adata,
        result,
        outcome="commitment_contribution",
        fate_names=["B"],
        layer="velocity",
        pseudotime_key="pt",
        min_feature_cells=3,
        verbose=False,
    )
    assert association.metadata["matrix_source"] == "layer:velocity"


def test_fate_markers_are_labeled_as_annotation_markers():
    scorer, result = make_gene_scorer()
    markers = get_fate_markers(
        scorer.adata,
        result,
        n_genes=5,
        min_cells=5,
        verbose=False,
    )
    assert "A" in markers
    assert "analysis_type" in markers["A"]
    assert set(markers["A"]["analysis_type"]) == {"terminal_annotation_marker_vs_root"}


def test_local_enrichment_uses_tested_gene_background():
    scorer, result = make_gene_scorer()
    association = get_commitment_associated_genes(
        scorer.adata,
        result,
        outcome="directional_affinity",
        fate_names=["B"],
        pseudotime_key="pt",
        min_feature_cells=3,
        fdr_threshold=1.0,
        verbose=False,
    )
    gene_sets = {
        "Target pathway": ["GeneB", "GenePT"],
        "Noise pathway": ["Noise", "Constant"],
    }
    enriched = run_commitment_enrichment(
        association,
        gene_sets=gene_sets,
        direction="positive",
        significant_only=False,
        max_genes=1,
        min_query_genes=1,
        min_set_size=1,
        verbose=False,
    )
    table = enriched.tables["B"]
    assert table.iloc[0]["term"] == "Target pathway"
    assert enriched.metadata["background_source"] == "union_of_tested_genes"
    assert enriched.metadata["gene_set_source"]["source_type"] == "local_mapping"


def test_load_gmt(tmp_path):
    path = tmp_path / "sets.gmt"
    path.write_text("SetA\tdescription\tG1\tG2\nSetB\tdescription\tG3\n")
    sets = load_gmt(path)
    assert sets == {"SetA": {"G1", "G2"}, "SetB": {"G3"}}


def test_replicate_association_filters_low_cell_units():
    scorer, result = make_gene_scorer()
    # Make one root replicate contain only one selected root cell by assigning
    # the remaining cells to a distinct low-cell unit.
    root_ids = scorer.adata.obs.index[scorer.adata.obs["cluster"].astype(str).eq("root")]
    first_rep = root_ids[:6]
    scorer.adata.obs.loc[first_rep[1:], "replicate"] = "low_cell_unit"
    association = scorer.get_commitment_associated_genes(
        outcome="directional_affinity",
        fate_names=["B"],
        inference_unit="replicate",
        replicate_key="replicate",
        pseudotime_key=None,
        min_feature_cells=3,
        min_replicates=6,
        min_cells_per_replicate=5,
        verbose=False,
    )
    assert association.metadata["min_cells_per_replicate"] == 5
    assert association.metadata["n_excluded_replicate_units"] >= 1
    assert any(row["n_cells"] < 5 for row in association.metadata["excluded_replicate_units"])
    assert all(row["n_cells"] >= 5 for row in association.metadata["replicate_units"])


def test_future_fate_outcome_aliases_are_accepted_by_gene_association():
    scorer, result = make_gene_scorer()
    future_named = get_commitment_associated_genes(
        scorer.adata,
        result,
        outcome="future_fate_affinity",
        fate_names=["B"],
        pseudotime_key="pt",
        min_feature_cells=3,
        min_cells=20,
        verbose=False,
    )
    canonical = get_commitment_associated_genes(
        scorer.adata,
        result,
        outcome="directional_affinity",
        fate_names=["B"],
        pseudotime_key="pt",
        min_feature_cells=3,
        min_cells=20,
        verbose=False,
    )
    pd.testing.assert_frame_equal(future_named.tables["B"], canonical.tables["B"])
