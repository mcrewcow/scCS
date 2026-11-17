"""Scientific missingness, state, and visualization contracts for the release."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import anndata as ad

from scCS import PairScorer, SingleScorer
from test_release_regressions import fitted


def missingness_pair():
    rows = []
    for condition in ("control", "treated"):
        for rep in range(5):
            for cell in range(10):
                rows.append(
                    dict(
                        cluster="root",
                        pt=cell / 20,
                        condition=condition,
                        replicate=f"R{rep}",
                        valid=condition == "control" or cell < 5,
                    )
                )
    for fate in ("A", "B"):
        for pt in (0.95, 1.0):
            rows.append(
                dict(
                    cluster=fate, pt=pt, condition="control", replicate="R0", valid=True
                )
            )
    obs = pd.DataFrame(rows, index=[f"cell{i}" for i in range(len(rows))])
    a = ad.AnnData(X=np.zeros((len(obs), 2)), obs=obs)
    matrix = sparse.lil_matrix((len(obs), len(obs)))
    for i, row in enumerate(rows):
        if row["cluster"] != "root" or not row["valid"]:
            matrix[i, i] = 1
        else:
            matrix[i, len(obs) - 3] = 0.9
            matrix[i, len(obs) - 1] = 0.1
    scorer = PairScorer(
        a,
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
        replicate_obs_key="replicate",
    )
    scorer.build_embedding(ordering_metric="pt", verbose=False)
    scorer.fit(
        matrix.tocsr(),
        scoring_mode="future_fate",
        future_fate_options={"min_anchor_cells": 1},
        verbose=False,
    )
    scorer.score_all_conditions(verbose=False)
    return scorer, matrix.tocsr()


def test_missing_cfa_does_not_create_preference_effect():
    s, _ = missingness_pair()
    table = s.replicate_table()
    np.testing.assert_allclose(table["directional_affinity::A"], 0.9)
    assert set(table["mean_affinity_defined_fraction"]) == {1.0, 0.5}
    stats = s.compare_conditions(
        metric="future_fate_affinity", fate="A", n_bootstrap=50, verbose=False
    ).iloc[0]
    assert abs(stats.effect_b_minus_a) < 1e-14
    assert stats.pvalue == 1.0
    assert abs(stats.ci_lower) < 1e-14 and abs(stats.ci_upper) < 1e-14


def test_unresolved_reach_and_coverage_are_distinct_outcomes():
    s, _ = missingness_pair()
    reach = s.compare_conditions(metric="future_fate_reach", verbose=False).iloc[0]
    unresolved = s.compare_conditions(
        metric="unresolved_future_probability", verbose=False
    ).iloc[0]
    defined = s.compare_conditions(
        metric="affinity_defined_fraction", verbose=False
    ).iloc[0]
    assert reach.effect_b_minus_a < 0
    assert unresolved.effect_b_minus_a > 0
    assert defined.effect_b_minus_a == -0.5
    competing = s.compare_conditions(metric="competing_reach", verbose=False).iloc[0]
    assert competing.effect_b_minus_a == 0


def test_no_fabricated_entropy_or_probability_for_unknown_futures():
    s, _ = missingness_pair()
    r = s.result
    unknown = ~r.affinity_defined
    assert unknown.any()
    assert np.isnan(r.future_fate_affinity[unknown]).all()
    assert np.isnan(r.future_fate_specificity[unknown]).all()
    assert np.isnan(r.future_fate_entropy[unknown]).all()
    np.testing.assert_equal(r.resolved_commitment[unknown], 0)
    np.testing.assert_allclose(r.future_fate_contribution, r.selected_probability)


def test_invalid_alias_rejected_for_instantaneous_inference_and_plots():
    s, matrix = missingness_pair()
    s.fit(matrix, scoring_mode="instantaneous", verbose=False)
    s.score_all_conditions(verbose=False)
    with pytest.raises(ValueError, match="future_fate"):
        s.compare_conditions(metric="future_fate_affinity", fate="A", verbose=False)
    with pytest.raises(ValueError, match="future_fate"):
        s._scorer.plot_star(color_by="future_fate_affinity:A")
    with pytest.raises(ValueError, match="future_fate"):
        s._scorer.plot_commitment_heatmap(metric="future_fate_affinity")


def test_missing_id_export_fails_before_writing():
    s, _ = fitted()
    target = s.adata.copy()
    target.obs_names = [f"different{i}" for i in range(target.n_obs)]
    with pytest.raises(ValueError, match="missing result cell IDs"):
        s.result.write_to_adata(target)
    assert "sccs_future_fate_reach" not in target.obs


def test_duplicate_ids_fail_preflight_and_embedding():
    s, _ = fitted()
    a = s.adata.copy()
    a.obs_names = ["duplicate"] * len(a)
    scorer = SingleScorer(a, root="root", branches=["A", "B"], obs_key="cluster")
    assert not scorer.preflight(ordering_metric="pt", check_velocity=False).ok
    with pytest.raises(ValueError, match="unique"):
        scorer.build_embedding(ordering_metric="pt", verbose=False)


def test_reordered_loaded_scorer_replays_graph_and_mode(tmp_path):
    s, _ = fitted()
    path = tmp_path / "scorer.pkl"
    s.save(str(path))
    restored = SingleScorer.load(str(path), s.adata[::-1].copy())
    expected = s.result.future_fate_affinity.copy()
    assert restored.scoring_mode == "future_fate"
    assert restored._transition_matrix is not None
    restored.score(verbose=False)
    restored.fit_future_fate(min_anchor_cells=1, verbose=False)
    index = pd.Index(restored.result.cell_ids).get_indexer(s.result.cell_ids)
    np.testing.assert_allclose(restored.result.future_fate_affinity[index], expected)


def test_scope_refit_can_return_to_unblocked_graph():
    s, matrix = missingness_pair()
    original = s.result.selected_probability.copy()
    s.fit(transition_scope="condition", scoring_mode="future_fate", verbose=False)
    s.fit(
        transition_scope="pooled",
        scoring_mode="future_fate",
        future_fate_options={"min_anchor_cells": 1},
        verbose=False,
    )
    np.testing.assert_allclose(s.result.selected_probability, original)
    assert (s._source_transition_matrix != matrix).nnz == 0


def test_unknown_futures_render_gray_and_ufp_is_available():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, _ = missingness_pair()
    for metric in [
        "future_fate_affinity:A",
        "unresolved_future_probability",
        "competing_reach",
    ]:
        fig = s._scorer.plot_star(color_by=metric)
        fig.canvas.draw()
        assert fig.axes[0].collections
        plt.close(fig)
    fig = s.plot_star_grid(color_by="signed_ordering_flux", population="root")
    axes = [a for a in fig.axes if a.get_title() in s.conditions]
    np.testing.assert_allclose(
        axes[0].collections[0].get_clim(), axes[1].collections[0].get_clim()
    )
    plt.close(fig)
    fig = s._scorer.plot_commitment_heatmap(metric="future_fate_affinity")
    image = next(ax.images[0] for ax in fig.axes if ax.images)
    np.testing.assert_allclose(image.get_clim(), (0.0, 1.0))
    np.testing.assert_allclose(image.cmap.get_bad()[:3], (0.72, 0.72, 0.72))
    plt.close(fig)


def test_future_metadata_records_parameters_and_anchor_ids(tmp_path):
    s, _ = fitted()
    s.score(verbose=False)
    path = tmp_path / "scores.h5ad"
    s.adata.write_h5ad(path)
    metadata = ad.read_h5ad(path).uns["sccs_v08"]["future_fate"]
    assert metadata["committed_reach_threshold"] == 0.25
    assert metadata["gamma"] == 64 / 65
    assert set(metadata["anchor_cell_ids"]) == {"A", "B"}
    assert metadata["tolerance"] == 1e-10
    assert metadata["progression_values"].shape == (6,)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, np.nan])
def test_anchor_count_is_a_positive_integer(count):
    s, matrix = fitted()
    with pytest.raises(ValueError, match="positive integer"):
        s.fit_future_fate(matrix, min_anchor_cells=count, verbose=False)
