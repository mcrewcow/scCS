from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from scCS import MultiScorer, PairScorer


def make_condition_adata(conditions, n_replicates=5, root_cells_per_replicate=4):
    rows = []
    for condition in conditions:
        for replicate in range(n_replicates):
            for cell in range(root_cells_per_replicate):
                rows.append(
                    {
                        "cluster": "root",
                        "condition": condition,
                        "replicate": f"R{replicate}",
                        "pt": 0.1 + 0.6 * (cell / max(root_cells_per_replicate - 1, 1)),
                    }
                )
            # Terminal anchors are annotation-defined endpoints and need not be numerous.
            rows.append(
                {
                    "cluster": "A",
                    "condition": condition,
                    "replicate": f"R{replicate}",
                    "pt": 0.9,
                }
            )
            rows.append(
                {
                    "cluster": "B",
                    "condition": condition,
                    "replicate": f"R{replicate}",
                    "pt": 0.9,
                }
            )
    obs = pd.DataFrame(rows, index=[f"cell_{i}" for i in range(len(rows))])
    return anndata.AnnData(X=np.zeros((len(obs), 2)), obs=obs)


def load_condition_velocities(scorer, condition_lambda):
    result = scorer._scorer._embedding_result
    assert result is not None
    directions = result.geometry.terminal_directions
    metadata = scorer.adata.obs.loc[result.selected_cell_ids]
    velocity = np.zeros_like(result.coordinates)
    offsets = np.linspace(-0.04, 0.04, 5)
    for index, (_, row) in enumerate(metadata.iterrows()):
        if row["cluster"] != "root":
            continue
        lam = float(condition_lambda[str(row["condition"])])
        replicate = int(str(row["replicate"])[1:])
        lam = float(np.clip(lam + offsets[replicate], 0.0, 1.0))
        vector = (1.0 - lam) * directions[0] + lam * directions[1]
        norm = np.linalg.norm(vector)
        velocity[index] = vector / norm if norm > np.finfo(float).eps else 0.0
    scorer.load_velocity_vectors(velocity)
    scorer.fit(verbose=False)


def make_pair_scorer(with_replicates=True):
    adata = make_condition_adata(["control", "treated"])
    scorer = PairScorer(
        adata,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate" if with_replicates else None,
        obs_key="cluster",
    )
    scorer.build_embedding(ordering_metric="pt", verbose=False)
    load_condition_velocities(scorer, {"control": 0.15, "treated": 0.85})
    return scorer


def test_pair_scorer_uses_condition_specific_cells_and_replicates():
    scorer = make_pair_scorer()
    results = scorer.score_all_conditions(verbose=False)
    assert set(results) == {"control", "treated"}
    assert results["control"].n_cells == 20
    assert results["treated"].n_cells == 20
    assert results["control"].n_replicates == 5
    assert results["treated"].n_replicates == 5
    assert (
        results["treated"].population_summary.mean_commitment_contribution[1]
        > (results["control"].population_summary.mean_commitment_contribution[1])
    )


def test_pair_replicate_permutation_recovers_known_effect():
    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    comparison = scorer.compare_conditions(
        metric="mean_commitment",
        fate="B",
        n_permutations=499,
        n_bootstrap=100,
        random_state=7,
        verbose=False,
    )
    row = comparison.iloc[0]
    assert row["effect_b_minus_a"] > 0
    assert row["pvalue"] < 0.05
    assert row["ci_lower"] > 0
    assert row["n_replicates_a"] == 5
    assert row["n_replicates_b"] == 5


def test_compute_delta_cs_is_new_mean_commitment_effect():
    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    delta = scorer.compute_delta_CS(
        n_permutations=99,
        n_bootstrap=20,
        random_state=1,
        verbose=False,
    )
    assert set(delta["fate"]) == {"A", "B"}
    np.testing.assert_allclose(delta["delta_CS"], delta["effect_b_minus_a"])
    assert delta.loc[delta["fate"] == "B", "delta_CS"].iloc[0] > 0


def test_formal_pair_inference_requires_replicate_key():
    scorer = make_pair_scorer(with_replicates=False)
    scorer.score_all_conditions(verbose=False)
    with pytest.raises(ValueError, match="replicate_obs_key"):
        scorer.compare_conditions(metric="mean_commitment", fate="B", verbose=False)


def test_hierarchical_bootstrap_is_deterministic_under_fixed_seed():
    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    first = scorer.hierarchical_bootstrap(
        condition_a="control",
        condition_b="treated",
        metric="mean_commitment",
        fate="B",
        n_bootstrap=50,
        random_state=3,
    )
    second = scorer.hierarchical_bootstrap(
        condition_a="control",
        condition_b="treated",
        metric="mean_commitment",
        fate="B",
        n_bootstrap=50,
        random_state=3,
    )
    pd.testing.assert_frame_equal(first, second)


def make_multi_scorer():
    adata = make_condition_adata(["low", "middle", "high"])
    scorer = MultiScorer(
        adata,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
        condition_order=["low", "middle", "high"],
    )
    scorer.build_embedding(ordering_metric="pt", verbose=False)
    load_condition_velocities(
        scorer,
        {"low": 0.15, "middle": 0.50, "high": 0.85},
    )
    return scorer


def test_multi_omnibus_and_posthoc_recover_graded_effect():
    scorer = make_multi_scorer()
    scorer.score_all_conditions(verbose=False)
    omnibus = scorer.compare_omnibus(
        metric="mean_commitment",
        fate="B",
        n_permutations=499,
        random_state=4,
        verbose=False,
    )
    assert omnibus.iloc[0]["pvalue"] < 0.05
    posthoc = scorer.compare_posthoc(
        metric="mean_commitment",
        fate="B",
        n_permutations=199,
        random_state=5,
        verbose=False,
    )
    high_low = posthoc.loc[
        (posthoc["condition_a"] == "low") & (posthoc["condition_b"] == "high")
    ].iloc[0]
    assert high_low["effect_b_minus_a"] > 0
    assert high_low["pvalue"] < 0.05


def test_multi_planned_linear_contrast():
    scorer = make_multi_scorer()
    scorer.score_all_conditions(verbose=False)
    contrast = scorer.compare_contrast(
        {"low": -1.0, "middle": 0.0, "high": 1.0},
        metric="mean_commitment",
        fate="B",
        n_permutations=499,
        random_state=6,
    )
    assert contrast.iloc[0]["estimate"] > 0
    assert contrast.iloc[0]["pvalue"] < 0.05


def test_multi_posthoc_can_filter_by_significant_omnibus():
    scorer = make_multi_scorer()
    scorer.score_all_conditions(verbose=False)
    omnibus = scorer.compare_omnibus(
        metric="mean_commitment", fate="B", n_permutations=199, random_state=8, verbose=False
    )
    posthoc = scorer.compare_posthoc(
        metric="mean_commitment",
        fate="B",
        omnibus_results=omnibus,
        only_significant_omnibus=True,
        n_permutations=99,
        random_state=9,
        verbose=False,
    )
    assert len(posthoc) == 3


def test_metric_alias_is_canonicalized_in_outputs():
    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    comparison = scorer.compare_conditions(
        metric="mean_commitment",
        fate="B",
        n_permutations=99,
        verbose=False,
    )
    assert comparison.iloc[0]["metric"] == "mean_commitment_contribution"
    table = scorer.replicate_table()
    assert "mean_commitment_contribution::B" in table.columns
    assert "mean_commitment::B" not in table.columns



def test_future_fate_alias_adds_public_metric_metadata():
    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    comparison = scorer.compare_conditions(
        metric="future_fate_affinity",
        fate="B",
        n_permutations=99,
        verbose=False,
    )
    row = comparison.iloc[0]
    assert row["metric"] == "directional_affinity"
    assert row["metric_public"] == "future_fate_affinity"
    assert row["metric_label"] == "Conditional Fate Affinity (CFA) toward B"

def test_score_all_conditions_rejects_partial_designs():
    adata = make_condition_adata(
        ["control", "treated"],
        n_replicates=2,
        root_cells_per_replicate=4,
    )
    # Remove almost all treated root cells while retaining terminal anchors.
    keep = ~(adata.obs["condition"].eq("treated") & adata.obs["cluster"].eq("root"))
    treated_root = adata.obs.index[
        adata.obs["condition"].eq("treated") & adata.obs["cluster"].eq("root")
    ][:1]
    keep.loc[treated_root] = True
    adata = adata[keep].copy()
    scorer = PairScorer(
        adata,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )
    scorer.build_embedding(ordering_metric="pt", verbose=False)
    velocity = np.zeros_like(scorer._scorer._embedding_result.coordinates)
    scorer.load_velocity_vectors(velocity)
    scorer.fit(verbose=False)
    with pytest.raises(ValueError, match="partial designs"):
        scorer.score_all_conditions(min_cells=2, verbose=False)


def test_low_replicate_count_warns_but_scores_at_hard_minimum():
    adata = make_condition_adata(
        ["control", "treated"],
        n_replicates=2,
        root_cells_per_replicate=4,
    )
    scorer = PairScorer(
        adata,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )
    scorer.build_embedding(ordering_metric="pt", verbose=False)
    load_condition_velocities(scorer, {"control": 0.2, "treated": 0.8})
    with pytest.warns(RuntimeWarning, match="at least 4"):
        results = scorer.score_all_conditions(verbose=False)
    assert set(results) == {"control", "treated"}


def test_transition_scope_condition_blocks_cross_condition_edges():
    adata = make_condition_adata(
        ["control", "treated"],
        n_replicates=2,
        root_cells_per_replicate=2,
    )
    n = adata.n_obs
    matrix = np.zeros((n, n), dtype=float)
    labels = adata.obs["cluster"].astype(str).to_numpy()
    conditions = adata.obs["condition"].astype(str).to_numpy()
    for i in range(n):
        if labels[i] != "root":
            matrix[i, i] = 1.0
            continue
        same_a = np.flatnonzero((conditions == conditions[i]) & (labels == "A"))[0]
        other_b = np.flatnonzero((conditions != conditions[i]) & (labels == "B"))[0]
        matrix[i, same_a] = 0.5
        matrix[i, other_b] = 0.5

    pooled = PairScorer(
        adata.copy(),
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )
    pooled.build_embedding(ordering_metric="pt", verbose=False)
    pooled.fit(transition_matrix=matrix, transition_scope="pooled", verbose=False)

    blocked = PairScorer(
        adata.copy(),
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )
    blocked.build_embedding(ordering_metric="pt", verbose=False)
    blocked.fit(transition_matrix=matrix, transition_scope="condition", verbose=False)

    root = pooled.result.root_mask
    mean_b_pooled = pooled.result.directional_affinity[root, 1].mean()
    mean_b_blocked = blocked.result.directional_affinity[root, 1].mean()
    assert mean_b_blocked < mean_b_pooled
    assert blocked.transition_scope == "condition"


def test_condition_visualizations_smoke():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    stats = scorer.compare_conditions(
        metric="mean_commitment_contribution",
        fate="B",
        n_permutations=99,
        n_bootstrap=20,
        random_state=17,
        verbose=False,
    )
    figures = [
        scorer.plot_replicate_outcomes(metric="mean_commitment_contribution", fate="B"),
        scorer.plot_effects(stats),
        scorer.plot_commitment_heatmap(),
        scorer.plot_pseudotime_trends(pseudotime_key="pt", metric="specific_commitment", n_bins=4),
        scorer.plot_transition_coverage(),
        scorer.plot_status_composition(),
        scorer.plot_affinity_distributions(metric="mean_commitment", fate="B"),
        scorer.plot_star_grid(ncols=2, terminal_layout="ordering"),
        scorer.plot_gene_expression_star_grid("0", ncols=2),
        scorer.plot_rose_grid(ncols=2),
        scorer.plot_compare_conditions_bar(),
        scorer.plot_commitment_vector_radar(),
        scorer.plot_trajectory_shift(),
        scorer.plot_delta_CS_heatmap(stats),
    ]
    assert all(figure is not None for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_radar_replaces_rectangular_placeholder_axis():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    scorer.plot_compare_conditions_bar(ax=axes[0])
    returned = scorer.plot_commitment_vector_radar(ax=axes[1])
    assert returned is figure
    assert any(getattr(axis, "name", "") == "polar" for axis in figure.axes)
    plt.close(figure)


def test_multi_visualizations_smoke():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_multi_scorer()
    scorer.score_all_conditions(verbose=False)
    omnibus = scorer.compare_omnibus(
        metric="mean_commitment_contribution",
        n_permutations=49,
        random_state=23,
        verbose=False,
    )
    posthoc = scorer.compare_posthoc(
        metric="mean_commitment_contribution",
        n_permutations=49,
        random_state=24,
        verbose=False,
    )
    figures = [
        scorer.plot_star_grid(ncols=3),
        scorer.plot_gene_expression_star_grid("1", ncols=3),
        scorer.plot_rose_grid(ncols=3),
        scorer.plot_compare_conditions_bar(),
        scorer.plot_commitment_vector_radar(),
        scorer.plot_trajectory_shift(),
        scorer.plot_omnibus_summary(omnibus),
        scorer.plot_posthoc_heatmap(posthoc, fate="B"),
        scorer.plot_pairwise_delta_grid(posthoc, ncols=2),
    ]
    assert all(figure is not None for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_transition_scope_summary_reports_removed_cross_group_mass():
    adata = make_condition_adata(
        ["control", "treated"],
        n_replicates=2,
        root_cells_per_replicate=2,
    )
    n = adata.n_obs
    matrix = np.zeros((n, n), dtype=float)
    labels = adata.obs["cluster"].astype(str).to_numpy()
    conditions = adata.obs["condition"].astype(str).to_numpy()
    for i in range(n):
        if labels[i] != "root":
            matrix[i, i] = 1.0
            continue
        same_a = np.flatnonzero((conditions == conditions[i]) & (labels == "A"))[0]
        other_b = np.flatnonzero((conditions != conditions[i]) & (labels == "B"))[0]
        matrix[i, same_a] = 0.6
        matrix[i, other_b] = 0.4

    scorer = PairScorer(
        adata,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )
    scorer.build_embedding(ordering_metric="pt", verbose=False)
    scorer.fit(transition_matrix=matrix, transition_scope="condition", verbose=False)
    summary = scorer.transition_scope_summary(population="root")
    assert np.all(summary["mean_scope_coverage"] < 1.0)
    assert np.all(summary["mean_scope_removed_mass"] > 0.0)


def test_condition_preflight_and_decomposition_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    scorer = make_pair_scorer()
    scorer.score_all_conditions(verbose=False)
    report = scorer.preflight(ordering_metric="pt")
    assert report.ok
    fig = scorer.plot_commitment_decomposition(
        fate="B",
        n_bootstrap=20,
        random_state=12,
    )
    assert "decomposition" in fig.axes[0].get_title().lower()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_supplied_velocity_coverage_plot_is_informational_panel():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    scorer = make_pair_scorer()
    results = scorer.score_all_conditions(verbose=False)
    with pytest.warns(RuntimeWarning, match="fixed at 1"):
        fig = scorer.plot_transition_coverage(results)
    texts = [text.get_text() for text in fig.axes[0].texts]
    assert any("fixed at 1" in text for text in texts)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_conditions_ignore_missing_metadata_outside_selected_furcation():
    adata = make_condition_adata(["control", "treated"])
    extra_obs = pd.DataFrame(
        {
            "cluster": ["Other", "Other"],
            "condition": [np.nan, np.nan],
            "replicate": [np.nan, np.nan],
            "pt": [0.4, 0.6],
        },
        index=["outside_0", "outside_1"],
    )
    extra = anndata.AnnData(X=np.zeros((2, 2)), obs=extra_obs)
    combined = anndata.concat([adata, extra], join="outer", merge="same")

    scorer = PairScorer(
        combined,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )

    assert scorer.conditions == ["control", "treated"]


def test_missing_condition_on_selected_furcation_cell_raises():
    adata = make_condition_adata(["control", "treated"])
    selected_root = adata.obs.index[adata.obs["cluster"].eq("root")][0]
    adata.obs.loc[selected_root, "condition"] = np.nan

    with pytest.raises(ValueError, match="selected furcation cells"):
        PairScorer(
            adata,
            root="root",
            branches=["A", "B"],
            condition_obs_key="condition",
            replicate_obs_key="replicate",
            obs_key="cluster",
        )


def test_pairwise_log_ratio_accepts_reversed_fate_order():
    scorer = make_pair_scorer()
    results = scorer.score_all_conditions(verbose=False)
    forward = scorer.compare_conditions(
        results,
        metric="pairwise_log_commitment_ratio",
        fate_pair=("A", "B"),
        n_permutations=199,
        n_bootstrap=40,
        random_state=23,
        verbose=False,
    )
    reverse = scorer.compare_conditions(
        results,
        metric="pairwise_log_commitment_ratio",
        fate_pair=("B", "A"),
        n_permutations=199,
        n_bootstrap=40,
        random_state=23,
        verbose=False,
    )
    np.testing.assert_allclose(
        reverse["effect_b_minus_a"],
        -forward["effect_b_minus_a"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(reverse["pvalue"], forward["pvalue"], rtol=0.0, atol=0.0)


def test_default_condition_order_preserves_categorical_categories():
    adata = make_condition_adata(["control", "high", "low"])
    adata.obs["condition"] = pd.Categorical(
        adata.obs["condition"],
        categories=["control", "low", "high"],
        ordered=True,
    )
    scorer = MultiScorer(
        adata,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )
    assert scorer.conditions == ["control", "low", "high"]
    assert scorer.condition_order_source == "categorical_categories"


def test_default_condition_order_preserves_first_appearance_for_plain_strings():
    adata = make_condition_adata(["baseline", "late", "middle"])
    scorer = MultiScorer(
        adata,
        root="root",
        branches=["A", "B"],
        condition_obs_key="condition",
        replicate_obs_key="replicate",
        obs_key="cluster",
    )
    assert scorer.conditions == ["baseline", "late", "middle"]
    assert scorer.condition_order_source == "first_appearance"


def _invalid_mixed_audit():
    from scCS.mixed_models import MixedModelAudit

    return MixedModelAudit(
        fit=None,
        valid=False,
        failure_reason="synthetic singular fit",
        warning_messages=("Random effects covariance is singular",),
        fixed_effect_covariance=None,
        fixed_effect_covariance_min_eigenvalue=np.nan,
        fixed_effect_covariance_condition_number=np.nan,
        random_effect_variance_min=0.0,
    )


def test_pair_mixed_model_interface_fails_closed(monkeypatch):
    pytest.importorskip("statsmodels.api")
    import scCS.pairwise as pairwise_module

    scorer = make_pair_scorer()
    results = scorer.score_all_conditions(verbose=False)
    monkeypatch.setattr(
        pairwise_module,
        "fit_mixedlm_fail_closed",
        lambda *args, **kwargs: _invalid_mixed_audit(),
    )
    table = scorer.fit_mixed_model(
        metric="directional_affinity",
        fate="B",
        results=results,
    )
    assert not bool(table.iloc[0]["valid_fit"])
    assert np.isnan(table.iloc[0]["effect_b_minus_a"])
    assert np.isnan(table.iloc[0]["pvalue"])
    assert "singular" in table.iloc[0]["failure_reason"]


def test_multi_mixed_model_interface_fails_closed(monkeypatch):
    pytest.importorskip("statsmodels.api")
    import scCS.multicomparison as multi_module

    scorer = make_multi_scorer()
    results = scorer.score_all_conditions(verbose=False)
    monkeypatch.setattr(
        multi_module,
        "fit_mixedlm_fail_closed",
        lambda *args, **kwargs: _invalid_mixed_audit(),
    )
    table = scorer.fit_mixed_model(
        metric="directional_affinity",
        fate="B",
        results=results,
    )
    assert not bool(table.iloc[0]["valid_fit"])
    assert np.isnan(table.iloc[0]["wald_chi2"])
    assert np.isnan(table.iloc[0]["pvalue"])
    assert "singular" in table.iloc[0]["failure_reason"]
