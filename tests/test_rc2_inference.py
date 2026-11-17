"""Regression coverage for explicit replicate eligibility and rc2 diagnostics."""

from importlib.metadata import PackageNotFoundError

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from scCS import MultiScorer, PairScorer
from scCS.inference import two_group_permutation_test
from scCS import tutorial_cache
from test_v08_condition import make_condition_adata
from test_release_regressions import fitted


def scorer_with_graph(
    conditions=("control", "treated"), *, blocked=True, sparse_signal=False
):
    a = make_condition_adata(conditions)
    cls = PairScorer if len(conditions) == 2 else MultiScorer
    s = cls(
        a,
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
        replicate_obs_key="replicate",
    )
    s.build_embedding(ordering_metric="pt", verbose=False, cache_subset=False)
    p = sparse.lil_matrix((a.n_obs, a.n_obs))
    ia, ib = [np.flatnonzero(a.obs.cluster.eq(c))[0] for c in ["A", "B"]]
    for i, row in enumerate(a.obs.itertuples()):
        dead = (
            row.cluster != "root"
            or (blocked and row.condition == "treated" and row.replicate == "R0")
            or (sparse_signal and row.cluster == "root" and row.pt > 0.1)
        )
        if dead:
            p[i, i] = 1
        else:
            p[i, ia], p[i, ib] = 0.9, 0.1
    s.fit(
        p.tocsr(),
        scoring_mode="future_fate",
        future_fate_options={"min_anchor_cells": 1},
        verbose=False,
    )
    s.score_all_conditions(min_cells=2, verbose=False, write_to_adata=False)
    return s


@pytest.mark.parametrize(
    "method", ["compare_conditions", "hierarchical_bootstrap", "fit_mixed_model"]
)
def test_pair_rejects_wholly_undefined_replicate(method):
    s = scorer_with_graph()
    kwargs = dict(metric="conditional_fate_affinity", fate="A")
    if method == "hierarchical_bootstrap":
        kwargs.update(condition_a="control", condition_b="treated", n_bootstrap=5)
    with pytest.raises(ValueError, match="treated::R0"):
        getattr(s, method)(**kwargs)


@pytest.mark.parametrize(
    "method",
    ["compare_omnibus", "compare_posthoc", "compare_contrast", "fit_mixed_model"],
)
def test_multi_rejects_wholly_undefined_replicate(method):
    s = scorer_with_graph(("control", "treated", "third"))
    kwargs = dict(metric="conditional_fate_affinity", fate="A")
    if method == "compare_contrast":
        kwargs["weights"] = {"control": -1, "treated": 1, "third": 0}
    with pytest.raises(ValueError, match="treated::R0"):
        getattr(s, method)(**kwargs)


@pytest.mark.parametrize("method", ["compare_conditions", "hierarchical_bootstrap"])
def test_explicit_drop_reports_ids_and_keeps_unconditional_reach(method):
    s = scorer_with_graph()
    kwargs = dict(
        metric="conditional_fate_affinity", fate="A", undefined_replicates="drop"
    )
    if method == "hierarchical_bootstrap":
        kwargs.update(condition_a="control", condition_b="treated", n_bootstrap=10)
    with pytest.warns(UserWarning, match="treated::R0"):
        result = getattr(s, method)(**kwargs).iloc[0]
    assert result.n_replicates_requested == {"control": 5, "treated": 5}
    assert result.n_replicates_included == {"control": 5, "treated": 4}
    assert result.excluded_replicates[0]["replicate_id"] == "treated::R0"
    assert result.replicate_diagnostics[5]["mean_affinity_defined_fraction"] == 0
    reach = s.compare_conditions(metric="discounted_fate_reach", verbose=False).iloc[0]
    assert reach.n_replicates_b == 5
    assert reach.effect_b_minus_a < 0


@pytest.mark.parametrize(
    "method", ["compare_omnibus", "compare_posthoc", "compare_contrast"]
)
def test_multi_explicit_drop_is_audited(method):
    s = scorer_with_graph(("control", "treated", "third"))
    kwargs = dict(
        metric="conditional_fate_affinity",
        fate="A",
        undefined_replicates="drop",
        n_permutations=19,
    )
    if method == "compare_contrast":
        kwargs["weights"] = {"control": -1, "treated": 1, "third": 0}
    with pytest.warns(UserWarning, match="treated::R0"):
        result = getattr(s, method)(**kwargs)
    affected = [row for row in result.excluded_replicates if row]
    assert affected and all(row[0]["replicate_id"] == "treated::R0" for row in affected)


def test_drop_cannot_remove_an_entire_condition():
    s = scorer_with_graph()
    table = s._condition_results["treated"].replicate_summary
    table.loc[:, "directional_affinity::A"] = np.nan
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="included counts"):
        s.compare_conditions(
            metric="conditional_fate_affinity", fate="A", undefined_replicates="drop"
        )


def test_bootstrap_undefined_composition_draws_are_not_discarded():
    s = scorer_with_graph(blocked=False, sparse_signal=True)
    kwargs = dict(
        condition_a="control",
        condition_b="treated",
        metric="commitment_composition",
        fate="A",
        n_bootstrap=50,
    )
    with pytest.raises(ValueError, match="no draws were discarded"):
        s.hierarchical_bootstrap(**kwargs)
    with pytest.warns(UserWarning, match="no draws were discarded"):
        result = s.hierarchical_bootstrap(**kwargs, undefined_bootstrap="report").iloc[
            0
        ]
    assert result.n_bootstrap == 50
    assert result.n_bootstrap_undefined > 0
    assert np.isnan(result.ci_lower) and np.isnan(result.ci_upper)
    assert result.bootstrap_status == "undefined_draws"


@pytest.mark.parametrize("multi", [False, True])
def test_two_fate_family_collapses_only_normalized_endpoints(multi):
    s = scorer_with_graph(
        ("control", "treated", "third") if multi else ("control", "treated"),
        blocked=False,
    )
    method = s.compare_omnibus if multi else s.compare_conditions
    result = method(
        metric="conditional_fate_affinity", n_permutations=19, verbose=False
    )
    assert set(result.hypothesis_family_size) == {1}
    assert result.complementary_fates_collapsed.all()
    assert result.pvalue.nunique() == 1
    raw = method(metric="future_fate_contribution", n_permutations=19, verbose=False)
    assert set(raw.hypothesis_family_size) == {2}
    assert not raw.complementary_fates_collapsed.any()


def test_posthoc_family_counts_condition_pairs_once_for_complementary_fates():
    s = scorer_with_graph(("control", "treated", "third"), blocked=False)
    result = s.compare_posthoc(
        metric="conditional_fate_affinity", n_permutations=19, verbose=False
    )
    assert len(result) == 6
    assert set(result.hypothesis_family_size) == {3}
    assert (
        result.groupby(["condition_a", "condition_b"])
        .hypothesis_id.nunique()
        .eq(1)
        .all()
    )


def test_exact_three_vs_three_reports_attainability():
    with pytest.warns(UserWarning, match="minimum attainable p-value"):
        result = two_group_permutation_test(
            [0, 1, 2, 9, 10, 11], ["a"] * 3 + ["b"] * 3, "a", "b"
        )
    assert result.minimum_attainable_pvalue == 0.1
    assert result.pvalue == 0.1


def test_existing_cache_handles_missing_optional_distribution(tmp_path, monkeypatch):
    original = tutorial_cache.version

    def missing_igraph(name):
        if name == "igraph":
            raise PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(tutorial_cache, "version", missing_igraph)
    assert (
        tutorial_cache.cache_signature("pancreas")["dependencies"]["igraph"]
        == "not-installed"
    )
    a = ad.AnnData(np.ones((3, 2)))
    a.layers["velocity"] = np.ones(a.shape)
    a.uns["velocity_graph"] = sparse.eye(3, format="csr")
    tutorial_cache.save_tutorial_cache(a, "pancreas", cache_dir=tmp_path)
    assert (
        tutorial_cache.load_tutorial_cache("pancreas", cache_dir=tmp_path).shape
        == a.shape
    )
    monkeypatch.setattr(tutorial_cache, "version", original)
    assert tutorial_cache.load_tutorial_cache("pancreas", cache_dir=tmp_path) is None


def test_score_respects_disabled_subset_cache_and_compact_export(tmp_path):
    s, _ = fitted()
    s._cache_subset = False
    result = s.score(verbose=False, compact_metadata=True)
    assert s.adata_sub is None
    assert "selected_cell_ids" not in s.adata.uns["sccs_v08"]
    metadata = s.adata.uns["sccs_v08"]["future_fate"]
    assert "source_observation_ids" not in metadata
    assert "progression_values" not in metadata
    assert metadata["source_observation_fingerprint"]
    s.adata.write_h5ad(tmp_path / "compact.h5ad")
    s.score(verbose=False, cache_subset=True)
    assert list(s.adata_sub.obs_names) == list(result.cell_ids)
    s.adata = s.adata[::-1].copy()
    with pytest.raises(ValueError, match="observation order changed"):
        s.score(verbose=False)


def test_preflight_counts_dead_ends_separately_from_unreachable_anchors():
    s = scorer_with_graph()
    report = s.preflight(ordering_metric="pt").to_frame().set_index("code")
    assert report.loc["graph_dead_end_nonanchors", "value"] == 4
    assert report.loc["root_unreachable_selected_anchors", "value"] == 4
    assert report.loc["undefined_affinity", "value"] == pytest.approx(0.1)


def test_sensitivity_grid_preserves_fitted_state_and_reports_coverage():
    s, _ = fitted()
    result = s.result
    table = s.sensitivity_analysis(
        effective_horizons=[result.effective_horizon],
        anchor_quantiles=[result.anchor_quantile],
    )
    assert s.result is result
    assert len(table) == 1
    assert table.iloc[0].median_cfa_js_bits_vs_fitted < 1e-12
    assert table.iloc[0].n_jointly_defined > 0


def test_disabled_subset_cache_survives_save_load(tmp_path):
    from scCS import SingleScorer

    s, _ = fitted()
    s._cache_subset = False
    path = tmp_path / "no_subset.pkl"
    s.save(path)
    restored = SingleScorer.load(path, s.adata)
    assert restored.adata_sub is None
    restored.score(verbose=False)
    assert restored.adata_sub is None


def test_two_fate_adjustment_preserves_actual_effect_and_exact_pvalue():

    s = scorer_with_graph(blocked=False)
    for condition, result in s._condition_results.items():
        a = (
            np.linspace(0.1, 0.3, 5)
            if condition == "control"
            else np.linspace(0.7, 0.9, 5)
        )
        result.replicate_summary.loc[:, "directional_affinity::A"] = a
        result.replicate_summary.loc[:, "directional_affinity::B"] = 1 - a
    result = s.compare_conditions(metric="conditional_fate_affinity", verbose=False)
    first, second = result.iloc[0], result.iloc[1]
    assert first.effect_b_minus_a == pytest.approx(-second.effect_b_minus_a)
    assert first.pvalue == pytest.approx(2 / 252)
    assert first.pvalue_adj == first.pvalue
    assert second.pvalue_adj == first.pvalue


def test_scalability_reports_skipped_and_unrequested_without_allocating(tmp_path):
    import json
    import os
    from pathlib import Path
    import pandas as pd

    path = (
        Path(__file__).resolve().parents[1]
        / "tutorials/12_scalability_no_chunking.ipynb"
    )
    if not path.exists():
        pytest.skip("Executed notebooks are provided in the full source archive.")
    cells = json.loads(path.read_text())["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]

    def forbidden(*args):
        raise AssertionError(
            "A skipped workload must not allocate its graph or arrays."
        )

    scope = dict(
        pd=pd,
        np=np,
        os=os,
        OUTPUT_DIR=tmp_path,
        display=lambda *args: None,
        TARGET_CELLS=[10_000],
        REALISTIC_TARGET_CELLS=[100_000],
        MEMORY_SAFETY_FRACTION=0,
        N_FATES=4,
        GRAPH_DEGREE=4,
        GRAPH_OUTCOMES=3,
        REALISTIC_GRAPH_DEGREE=30,
        SCALABILITY_PROFILE="standard",
        RANDOM_SEED=1,
        metric_transform_bytes=lambda *args: 1,
        graph_solve_bytes=lambda *args: 1,
        current_rss_bytes=lambda: 0,
        run_metric_transform=forbidden,
        run_graph_solve=forbidden,
    )
    for prefix in [
        "metric_results = []",
        "graph_results = []",
        "realistic_graph_results = []",
    ]:
        exec(next(s for s in code if s.startswith(prefix)), scope)
    for key in ["metric_results", "graph_results", "realistic_graph_results"]:
        assert set(scope[key]["status"]) == {"SKIPPED_INSUFFICIENT_MEMORY"}
    exec(next(s for s in code if s.startswith("status_200m =")), scope)
    assert set(scope["status_200m"].status) == {"NOT_REQUESTED"}
    assert len(pd.read_csv(tmp_path / "status_200m.csv")) == 2


@pytest.mark.parametrize(
    "method", ["plot_replicate_outcomes", "plot_affinity_distributions"]
)
def test_undefined_replicates_are_visible_in_descriptive_plot_counts(method):
    import matplotlib.pyplot as plt

    s = scorer_with_graph()
    with pytest.warns(UserWarning, match="treated::R0"):
        figure = getattr(s, method)(metric="conditional_fate_affinity", fate="A")
    assert any(
        "4/5 defined" in label.get_text() for label in figure.axes[0].get_xticklabels()
    )
    plt.close(figure)
