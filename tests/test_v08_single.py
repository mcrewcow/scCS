"""End-to-end tests for the v0.8 SingleScorer public workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

anndata = pytest.importorskip("anndata")

from scCS import Furcation, FurcationScoreResult, SingleScorer, __version__
from scCS.population import PopulationCommitmentSummary


def make_adata():
    labels = np.array(["root"] * 4 + ["A"] * 3 + ["B"] * 3)
    ordering = np.array([0.05, 0.15, 0.25, 0.35, 0.65, 0.80, 0.95, 0.65, 0.80, 0.95])
    X = np.arange(len(labels) * 4, dtype=float).reshape(len(labels), 4)
    adata = anndata.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(len(labels))]
    adata.obs["cluster"] = pd.Categorical(labels)
    adata.obs["velocity_pseudotime"] = ordering
    adata.obs["condition"] = pd.Categorical(
        ["control", "control", "treated", "treated"]
        + ["control", "treated", "treated"]
        + ["control", "control", "treated"]
    )
    adata.var["symbol"] = ["Gene0", "Gene1", "Gene2", "Gene3"]
    adata.layers["counts"] = sparse.csr_matrix(X + 1.0)
    # Native scVelo star-embedding plots require the velocity layer even when
    # the transition matrix is supplied explicitly.
    adata.layers["velocity"] = np.zeros_like(X, dtype=float)
    adata.raw = adata.copy()
    return adata


def make_transition():
    # Directed transition matrix over all 10 cells.
    T = np.zeros((10, 10), dtype=float)
    T[0, 1] = 1.0
    T[1, 2] = 1.0
    T[2, 4] = 0.8
    T[2, 7] = 0.2
    T[3, 5] = 0.9
    T[3, 8] = 0.1
    # Terminal cells progress outward or remain at terminal endpoints.
    T[4, 5] = 1.0
    T[5, 6] = 1.0
    T[6, 6] = 1.0
    T[7, 8] = 1.0
    T[8, 9] = 1.0
    T[9, 9] = 1.0
    return sparse.csr_matrix(T)


def make_fitted_scorer(write_to_adata=False):
    adata = make_adata()
    scorer = SingleScorer(
        adata,
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
    )
    scorer.build_embedding(
        ordering_metric="pseudotime",
        write_to_adata=write_to_adata,
        verbose=False,
    )
    scorer.fit(
        transition_matrix=make_transition(),
        min_transition_coverage=0.0,
        verbose=False,
    )
    return scorer


def test_constructor_preserves_familiar_root_branches_api():
    scorer = SingleScorer(
        make_adata(),
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
    )
    assert scorer.root == "root"
    assert scorer.branches == ["A", "B"]
    assert scorer.furcation.k == 2


def test_constructor_accepts_furcation_object():
    furcation = Furcation(obs_key="cluster", root="root", terminals=["A", "B"])
    scorer = SingleScorer(make_adata(), furcation=furcation)
    assert scorer.furcation is furcation


def test_build_embedding_is_simplex_and_deterministic():
    scorer = SingleScorer(make_adata(), root="root", branches=["A", "B"], obs_key="cluster")
    scorer.build_embedding(verbose=False)
    first = scorer.embedding
    scorer.build_embedding(verbose=False)
    second = scorer.embedding
    assert first.shape == (10, 2)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_fit_and_score_return_new_result():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    assert isinstance(result, FurcationScoreResult)
    assert result.fate_names == ("A", "B")
    assert result.k == 2
    assert result.directional_affinity.shape == (10, 2)
    np.testing.assert_allclose(result.directional_affinity.sum(axis=1), 1.0)
    assert np.all((result.commitment_strength >= 0) & (result.commitment_strength <= 1))


def test_score_writes_scverse_friendly_outputs():
    scorer = make_fitted_scorer()
    scorer.score(write_to_adata=True, verbose=False)
    adata = scorer.adata
    assert "X_sccs_score" in adata.obsm
    assert "sccs_directional_affinity" in adata.obsm
    assert "sccs_commitment_contribution" in adata.obsm
    assert "sccs_fate_cosine_similarity" in adata.obsm
    assert "sccs_directional_entropy" in adata.obs
    assert "sccs_commitment_entropy" in adata.obs
    assert "sccs_nearest_fate_angle_degrees" in adata.obs
    assert "sccs_specific_commitment" in adata.obs
    assert "sccs_transition_coverage" in adata.obs
    assert "sccs" in adata.uns
    assert adata.uns["sccs"]["schema_version"] == "0.8"


def test_explicit_population_mask_changes_only_summary():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    root = result.root_mask
    mask = root.copy()
    root_rows = np.flatnonzero(root)
    mask[root_rows[:2]] = False
    returned = scorer.score(cell_mask=mask, write_to_adata=False, verbose=False)
    assert returned is result
    assert scorer.population_summary.n_cells == 2
    np.testing.assert_allclose(
        scorer.population_summary.mean_commitment_contribution,
        result.commitment_contribution[mask].mean(axis=0),
    )


def test_score_per_subset_summarizes_root_only():
    scorer = make_fitted_scorer()
    scorer.score(write_to_adata=False, verbose=False)
    summaries = scorer.score_per_subset("condition", population="root", min_cells=1, verbose=False)
    assert set(summaries) == {"control", "treated"}
    assert all(isinstance(value, PopulationCommitmentSummary) for value in summaries.values())
    assert sum(value.n_cells for value in summaries.values()) == 4


def test_load_velocity_vectors_accepts_two_fate_vx_vy():
    scorer = SingleScorer(make_adata(), root="root", branches=["A", "B"], obs_key="cluster")
    scorer.build_embedding(verbose=False)
    vx = np.linspace(-1.0, 1.0, 10)
    vy = np.zeros(10)
    scorer.load_velocity_vectors(vx, vy)
    scorer.fit(verbose=False)
    result = scorer.score(write_to_adata=False, verbose=False)
    assert result.projection.transition_coverage.min() == 1.0


def test_display_jitter_does_not_modify_scientific_scores():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    before = result.directional_affinity.copy()
    fig1 = scorer.plot_star(result, jitter=0.0, seed=1)
    fig2 = scorer.plot_star(result, jitter=0.2, seed=99)
    np.testing.assert_allclose(result.directional_affinity, before, atol=0.0, rtol=0.0)
    import matplotlib.pyplot as plt

    plt.close(fig1)
    plt.close(fig2)


def test_terminal_display_uses_fitted_ordering_on_one_shared_scale_by_default():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    coordinates, directions = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="ordering",
        terminal_inner_radius=0.15,
    )
    terminal = result.terminal_mask
    selected_ordering = scorer._selected_display_ordering(result)
    progress = scorer._global_ordering_progress(selected_ordering)
    expected = result.embedding.arm_scale * (0.15 + 0.85 * progress)

    for fate in result.fate_names:
        mask = terminal & (result.embedding.terminal_names == fate)
        radii = coordinates[mask] @ directions[fate]
        np.testing.assert_allclose(radii, expected[mask], atol=1e-12, rtol=0.0)
        order = np.argsort(selected_ordering[mask], kind="stable")
        assert np.all(np.diff(radii[order]) >= -1e-12)


def test_terminal_pseudotime_alias_matches_generic_ordering_layout():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    ordering, _ = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="ordering",
    )
    pseudotime, _ = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="pseudotime",
    )
    branch, _ = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="branch",
    )
    np.testing.assert_allclose(pseudotime, ordering, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(branch, ordering, atol=0.0, rtol=0.0)


def test_terminal_rank_layout_equalizes_visual_branch_coverage():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    coordinates, directions = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="rank",
        terminal_inner_radius=0.15,
    )
    terminal = result.terminal_mask
    for fate in result.fate_names:
        mask = terminal & (result.embedding.terminal_names == fate)
        radii = coordinates[mask] @ directions[fate]
        assert np.isclose(radii.min(), 0.15 * result.embedding.arm_scale)
        assert np.isclose(radii.max(), result.embedding.arm_scale)
        assert np.ptp(radii) > 0.5 * result.embedding.arm_scale


def test_terminal_endpoint_layout_remains_available_and_display_only():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    before_affinity = result.directional_affinity.copy()
    before_strength = result.commitment_strength.copy()

    endpoint, directions = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="endpoint",
    )
    terminal = result.terminal_mask
    for fate in result.fate_names:
        mask = terminal & (result.embedding.terminal_names == fate)
        expected = result.embedding.arm_scale * directions[fate]
        np.testing.assert_allclose(
            endpoint[mask],
            np.repeat(expected[None, :], int(mask.sum()), axis=0),
            atol=0.0,
            rtol=0.0,
        )

    branch, _ = scorer._display_coordinates(
        result,
        jitter=0.03,
        terminal_radial_jitter=0.015,
        terminal_layout="branch",
        seed=11,
    )
    assert np.std(branch[terminal], axis=0).sum() > 0
    np.testing.assert_allclose(result.directional_affinity, before_affinity, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(result.commitment_strength, before_strength, atol=0.0, rtol=0.0)


def test_terminal_branch_display_uses_frozen_fitted_ordering():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    before, _ = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="branch",
    )
    scorer.adata.obs["velocity_pseudotime"] = scorer.adata.obs["velocity_pseudotime"].to_numpy()[
        ::-1
    ]
    after, _ = scorer._display_coordinates(
        result,
        jitter=0.0,
        terminal_radial_jitter=0.0,
        terminal_layout="branch",
    )
    np.testing.assert_allclose(after, before, atol=0.0, rtol=0.0)


def test_single_legacy_relevant_visualizations_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    subsets = scorer.score_per_subset(
        "condition",
        population="root",
        min_cells=1,
        verbose=False,
    )
    figures = [
        scorer.plot_star_panels(result, ncols=2),
        scorer.plot_rose(result),
        scorer.plot_pairwise_cs(result),
        scorer.plot_commitment_bar(result),
        scorer.plot_commitment_heatmap(result, max_cells=20),
        scorer.plot_subset_comparison(subsets),
        scorer.plot_expression_trends(["0", "1"], result, n_bins=4, ncols=2),
    ]
    assert all(figure is not None for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_star_figures_remain_compatible_with_explicit_tight_layout():
    """Package-created colorbar figures must not lock a layout engine."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figures = [
        scorer.plot_star(result, color_by="commitment_strength"),
        scorer.plot_star_panels(
            result,
            panels=["commitment_strength", "directional_entropy"],
            ncols=2,
        ),
        scorer.plot_gene_expression_star(["0", "1"], result, ncols=2),
    ]
    for figure in figures:
        # This is a common notebook pattern and previously raised on
        # Matplotlib 3.10 after a constrained-layout colorbar was created.
        figure.tight_layout()
        plt.close(figure)


def test_gene_expression_star_supports_x_layer_raw_and_gene_symbols():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    x_values, x_label = scorer._gene_expression_values(result, "1")
    layer_values, layer_label = scorer._gene_expression_values(
        result,
        "Gene1",
        layer="counts",
        gene_symbols="symbol",
    )
    raw_values, raw_label = scorer._gene_expression_values(
        result,
        "2",
        use_raw=True,
    )
    expected_x = np.asarray(scorer.adata.X[result.embedding.selected_indices, 1]).reshape(-1)
    expected_layer = np.asarray(
        scorer.adata.layers["counts"][result.embedding.selected_indices, 1].toarray()
    ).reshape(-1)
    np.testing.assert_allclose(x_values, expected_x)
    np.testing.assert_allclose(layer_values, expected_layer)
    assert x_label == "X"
    assert layer_label == "layer=counts"
    assert raw_label == "raw"
    assert raw_values.shape == (result.n_cells,)

    figures = [
        scorer.plot_gene_expression_star(["0", "1"], result, ncols=2),
        scorer.plot_gene_expression_star(
            "Gene1",
            result,
            layer="counts",
            gene_symbols="symbol",
            terminal_layout="ordering",
        ),
        scorer.plot_gene_expression_star("2", result, use_raw=True),
    ]
    assert all(figure is not None for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_gene_expression_star_rejects_unknown_or_ambiguous_inputs():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    with pytest.raises(KeyError, match="not found"):
        scorer.plot_gene_expression_star("missing_gene", result)
    with pytest.raises(ValueError, match="cannot be used together"):
        scorer.plot_gene_expression_star("0", result, layer="counts", use_raw=True)
    with pytest.raises(ValueError, match="duplicate"):
        scorer.plot_gene_expression_star(["0", "0"], result)


def test_external_star_color_values_are_display_only():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    before = result.directional_affinity.copy()
    values = np.linspace(0.0, 1.0, result.n_cells)
    fig = scorer.plot_star(
        result,
        color_values=values,
        color_label="external",
        sort_by_color=True,
        terminal_layout="ordering",
    )
    np.testing.assert_allclose(result.directional_affinity, before, atol=0.0, rtol=0.0)
    plt.close(fig)


def test_save_and_load_roundtrip(tmp_path):
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    path = tmp_path / "scorer.pkl"
    scorer.save(path)
    restored = SingleScorer.load(path, scorer.adata)
    restored_result = restored.score(write_to_adata=False, verbose=False)
    np.testing.assert_allclose(
        restored_result.directional_affinity,
        result.directional_affinity,
    )


def test_summary_contains_commitment_language():
    result = make_fitted_scorer().score(write_to_adata=False, verbose=False)
    text = result.summary()
    assert "FurcationScoreResult" in text
    assert "commitment" in text.lower()


def test_score_before_fit_raises():
    scorer = SingleScorer(make_adata(), root="root", branches=["A", "B"], obs_key="cluster")
    with pytest.raises(RuntimeError, match="not fitted"):
        scorer.score()


def test_fit_before_embedding_raises():
    scorer = SingleScorer(make_adata(), root="root", branches=["A", "B"], obs_key="cluster")
    with pytest.raises(RuntimeError, match="build_embedding"):
        scorer.fit(transition_matrix=make_transition())


def test_terminal_pseudotime_permutation_does_not_change_scientific_scores():
    first_adata = make_adata()
    second_adata = first_adata.copy()
    terminal = second_adata.obs["cluster"].isin(["A", "B"]).to_numpy()
    values = second_adata.obs["velocity_pseudotime"].to_numpy().copy()
    values[terminal] = values[terminal][::-1]
    second_adata.obs["velocity_pseudotime"] = values

    outputs = []
    for adata in (first_adata, second_adata):
        scorer = SingleScorer(
            adata,
            root="root",
            branches=["A", "B"],
            obs_key="cluster",
        )
        scorer.build_embedding(verbose=False)
        scorer.fit(
            transition_matrix=make_transition(),
            min_transition_coverage=0.0,
            verbose=False,
        )
        outputs.append(scorer.score(write_to_adata=False, verbose=False))

    np.testing.assert_allclose(
        outputs[0].embedding.coordinates,
        outputs[1].embedding.coordinates,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        outputs[0].directional_affinity,
        outputs[1].directional_affinity,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        outputs[0].commitment_strength,
        outputs[1].commitment_strength,
        atol=0.0,
        rtol=0.0,
    )


def test_all_terminal_scientific_coordinates_have_equal_radius():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    terminal_coordinates = result.embedding.coordinates[result.terminal_mask]
    np.testing.assert_allclose(
        np.linalg.norm(terminal_coordinates, axis=1),
        result.embedding.arm_scale,
        atol=1e-12,
    )


def test_preflight_and_direction_strength_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    scorer = make_fitted_scorer()
    scorer.score(write_to_adata=False, verbose=False)
    report = scorer.preflight(ordering_metric="pseudotime")
    assert report.ok
    assert {d.code for d in report.diagnostics} >= {"furcation_valid", "transition_coverage"}
    fig = scorer.plot_direction_strength_map()
    assert fig.axes[0].get_xlabel() == "Directional specificity"
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_result_metadata_contains_versioned_scientific_configuration():
    scorer = make_fitted_scorer()
    scorer.score(write_to_adata=True, verbose=False)
    metadata = scorer.adata.uns["sccs"]
    assert metadata["schema_version"] == "0.8"
    assert metadata["package_version"] == __version__
    assert metadata["affinity_model"] == "calibrated_cosine_softmax"
    assert metadata["geometry"]["type"].endswith("fixed_terminal_vertices")
    assert "projection" in metadata


def test_preflight_warns_for_highly_discrete_root_ordering():
    adata = make_adata()
    adata.obs["coarse_time"] = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2], dtype=float)
    scorer = SingleScorer(
        adata,
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
    )
    report = scorer.preflight(
        ordering_metric="coarse_time",
        check_velocity=False,
    )
    by_code = {diagnostic.code: diagnostic for diagnostic in report.diagnostics}
    assert "ordering_highly_discrete" in by_code
    assert by_code["ordering_highly_discrete"].level == "warning"


def test_result_exposes_cell_count_convenience_properties():
    result = make_fitted_scorer().score(write_to_adata=False, verbose=False)
    assert result.n_cells == 10
    assert result.n_root == 4
    assert result.n_terminal == 6


def test_star_supports_fate_specific_and_dominant_fate_colors():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figures = [
        scorer.plot_star(result, color_by="dominant_fate", jitter=0.0),
        scorer.plot_star(result, color_by="dominant_affinity", jitter=0.0),
        scorer.plot_star(result, color_by="root_dominant_affinity", jitter=0.0),
        scorer.plot_star(result, color_by="affinity:A", jitter=0.0),
        scorer.plot_star(result, color_by="commitment_affinity:A", jitter=0.0),
        scorer.plot_star(result, color_by="commitment_contribution:A", jitter=0.0),
    ]
    import matplotlib.pyplot as plt

    for figure in figures:
        assert figure is not None
        plt.close(figure)


def test_star_default_titles_describe_the_displayed_metric():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_star_panels(
        result,
        panels=["population", "root_dominant_affinity", "directional_entropy"],
        ncols=3,
        ordering_label="Velocity pseudotime",
    )
    titles = [axis.get_title() for axis in figure.axes[:3]]
    assert titles[0].startswith("Cell annotation")
    assert titles[1].startswith("Dominant directional affinity in root cells")
    assert titles[2].startswith("Directional entropy")
    assert all("Terminal placement: Velocity pseudotime" in title for title in titles)
    plt.close(figure)


def test_commitment_heatmap_supports_population_annotation_strip():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_commitment_heatmap(
        result,
        metric="directional_affinity",
        population="all",
        sort_by="population_then_ordering",
        row_annotation="population",
        max_cells=20,
    )
    titles = {
        axis.get_title(loc=loc) for axis in figure.axes for loc in ("left", "center", "right")
    }
    assert "Type" in titles
    assert "Fate" in titles
    assert any(axis.get_title() == "Directional Affinity" for axis in figure.axes)
    plt.close(figure)


def test_population_commitment_plots_all_supported_metrics():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figures = []
    for metric in [
        "total_commitment_mass",
        "mean_commitment_contribution",
        "commitment_composition",
        "pairwise_log_commitment_ratio",
    ]:
        figures.append(
            scorer.plot_population_commitment(
                result,
                population="root",
                metric=metric,
            )
        )
    import matplotlib.pyplot as plt

    for figure in figures:
        assert figure is not None
        plt.close(figure)


def test_furcation_velocity_pseudotime_uses_induced_graph(monkeypatch):
    """The public helper must subset the fitted graph, not recompute velocity."""
    import sys
    import types

    base = make_adata()
    extra_obs = pd.DataFrame(
        {
            "cluster": pd.Categorical(["Other", "Other"]),
            "velocity_pseudotime": [0.4, 0.6],
            "condition": pd.Categorical(["control", "treated"]),
        },
        index=["outside_0", "outside_1"],
    )
    extra = anndata.AnnData(X=np.zeros((2, base.n_vars)), obs=extra_obs)
    adata = anndata.concat([base, extra], join="outer", merge="same")

    graph = sparse.lil_matrix((adata.n_obs, adata.n_obs), dtype=float)
    for index in range(base.n_obs - 1):
        graph[index, index + 1] = 1.0
    graph[base.n_obs - 1, base.n_obs - 1] = 1.0
    graph[base.n_obs, base.n_obs] = 1.0
    graph[base.n_obs + 1, base.n_obs + 1] = 1.0
    graph = graph.tocsr()
    adata.uns["velocity_graph"] = graph
    adata.uns["velocity_graph_neg"] = sparse.csr_matrix(graph.shape)

    calls = {}
    fake = types.ModuleType("scvelo")
    fake.tl = types.SimpleNamespace()

    def fake_velocity_pseudotime(local, **kwargs):
        calls["shape"] = local.uns["velocity_graph"].shape
        calls["root_key"] = kwargs["root_key"]
        calls["n_dcs"] = kwargs["n_dcs"]
        local.obs["velocity_pseudotime"] = np.linspace(0.0, 1.0, local.n_obs)

    fake.tl.velocity_pseudotime = fake_velocity_pseudotime
    monkeypatch.setitem(sys.modules, "scvelo", fake)

    scorer = SingleScorer(
        adata,
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
    )
    scorer.compute_velocity_pseudotime(
        root_key="cell_1",
        scope="furcation",
        key_added="local_velocity_pseudotime",
        n_dcs=4,
        verbose=False,
    )

    selected = adata.obs["cluster"].astype(str).isin(["root", "A", "B"])
    assert calls["shape"] == (int(selected.sum()), int(selected.sum()))
    assert calls["root_key"] == "cell_1"
    assert calls["n_dcs"] == 4
    assert np.all(np.isfinite(adata.obs.loc[selected, "local_velocity_pseudotime"]))
    assert adata.obs.loc[~selected, "local_velocity_pseudotime"].isna().all()
    metadata = adata.uns["sccs_v08"]["pseudotime"]["local_velocity_pseudotime"]
    assert metadata["scope"] == "furcation"
    assert metadata["n_cells"] == int(selected.sum())
    assert metadata["n_connected_components"] == 1
    assert metadata["largest_component_fraction"] == 1.0


def test_build_embedding_can_skip_cached_subset_and_adata_output():
    scorer = SingleScorer(make_adata(), root="root", branches=["A", "B"], obs_key="cluster")
    scorer.build_embedding(
        ordering_metric="velocity_pseudotime",
        write_to_adata=False,
        cache_subset=False,
        verbose=False,
    )
    assert scorer.adata_sub is None
    assert "X_sccs_score" not in scorer.adata.obsm
    assert scorer.embedding.shape == (10, 2)


def test_score_per_subset_works_without_cached_subset():
    scorer = SingleScorer(make_adata(), root="root", branches=["A", "B"], obs_key="cluster")
    scorer.build_embedding(
        ordering_metric="velocity_pseudotime",
        write_to_adata=False,
        cache_subset=False,
        verbose=False,
    )
    scorer.fit(
        transition_matrix=make_transition(),
        min_transition_coverage=0.0,
        verbose=False,
    )
    scorer.score(write_to_adata=False, verbose=False)
    summaries = scorer.score_per_subset(
        "condition",
        population="root",
        min_cells=1,
        verbose=False,
    )
    assert scorer.adata_sub is None
    assert set(summaries) == {"control", "treated"}
    assert sum(summary.n_cells for summary in summaries.values()) == 4


def test_result_reuses_projection_velocity_storage():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    assert result.projected_velocity is result.projection.velocity


def test_score_per_subset_ignores_missing_values():
    """Nullable subset labels must not trigger ambiguous pd.NA comparisons."""
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    labels = pd.Series(pd.NA, index=scorer.adata.obs_names, dtype="string")
    root_ids = result.cell_ids[result.root_mask]
    midpoint = len(root_ids) // 2
    labels.loc[root_ids[:midpoint]] = "early"
    labels.loc[root_ids[midpoint:]] = "late"
    scorer.adata.obs["subset_with_missing"] = labels

    summaries = scorer.score_per_subset(
        "subset_with_missing",
        population="root",
        min_cells=1,
    )
    assert set(summaries) == {"early", "late"}
    assert sum(summary.n_cells for summary in summaries.values()) == int(result.root_mask.sum())


def test_projection_geometry_diagnostics_reconstruct_root_branch_velocity_exactly():
    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    diagnostics = scorer.projection_geometry_diagnostics(result)
    assert diagnostics.n_root_cells == result.n_root
    assert diagnostics.max_abs_branch_error < 1e-12
    assert diagnostics.max_abs_affinity_error < 1e-12
    assert diagnostics.branch_rmse < 1e-12
    if diagnostics.n_informative_cells:
        assert diagnostics.median_direction_cosine > 1.0 - 1e-12
    if diagnostics.n_decisive_cells:
        assert diagnostics.dominant_fate_agreement == 1.0


def test_projection_geometry_diagnostic_plot_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_projection_geometry_diagnostics(result, max_cells=20)
    titles = [axis.get_title() for axis in figure.axes]
    assert any("Direct retained transition mass" in title for title in titles)
    assert any("Directional affinity" in title for title in titles)
    plt.close(figure)


def test_projection_geometry_diagnostics_reject_preprojected_vectors():
    scorer = SingleScorer(make_adata(), root="root", branches=["A", "B"], obs_key="cluster")
    scorer.build_embedding(verbose=False)
    scorer.load_velocity_vectors(np.ones((10, 2)))
    scorer.fit(verbose=False)
    scorer.score(write_to_adata=False, verbose=False)
    with pytest.raises(RuntimeError, match="transition matrix"):
        scorer.projection_geometry_diagnostics()


def test_star_color_mask_keeps_uncolored_cells_as_neutral_context():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_star(
        result,
        color_by="affinity:A",
        color_mask=result.root_mask,
        jitter=0.0,
    )
    axis = figure.axes[0]
    # One neutral collection plus one colored collection.
    assert len(axis.collections) >= 2
    plt.close(figure)


def test_four_fate_auto_rose_uses_branch_direction_bins():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.array(["root"] * 8 + [fate for fate in ["A", "B", "C", "D"] for _ in range(3)])
    ordering = np.linspace(0.0, 1.0, len(labels))
    adata = anndata.AnnData(X=np.zeros((len(labels), 2)))
    adata.obs["cluster"] = pd.Categorical(labels)
    adata.obs["velocity_pseudotime"] = ordering
    transition = np.zeros((len(labels), len(labels)), dtype=float)
    terminal_starts = {"A": 8, "B": 11, "C": 14, "D": 17}
    for row in range(8):
        fate = ["A", "B", "C", "D"][row % 4]
        transition[row, terminal_starts[fate]] = 0.7
        transition[row, (row + 1) % 8] = 0.3
    for start in terminal_starts.values():
        transition[start : start + 3, start : start + 3] = np.eye(3)

    scorer = SingleScorer(adata, root="root", branches=["A", "B", "C", "D"], obs_key="cluster")
    scorer.build_embedding(verbose=False)
    scorer.fit(
        transition_matrix=sparse.csr_matrix(transition),
        min_transition_coverage=0.0,
        verbose=False,
    )
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_rose(result, mode="auto", n_bins=24)
    axis = figure.axes[0]
    assert "branch direction" in axis.get_title()
    # Binned mirrored petals should create many bars, not one bar per fate.
    assert len(axis.patches) > 2 * result.k
    plt.close(figure)


def test_commitment_heatmap_supports_multiple_row_and_fate_strips():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_commitment_heatmap(
        result,
        metric="directional_affinity",
        population="all",
        sort_by="population_then_ordering",
        row_annotation=("population", "dominant_affinity"),
        show_fate_strip=True,
        max_cells=20,
    )
    titles = {
        axis.get_title(loc=loc) for axis in figure.axes for loc in ("left", "center", "right")
    }
    assert "Type" in titles
    assert "Branch" in titles
    assert "Fate" in titles
    plt.close(figure)


def test_projection_diagnostic_plot_has_branch_and_fate_strips():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_projection_geometry_diagnostics(
        result,
        max_cells=20,
        show_branch_strip=True,
        show_fate_strip=True,
    )
    titles = {
        axis.get_title(loc=loc) for axis in figure.axes for loc in ("left", "center", "right")
    }
    ylabels = {axis.get_ylabel() for axis in figure.axes}
    assert "Direct\nbranch" in titles
    assert "Fate" in ylabels
    plt.close(figure)


def test_display_velocity_projection_and_star_plots_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    coords, projection = scorer.display_velocity_projection(result)
    assert coords.shape == (result.n_cells, 2)
    assert projection.velocity.shape == (result.n_cells, 2)
    assert np.any(projection.velocity_defined[result.root_mask])

    cell_figure = scorer.plot_velocity_star(
        result,
        mode="cell",
        population="root",
        max_arrows=20,
    )
    grid_figure = scorer.plot_velocity_star(
        result,
        mode="grid",
        population="root",
        grid_size=8,
        min_cells_per_bin=1,
    )
    assert "RNA-velocity vectors" in cell_figure.axes[0].get_title()
    assert "velocity grid" in grid_figure.axes[0].get_title().lower()
    plt.close(cell_figure)
    plt.close(grid_figure)


def test_four_fate_rose_within_fate_normalization_uses_fractional_scale():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = np.array(["root"] * 8 + [fate for fate in ["A", "B", "C", "D"] for _ in range(3)])
    ordering = np.linspace(0.0, 1.0, len(labels))
    adata = anndata.AnnData(X=np.zeros((len(labels), 2)))
    adata.obs["cluster"] = pd.Categorical(labels)
    adata.obs["velocity_pseudotime"] = ordering
    transition = np.zeros((len(labels), len(labels)), dtype=float)
    terminal_starts = {"A": 8, "B": 11, "C": 14, "D": 17}
    for row in range(8):
        fate = ["A", "B", "C", "D"][row % 4]
        transition[row, terminal_starts[fate]] = 0.7
        transition[row, (row + 1) % 8] = 0.3
    for start in terminal_starts.values():
        transition[start : start + 3, start : start + 3] = np.eye(3)

    scorer = SingleScorer(adata, root="root", branches=["A", "B", "C", "D"], obs_key="cluster")
    scorer.build_embedding(verbose=False)
    scorer.fit(
        transition_matrix=sparse.csr_matrix(transition),
        min_transition_coverage=0.0,
        verbose=False,
    )
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_rose(
        result,
        mode="deviation",
        n_bins=24,
        normalization="within_fate",
    )
    heights = [patch.get_height() for patch in figure.axes[0].patches]
    assert heights
    assert max(heights) <= 1.0 + 1e-12
    assert any(
        "Within-fate velocity-mass fraction" in text.get_text() for text in figure.axes[0].texts
    )
    plt.close(figure)


def test_native_scvelo_star_embedding_includes_root_and_terminals():
    pytest.importorskip("scvelo")
    scorer = make_fitted_scorer(write_to_adata=True)
    result = scorer.score(write_to_adata=False, verbose=False)
    star = scorer.prepare_scvelo_star_embedding(result, write_to_adata=True)
    assert star.n_obs == result.n_cells
    assert star.obsm["X_sccs"].shape == (result.n_cells, 2)
    assert star.obsm["velocity_sccs"].shape == (result.n_cells, 2)
    assert np.isfinite(star.obsm["X_sccs"]).all()
    assert "X_sccs" in scorer.adata.obsm
    assert "velocity_sccs" in scorer.adata.obsm
    selected = result.embedding.selected_indices
    np.testing.assert_allclose(
        scorer.adata.obsm["X_sccs"][selected],
        star.obsm["X_sccs"],
    )


def test_native_scvelo_velocity_grid_smoke_all_cells():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("scvelo")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_velocity_embedding_grid(
        result,
        population="all",
        density=0.2,
        smooth=0.5,
        min_mass=0.1,
        n_neighbors=3,
        write_to_adata=False,
    )
    assert "RNA-velocity grid" in figure.axes[0].get_title()
    plt.close(figure)


def test_display_rose_uses_scvelo_star_angles_and_bins():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("scvelo")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_rose(
        result,
        population="all",
        mode="display",
        n_bins=24,
        normalization="mass",
    )
    axis = figure.axes[0]
    assert "scVelo velocity magnitude" in axis.get_title()
    assert len(axis.patches) == 24
    plt.close(figure)


def test_transition_expected_star_velocity_points_root_toward_furcation():
    scorer = make_fitted_scorer(write_to_adata=True)
    result = scorer.score(write_to_adata=False, verbose=False)
    star = scorer.prepare_scvelo_star_embedding(
        result,
        projection_mode="transition",
        write_to_adata=False,
    )
    root_velocity = np.asarray(star.obsm["velocity_sccs"])[result.root_mask]
    # The display incoming root arm points left, so positive x is centerward.
    assert np.all(root_velocity[:, 0] > 0.0)


def test_root_progression_direction_diagnostics_validate_geometry_identity():
    scorer = make_fitted_scorer(write_to_adata=True)
    result = scorer.score(write_to_adata=False, verbose=False)
    diagnostics = scorer.root_progression_direction_diagnostics(result)
    assert diagnostics.max_abs_progression_identity_error < 1e-12
    assert diagnostics.selected_root_progress == pytest.approx(0.0)
    assert diagnostics.selected_root_radius == pytest.approx(result.embedding.arm_scale)
    assert diagnostics.forward_expected_progress_fraction == pytest.approx(0.75)
    assert diagnostics.forward_scientific_fraction >= 0.75
    assert diagnostics.forward_transition_display_fraction == pytest.approx(1.0)


def test_branch_rose_excludes_incoming_root_direction_from_fate_labels():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_rose(
        result,
        population="root",
        mode="branch",
        n_bins=24,
        normalization="mass",
    )
    axis = figure.axes[0]
    assert len(axis.patches) == 24
    assert [tick.get_text() for tick in axis.get_xticklabels()] == list(result.fate_names)
    legend = axis.get_legend()
    assert legend is not None
    legend_text = [text.get_text() for text in legend.get_texts()]
    assert all(text.startswith(("A ", "B ")) for text in legend_text)
    assert not any("root" in text.lower() for text in legend_text)
    plt.close(figure)


def test_root_progression_identity_preserves_realistic_self_transition_mass():
    """Diagnostics must replay the same normalized operator as scoring.

    Real scVelo transition graphs commonly contain non-zero diagonal mass.
    Self transitions contribute zero displacement, but they remain in the row
    normalization and therefore scale all non-self displacements.  Removing
    them only in display/diagnostic code breaks the exact progression identity.
    """
    adata = make_adata()
    transition = make_transition().tolil()

    # Add realistic self-transition mass to every root row while preserving a
    # row total of one.  Terminal rows already contain endpoint self loops.
    transition[0, 0] = 0.50
    transition[0, 1] = 0.50
    transition[1, 1] = 0.40
    transition[1, 2] = 0.60
    transition[2, 2] = 0.30
    transition[2, 4] = 0.56
    transition[2, 7] = 0.14
    transition[3, 3] = 0.20
    transition[3, 5] = 0.72
    transition[3, 8] = 0.08

    scorer = SingleScorer(
        adata,
        root="root",
        branches=["A", "B"],
        obs_key="cluster",
    )
    scorer.build_embedding(
        ordering_metric="velocity_pseudotime",
        write_to_adata=True,
        verbose=False,
    )
    scorer.fit(
        transition_matrix=transition.tocsr(),
        min_transition_coverage=0.0,
        verbose=False,
    )
    result = scorer.score(write_to_adata=False, verbose=False)

    replay = scorer._selected_transition_weights(result)
    np.testing.assert_allclose(
        replay.diagonal()[:4],
        np.array([0.50, 0.40, 0.30, 0.20]),
        atol=0.0,
        rtol=0.0,
    )

    diagnostics = scorer.root_progression_direction_diagnostics(result)
    assert diagnostics.max_abs_progression_identity_error < 1e-12
    np.testing.assert_allclose(
        result.embedding.arm_scale * diagnostics.expected_progress_change,
        diagnostics.scientific_progression_velocity,
        atol=1e-12,
        rtol=0.0,
    )

    # The direct display field must replay the same transition operator.  The
    # root arm points left, so positive x remains centerward.
    star = scorer.prepare_scvelo_star_embedding(
        result,
        projection_mode="transition",
        write_to_adata=False,
    )
    root_velocity = np.asarray(star.obsm["velocity_sccs"])[result.root_mask]
    assert np.all(root_velocity[:, 0] >= -1e-12)


def test_commitment_heatmap_accepts_future_fate_metric_aliases():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_commitment_heatmap(
        result,
        metric="future_fate_affinity",
        population="all",
        max_cells=20,
    )
    assert figure.axes
    plt.close(figure)


def test_rose_accepts_terminal_population():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_rose(
        result,
        population="terminal",
        mode="fate_mass",
        normalization="mass",
    )
    assert len(figure.axes[0].patches) == result.k
    plt.close(figure)


def test_branch_velocity_profiles_keep_terminal_populations_separate():
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("scvelo")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorer = make_fitted_scorer()
    result = scorer.score(write_to_adata=False, verbose=False)
    figure = scorer.plot_branch_velocity_profiles(
        result,
        n_bins=16,
        normalization="within_branch",
        projection_mode="scvelo",
    )
    visible_axes = [axis for axis in figure.axes if axis.get_visible()]
    assert len(visible_axes) == result.k
    assert all(len(axis.patches) in {0, 16} for axis in visible_axes)
    assert all(any(name in axis.get_title() for name in result.fate_names) for axis in visible_axes)
    plt.close(figure)
