import numpy as np
import pytest

ad = pytest.importorskip("anndata")

from scCS.furcation import Furcation
from scCS.ordering import FurcationOrderingScaler
from scCS.scoring_embedding import build_scoring_embedding


def make_adata():
    import pandas as pd

    obs = pd.DataFrame(
        {
            "clusters": ["Root", "Root", "A", "A", "B", "B", "Other"],
            "pt": [0.0, 0.4, 0.6, 1.0, 0.7, 0.9, 0.5],
        },
        index=[f"c{i}" for i in range(7)],
    )
    return ad.AnnData(X=np.zeros((7, 1)), obs=obs)


def test_scoring_embedding_selects_only_manual_furcation_and_is_deterministic():
    adata = make_adata()
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    first = build_scoring_embedding(
        adata, furcation, ordering="pt", ordering_scaler=FurcationOrderingScaler()
    )
    second = build_scoring_embedding(
        adata,
        furcation,
        ordering="pt",
        ordering_scaler=FurcationOrderingScaler(),
    )
    assert first.n_selected == 6
    assert list(first.selected_cell_ids) == [f"c{i}" for i in range(6)]
    np.testing.assert_allclose(first.coordinates, second.coordinates, atol=0, rtol=0)


def test_root_and_terminal_coordinates_obey_simplex_axes_and_fixed_radius():
    adata = make_adata()
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    result = build_scoring_embedding(
        adata,
        furcation,
        ordering="pt",
        ordering_scaler=FurcationOrderingScaler(),
    )
    root_coords = result.coordinates[result.root_mask]
    terminal_coords = result.coordinates[result.terminal_mask]

    # Root is collinear with the incoming root axis.
    cross = root_coords @ result.geometry.terminal_directions.T
    np.testing.assert_allclose(cross, 0.0, atol=1e-12)

    # Every terminal cell occupies the equal-radius vertex for its fate.
    np.testing.assert_allclose(
        np.linalg.norm(terminal_coords, axis=1),
        result.arm_scale,
        atol=1e-12,
    )
    for coordinate, name in zip(terminal_coords, result.terminal_names[result.terminal_mask]):
        np.testing.assert_allclose(
            coordinate,
            result.arm_scale * result.geometry.direction_for(name),
            atol=1e-12,
        )


def test_terminal_pseudotime_permutation_does_not_change_scientific_coordinates():
    adata = make_adata()
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    first = build_scoring_embedding(adata, furcation, ordering="pt")

    adata_permuted = adata.copy()
    terminal = adata_permuted.obs["clusters"].isin(["A", "B"]).to_numpy()
    values = adata_permuted.obs["pt"].to_numpy().copy()
    values[terminal] = values[terminal][::-1]
    adata_permuted.obs["pt"] = values
    second = build_scoring_embedding(adata_permuted, furcation, ordering="pt")

    np.testing.assert_allclose(first.coordinates, second.coordinates, atol=0, rtol=0)


def test_unequal_terminal_population_sizes_do_not_change_vertex_radius():
    import pandas as pd

    labels = ["Root", "Root"] + ["A"] * 2 + ["B"] * 8
    obs = pd.DataFrame(
        {
            "clusters": labels,
            "pt": np.linspace(0.0, 1.0, len(labels)),
        },
        index=[f"u{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.zeros((len(labels), 1)), obs=obs)
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    result = build_scoring_embedding(adata, furcation, ordering="pt")
    radii = np.linalg.norm(result.coordinates[result.terminal_mask], axis=1)
    np.testing.assert_allclose(radii, result.arm_scale, atol=1e-12)


def test_write_to_adata_uses_full_length_nan_outside_furcation():
    adata = make_adata()
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    build_scoring_embedding(adata, furcation, ordering="pt", write_to_adata=True)
    assert adata.obsm["X_sccs_score"].shape == (7, 2)
    assert np.isnan(adata.obsm["X_sccs_score"][6]).all()
    assert adata.uns["sccs_v08"]["fate_names"] == ["A", "B"]
    assert adata.uns["sccs_v08"]["terminal_coordinate_mode"] == "fixed_vertex"


def test_scoring_embedding_ignores_nonfinite_ordering_outside_furcation():
    adata = make_adata()
    adata.obs.loc["c6", "pt"] = np.nan
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    result = build_scoring_embedding(adata, furcation, ordering="pt")
    assert result.n_selected == 6
    assert np.all(np.isfinite(result.coordinates))
