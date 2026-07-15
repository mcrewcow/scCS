import numpy as np
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")

from scCS.affinity import MagnitudeScaler
from scCS.furcation import Furcation
from scCS.pipeline import score_furcation


def make_data():
    import pandas as pd

    obs = pd.DataFrame(
        {
            "clusters": ["Root", "Root", "A", "A", "B", "B"],
            "pt": [0.0, 0.4, 0.6, 1.0, 0.7, 0.9],
        },
        index=["r0", "r1", "a0", "a1", "b0", "b1"],
    )
    adata = ad.AnnData(X=np.zeros((6, 1)), obs=obs)
    transition = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.8, 0.0, 0.2],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    return adata, transition


def test_integrated_pipeline_recovers_preferential_root_to_a_transition():
    adata, transition = make_data()
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    result = score_furcation(
        adata,
        furcation,
        ordering="pt",
        transition_matrix=transition,
        min_transition_coverage=0.0,
    )
    r1 = list(result.cell_ids).index("r1")
    a_index = result.fate_names.index("A")
    b_index = result.fate_names.index("B")
    assert (
        result.commitment.directional_affinity[r1, a_index]
        > result.commitment.directional_affinity[r1, b_index]
    )
    assert result.progression_velocity[0] > 0
    assert result.root_population_summary.n_cells == 2
    assert isinstance(result.magnitude_scaler, MagnitudeScaler)
    assert result.magnitude_scaler.scale_quantile == 0.75


def test_pipeline_write_to_adata_records_final_v08_model():
    adata, transition = make_data()
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    score_furcation(
        adata,
        furcation,
        ordering="pt",
        transition_matrix=transition,
        min_transition_coverage=0.0,
        write_to_adata=True,
    )
    assert adata.obsm["sccs_directional_affinity"].shape == (6, 2)
    assert "sccs_commitment_strength" in adata.obs
    assert "sccs_commitment_entropy" in adata.obs
    assert "sccs_nearest_fate_angle_degrees" in adata.obs
    assert adata.uns["sccs"]["schema_version"] == "0.8"
    assert adata.uns["sccs"]["aligned_probability"] == 0.90
    assert adata.uns["sccs"]["entropy"]["aligned_directional_entropy_floor"] > 0


def test_external_only_transition_is_low_coverage_not_no_outgoing():
    import pandas as pd

    obs = pd.DataFrame(
        {
            "clusters": ["Root", "Root", "A", "B", "Other"],
            "pt": [0.0, 0.4, 0.8, 0.9, 0.5],
        },
        index=["r0", "r1", "a0", "b0", "x0"],
    )
    adata = ad.AnnData(X=np.zeros((5, 1)), obs=obs)
    transition = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
    )
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"], min_cells=1)
    result = score_furcation(
        adata,
        furcation,
        ordering="pt",
        transition_matrix=transition,
        min_transition_coverage=0.05,
    )
    r1 = list(result.cell_ids).index("r1")
    assert result.status[r1] == "low_transition_coverage"


def test_progression_only_transitions_create_no_fate_commitment():
    import pandas as pd

    obs = pd.DataFrame(
        {
            "clusters": ["Root", "Root", "A", "A", "B", "B"],
            "pt": [0.0, 0.4, 0.6, 1.0, 0.6, 1.0],
        },
        index=["r0", "r1", "a0", "a1", "b0", "b1"],
    )
    adata = ad.AnnData(X=np.zeros((6, 1)), obs=obs)
    # r0 -> r1 only.  No root transition points into a terminal arm.
    transition = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
    )
    furcation = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])
    result = score_furcation(
        adata,
        furcation,
        ordering="pt",
        transition_matrix=transition,
        min_transition_coverage=0.0,
    )
    np.testing.assert_allclose(
        result.commitment.commitment_strength[result.root_mask], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.commitment.specific_commitment[result.root_mask], 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.commitment.directional_affinity[result.root_mask], 0.5, atol=1e-12
    )
