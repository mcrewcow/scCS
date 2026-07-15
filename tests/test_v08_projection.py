import numpy as np
from scipy import sparse

from scCS.projection import project_transition_velocity


def test_full_matrix_projection_ignores_external_coordinates_and_reports_mass():
    # Selected cells 0 and 1 have coordinates 0 and 1. Cell 2 is outside.
    transition = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coordinates = np.array([[0.0], [1.0]])
    result = project_transition_velocity(
        transition,
        coordinates,
        selected_indices=[0, 1],
        renormalize_retained=True,
        min_coverage=0.0,
    )

    # Cell 0's retained transition points fully to coordinate 1 after
    # renormalization. The excluded cell never receives an artificial origin.
    np.testing.assert_allclose(result.velocity[:, 0], [1.0, -0.5], atol=1e-12)
    np.testing.assert_allclose(result.retained_transition_mass, [0.8, 1.0])
    np.testing.assert_allclose(result.external_transition_mass, [0.2, 0.0])
    np.testing.assert_allclose(result.transition_coverage, [0.8, 1.0])
    assert result.velocity_defined.all()


def test_low_coverage_cell_is_explicitly_undefined():
    transition = np.array(
        [
            [0.0, 0.01, 0.99],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coordinates = np.array([[0.0], [1.0]])
    result = project_transition_velocity(
        transition,
        coordinates,
        selected_indices=[0, 1],
        min_coverage=0.05,
    )
    assert not result.velocity_defined[0]
    assert np.isnan(result.velocity[0]).all()
    assert result.transition_coverage[0] == 0.01
    assert result.velocity_defined[1]


def test_no_outgoing_mass_is_undefined_not_zero_velocity():
    transition = np.zeros((2, 2))
    coordinates = np.array([[0.0], [1.0]])
    result = project_transition_velocity(
        transition,
        coordinates,
        min_coverage=0.0,
    )
    assert not result.velocity_defined.any()
    assert np.isnan(result.velocity).all()


def test_sparse_and_dense_projection_are_equivalent():
    transition = np.array(
        [
            [0.2, 0.8, 0.0],
            [0.1, 0.2, 0.7],
            [0.0, 0.4, 0.6],
        ]
    )
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dense = project_transition_velocity(transition, coordinates, min_coverage=0.0)
    sparse_result = project_transition_velocity(
        sparse.csr_matrix(transition), coordinates, min_coverage=0.0
    )
    np.testing.assert_allclose(dense.velocity, sparse_result.velocity, atol=1e-12)
    np.testing.assert_allclose(
        dense.transition_coverage,
        sparse_result.transition_coverage,
        atol=1e-12,
    )


def test_nonrenormalized_projection_preserves_retained_mass_scale():
    transition = np.array(
        [
            [0.0, 0.4, 0.6],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coordinates = np.array([[0.0], [1.0]])
    result = project_transition_velocity(
        transition,
        coordinates,
        selected_indices=[0, 1],
        renormalize_retained=False,
        min_coverage=0.0,
    )
    assert result.velocity[0, 0] == 0.4
