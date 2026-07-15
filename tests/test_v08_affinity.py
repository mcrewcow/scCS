import numpy as np
import pytest

from scCS.affinity import (
    MagnitudeScaler,
    aligned_directional_entropy,
    calibrated_softmax_beta,
    combine_direction_and_strength,
    cosine_softmax_affinity,
    normalized_entropy,
    support_adjusted_directional_specificity,
)
from scCS.geometry import SimplexStarGeometry


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_calibrated_softmax_hits_requested_aligned_probability(k):
    geometry = SimplexStarGeometry([f"F{i}" for i in range(k)])
    scores = cosine_softmax_affinity(
        geometry.direction_for("F0")[None, :],
        geometry.terminal_directions,
        aligned_probability=0.90,
    )
    assert scores[0, 0] == pytest.approx(0.90, abs=1e-12)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-12)


def test_four_fate_aligned_entropy_is_well_below_legacy_floor():
    geometry = SimplexStarGeometry(["Alpha", "Beta", "Delta", "Epsilon"])
    scores = cosine_softmax_affinity(
        geometry.direction_for("Beta")[None, :],
        geometry.terminal_directions,
        aligned_probability=0.90,
    )
    entropy = normalized_entropy(scores)[0]
    assert entropy == pytest.approx(0.313745, abs=1e-5)
    assert entropy < 0.75


def test_zero_branch_velocity_is_uniform_and_explicitly_undefined():
    geometry = SimplexStarGeometry(["A", "B", "C", "D"])
    zero = np.zeros((3, geometry.dimension))
    q = cosine_softmax_affinity(zero, geometry.terminal_directions)
    np.testing.assert_allclose(q, 0.25)
    combined = combine_direction_and_strength(q, zero, np.zeros(3))
    assert not combined.velocity_defined.any()
    np.testing.assert_allclose(combined.commitment_affinity, 0.25)
    np.testing.assert_allclose(combined.specific_commitment, 0.0)


def test_alpha_to_beta_interpolation_is_smooth_and_monotonic():
    geometry = SimplexStarGeometry(["Alpha", "Beta", "Delta", "Epsilon"])
    alpha = geometry.direction_for("Alpha")
    beta = geometry.direction_for("Beta")
    levels = np.linspace(0.0, 1.0, 101)
    vectors = np.array([(1.0 - t) * alpha + t * beta for t in levels])
    q = cosine_softmax_affinity(
        vectors,
        geometry.terminal_directions,
        aligned_probability=0.90,
    )
    beta_scores = q[:, geometry.fate_names.index("Beta")]
    alpha_scores = q[:, geometry.fate_names.index("Alpha")]
    assert np.all(np.diff(beta_scores) >= -1e-12)
    assert np.all(np.diff(alpha_scores) <= 1e-12)
    crossing = np.argmin(np.abs(beta_scores - alpha_scores))
    assert abs(levels[crossing] - 0.5) <= 0.01


def test_beta_formula_is_positive_and_finite():
    beta = calibrated_softmax_beta(4, 0.90)
    assert np.isfinite(beta)
    assert beta > 0


def test_support_adjusted_specificity_maps_uniform_to_zero_and_alignment_to_one():
    geometry = SimplexStarGeometry(["A", "B", "C"])
    aligned = cosine_softmax_affinity(
        geometry.direction_for("A")[None, :],
        geometry.terminal_directions,
        aligned_probability=0.90,
    )
    aligned_entropy = normalized_entropy(aligned)
    specificity = support_adjusted_directional_specificity(
        aligned_entropy,
        k=3,
        aligned_probability=0.90,
    )
    assert aligned_directional_entropy(3, 0.90) == pytest.approx(aligned_entropy[0])
    assert specificity[0] == pytest.approx(1.0, abs=1e-12)

    uniform_entropy = normalized_entropy(np.full((1, 3), 1 / 3))
    uniform_specificity = support_adjusted_directional_specificity(
        uniform_entropy,
        k=3,
        aligned_probability=0.90,
    )
    assert uniform_specificity[0] == pytest.approx(0.0, abs=1e-12)


def test_two_fate_directional_entropy_has_no_continuous_angular_degree_of_freedom():
    geometry = SimplexStarGeometry(["A", "B"])
    axis = geometry.direction_for("A")
    vectors = np.vstack([0.1 * axis, axis, 10.0 * axis, -0.2 * axis, -3.0 * axis])
    affinity = cosine_softmax_affinity(vectors, geometry.terminal_directions)
    entropy = normalized_entropy(affinity)
    np.testing.assert_allclose(entropy, aligned_directional_entropy(2, 0.90), atol=1e-12)


def test_three_fate_equal_nearest_angles_have_equal_entropy_up_to_axis_permutation():
    geometry = SimplexStarGeometry(["A", "B", "C"])
    a = geometry.direction_for("A")
    # Reflections around fate A exchange B and C while preserving the angle to A.
    toward_b = a + 0.35 * geometry.direction_for("B")
    toward_c = a + 0.35 * geometry.direction_for("C")
    affinity = cosine_softmax_affinity(
        np.vstack([toward_b, toward_c]), geometry.terminal_directions
    )
    entropy = normalized_entropy(affinity)
    assert entropy[0] == pytest.approx(entropy[1], abs=1e-12)
    assert affinity[0, 0] == pytest.approx(affinity[1, 0], abs=1e-12)
    assert affinity[0, 1] == pytest.approx(affinity[1, 2], abs=1e-12)
    assert affinity[0, 2] == pytest.approx(affinity[1, 1], abs=1e-12)


def test_three_fate_entropy_is_monotone_in_nearest_axis_angle():
    geometry = SimplexStarGeometry(["A", "B", "C"])
    a = geometry.direction_for("A")
    b = geometry.direction_for("B")
    tangent = b - float(b @ a) * a
    tangent /= np.linalg.norm(tangent)
    theta = np.linspace(0.0, np.pi / 3.0, 121)
    vectors = np.cos(theta)[:, None] * a + np.sin(theta)[:, None] * tangent
    affinity = cosine_softmax_affinity(vectors, geometry.terminal_directions)
    entropy = normalized_entropy(affinity)
    assert np.all(np.diff(entropy) >= -1e-12)
    assert entropy[0] == pytest.approx(aligned_directional_entropy(3, 0.90), abs=1e-12)


def test_entropy_is_derived_from_cosine_softmax_angular_profile():
    geometry = SimplexStarGeometry(["A", "B", "C"])
    a = geometry.direction_for("A")
    b = geometry.direction_for("B")
    vectors = np.vstack([a, a + b, np.zeros_like(a)])
    affinity = cosine_softmax_affinity(vectors, geometry.terminal_directions)
    entropy = normalized_entropy(affinity)
    assert entropy[0] < entropy[1] < entropy[2]
    combined = combine_direction_and_strength(
        affinity,
        vectors,
        np.array([1.0, 1.0, 0.0]),
        aligned_probability=0.90,
    )
    assert combined.directional_specificity[0] == pytest.approx(1.0, abs=1e-12)
    assert combined.directional_specificity[2] == pytest.approx(0.0, abs=1e-12)


def test_default_magnitude_scaler_is_zero_preserving_smooth_and_saturating():
    scaler = MagnitudeScaler().fit(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    transformed = scaler.transform(np.array([0.0, 1.0, 3.0, 1_000.0]))
    assert transformed[0] == 0.0
    assert np.all(np.diff(transformed) > 0)
    assert transformed[-1] < 1.0
    assert scaler.scale_quantile == 0.75


def test_no_signal_reference_maps_all_strengths_to_zero():
    scaler = MagnitudeScaler().fit(np.zeros(5))
    np.testing.assert_allclose(scaler.transform(np.array([0.0, 1.0])), 0.0)
