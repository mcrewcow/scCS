import numpy as np

from scCS.geometry import SimplexStarGeometry, regular_simplex_directions


def test_regular_simplex_gram_matrix():
    for k in range(2, 8):
        directions = regular_simplex_directions(k)
        gram = directions @ directions.T
        expected = np.full((k, k), -1.0 / (k - 1))
        np.fill_diagonal(expected, 1.0)
        np.testing.assert_allclose(gram, expected, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(directions.sum(axis=0), 0.0, atol=1e-12)


def test_root_axis_is_orthogonal_to_all_terminal_axes():
    geometry = SimplexStarGeometry(["Alpha", "Beta", "Delta", "Epsilon"])
    np.testing.assert_allclose(
        geometry.terminal_directions @ geometry.root_direction,
        0.0,
        atol=1e-12,
        rtol=0.0,
    )


def test_root_and_terminal_coordinates_have_expected_radius():
    geometry = SimplexStarGeometry(["A", "B", "C"])
    root_progress = np.array([0.0, 0.5, 1.0])
    root = geometry.root_coordinates(root_progress, arm_scale=10.0)
    np.testing.assert_allclose(np.linalg.norm(root, axis=1), [10.0, 5.0, 0.0])

    terminal = geometry.terminal_coordinates(
        ["A", "B", "C"],
        arm_scale=10.0,
    )
    np.testing.assert_allclose(np.linalg.norm(terminal, axis=1), [10.0, 10.0, 10.0], atol=1e-12)


def test_fate_reordering_is_equivariant_after_label_alignment():
    names_a = ["Alpha", "Beta", "Delta", "Epsilon"]
    names_b = ["Delta", "Alpha", "Epsilon", "Beta"]
    geom_a = SimplexStarGeometry(names_a)
    geom_b = SimplexStarGeometry(names_b)

    coefficients = {"Alpha": 0.8, "Beta": -0.1, "Delta": 0.3, "Epsilon": 0.2}
    velocity_a = sum(coefficients[name] * geom_a.direction_for(name) for name in names_a)
    velocity_b = sum(coefficients[name] * geom_b.direction_for(name) for name in names_b)

    similarities_a = {name: float(velocity_a @ geom_a.direction_for(name)) for name in names_a}
    similarities_b = {name: float(velocity_b @ geom_b.direction_for(name)) for name in names_a}
    for name in names_a:
        np.testing.assert_allclose(similarities_a[name], similarities_b[name], atol=1e-12, rtol=0.0)


def test_progression_and_branch_velocity_decompose_exactly():
    geometry = SimplexStarGeometry(["A", "B", "C", "D"])
    vectors = np.vstack(
        [
            2.0 * geometry.root_direction,
            3.0 * geometry.direction_for("B"),
            1.5 * geometry.root_direction + 0.7 * geometry.direction_for("D"),
        ]
    )
    progression, branch = geometry.decompose_velocity(vectors)
    np.testing.assert_allclose(progression, [2.0, 0.0, 1.5], atol=1e-12)
    np.testing.assert_allclose(branch[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(branch[1], 3.0 * geometry.direction_for("B"), atol=1e-12)
    np.testing.assert_allclose(branch[2], 0.7 * geometry.direction_for("D"), atol=1e-12)
