import numpy as np

from scCS.furcation import Furcation
from scCS.ordering import FurcationOrderingScaler


LABELS = np.array(["Root", "Root", "A", "A", "B", "B"])
ORDERING = np.array([0.0, 0.4, 0.6, 1.0, 0.7, 0.9])
FURCATION = Furcation(obs_key="clusters", root="Root", terminals=["A", "B"])


def test_ordering_maps_only_root_inward():
    scaler = FurcationOrderingScaler(
        root_lower_quantile=0.0,
        root_upper_quantile=1.0,
    )
    result = scaler.fit_transform(ORDERING, LABELS, FURCATION)
    root = result.root_progress[result.root_mask]
    assert root[0] < root[1]
    np.testing.assert_allclose(root, [0.0, 1.0])
    assert result.diagnostics.root_lower_bound == 0.0
    assert result.diagnostics.root_upper_bound == 0.4
    assert set(result.terminal_names[result.terminal_mask]) == {"A", "B"}


def test_terminal_ordering_values_do_not_affect_scientific_root_transform():
    first = FurcationOrderingScaler(
        root_lower_quantile=0.0,
        root_upper_quantile=1.0,
    ).fit_transform(ORDERING, LABELS, FURCATION)

    permuted = ORDERING.copy()
    permuted[2:] = permuted[[5, 4, 3, 2]]
    second = FurcationOrderingScaler(
        root_lower_quantile=0.0,
        root_upper_quantile=1.0,
    ).fit_transform(permuted, LABELS, FURCATION)

    np.testing.assert_allclose(first.root_progress, second.root_progress, atol=0, rtol=0)
    np.testing.assert_array_equal(first.terminal_names, second.terminal_names)


def test_overlapping_annotation_distributions_do_not_change_root_only_scaling():
    root = np.linspace(0.20, 0.80, 100)
    terminal_a = np.linspace(0.20, 0.80, 100)
    terminal_b = np.linspace(0.20, 0.80, 100)
    ordering = np.r_[root, terminal_a, terminal_b]
    labels = np.array(["Root"] * 100 + ["A"] * 100 + ["B"] * 100)

    result = FurcationOrderingScaler().fit_transform(
        ordering,
        labels,
        FURCATION,
    )
    root_progress = result.root_progress[result.root_mask]

    assert np.mean(root_progress == 1.0) <= 0.06
    assert np.std(root_progress) > 0.20


def test_constant_root_metric_uses_common_mid_arm_position():
    ordering = np.ones(len(LABELS))
    result = FurcationOrderingScaler().fit_transform(
        ordering,
        LABELS,
        FURCATION,
    )
    np.testing.assert_allclose(result.root_progress[result.root_mask], 0.5)


def test_inverted_orientation_is_explicit_and_deterministic():
    forward = FurcationOrderingScaler().fit_transform(
        ORDERING,
        LABELS,
        FURCATION,
    )
    reverse_1 = FurcationOrderingScaler(higher_is_later=False).fit_transform(
        ORDERING,
        LABELS,
        FURCATION,
    )
    reverse_2 = FurcationOrderingScaler(higher_is_later=False).fit_transform(
        ORDERING,
        LABELS,
        FURCATION,
    )
    np.testing.assert_allclose(reverse_1.root_progress, reverse_2.root_progress)
    assert not np.allclose(
        forward.root_progress[forward.root_mask],
        reverse_1.root_progress[reverse_1.root_mask],
    )


def test_ordering_diagnostics_report_ties_and_unique_fraction():
    labels = np.array(["Root"] * 10 + ["A"] * 2 + ["B"] * 2)
    ordering = np.array([0.0] * 5 + [1.0] * 5 + [0.8, 0.9, 0.8, 0.9])
    result = FurcationOrderingScaler(
        root_lower_quantile=0.0,
        root_upper_quantile=1.0,
    ).fit_transform(ordering, labels, FURCATION)

    diagnostics = result.diagnostics
    assert diagnostics.root_unique_values == 2
    assert diagnostics.root_unique_fraction == 0.2
    assert diagnostics.root_largest_tie_fraction == 0.5


def test_nonfinite_ordering_outside_furcation_is_ignored():
    labels = np.array(["Root", "Root", "A", "A", "B", "B", "Other"])
    ordering = np.array([0.0, 1.0, 0.8, 0.9, 0.8, 0.9, np.nan])
    result = FurcationOrderingScaler(
        root_lower_quantile=0.0,
        root_upper_quantile=1.0,
    ).fit_transform(ordering, labels, FURCATION)

    np.testing.assert_allclose(
        result.root_progress[result.root_mask],
        [0.0, 1.0],
    )


def test_nonfinite_ordering_inside_furcation_raises():
    labels = np.array(["Root", "Root", "A", "A", "B", "B"])
    ordering = np.array([0.0, np.nan, 0.8, 0.9, 0.8, 0.9])
    with np.testing.assert_raises_regex(
        ValueError,
        "selected furcation cells",
    ):
        FurcationOrderingScaler().fit_transform(ordering, labels, FURCATION)
