import numpy as np
import pytest

from scCS.furcation import Furcation


def test_simple_manual_furcation_validation():
    furcation = Furcation(
        obs_key="clusters",
        root="Pre-endocrine",
        terminals=["Alpha", "Beta", "Delta", "Epsilon"],
    )
    labels = np.array(
        ["Pre-endocrine"] * 5
        + ["Alpha"] * 3
        + ["Beta"] * 4
        + ["Delta"] * 2
        + ["Epsilon"] * 2
        + ["Ductal"] * 7
    )
    report = furcation.validate_labels(labels)
    assert report.root_count == 5
    assert report.terminal_counts == {
        "Alpha": 3,
        "Beta": 4,
        "Delta": 2,
        "Epsilon": 2,
    }
    assert report.selected_count == 16
    assert report.selected_mask.dtype == bool
    assert furcation.k == 4


def test_grouped_labels_are_only_population_grouping():
    furcation = Furcation(
        obs_key="clusters",
        root=["Pre-endocrine_1", "Pre-endocrine_2"],
        terminals={
            "Alpha": ["Alpha_1", "Alpha_2"],
            "Beta": "Beta",
        },
        min_cells=1,
    )
    labels = np.array(["Pre-endocrine_1", "Pre-endocrine_2", "Alpha_1", "Alpha_2", "Beta"])
    report = furcation.validate_labels(labels)
    assert report.root_count == 2
    assert report.terminal_counts == {"Alpha": 2, "Beta": 1}
    assert furcation.terminal_names == ("Alpha", "Beta")


def test_root_terminal_overlap_is_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        Furcation(
            obs_key="clusters",
            root="Pre-endocrine",
            terminals={
                "Alpha": ["Pre-endocrine", "Alpha"],
                "Beta": "Beta",
            },
        )


def test_missing_terminal_fails_instead_of_being_skipped():
    furcation = Furcation(
        obs_key="clusters",
        root="Root",
        terminals=["A", "B"],
        min_cells=1,
    )
    with pytest.raises(ValueError, match="Terminal population 'B'"):
        furcation.validate_labels(np.array(["Root", "A"]))
