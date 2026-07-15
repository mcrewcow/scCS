from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "v08" / "pancreas_pseudoconditions.py"
spec = importlib.util.spec_from_file_location("pancreas_pseudoconditions", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def make_adata():
    labels = (
        ["Pre-endocrine"] * 120
        + ["Alpha"] * 40
        + ["Beta"] * 50
        + ["Delta"] * 24
        + ["Epsilon"] * 24
        + ["Ductal"] * 30
    )
    obs = pd.DataFrame(
        {
            "clusters": labels,
            "velocity_pseudotime": np.linspace(0.0, 1.0, len(labels)),
        },
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    return anndata.AnnData(X=np.zeros((len(labels), 2)), obs=obs)


def test_stratified_pseudoreplicates_assign_every_cell_and_balance_strata():
    adata = make_adata()
    assignments = module.assign_stratified_pseudoreplicates(
        adata,
        annotation_key="clusters",
        root="Pre-endocrine",
        terminals=["Alpha", "Beta", "Delta", "Epsilon"],
        pseudotime_key="velocity_pseudotime",
        conditions=["control", "treated"],
        replicates_per_condition=4,
        random_state=2,
    )
    assert len(assignments) == adata.n_obs
    assert not assignments.isna().any().any()
    assert set(adata.obs["sccs_condition"].astype(str)) == {"control", "treated"}
    grouped = assignments.groupby(["stratum", "sccs_condition", "sccs_replicate"]).size()
    for _, values in grouped.groupby(level=0):
        assert values.max() - values.min() <= 1


def test_directional_injection_preserves_nonroot_vectors():
    velocity = np.array(
        [
            [-0.2, 0.3, 0.0],
            [-0.1, 0.1, 0.2],
            [0.0, 1.0, 0.0],
        ]
    )
    output = module.inject_root_directional_effect(
        velocity,
        root_mask=np.array([True, True, False]),
        condition_labels=["control", "treated", "treated"],
        replicate_labels=["R1", "R1", "R1"],
        root_direction=np.array([1.0, 0.0, 0.0]),
        target_direction=np.array([0.0, 0.0, 1.0]),
        effect_by_condition={"control": 0.0, "treated": 1.0},
        replicate_effect_sd=0.0,
        replicate_magnitude_log_sd=0.0,
        random_state=0,
    )
    np.testing.assert_allclose(output[2], velocity[2])
    assert output[1, 2] > output[1, 1]


def test_directional_injection_uses_angular_fraction_and_preserves_magnitude():
    root_direction = np.array([1.0, 0.0, 0.0])
    source = np.array([[0.2, 1.0, 0.0]])
    target = np.array([0.0, 0.0, 1.0])
    output = module.inject_root_directional_effect(
        source,
        root_mask=np.array([True]),
        condition_labels=["treated"],
        replicate_labels=["R1"],
        root_direction=root_direction,
        target_direction=target,
        effect_by_condition={"treated": 0.5},
        random_state=0,
    )
    progression_before = source @ root_direction
    progression_after = output @ root_direction
    np.testing.assert_allclose(progression_after, progression_before)
    branch_before = source - progression_before[:, None] * root_direction
    branch_after = output - progression_after[:, None] * root_direction
    np.testing.assert_allclose(
        np.linalg.norm(branch_after, axis=1),
        np.linalg.norm(branch_before, axis=1),
    )
    source_angle = np.arccos(
        np.clip(
            (branch_before[0] / np.linalg.norm(branch_before[0])) @ target,
            -1.0,
            1.0,
        )
    )
    output_angle = np.arccos(
        np.clip(
            (branch_after[0] / np.linalg.norm(branch_after[0])) @ target,
            -1.0,
            1.0,
        )
    )
    np.testing.assert_allclose(output_angle, 0.5 * source_angle, atol=1e-10)


def test_magnitude_injection_preserves_progression_and_branch_direction():
    velocity = np.array(
        [
            [0.2, 0.3, 0.4],
            [-0.1, 0.2, -0.3],
            [0.0, 1.0, 0.0],
        ]
    )
    root_direction = np.array([1.0, 0.0, 0.0])
    output = module.inject_root_magnitude_effect(
        velocity,
        root_mask=np.array([True, True, False]),
        condition_labels=["control", "treated", "treated"],
        replicate_labels=["R1", "R1", "R1"],
        root_direction=root_direction,
        factor_by_condition={"control": 1.0, "treated": 2.0},
        random_state=0,
    )
    np.testing.assert_allclose(output[2], velocity[2])
    progression_before = velocity[:2] @ root_direction
    progression_after = output[:2] @ root_direction
    np.testing.assert_allclose(progression_after, progression_before)
    branch_before = velocity[:2] - progression_before[:, None] * root_direction
    branch_after = output[:2] - progression_after[:, None] * root_direction
    np.testing.assert_allclose(branch_after[0], branch_before[0])
    np.testing.assert_allclose(branch_after[1], 2.0 * branch_before[1])


def test_combined_injection_changes_only_root_cells():
    velocity = np.array(
        [
            [0.1, 0.4, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    output = module.inject_root_combined_effect(
        velocity,
        root_mask=np.array([True, False]),
        condition_labels=["treated", "treated"],
        replicate_labels=["R1", "R1"],
        root_direction=np.array([1.0, 0.0, 0.0]),
        target_direction=np.array([0.0, 0.0, 1.0]),
        directional_effect_by_condition={"treated": 1.0},
        magnitude_factor_by_condition={"treated": 1.5},
        random_state=0,
    )
    np.testing.assert_allclose(output[1], velocity[1])
    assert output[0, 2] > 0.0
