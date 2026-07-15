import numpy as np

from scCS.population import summarize_commitment


def test_duplicate_cells_double_total_but_preserve_mean_and_composition():
    contributions = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    original = summarize_commitment(contributions)
    duplicated = summarize_commitment(np.vstack([contributions, contributions]))

    np.testing.assert_allclose(
        duplicated.total_commitment_mass,
        2.0 * original.total_commitment_mass,
    )
    np.testing.assert_allclose(
        duplicated.mean_commitment_contribution, original.mean_commitment_contribution
    )
    np.testing.assert_allclose(
        duplicated.commitment_composition,
        original.commitment_composition,
    )


def test_pairwise_log_ratios_are_antisymmetric():
    summary = summarize_commitment(
        np.array(
            [
                [0.7, 0.2, 0.1],
                [0.5, 0.4, 0.1],
            ]
        )
    )
    matrix = summary.pairwise_log_commitment_ratio
    np.testing.assert_allclose(matrix, -matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)


def test_no_commitment_has_undefined_composition_not_fake_balance():
    summary = summarize_commitment(np.zeros((10, 4)))
    assert summary.total_mass == 0.0
    assert not summary.composition_defined
    assert np.isnan(summary.commitment_composition).all()
    assert np.isnan(summary.population_balance_entropy)
