from __future__ import annotations

import warnings

import numpy as np
import pytest

from scCS.mixed_models import fit_mixedlm_fail_closed


class _FakeFit:
    def __init__(self, *, cov_re=1.0, converged=True):
        self.converged = converged
        self.fe_params = np.array([0.0, 1.0])
        self.cov_re = np.array([[cov_re]], dtype=float)

    def cov_params(self):
        return np.array([[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])


class _FakeModel:
    def __init__(self, *, warning_message=None, cov_re=1.0, converged=True):
        self.warning_message = warning_message
        self.cov_re = cov_re
        self.converged = converged

    def fit(self, **kwargs):
        del kwargs
        if self.warning_message:
            warnings.warn(self.warning_message, UserWarning)
        return _FakeFit(cov_re=self.cov_re, converged=self.converged)


def test_mixed_model_audit_accepts_finite_positive_definite_fit():
    audit = fit_mixedlm_fail_closed(_FakeModel(), n_fixed_effects=2)
    assert audit.valid
    assert audit.failure_reason is None
    assert audit.fixed_effect_covariance_min_eigenvalue > 0
    assert audit.random_effect_variance_min > 0


@pytest.mark.parametrize(
    "warning_message",
    [
        "Random effects covariance is singular",
        "The MLE may be on the boundary of the parameter space",
        "The Hessian matrix is not positive definite",
    ],
)
def test_mixed_model_audit_rejects_critical_warnings(warning_message):
    audit = fit_mixedlm_fail_closed(
        _FakeModel(warning_message=warning_message),
        n_fixed_effects=2,
    )
    assert not audit.valid
    assert audit.failure_reason is not None
    assert warning_message in " | ".join(audit.warning_messages)


def test_mixed_model_audit_rejects_boundary_random_effect_variance():
    audit = fit_mixedlm_fail_closed(_FakeModel(cov_re=0.0), n_fixed_effects=2)
    assert not audit.valid
    assert "random-effect variance" in audit.failure_reason


def test_mixed_model_audit_can_raise_on_invalid():
    with pytest.raises(RuntimeError, match="singular"):
        fit_mixedlm_fail_closed(
            _FakeModel(warning_message="Random effects covariance is singular"),
            n_fixed_effects=2,
            on_invalid="raise",
        )
