"""Fail-closed auditing for optional statsmodels mixed-effects models.

Replicate permutation and hierarchical bootstrap are the primary scCS
inference methods. Mixed models are optional sensitivity analyses and must not
return apparently valid p-values when optimization, covariance estimation, or
the random-effects structure is singular.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
import warnings

import numpy as np


_CRITICAL_WARNING_FRAGMENTS = (
    "singular",
    "boundary",
    "hessian",
    "not positive definite",
    "failed to converge",
    "optimization failed",
    "gradient optimization failed",
)


@dataclass(frozen=True)
class MixedModelAudit:
    """Audited result of one statsmodels ``MixedLM`` fit."""

    fit: Optional[Any]
    valid: bool
    failure_reason: Optional[str]
    warning_messages: tuple[str, ...]
    fixed_effect_covariance: Optional[np.ndarray]
    fixed_effect_covariance_min_eigenvalue: float
    fixed_effect_covariance_condition_number: float
    random_effect_variance_min: float


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _invalid_audit(
    *,
    reason: str,
    warning_messages: Sequence[str] = (),
    fit: Optional[Any] = None,
    fixed_covariance: Optional[np.ndarray] = None,
    min_eigenvalue: float = float("nan"),
    condition_number: float = float("nan"),
    random_variance_min: float = float("nan"),
) -> MixedModelAudit:
    return MixedModelAudit(
        fit=fit,
        valid=False,
        failure_reason=str(reason),
        warning_messages=tuple(str(value) for value in warning_messages),
        fixed_effect_covariance=fixed_covariance,
        fixed_effect_covariance_min_eigenvalue=float(min_eigenvalue),
        fixed_effect_covariance_condition_number=float(condition_number),
        random_effect_variance_min=float(random_variance_min),
    )


def fit_mixedlm_fail_closed(
    model,
    *,
    n_fixed_effects: int,
    on_invalid: str = "return",
    methods: Sequence[str] = ("lbfgs", "powell", "cg"),
    covariance_rtol: float = 1e-10,
    condition_number_limit: float = 1e12,
) -> MixedModelAudit:
    """Fit and audit a ``statsmodels.MixedLM`` result.

    Parameters
    ----------
    model
        Constructed statsmodels ``MixedLM`` object.
    n_fixed_effects
        Number of fixed-effect coefficients, including the intercept.
    on_invalid
        ``"return"`` returns an invalid audit with no inferential estimates.
        ``"raise"`` raises ``RuntimeError`` with the audit reason.

    Notes
    -----
    The audit rejects convergence/covariance warnings, non-finite estimates,
    indefinite or numerically singular fixed-effect covariance, and boundary
    random-effect variances. This is deliberately conservative because mixed
    models are only a sensitivity analysis in scCS.
    """
    if on_invalid not in {"return", "raise"}:
        raise ValueError("on_invalid must be 'return' or 'raise'.")
    if not isinstance(n_fixed_effects, int) or n_fixed_effects < 2:
        raise ValueError("n_fixed_effects must be an integer of at least 2.")
    if covariance_rtol <= 0 or condition_number_limit <= 1:
        raise ValueError("Covariance tolerances must be positive.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = model.fit(reml=False, method=list(methods), disp=False)
        except Exception as exc:  # statsmodels may raise many numerical exceptions
            audit = _invalid_audit(
                reason=f"Mixed-model optimization failed: {_format_exception(exc)}",
                warning_messages=[str(item.message) for item in caught],
            )
            if on_invalid == "raise":
                raise RuntimeError(audit.failure_reason) from exc
            return audit

    warning_messages = tuple(str(item.message) for item in caught)
    critical_warnings = [
        message
        for message in warning_messages
        if any(fragment in message.lower() for fragment in _CRITICAL_WARNING_FRAGMENTS)
    ]

    reasons: list[str] = []
    if not bool(getattr(fit, "converged", False)):
        reasons.append("optimizer did not report convergence")
    if critical_warnings:
        reasons.append("critical statsmodels warning(s): " + " | ".join(critical_warnings))

    fixed_params = np.asarray(getattr(fit, "fe_params", []), dtype=float)
    if fixed_params.shape != (n_fixed_effects,) or not np.all(np.isfinite(fixed_params)):
        reasons.append("fixed-effect estimates are missing or non-finite")

    fixed_covariance: Optional[np.ndarray] = None
    min_eigenvalue = float("nan")
    condition_number = float("nan")
    try:
        covariance = np.asarray(fit.cov_params(), dtype=float)
        fixed_covariance = covariance[:n_fixed_effects, :n_fixed_effects]
        if fixed_covariance.shape != (n_fixed_effects, n_fixed_effects):
            reasons.append("fixed-effect covariance has the wrong shape")
        elif not np.all(np.isfinite(fixed_covariance)):
            reasons.append("fixed-effect covariance contains non-finite values")
        else:
            symmetric = 0.5 * (fixed_covariance + fixed_covariance.T)
            eigenvalues = np.linalg.eigvalsh(symmetric)
            min_eigenvalue = float(eigenvalues.min())
            max_eigenvalue = float(eigenvalues.max())
            scale = max(abs(max_eigenvalue), 1.0)
            if min_eigenvalue <= covariance_rtol * scale:
                reasons.append("fixed-effect covariance is singular or not positive definite")
            else:
                condition_number = float(max_eigenvalue / min_eigenvalue)
                if not np.isfinite(condition_number) or condition_number > condition_number_limit:
                    reasons.append("fixed-effect covariance is numerically ill-conditioned")
            fixed_covariance = symmetric
    except Exception as exc:
        reasons.append(f"could not evaluate fixed-effect covariance: {_format_exception(exc)}")

    random_variance_min = float("nan")
    try:
        cov_re = np.asarray(getattr(fit, "cov_re"), dtype=float)
        if cov_re.ndim != 2 or cov_re.shape[0] != cov_re.shape[1] or cov_re.size == 0:
            reasons.append("random-effect covariance is unavailable")
        elif not np.all(np.isfinite(cov_re)):
            reasons.append("random-effect covariance contains non-finite values")
        else:
            re_eigenvalues = np.linalg.eigvalsh(0.5 * (cov_re + cov_re.T))
            random_variance_min = float(re_eigenvalues.min())
            re_scale = max(float(np.max(np.abs(re_eigenvalues))), 1.0)
            if random_variance_min <= covariance_rtol * re_scale:
                reasons.append("random-effect variance is on the boundary or singular")
    except Exception as exc:
        reasons.append(f"could not evaluate random-effect covariance: {_format_exception(exc)}")

    if reasons:
        audit = _invalid_audit(
            reason="; ".join(dict.fromkeys(reasons)),
            warning_messages=warning_messages,
            fit=fit,
            fixed_covariance=fixed_covariance,
            min_eigenvalue=min_eigenvalue,
            condition_number=condition_number,
            random_variance_min=random_variance_min,
        )
        if on_invalid == "raise":
            raise RuntimeError(audit.failure_reason)
        return audit

    return MixedModelAudit(
        fit=fit,
        valid=True,
        failure_reason=None,
        warning_messages=warning_messages,
        fixed_effect_covariance=fixed_covariance,
        fixed_effect_covariance_min_eigenvalue=min_eigenvalue,
        fixed_effect_covariance_condition_number=condition_number,
        random_effect_variance_min=random_variance_min,
    )


__all__ = ["MixedModelAudit", "fit_mixedlm_fail_closed"]
