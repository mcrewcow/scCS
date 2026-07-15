"""Replicate-aware pairwise condition comparison for scCS v0.8."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .condition import (
    ConditionCommitmentResult,
    ConditionScorer,
    _canonical_metric,
    _metric_label,
)
from .inference import holm_adjust, two_group_permutation_test
from .mixed_models import fit_mixedlm_fail_closed


class PairScorer(ConditionScorer):
    """Compare commitment between exactly two conditions.

    One shared :class:`~scCS.SingleScorer` is fit on the pooled data.  Formal
    inference is performed on biological-replicate summaries supplied through
    ``replicate_obs_key`` (or its alias ``replicate_key``).
    """

    minimum_conditions = 2
    maximum_conditions = 2

    def compare_conditions(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        condition_a: Optional[str] = None,
        condition_b: Optional[str] = None,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
        n_permutations: int = 9999,
        max_exact: int = 100_000,
        n_bootstrap: int = 0,
        confidence_level: float = 0.95,
        resample_cells_within_replicate: bool = True,
        random_state: int = 0,
        adjust_pvalues: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Compare replicate-level commitment outcomes between two conditions.

        The reported effect is ``mean(condition_b) - mean(condition_a)``.
        Exact label permutation is used whenever the number of allocations is
        small enough; otherwise a corrected Monte Carlo p-value is used.
        """
        chosen = self._resolve_results(results)
        metric_public = str(metric)
        metric = _canonical_metric(metric)
        condition_a = self.conditions[0] if condition_a is None else str(condition_a)
        condition_b = self.conditions[1] if condition_b is None else str(condition_b)
        if condition_a == condition_b:
            raise ValueError("condition_a and condition_b must differ.")
        if condition_a not in chosen or condition_b not in chosen:
            raise ValueError("Both requested conditions must have scored results.")
        if self.replicate_obs_key is None:
            raise ValueError(
                "compare_conditions requires replicate_obs_key. Cell-level "
                "permutation is intentionally not used for formal inference."
            )

        table = self.replicate_table(chosen)
        rows = []
        specs = self._outcome_specs(metric, fate=fate, fate_pair=fate_pair)
        for offset, (fate_name, pair, column) in enumerate(specs):
            subset = table.loc[
                table["condition"].isin([condition_a, condition_b]),
                ["condition", column],
            ].dropna()
            test = two_group_permutation_test(
                subset[column].to_numpy(dtype=float),
                subset["condition"].to_numpy(str),
                condition_a,
                condition_b,
                n_permutations=n_permutations,
                max_exact=max_exact,
                random_state=random_state + offset,
            )
            values_a = subset.loc[subset["condition"] == condition_a, column]
            values_b = subset.loc[subset["condition"] == condition_b, column]
            rows.append(
                {
                    "metric": metric,
                    "metric_public": metric_public,
                    "metric_label": _metric_label(metric_public, fate=fate_name),
                    "fate": fate_name,
                    "fate_a": None if pair is None else pair[0],
                    "fate_b": None if pair is None else pair[1],
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "mean_a": float(values_a.mean()),
                    "mean_b": float(values_b.mean()),
                    "effect_b_minus_a": test.effect,
                    "pvalue": test.pvalue,
                    "permutation_method": test.method,
                    "n_permutations": test.n_permutations,
                    "n_replicates_a": test.n_group_a,
                    "n_replicates_b": test.n_group_b,
                }
            )

        output = pd.DataFrame(rows)
        output["pvalue_adj"] = (
            holm_adjust(output["pvalue"].to_numpy())
            if adjust_pvalues and len(output) > 1
            else output["pvalue"].to_numpy()
        )

        if n_bootstrap > 0:
            intervals = self.hierarchical_bootstrap(
                condition_a=condition_a,
                condition_b=condition_b,
                metric=metric_public,
                fate=fate,
                fate_pair=fate_pair,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                resample_cells_within_replicate=resample_cells_within_replicate,
                random_state=random_state + 100_000,
                results=chosen,
            )
            merge_keys = [
                "metric",
                "metric_public",
                "metric_label",
                "fate",
                "fate_a",
                "fate_b",
                "condition_a",
                "condition_b",
            ]
            output = output.merge(
                intervals[
                    merge_keys
                    + [
                        "ci_lower",
                        "ci_upper",
                        "confidence_level",
                        "n_bootstrap",
                        "resample_cells_within_replicate",
                    ]
                ],
                on=merge_keys,
                how="left",
                validate="one_to_one",
            )

        if verbose:
            print(
                f"[scCS] Pairwise replicate inference: {condition_b!r} - "
                f"{condition_a!r}; metric={metric!r}."
            )
        return output

    def compute_delta_CS(
        self,
        condition_a: Optional[str] = None,
        condition_b: Optional[str] = None,
        *,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        n_permutations: int = 9999,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        random_state: int = 0,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Return fate-specific changes in mean commitment contribution.

        This preserves the familiar function name while replacing the old
        angular-sector nCS difference.  ``delta_CS`` is now a finite,
        replicate-aware difference in mean soft commitment contribution.
        """
        result = self.compare_conditions(
            results=results,
            condition_a=condition_a,
            condition_b=condition_b,
            metric="mean_commitment_contribution",
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
            verbose=verbose,
        )
        result = result.copy()
        result["delta_CS"] = result["effect_b_minus_a"]
        return result

    def trajectory_shift(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compare replicate-level mean root progression velocity.

        The old cell-level pseudotime distribution test was removed.  This
        preserved method name now performs replicate-aware inference on the
        progression component of the projected velocity.
        """
        return self.compare_conditions(
            results=results,
            metric="progression_velocity",
            **kwargs,
        )

    def fit_mixed_model(
        self,
        *,
        fate: Optional[str] = None,
        metric: str = "mean_commitment_contribution",
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        condition_a: Optional[str] = None,
        condition_b: Optional[str] = None,
        on_invalid: str = "return",
    ) -> pd.DataFrame:
        """Fit optional cell-level mixed models with replicate random intercepts.

        Replicate-level permutation and hierarchical bootstrap remain the
        primary inference methods.  This sensitivity analysis fails closed:
        singular, boundary, non-converged, or ill-conditioned fits return no
        coefficient or p-value when ``on_invalid="return"`` (default), or
        raise ``RuntimeError`` when ``on_invalid="raise"``.
        """
        try:
            import statsmodels.api as sm
            from scipy.stats import norm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "fit_mixed_model requires statsmodels. Install scCS-py[drivers]."
            ) from exc

        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        if self.replicate_obs_key is None:
            raise ValueError("fit_mixed_model requires replicate_obs_key.")
        condition_a = self.conditions[0] if condition_a is None else str(condition_a)
        condition_b = self.conditions[1] if condition_b is None else str(condition_b)
        specs = self._outcome_specs(metric, fate=fate)
        rows = []

        for fate_name, _, _ in specs:
            values = []
            groups = []
            indicators = []
            for condition in (condition_a, condition_b):
                result = chosen[condition]
                if result.replicate_ids is None:
                    raise ValueError("Missing replicate IDs.")
                if metric == "mean_commitment_contribution":
                    j = result.fate_names.index(fate_name)
                    cell_values = result.commitment_contribution[:, j]
                elif metric == "directional_affinity":
                    j = result.fate_names.index(fate_name)
                    cell_values = result.directional_affinity[:, j]
                elif metric == "commitment_strength":
                    cell_values = result.commitment_strength
                elif metric == "directional_entropy":
                    cell_values = result.directional_entropy
                elif metric == "commitment_entropy":
                    cell_values = result.commitment_entropy
                elif metric == "directional_specificity":
                    cell_values = result.directional_specificity
                elif metric == "nearest_fate_angle_degrees":
                    cell_values = result.nearest_fate_angle_degrees
                elif metric == "specific_commitment":
                    cell_values = result.specific_commitment
                elif metric == "progression_velocity":
                    cell_values = result.progression_velocity
                else:
                    raise ValueError(
                        "fit_mixed_model does not support this metric. Use a scalar "
                        "cell-level outcome or one fate-specific affinity/contribution."
                    )
                finite = np.isfinite(cell_values)
                values.extend(cell_values[finite].tolist())
                # Replicate labels may be reused across independent conditions.
                # Prefixing prevents accidental pairing of unrelated samples.
                groups.extend(
                    [f"{condition}::{replicate}" for replicate in result.replicate_ids[finite]]
                )
                indicators.extend([1.0 if condition == condition_b else 0.0] * int(finite.sum()))

            endog = np.asarray(values, dtype=float)
            exog = np.column_stack([np.ones(len(endog)), np.asarray(indicators, dtype=float)])
            model = sm.MixedLM(endog, exog, groups=np.asarray(groups, dtype=str))
            audit = fit_mixedlm_fail_closed(
                model,
                n_fixed_effects=2,
                on_invalid=on_invalid,
            )

            common = {
                "metric": metric,
                "fate": fate_name,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_cells": len(endog),
                "n_replicates": len(np.unique(groups)),
                "valid_fit": bool(audit.valid),
                "failure_reason": audit.failure_reason,
                "warning_messages": list(audit.warning_messages),
                "fixed_effect_covariance_min_eigenvalue": (
                    audit.fixed_effect_covariance_min_eigenvalue
                ),
                "fixed_effect_covariance_condition_number": (
                    audit.fixed_effect_covariance_condition_number
                ),
                "random_effect_variance_min": audit.random_effect_variance_min,
                "converged": bool(audit.fit is not None and getattr(audit.fit, "converged", False)),
            }
            if not audit.valid:
                rows.append(
                    {
                        **common,
                        "effect_b_minus_a": np.nan,
                        "standard_error": np.nan,
                        "zvalue": np.nan,
                        "pvalue": np.nan,
                    }
                )
                continue

            assert audit.fit is not None
            assert audit.fixed_effect_covariance is not None
            fixed_params = np.asarray(audit.fit.fe_params, dtype=float)
            effect = float(fixed_params[1])
            variance = float(audit.fixed_effect_covariance[1, 1])
            standard_error = float(np.sqrt(variance))
            zvalue = effect / standard_error
            pvalue = float(2.0 * norm.sf(abs(zvalue)))
            rows.append(
                {
                    **common,
                    "effect_b_minus_a": effect,
                    "standard_error": standard_error,
                    "zvalue": zvalue,
                    "pvalue": pvalue,
                }
            )

        output = pd.DataFrame(rows)
        output["pvalue_adj"] = np.nan
        finite = output["pvalue"].notna().to_numpy()
        if finite.any():
            output.loc[finite, "pvalue_adj"] = holm_adjust(
                output.loc[finite, "pvalue"].to_numpy(dtype=float)
            )
        return output

    def plot_replicate_outcomes(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
        show_ci: bool = True,
        ax=None,
    ):
        """Plot biological-replicate outcomes for the two conditions."""
        return super().plot_replicate_outcomes(
            results=results,
            metric=metric,
            fate=fate,
            fate_pair=fate_pair,
            show_ci=show_ci,
            ax=ax,
        )

    def plot_affinity_distributions(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "directional_affinity",
        fate: Optional[str] = None,
        plot_type: str = "box",
        figsize: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        ax=None,
    ):
        """Plot biological-replicate outcome distributions by condition.

        This retains the familiar pre-v0.8 method name while using biological
        replicates, rather than cells, as the displayed independent units.
        ``plot_type`` may be ``"box"``, ``"violin"``, or ``"strip"``.
        """
        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        plot_type = str(plot_type).lower()
        if plot_type not in {"box", "violin", "strip"}:
            raise ValueError("plot_type must be 'box', 'violin', or 'strip'.")
        specs = self._outcome_specs(metric, fate=fate)
        table = self.replicate_table(chosen)

        import matplotlib.pyplot as plt

        if ax is not None and len(specs) != 1:
            raise ValueError("ax can be supplied only when one outcome is plotted.")
        if ax is None:
            width = 5.2 * len(specs)
            if figsize is None:
                figsize = (width, 4.8)
            figure, axes = plt.subplots(1, len(specs), figsize=figsize, squeeze=False)
            axes = axes.ravel()
        else:
            figure = ax.figure
            axes = np.asarray([ax], dtype=object)

        for axis, (fate_name, pair, column) in zip(axes, specs):
            distributions = [
                table.loc[table["condition"] == condition, column].dropna().to_numpy(float)
                for condition in self.conditions
            ]
            positions = np.arange(1, len(self.conditions) + 1)
            if plot_type == "box":
                axis.boxplot(distributions, positions=positions, widths=0.55)
            elif plot_type == "violin":
                axis.violinplot(
                    distributions,
                    positions=positions,
                    widths=0.75,
                    showmeans=True,
                    showextrema=True,
                )
            else:
                for position, values in zip(positions, distributions):
                    offsets = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else [0.0]
                    axis.scatter(position + np.asarray(offsets), values, alpha=0.85)
            axis.set_xticks(positions)
            axis.set_xticklabels(self.conditions, rotation=25, ha="right")
            axis.set_ylabel(_metric_label(metric, fate=fate_name))
            if fate_name is not None:
                axis.set_title(str(fate_name))
            elif pair is not None:
                axis.set_title(f"{pair[0]} / {pair[1]}")
            else:
                axis.set_title(_metric_label(metric))
        figure.suptitle(title or "Biological-replicate scCS distributions", y=1.02)
        return figure

    def plot_commitment_decomposition(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        fate: str,
        condition_a: Optional[str] = None,
        condition_b: Optional[str] = None,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        resample_cells_within_replicate: bool = True,
        random_state: int = 0,
        ax=None,
    ):
        """Decompose a condition effect into direction, strength, and contribution.

        The three displayed effects are condition B minus condition A for:
        directional affinity toward ``fate``, scalar commitment strength, and
        fate-specific mean commitment contribution.
        """
        chosen = self._resolve_results(results)
        condition_a = self.conditions[0] if condition_a is None else str(condition_a)
        condition_b = self.conditions[1] if condition_b is None else str(condition_b)
        specifications = [
            ("Directional affinity", "directional_affinity", fate),
            ("Commitment strength", "commitment_strength", None),
            ("Commitment contribution", "mean_commitment_contribution", fate),
        ]
        rows = []
        for offset, (label, metric, metric_fate) in enumerate(specifications):
            stats = self.hierarchical_bootstrap(
                condition_a=condition_a,
                condition_b=condition_b,
                metric=metric,
                fate=metric_fate,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                resample_cells_within_replicate=resample_cells_within_replicate,
                random_state=random_state + offset,
                results=chosen,
            )
            row = stats.iloc[0]
            rows.append(
                {
                    "component": label,
                    "effect": float(row["effect_b_minus_a"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                }
            )
        frame = pd.DataFrame(rows)

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7.4, 4.4))
        y = np.arange(len(frame))
        effect = frame["effect"].to_numpy(float)
        lower = frame["ci_lower"].to_numpy(float)
        upper = frame["ci_upper"].to_numpy(float)
        ax.errorbar(
            effect,
            y,
            xerr=np.vstack([effect - lower, upper - effect]),
            fmt="o",
            capsize=4,
        )
        ax.axvline(0.0, color="0.45", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(frame["component"])
        ax.invert_yaxis()
        ax.set_xlabel(f"{condition_b} minus {condition_a}")
        ax.set_title(f"scCS commitment-effect decomposition toward {fate}")
        return ax.figure

    def plot_effects(
        self,
        statistics: pd.DataFrame,
        *,
        alpha: float = 0.05,
        use_adjusted_pvalue: bool = True,
        ax=None,
    ):
        """Plot effect sizes and hierarchical-bootstrap confidence intervals.

        ``statistics`` should normally be returned by ``compare_conditions``
        with ``n_bootstrap > 0`` or by ``compute_delta_CS``.
        """
        required = {"effect_b_minus_a", "ci_lower", "ci_upper"}
        missing = sorted(required - set(statistics.columns))
        if missing:
            raise ValueError(
                "plot_effects requires confidence intervals; missing columns: " + ", ".join(missing)
            )
        if len(statistics) == 0:
            raise ValueError("statistics is empty.")

        frame = statistics.copy().reset_index(drop=True)
        labels = []
        for row in frame.itertuples(index=False):
            fate = getattr(row, "fate", None)
            fate_a = getattr(row, "fate_a", None)
            fate_b = getattr(row, "fate_b", None)
            labels.append(
                str(fate) if fate is not None and not pd.isna(fate) else f"{fate_a} / {fate_b}"
            )

        import matplotlib.pyplot as plt

        if ax is None:
            height = max(3.2, 0.48 * len(frame) + 1.6)
            _, ax = plt.subplots(figsize=(7.2, height))
        y = np.arange(len(frame))
        effect = frame["effect_b_minus_a"].to_numpy(dtype=float)
        lower = frame["ci_lower"].to_numpy(dtype=float)
        upper = frame["ci_upper"].to_numpy(dtype=float)
        ax.errorbar(
            effect,
            y,
            xerr=np.vstack([effect - lower, upper - effect]),
            fmt="o",
            capsize=4,
        )
        ax.axvline(0.0, linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        metric_name = (
            str(frame["metric"].iloc[0])
            if "metric" in frame.columns
            else "mean_commitment_contribution"
        )
        condition_a = (
            str(frame["condition_a"].iloc[0]) if "condition_a" in frame.columns else "condition A"
        )
        condition_b = (
            str(frame["condition_b"].iloc[0]) if "condition_b" in frame.columns else "condition B"
        )
        ax.set_xlabel(f"{_metric_label(metric_name)} effect: {condition_b} minus {condition_a}")
        ax.set_title("scCS replicate-level effects")

        pvalue_column = (
            "pvalue_adj" if use_adjusted_pvalue and "pvalue_adj" in frame.columns else "pvalue"
        )
        if pvalue_column in frame.columns:
            pvalues = frame[pvalue_column].to_numpy(dtype=float)
            right = np.nanmax(upper)
            span = max(np.nanmax(upper) - np.nanmin(lower), 1e-6)
            for index, pvalue in enumerate(pvalues):
                if np.isfinite(pvalue) and pvalue < alpha:
                    ax.text(right + 0.03 * span, index, "*", va="center")
        return ax.figure

    def plot_delta_CS_heatmap(
        self,
        statistics: pd.DataFrame,
        *,
        annotate: bool = True,
        use_adjusted_pvalue: bool = True,
        title: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        ax=None,
    ):
        """Plot fate-specific v0.8 ``delta_CS`` values as a heatmap.

        ``delta_CS`` is the replicate-aware condition-B minus condition-A
        difference in mean soft commitment contribution.  It is not the old
        angular-sector nCS statistic from pre-v0.8 releases.
        """
        if len(statistics) == 0:
            raise ValueError("statistics is empty.")
        frame = statistics.copy()
        value_column = "delta_CS" if "delta_CS" in frame.columns else "effect_b_minus_a"
        required = {value_column, "fate"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError("plot_delta_CS_heatmap is missing columns: " + ", ".join(missing))
        frame = frame.loc[frame["fate"].notna()].copy()
        if frame.empty:
            raise ValueError("statistics contains no fate-specific rows.")
        indexed = frame.set_index(frame["fate"].astype(str))
        ordered_fates = [fate for fate in self.branches if fate in indexed.index]
        if not ordered_fates:
            ordered_fates = indexed.index.astype(str).tolist()
        values = indexed.loc[ordered_fates, value_column].to_numpy(float)[None, :]

        import matplotlib.pyplot as plt

        if ax is None:
            default_size = (max(6.0, 1.25 * len(ordered_fates)), 2.8)
            _, ax = plt.subplots(figsize=default_size if figsize is None else figsize)
        limit = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
        if limit <= 0:
            limit = 1.0
        image = ax.imshow(values, aspect="auto", vmin=-limit, vmax=limit, cmap="coolwarm")
        ax.set_xticks(np.arange(len(ordered_fates)))
        ax.set_xticklabels(ordered_fates, rotation=30, ha="right")
        ax.set_yticks([0])
        condition_a = str(frame["condition_a"].iloc[0]) if "condition_a" in frame else "A"
        condition_b = str(frame["condition_b"].iloc[0]) if "condition_b" in frame else "B"
        ax.set_yticklabels([f"{condition_b} − {condition_a}"])
        ax.set_title(title or "Fate-specific change in mean commitment contribution")
        ax.figure.colorbar(image, ax=ax, label="delta_CS")

        if annotate:
            pvalue_column = (
                "pvalue_adj"
                if use_adjusted_pvalue and "pvalue_adj" in indexed.columns
                else "pvalue"
            )
            for column, fate in enumerate(ordered_fates):
                value = float(indexed.loc[fate, value_column])
                label = f"{value:.3f}"
                if pvalue_column in indexed.columns:
                    pvalue = float(indexed.loc[fate, pvalue_column])
                    if np.isfinite(pvalue):
                        label += f"\np={pvalue:.3g}"
                ax.text(column, 0, label, ha="center", va="center")
        return ax.figure

    def plot_delta_cs_heatmap(self, *args, **kwargs):
        """Lower-case compatibility alias for :meth:`plot_delta_CS_heatmap`."""
        return self.plot_delta_CS_heatmap(*args, **kwargs)
