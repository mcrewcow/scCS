"""Replicate-aware multi-condition comparison for scCS v0.8."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .condition import (
    ConditionCommitmentResult,
    ConditionScorer,
    _canonical_metric,
    _metric_label,
)
from .inference import (
    holm_adjust,
    one_way_permutation_test,
    two_group_permutation_test,
)
from .mixed_models import fit_mixedlm_fail_closed


class MultiScorer(ConditionScorer):
    """Compare commitment across three or more biological conditions."""

    minimum_conditions = 3
    maximum_conditions = None

    def compare_omnibus(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
        n_permutations: int = 9999,
        random_state: int = 0,
        adjust_pvalues: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Replicate-label permutation omnibus test across all conditions."""
        chosen = self._resolve_results(results)
        metric_public = str(metric)
        metric = _canonical_metric(metric)
        if self.replicate_obs_key is None:
            raise ValueError("compare_omnibus requires replicate_obs_key.")
        table = self.replicate_table(chosen)
        rows = []
        specs = self._outcome_specs(metric, fate=fate, fate_pair=fate_pair)
        for offset, (fate_name, pair, column) in enumerate(specs):
            subset = table[["condition", column]].dropna()
            test = one_way_permutation_test(
                subset[column].to_numpy(dtype=float),
                subset["condition"].to_numpy(str),
                n_permutations=n_permutations,
                random_state=random_state + offset,
            )
            means = {
                condition: float(subset.loc[subset["condition"] == condition, column].mean())
                for condition in self.conditions
            }
            rows.append(
                {
                    "metric": metric,
                    "metric_public": metric_public,
                    "metric_label": _metric_label(metric_public, fate=fate_name),
                    "fate": fate_name,
                    "fate_a": None if pair is None else pair[0],
                    "fate_b": None if pair is None else pair[1],
                    "statistic": test.statistic,
                    "pvalue": test.pvalue,
                    "permutation_method": test.method,
                    "n_permutations": test.n_permutations,
                    "n_conditions": len(self.conditions),
                    "condition_means": means,
                }
            )
        output = pd.DataFrame(rows)
        output["pvalue_adj"] = (
            holm_adjust(output["pvalue"].to_numpy())
            if adjust_pvalues and len(output) > 1
            else output["pvalue"].to_numpy()
        )
        if verbose:
            print(f"[scCS] Multi-condition replicate omnibus test; metric={metric!r}.")
        return output

    def compare_posthoc(
        self,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
        omnibus_results: Optional[pd.DataFrame] = None,
        only_significant_omnibus: bool = True,
        omnibus_alpha: float = 0.05,
        n_permutations: int = 9999,
        max_exact: int = 100_000,
        random_state: int = 0,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Pairwise replicate-label tests with Holm correction."""
        chosen = self._resolve_results(results)
        metric_public = str(metric)
        metric = _canonical_metric(metric)
        if self.replicate_obs_key is None:
            raise ValueError("compare_posthoc requires replicate_obs_key.")
        specs = self._outcome_specs(metric, fate=fate, fate_pair=fate_pair)
        if omnibus_results is not None and only_significant_omnibus:
            significant_keys = set()
            for row in omnibus_results.itertuples(index=False):
                adjusted = getattr(row, "pvalue_adj", getattr(row, "pvalue"))
                if adjusted < omnibus_alpha:
                    significant_keys.add((row.fate, row.fate_a, row.fate_b))
            specs = [
                spec
                for spec in specs
                if (
                    spec[0],
                    None if spec[1] is None else spec[1][0],
                    None if spec[1] is None else spec[1][1],
                )
                in significant_keys
            ]
        if not specs:
            return pd.DataFrame(
                columns=[
                    "metric",
                    "metric_public",
                    "metric_label",
                    "fate",
                    "fate_a",
                    "fate_b",
                    "condition_a",
                    "condition_b",
                    "effect_b_minus_a",
                    "pvalue",
                    "pvalue_adj",
                ]
            )

        table = self.replicate_table(chosen)
        rows = []
        counter = 0
        for fate_name, pair, column in specs:
            for condition_a, condition_b in combinations(self.conditions, 2):
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
                    random_state=random_state + counter,
                )
                counter += 1
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
                        "effect_b_minus_a": test.effect,
                        "pvalue": test.pvalue,
                        "permutation_method": test.method,
                        "n_permutations": test.n_permutations,
                        "n_replicates_a": test.n_group_a,
                        "n_replicates_b": test.n_group_b,
                    }
                )
        output = pd.DataFrame(rows)
        output["pvalue_adj"] = holm_adjust(output["pvalue"].to_numpy())
        if verbose:
            print(f"[scCS] Multi-condition post-hoc replicate tests; metric={metric!r}.")
        return output

    def compare_contrast(
        self,
        weights: Mapping[object, float],
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        fate_pair: Optional[tuple[str, str]] = None,
        n_permutations: int = 9999,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """Test a planned zero-sum contrast on replicate-level outcomes."""
        chosen = self._resolve_results(results)
        metric_public = str(metric)
        metric = _canonical_metric(metric)
        if self.replicate_obs_key is None:
            raise ValueError("compare_contrast requires replicate_obs_key.")
        normalized = {str(key): float(value) for key, value in weights.items()}
        if set(normalized) != set(self.conditions):
            raise ValueError("weights must specify every condition exactly once.")
        if not np.isclose(sum(normalized.values()), 0.0, atol=1e-12):
            raise ValueError("Contrast weights must sum to zero.")
        if n_permutations < 1:
            raise ValueError("n_permutations must be positive.")

        table = self.replicate_table(chosen)
        labels = table["condition"].to_numpy(str)
        rng = np.random.default_rng(random_state)
        rows = []
        specs = self._outcome_specs(metric, fate=fate, fate_pair=fate_pair)
        for fate_name, pair, column in specs:
            values = table[column].to_numpy(dtype=float)

            def statistic(group_labels: np.ndarray) -> float:
                return float(
                    sum(
                        normalized[condition] * values[group_labels == condition].mean()
                        for condition in self.conditions
                    )
                )

            observed = statistic(labels)
            exceed = 0
            for _ in range(n_permutations):
                permuted = rng.permutation(labels)
                exceed += abs(statistic(permuted)) >= abs(observed) - 1e-15
            pvalue = (exceed + 1) / (n_permutations + 1)
            rows.append(
                {
                    "metric": metric,
                    "metric_public": metric_public,
                    "metric_label": _metric_label(metric_public, fate=fate_name),
                    "fate": fate_name,
                    "fate_a": None if pair is None else pair[0],
                    "fate_b": None if pair is None else pair[1],
                    "contrast": dict(normalized),
                    "estimate": observed,
                    "pvalue": float(pvalue),
                    "n_permutations": int(n_permutations),
                }
            )
        output = pd.DataFrame(rows)
        output["pvalue_adj"] = (
            holm_adjust(output["pvalue"].to_numpy()) if len(output) > 1 else output["pvalue"]
        )
        return output

    def fit_mixed_model(
        self,
        *,
        metric: str = "mean_commitment_contribution",
        fate: Optional[str] = None,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        on_invalid: str = "return",
    ) -> pd.DataFrame:
        """Fit optional mixed models with fail-closed omnibus Wald tests.

        Replicate-level permutation remains the primary omnibus inference.
        Singular, boundary, non-converged, or ill-conditioned fits return no
        coefficient or p-value by default instead of reporting misleading
        ``converged=True`` results from an invalid covariance estimate.
        """
        try:
            import statsmodels.api as sm
            from scipy.stats import chi2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "fit_mixed_model requires statsmodels. Install scCS-py[drivers]."
            ) from exc

        chosen = self._resolve_results(results)
        metric = _canonical_metric(metric)
        if self.replicate_obs_key is None:
            raise ValueError("fit_mixed_model requires replicate_obs_key.")
        specs = self._outcome_specs(metric, fate=fate)
        rows = []
        reference = self.conditions[0]
        alternatives = self.conditions[1:]

        for fate_name, _, _ in specs:
            values = []
            groups = []
            condition_values = []
            for condition in self.conditions:
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
                groups.extend(
                    [f"{condition}::{replicate}" for replicate in result.replicate_ids[finite]]
                )
                condition_values.extend([condition] * int(finite.sum()))

            condition_values_array = np.asarray(condition_values, dtype=str)
            design = [np.ones(len(values), dtype=float)]
            for condition in alternatives:
                design.append((condition_values_array == condition).astype(float))
            exog = np.column_stack(design)
            model = sm.MixedLM(
                np.asarray(values, dtype=float),
                exog,
                groups=np.asarray(groups, dtype=str),
            )
            audit = fit_mixedlm_fail_closed(
                model,
                n_fixed_effects=len(self.conditions),
                on_invalid=on_invalid,
            )

            common = {
                "metric": metric,
                "fate": fate_name,
                "reference_condition": reference,
                "n_cells": len(values),
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
                        "condition_coefficients": {condition: np.nan for condition in alternatives},
                        "wald_chi2": np.nan,
                        "df": len(alternatives),
                        "pvalue": np.nan,
                    }
                )
                continue

            assert audit.fit is not None
            assert audit.fixed_effect_covariance is not None
            fixed_params = np.asarray(audit.fit.fe_params, dtype=float)
            beta = fixed_params[1 : len(self.conditions)]
            covariance = audit.fixed_effect_covariance[
                1 : len(self.conditions), 1 : len(self.conditions)
            ]
            statistic = float(beta.T @ np.linalg.solve(covariance, beta))
            df = len(alternatives)
            pvalue = float(chi2.sf(statistic, df))
            rows.append(
                {
                    **common,
                    "condition_coefficients": {
                        condition: float(value) for condition, value in zip(alternatives, beta)
                    },
                    "wald_chi2": statistic,
                    "df": df,
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

    def plot_omnibus_summary(
        self,
        omnibus_results: pd.DataFrame,
        results: Optional[Mapping[str, ConditionCommitmentResult]] = None,
        *,
        posthoc_df: Optional[pd.DataFrame] = None,
        annotate: bool = True,
        use_adjusted_pvalue: bool = True,
        ax=None,
    ):
        """Plot condition means and omnibus significance for each outcome.

        ``results`` and ``posthoc_df`` are accepted for tutorial compatibility.
        Omnibus annotations are taken from ``omnibus_results``; post-hoc effects
        should be shown with :meth:`plot_posthoc_heatmap`.
        """
        del results, posthoc_df
        if len(omnibus_results) == 0:
            raise ValueError("omnibus_results is empty.")
        required = {"condition_means", "metric"}
        missing = sorted(required - set(omnibus_results.columns))
        if missing:
            raise ValueError("plot_omnibus_summary is missing columns: " + ", ".join(missing))
        frame = omnibus_results.reset_index(drop=True)
        row_labels = []
        rows = []
        for row in frame.itertuples(index=False):
            fate = getattr(row, "fate", None)
            fate_a = getattr(row, "fate_a", None)
            fate_b = getattr(row, "fate_b", None)
            if fate is not None and not pd.isna(fate):
                label = str(fate)
            elif fate_a is not None and fate_b is not None:
                label = f"{fate_a} / {fate_b}"
            else:
                label = _metric_label(str(row.metric))
            means = dict(row.condition_means)
            row_labels.append(label)
            rows.append([float(means.get(condition, np.nan)) for condition in self.conditions])
        matrix = np.asarray(rows, dtype=float)

        import matplotlib.pyplot as plt

        if ax is None:
            height = max(3.4, 0.48 * len(row_labels) + 1.8)
            _, ax = plt.subplots(figsize=(7.5, height))
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(np.arange(len(self.conditions)))
        ax.set_xticklabels(self.conditions, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        metric = str(frame["metric"].iloc[0])
        ax.set_title(f"Omnibus summary: {_metric_label(metric)}")
        ax.figure.colorbar(image, ax=ax, label=_metric_label(metric))

        if annotate and matrix.size <= 150:
            pvalue_column = (
                "pvalue_adj" if use_adjusted_pvalue and "pvalue_adj" in frame.columns else "pvalue"
            )
            for row_index in range(matrix.shape[0]):
                pvalue = (
                    float(frame.loc[row_index, pvalue_column])
                    if pvalue_column in frame.columns
                    else float("nan")
                )
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    if np.isfinite(value):
                        text = f"{value:.3f}"
                        if column_index == matrix.shape[1] - 1 and np.isfinite(pvalue):
                            text += f"\np={pvalue:.3g}"
                        ax.text(column_index, row_index, text, ha="center", va="center")
        return ax.figure

    def plot_posthoc_heatmap(
        self,
        posthoc_results: pd.DataFrame,
        *,
        fate: Optional[str] = None,
        annotate: bool = True,
        use_adjusted_pvalue: bool = True,
        ax=None,
    ):
        """Plot an antisymmetric condition-effect matrix for one outcome."""
        if len(posthoc_results) == 0:
            raise ValueError("posthoc_results is empty.")
        required = {"condition_a", "condition_b", "effect_b_minus_a"}
        missing = sorted(required - set(posthoc_results.columns))
        if missing:
            raise ValueError("plot_posthoc_heatmap is missing columns: " + ", ".join(missing))
        frame = posthoc_results.copy()
        if "fate" in frame.columns:
            available = [str(value) for value in pd.unique(frame["fate"].dropna())]
            if fate is None and len(available) > 1:
                if ax is not None:
                    raise ValueError(
                        "fate is required when plotting multiple-fate post-hoc "
                        "results into one supplied axis."
                    )
                return self.plot_pairwise_delta_grid(
                    posthoc_results,
                    annotate=annotate,
                    use_adjusted_pvalue=use_adjusted_pvalue,
                )
            if fate is not None:
                fate = str(fate)
                frame = frame.loc[frame["fate"].astype(str) == fate]
                if frame.empty:
                    raise ValueError(f"No post-hoc rows were found for fate {fate!r}.")
        if frame.empty:
            raise ValueError("No post-hoc rows remain after filtering.")

        size = len(self.conditions)
        effects = np.full((size, size), np.nan, dtype=float)
        pvalues = np.full((size, size), np.nan, dtype=float)
        np.fill_diagonal(effects, 0.0)
        np.fill_diagonal(pvalues, 1.0)
        condition_to_index = {condition: index for index, condition in enumerate(self.conditions)}
        pvalue_column = (
            "pvalue_adj" if use_adjusted_pvalue and "pvalue_adj" in frame.columns else "pvalue"
        )
        for row in frame.itertuples(index=False):
            condition_a = str(row.condition_a)
            condition_b = str(row.condition_b)
            if condition_a not in condition_to_index or condition_b not in condition_to_index:
                continue
            index_a = condition_to_index[condition_a]
            index_b = condition_to_index[condition_b]
            effect = float(row.effect_b_minus_a)
            effects[index_a, index_b] = effect
            effects[index_b, index_a] = -effect
            if pvalue_column in frame.columns:
                pvalue = float(getattr(row, pvalue_column))
                pvalues[index_a, index_b] = pvalue
                pvalues[index_b, index_a] = pvalue

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 5.5))
        finite = effects[np.isfinite(effects)]
        limit = float(np.max(np.abs(finite))) if len(finite) else 1.0
        if limit <= 0:
            limit = 1.0
        image = ax.imshow(effects, vmin=-limit, vmax=limit, cmap="coolwarm")
        ticks = np.arange(size)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.conditions, rotation=30, ha="right")
        ax.set_yticklabels(self.conditions)
        title = "Post-hoc effects (column minus row)"
        if fate is not None:
            title += f": {fate}"
        ax.set_title(title)
        ax.figure.colorbar(image, ax=ax, label="Effect")
        if annotate:
            for row_index in range(size):
                for column_index in range(size):
                    value = effects[row_index, column_index]
                    if not np.isfinite(value):
                        continue
                    text = f"{value:.3f}"
                    pvalue = pvalues[row_index, column_index]
                    if row_index != column_index and np.isfinite(pvalue):
                        text += f"\np={pvalue:.3g}"
                    ax.text(column_index, row_index, text, ha="center", va="center")
        return ax.figure

    def plot_pairwise_delta_grid(
        self,
        posthoc_results: pd.DataFrame,
        *,
        ncols: int = 3,
        annotate: bool = True,
        use_adjusted_pvalue: bool = True,
        figsize_per_panel: tuple[float, float] = (5.6, 4.8),
    ):
        """Plot one post-hoc condition-effect heatmap per fate."""
        if len(posthoc_results) == 0:
            raise ValueError("posthoc_results is empty.")
        if "fate" not in posthoc_results.columns:
            raise ValueError("posthoc_results must contain a 'fate' column.")
        fates = [
            fate
            for fate in self.branches
            if fate in set(posthoc_results["fate"].dropna().astype(str))
        ]
        if not fates:
            raise ValueError("posthoc_results contains no fate-specific rows.")
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")

        import matplotlib.pyplot as plt

        ncols = min(ncols, len(fates))
        nrows = int(np.ceil(len(fates) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )
        for fate, axis in zip(fates, axes.ravel()):
            self.plot_posthoc_heatmap(
                posthoc_results,
                fate=fate,
                annotate=annotate,
                use_adjusted_pvalue=use_adjusted_pvalue,
                ax=axis,
            )
        for axis in axes.ravel()[len(fates) :]:
            axis.set_visible(False)
        fig.suptitle("Multi-condition post-hoc effects", y=1.01)
        return fig
