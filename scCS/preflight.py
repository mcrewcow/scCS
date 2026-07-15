"""Preflight diagnostics for scCS v0.8 workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreflightDiagnostic:
    code: str
    level: str
    message: str
    value: object = None


@dataclass(frozen=True)
class PreflightReport:
    diagnostics: tuple[PreflightDiagnostic, ...]

    @property
    def errors(self):
        return tuple(d for d in self.diagnostics if d.level == "error")

    @property
    def warnings(self):
        return tuple(d for d in self.diagnostics if d.level == "warning")

    @property
    def info(self):
        return tuple(d for d in self.diagnostics if d.level == "info")

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def raise_for_errors(self) -> None:
        if self.errors:
            text = "; ".join(f"{d.code}: {d.message}" for d in self.errors)
            raise ValueError(f"scCS preflight failed: {text}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"level": d.level, "code": d.code, "message": d.message, "value": d.value}
                for d in self.diagnostics
            ]
        )

    def summary(self) -> str:
        lines = [
            f"scCS preflight: {'PASS' if self.ok else 'FAIL'}",
            f"  errors={len(self.errors)}, warnings={len(self.warnings)}, info={len(self.info)}",
        ]
        for diagnostic in self.diagnostics:
            lines.append(f"  [{diagnostic.level.upper()}] {diagnostic.code}: {diagnostic.message}")
        return "\n".join(lines)

    def display(self):
        try:
            from IPython.display import display

            display(self.to_frame())
        except Exception:
            print(self.summary())
        return self


def _append(items: list[PreflightDiagnostic], code: str, level: str, message: str, value=None):
    items.append(PreflightDiagnostic(code=code, level=level, message=message, value=value))


def single_preflight(
    scorer, *, ordering_metric="pseudotime", check_velocity: bool = True
) -> PreflightReport:
    diagnostics: list[PreflightDiagnostic] = []
    adata = scorer.adata
    try:
        validation = scorer.furcation.validate_adata(adata)
        _append(
            diagnostics,
            "furcation_valid",
            "info",
            "Root and terminal annotations are valid.",
            validation.selected_count,
        )
        _append(
            diagnostics,
            "root_cells",
            "info",
            f"Root population contains {validation.root_count} cells.",
            validation.root_count,
        )
        for fate, count in validation.terminal_counts.items():
            level = "warning" if count < 20 else "info"
            _append(
                diagnostics,
                f"terminal_{fate}",
                level,
                f"Terminal {fate!r} contains {count} cells.",
                count,
            )
    except Exception as exc:
        _append(diagnostics, "furcation_invalid", "error", str(exc))
        return PreflightReport(tuple(diagnostics))

    try:
        resolved = scorer._resolve_ordering_alias(adata, ordering_metric)
        if isinstance(resolved, str):
            if resolved not in adata.obs:
                raise KeyError(f"Ordering column {resolved!r} is missing.")
            values = adata.obs[resolved].to_numpy(dtype=float)
        else:
            values = np.asarray(resolved, dtype=float)
        if len(values) != adata.n_obs or values.ndim != 1:
            raise ValueError("Ordering must be one-dimensional and match adata.n_obs.")
        label_values = adata.obs[scorer.obs_key].astype(str).to_numpy()
        selected_mask = validation.selected_mask
        selected_values = values[selected_mask]
        finite_fraction = float(np.mean(np.isfinite(selected_values)))
        if finite_fraction < 1.0:
            _append(
                diagnostics,
                "ordering_nonfinite",
                "error",
                "Ordering contains non-finite values among selected furcation cells.",
                finite_fraction,
            )
        elif float(np.ptp(selected_values)) <= np.finfo(float).eps:
            _append(
                diagnostics,
                "ordering_constant",
                "error",
                "Ordering metric is constant among selected furcation cells.",
            )
        else:
            _append(
                diagnostics,
                "ordering_valid",
                "info",
                "Ordering metric is finite and non-constant among selected furcation cells.",
                finite_fraction,
            )
            root_mask = np.isin(label_values, scorer.furcation.root_labels)
            root_values = values[root_mask]
            unique_values, unique_counts = np.unique(root_values, return_counts=True)
            unique_fraction = float(len(unique_values) / len(root_values))
            largest_tie_fraction = float(unique_counts.max() / len(root_values))
            if len(unique_values) < 20 or unique_fraction < 0.05 or largest_tie_fraction > 0.20:
                _append(
                    diagnostics,
                    "ordering_highly_discrete",
                    "warning",
                    (
                        f"Root ordering has only {len(unique_values)} unique values across "
                        f"{len(root_values)} cells (largest tied group "
                        f"{largest_tie_fraction:.1%}). Scientific root coordinates will "
                        "form radial bands. Use a continuous pseudotime or another "
                        "biologically justified continuous ordering; scCS will not "
                        "break ties arbitrarily."
                    ),
                    unique_fraction,
                )
            else:
                _append(
                    diagnostics,
                    "ordering_resolution",
                    "info",
                    (
                        f"Root ordering has {len(unique_values)} unique values across "
                        f"{len(root_values)} cells."
                    ),
                    unique_fraction,
                )
    except Exception as exc:
        _append(diagnostics, "ordering_invalid", "error", str(exc))

    if check_velocity:
        has_graph = "velocity_graph" in adata.uns
        has_vectors = scorer._projection_result is not None
        if not has_graph and not has_vectors:
            _append(
                diagnostics,
                "velocity_missing",
                "error",
                "No scVelo velocity graph or supplied projected velocities are available.",
            )
        else:
            _append(diagnostics, "velocity_available", "info", "Velocity information is available.")

    if scorer._result is not None:
        result = scorer._result
        root = result.root_mask
        coverage = result.projection.transition_coverage[root]
        valid = result.projection.velocity_defined[root]
        median = float(np.nanmedian(coverage)) if len(coverage) else float("nan")
        q05 = float(np.nanquantile(coverage, 0.05)) if len(coverage) else float("nan")
        _append(
            diagnostics,
            "transition_coverage",
            "info" if q05 >= 0.5 else "warning",
            f"Root transition coverage median={median:.3f}, q05={q05:.3f}.",
            q05,
        )
        undefined = float(np.mean(~valid)) if len(valid) else 1.0
        _append(
            diagnostics,
            "undefined_velocity",
            "warning" if undefined > 0.05 else "info",
            f"Undefined root projected-velocity fraction={undefined:.3f}.",
            undefined,
        )
        no_signal = not result.root_population_summary.composition_defined
        if no_signal:
            _append(
                diagnostics,
                "no_commitment_signal",
                "warning",
                "Root commitment composition is undefined because total commitment mass is zero.",
            )

    return PreflightReport(tuple(diagnostics))


def condition_preflight(
    scorer, *, ordering_metric="pseudotime", check_velocity: bool = True
) -> PreflightReport:
    diagnostics = list(
        single_preflight(
            scorer._scorer, ordering_metric=ordering_metric, check_velocity=check_velocity
        ).diagnostics
    )
    adata = scorer.adata
    condition_values = adata.obs[scorer.condition_obs_key].astype(str)
    for condition in scorer.conditions:
        n_cells = int(np.sum(condition_values.to_numpy() == condition))
        _append(
            diagnostics,
            f"condition_{condition}_cells",
            "warning" if n_cells < 20 else "info",
            f"Condition {condition!r} contains {n_cells} cells.",
            n_cells,
        )
    if scorer.replicate_obs_key is None:
        _append(
            diagnostics,
            "replicate_key_missing",
            "warning",
            "No replicate key was supplied; formal condition inference will be unavailable.",
        )
    else:
        raw_rep = adata.obs[scorer.replicate_obs_key].astype(str)
        cross = pd.crosstab(raw_rep, condition_values)
        shared = cross.index[(cross > 0).sum(axis=1) > 1].tolist()
        if shared:
            _append(
                diagnostics,
                "replicate_labels_reused",
                "info",
                (
                    "Some raw replicate labels occur in multiple conditions. They are "
                    "condition-qualified and treated as independent groups; use distinct "
                    "biological identifiers if this reuse is accidental."
                ),
                shared[:10],
            )
        counts = (
            pd.DataFrame({"condition": condition_values, "replicate": raw_rep})
            .drop_duplicates()
            .groupby("condition")
            .size()
        )
        for condition in scorer.conditions:
            count = int(counts.get(condition, 0))
            level = "error" if count < 2 else ("warning" if count < 4 else "info")
            _append(
                diagnostics,
                f"condition_{condition}_replicates",
                level,
                f"Condition {condition!r} contains {count} biological replicates.",
                count,
            )
    return PreflightReport(tuple(diagnostics))
