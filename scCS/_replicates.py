"""Shared eligibility and hypothesis-family rules for replicate inference."""

from __future__ import annotations

import warnings

import numpy as np

from .inference import holm_adjust


def prepare_replicates(table, column, conditions, *, undefined_replicates="raise"):
    """Validate an outcome without silently changing the replicate population."""
    if undefined_replicates not in {"raise", "drop"}:
        raise ValueError("undefined_replicates must be 'raise' or 'drop'.")
    conditions = list(conditions)
    selected = table.loc[table["condition"].isin(conditions)].copy()
    finite = np.isfinite(selected[column].to_numpy(dtype=float))
    excluded = selected.loc[~finite, ["condition", "replicate_id"]].to_dict("records")
    for row in excluded:
        row["reason"] = f"Non-finite replicate outcome: {column}"
    if excluded:
        names = ", ".join(str(row["replicate_id"]) for row in excluded)
        message = (
            f"Undefined replicate outcome {column!r} for {names}. "
            "Inspect fate reach and defined-affinity coverage. "
            "Use undefined_replicates='drop' only for an explicit conditional analysis."
        )
        if undefined_replicates == "raise":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)
    included = selected.loc[finite].copy()
    requested_counts = {c: int((selected.condition == c).sum()) for c in conditions}
    included_counts = {c: int((included.condition == c).sum()) for c in conditions}
    insufficient = {c: n for c, n in included_counts.items() if n < 2}
    if insufficient:
        raise ValueError(
            f"At least two defined biological replicates per condition are required; "
            f"included counts={included_counts}, requested counts={requested_counts}; "
            f"excluded replicates={excluded}."
        )
    diagnostic_columns = [
        c
        for c in (
            "condition",
            "replicate_id",
            "n_cells",
            "n_defined_affinity",
            "mean_affinity_defined_fraction",
            "mean_commitment_strength",
            "mean_transition_coverage",
        )
        if c in selected
    ]
    audit = {
        "undefined_replicates": undefined_replicates,
        "n_replicates_requested": requested_counts,
        "n_replicates_included": included_counts,
        "n_replicates_excluded": {
            c: requested_counts[c] - included_counts[c] for c in conditions
        },
        "excluded_replicates": excluded,
        "replicate_diagnostics": selected[diagnostic_columns].to_dict("records"),
    }
    return included, audit


def adjust_inference(output, table, specs, metric, fate_names, *, adjust=True):
    """Keep both fate rows but count a verified complementary K=2 test once.

    This applies only to two-sided replicate tests of normalized affinity or
    composition. Raw fate probabilities and K>2 outcomes remain separate tests.
    """
    complementary = False
    if (
        metric in {"directional_affinity", "commitment_composition"}
        and len(fate_names) == 2
        and len(specs) == 2
        and {s[0] for s in specs} == set(fate_names)
    ):
        a = table[specs[0][2]].to_numpy(float)
        b = table[specs[1][2]].to_numpy(float)
        fa, fb = np.isfinite(a), np.isfinite(b)
        complementary = bool(
            np.array_equal(fa, fb)
            and fa.any()
            and np.allclose(a[fa] + b[fa], 1.0, rtol=0, atol=1e-12)
        )
    keys = []
    for i, row in output.iterrows():
        contrast = tuple(
            (c, row[c]) for c in ("condition_a", "condition_b") if c in output
        )
        endpoint = (
            "complementary_two_fate"
            if complementary
            else (row.get("fate"), row.get("fate_a"), row.get("fate_b"))
        )
        keys.append((contrast, endpoint))
    representatives = {}
    ids = []
    for i, key in enumerate(keys):
        if key not in representatives:
            representatives[key] = i
        ids.append(list(representatives).index(key))
    first = list(representatives.values())
    p = output["pvalue"].to_numpy(float)
    canonical = p[first]
    adjusted = holm_adjust(canonical) if adjust else canonical
    output["pvalue"] = [canonical[i] for i in ids]
    output["pvalue_adj"] = [adjusted[i] for i in ids]
    output["hypothesis_id"] = [f"H{i + 1}" for i in ids]
    output["hypothesis_family_size"] = len(first)
    output["multiplicity_method"] = "holm" if adjust else "none"
    output["complementary_fates_collapsed"] = complementary
    return output


def replicate_plot_values(table, column, condition):
    """Make undefined display values and their replicate denominator visible."""
    group = table.loc[table.condition == condition]
    values = group[column].to_numpy(float)
    finite = np.isfinite(values)
    label = str(condition)
    if not finite.all():
        ids = group.loc[~finite, "replicate_id"].tolist()
        warnings.warn(
            f"Undefined {column!r} values cannot be plotted for replicates {ids}; "
            "the axis reports defined/requested replicate counts. "
            "Use replicate_table() to inspect reach and coverage.",
            UserWarning,
            stacklevel=3,
        )
        label += f"\n({int(finite.sum())}/{len(values)} defined)"
    return values[finite], label
