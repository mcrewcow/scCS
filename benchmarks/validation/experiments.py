"""Executable, seeded graph-level experiments backing the validation notebooks.

Synthetic coordinates and velocities are not transcript/splicing simulations.
Prescribed-vector manifold generators are implemented in simulations.py.
Statistical intervals below are empirical diagnostics, not guarantees.
"""

from __future__ import annotations

from math import comb
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

import scCS
from scCS.future_fate import solve_discounted_outcomes
from scCS.inference import two_group_permutation_test
from . import simulations as sim


def fit_simulation(
    a,
    *,
    horizon=64,
    quantile=0.9,
    root="Root",
    branches=("A", "B"),
    ordering="latent_time",
    competing=None,
):
    scorer = scCS.SingleScorer(a, root=root, branches=list(branches), obs_key="clusters")
    scorer.build_embedding(
        ordering_metric=ordering,
        cache_subset=False,
        write_to_adata=False,
        verbose=False,
    )
    scorer.fit(
        a.obsp["T_velocity"],
        scoring_mode="future_fate",
        future_fate_options={
            "effective_horizon": horizon,
            "anchor_quantile": quantile,
            "min_anchor_cells": 3,
            "competing_outcomes": competing,
        },
        verbose=False,
    )
    return scorer, scorer.score(write_to_adata=False, verbose=False)


def root_summary(result):
    mask = result.root_mask
    defined = mask & result.affinity_defined
    row = {
        "DFR": float(result.discounted_fate_reach[mask].mean()),
        "UFP": float(result.unresolved_probability[mask].mean()),
        "coverage": float(defined.sum() / mask.sum()),
        "competing_reach": float(result.competing_reach[mask].mean()),
        "residual": result.solver_residual,
    }
    for j, fate in enumerate(result.fate_names):
        row[f"CFA_{fate}"] = (
            float(result.conditional_fate_affinity[defined, j].mean()) if defined.any() else np.nan
        )
    return row


def unequal_path_chain(split=0.5):
    # Root -> A directly, or Root -> B1 -> B2 -> B. Anchors are A and B.
    matrix = sparse.csr_matrix(
        [
            [0, split, 1 - split, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ]
    )
    anchors = np.zeros((5, 2), dtype=bool)
    anchors[1, 0], anchors[4, 1] = True, True
    return matrix, anchors


def exact_calibration(n_walks=100_000, seed=13):
    rng = np.random.default_rng(seed)
    rows = []
    for split in [0.2, 0.5, 0.8]:
        matrix, anchors = unequal_path_chain(split)
        for h in [4, 16, 64, 256]:
            gamma = h / (h + 1)
            result = solve_discounted_outcomes(
                matrix, anchors, ("A", "B"), effective_horizon=h, solver="direct"
            )
            # Independent geometric stopping and Bernoulli route sampling.
            takes_a = rng.random(n_walks) < split
            budget = rng.geometric(1 - gamma, size=n_walks) - 1
            empirical = np.array(
                [np.mean(takes_a & (budget >= 1)), np.mean(~takes_a & (budget >= 3))]
            )
            truth = np.array([split * gamma, (1 - split) * gamma**3])
            rows.append(
                {
                    "split": split,
                    "h": h,
                    "analytic_A": truth[0],
                    "analytic_B": truth[1],
                    "computed_A": result.probability[0, 0],
                    "computed_B": result.probability[0, 1],
                    "CFA_A": result.probability[0, 0] / result.probability[0].sum(),
                    "simulation_max_error": float(abs(empirical - truth).max()),
                    "solver_max_error": float(abs(result.probability[0] - truth).max()),
                    "mass_error": abs(
                        result.probability[0].sum() + result.unresolved_probability[0] - 1
                    ),
                }
            )
    return pd.DataFrame(rows)


def proliferation(n_seeds=20):
    scenarios = {
        "Control": {},
        "A expansion from entry": {"rate_A": 2},
        "A downstream expansion": {"rate_A": 1, "onset_A": 0.75},
        "B depletion from entry": {"rate_B": -1.5},
        "Changed entry density 60/40": {"density_A": 1.5},
    }
    rows = []
    for name, opts in scenarios.items():
        for seed in range(n_seeds):
            a = sim.make_fork(n_root=400, n_entry=180, seed=seed, **opts)
            labels = a.obs.clusters.astype(str)
            abundance = (labels == "A").sum() / labels.isin(["A", "B"]).sum()
            for q in [0.01, 0.90]:
                _, result = fit_simulation(a, quantile=q)
                rows.append(
                    {
                        "scenario": name,
                        "seed": seed,
                        "q": q,
                        "nominal_entry_share": opts.get("density_A", 1)
                        / (1 + opts.get("density_A", 1)),
                        "abundance_A": abundance,
                        **root_summary(result),
                    }
                )
    return pd.DataFrame(rows)


def horizon_sensitivity(n_seeds=5):
    rows = []
    for seed in range(n_seeds):
        a = sim.make_furcation(
            branches=(("A", 1, 1, False), ("B", 3, 1, False)),
            n_root=400,
            n_per_unit=180,
            seed=seed,
        )
        for q in [0.01, 0.50, 0.90, 0.95]:
            for h in [16, 32, 64, 128, 256, 4096]:
                _, result = fit_simulation(a, horizon=h, quantile=q)
                rows.append({"seed": seed, "q": q, "h": h, **root_summary(result)})
    return pd.DataFrame(rows)


def complex_dynamics():
    rows = []
    a = sim.make_furcation(
        branches=(("A", 1, 1, False), ("B", 1, 1, False), ("C", 1, 1, True)), seed=2
    )
    _, result = fit_simulation(a, branches=("A", "B", "C"))
    for label in ["Root", "A", "B", "C"]:
        mask = result.embedding.selected_labels == label
        rows.append(
            {
                "case": "retrograde",
                "population": label,
                "mean_SOF": result.signed_progression[mask].mean(),
                "DFR": result.discounted_fate_reach[mask].mean(),
            }
        )
    retrograde = pd.DataFrame(rows)
    a = sim.make_furcation(seed=3, outside={"name": "Cycling", "n": 500, "attach": 0.5})
    labels = a.obs.clusters.astype(str)
    cycling = labels.eq("Cycling").to_numpy()
    threshold = np.quantile(a.obs.loc[cycling, "latent_time"], 0.9)
    anchors = cycling & (a.obs.latent_time.to_numpy() >= threshold)
    _, first = fit_simulation(a)
    _, second = fit_simulation(a, competing={"Cycling": anchors})
    competing = pd.DataFrame(
        [
            {"case": "unmodeled", **root_summary(first)},
            {"case": "explicit competing", **root_summary(second)},
        ]
    )
    loops = []
    for w in [0, 0.25, 0.5, 1]:
        a = sim.make_reversible_loop(w_res=w, n_blob=300, n_arc=180, seed=0)
        labels = a.obs.clusters.astype(str)
        _, r = fit_simulation(
            a,
            root="Transitory",
            branches=("Homeostatic", "Activated"),
            ordering="outward",
        )
        mask = r.embedding.selected_labels == "Activated"
        j = r.fate_names.index("Homeostatic")
        occupancy = sim.discounted_occupancy(
            a.obsp["T_velocity"], {"Homeostatic": labels.eq("Homeostatic")}
        )
        loops.append(
            {
                "return_density": w,
                "CFA_return_nonanchor": np.nanmean(
                    r.conditional_fate_affinity[mask & ~r.selected_anchor_mask.any(1), j]
                ),
                "occupancy_homeostatic": occupancy.loc[
                    labels.eq("Activated").to_numpy(), "Homeostatic"
                ].mean(),
                "dead_ends": int(sim.dead_end_cells(a.obsp["T_velocity"]).sum()),
            }
        )
    return retrograde, competing, pd.DataFrame(loops)


def sampling_uncertainty(n_datasets=12, n_subsamples=20, fraction=0.5):
    rows = []
    for seed in range(n_datasets):
        a = sim.make_fork(n_root=400, n_entry=180, seed=seed)
        _, r = fit_simulation(a)
        estimate = root_summary(r)["CFA_A"]
        rng = np.random.default_rng(17000 + seed)
        values = []
        for _ in range(n_subsamples):
            idx = np.sort(rng.choice(a.n_obs, int(fraction * a.n_obs), replace=False))
            b = a[idx].copy()
            b.obsp["T_velocity"] = sim.velocity_transition_matrix(
                np.asarray(b.X), b.layers["velocity"]
            )
            _, rb = fit_simulation(b)
            values.append(root_summary(rb)["CFA_A"])
        se = np.std(values, ddof=1) * np.sqrt(fraction / (1 - fraction))
        rows.append(
            {
                "seed": seed,
                "estimate": estimate,
                "subsample_sd": np.std(values, ddof=1),
                "rescaled_se": se,
                "lower": estimate - 1.96 * se,
                "upper": estimate + 1.96 * se,
                "covers_nominal_half": abs(estimate - 0.5) <= 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def inference_calibration(n_null=500, n_effect=200, n_replicates=5):
    rng = np.random.default_rng(1701)
    rows = []
    for effect, repetitions in [(0, n_null), (1.25, n_effect)]:
        for _ in range(repetitions):
            # Independent experimental units; cells are never permuted here.
            x = rng.normal(size=n_replicates)
            y = rng.normal(loc=effect, size=n_replicates)
            result = two_group_permutation_test(
                np.r_[x, y],
                ["control"] * n_replicates + ["treated"] * n_replicates,
                "control",
                "treated",
                n_permutations=999,
                random_state=0,
            )
            rows.append(
                {
                    "effect": effect,
                    "pvalue": result.pvalue,
                    "rejected": result.pvalue < 0.05,
                }
            )
    return pd.DataFrame(rows)


def replicate_safeguards():
    a = sim.make_conditions(n_reps=5, effect=1, seed=3, n_root=120, n_per_unit=100)
    s = scCS.PairScorer(
        a,
        root="Root",
        branches=["A", "B"],
        obs_key="clusters",
        condition_obs_key="condition",
        replicate_obs_key="sample_id",
    )
    s.build_embedding(ordering_metric="latent_time", cache_subset=False, verbose=False)
    s.fit(
        a.obsp["T_velocity"],
        scoring_mode="future_fate",
        future_fate_options={"min_anchor_cells": 3},
        verbose=False,
    )
    s.score_all_conditions(min_cells=2, verbose=False, write_to_adata=False)
    test = s.compare_conditions(metric="conditional_fate_affinity", verbose=False)
    idx = np.flatnonzero(a.obs.sample_id.eq("treated_2") & a.obs.clusters.eq("Root"))
    a.obsp["T_velocity"] = sim.make_dead_ends(a.obsp["T_velocity"], idx)
    s.fit(
        a.obsp["T_velocity"],
        scoring_mode="future_fate",
        future_fate_options={"min_anchor_cells": 3},
        verbose=False,
    )
    s.score_all_conditions(min_cells=2, verbose=False, write_to_adata=False)
    try:
        s.compare_conditions(metric="conditional_fate_affinity", fate="A", verbose=False)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("An undefined replicate must raise.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dropped = s.compare_conditions(
            metric="conditional_fate_affinity",
            fate="A",
            undefined_replicates="drop",
            verbose=False,
        )
    return test, error, dropped, [str(w.message) for w in caught]


def pvalue_floors():
    return pd.DataFrame(
        [
            {
                "replicates_per_group": n,
                "independent_two_sided_floor": (2 / comb(2 * n, n)),
                "paired_sign_flip_floor": 2 / (2**n),
            }
            for n in range(2, 9)
        ]
    )


def occupancy_sensitivity():
    a = sim.make_reversible_loop(w_res=1, n_blob=300, n_arc=180, seed=0)
    labels = a.obs.clusters.astype(str)
    masks = {name: labels.eq(name).to_numpy() for name in labels.unique()}
    matrix = a.obsp["T_velocity"]
    repaired, dead = sim.repair_dead_ends(matrix, np.asarray(a.X))
    rows = []
    for name, transition in [("original", matrix), ("added neighbor edges", repaired)]:
        for h in [8, 64, 1024, 20_000]:
            occupancy = sim.discounted_occupancy(transition, masks, effective_horizon=h)
            rows.append(
                {
                    "graph": name,
                    "h": h,
                    "dead_end_count": len(dead),
                    "activated_from_H": occupancy.loc[masks["Homeostatic"], "Activated"].mean(),
                    "activated_from_A": occupancy.loc[masks["Activated"], "Activated"].mean(),
                    "mass_error": abs(occupancy.sum(1) - 1).max(),
                }
            )
    return pd.DataFrame(rows)
