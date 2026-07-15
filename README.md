# scCS

**scCS** is a supervised framework for quantifying cell-fate commitment at a
manually annotated furcation. You provide one root population, two or more
candidate fate populations, an ordering coordinate, and an RNA-velocity
transition graph. scCS standardizes the trajectory as a root-plus-star display
while keeping the scientific calculation tied to the original biological graph.

scCS does **not** infer topology, discover terminal states, or replace RNA
velocity. It answers a narrower question:

> Given a biologically justified root and candidate fates, how strongly and how
> specifically is each cell associated with those futures, and is it moving
> forward, backward, or ambiguously along the supplied ordering?


## Start here

New to scCS? Read the **Introduction** in Read the Docs before the API pages. It explains what scCS assumes, how DFFP differs from a projected velocity vector, how to choose a scorer and ordering, and how to interpret CFA, DFR, FFS, RC, UFP, and SOF together. The **Mathematical framework** page derives the discounted hitting-probability equations and Signed Ordering Flux step by step.

Recommended learning order:

1. Introduction and mathematical framework;
2. pancreas or Schwann SingleScorer tutorial;
3. method-selection and complex-branch tutorials;
4. PairScorer or MultiScorer for replicate-aware conditions;
5. downstream analysis and scalability.

## Two explicit scoring modes

scCS keeps two scientifically different questions separate.

### Discounted Future-Fate Propagation — recommended for fate identity

**Discounted Future-Fate Propagation (DFFP)** is implemented by
`scoring_mode="future_fate"`. It calculates geometrically discounted hitting
probabilities on the original RNA-velocity transition graph. Paths may leave the
selected cells and return. The star does not define the probabilities.

The principal outputs and documentation terms are:

- **Conditional Fate Affinity (CFA)** — `future_fate_affinity` — conditional probability across the supervised fates;
- **Discounted Fate Reach (DFR)** — `future_fate_reach`, the discounted probability of reaching any supervised fate anchor;
- **Future-Fate Specificity (FFS)** — `1 - normalized entropy(affinity)`;
- **Resolved Commitment (RC)** — `reach_supported_specificity`, reach multiplied by specificity;
- **Unresolved Future Probability (UFP)** — `unresolved_probability`, the probability that the discounted process stops first;
- **Signed Ordering Flux (SOF)** — `signed_progression`, the expected change in the supplied ordering after
  conditioning on transitions retained within the selected furcation.

Fate identity and progression are independent. A cell can retain a strong future
association with a fate while moving backward, turning, looping, or showing mixed
motion. scCS therefore does not require every annotated branch to move monotonically
outward.

### Instantaneous transition pushforward — retained for local direction

`scoring_mode="instantaneous"` projects the immediate transition-induced
displacement into the supervised simplex geometry and calculates cosine-softmax
directional affinity. This remains useful when the scientific question is local
velocity direction rather than future reachability.

The two modes are not interchangeable. The documentation and method-selection
tutorial explain the distinction and the validation that led to the discounted
future-fate mode.

## Installation

```bash
pip install scCS-py
```

RNA-velocity utilities require the optional scVelo dependency:

```bash
pip install "scCS-py[velocity]"
```

For all optional analysis features:

```bash
pip install "scCS-py[all]"
```

## Quick start: future-fate scoring

```python
import scCS

scorer = scCS.SingleScorer(
    adata,
    root=("Ngn3 high EP", "Pre-endocrine"),
    branches=["Alpha", "Beta", "Delta", "Epsilon"],
    obs_key="clusters",
)

# Any validated continuous ordering may be used: latent time, velocity
# pseudotime, diffusion pseudotime, Palantir pseudotime, or another justified
# progression coordinate.
scorer.build_embedding(ordering_metric="latent_time")

scorer.fit(
    scoring_mode="future_fate",
    future_fate_options={
        "effective_horizon": 64,
        "anchor_quantile": 0.90,
        "min_anchor_cells": 10,
        "progression_scale": "rank",
    },
)

result = scorer.score()
print(result.summary())

scorer.plot_star(result, color_by="future_fate_reach")
scorer.plot_star(result, color_by="future_fate_specificity")
scorer.plot_star(result, color_by="reach_supported_specificity")
scorer.plot_star(result, color_by="signed_progression")
scorer.plot_star(result, color_by="future_fate_affinity:Beta")
```

The established metric names remain available for compatibility. In future-fate
mode their explicit meanings are:

| Compatibility name | Future-fate meaning |
|---|---|
| `directional_affinity` | future-fate affinity |
| `commitment_strength` | future-fate reach |
| `directional_entropy` | future-fate entropy |
| `directional_specificity` | future-fate specificity |
| `specific_commitment` | reach-supported specificity |
| `commitment_contribution` | discounted fate probability by fate |
| `progression_velocity` | signed ordering progression |
| `transition_coverage` | one-step selected-path coverage |

Additional read-only terminology aliases are available as
`conditional_fate_affinity`, `discounted_fate_reach`, `resolved_commitment`,
`signed_ordering_flux`, and `unresolved_future_probability`.

Condition scorers also accept explicit aliases such as
`metric="future_fate_affinity"`, `metric="future_fate_reach"`, and
`metric="reach_supported_specificity"`.

## Optional competing outcomes

Competing outcomes are never guessed silently. Supply them as full-data Boolean
masks or full-data indices when they are biologically justified:

```python
scorer.fit(
    scoring_mode="future_fate",
    future_fate_options={
        "effective_horizon": 64,
        "competing_outcomes": {
            "Sensory": adata.obs["assignments"].eq("Sensory").to_numpy(),
            "Sympathetic": adata.obs["assignments"].eq("Symp").to_numpy(),
        },
    },
)
```

Without explicit competitors, unmodelled futures remain unresolved. This is safer
than automatically converting large annotated groups into terminal outcomes.

## Quick start: instantaneous scoring

```python
scorer.build_embedding(ordering_metric="latent_time")
scorer.fit(scoring_mode="instantaneous")
instantaneous = scorer.score()

scorer.plot_star(instantaneous, color_by="affinity:Beta")
scorer.plot_direction_strength_map(instantaneous)
scorer.plot_rose(instantaneous)
```

Scientific velocity-vector diagnostics, rose plots derived from velocity angles,
and scVelo grids on the display star are defined only for instantaneous mode.
Future-fate mode deliberately does not manufacture a star-space scientific
velocity vector.

## Choosing an ordering

The ordering is a supervised model input, not an estimate invented by scCS.
Use a continuous coordinate that is coherent for the selected trajectory and
oriented so larger values mean later progression. Typical choices include:

- dynamical latent time;
- velocity pseudotime;
- diffusion pseudotime;
- Palantir pseudotime;
- CytoTRACE-derived ordering;
- experimentally justified continuous developmental time.

Validate the ordering against independent biology whenever possible. A smooth
star is not evidence that an ordering is correct.

## Interpreting complex branches

Unusual branch dynamics are not automatically errors.

- **Retrograde branch:** strong fate affinity with negative signed progression.
- **Loop or turn:** fate identity may remain stable while progression changes sign.
- **Mixed branch:** low specificity or low probability margin may be biologically real.
- **Non-sink-like endpoint:** anchor diagnostics warn that late annotated cells
  transition toward root, another fate, or outside the selected path.

Anchor diagnostics are warnings for interpretation, not requirements that all
terminal populations behave as simple outward sinks.

## Pairwise and multi-condition inference

`PairScorer` and `MultiScorer` fit one pooled scientific model and perform formal
inference on biological-replicate summaries.

```python
pair = scCS.PairScorer(
    adata,
    root="Common Progenitor",
    branches=["Gut", "Gut neuron", "ChC"],
    obs_key="cell_type_new",
    condition_obs_key="condition",
    replicate_obs_key="sample_id",
)
pair.build_embedding(ordering_metric="inverse_cytotrace_pseudotime")
pair.fit(
    scoring_mode="future_fate",
    future_fate_options={"effective_horizon": 64},
)
condition_results = pair.score_all_conditions(population="root")

stats = pair.compare_conditions(
    condition_results,
    metric="future_fate_affinity",
    fate="ChC",
)
```

Formal inference requires biological replicates. Cell-level values are descriptive
and enter only the within-replicate level of the hierarchical bootstrap. At least
four, and preferably five to six or more, independent replicates per condition are
recommended.

## Why this method

The methodological tutorials document the alternatives tested during development:

1. direct transition pushforward into fixed star vertices;
2. continuous straight branch coordinates;
3. scVelo projection onto the display star;
4. re-fitting neighbors or velocity on scCS coordinates;
5. local trajectory-tangent straightening;
6. first-exit absorbing models;
7. unlimited full-graph absorption;
8. discounted future-fate hitting probabilities.

The selected method preserves source-graph dynamics, allows leave-and-return paths,
avoids circular neighborhoods defined by the supervised star, does not force loops
or retrograde branches to look outward, and stabilizes before unlimited graph mixing.
See **Method selection** in the documentation and the package-backed decision notebook.

## Documentation and tutorials

The repository contains six complete dataset-specific scorer guides:

- pancreas SingleScorer;
- RegVelo Schwann SingleScorer;
- pancreas PairScorer;
- RegVelo Schwann PairScorer;
- pancreas MultiScorer;
- RegVelo Schwann MultiScorer.

It also contains:

- a visual DFFP method-selection tutorial;
- a visual guide to turning, loop-like, mixed, and retrograde branches;
- a package-backed pancreas and Schwann method-decision reproduction;
- dedicated pancreas and Schwann downstream-analysis tutorials;
- a single-process, in-memory, no-chunking scalability notebook whose exact
  target ladder extends to 200 million cells.

Each primary notebook loads its public dataset directly and recomputes RNA velocity;
no hidden tutorial cache is assumed. The Schwann tutorials use dynamical velocity as
their primary model. Every guide covers ordering validation, preflight diagnostics,
discounted future-fate scoring, anchor and horizon sensitivity, population and
cell-level visualization, native/display velocity QC, gene-expression visualization,
replicate-aware inference where applicable, and reproducible export. PairScorer and
MultiScorer notebooks include a clearly labeled controlled demonstration mode and a
separate real-study path requiring genuine condition and biological-replicate
metadata.

The repository also contains a method-selection tutorial explaining rejected
alternatives, a guide to complex/loop-like/retrograde branches, and the package-API
reproduction of the pancreas and RegVelo Schwann decision benchmark, together with
API reference, diagnostics, scalability guidance, and release notes.

Documentation: https://sccs-py.readthedocs.io/

## Scope and limitations

- scCS is supervised and does not discover topology.
- Future-fate results depend on the velocity graph, ordering, fate annotations,
  endpoint-anchor definition, and effective horizon.
- `effective_horizon=64` is a validated default, not a universal biological time unit.
- Endpoint anchors need not be perfect sinks, but weak sink behavior should be reported.
- Paired and repeated-measure condition designs are not silently approximated.
- Gene associations are candidate associations, not causal lineage drivers.
- The star is a standardized display; it is not an inverse map of UMAP or gene space.

## Citation

A manuscript describing the redesigned method is in preparation. Until then, cite
the software repository and record the exact package version, scoring mode, ordering,
effective horizon, anchor quantile, and velocity model used in the analysis.


## Exact 200-million-cell no-chunk run

The scalability tutorial measures complete in-memory DFFP problems only. On a
high-memory host, set `SCCS_SCALABILITY_PROFILE=exact_200m` before running the
scalability notebook, or use
`benchmarks/v08/run_dffp_200m_no_chunk.py`. The runner fails when the entire
problem cannot be allocated; it never substitutes a chunked calculation.
