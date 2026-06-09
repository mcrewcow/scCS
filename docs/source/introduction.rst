Introduction
============

**scCS** (Single-Cell Commitment Scores) is a Python package for quantifying
the degree of transcriptional commitment of single cells toward specific
differentiation fates, using RNA velocity as the underlying signal.

It generalizes the 2-state (homeostatic/activated) commitment score framework
introduced in Kriukov et al. (2025) to arbitrary **k-furcations** — branching
points where a progenitor population splits into k ≥ 2 terminal fates.

Three-Scorer Architecture
-------------------------

v0.7 introduces a three-scorer architecture, each designed for a specific
experimental design:

.. code-block:: text

    How many experimental conditions?
           │
           ├── 1 condition  →  SingleScorer
           │                    Single-condition analysis: embedding,
           │                    scoring, driver genes, enrichment
           │
           ├── 2 conditions →  PairScorer
           │                    Pairwise comparison: permutation test,
           │                    delta-CS with bootstrap CI, mixed model
           │
           └── 3+ conditions → MultiScorer
                                Multi-comparison: omnibus tests (Kruskal-Wallis,
                                ANOVA), post-hoc (Dunn, Tukey, Conover),
                                pairwise deltas, mixed model contrasts

**SingleScorer** — single-condition analysis
    Build a radial star embedding, compute commitment scores, per-cell fate
    affinities, entropy metrics, driver genes, and pathway enrichment.

**PairScorer** — pairwise comparison (exactly 2 conditions)
    Build a shared embedding on pooled data, score each condition separately,
    then compare using permutation tests, delta-CS with bootstrap CI,
    mixed-effects models, and trajectory shift analysis.

**MultiScorer** — multi-condition comparison (3+ conditions)
    Same shared-embedding approach as PairScorer, plus tiered statistical
    testing: omnibus tests (Kruskal-Wallis, ANOVA) followed by post-hoc
    pairwise comparisons (Dunn, Tukey HSD, Conover-Iman) with FDR/Bonferroni
    correction, all-pairs delta-CS, and mixed-model contrasts.

Motivation
----------

Classical RNA velocity tools (scVelo, CellRank) describe *where* cells are
going. scCS answers a complementary question: **how strongly committed is each
cell to a given fate, relative to the alternatives?**

This is particularly useful when:

- Comparing commitment levels across experimental conditions (e.g. treated vs control)
- Identifying driver genes that correlate with commitment
- Quantifying reversibility of cell state transitions
- Studying multi-fate branching points (k ≥ 3)
- Testing whether commitment shifts across 3+ treatment groups

Quick Start — SingleScorer
---------------------------

.. code-block:: python

    import scCS

    scorer = scCS.SingleScorer(
        adata,
        root='17',
        branches=['FateA', 'FateB', 'FateC'],
        obs_key='leiden',
    )
    scorer.build_embedding(ordering_metric='pseudotime')
    scorer.refit_pseudotime()
    scorer.fit()
    result = scorer.score(n_bootstrap=500)
    print(result.summary())
    scorer.plot_star(result)

Quick Start — PairScorer
-------------------------

.. code-block:: python

    pscorer = scCS.PairScorer(
        adata,
        root='17',
        branches=['homeostatic', 'activated'],
        condition_obs_key='treatment',
        obs_key='leiden',
    )
    pscorer.build_embedding(ordering_metric='pseudotime')
    pscorer.fit()
    results = pscorer.score_all_conditions()
    delta = pscorer.compute_delta_CS('control', 'treated')
    stats = pscorer.compare_conditions(results)

Quick Start — MultiScorer
--------------------------

.. code-block:: python

    mscorer = scCS.MultiScorer(
        adata,
        root='17',
        branches=['homeostatic', 'activated'],
        condition_obs_key='treatment',
        obs_key='leiden',
    )
    mscorer.build_embedding(ordering_metric='pseudotime')
    mscorer.fit()
    results = mscorer.score_all_conditions()
    omnibus = mscorer.compare_omnibus(results)
    posthoc = mscorer.compare_posthoc(results, omnibus_results=omnibus)
    deltas = mscorer.compute_pairwise_deltas()

Mathematical Framework
----------------------

Given per-cell RNA velocity vectors projected into a radial star embedding:

1. **Magnitude** — ``magnitude_i = sqrt(vx_i² + vy_i²)``
2. **Angle** — ``theta_i = atan2(vy_i, vx_i)`` mapped to [0°, 360°)
3. **Binning** — angles binned into N equal sectors of width 360°/N
4. **Sector magnitude** — ``M_sector(j) = sum of magnitudes in sector j``
5. **Unnormalized CS** — ``unCS(i, j) = M_sector(i) / M_sector(j)``
6. **Normalized CS** — ``nCS(i, j) = unCS(i, j) × n_cells(j) / n_cells(i)``

For k fates, a full k×k pairwise matrix of unCS and nCS is computed.
Per-cell scores are derived from the dot product of each cell's velocity
vector with the unit direction toward each fate centroid.

See :doc:`mathematical_framework` for full derivations with LaTeX.

Entropy Metrics
---------------

scCS provides three complementary entropy metrics, each answering a different
question about commitment uncertainty:

**Population entropy** (single scalar)
    How evenly is total velocity mass distributed across fate sectors?
    ``H_pop = 0`` → all mass in one sector; ``H_pop = 1`` → uniform.
    Note: can be misleadingly high for split-committed populations where
    cells are individually decisive but split between fates.

**Per-fate cell entropy** (shape ``(k,)``)
    For each fate *j*: how individually decisive are cells about fate *j*?
    Computed as the mean binary entropy of each cell's affinity score
    ``s_ij`` treated as ``[s_ij, 1−s_ij]``, averaged over all cells.
    Low → cells are sharply committed (or sharply not committed) to fate *j*.
    High → cells are ambiguous about fate *j* (scores cluster near 0.5).

**NN-smoothed per-cell entropy** (shape ``(n_cells,)``)
    For each cell: average ``cell_scores`` over its *k* nearest neighbors
    in the scCS embedding (``X_sccs``), then compute k-way Shannon entropy
    on the smoothed scores. Removes single-cell velocity noise while
    preserving local commitment structure. Use
    ``plot_nn_entropy_elbow()`` to choose the optimal *k*.

Workflow
--------

.. code-block:: text

    AnnData (with scVelo velocity)
           │
           ▼
    SingleScorer(adata, root=..., branches=[...], obs_key=...)
           │
           ├── build_embedding(ordering_metric='pseudotime')
           │                          → radial star layout in obsm['X_sccs']
           ├── refit_pseudotime()     → fix arm coverage with subset pseudotime
           │
           ├── fit()                  → builds FateMap, projects velocity
           │
           ├── score(k_nn=15)         → CommitmentScoreResult
           │     ├── M_sector         (per-fate velocity magnitude)
           │     ├── pairwise_unCS    (k×k unnormalized scores)
           │     ├── pairwise_nCS     (k×k cell-count corrected scores)
           │     ├── cell_scores      (per-cell fate affinities)
           │     ├── population_entropy   (scalar)
           │     ├── per_fate_entropy     (shape k)
           │     └── nn_cell_entropy      (shape n_cells)
           │
           ├── plot_star()                → radial embedding visualization
           ├── plot_commitment_bar()      → unCS/nCS bar chart
           ├── plot_rose()                → polar velocity magnitude plot
           ├── plot_pairwise_cs()         → k×k heatmap
           ├── plot_nn_entropy_elbow()    → choose optimal k_nn
           ├── plot_expression_trends()   → gene expression vs CS axis
           ├── get_velocity_drivers()     → ranked driver genes per fate
           ├── get_deg_drivers()          → DEG analysis per fate arm
           ├── get_velocity_fate_drivers() → velocity-fate Spearman correlation
           └── get_enrichment()           → pathway enrichment per fate

    PairScorer(adata, root=..., branches=[...],
               condition_obs_key=..., obs_key=...)
           │
           ├── build_embedding() / fit() / refit_pseudotime()
           │
           ├── score_all_conditions()     → dict[condition → CommitmentScoreResult]
           ├── compute_delta_CS()         → ΔnCS matrix with bootstrap CI
           ├── compare_conditions()       → permutation test (k=2)
           ├── fit_mixed_model()          → LMM with replicate random effect
           ├── trajectory_shift()         → KS + Wasserstein pseudotime shift
           │
           ├── plot_star_grid()           → side-by-side star panels per condition
           ├── plot_rose_grid()           → side-by-side rose plots per condition
           ├── plot_affinity_distributions() → violin/box/strip per fate
           ├── plot_delta_cs_heatmap()    → diverging ΔCS heatmap with CI
           ├── plot_compare_conditions_bar() → grouped nCS bar chart
           ├── plot_commitment_vector_radar() → radar chart of commitment vectors
           └── plot_trajectory_shift()    → KDE pseudotime distributions

    MultiScorer(adata, root=..., branches=[...],
                condition_obs_key=..., obs_key=...)
           │
           ├── build_embedding() / fit()
           │
           ├── score_all_conditions()     → dict[condition → CommitmentScoreResult]
           │
           ├── Tier 2: Omnibus + post-hoc
           │   ├── compare_omnibus()      → Kruskal-Wallis / ANOVA per fate
           │   ├── compare_posthoc()      → Dunn / Tukey / Conover pairwise
           │   └── compute_pairwise_deltas() → ΔCS for ALL condition pairs
           │
           ├── Tier 3: Mixed models + trajectory
           │   ├── fit_mixed_model()      → LMM with replicate random effect
           │   ├── fit_mixed_model_contrasts() → custom condition contrasts
           │   └── trajectory_shift()     → KS + Wasserstein pseudotime shift
           │
           ├── plot_omnibus_summary()     → fates × conditions heatmap + significance
           ├── plot_posthoc_heatmap()     → condition × condition p-value heatmap
           ├── plot_pairwise_delta_grid() → grid of ΔCS heatmaps for all pairs
           └── [all PairScorer plots also available]

Citation
--------

If you use scCS in your research, please cite:

    Kriukov et al. (2025) *Single-cell transcriptome of myeloid cells in
    response to transplantation of human retinal neurons reveals reversibility
    of microglial activation.* DOI: 10.XXXX
