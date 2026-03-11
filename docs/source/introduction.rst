Introduction
============

**scCS** (Single-Cell Commitment Scores) is a Python package for quantifying
the degree of transcriptional commitment of single cells toward specific
differentiation fates, using RNA velocity as the underlying signal.

It generalizes the 2-state (homeostatic/activated) commitment score framework
introduced in Kriukov et al. (2025) to arbitrary **k-furcations** — branching
points where a progenitor population splits into k ≥ 2 terminal fates.

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

The **commitment entropy** H summarizes how evenly distributed velocity is
across all fates:

- H = 0 → all velocity points toward one fate (maximally committed)
- H = 1 → velocity evenly distributed across all fates (uncommitted)

Workflow
--------

.. code-block:: text

    AnnData (with scVelo velocity)
           │
           ▼
    CommitmentScorer(adata, cluster_key)
           │
           ├── build_embedding()   → radial star layout in obsm['X_sccs']
           │
           ├── score()             → CommitmentScoreResult
           │     ├── M_sector      (per-fate velocity magnitude)
           │     ├── pairwise_unCS (k×k unnormalized scores)
           │     ├── pairwise_nCS  (k×k cell-count corrected scores)
           │     ├── cell_scores   (per-cell fate affinities)
           │     └── commitment_entropy
           │
           ├── plot_star()         → radial embedding visualization
           ├── plot_commitment_bar() → unCS/nCS bar chart
           ├── plot_expression_trends() → gene expression vs CS axis
           └── get_velocity_drivers()   → ranked driver genes per fate

Citation
--------

If you use scCS in your research, please cite:

    Kriukov et al. (2025) *Single-cell transcriptome of myeloid cells in
    response to transplantation of human retinal neurons reveals reversibility
    of microglial activation.* DOI: 10.XXXX
