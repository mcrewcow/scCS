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
    CommitmentScorer(adata, bifurcation_cluster, terminal_cell_types)
           │
           ├── build_embedding()      → radial star layout in obsm['X_sccs']
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
           └── get_enrichment()           → pathway enrichment per fate

Citation
--------

If you use scCS in your research, please cite:

    Kriukov et al. (2025) *Single-cell transcriptome of myeloid cells in
    response to transplantation of human retinal neurons reveals reversibility
    of microglial activation.* DOI: 10.XXXX
