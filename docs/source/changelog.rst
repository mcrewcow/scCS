Changelog
=========

v0.3.2 (2026-03-12)
--------------------

**New features**

- ``compute_per_fate_cell_entropy(cell_scores)`` → ``ndarray shape (k,)``.
  For each fate *j*: mean binary Shannon entropy of each cell's affinity
  score ``s_ij`` treated as a Bernoulli distribution ``[s_ij, 1−s_ij]``,
  averaged over all cells. Low = cells are sharply decisive about that fate;
  high = cells are ambiguous (scores cluster near 0.5).
- ``compute_nn_cell_entropy(cell_scores, coords, k_nn)`` → ``ndarray shape (n_cells,)``.
  For each cell: average ``cell_scores`` over its ``k_nn`` nearest neighbors
  in the scCS embedding (``X_sccs``), then compute normalized k-way Shannon
  entropy on the smoothed scores. Removes single-cell velocity noise while
  preserving local commitment structure.
- ``CommitmentScorer.score(k_nn=...)`` — new optional parameter. When set,
  computes NN-smoothed per-cell entropy and stores it in
  ``result.nn_cell_entropy`` and ``adata_sub.obs['cs_nn_entropy']``.
- ``plot_nn_entropy_elbow(scorer, k_nn_range)`` — two-panel figure for
  choosing ``k_nn``: mean NN entropy across all cells (left) and per fate
  arm (right) vs k. Also accessible as ``scorer.plot_nn_entropy_elbow()``.

**Changed**

- ``CommitmentScoreResult`` gains three new fields:
  ``per_fate_entropy`` (shape ``(k,)``),
  ``nn_cell_entropy`` (shape ``(n_cells,)`` or ``None``),
  ``nn_k`` (``int`` or ``None``).
- ``summary()`` now prints per-fate entropy and NN entropy (when computed).
- Version bumped to ``0.3.2``.

v0.3.1 (2026-03-12)
--------------------

**Fixed — entropy quantification redesign**

The previous ``commitment_entropy`` metric operated on the aggregate
commitment vector ``p_vec = M_sector / sum(M_sector)``. A population split
50/50 between two strongly committed sub-groups yielded ``H ≈ 1`` (maximum
uncertainty) even though every individual cell was decisive, making the
metric uninformative for real bifurcations.

- ``compute_population_entropy(p_vec)`` → ``float``. Renamed from
  ``compute_commitment_entropy``. Same math, clarified semantics: measures
  how evenly total velocity mass is distributed across fate sectors.
- ``compute_mean_cell_entropy(cell_scores)`` → ``float``. New primary metric.
  Computes normalized Shannon entropy independently for each cell's
  fate-affinity vector, then averages. Correctly distinguishes a
  split-committed bifurcation (``H_cell ≈ 0``) from a genuinely uncommitted
  population (``H_cell ≈ 1``).
- ``CommitmentScoreResult``: field ``commitment_entropy`` renamed to
  ``population_entropy``; new field ``mean_cell_entropy`` added.
  ``commitment_entropy`` retained as a deprecated property that returns
  ``population_entropy`` with a ``DeprecationWarning``.
- ``adata_sub.obs['cs_entropy']`` now stores per-cell normalized Shannon
  entropy (formula unchanged, now consistent with ``mean_cell_entropy``).
- Version bumped to ``0.3.1``.

v0.2.2 (2025-03-11)
--------------------

**Bug fixes**

- ``plot_expression_trends``: fixed ``IndexError`` when ``adata`` contains
  more cells than the scored subset. ``CommitmentScoreResult`` now stores
  ``cell_obs_names`` so expression extraction is always correctly aligned.
- ``plot_commitment_bar``: fixed all-``-1`` values for k ≥ 3 furcations.
  Now produces **k subplots** (one per reference fate) so every population
  is shown as both query and reference. Nothing is hidden.

v0.2.1 (2025-03-10)
--------------------

**New features**

- ``plot_expression_trends()``: CellRank-style gene expression vs commitment
  axis plot. Cells binned by per-cell fate affinity score; mean expression
  per bin plotted with LOWESS smooth. Supports any gene in ``adata.var_names``,
  any AnnData layer, and custom fate selection.
- ``color_map`` parameter added to all plot functions. Pass a dict of
  ``{fate_name: hex_color}`` to preserve your original scanpy/Seurat cluster
  colors across all scCS plots. Progenitor cells always remain gray.
- ``plot_commitment_bar`` rewritten: now shows **unCS** (solid bars) and
  **nCS** (hatched bars, same fate color) side by side. CS = 1 reference
  line included.

**Internal**

- ``_fate_colors()`` updated to accept optional ``color_map`` override.
- ``CommitmentScoreResult.cell_obs_names`` field added.

v0.2.0 (2025-03-07)
--------------------

**New features**

- Generalized k-furcation support (k ≥ 2).
- ``plot_pairwise_cs()``: heatmap of full k×k unCS/nCS matrix.
- ``plot_commitment_heatmap()``: per-cell fate affinity heatmap.
- ``plot_subset_comparison()``: compare CS across experimental subsets
  via ``scorer.score_per_subset()``.
- ``get_velocity_drivers()``: rank genes by mean scVelo velocity per fate arm.
- ``get_deg_drivers()``: Wilcoxon rank-sum DEG analysis per fate arm.
- ``run_enrichment_per_fate()``: Enrichr ORA (KEGG, GO BP, Reactome).
- Fate detection backends: GMM, PAGA, CellRank, supervised.

v0.1.0 (2025-03-01)
--------------------

**Initial release**

- 2-state (homeostatic/activated) commitment score framework.
- unCS and nCS for bifurcation (k=2).
- Radial star embedding (``X_sccs`` in ``obsm``).
- ``plot_star_embedding()``, ``plot_rose()``.
- Based on: Kriukov et al. (2025).
