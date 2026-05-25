Changelog
=========

v0.6.2 (2026-05-25)
-------------------

**Bug fixes**

- **Bug 1** — ``plot_nn_entropy_elbow()`` raised ``AttributeError: 'CommitmentScorer' object
  has no attribute 'cluster_key'``. Fixed: ``scorer.cluster_key`` → ``scorer.obs_key``
  in ``plot.py`` (the v0.6.1 rename was not propagated to this call site).

- **Bug 2** — ``score_per_subset()`` printed each subset result twice when
  ``verbose=True``: once from the internal ``score()`` call and once from
  ``score_per_subset`` itself. Fixed: ``score()`` is now always called with
  ``verbose=False`` inside ``score_per_subset``; the subset header and summary
  are printed only by ``score_per_subset``.

- **Bug 3** — ``score_per_subset()`` produced ``inf`` in ``pairwise_nCS`` for
  progenitor-only subsets (e.g., "Pre-endocrine") with no explanation.
  This is mathematically correct (nCS is undefined when a fate arm has 0 cells),
  but was confusing. Fixed:

  - ``score_per_subset()`` now emits a ``UserWarning`` when all off-diagonal
    nCS entries are ``inf``, explaining that the subset contains no cells from
    any fate arm.
  - ``CommitmentScoreResult.summary()`` now appends a footnote line when any
    ``pairwise_nCS`` entry is ``inf``:
    ``"(inf = fate arm has 0 cells in this subset; expected for progenitor-only subsets)"``.

v0.6.1 (2026-05-24)
-------------------

**New features**

- **``get_velocity_fate_drivers()``** — velocity-fate correlation driver method.
  Computes Spearman correlation between each gene's velocity and per-cell fate
  affinity scores (CellRank-style). Returns FDR-corrected p-values via
  Benjamini-Hochberg. Available as ``scorer.get_velocity_fate_drivers(result)`` and
  as standalone ``scCS.get_velocity_fate_drivers()``.

- **``plot_rose_grid()``** — per-condition rose plot grid.
  One polar subplot per condition, all panels sharing the same radial scale for
  direct magnitude comparison. Available as ``mscorer.plot_rose_grid(results)`` and
  as standalone ``scCS.plot_rose_grid()``.

- **``plot_delta_cs_heatmap()``** — ΔCS heatmap with CI annotation.
  Visualizes ``compute_delta_CS()`` output as a diverging heatmap annotated with
  Δ ± CI_half per cell. Available as ``mscorer.plot_delta_cs_heatmap(delta)`` and
  as standalone ``scCS.plot_delta_cs_heatmap()``.

- **``plot_compare_conditions_bar()``** — grouped bar chart of nCS per condition.
  One bar group per fate pair, one bar per condition, colored by ``CONDITION_PALETTE``.
  Available as ``mscorer.plot_compare_conditions_bar(results)`` and standalone.

- **``plot_commitment_vector_radar()``** — radar/spider chart of commitment vectors.
  Each condition is one closed polygon; axes = fate names; values = commitment
  vector (sums to 1). Falls back to bar chart for k < 3. Available as
  ``mscorer.plot_commitment_vector_radar(results)`` and standalone.

- **``CONDITION_PALETTE``** — new colorblind-safe palette for condition coloring
  (distinct from ``FATE_PALETTE``). Used automatically in all multi-condition plots.

- **``_condition_colors()``** — helper mirroring ``_fate_colors()`` but drawing from
  ``CONDITION_PALETTE``. Used in ``plot_affinity_distributions()``,
  ``plot_trajectory_shift()``, ``plot_rose_grid()``, and the three new plots.

**API renames (hard rename — no deprecation shims)**

All renames are breaking changes. Update call sites accordingly.

.. list-table::
   :header-rows: 1

   * - Old name
     - New name
     - Scope
   * - ``bifurcation_cluster``
     - ``root``
     - All classes and functions
   * - ``terminal_cell_types``
     - ``branches``
     - All classes and functions
   * - ``cluster_key``
     - ``obs_key``
     - All classes and functions
   * - ``condition_key``
     - ``condition_obs_key``
     - ``MultiConditionScorer``
   * - ``sector_mode``
     - ``sector_method``
     - Both scorers
   * - ``differentiation_metric``
     - ``ordering_metric``
     - ``build_embedding()``
   * - ``invert_metric``
     - ``invert_ordering``
     - ``build_embedding()``
   * - ``scale_metric``
     - ``scale_ordering``
     - ``build_embedding()``
   * - ``n_bins`` (constructor)
     - ``n_angle_bins``
     - Both scorers
   * - ``pval_cutoff``
     - ``pval_threshold``
     - drivers, enrichment, ``compare_conditions()``
   * - ``logfc_cutoff``
     - ``logfc_threshold``
     - drivers, enrichment
   * - ``n_top``
     - ``n_top_genes``
     - ``get_velocity_drivers()``, ``get_deg_drivers()``
   * - ``n_top_terms``
     - ``n_top_pathways``
     - enrichment functions
   * - ``compute_cell_level``
     - ``cell_level``
     - ``score()``, ``score_all_conditions()``
   * - ``subset_key``
     - ``split_by``
     - ``score_per_subset()``
   * - ``pseudotime_col``
     - ``pseudotime_key``
     - ``trajectory_shift()``, ``plot_trajectory_shift()``
   * - ``sample_key``
     - ``replicate_key``
     - ``fit_mixed_model()``
   * - ``reference_condition``
     - ``ref_condition``
     - ``fit_mixed_model()``
   * - ``reference_fate``
     - ``ref_fate``
     - ``plot_commitment_bar()``, ``plot_subset_comparison()``
   * - ``sccs_arm_name`` (obs col)
     - ``sccs_branch``
     - ``embedding.py``, ``plot.py``
   * - ``velocity_pseudotime_sub`` (obs col)
     - ``sccs_pseudotime``
     - multiple files
   * - ``uns["sccs"]["bifurcation_cluster"]``
     - ``uns["sccs"]["root"]``
     - multiple files
   * - ``FateMap.bifurcation_cluster``
     - ``FateMap.root``
     - ``bifurcation.py``
   * - ``FateMap.cluster_key``
     - ``FateMap.obs_key``
     - ``bifurcation.py``
   * - ``rebuild_embedding_with_subset_pseudotime()``
     - ``refit_pseudotime()``
     - Both scorers
   * - ``recompute_subset_pseudotime()``
     - ``compute_local_pseudotime()``
     - ``CommitmentScorer``
   * - ``plot_condition_comparison()``
     - ``plot_affinity_distributions()``
     - ``MultiConditionScorer``
   * - ``plot_condition_star()``
     - ``plot_star_grid()``
     - ``MultiConditionScorer``

**Removed**

- ``MultiConditionScorer.score_per_condition()`` — was a thin alias for
  ``score_all_conditions()``. Use ``score_all_conditions()`` directly.

**Bug fixes**

- **Bug E** — ``plot_affinity_distributions()`` and ``plot_trajectory_shift()`` now
  use ``CONDITION_PALETTE`` for condition colors instead of ``FATE_PALETTE``.
- **Bug F** — ``plot_expression_trends()`` error message now correctly references
  ``compute_local_pseudotime()`` (was ``recompute_subset_pseudotime()``).

v0.6.0 (2026-05-23)
-------------------

**Bug fixes (13 total)**

- **Fix #1** — ``plot_nn_entropy_elbow`` docstring: removed false prerequisite
  claiming ``score()`` must be called before the elbow plot.
- **Fix #2** — ``write_to_obs=False`` in ``score()``, ``score_per_subset()``,
  ``score_all_conditions()``: prevents obs column clobbering when called in loops.
- **Fix #3** — f-string bug in ``compare_conditions()`` verbose path: condition
  label was not interpolated correctly in the "no significant differences" message.
- **Fix #4** — Removed dead ``PROGENITOR_COLOR`` import in ``multiconditional.py``
  (was imported but never used, causing a linting warning).
- **Fix #5** — ``try/except/finally`` in ``embedding.py`` Strategy 1 cleanup:
  ensures temporary obs columns are removed even if an exception is raised.
- **Fix #6** — ``_needs_refit`` flag + improved ``_check_fitted()`` error message:
  raises a clear error if ``score()`` is called after ``refit_pseudotime()`` without
  calling ``fit()`` again.
- **Fix #7** — ``pct_fate`` / ``pct_progenitor`` columns from ``pts`` in
  ``get_deg_drivers()``: correctly extracts percent-expressed values from scanpy's
  ``rank_genes_groups`` output.
- **Fix #8** — ``__repr__`` on ``CommitmentScorer`` and ``MultiConditionScorer``:
  now shows root, branches, conditions, and status.
- **Fix #9** — ``statsmodels`` ImportError guard in ``plot_expression_trends()``:
  raises a clear error with install instructions when statsmodels is absent.
- **Fix #10** — ``save()`` / ``load()`` serialization on ``CommitmentScorer``:
  correctly round-trips all scorer state including ``_needs_refit``.
- **Fix #11** — Stratified ``bootstrap_cs()``: added ``stratified=`` and
  ``fate_cell_indices=`` parameters for stratified resampling within fate arms.
- **Fix #12** — ``_resolve_gene_sets()`` fuzzy year-suffix matching in
  ``enrichment.py``: handles Enrichr library names with year suffixes
  (e.g., ``KEGG_2021_Human`` vs ``KEGG_2019_Mouse``).
- **Fix #13** — ``TestMultiConditionScorer`` test class: 26 tests covering all
  ``MultiConditionScorer`` methods, bringing the total to 130 passing tests.

v0.5.0 (2026-03-27)
--------------------

**New module: multiconditional.py**

- ``MultiConditionScorer`` — new top-level class for multi-condition experiments.
  Builds a **shared** star embedding on pooled data from all conditions, ensuring
  arm geometry is identical across conditions and CS values are directly comparable.
  Wraps ``CommitmentScorer`` internally.

  *Tier 1 — Core multi-condition API*

  - ``build_embedding()`` / ``fit()`` — same interface as ``CommitmentScorer``,
    operates on pooled data.
  - ``score_all_conditions()`` — scores each condition separately using cell masks
    on the shared embedding. Returns ``dict[condition -> CommitmentScoreResult]``.
  - ``score_per_condition()`` — alias with pseudotime-aware documentation.
  - ``rebuild_embedding_with_subset_pseudotime()`` — delegates to the shared scorer.
  - ``plot_condition_star()`` — side-by-side star embedding panels, one per condition,
    with identical arm geometry and color scale.
  - ``transfer_labels()`` — writes per-condition commitment scores to full adata.

  *Tier 2 — Statistical comparison*

  - ``compute_delta_CS(condition_a, condition_b, n_bootstrap=500)`` — computes
    ΔCS = nCS_A − nCS_B with bootstrap confidence intervals (cell resampling
    within each condition). Returns full k×k delta matrix with CI bounds.
  - ``compare_conditions(results, test='auto')`` — statistical comparison of
    per-cell fate affinity scores across conditions. Permutation test for k=2
    conditions; Kruskal-Wallis + pairwise Mann-Whitney with Bonferroni correction
    for k>2. Returns tidy DataFrame with p-values and significance flags.
  - ``plot_condition_comparison(results, plot_type='violin')`` — violin/box/strip
    plots of per-cell fate affinity distributions split by condition, one panel
    per fate.

  *Tier 3 — Advanced*

  - ``fit_mixed_model(results, sample_key=None)`` — linear mixed-effects model
    on per-cell fate affinity scores (condition as fixed effect, sample/replicate
    as optional random effect) via ``statsmodels MixedLM``. Correct approach for
    datasets with multiple biological replicates per condition.
  - ``trajectory_shift(results, pseudotime_col='velocity_pseudotime_sub')`` —
    tests whether pseudotime distributions differ across conditions per fate arm.
    Computes KS statistic + p-value and Wasserstein distance with bootstrap CI.
    Answers: "do cells commit earlier/later under condition B?"
  - ``plot_trajectory_shift(shift_df)`` — KDE plots of pseudotime distributions
    per condition per fate arm, annotated with Wasserstein distance and KS p-value.

**Bug fixes**

- ``CommitmentScorer.score_per_subset()``: fixed cell mask misalignment.
  The mask was previously applied to ``self.adata.obs`` (full adata) but
  ``_vx``/``_vy`` are indexed to ``adata_sub``. Now correctly uses
  ``self.adata_sub.obs[subset_key]``.
- ``get_velocity_drivers()``: now computes **delta velocity** (fate arm mean
  minus progenitor mean) instead of raw arm mean. This removes genes
  constitutively active in the progenitor, highlighting fate-specific
  upregulation. New column ``delta_velocity`` added to output DataFrames;
  results are sorted by ``delta_velocity`` (descending).
- ``plot_expression_trends()``: added ``x_axis`` parameter (``'affinity'``,
  ``'pseudotime'``, ``'radial_distance'``). Previously the x-axis was always
  per-cell fate affinity but was misleadingly labeled. Now supports ordering
  cells by pseudotime or radial distance from origin in X_sccs.
- ``compute_cell_scores()``: added ``mag_weight=True`` and
  ``mag_threshold_pct=5.0`` parameters. Cells with near-zero velocity
  magnitude (typically progenitors at the origin) are now down-weighted
  toward the uniform distribution (1/k), reducing noise from near-stationary
  cells. Set ``mag_weight=False`` to restore original behavior.

**New features**

- ``CommitmentScorer.score(n_bootstrap=0, bootstrap_ci=0.95)`` — optional
  bootstrap confidence intervals on pairwise CS values. Resamples cells with
  replacement ``n_bootstrap`` times and returns empirical CI bounds stored in
  ``result.bootstrap_ci``. Shown in ``result.summary()`` when computed.
- ``bootstrap_cs(vx, vy, sectors, ...)`` — standalone bootstrap function
  exported from ``scores.py`` for advanced users.
- ``CommitmentScorer.transfer_labels(adata, result)`` — writes per-cell fate
  affinities, dominant fate, entropy, NN entropy, and subset pseudotime from
  ``adata_sub.obs`` back to the full adata. Cells outside the embedding subset
  receive NaN / 'unassigned'.
- ``CommitmentScorer.build_embedding(scale_metric=False)`` — new parameter.
  When ``True``, min-max scales the metric array to [0, 1] before embedding.
  For pseudotime, prefer ``rebuild_embedding_with_subset_pseudotime()`` instead.
- ``CommitmentScoreResult.bootstrap_ci`` — new optional field storing the
  bootstrap CI dict (keys: ``mean``, ``ci_low``, ``ci_high``, ``std``,
  ``n_bootstrap``, ``ci_level``).

**Pseudotime recomputation (from v0.4.x preview)**

- ``recompute_subset_pseudotime(adata_sub, adata_full, scale_01=True)`` —
  recomputes velocity pseudotime on the subset's induced velocity subgraph.
  Corrects the arm-coverage problem where full-adata pseudotime is compressed
  within the subset. Falls back to scanpy DPT, then radial distance.
- ``scale_metric_01(scores)`` — standalone min-max scaler for any metric.
- ``CommitmentScorer.recompute_subset_pseudotime(scale_01=True)`` — convenience
  wrapper.
- ``CommitmentScorer.rebuild_embedding_with_subset_pseudotime()`` — full
  pipeline: recompute → map back to full-adata indices → rebuild embedding.
  Resets ``_fitted=False``; call ``fit()`` again after.

**API changes**

- ``score_per_subset()`` now accepts ``n_bootstrap`` parameter.
- ``plot_expression_trends()`` ``x_axis`` parameter added (default ``'affinity'``
  preserves backward compatibility).
- ``get_velocity_drivers()`` output DataFrames now include ``delta_velocity``
  and ``progenitor_velocity`` columns in addition to ``mean_velocity``.
- Version bumped to ``0.5.0``.

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
