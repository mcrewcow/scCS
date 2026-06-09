Changelog
=========

v0.7.4 (2026-06-08)
-------------------

**Fixed**

- ``scCS.embedding.build_star_embedding`` — the ``(s_min, s_max)`` range
  used to map the ordering metric onto each arm is now computed from
  **fate cells only**, excluding the bifurcation/progenitor cells.
  Previously the range was taken across ``fate_mask | bif_mask`` in
  per-arm mode (and across the whole subset in global mode), which let
  bifurcation cells — typically clustered at the low end of the
  pseudotime — push the lower edge of the rescale interval down and
  shift the closest fate cell off the origin. After this fix, the
  earliest fate cell on each arm (per-arm mode) or the earliest fate
  cell anywhere in the subset (global mode) sits at radius ≈ 0, which
  matches the published star-plot semantics. Bifurcation cells continue
  to be clustered tightly around the origin via the separate
  jitter-around-origin logic and are unaffected by this change.

  *Visible effect on real data*: on the pancreas tutorial subset
  (Pre-endocrine → Alpha/Beta/Delta/Epsilon), v0.7.3 placed every fate
  cluster at least 2.7 radial units away from the origin even with
  ``arm_norm="per_arm"``; v0.7.4 places the earliest fate cell on each
  arm within ~0.5 units of origin in per-arm mode and within ~0.05
  units of origin for the global-min fate (Delta) in global mode.

  Multi-condition embeddings (``PairScorer`` / ``MultiScorer``) share
  one rescale across the full subset, then each condition panel in
  ``plot_star_grid`` shows only that condition's cells. If a
  particular condition does **not** contain the cell with the
  global-min pseudotime, its panel will show all arms starting
  slightly off origin — this is the intended behavior for global
  rescale and reflects the biological fact that the condition is
  enriched in later-pseudotime cells. Use ``arm_norm="per_arm"`` if
  you instead want every condition's earliest cells to land at
  origin (at the cost of losing inter-condition timing comparison).

- All three tutorials — the ``sc.pp.highly_variable_genes`` fallback in
  the scvelo HVG path now uses ``flavor="cell_ranger"`` instead of the
  default ``flavor="seurat"``. The seurat flavor passes an integer
  ``n_bins`` to ``pandas.cut`` on the log-dispersion vector, which
  rejects integer bin counts when the input contains ``±inf`` from
  pandas 2.2 onward. Genes with zero mean expression produce
  ``-inf`` log-dispersions and trigger this error on Python 3.12
  + pandas ≥ 2.2 environments. The ``cell_ranger`` flavor uses explicit
  bin edges that already include ``±inf`` and is robust across pandas
  versions. The IF-branch (``scv.pp.filter_and_normalize`` with
  ``n_top_genes=2000``) is unchanged and still preferred when the
  installed scvelo supports it.

**Tests**

- Added ``tests/test_v074_fate_only_rescale.py`` (+6 tests):

  - ``TestPerArmFateOnly`` — every fate's closest cell touches origin
    in ``arm_norm="per_arm"`` mode; bifurcation cells remain near
    origin (regression test for the inner cluster).
  - ``TestGlobalFateOnly`` — at least one fate touches origin in
    ``arm_norm="global"`` mode (the fate containing the global-min
    pseudotime cell); the longest-range fate reaches ``arm_scale``;
    explicit regression check against the v0.7.3 fate+bif rescale
    behavior on a fixture where bifurcation cells are pseudotime-disjoint
    from fate cells.
  - ``TestScveloHvgFallbackCellRanger`` — synthetic adata with all-zero
    genes (the failure mode for the seurat flavor on pandas ≥ 2.2)
    successfully runs through ``flavor="cell_ranger"``.

  Test count: 183 passed, 1 skipped (was 177 + 1 in v0.7.3).

v0.7.3 (2026-06-08)
-------------------

**Added**

- ``arm_norm`` keyword on ``scCS.embedding.build_star_embedding`` and on
  ``SingleScorer.build_embedding``, ``PairScorer.build_embedding``,
  ``MultiScorer.build_embedding``, plus the matching ``refit_pseudotime``
  wrappers. Accepts ``"global"`` (new default) or ``"per_arm"`` (legacy
  behavior). See *Changed* below for the rationale.
- ``vmin`` / ``vmax`` keyword arguments on
  ``MultiScorer.plot_omnibus_summary``. When both are ``None`` (default),
  the colormap limits are derived from the finite values of the
  mean-affinity matrix; pass explicit floats to pin a fixed scale across
  figures.

**Changed**

- ``build_star_embedding`` now rescales the ordering metric **globally**
  across the entire subset by default (``arm_norm="global"``). Previously
  each arm received its own ``(s_min, s_max)`` from
  ``fate_mask | bif_mask`` and was mapped to ``[0, arm_scale]``
  independently, so all arms reached the full radial cap regardless of
  the underlying pseudotime range. The new default uses one
  ``(s_min, s_max)`` from all subset cells and applies it uniformly, so
  arms whose cells span shorter pseudotime intervals stay visibly
  shorter. This preserves the relative ordering of cells across arms and
  matches the biological intuition that arm length reflects how far each
  fate has differentiated from the progenitor on a shared scale. Pass
  ``arm_norm="per_arm"`` to reproduce pre-v0.7.3 plots.
- ``MultiScorer.plot_omnibus_summary`` no longer pins the mean-affinity
  heatmap to ``[0, 1]``. The colormap now spans the realized data range
  by default, so per-condition contrast is visible on datasets where
  affinities cluster well below 1.0 (the previous behavior rendered
  nearly uniform pale yellow on real data). The colorbar label includes
  the realized ``[vmin, vmax]`` so the scale stays explicit.
- ``scCS.enrichment.run_enrichment_per_fate`` — ``fate_names`` is now an
  optional second argument. If omitted, fate names are inferred from
  ``deg_drivers.keys()`` in their natural insertion order. If provided
  but mismatched with ``deg_drivers.keys()``, a ``UserWarning`` is
  emitted and only the intersection is used. The previous positional
  contract is preserved for callers (including
  ``SingleScorer.get_enrichment``) that pass it explicitly.

**Fixed**

- ``scCS_tutorial_pairwise.ipynb`` and ``scCS_tutorial_multi.ipynb`` —
  ``run_enrichment_per_fate`` calls now pass ``fate_names=fate_names``
  explicitly. Previously the missing positional argument raised
  ``TypeError`` which was silently swallowed by the surrounding
  ``except Exception`` clause, leaving the enrichment table empty with
  no clear error message.
- All three tutorials — ``scv.pp.filter_and_normalize`` is now invoked
  through an ``inspect.signature``-based guard that uses
  ``n_top_genes=2000`` when the installed scvelo supports the keyword,
  and falls back to ``sc.pp.highly_variable_genes(adata, n_top_genes=2000,
  subset=True)`` otherwise. This keeps the notebooks runnable across
  scvelo releases where the keyword has been removed.

- ``MultiScorer.plot_omnibus_summary`` layout cleanup —
  the default figsize is widened so the colorbar label (which now
  carries the auto-derived range) and the fate row labels render without
  overlap or truncation, and y-tick labels are explicitly set to
  ``rotation=0`` on both panels.

**Tutorial hygiene**

- Uniform warning filters added to ``scCS_tutorial_pairwise.ipynb`` and
  ``scCS_tutorial_single.ipynb``: ``DeprecationWarning``,
  ``FutureWarning``, and ``statsmodels.ConvergenceWarning`` are
  suppressed. The previous blanket ``warnings.filterwarnings("ignore")``
  in ``scCS_tutorial_multi.ipynb`` is narrowed to the same category set,
  so ``UserWarning`` emitted by scCS itself (e.g. the
  ``plot_expression_trends`` slice notice introduced in v0.7.2) stays
  visible.

**Tests**

- Added ``tests/test_arm_norm_and_enrichment.py`` covering the new
  ``arm_norm`` branches in ``build_star_embedding``, the
  ``fate_names`` inference path in ``run_enrichment_per_fate``, and
  the auto-scale defaults of ``plot_omnibus_summary``.

v0.7.2 (2026-06-07)
-------------------

**Bug fixes — correctness**

- ``MultiScorer.compare_omnibus``, ``MultiScorer.compare_posthoc``, and
  ``MultiScorer.fit_mixed_model_contrasts`` previously compared identical
  full-embedding ``cell_scores`` arrays across all conditions because
  ``SingleScorer.score(cell_mask=...)`` returns ``cell_scores`` sized to
  the **full** ``adata_sub`` regardless of the mask (a semantic the
  ``transfer_labels`` pipeline depends on). The downstream multi-condition
  consumers wrongly assumed condition-only sizing. They now correctly
  slice ``cell_scores`` per condition via the new
  ``MultiScorer._per_condition_cell_scores`` helper, so omnibus,
  posthoc, and LMM analyses operate on the actual per-condition cell
  distributions. Same fix applied to ``plot_omnibus_summary``'s
  mean-affinity matrix and to ``plot_rose_grid``'s long-form table.

- ``MultiScorer.plot_pairwise_delta_grid`` — previously called
  ``plot_delta_cs_heatmap(delta_result, ax=ax)`` but
  ``plot_delta_cs_heatmap`` does not accept ``ax`` (it constructs its
  own ``Figure``). Heatmap rendering is now inlined in the grid loop so
  the grid renders correctly. Added a ``cmap`` keyword (default
  ``"RdBu_r"``) on the grid signature.

- ``embedding._fallback_dpt`` — previously called ``sc.tl.dpt`` without
  first running ``sc.tl.diffmap``, causing scanpy to silently use a
  default-parameter diffmap and producing ``inf`` pseudotime values on
  disconnected subgraph components. Now explicitly runs
  ``sc.tl.diffmap(n_comps=15)`` (refitting neighbors on ``X_sccs`` if
  needed) before DPT and clips any remaining non-finite values to the
  finite range. This makes ``MultiScorer.refit_pseudotime()`` safe on
  velocity-tertile and other split datasets.

- ``embedding._fill_nan`` — generalized to handle ``±inf`` in addition
  to ``NaN`` (positive infinities clip to the finite max, negative
  infinities to the finite min, and NaNs to the finite median).

- ``plot_expression_trends`` — now gracefully handles a common usage
  pattern where the caller passes a condition-masked subset of the
  scored ``adata`` together with a full-embedding
  ``CommitmentScoreResult``. Previously this raised
  ``KeyError: '... are not valid obs/var names or indices'`` because
  ``result.cell_obs_names`` referenced cells outside the passed
  ``adata``. The function now intersects ``result.cell_obs_names`` with
  ``adata.obs_names``, slices ``cell_scores`` to keep only the
  overlapping rows, and emits a ``UserWarning`` indicating how many
  cells were kept. Calls using the legacy patterns (full ``adata``
  paired with full ``result``, or a manually pre-sliced ``result``) are
  unchanged and produce no warning.

**Tutorial fixes**

- ``scCS_tutorial_multi.ipynb`` — corrected several API calls that
  broke under v0.7.1:

  - ``MultiScorer.build_embedding`` now uses
    ``ordering_metric="velocity_pseudotime"`` (the value
    ``"pseudotime"`` is no longer accepted by the embedding builder
    when velocity is available).
  - ``compare_posthoc`` now passes ``pval_correction=...`` (not
    ``correction=...``).
  - ``plot_expression_trends`` is now called with a per-condition
    ``CommitmentScoreResult`` constructed via ``dataclasses.replace``
    that slices ``cell_scores`` / ``cell_obs_names`` /
    ``nn_cell_entropy`` to the condition's mask. This works around the
    full-embedding ``cell_scores`` sizing without breaking
    ``transfer_labels``.
  - ``MultiScorer.transfer_labels(results, prefix=...)`` no longer
    takes an ``adata`` positional argument.
  - ``MultiScorer.compute_pairwise_deltas()`` no longer takes a
    ``results`` argument (it reads cached scores internally).
  - Final UMAP visualization uses ``mscorer.adata`` (the scorer
    re-exposes its working AnnData).
  - Added a markdown caveat that the velocity-tertile split is
    illustrative, not a real biological perturbation; fate prevalence
    is imbalanced across tertiles.

- ``scCS_tutorial_pairwise.ipynb`` — ``_driver_overlap`` helper made
  defensive against asymmetric fate sets across conditions: uses
  ``dict.get(fate)`` with empty-set fallback and unions fates across
  all conditions instead of intersecting on the first one (which raised
  ``KeyError`` when one condition had no top drivers for a given fate).

- ``scCS_tutorial_single.ipynb`` — removed an unsupported
  ``verbose=False`` keyword argument from the
  ``scorer.get_deg_drivers(...)`` call (the method does not accept
  ``verbose``; previously raised ``TypeError`` on first run).

- All three tutorial notebooks now start with ``%matplotlib inline``
  so that figures render correctly when the notebooks are executed
  headlessly (``jupyter nbconvert --execute``).

**Tests**

- Updated 6 stale tests under ``TestPairScorerPipeline`` /
  ``TestMultiScorerPipeline`` to match the v0.7.x APIs:

  - ``test_compare_conditions_kruskal_path`` skipped — ``PairScorer``
    now enforces exactly 2 conditions.
  - ``test_single_condition_raises`` regex updated to
    ``"exactly 2"``.
  - ``test_validation_rejects_2_conditions`` regex updated to
    ``"at least 3"``.
  - ``test_omnibus_anova`` uses
    ``df["test"].str.contains("anova")`` instead of equality on
    ``df["test"].values``.
  - Three plot tests now import ``matplotlib.pyplot as plt`` locally.

- Suite size: 168 passed, 1 skipped.

v0.7.1 (2026-06-07)
-------------------

**Bug fixes**

- ``plot_star(color_by="entropy"|"cs_entropy")`` — colorbar limits now
  auto-scale to the data range instead of being hardcoded to ``[0, 1]``,
  so plots dominated by high entropy no longer appear uniformly red.
  Added new ``vmin``, ``vmax``, and ``cmap`` keyword arguments to
  ``plot_star_embedding()`` (and forwarded through
  ``SingleScorer.plot_star`` / ``PairScorer.plot_star`` /
  ``MultiScorer.plot_star``) for users who want to pin limits across
  figures or change the colormap.

- ``plot_star(color_by="nn_entropy"|"cs_nn_entropy")`` — previously fell
  through to the generic numeric branch and looked up the missing column
  ``nn_entropy`` (actual column is ``cs_nn_entropy``), producing a
  fully gray scatter. Now uses a dedicated branch that reads the correct
  column and emits a clear warning when ``score(k_nn=...)`` was not run.

- ``plot_star(color_by="pseudotime")`` — added explicit support
  (reads ``sccs_pseudotime`` with fallback to ``velocity_pseudotime``).
  Removed unsupported ``"cytotrace"`` claim from the docstring.

- ``plot_subset_comparison()`` — subsets containing only progenitor cells
  produce ``pairwise_nCS = inf`` for cross-fate pairs; matplotlib was
  silently dropping these bars, leaving an empty plot. The bars are now
  rendered as gray-hatched placeholders at a small fraction of the finite
  maximum with an "inf" annotation, and a ``UserWarning`` listing the
  affected subsets is emitted.

**Tutorial improvements**

- Tutorial notebooks (single, pairwise, multi) now use long-form tidy
  DataFrames for driver and enrichment displays. New helpers
  ``_stack_drivers``, ``_stack_enrichment``, ``_stack_drivers_by_condition``,
  ``_stack_enrichment_by_condition``, and ``_driver_overlap`` are defined
  inline in each tutorial's setup cell. One ~20-row table replaces the
  previous per-fate / per-condition print loops.

- ``scCS_tutorial_pairwise.ipynb`` extended with full downstream sections
  matching the single-condition tutorial:

  - §6 Driver genes per condition (velocity + DEG drivers; per-condition
    masking of ``adata_sub``; driver overlap table across conditions)
  - §7 Pathway enrichment per condition (offline-safe via try/except)
  - §8 Expression trends along fate arms — per condition
    (``plot_expression_trends`` side by side)

- ``scCS_tutorial_multi.ipynb`` rebuilt on the scVelo pancreas dataset
  split into three RNA-velocity-magnitude tertiles
  (``low_velocity`` / ``med_velocity`` / ``high_velocity``) instead of
  synthetic random data. Includes all three tiers of statistical
  comparison plus per-condition drivers (§10), enrichment (§11),
  expression trends (§12), and ``transfer_labels`` + UMAP (§13).

**Tests**

- Added ``TestPlotStarAutoScale`` to ``tests/test_scores.py`` covering:

  - ``test_plot_star_entropy_autoscale`` — asserts that the colorbar
    ``norm.vmin``/``vmax`` track the per-cell entropy data range.
  - ``test_plot_star_entropy_explicit_range`` — asserts that explicit
    ``vmin``/``vmax`` kwargs are honored verbatim.
  - ``test_plot_star_nn_entropy_renders`` — asserts a real norm (not the
    gray-fallback) is built for ``color_by="nn_entropy"``.
  - ``test_plot_subset_comparison_inf_handling`` — asserts the
    ``UserWarning`` is emitted, at least one bar is hatched, and the
    "inf" annotation is added.

v0.7.0 (2026-06-06)
--------------------

**Breaking changes — class rename**

- ``CommitmentScorer`` → ``SingleScorer`` (hard rename, no alias)
- ``MultiConditionScorer`` → ``PairScorer`` (hard rename, no alias)
- All references to old names removed from code, docs, and notebooks

**New module: multicomparison.py**

- ``MultiScorer`` — new top-level class for experiments with 3+ conditions.
  Validates >= 3 conditions at init; suggests PairScorer for 2 conditions.

  *Tier 2 — Omnibus + post-hoc statistical comparison*

  - ``compare_omnibus(results, test='kruskal')`` — omnibus test across all
    conditions per fate arm. Supports Kruskal-Wallis (non-parametric) and
    one-way ANOVA (parametric). Returns tidy DataFrame with per-fate
    statistics and adjusted p-values.
  - ``compare_posthoc(results, method='dunn', pval_correction='fdr')`` —
    post-hoc pairwise comparisons per fate arm. Supports Dunn's test,
    Tukey HSD, and Conover-Iman test. Multiple testing correction via
    FDR (Benjamini-Hochberg), Bonferroni, or Holm. Optionally filters
    to fates where omnibus test was significant.
  - ``compute_pairwise_deltas(n_bootstrap=500)`` — ΔCS with bootstrap CI
    for ALL condition pairs (not just one pair like PairScorer).
  - ``fit_mixed_model_contrasts(results, contrasts=None)`` — LMM with
    custom condition contrasts via Wald tests.

  *New visualizations*

  - ``plot_omnibus_summary()`` — fates × conditions heatmap with omnibus
    p-value annotation.
  - ``plot_posthoc_heatmap()`` — condition × condition post-hoc p-value
    heatmap per fate arm.
  - ``plot_pairwise_delta_grid()`` — grid of ΔCS heatmaps for all pairs.

**New documentation**

- ``mathematical_framework.rst`` — dedicated page with full LaTeX derivations
  of the scCS scoring framework, entropy metrics, and statistical tests.
- ``scCS_tutorial_multi.ipynb`` — new tutorial notebook for MultiScorer
  with 3+ conditions.
- Expanded ``introduction.rst`` with three-scorer decision flowchart.
- Restructured ``api.rst`` organized by scorer class.

**New dependency**

- ``scikit-posthocs>=0.8`` — required for Dunn's test and Conover-Iman
  post-hoc comparisons in MultiScorer.

**Internal**

- ``_base.py`` — new module with ``_BaseScorer`` abstract class extracting
  shared initialization and embedding logic from SingleScorer.
- ``single.py`` — renamed from ``trajectory.py``.
- ``pairwise.py`` — renamed from ``multiconditional.py``.
- ``CONDITION_PALETTE`` extended to 12 colors for 3+ condition support.
- ``conf.py`` — added ``sphinx.ext.mathjax`` for LaTeX rendering.

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
