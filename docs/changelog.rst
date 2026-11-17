Changelog
=========

0.8.1rc2
---------

* Fail closed on undefined replicate outcomes, with explicit audited exclusions.
* Report undefined bootstrap draws and exact-test p-value limits.
* Count verified complementary two-fate hypotheses once in replicate tests.
* Add graph dead-end/reachability diagnostics and future-fate sensitivity summaries.
* Fix optional dependency cache signatures and centralize plot metric aliases.
* Respect disabled AnnData subset caching and offer compact export metadata.
* Simplify the API guide, publish executed tutorials, and correct benchmark status reporting.
* Exclude generated artifacts from distributions.

See :doc:`release_notes` for behavior changes and migration guidance.

0.8.1rc1 — 2026-11-16
=====================

Cell-ID-safe export and reload; tie-invariant rank progression; explicit undefined
conditional affinities; valid-cell-aware summaries and gene associations; shared
condition colors; restored graph/mode state; complete scientific provenance;
mode-aware aliases; solver validation; compatible tutorial environment, versioned
caches and twelve refreshed notebooks. See :doc:`release_notes` for current migration guidance.

**Behavior change from development builds:** undefined conditional affinity,
entropy and specificity are NaN. Conditional summaries require a defined-cell
mask and coverage reporting.

0.8.0.dev34
-----------

- Preserved the original RegVelo neighbor graph in the final Schwann
  deterministic/dynamical reproduction gate.
- Added anchor-quantile stability and cell-ID-aligned comparisons to the final
  package decision notebook.
- Strengthened the controlled Schwann PairScorer and MultiScorer demonstrations
  so their predeclared ChC CFA effects are clearer.
- Added public metric metadata (`metric_public`, `metric_label`) to condition
  inference tables.
- Separated downstream tutorial output directories.
- Added an exact no-chunk 200-million-cell execution profile, a degree-30
  measured ladder, and high-memory command-line/Slurm runners.


* Added a first-time-user Introduction and a full Mathematical framework page.
* Expanded the DFFP derivation, metric semantics, Signed Ordering Flux, anchor rules, and instantaneous-mode equations.
* Expanded methodology tutorials with geometric-stopping, identity-versus-motion, and branch-angle visualizations.
* Added a validation-evidence page based on the final preserved-graph pancreas and Schwann runs.
* Embedded the final executed tutorial suite; scientific calculations are unchanged from dev33.




0.8.0.dev32
-----------

* named the graph-based method Discounted Future-Fate Propagation (DFFP);
* added documentation aliases for CFA, DFR, RC, UFP, and SOF;
* made condition radar plots replace rectangular placeholder axes automatically;
* expanded method-selection and complex-branch tutorials with visual examples;
* added pancreas and Schwann downstream-analysis tutorials;
* added a single-process, in-memory, no-chunking scalability notebook with
  targets through 200 million cells;
* fixed the Schwann package-decision notebook so inverse CytoTRACE is prepared
  reproducibly from the public RegVelo dataset.


0.8.0.dev31
------------

* Use dynamical RNA velocity as the primary model in every RegVelo Schwann tutorial.
* Rework the six dataset/scorer tutorials to load public datasets directly and
  recompute velocity without hidden tutorial caches.
* Remove ``try``/``except`` control flow from public tutorial code and add
  explanatory notes for first-time users.
* Make display-star velocity grids explicitly use scVelo's embedding projection.
* Replace the misleading pooled terminal display rose with two explicit views:
  a root fate-directed branch rose and branch-relative terminal velocity profiles.
  The latter keep every cell attached to its annotated branch, so inward or curved
  vectors cannot be relabeled as the nearest different fate sector.
* Add ``SingleScorer.plot_branch_velocity_profiles`` and
  ``population="terminal"`` support to ``SingleScorer.plot_rose``.
* Make all tutorial star grids and roses use a consistent scVelo display projection;
  display-only RNA-velocity figures remain separate from future-fate scores.
* Make future-fate metadata fully H5AD-serializable by storing anchor diagnostics
  as a mapping keyed by fate.
* Add H5AD export and terminal-rose regression tests.

0.8.0.dev30
------------

* Added six dataset-specific public tutorials: SingleScorer, PairScorer, and
  MultiScorer guides for pancreas and RegVelo Schwann.
* Expanded tutorials from dataset loading and velocity fitting through ordering
  validation, future-fate diagnostics, visualization, sensitivity analysis,
  replicate-aware inference, gene-expression analysis, and export.
* Added clearly separated real-study and controlled demonstration paths for
  PairScorer and MultiScorer.
* Added future-fate aliases to gene-association outcomes and cell-level heatmaps.
* Kept display-space velocity grids and rose plots on independent instantaneous
  scorers so they cannot alter or masquerade as future-fate calculations.
* Reorganized Read the Docs tutorials by dataset and scorer.

0.8.0.dev29
-------------

* Expose ``FutureFateScoreResult.signed_progression`` as the documented public
  alias of ``progression_velocity``.
* Expose ``FutureFateScoreResult.selected_path_coverage`` as the documented
  public alias of the retained one-step transition mass.
* Add regression coverage for both result-level semantic aliases.

0.8.0.dev28
------------

* Added discounted future-fate scoring on the original RNA-velocity graph.
* Added direct and iterative discounted hitting solvers.
* Added explicit optional competing outcomes and unresolved probability.
* Added future-fate affinity, reach, entropy, specificity, reach-supported
  specificity, and signed progression.
* Added compatibility aliases for PairScorer and MultiScorer metrics.
* Added anchor and transition-normalization diagnostics.
* Prevented scientific velocity-vector plots in future-fate mode.
* Rebuilt README, documentation, and tutorials around explicit method semantics.
* Retained instantaneous transition pushforward as a separate scoring mode.

0.8.0.dev27
------------

* Completed the validated instantaneous supervised-geometry implementation.
* Added native scVelo display-star projection, binned rose plots, gene-expression
  stars, projection reconstruction diagnostics, and comprehensive pancreas and
  Schwann validation.

