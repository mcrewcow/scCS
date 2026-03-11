Changelog
=========

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
