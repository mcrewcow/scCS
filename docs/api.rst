API guide
=========

Choose a scorer
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Scorer
     - Use
     - Main methods
   * - :class:`scCS.SingleScorer`
     - One supervised root-to-fates analysis
     - ``build_embedding``, ``fit``, ``score``, ``plot_star``
   * - :class:`scCS.PairScorer`
     - Two independent conditions
     - ``score_all_conditions``, ``replicate_table``, ``compare_conditions``
   * - :class:`scCS.MultiScorer`
     - Three or more independent conditions
     - ``compare_omnibus``, ``compare_posthoc``, ``compare_contrast``

Single-population workflow
--------------------------

.. code-block:: python

   import scCS

   scorer = scCS.SingleScorer(
       adata, root="Progenitor", branches=["Fate A", "Fate B"], obs_key="cell_type"
   )
   scorer.build_embedding(ordering_metric="pseudotime")
   scorer.fit(scoring_mode="future_fate")
   result = scorer.score()
   scorer.plot_star(color_by="conditional_fate_affinity:Fate A")

``fit`` uses an existing velocity graph or a supplied ``transition_matrix``.
Prepare velocity in the original expression manifold; see :doc:`quickstart`.
Use ``result.affinity_defined`` when summarizing conditional affinity and report
reach and coverage separately. Scientific definitions are in :doc:`metrics`.

Condition workflow
------------------

Use the same build and fit steps with a condition scorer, providing
``condition_obs_key`` and ``replicate_obs_key`` at construction. Then:

.. code-block:: python

   results = pair.score_all_conditions()
   replicates = pair.replicate_table(results)
   comparison = pair.compare_conditions(
       results, metric="conditional_fate_affinity", fate="Fate A"
   )

Formal inference uses biological replicates. Undefined replicate outcomes raise
by default; explicit exclusions are reported. See :doc:`condition_inference`.

Common follow-up tasks
----------------------

* Diagnostics: ``preflight()``, ``result.anchor_diagnostics_frame()``.
* Sensitivity: ``scorer.sensitivity_analysis()`` for future-fate horizon/anchor grids.
* Figures: ``plot_star_panels()``, ``plot_commitment_heatmap()``, ``plot_rose()``.
* Expression: ``plot_gene_expression_star()``, ``get_commitment_associated_genes()``.
* Persistence: ``save()`` and ``SingleScorer.load()``.

All supported methods, parameters, result fields, compatibility aliases, and
low-level solvers remain available in :doc:`api_advanced`.
