Quick start
===========

This page shows the minimum future-fate workflow. Read :doc:`introduction` for
conceptual orientation and :doc:`mathematical_framework` for the equations.

Prepare the data
----------------

The AnnData object must contain annotations, a continuous ordering, and a
velocity graph. Inspect the native embedding and velocity field before fitting
scCS.

Fit Discounted Future-Fate Propagation
--------------------------------------

.. code-block:: python

   import scCS

   scorer = scCS.SingleScorer(
       adata,
       root=("Ngn3 high EP", "Pre-endocrine"),
       branches=["Alpha", "Beta", "Delta", "Epsilon"],
       obs_key="clusters",
   )

   report = scorer.preflight(ordering_metric="latent_time")
   print(report.to_frame())

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

Read the result
---------------

.. code-block:: python

   scorer.plot_star(result, color_by="future_fate_affinity:Beta")
   scorer.plot_star(result, color_by="future_fate_reach")
   scorer.plot_star(result, color_by="future_fate_specificity")
   scorer.plot_star(result, color_by="reach_supported_specificity")
   scorer.plot_star(result, color_by="signed_progression")

CFA answers which future is favored. DFR answers how much probability reaches a
supplied fate. FFS answers how decisive the distribution is. SOF independently
reports forward or retrograde motion.

Check sensitivity
-----------------

Repeat the fit across plausible effective horizons and anchor quantiles. Inspect
``result.anchor_diagnostics_frame()`` and compare per-cell CFA rather than only
population averages.

Instantaneous mode
------------------

.. code-block:: python

   scorer.fit(scoring_mode="instantaneous")
   instantaneous = scorer.score()
   scorer.plot_direction_strength_map(instantaneous)
   scorer.plot_rose(instantaneous)

Instantaneous mode measures local direction in the supervised geometry. It is
not a substitute for DFFP future probabilities.
