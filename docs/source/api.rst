API Reference
=============

This page documents all public classes and functions in scCS.
Full source-linked documentation is also available in the **autoapi** section
in the sidebar.

CommitmentScorer
----------------

The main entry point for single-condition analysis. Wraps an AnnData object
and exposes all scoring and plotting methods.

.. autoclass:: scCS.CommitmentScorer
   :members:
   :undoc-members:
   :show-inheritance:

MultiConditionScorer
--------------------

Multi-condition extension. Builds a shared star embedding on pooled data,
then scores each condition separately. Provides statistical comparison,
mixed-effects modeling, and trajectory shift analysis.

.. autoclass:: scCS.MultiConditionScorer
   :members:
   :undoc-members:
   :show-inheritance:

CommitmentScoreResult
---------------------

Dataclass returned by ``scorer.score()``. Contains all computed scores,
matrices, and metadata.

.. autoclass:: scCS.CommitmentScoreResult
   :members:
   :undoc-members:
   :show-inheritance:

FateMap
-------

Stores the fate topology: which cells belong to which arm, centroids,
and arm angles.

.. autoclass:: scCS.FateMap
   :members:
   :undoc-members:
   :show-inheritance:

Plotting Functions
------------------

All plotting functions accept a ``color_map`` dict (fate name → hex color)
to preserve your original scanpy/Seurat cluster colors.

.. autofunction:: scCS.plot_star_embedding
.. autofunction:: scCS.plot_star_panels
.. autofunction:: scCS.plot_commitment_bar
.. autofunction:: scCS.plot_expression_trends
.. autofunction:: scCS.plot_rose
.. autofunction:: scCS.plot_pairwise_cs
.. autofunction:: scCS.plot_commitment_heatmap
.. autofunction:: scCS.plot_subset_comparison
.. autofunction:: scCS.plot_nn_entropy_elbow

Embedding
---------

.. autofunction:: scCS.build_star_embedding
.. autofunction:: scCS.recompute_subset_pseudotime
.. autofunction:: scCS.scale_metric_01

Fate Detection
--------------

.. autofunction:: scCS.build_fate_map

Core Math — Entropy
-------------------

.. autofunction:: scCS.compute_population_entropy
.. autofunction:: scCS.compute_mean_cell_entropy
.. autofunction:: scCS.compute_per_fate_cell_entropy
.. autofunction:: scCS.compute_nn_cell_entropy

Core Math — Scores
------------------

.. autofunction:: scCS.compute_unCS
.. autofunction:: scCS.compute_nCS
.. autofunction:: scCS.compute_commitment_vector
.. autofunction:: scCS.compute_pairwise_cs_matrix
.. autofunction:: scCS.compute_cell_scores
.. autofunction:: scCS.compute_magnitudes
.. autofunction:: scCS.compute_angles
.. autofunction:: scCS.bin_angles
.. autofunction:: scCS.equal_sectors
.. autofunction:: scCS.centroid_sectors
.. autofunction:: scCS.compute_sector_magnitudes
.. autofunction:: scCS.bootstrap_cs
