API Reference
=============

This page documents all public classes and functions in scCS.
Full source-linked documentation is also available in the **autoapi** section
in the sidebar.

CommitmentScorer
----------------

The main entry point. Wraps an AnnData object and exposes all scoring
and plotting methods.

.. autoclass:: scCS.CommitmentScorer
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

Embedding
---------

.. autofunction:: scCS.build_star_embedding

Fate Detection
--------------

.. autofunction:: scCS.build_fate_map
