API reference
=============

Core scorers
------------

.. autoclass:: scCS.SingleScorer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scCS.PairScorer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: scCS.MultiScorer
   :members:
   :undoc-members:
   :show-inheritance:

Future-fate engine
------------------

.. autoclass:: scCS.FutureFateScoreResult
   :members:

.. autoclass:: scCS.DiscountedOutcomeSolution
   :members:

.. autofunction:: scCS.score_future_fate

.. autofunction:: scCS.solve_discounted_outcomes

.. autofunction:: scCS.canonicalize_transition_matrix

Supervised geometry and ordering
--------------------------------

.. autoclass:: scCS.Furcation
   :members:

.. autoclass:: scCS.FurcationOrderingScaler
   :members:

.. autofunction:: scCS.build_scoring_embedding

Instantaneous engine
--------------------

.. autoclass:: scCS.FurcationScoreResult
   :members:

.. autofunction:: scCS.project_transition_velocity

.. autofunction:: scCS.cosine_softmax_affinity
