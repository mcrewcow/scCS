Condition inference
===================

PairScorer and MultiScorer fit one pooled model and summarize outcomes by
biological replicate. The future-fate aliases can be used directly:

.. code-block:: python

   pair.compare_conditions(
       metric="future_fate_affinity",
       fate="ChC",
   )

   pair.compare_conditions(metric="future_fate_reach")
   pair.compare_conditions(metric="future_fate_specificity")
   pair.compare_conditions(metric="reach_supported_specificity")
   pair.compare_conditions(metric="signed_progression")

Replicate-first inference
-------------------------

Formal tests use replicate summaries, exact or Monte Carlo label permutations,
and hierarchical bootstrap where requested. Cells are not treated as independent
biological replicates.

Transition scope
----------------

Pooled transition graphs remain the default. Condition- or replicate-blocked
graphs are sensitivity analyses because blocking can remove valid manifold
neighbors. The selected scope must be reported.

Design scope
------------

Independent-group designs are supported. Paired and repeated-measure designs are
not silently approximated.
