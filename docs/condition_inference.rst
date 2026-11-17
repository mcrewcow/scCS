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

Undefined outcomes and coverage
-------------------------------

A replicate with no defined conditional affinity is retained in descriptive
summaries. Formal inference raises by default rather than silently excluding it.
This applies to pairwise, omnibus, post-hoc, planned-contrast, mixed-model and
hierarchical-bootstrap methods. When scientifically justified, opt in explicitly:

.. code-block:: python

   comparison = pair.compare_conditions(
       metric="conditional_fate_affinity", fate="ChC",
       undefined_replicates="drop",
   )
   comparison[["n_replicates_requested", "n_replicates_included",
               "excluded_replicates", "replicate_diagnostics"]]

The resulting comparison is conditional on eligible replicates. Exclusion can
select for samples that differentiated; report the exclusion and interpret it
alongside DFR and defined-affinity coverage, which retain zero-reach replicates.
An undefined CFA is not zero preference. Graph dead ends can reflect biology,
modeling or data quality; they alone do not demonstrate differentiation arrest.
Every tested condition must retain at least two eligible replicates.
Descriptive replicate scatter/distribution plots warn about undefined values
and show defined/requested replicate counts when values cannot be plotted.

For conditional CFA, entropy and specificity, within-replicate bootstraps sample
from defined cells. If another outcome (such as composition) becomes undefined
in a resampled draw, the bootstrap raises. ``undefined_bootstrap="report"`` returns
NaN interval bounds, ``n_bootstrap_undefined`` and ``bootstrap_status`` instead.
No undefined draw is discarded or redrawn. ``n_bootstrap`` counts all requested draws.

Hypothesis families and p-value resolution
------------------------------------------

Holm correction applies to the outcome family returned by each comparison call.
Separate calls do not provide a study-wide adjustment; predefine the family of claims.
For exactly two fates, verified complementary CFA or composition endpoints on the
same replicate observations represent one two-sided hypothesis. Both fate rows
are retained with a shared ``hypothesis_id`` and p-value. Raw fate probabilities
and outcomes with more than two fates remain separate hypotheses. Post-hoc tests
also distinguish condition pairs. The returned ``hypothesis_family_size`` and
``multiplicity_method`` describe the correction. Mixed-model sensitivity tests
retain separate endpoint corrections.

Exact pairwise tests report ``minimum_attainable_pvalue`` and warn when that
lower bound exceeds ``alpha`` (default 0.05). For equal independent groups of
three replicates, the two-sided mean-difference test has 20 allocations and a
minimum p-value of 0.10 before multiplicity correction. Ties can increase the
minimum. Monte Carlo results report the finite-resampling resolution instead;
this is not a power calculation. Paired, repeated-measure and other test designs
have different attainable p-values and are not approximated here.
