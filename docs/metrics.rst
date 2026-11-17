Metrics and semantics
=====================

The recommended graph-based method is **Discounted Future-Fate Propagation
(DFFP)**. The public field names remain explicit, while concise terms are used
throughout the documentation.

Conditional Fate Affinity
-------------------------

For supervised fate probabilities :math:`p_{if}`, Conditional Fate Affinity
(CFA) is

.. math::

   q_{if} = \frac{p_{if}}{\sum_g p_{ig}}.

It describes fate identity conditional on reaching a supervised fate. It is
exposed as ``future_fate_affinity`` and ``conditional_fate_affinity``.

Discounted Fate Reach
---------------------

.. math::

   R_i = \sum_f p_{if}.

Discounted Fate Reach (DFR) is exposed as ``future_fate_reach`` and
``discounted_fate_reach``. It is horizon-dependent and is interpreted as
time-bounded resolution or endpoint reach, not universal physical time.

Future-Fate Specificity
-----------------------

.. math::

   S_i = 1 - \frac{-\sum_f q_{if}\log q_{if}}{\log K}.

Future-Fate Specificity (FFS) is zero for a uniform fate distribution and one
for a point mass.

Resolved Commitment
-------------------

.. math::

   C_i = R_i S_i.

Resolved Commitment (RC) combines reach and specificity. It is exposed as
``reach_supported_specificity`` and ``resolved_commitment``. Always report DFR
and FFS separately alongside RC.

Unresolved Future Probability
-----------------------------

Unresolved Future Probability (UFP) is the chance that geometric stopping occurs
before any supplied selected or competing anchor is reached. It prevents
unlimited random-walk mixing from assigning every cell to an outcome.

Signed Ordering Flux
--------------------

Signed Ordering Flux (SOF) is the expected one-step change in the supplied
ordering, conditioned on transitions retained in the selected furcation. It is
exposed as ``signed_progression`` and ``signed_ordering_flux``. Positive values
are forward, negative values are retrograde, and values near zero may indicate
stationary, mixed, turning, or loop-like motion.

Compatibility names
-------------------

The established result properties remain available. In future-fate mode:

.. list-table::
   :header-rows: 1

   * - Established property
     - DFFP meaning
   * - ``directional_affinity``
     - Conditional Fate Affinity
   * - ``commitment_strength``
     - Discounted Fate Reach
   * - ``directional_entropy``
     - future-fate entropy
   * - ``directional_specificity``
     - Future-Fate Specificity
   * - ``specific_commitment``
     - Resolved Commitment
   * - ``commitment_contribution``
     - discounted selected-fate probability by fate
   * - ``progression_velocity``
     - Signed Ordering Flux
   * - ``transition_coverage``
     - one-step selected-path coverage

Undefined conditional quantities
----------------------------------

Below ``min_reach``, CFA, future-fate entropy and FFS are undefined (NaN).
Conditional summaries exclude those cells; per-replicate counts and the fraction
with defined affinity are reported separately. RC is conservatively zero while
discounted fate contributions retain their actual probabilities. Interpret
conditional preference with DFR, UFP and defined-affinity coverage; missing CFA
is not a uniform preference. Gene associations use only cells with the target
outcome defined and enforce the minimum cell count for each replicate.

Ordering ranks use average ranks for ties. Thus permuting cells tied at the
same ordering cannot create SOF. SOF remains conditional on transitions retained
within the selected furcation. DFFP horizons count discounted graph steps, not
physical elapsed time, and these associations alone do not prove irreversible
biological commitment.

Instantaneous mode remains the default for backward compatibility. In two-fate
instantaneous geometry, cosine affinity measures the sign of a collinear
displacement and cannot distinguish mixture strength along that ray. The
biological tutorials explicitly choose future-fate mode.

Export and reload validate unique cell IDs. New saves retain the full graph
axis and can be loaded against a reordered copy of the same full cell set.
Reordering the scorer's live AnnData requires reload or rebuilding. Partial
legacy saves lacking the full cell axis must be rebuilt.
