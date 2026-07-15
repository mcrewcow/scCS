Terminology
===========

scCS uses a small set of names for the graph-based future-fate formulation.
The names describe distinct quantities and should not be collapsed into one
generic "commitment score."

Discounted Future-Fate Propagation
----------------------------------

**Discounted Future-Fate Propagation (DFFP)** is the recommended future-identity
mode. It propagates transition probability on the original RNA-velocity graph
with geometric stopping. It is not a transformation of UMAP arrows and does not
use the star to define neighbors.

Conditional Fate Affinity
-------------------------

**Conditional Fate Affinity (CFA)** is the normalized distribution across the
supervised fates after conditioning on selected-fate reach. The package exposes
it as ``future_fate_affinity`` and the exact alias
``conditional_fate_affinity``.

Discounted Fate Reach
---------------------

**Discounted Fate Reach (DFR)** is the probability of reaching any supervised
fate anchor before geometric stopping. It is exposed as ``future_fate_reach``
and ``discounted_fate_reach``. Reach depends on the effective horizon and is
interpreted as time-bounded resolution, not as physical time.

Future-Fate Specificity
-----------------------

**Future-Fate Specificity (FFS)** is one minus normalized Shannon entropy of
Conditional Fate Affinity. It measures decisiveness among supplied fates but
does not measure how much probability reaches those fates.

Resolved Commitment
-------------------

**Resolved Commitment (RC)** is the product of Discounted Fate Reach and
Future-Fate Specificity. It is exposed as ``reach_supported_specificity`` and
the alias ``resolved_commitment``. Report reach and specificity separately
alongside this composite.

Signed Ordering Flux
--------------------

**Signed Ordering Flux (SOF)** is the expected one-step change in the supplied
ordering after conditioning on transitions retained within the selected
furcation. It is exposed as ``signed_progression`` and
``signed_ordering_flux``. Positive values are forward; negative values preserve
retrograde motion; values near zero can indicate stationary, mixed, turning, or
loop-like dynamics.

Unresolved Future Probability
-----------------------------

**Unresolved Future Probability (UFP)** is the probability that geometric
stopping occurs before any supplied selected or competing anchor is reached. It
is exposed as ``unresolved_probability`` and
``unresolved_future_probability``.

Display-only velocity projection
--------------------------------

The scVelo grid, root rose, and branch-relative velocity profiles on the star
are visualization diagnostics. They show how the source velocity graph appears
after the standardized layout is drawn. They do not define DFFP probabilities,
CFA, DFR, FFS, RC, or SOF.
