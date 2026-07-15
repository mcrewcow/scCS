Diagnostics
===========

Ordering diagnostics
--------------------

Check finiteness, orientation, unique values, large ties, root progression, and
agreement with independent developmental information. Visual smoothness alone is
not validation.

Horizon sensitivity
-------------------

Compare future-fate affinity at effective horizons 32, 64, and 128. Fate identity
should be substantially more stable than absolute reach. Large changes in
conditional affinity indicate that the graph has not resolved the future within
the chosen scale.

Anchor sensitivity
------------------

Compare anchor quantiles such as 0.85, 0.90, and 0.95. Inspect per-cell affinity
stability, not only mean entropy.

Anchor quality
--------------

``result.anchor_diagnostics_frame()`` reports transitions from endpoint anchors
to the same fate, the root, other selected fates, and outside the selected path.
A non-sink-like anchor is an interpretation warning, not an automatic failure.
Curved or retrograde biology can make an annotated late state dynamically
non-terminal.

Selected-path coverage
----------------------

One-step selected-path coverage quantifies how much transition mass remains in
the selected furcation. It does not irreversibly remove outside paths from the
future-fate solver.

Unresolved probability
----------------------

High unresolved probability means the selected anchors are not reached at the
chosen effective horizon. It may reflect early cells, omitted outcomes, slow
dynamics, poor anchors, or graph uncertainty.

Velocity-model sensitivity
--------------------------

When feasible, compare deterministic, stochastic, or dynamical velocity models.
Use continuous probability divergence and progression correlation rather than
only dominant-fate agreement, which is unstable for nearly tied probabilities.
