Method selection
================

Why not transform UMAP arrows directly?
---------------------------------------

UMAP is nonlinear and does not provide a unique invertible Jacobian for
transporting gene-space velocity vectors into a supervised star. scVelo instead
uses transition-weighted displacements in the target embedding. That projection
is useful display quality control but is not a scientific definition of fate
commitment.

Alternatives evaluated
----------------------

**Fixed-vertex instantaneous pushforward.** Exact for the simplified simplex,
but collapses curvature and long-term reachability. Retained as
``scoring_mode="instantaneous"``.

**Continuous straight branches.** Improves cell display but still reduces each
fate to one ray and cannot retain loops or turns.

**Regraphing on scCS coordinates.** Circular: annotations define the geometry,
which defines neighbors, which then appear to validate the annotations. Root
fate magnitude collapses and terminal entropy becomes artificially axial.

**Local trajectory frames.** Theoretically flexible, but requires trajectory
smoothing, local tangent estimation, regularization, and choices for overlapping
or looping paths.

**First-exit absorption.** Measures selected-path containment but incorrectly
makes temporary excursions irreversible.

**Unlimited absorption.** Allows return, but in connected finite graphs it can
assign nearly every state to a selected anchor after biologically irrelevant
long-time mixing.

Selected formulation
--------------------

DFFP solves

.. math::

   H_i=\gamma\sum_jP^*_{ij}H_j,
   \qquad \gamma=\frac{h}{h+1},

with fixed anchor boundary values. It preserves the original graph, allows
leave-and-return paths, keeps unresolved probability explicit, and stabilizes
before unlimited mixing. See :doc:`mathematical_framework` for the complete
derivation and tutorial 07 for visual comparisons.

Reporting requirements
----------------------

Report the velocity model, graph construction, root and fates, ordering,
effective horizon, anchor quantile, minimum anchor count, progression scaling,
explicit competitors, horizon sensitivity, anchor sensitivity, and anchor
diagnostics. Do not describe DFFP as a transformed UMAP velocity vector.
