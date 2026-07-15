Core concepts
=============

Supervised furcation
--------------------

The user supplies one root population and at least two candidate terminal
populations. scCS validates the specification but does not infer topology.

Source graph versus display
---------------------------

The RNA-velocity graph is the scientific source. The star is a standardized
visual display. DFFP never recomputes neighbors on the star and never treats
UMAP arrows as vectors that can be directly warped into the star.

Ordering as a model input
-------------------------

The ordering determines cell placement, late anchor selection, and Signed
Ordering Flux. It can be latent time, velocity pseudotime, diffusion
pseudotime, a CytoTRACE-derived coordinate, or another independently justified
continuous progression variable. Smooth display alone is not validation.

Two modes, two questions
------------------------

``future_fate`` asks which supplied outcomes are reachable over a discounted
future. ``instantaneous`` asks where immediate transition-induced motion points
in the supervised geometry. Both are retained because they answer different
scientific questions.

Complex dynamics
----------------

Future identity and progression are independent. High fate affinity can coexist
with negative or sign-changing progression. Curves, loops, returns, and
retrograde branches remain visible rather than being forced into monotonic rays.
