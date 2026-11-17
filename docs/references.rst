References and software context
===============================

scCS is designed to work with AnnData, Scanpy, and scVelo objects. RNA velocity,
latent time, velocity pseudotime, terminal-state estimates, and transition graphs
remain the responsibility of the source method. scCS adds supervised
furcation-specific quantification and standardized visualization.

The package decision notebooks document comparisons with transition pushforward,
scVelo embedding projection, graph refitting on supervised coordinates, local
trajectory frames, first-exit absorption, and unlimited absorption. These
notebooks are included as methodological provenance rather than as required user
workflows.

Published foundation and methods
--------------------------------

Kriukov et al. (2026), *Single-cell transcriptomics reveals reversible
microglia/macrophage activation after human retinal neuron transplantation into
mouse retina*, iScience, DOI: https://doi.org/10.1016/j.isci.2026.117232.
The redesigned methods are described separately in the methods preprint:
https://doi.org/10.64898/2026.07.27.741070. Cite the exact software version and
scientific settings alongside the relevant work.
