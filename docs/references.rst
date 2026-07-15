References and software context
===============================

scCS is designed to operate on AnnData objects and to complement established
single-cell analysis and RNA-velocity workflows.

Core software ecosystem
-----------------------

scCS interoperates with:

- AnnData for annotated matrix storage;
- Scanpy for preprocessing, neighborhood graphs, dimensionality reduction,
  clustering, and visualization;
- scVelo for RNA-velocity estimation, velocity graphs, latent time, and
  velocity pseudotime;
- RegVelo datasets and models where available;
- SciPy, NumPy, pandas, statsmodels, and related scientific Python tools.

Scientific scope
----------------

RNA velocity, latent time, velocity pseudotime, source transition graphs, and
terminal-state evidence remain responsibilities of the corresponding upstream
methods. scCS adds supervised furcation-specific quantification, condition
comparison, interpretation, downstream analysis, and standardized
visualization.

Methodological context
----------------------

The methodological tutorials compare DFFP with immediate transition
pushforward, scVelo embedding projection, graph refitting on supervised
coordinates, local trajectory-frame projection, first-exit absorption, and
unlimited absorption. These comparisons explain the final method choice and
are not required steps in routine user analyses.

For the equations, assumptions, and interpretation of scCS outputs, see:

- :doc:`mathematical_framework`
- :doc:`metrics`
- :doc:`method_selection`
- :doc:`limitations`
