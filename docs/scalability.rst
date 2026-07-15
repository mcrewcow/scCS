Scalability
===========

The scalability tutorial uses **single-process, in-memory, no-chunking**
benchmarks. Each measured cell count is allocated and scored as one complete
problem. This preserves the meaning of the benchmark and avoids presenting
streamed partial calculations as an exact full-scale run.

Two distinct workloads
----------------------

Cellwise DFFP metric transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once selected-fate probabilities are available, CFA, DFR, FFS, RC, and UFP are
cellwise operations with memory proportional to ``n_cells × n_fates``. The
notebook includes an exact ladder whose target list extends to 200 million
cells. A target is measured only when the current machine can allocate the
complete arrays under the configured safety fraction.

Full DFFP graph solve
~~~~~~~~~~~~~~~~~~~~~

The iterative DFFP solver stores the sparse transition graph plus at least one
state-by-outcome probability matrix. Memory scales approximately with

``number of retained edges + n_cells × n_outcomes``.

The notebook also benchmarks this complete sparse solve without partitioning.
The exact target list includes 200 million cells, but a run is recorded as
``SKIPPED_INSUFFICIENT_MEMORY`` rather than silently chunked when the host cannot
hold the full graph and solver state.

Interpretation
--------------

Measured runs, analytic memory estimates, and extrapolated times are reported in
separate columns. A 200-million-cell claim is valid only when the row is marked
``MEASURED`` on the stated hardware, solver, graph degree, fate count, dtype, and
tolerance.

RNA-velocity preprocessing
---------------------------

Neighbor construction, kinetic fitting, and velocity-graph estimation are
outside the scCS scoring benchmark. They must be reported separately and should
not be included in an scCS-only throughput claim.

Plotting
--------

Plotting may subsample cells because it is a rendering operation. Scientific
scoring, probability propagation, population summaries, and inference are not
subsampled in the no-chunk benchmark.


High-memory exact 200-million-cell run
--------------------------------------

The standard tutorial measures every complete problem that fits safely on the
current host. To require the exact 200-million-cell full graph solve, run with
``SCCS_SCALABILITY_PROFILE=exact_200m``. The notebook raises an error if the
complete allocation cannot be made. No chunked or out-of-core substitute is
used. A degree-30 no-chunk ladder is reported separately from the degree-4
200-million-cell performance graph.
