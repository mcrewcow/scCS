Tutorials
=========

The tutorials are organized by biological dataset and analysis stage. Each
primary notebook loads its public dataset, prepares RNA velocity in the original
manifold, defines the supervised furcation, runs diagnostics, fits DFFP, creates
visualizations, and exports reproducible outputs.

Getting started
---------------

Create the environment described in :doc:`installation`. Begin with a
SingleScorer notebook, then continue with the corresponding PairScorer,
MultiScorer and downstream analysis notebooks:

* Pancreas: 01, 03, 05, then 10.
* Schwann: 02, 04, 06, then 11.
* Method interpretation: 07 and 08; package decision reproduction: 09.
* Synthetic scalability: 12, using the standard profile first.

The Schwann tutorials use dynamical RNA velocity. Reconstructed inverse
CytoTRACE is specific to that benchmark, rather than a universal ordering
recommendation. Completed public-data preprocessing fits are cached for reuse.

PairScorer and MultiScorer notebooks offer a controlled demonstration mode for
software examples and a real-study path requiring genuine condition and
biological-replicate metadata. Pseudo-conditions are not biological evidence.
Cell-level gene associations are exploratory; formal inference requires
independent biological replicates.

The notebooks retain their recorded outputs and environment information.
Their figures can be read here without running a notebook locally.

Tutorial sections
-----------------

* :doc:`tutorials_single`: score and visualize one furcation.
* :doc:`tutorials_pair`: compare two conditions using biological replicates.
* :doc:`tutorials_multi`: compare three or more conditions.
* :doc:`tutorials_methods`: understand method choices and complex dynamics.
* :doc:`tutorials_downstream`: investigate expression and gene associations.
* :doc:`tutorials_scalability`: measure complete numerical workloads.

* :doc:`tutorials_validation`: analytical and simulated robustness checks.
* :doc:`tutorials_benchmarks`: AML, held-out lineage and matched-method benchmarks.
