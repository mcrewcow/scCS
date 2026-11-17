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

Single-condition workflows
--------------------------

.. toctree::
   :maxdepth: 1

   tutorials/01_pancreas_single_scorer
   tutorials/02_schwann_single_scorer

Two-condition workflows
-----------------------

.. toctree::
   :maxdepth: 1

   tutorials/03_pancreas_pair_scorer
   tutorials/04_schwann_pair_scorer

Three-or-more-condition workflows
---------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/05_pancreas_multi_scorer
   tutorials/06_schwann_multi_scorer

Methodology, interpretation, and validation
-------------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/07_method_selection
   tutorials/08_complex_branches
   tutorials/09_package_decision_reproduction

Downstream analysis
-------------------

.. toctree::
   :maxdepth: 1

   tutorials/10_pancreas_downstream_analysis
   tutorials/11_schwann_downstream_analysis

Scalability
-----------

.. toctree::
   :maxdepth: 1

   tutorials/12_scalability_no_chunking
