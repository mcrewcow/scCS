Tutorials
=========

The pancreas and Schwann tutorials introduce the scorers using public RNA-velocity
data. The :doc:`Practical cases <tutorials_practical>` use observed developmental
stages and treatment conditions. Each case identifies its data source, required
inputs and the graph used for the comparison.

Getting started
---------------

Create the environment described in :doc:`installation`. Begin with a
SingleScorer notebook, then continue with the corresponding PairScorer,
MultiScorer and downstream analysis notebooks:

* Pancreas: 01, 03, 05, then 10.
* Schwann: 02, 04, 06, then 11.
* Method interpretation: 07 and 08; sensitivity to model choices: 09.
* Synthetic scalability: 12, using the standard profile first.

The Schwann tutorials use dynamical RNA velocity. Reconstructed inverse
CytoTRACE is specific to that benchmark, rather than a universal ordering
recommendation. Completed public-data preprocessing fits are cached for reuse.

The pancreas and Schwann PairScorer and MultiScorer notebooks offer a controlled demonstration mode for
software examples and a real-study path requiring genuine condition and
biological-replicate metadata. Pseudo-conditions are not biological evidence.
Cell-level gene associations are exploratory; formal inference requires
independent biological replicates.

The notebooks retain their recorded outputs and environment information.
Their figures can be read here without running a notebook locally.

Tutorial sections
-----------------

* :doc:`tutorials_single`: score and visualize one furcation.
* :doc:`tutorials_pair`: compare two conditions on a shared annotated fork.
* :doc:`tutorials_practical`: work through AML, LARRY, chromaffin, B-cell and neuroblastoma studies.
* :doc:`tutorials_multi`: compare three or more conditions.
* :doc:`tutorials_methods`: understand method choices and complex dynamics.
* :doc:`tutorials_downstream`: investigate expression and gene associations.
* :doc:`tutorials_scalability`: measure complete numerical workloads.
* :doc:`tutorials_validation`: non-trivial questions about abundance, reach,
  reversible dynamics, uncertainty and comparisons with CellRank.
