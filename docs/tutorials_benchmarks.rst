Real-data and method benchmarks
==================================

The AML notebook is a single-patient descriptive pilot, with explicit limitations
on malignant-cell annotation and biological replication. It includes a failed
inference check rather than reporting cell-level treatment p-values. The recorded
run has very low selected-fate coverage and does not establish differentiation
redirection.

The LARRY notebook evaluates predictions against held-out clonal outcomes. Its
time-directed graph is expression-derived and is explicitly not RNA velocity.
The recorded run scores 25 of 119 test clones and does not outperform the constant
training-clone baseline on that common subset.
The matched CellRank comparison demonstrates agreement and differences in the
estimands under controlled graph and endpoint definitions.

.. toctree::
   :maxdepth: 1

   tutorials/18_aml_treatment_pilot
   tutorials/19_larry_clone_benchmark
   tutorials/20_cellrank_matched_comparison
   neuroblastoma_data
