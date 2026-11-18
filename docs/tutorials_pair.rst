PairScorer
==========

Compare two conditions on a shared supervised furcation. Learn how descriptive
scores relate to formal inference across independent biological replicates, and
how to recognize a design that supports descriptive analysis only.

The pancreas and Schwann examples demonstrate the replicated API with controlled
pseudo-conditions. The AML and LARRY examples instead start from public, genuine
spliced/unspliced counts, estimate RNA velocity, inspect it on the native
manifold, define roots and outcomes, and run PairScorer without artificial
transition reweighting. They retain the original experimental design: the AML
case is one patient and LARRY compares sampled culture time points. Cells, time
points, library batches and lineage barcodes are not silently relabeled as
independent treatment replicates. Those examples report descriptive scores and
coverage rather than unsupported treatment p-values.

.. toctree::
   :maxdepth: 1

   tutorials/03_pancreas_pair_scorer
   tutorials/04_schwann_pair_scorer
   tutorials/18_aml_pair_scorer
   tutorials/19_larry_pair_scorer

Experimental comparisons
------------------------

The chromaffin tutorial compares the same pro-chromaffin starting population
across embryonic stages. It shows how stage-specific sampling affects the
availability of oxygen-sensing and maturing chromaffin endpoints. The B-cell
tutorial compares eight libraries from vaccinated mouse recipients under two
antigen conditions. It tests whether RNA-defined plasma-like and pre-plasma-like
states support a comparison of activated B cells. Both tutorials start from the
original velocyto looms and carry endpoint counts alongside the scores.

.. toctree::
   :maxdepth: 1

   tutorials/21_akkuratova_faure_pair_scorer
   tutorials/22_gse230862_pair_scorer

Preparing another treatment dataset
-----------------------------------

.. toctree::
   :maxdepth: 1

   neuroblastoma_data
