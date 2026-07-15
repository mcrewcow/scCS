Downstream analysis
===================

scCS downstream analysis begins only after the supervised topology, ordering,
velocity graph, anchor diagnostics, and future-fate sensitivity checks are
acceptable.

Expression on the star
----------------------

``plot_gene_expression_star`` places measured expression on the same standardized
cell positions used for scCS metrics. It is a visualization of expression, not a
new scoring model. Use shared scales when comparing conditions and separate
scales when genes have very different dynamic ranges.

Expression trends
-----------------

``plot_expression_trends`` summarizes binned expression along ordering,
Conditional Fate Affinity, fate contribution, or Resolved Commitment. Binned
means are descriptive and should not be treated as independent observations.

Candidate commitment-associated genes
-------------------------------------

``get_commitment_associated_genes`` estimates partial rank associations between
gene expression and a fitted scCS outcome.

* ``inference_unit="cell_exploratory"`` is candidate generation only.
* ``inference_unit="replicate"`` aggregates biological replicate units before
  formal inference and is preferred when genuine replicate metadata exist.
* Association is not proof of a causal lineage driver.

Fate markers
------------

``get_fate_markers`` compares each annotated terminal population with the root.
These markers describe cell identity and are distinct from genes associated with
variation in future-fate commitment within the root.

Enrichment
----------

``run_commitment_enrichment`` accepts a local mapping or GMT file. Local gene
sets and an assay-specific background are preferred for publication-grade
reproducibility. Remote Enrichr libraries are optional and can change over time.

Interpretation order
--------------------

A defensible downstream analysis reports:

#. velocity model and selected topology;
#. ordering and its validation;
#. DFFP parameters and anchor diagnostics;
#. CFA, DFR, FFS, RC, UFP, and SOF;
#. candidate genes or markers;
#. gene-set source, background, and enrichment thresholds;
#. biological replicate unit for any inferential claim.

See the pancreas and Schwann downstream-analysis tutorials for complete examples.
