:orphan:

Neuroblastoma differentiation: preparing velocity inputs
===========================================================

GSE288099 contains human SK-N-BE(2) neuroblastoma xenografts exposed to control,
retinoic acid (RA), niclosamide ethanolamine (NEN), or RA+NEN diets, with two
independent tumors per group. The study reports differentiation-associated
phenotypes and RNA-velocity analyses. Its public processed deposit contains
MTX/TSV expression counts rather than a verified spliced/unspliced loom.

The experiment is described in `Jiang et al., PNAS (2025)
<https://doi.org/10.1073/pnas.2502483122>`_. Source records and raw sequencing
links are available at `GSE288099
<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288099>`_.

What can be generated
------------------------

Genuine velocity layers require aligned reads with gene, cell-barcode and UMI
information, and a matching annotation. Renaming an expression matrix as loom,
or copying counts into two named layers, does not reconstruct this information.
The study used PIP-seq: process the original reads with the correct
chemistry-aware barcode/UMI workflow before attempting velocyto. Do not assume
the standard 10x read layout. Use a species-aware alignment to distinguish human
tumor cells from mouse host cells in these xenografts.

The input audit records all eight sample IDs and leaves them explicitly marked
as requiring velocity counts or tagged BAM input:

.. code-block:: bash

   python -m benchmarks.validation.neuroblastoma_inputs

If tagged BAM, filtered barcodes and matching GTF are available, inspect them
before running a locally installed velocyto CLI:

.. code-block:: bash

   python -m benchmarks.validation.neuroblastoma_inputs \
       --bam /path/to/tagged.bam \
       --barcodes /path/to/filtered_barcodes.tsv \
       --gtf /path/to/matching_annotation.gtf \
       --sample C1

This checks a sample of mapped reads for CB/UB tags and prints the command
arguments. Add ``--execute`` only when those inputs are ready. The tag check is
not a complete alignment or reference audit. Inspect the resulting loom for
nonempty, aligned spliced/unspliced layers before analysis.

Interpreting the inputs
--------------------------

No neuroblastoma loom or velocity fit is supplied by this workflow: the required
aligned inputs are not included in the public expression-matrix download.
Author-provided velocity counts could avoid repeating the raw-read alignment.
Two tumors per condition allow descriptive comparisons, but the independent
2-versus-2 exact two-sided permutation test cannot attain p<0.05. A treatment
claim also needs tumor-cell identity, a justified root and candidate outcomes,
and evidence independent of the expression markers used to define them.

Matched velocity counts for the four-arm comparison
---------------------------------------------------

Once the eight aligned looms are available, assemble a tab-separated table
with columns ``tumor``, ``loom_cell_id``, ``cell_id`` and ``species``.
Use the sample names C1, C2, R1, R2, N1, N2, RN1 and RN2 as the loom file
stems. Obtain the cell crosswalk from barcode processing, and the species
calls from the joint alignment. The expression cell IDs are in notebook 23's
``expression.h5ad``. Every modeled barcode must map to a human profile in
the same tumor. Inspect RNA-velocity confidence, phase portraits and the
embedded field before interpreting a treatment contrast.

.. code-block:: bash

   python -m benchmarks.validation.neuroblastoma_velocity \
       --expression tutorial_outputs/neuroblastoma/expression.h5ad \
       --loom-dir /path/to/verified/looms --mapping /path/to/cell_mapping.tsv \
       --out tutorial_outputs/neuroblastoma/velocity.h5ad
   export SCCS_NEUROBLASTOMA_VELOCITY_H5AD="$PWD/tutorial_outputs/neuroblastoma/velocity.h5ad"

Start Jupyter in this shell and run :doc:`tutorials/24_neuroblastoma_multi_scorer`
to compare results on the expression and RNA-velocity graphs.
