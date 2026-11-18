Practical cases
===============

Choose a study by its biological question and experimental unit. Each case
identifies its starting population, the measured inputs and the conditions
being compared. Read the condition difference alongside the number of roots
with defined affinity and the representation of endpoints in each group.

Data needed to run the cases
----------------------------

Open the notebooks from a source checkout with the :doc:`tutorial environment
<installation>`. The checkout provides the analysis code and the sequencing-run
manifest. Downloads go into ``data/`` and generated fits and tables go into
``tutorial_outputs/``. For a new checkout, follow the run order below.

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Tutorial
     - Public input
     - How to start
   * - 18 · AML
     - `Zenodo merged loom <https://doi.org/10.5281/zenodo.6587903>`_ and
       `GSE116481 cell metadata <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116481>`_
     - Run 18 directly.
   * - 19 · LARRY
     - `Figshare count object <https://ndownloader.figshare.com/files/37028569>`_
       with spliced and unspliced layers
     - Run 19 to create the analysis-subset loom.
   * - 21 · Chromaffin
     - Nine looms and the authors' annotated ChC object from
       `GSE204700 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE204700>`_
     - Run 21 directly.
   * - 22 · B-cell differentiation
     - Eight recipient-library looms from
       `GSE230862 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE230862>`_
     - Run the cells in order to build the inputs for the endpoint sensitivity check.
   * - 23–24 · Neuroblastoma
     - Eight expression matrices with features and barcodes from
       `GSE288099 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288099>`_
     - Run 23, then 24 in the same checkout.

Notebook 23 writes ``expression.h5ad`` and ``eligible_profiles.csv.gz`` to
``tutorial_outputs/neuroblastoma/``. These are the two inputs for notebook 24's
expression-neighborhood comparison. The repository includes the
:download:`ENA run manifest <../tutorials/resources/GSE288099_fastq_manifest.tsv>`
with sample assignments, read locations, file sizes and MD5 checksums for all
24 runs. The optional RNA-velocity comparison requires eight species-verified
looms and a barcode mapping, or the resulting matched velocity fit. See
:doc:`neuroblastoma_data` for the input format and preparation steps.

Hematopoiesis and treatment response
------------------------------------

AML compares a single patient's baseline and day-2 blast compartment. LARRY
compares sampled culture days and tests how later cells support earlier fate
assignments. Both report descriptive differences on measured velocity graphs.

.. toctree::
   :maxdepth: 1

   tutorials/18_aml_pair_scorer
   tutorials/19_larry_pair_scorer

Embryonic and immune differentiation
------------------------------------

Chromaffin cells illustrate a developmentally inferred fork and the effect of
stage-specific anchors. The B-cell case compares first arrival at two observed
RNA states. Broader anchors change the direction of the antigen-condition contrast.

.. toctree::
   :maxdepth: 1

   tutorials/21_akkuratova_faure_pair_scorer
   tutorials/22_gse230862_pair_scorer

Tumor differentiation across four conditions
--------------------------------------------

Follow eight neuroblastoma xenografts from public expression matrices to an
explicit MultiScorer expression-neighborhood reference. Two tumors per arm
contribute equally to the control, RA, NEN and combination comparisons.
The study describes a maturation continuum. Its intermediate clusters and
experimental treatment groups serve different purposes from fate branches.

.. toctree::
   :maxdepth: 1

   tutorials/23_neuroblastoma_inputs
   tutorials/24_neuroblastoma_multi_scorer
   neuroblastoma_data
