Practical cases
===============

A population can shrink during treatment because cells mature, die or leave the
sampled compartment. A mature population can grow while progenitors make little
progress toward its late state. These studies start with the observed cells and
ask which of these explanations their RNA dynamics can distinguish.

Data needed to run the cases
----------------------------

Open the notebooks from a source checkout with the :doc:`tutorial environment
<installation>`. The papers and public sequencing inputs are listed below. The retina preparation guide also works from a standalone copy of the workflow.

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
   * - 25 · Embryonic mouse retina
     - Six original Cell Ranger BAMs and matched author annotations from
       `GSE118614 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118614>`_
     - Follow :doc:`clark_retina_data`, then run 25 locally.

Blood progenitors during treatment and differentiation
------------------------------------------------------

**AML.** In one treated patient, only 24 of 800 sampled cells from the mixed
blast and immature erythroid compartment supported a late monocytic or
erythroid assignment. All 24 favored monocytes. How much does that explain
about the loss of blasts during treatment?

**LARRY.** In differentiating cultures, early progenitors were much more likely
to encounter a neutrophilic state first, yet their relative access to late
monocytic and neutrophilic cells was close to balanced. How does the first
recognizable lineage state relate to mature descendants?

.. toctree::
   :maxdepth: 1

   tutorials/18_aml_pair_scorer
   tutorials/19_larry_pair_scorer

Adrenal maturation and the response to persistent antigen
----------------------------------------------------------

**Chromaffin cells.** Late pro-chromaffin cells retained access to the oxygen-sensing
state on a common developmental reference. Which mature populations were
actually available in each embryonic window?

**B cells.** Plasma-like RNA profiles were more abundant in antigen-exposed
recipients, yet activated cells had less access to stringent late secretory
endpoints. The difference reversed when earlier secretory states counted as
destinations. Where along maturation does the condition difference arise?

.. toctree::
   :maxdepth: 1

   tutorials/21_akkuratova_faure_pair_scorer
   tutorials/22_gse230862_pair_scorer

Changing developmental competence in the embryonic retina
-----------------------------------------------------------

RGC production begins early in mouse retinogenesis. As amacrine and cone
populations emerge, do cells in the same neurogenic progenitor compartment
change their access to these destinations? Compare E12, E14, E16 and E18 on a
common splicing-derived reference and examine what each age can resolve on its
own. The tutorial is ready for local execution from the six looms.

.. toctree::
   :maxdepth: 1

   clark_retina_data
   tutorials/25_clark_fetal_retina_multiscorer
