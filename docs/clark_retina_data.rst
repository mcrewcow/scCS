Preparing the embryonic mouse retina
====================================

Clark and colleagues profiled mouse retinal neurogenesis from embryonic to
postnatal life. Tutorial 25 follows the authors' neurogenic cells in six
unperturbed embryonic libraries. The annotation contains 5,567 neurogenic
cells before velocyto filtering. At E12 only 25 cells fall in this compartment,
and the selected sample contains eight RGCs, one amacrine cell and no cones.
Any modeled access to cones at E12 therefore depends on the shared reference.

.. list-table:: Source libraries and annotated cells
   :header-rows: 1

   * - GEO sample
     - Library
     - SRA run
     - Annotated cells
   * - GSM3334485
     - E12
     - SRR7699771
     - 1,503
   * - GSM3334486
     - E14_rep1
     - SRR7699772
     - 14,309
   * - GSM3334487
     - E14_rep2
     - SRR7699773
     - 9,897
   * - GSM3334488
     - E16
     - SRR7699774
     - 4,763
   * - GSM3334489
     - E18_rep2
     - SRR7699775
     - 10,181
   * - GSM3334490
     - E18_rep3
     - SRR7699776
     - 10,397

The :download:`sample manifest <../tutorials/resources/clark_retina/samples.tsv>`
links directly to each submitted coordinate-sorted BAM and its checksum.
All six contain corrected cell barcodes and UMIs. Use the GTF from the 10x
mm10 v1.2.0 reference used for alignment. Labels come from the
`authors' updated annotation <https://github.com/gofflab/developing_mouse_retina_scRNASeq#cellular-phenotype-data>`_.
GEO's E12 is named ``E12_rep1`` in the annotation table. The checkout
includes a compact six-library subset with these names reconciled.

From the checkout, activate your velocyto environment. Use the reference command below,
or set ``RETINA_GTF`` to an existing copy of that uncompressed GTF. The following commands create sample-specific whitelist
files, download the original BAMs and produce one loom per library. To test the
smallest library first, append ``E16`` to the download and velocity commands.

.. code-block:: bash

   cd tutorials/resources/clark_retina
   export RETINA_WORK="$HOME/projects/clark-retina"
   bash retina.sh reference
   bash retina.sh barcodes
   bash retina.sh download
   bash retina.sh velocity

See the :download:`step-by-step guide <../tutorials/resources/clark_retina/README.md>`
for the reference download, local filenames and environment setup. Open
:doc:`tutorials/25_clark_fetal_retina_multiscorer` in the scCS tutorial kernel
with the same ``RETINA_WORK``. The notebook audits the barcode joins, fits RNA
velocity, exports age-specific scores and rose plots, and reports absolute
changes alongside reach and coverage. E12 and E16 each have one library, so
the study supports descriptive age contrasts.
