References and software context
===============================

scCS is designed to work with AnnData, Scanpy, and scVelo objects. RNA velocity,
latent time, velocity pseudotime, terminal-state estimates, and transition graphs
remain the responsibility of the source method. scCS adds supervised
furcation-specific quantification and standardized visualization.

The package decision notebooks document comparisons with transition pushforward,
scVelo embedding projection, graph refitting on supervised coordinates, local
trajectory frames, first-exit absorption, and unlimited absorption. These
notebooks are included as methodological provenance rather than as required user
workflows.

Published foundation and methods
--------------------------------

Kriukov et al. (2026), *Single-cell transcriptomics reveals reversible
microglia/macrophage activation after human retinal neuron transplantation into
mouse retina*, iScience, DOI: https://doi.org/10.1016/j.isci.2026.117232.
The redesigned methods are described separately in the methods preprint:
https://doi.org/10.64898/2026.07.27.741070. Cite the exact software version and
scientific settings alongside the relevant work.

Validation and benchmark data
-----------------------------

* Weinreb et al. (2020), *Lineage tracing on transcriptional landscapes links
  state to fate during differentiation*, Science. DOI:
  https://doi.org/10.1126/science.aaw3381.
* Weiler et al. (2024), *CellRank 2: unified fate mapping in multiview single-cell
  data*, Nature Methods. DOI: https://doi.org/10.1038/s41592-024-02303-9.
* Ranek, Stanley and Purvis, versioned velocity datasets and preprocessing:
  https://doi.org/10.5281/zenodo.6587903. Original AML treatment data:
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116481.
* LARRY raw splicing-count object distributed with Pyro-Velocity, version 1:
  https://doi.org/10.6084/m9.figshare.20780344.v1. The data providers report
  dropEst/velocyto processing of the original inDrop reads:
  https://doi.org/10.1101/2022.09.12.507691.
* Jiang et al. (2025), *Restoring mitochondrial quantity and quality to reverse
  the Warburg effect and drive neuroblastoma differentiation*, PNAS. DOI:
  https://doi.org/10.1073/pnas.2502483122.
