# scCS — Single-Cell Commitment Scores

[![Documentation Status](https://readthedocs.org/projects/sccs-py/badge/?version=latest)](https://sccs.readthedocs.io/en/latest/)

**scCS** computes RNA velocity-based commitment scores for single-cell data,
generalizing the 2-state (homeostatic/activated) framework from
Kriukov et al. (2025) to arbitrary k-furcations.

## Installation

    pip install git+https://github.com/mcrewcow/scCS.git

## Quickstart

    import scCS
    import scanpy as sc

    adata = sc.read_h5ad("your_data.h5ad")
    scorer = scCS.CommitmentScorer(adata, cluster_key="cell_type")
    result = scorer.score(compute_cell_level=True)
    scorer.plot_star(result, color_by="fate")
    scorer.plot_commitment_bar(result)

## Key features

- k-furcation support — works for any number of fate branches
- Star embedding — radial layout with one arm per fate
- unCS / nCS — unnormalized and cell-count-corrected commitment scores
- Expression trends — gene expression vs commitment axis
- Color map support — preserves your original scanpy cluster colors

## Citation

Kriukov et al. (2025) Single-cell transcriptome of myeloid cells in response
to transplantation of human retinal neurons reveals reversibility of microglial
activation. DOI: 10.XXXX
