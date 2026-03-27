# scCS — Single-Cell Commitment Scores

[![PyPI](https://img.shields.io/pypi/v/scCS-py)](https://pypi.org/project/scCS-py)
[![PyPI Downloads](https://img.shields.io/pypi/dm/scCS-py)](https://pypi.org/project/scCS-py)
[![Documentation Status](https://readthedocs.org/projects/sccs-py/badge/?version=latest)](https://sccs-py.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/mcrewcow/scCS/actions/workflows/python-package.yml/badge.svg)](https://github.com/mcrewcow/scCS/actions/workflows/python-package.yml)

**scCS** quantifies RNA velocity-based commitment scores for single-cell data,
generalizing the 2-state framework from
Kriukov et al. (2025) to arbitrary **k-furcations** — branching points where
a progenitor population splits into k ≥ 2 terminal fates.

---

## What it does

Classical RNA velocity tools (scVelo, CellRank) describe *where* cells are
going. scCS answers a complementary question: **how strongly committed is each
cell to a given fate, relative to the alternatives?**

Given RNA velocity vectors projected into a radial star embedding, scCS computes:

- **unCS / nCS** — unnormalized and cell-count-corrected pairwise commitment scores
- **Per-cell fate affinities** — cosine similarity of each cell's velocity to each fate direction
- **Population entropy** — how evenly velocity mass is distributed across fates
- **Per-fate cell entropy** — how individually decisive cells are toward each fate specifically
- **NN-smoothed entropy** — spatially local commitment uncertainty, noise-robust

---

## Installation

```bash
pip install scCS-py
```

Or from source:

```bash
pip install git+https://github.com/mcrewcow/scCS.git
```

---

## Quickstart

```python
import scCS

scorer = scCS.CommitmentScorer(
    adata,
    bifurcation_cluster="17",           # leiden cluster at the branching point
    terminal_cell_types=["homeostatic", "activated"],
    cluster_key="leiden",
)
scorer.build_embedding(differentiation_metric="pseudotime")
scorer.fit()
result = scorer.score(compute_cell_level=True)

print(result.summary())
scorer.plot_star(result)
scorer.plot_commitment_bar(result)
```

---

## Key features

| Feature | Description |
|---------|-------------|
| **k-furcation support** | Works for any number of fate branches (k ≥ 2) |
| **Radial star embedding** | Progenitor at origin, each fate on its own arm, cells ordered by pseudotime / CytoTRACE2 |
| **unCS / nCS** | Pairwise commitment scores, unnormalized and cell-count-corrected |
| **Per-fate entropy** | Binary cell entropy per fate — how decisive cells are toward each fate individually |
| **NN-smoothed entropy** | Nearest-neighbor smoothed per-cell entropy in the scCS embedding; elbow plots to choose k |
| **Driver genes** | Velocity-based and DEG-based driver genes per fate arm |
| **Pathway enrichment** | Enrichr ORA (KEGG, GO BP, Reactome) per fate, up and down |
| **Color map support** | Pass your original scanpy/Seurat cluster colors to all plots |

---

## Entropy metrics

scCS provides three complementary entropy metrics:

```python
# 1. Population entropy — single scalar, aggregate velocity-mass balance
result.population_entropy

# 2. Per-fate cell entropy — shape (k,), one value per fate
#    Binary entropy of each cell's affinity toward fate j, averaged over cells
result.per_fate_entropy   # e.g. array([0.31, 0.28]) for k=2

# 3. NN-smoothed per-cell entropy — shape (n_cells,)
#    Average cell_scores over k nearest neighbors in X_sccs, then compute entropy
result = scorer.score(compute_cell_level=True, k_nn=15)
result.nn_cell_entropy    # also stored in adata_sub.obs["cs_nn_entropy"]

# Find the optimal k_nn with elbow plots
fig = scorer.plot_nn_entropy_elbow(k_nn_range=range(5, 51, 5))
```

---

## Full workflow

```python
import scCS

# 1. Initialize
scorer = scCS.CommitmentScorer(
    adata,
    bifurcation_cluster="17",
    terminal_cell_types=["homeostatic", "activated"],
    cluster_key="leiden",
)

# 2. Build radial star embedding
scorer.build_embedding(differentiation_metric="pseudotime")

# 3. Fit (builds FateMap, projects velocity)
scorer.fit()

# 4. Score
result = scorer.score(compute_cell_level=True, k_nn=15)
print(result.summary())

# 5. Plots
scorer.plot_star(result)
scorer.plot_commitment_bar(result)
scorer.plot_rose(result)
scorer.plot_pairwise_cs(result)
scorer.plot_nn_entropy_elbow(k_nn_range=range(5, 51, 5))

# 6. Driver genes
vel_drivers = scorer.get_velocity_drivers(n_top=50)
deg_drivers = scorer.get_deg_drivers(n_top=50)

# 7. Pathway enrichment
enrichment = scorer.get_enrichment(deg_drivers, organism="mouse")

# 8. Compare across conditions
subset_results = scorer.score_per_subset("condition")
scorer.plot_subset_comparison(subset_results)
```

---

## Manuscript values

Reproducing the k=2 microglia bifurcation from Kriukov et al. (2025)
(GEO: [GSE285564](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285564)):

```python
scorer = scCS.CommitmentScorer(
    adata,
    bifurcation_cluster="17",
    terminal_cell_types=["homeostatic", "activated"],
    cluster_key="leiden",
)
scorer.build_embedding(differentiation_metric="pseudotime")
scorer.fit()
result = scorer.score()

result.pairwise_unCS[0, 1]  # → 9.335
result.pairwise_nCS[0, 1]   # → 8.066
```

---

## Citation

If you use scCS in your research, please cite:

> Kriukov et al. (2025) *Single-cell transcriptome of myeloid cells in response
> to transplantation of human retinal neurons reveals reversibility of microglial
> activation.* DOI: 10.XXXX
