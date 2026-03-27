"""
scCS — Single-cell Commitment Scores with radial star embedding.

Generalizes the 2-state commitment score framework from:

    Kriukov et al. (2025) "Single-cell transcriptome of myeloid cells in
    response to transplantation of human retinal neurons reveals reversibility
    of microglial activation"

to any number of cell fates (k-furcations), with:
- User-supplied bifurcation cluster (e.g., leiden cluster '17')
- Radial star embedding: progenitor at origin, each fate on its own arm
- Cells ordered along arms by differentiation metric (pseudotime,
  CytoTRACE2, pathway score, or any custom per-cell score)
- Population-level scores: unCS, nCS, commitment vector, entropy
- Per-cell fate affinity scores with magnitude weighting
- Bootstrap confidence intervals on CS values
- Multi-condition analysis (MultiConditionScorer)

Quick start — single condition
-------------------------------
>>> import scCS
>>> scorer = scCS.CommitmentScorer(
...     adata,
...     bifurcation_cluster='17',
...     terminal_cell_types=['FateA', 'FateB', 'FateC'],
...     cluster_key='leiden',
... )
>>> scorer.build_embedding(differentiation_metric='pseudotime')
>>> scorer.rebuild_embedding_with_subset_pseudotime()   # fix arm coverage
>>> scorer.fit()
>>> result = scorer.score(n_bootstrap=500)
>>> print(result.summary())
>>> scorer.plot_star(result)
>>> scorer.transfer_labels(adata, result)               # write back to full adata

Quick start — multi-condition
------------------------------
>>> mscorer = scCS.MultiConditionScorer(
...     adata,
...     bifurcation_cluster='17',
...     terminal_cell_types=['homeostatic', 'activated'],
...     condition_key='treatment',
...     cluster_key='leiden',
... )
>>> mscorer.build_embedding(differentiation_metric='pseudotime')
>>> mscorer.rebuild_embedding_with_subset_pseudotime(scale_01=False)
>>> mscorer.fit()
>>> results = mscorer.score_all_conditions()
>>> delta = mscorer.compute_delta_CS('control', 'treated')
>>> stats = mscorer.compare_conditions(results)
>>> shift = mscorer.trajectory_shift(results)
>>> mscorer.plot_condition_comparison(results)
>>> mscorer.plot_trajectory_shift(shift)

For k=2 (reproducing manuscript):
>>> scorer = scCS.CommitmentScorer(
...     adata,
...     bifurcation_cluster='17',
...     terminal_cell_types=['homeostatic', 'activated'],
...     cluster_key='leiden',
... )
>>> scorer.build_embedding(differentiation_metric='pseudotime')
>>> scorer.fit()
>>> result = scorer.score()
>>> # result.pairwise_nCS[0, 1] should be ~8.066 (manuscript value)
"""

__version__ = "0.5.0"
__author__ = "Emil Kriukov"

# Main API — single condition
from .trajectory import CommitmentScorer

# Main API — multi-condition
from .multiconditional import MultiConditionScorer

# Fate map
from .bifurcation import FateMap, build_fate_map

# Embedding
from .embedding import (
    build_star_embedding,
    project_velocity_star,
    run_velocity_pipeline,
    recompute_subset_pseudotime,
    scale_metric_01,
)

# Core math (for advanced users)
from .scores import (
    CommitmentScoreResult,
    compute_magnitudes,
    compute_angles,
    bin_angles,
    equal_sectors,
    centroid_sectors,
    compute_sector_magnitudes,
    compute_unCS,
    compute_nCS,
    compute_commitment_vector,
    # Entropy
    compute_population_entropy,
    compute_mean_cell_entropy,
    compute_per_fate_cell_entropy,
    compute_nn_cell_entropy,
    compute_commitment_entropy,      # backward-compat alias
    compute_pairwise_cs_matrix,
    compute_cell_scores,
    # Bootstrap
    bootstrap_cs,
)

# Driver genes
from .drivers import (
    get_velocity_drivers,
    get_deg_drivers,
)

# Pathway enrichment
from .enrichment import (
    run_enrichment_per_fate,
    export_enrichment_tables,
)

# Plotting
from .plot import (
    plot_star_embedding,
    plot_star_panels,
    plot_rose,
    plot_pairwise_cs,
    plot_commitment_bar,
    plot_commitment_heatmap,
    plot_subset_comparison,
    plot_expression_trends,
    plot_nn_entropy_elbow,
)

__all__ = [
    # Main classes
    "CommitmentScorer",
    "MultiConditionScorer",
    # Fate map
    "FateMap",
    "build_fate_map",
    # Embedding
    "build_star_embedding",
    "project_velocity_star",
    "run_velocity_pipeline",
    "recompute_subset_pseudotime",
    "scale_metric_01",
    # Results
    "CommitmentScoreResult",
    # Core math
    "compute_magnitudes",
    "compute_angles",
    "bin_angles",
    "equal_sectors",
    "centroid_sectors",
    "compute_sector_magnitudes",
    "compute_unCS",
    "compute_nCS",
    "compute_commitment_vector",
    "compute_population_entropy",
    "compute_mean_cell_entropy",
    "compute_per_fate_cell_entropy",
    "compute_nn_cell_entropy",
    "compute_commitment_entropy",    # backward-compat alias
    "compute_pairwise_cs_matrix",
    "compute_cell_scores",
    "bootstrap_cs",
    # Driver genes
    "get_velocity_drivers",
    "get_deg_drivers",
    # Pathway enrichment
    "run_enrichment_per_fate",
    "export_enrichment_tables",
    # Plots
    "plot_star_embedding",
    "plot_star_panels",
    "plot_rose",
    "plot_pairwise_cs",
    "plot_commitment_bar",
    "plot_commitment_heatmap",
    "plot_subset_comparison",
    "plot_expression_trends",
    "plot_nn_entropy_elbow",
]
