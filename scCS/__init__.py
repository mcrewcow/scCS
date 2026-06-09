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
- Multi-condition analysis (PairScorer for 2, MultiScorer for 3+)

Three-scorer architecture
-------------------------
- **SingleScorer**: single-condition analysis (1 experimental group)
- **PairScorer**: pairwise comparison (exactly 2 conditions)
- **MultiScorer**: multi-condition comparison (3+ conditions) with
  tiered statistical testing (omnibus + post-hoc)

Quick start — single condition
-------------------------------
>>> import scCS
>>> scorer = scCS.SingleScorer(
...     adata,
...     root='17',
...     branches=['FateA', 'FateB', 'FateC'],
...     obs_key='leiden',
... )
>>> scorer.build_embedding(ordering_metric='pseudotime')
>>> scorer.refit_pseudotime()   # fix arm coverage
>>> scorer.fit()
>>> result = scorer.score(n_bootstrap=500)
>>> print(result.summary())
>>> scorer.plot_star(result)
>>> scorer.transfer_labels(adata, result)

Quick start — pairwise comparison (2 conditions)
--------------------------------------------------
>>> pscorer = scCS.PairScorer(
...     adata,
...     root='17',
...     branches=['homeostatic', 'activated'],
...     condition_obs_key='treatment',
...     obs_key='leiden',
... )
>>> pscorer.build_embedding(ordering_metric='pseudotime')
>>> pscorer.refit_pseudotime(scale_01=False)
>>> pscorer.fit()
>>> results = pscorer.score_all_conditions()
>>> delta = pscorer.compute_delta_CS('control', 'treated')
>>> stats = pscorer.compare_conditions(results)
>>> shift = pscorer.trajectory_shift(results)

Quick start — multi-condition (3+ conditions)
-----------------------------------------------
>>> mscorer = scCS.MultiScorer(
...     adata,
...     root='17',
...     branches=['homeostatic', 'activated'],
...     condition_obs_key='treatment',
...     obs_key='leiden',
... )
>>> mscorer.build_embedding(ordering_metric='pseudotime')
>>> mscorer.fit()
>>> results = mscorer.score_all_conditions()
>>> omnibus = mscorer.compare_omnibus(results)
>>> posthoc = mscorer.compare_posthoc(results, omnibus_results=omnibus)
>>> deltas = mscorer.compute_pairwise_deltas()
"""

__version__ = "0.7.4"
__author__ = "Emil Kriukov"

# Main API — single condition
from .single import SingleScorer

# Main API — pairwise comparison (2 conditions)
from .pairwise import PairScorer

# Main API — multi-condition comparison (3+ conditions)
from .multicomparison import MultiScorer

# Fate map
from .bifurcation import FateMap, build_fate_map

# Embedding
from .embedding import (
    build_star_embedding,
    project_velocity_star,
    run_velocity_pipeline,
    compute_local_pseudotime,
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
    get_velocity_fate_drivers,
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
    plot_rose_grid,
    plot_pairwise_cs,
    plot_commitment_bar,
    plot_commitment_heatmap,
    plot_subset_comparison,
    plot_expression_trends,
    plot_nn_entropy_elbow,
    plot_delta_cs_heatmap,
    plot_compare_conditions_bar,
    plot_commitment_vector_radar,
    plot_omnibus_summary,
    plot_posthoc_heatmap,
    plot_pairwise_delta_grid,
)

__all__ = [
    # Main classes
    "SingleScorer",
    "PairScorer",
    "MultiScorer",
    # Fate map
    "FateMap",
    "build_fate_map",
    # Embedding
    "build_star_embedding",
    "project_velocity_star",
    "run_velocity_pipeline",
    "compute_local_pseudotime",
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
    "get_velocity_fate_drivers",
    # Pathway enrichment
    "run_enrichment_per_fate",
    "export_enrichment_tables",
    # Plots
    "plot_star_embedding",
    "plot_star_panels",
    "plot_rose",
    "plot_rose_grid",
    "plot_pairwise_cs",
    "plot_commitment_bar",
    "plot_commitment_heatmap",
    "plot_subset_comparison",
    "plot_expression_trends",
    "plot_nn_entropy_elbow",
    "plot_delta_cs_heatmap",
    "plot_compare_conditions_bar",
    "plot_commitment_vector_radar",
    "plot_omnibus_summary",
    "plot_posthoc_heatmap",
    "plot_pairwise_delta_grid",
]
