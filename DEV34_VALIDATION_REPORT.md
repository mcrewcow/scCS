# scCS 0.8.0.dev34 focused release-gate update

## Scope

This pass resolves the remaining release blockers identified after executing all
12 public notebooks. It does not change the Discounted Future-Fate Propagation
(DFFP) equations.

## Changes

### Final package decision notebook

- Preserves the original RegVelo Schwann neighbor graph for both deterministic
  and dynamical velocity fits.
- Changes only the velocity model in the cross-model sensitivity comparison.
- Aligns all result arrays by stable cell IDs before comparison.
- Adds endpoint-anchor quantile sensitivity (`0.85` versus `0.95`).
- Retains horizon sensitivity (`64` versus `128`).
- Keeps dynamical velocity as the primary Schwann analysis and deterministic
  velocity as sensitivity analysis.
- The notebook must print `FINAL: READY_TO_FREEZE` before release.

### PairScorer and MultiScorer tutorials

- The controlled Schwann demonstrations now target Conditional Fate Affinity
  directly.
- The injected ChC effect was strengthened and pseudo-replicate perturbation
  noise reduced. This is software validation only and is not a biological
  treatment model.
- Public inference tables now include `metric_public` and `metric_label` while
  retaining the established canonical `metric` field for compatibility.
- Rectangular placeholder axes are replaced by polar axes for radar plots,
  including two-fate analyses.

### Downstream tutorials

- Pancreas outputs now write to `tutorial_outputs/pancreas_downstream`.
- Schwann outputs now write to `tutorial_outputs/schwann_downstream`.
- SingleScorer outputs are no longer overwritten.

### Scalability

- The standard notebook still measures only complete in-memory problems and
  never substitutes chunking.
- `SCCS_SCALABILITY_PROFILE=exact_200m` requires the complete 200-million-cell
  graph solve and raises an error if the full allocation is not possible.
- A separate degree-30 no-chunk ladder was added for a more realistic graph
  density.
- `benchmarks/v08/run_dffp_200m_no_chunk.py` and a 256 GB Slurm template were
  added for the final high-memory run.

## Validation completed

- 163 tests passed with AnnData, Scanpy, and scVelo installed.
- Ruff passed for package code, tests, benchmarks, and all 12 notebooks.
- Every notebook code cell parsed successfully.
- No tutorial contains `try`/`except` control flow.
- The method-selection and complex-branch notebooks were re-executed with
  scCS 0.8.0.dev34 and completed without errors.
- Strict Sphinx/Read the Docs build passed with warnings treated as errors.
- Wheel and source distributions built successfully.
- Twine metadata checks passed.
- An isolated wheel import returned `0.8.0.dev34`.
- A built-wheel synthetic PairScorer test returned the public DFFP fields:
  `metric_public='future_fate_affinity'` and
  `metric_label='Conditional Fate Affinity (CFA) toward B'`.
- The no-chunk command-line runner completed a 100,000-cell smoke test with a
  converged solver and probability checksum.

## Still requires execution outside this build host

The following evidence cannot be honestly claimed until it is executed:

1. The revised real-data package decision notebook must print
   `FINAL: READY_TO_FREEZE`.
2. The revised controlled Schwann PairScorer and MultiScorer notebooks must
   recover the predeclared ChC CFA effects clearly.
3. The exact complete 200-million-cell DFFP graph solve must be run on a
   high-memory host. The provided workflow does not use chunks.

The first two items are expected to run on the existing analysis workstation.
The degree-4 200-million-cell solve should be run with at least 128 GB RAM; the
provided Slurm template requests 256 GB to retain safe headroom.
