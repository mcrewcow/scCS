# scCS tutorial suite

## First-time-user path

Begin with the Read the Docs **Introduction** and **Mathematical framework** pages. They define the scientific inputs, derive DFFP, separate fate identity from signed progression, and explain why the star is a display rather than the source graph. Then run the SingleScorer tutorial for the dataset closest to your analysis.

The public tutorial suite is organized by biological dataset and analysis
stage. Every primary biological notebook loads its public dataset directly,
prepares RNA velocity in the original expression/PCA manifold, defines the
supervised topology, validates the ordering, fits scCS, creates figures, and
exports reproducible outputs. No hidden tutorial cache is required.

## Recommended run order

### Pancreas

1. `01_pancreas_single_scorer.ipynb`
2. `03_pancreas_pair_scorer.ipynb`
3. `05_pancreas_multi_scorer.ipynb`
4. `10_pancreas_downstream_analysis.ipynb`

### RegVelo Schwann

1. `02_schwann_single_scorer.ipynb`
2. `04_schwann_pair_scorer.ipynb`
3. `06_schwann_multi_scorer.ipynb`
4. `11_schwann_downstream_analysis.ipynb`

The Schwann notebooks use dynamical RNA velocity as the primary model. Inverse
CytoTRACE is reconstructed directly from the public dataset and is
benchmark-specific rather than a universal scCS ordering recommendation.

## Methodology and interpretation

- `07_method_selection.ipynb` introduces **Discounted Future-Fate Propagation
  (DFFP)** and visually compares it with instantaneous pushforward, display
  projection, circular star regraphing, local trajectory frames, first exit,
  and unlimited absorption.
- `08_complex_branches.ipynb` visualizes clean forward, retrograde, turning,
  loop-like, and ambiguous dynamics.
- `09_package_decision_reproduction.ipynb` reproduces the pancreas and Schwann
  method decision using package APIs.

The public DFFP terms are:

- Conditional Fate Affinity (CFA);
- Discounted Fate Reach (DFR);
- Future-Fate Specificity (FFS);
- Resolved Commitment (RC);
- Unresolved Future Probability (UFP);
- Signed Ordering Flux (SOF).

## Downstream analysis

The two downstream notebooks cover expression stars, ordering and affinity
trends, candidate commitment-associated genes, terminal fate markers, overlap
between association and marker programs, local reproducible enrichment,
ordering-third summaries, and complete export.

Cell-level gene association is exploratory. Formal inference requires genuine
independent biological replicates.

## Scalability

`12_scalability_no_chunking.ipynb` contains a single-process, in-memory benchmark.
The target ladder includes 200 million cells. Each target is measured only as
one complete allocation and calculation. If the current host cannot hold the
full problem, the row is marked `SKIPPED_INSUFFICIENT_MEMORY`; the notebook does
not substitute a chunked scientific run.

The notebook separately benchmarks:

- the cellwise DFFP metric transform after fate probabilities are available;
- the complete sparse DFFP graph solve.

Measured, extrapolated, and skipped targets are reported separately. RNA
velocity preprocessing is outside the scCS benchmark.

## PairScorer and MultiScorer demonstration mode

The public pancreas and Schwann datasets do not contain a controlled treatment
design suitable for every condition-inference feature. The condition notebooks
therefore provide two explicit paths:

- `DEMO_MODE=True` creates balanced pseudo-conditions and controlled graph
  perturbations for software validation only;
- `DEMO_MODE=False` requires genuine condition and biological-replicate columns.

Pseudo-conditions must never be presented as biological evidence.

## Visualization semantics

DFFP probabilities and Signed Ordering Flux are the scientific results. The
scVelo grid on the star, root rose, and branch-relative velocity profiles are
display-only quality-control views.

Unusual Delta, Epsilon, Alpha, or Gut motion must not be corrected merely to
make a branch appear outward. Fate identity and progression are reported
separately.

## Radar plots

Condition commitment-vector radar plots use polar axes. The notebooks create
those axes explicitly. scCS also replaces a rectangular placeholder axis with a
polar axis when one is supplied, preventing a late plotting error after a long
analysis.


## Final release-gate workflow

1. Run the two SingleScorer tutorials.
2. Run PairScorer and MultiScorer; controlled demonstrations are software tests,
   not biological experiments.
3. Run notebook 09. It must print `FINAL: READY_TO_FREEZE`.
4. Run downstream notebooks; outputs are written to dedicated downstream
   directories.
5. Run the standard scalability notebook.
6. Run the exact 200M profile on a high-memory host without chunking.
