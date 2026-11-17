# scCS 0.8.1rc2 validation

Checks used a separate Python 3.12 environment with constraints-tutorials.txt and
the dev/tutorial dependencies. Exact versions are in release_checks/environment.json.

## Completed checks

- **216 tests passed**, covering earlier release fixes plus undefined-replicate
  handling, explicit exclusions, bootstrap draws, complementary hypotheses,
  p-value resolution, cache dependencies, compact exports, subset caching,
  graph diagnostics, sensitivity and descriptive plot counts.
- Ruff passes for package code, tests, benchmarks, documentation configuration
  and the independent replay script. `pip check` finds no broken requirements.
- Documentation builds with warnings treated as errors and notebook execution
  disabled. All documentation notebooks are byte-identical to tutorials/.
- No existing public definitions or parameters were removed in the source API
  comparison. Behavior changes and new options are documented.
- Wheel and sdist builds and separate installs were checked. Distribution contents
  exclude checkpoints, bytecode and notebook media. The full source ZIP includes
  the executed notebooks and documentation.

## Executed notebooks

All 12 revised notebooks completed with **132 embedded PNG figures** and no
recorded execution errors. Cells ran with IPython display hooks in process because
this host restricts socket-based notebook kernels. Mathematical and biological
cells ran as written, using matching shared preprocessing fits. Default optional
flags were retained; default runs do not exercise every optional branch.

| Notebook | Code cells | Figures | Seconds |
| --- | ---: | ---: | ---: |
| 01_pancreas_single_scorer.ipynb | 21 | 13 | 46.7 |
| 02_schwann_single_scorer.ipynb | 20 | 13 | 54.6 |
| 03_pancreas_pair_scorer.ipynb | 22 | 18 | 41.0 |
| 04_schwann_pair_scorer.ipynb | 22 | 18 | 38.3 |
| 05_pancreas_multi_scorer.ipynb | 23 | 19 | 34.8 |
| 06_schwann_multi_scorer.ipynb | 23 | 19 | 34.6 |
| 07_method_selection.ipynb | 8 | 7 | 2.0 |
| 08_complex_branches.ipynb | 9 | 6 | 2.2 |
| 09_package_decision_reproduction.ipynb | 8 | 2 | 49.0 |
| 10_pancreas_downstream_analysis.ipynb | 19 | 8 | 29.2 |
| 11_schwann_downstream_analysis.ipynb | 19 | 8 | 23.5 |
| 12_scalability_no_chunking.ipynb | 10 | 1 | 12.5 |

Fresh public downloads matched the audited SHA-256 values: 3,696 pancreas cells
and 8,821 Schwann cells. Dynamical fits used max_iter=20 and four thread workers;
subsequent notebooks reused these complete fits. Independent deterministic
Schwann preprocessing also completed. Velocity fitting is outside the numerical
scCS no-chunk benchmark.

Earlier memory-heavy attempts were killed by the host. The successful Schwann
workflow used a lossless H5AD checkpoint between kinetic fitting and velocity
calculation. This does not establish that a single cold-cache notebook process
fits in this host's memory limit. Tutorials now disable redundant AnnData subsets
and release kinetic working layers after the complete fit is cached. The full
fit remains in its preprocessing cache. Run one biological notebook kernel at
a time; cold preprocessing can require more memory than cached analysis.

## Scientific checks

Independent boundary-clamped fixed-point calculations on all four audited full
cell graphs agreed with scCS fate probabilities within **6e-11**. Signed ordering
flux matched a separate calculation using average ranks for ties. These replays
validate scoring on audited graphs, separately from fresh preprocessing.

Fresh sensitivity runs reproduced mean deterministic/dynamical Schwann CFA
divergence **0.015043** and SOF Spearman correlation **0.951161**. Horizon/anchor
stability and defined-CFA coverage criteria passed. FFS correlation **0.742901**
is descriptive. The values are in release_checks/method_sensitivity.csv.
Controlled condition perturbations test software behavior; they are not biological
treatment experiments or proof of irreversible commitment.

## Scalability limits

The standard profile measured complete numerical problems through one million
cells and the requested degree-30 ladder. The 200M table explicitly reports
NOT_REQUESTED. Low-memory skipping and failed-run handling were checked. The
cellwise transform respects the undefined-CFA threshold and records coverage.

No fresh 200M computation was requested. These synthetic workloads exclude
velocity fitting, AnnData copies, identifier storage and full score export.
They do not establish a 200M-cell end-to-end biological workflow.

## Evidence

release_checks/ contains test, lint, environment, documentation/build, notebook,
source-hash, API and full-graph replay records. Full replay inputs remain in the
previously supplied scCS_Audit_Evidence.zip. Run instructions are in
release_checks/README.md.
