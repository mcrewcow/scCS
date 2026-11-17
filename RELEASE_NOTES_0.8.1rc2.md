# scCS 0.8.1rc2

This candidate retains the scoring definitions and supported scorer calls while
making replicate eligibility explicit and simplifying the user documentation.

## Behavior changes and migration

- **Undefined replicate outcomes now raise by default.** This covers pairwise,
  omnibus, post-hoc, planned-contrast, mixed-model and hierarchical-bootstrap
  inference. `undefined_replicates="drop"` explicitly requests a conditional
  analysis and returns excluded IDs, reasons, sample sizes and coverage diagnostics.
  At least two eligible replicates must remain in each tested condition.
  Descriptive replicate plots warn about missing values and display their sample denominators.
- **Undefined CFA, conditional entropy and specificity remain NaN**, as introduced
  in rc1. This differs from earlier development builds. Use a defined-cell mask
  and report coverage and reach; do not replace missing affinity with zero or
  uniform preference. A wholly undefined group requires explicit handling.
- Undefined bootstrap draws raise. `undefined_bootstrap="report"` returns NaN
  interval bounds with the undefined-draw count; no draws are dropped or redrawn.
- Verified complementary two-fate CFA/composition endpoints share one two-sided
  replicate hypothesis. Both rows remain in output with a shared hypothesis ID.
  Raw fate probabilities, K>2 endpoints and distinct condition pairs are retained
  as separate hypotheses. Mixed-model endpoint corrections remain conservative.
- Exact pairwise inference reports a minimum p-value bound and warns when it
  exceeds `alpha`. The 3-versus-3 two-sided exact mean-difference test cannot attain
  p<0.05. Monte Carlo output reports finite-resampling resolution.

## Diagnostics and performance

Preflight separates graph dead ends from root cells with no path to selected
anchors. Graph structure, discounted reach and biological differentiation are
different questions. `SingleScorer.sensitivity_analysis()` returns horizon/anchor
comparisons without changing the fitted model. Discounting favors shorter paths;
CFA normalization does not remove this effect.

`score()` respects `build_embedding(cache_subset=False)`. Optional compact export
metadata omits redundant full-length cell-ID/progression arrays while retaining
identity fingerprints, selected indices and anchor IDs. Full-length score exports
still consume memory; `write_to_adata=False` avoids them. Existing default export
metadata remains available.

Optional cache dependencies can be absent without crashing cache signatures.
Star color aliases resolve through the shared metric registry.

## Tutorials, documentation and packaging

The API landing page emphasizes common workflows; the complete reference retains
all supported methods. Reviewed, executed tutorials are synchronized with the
documentation. Development provenance and release-gate commentary were removed;
scientific explanations remain. Tutorials avoid redundant AnnData subsets, release
kinetic-fit working arrays after caching, and keep tied ordering values together
in descriptive ordering bins. Tutorial 12 explicitly reports unrequested 200M
workloads and records failed runs. Synthetic numerical benchmarks exclude AnnData,
identity storage, exports and velocity fitting.

The full source ZIP and repository include the executed notebooks and documentation.
PyPI wheel/sdist artifacts contain the package, tests and supporting text rather
than notebook media. Checkpoints, bytecode, caches, build outputs and generated
development metadata are excluded from the full source ZIP. Built distributions
retain their required packaging metadata. See RELEASE_VALIDATION.md for checks actually completed.
