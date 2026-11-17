Release notes and migration
===========================

0.8.1rc2
---------

The scientific score definitions remain unchanged from rc1. Supported scorer
calls remain available. The primary behavior change is strict handling of
undefined replicate outcomes; see :doc:`condition_inference`.

* Non-finite replicate outcomes raise unless ``undefined_replicates="drop"``
  is explicitly selected. Excluded IDs, reasons and sample sizes are returned.
* Undefined bootstrap draws raise; ``undefined_bootstrap="report"`` returns
  NaN bounds and diagnostic counts without discarding or redrawing samples.
* Verified complementary two-fate CFA/composition outcomes count once in the
  two-sided replicate hypothesis family, with both fate rows preserved.
* Exact pairwise tests report minimum attainable p-value bounds and warn for
  insufficient replicate designs. Monte Carlo output reports resampling resolution.
* Preflight reports graph dead ends and root reachability separately. A new
  future-fate sensitivity helper compares horizons and anchor quantiles.
* Optional cache distributions can be absent. Plot aliases use one registry.
* ``score()`` respects disabled subset caching. Optional compact metadata
  reduces duplicate identifiers while retaining alignment safeguards.
* The API guide is concise; :doc:`api_advanced` retains the complete reference.
* Executed tutorials are published, and scalability status distinguishes
  unrequested, memory-skipped, failed and measured workloads.

Migration from development builds
----------------------------------

Rc1 introduced NaN conditional affinity, entropy and specificity when a cell's
selected-fate affinity is undefined. This is a behavior change from development
builds that substituted a uniform distribution. Raw per-fate probability is
preserved. A safe descriptive calculation is:

.. code-block:: python

   root = result.root_mask
   defined = root & result.affinity_defined
   coverage = defined.sum() / root.sum()
   mean_cfa = (
       result.conditional_fate_affinity[defined].mean(axis=0)
       if defined.any() else None
   )
   mean_reach = result.discounted_fate_reach[root].mean()

``np.nanmean`` is also suitable when an all-undefined population is handled
explicitly. Neither approach makes exclusions scientifically neutral: report
coverage and reach. A wholly undefined replicate must not disappear from an
inferential comparison.

Review :doc:`limitations` for the distance effect of discounting and the allocation
scope of large synthetic benchmarks. The source archive's ``RELEASE_VALIDATION.md``
records package, documentation and notebook checks and their practical limits.
