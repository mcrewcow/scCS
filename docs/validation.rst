Validation evidence
===================

Package-backed validation used dynamical pancreas velocity and
independent deterministic and dynamical velocity fits on the RegVelo Schwann
dataset.

Benchmark evidence
-----------------------------

Across pancreas and both Schwann velocity models, root CFA was stable between
effective horizons 64 and 128 and between anchor quantiles 0.85 and 0.95. The
deterministic and dynamical Schwann models showed low mean CFA divergence and
strong SOF correlation.

Biological stress tests
-----------------------

The validation intentionally retained difficult dynamics:

* Beta was the clearest forward and fate-resolved pancreas branch.
* Alpha showed turning behavior.
* Delta was frequently retrograde and had non-sink-like late anchors.
* Epsilon showed mixed or loop-like motion.
* Schwann Gut remained annotation-coherent but strongly retrograde.

These patterns demonstrate why future identity and signed progression must be
reported independently.

Condition-scorer validation
---------------------------

Controlled pseudo-condition experiments were used only as software tests. The
final Schwann PairScorer recovered a positive ChC CFA shift with adjusted
``p=0.00794``. The Schwann MultiScorer recovered a ChC omnibus adjusted
``p=0.0162`` and a planned control-to-high contrast ``p=0.0041``. These values
validate effect recovery; they are not biological treatment results.

Scalability evidence
--------------------

The no-chunk benchmark measured the complete cellwise DFFP metric transform at
200 million cells and the complete degree-4 sparse graph solve at 100 million
cells on the available host. The 200-million-cell full graph solve was marked
``SKIPPED_INSUFFICIENT_MEMORY`` rather than replaced by chunking. The separate
high-memory runner is required for a direct 200-million-cell full-solve claim.

Software and tutorial checks
----------------------------

The release source passes 216 automated tests on Python 3.12. Repository-wide
linting includes notebook sources, and documentation builds treat warnings as
errors. The wheel and source distribution are built separately from the full
repository; they omit notebook media.

All twelve distributed notebooks contain recorded executed outputs, with 132
embedded figures. Their environment output identifies the package and dependency
versions used for those executions. Release preparation updated notebook setup
and formatting while retaining the scientific calculations and recorded figures.
The tutorial files and documentation copies are synchronized.

Independent fixed-point calculations on four complete audited graphs agreed
with scCS fate probabilities within ``6e-11``. Signed ordering flux matched a
separate calculation using average ranks for ties. These checks validate scoring
on the supplied graphs, separately from velocity preprocessing.

Fresh sensitivity checks gave mean deterministic/dynamical Schwann CFA
divergence ``0.015043`` and SOF Spearman correlation ``0.951161``. Horizon and
anchor-quantile stability criteria passed. FFS correlation ``0.742901`` is
descriptive rather than an acceptance criterion.

The successful fresh Schwann preprocessing used a lossless H5AD checkpoint
between kinetic fitting and velocity calculation. This does not establish that
a single cold-cache notebook process fits within a small memory budget.
Standard scalability tutorial executions stop at one million cells; their 200M
rows report ``NOT_REQUESTED``. Synthetic numerical benchmarks exclude velocity
fitting, AnnData copies, identifier storage and full score export, and therefore
do not establish a 200M-cell end-to-end biological workflow.

See :doc:`limitations` for interpretation and computational scope, and
:doc:`condition_inference` for replicate eligibility rules.
