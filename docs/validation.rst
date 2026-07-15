Validation evidence
===================

The final package-backed validation used dynamical pancreas velocity and
independent deterministic and dynamical velocity fits on the RegVelo Schwann
dataset.

The release gate passed
-----------------------

The preserved-graph package reproduction reported ``FINAL: READY_TO_FREEZE``.
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
