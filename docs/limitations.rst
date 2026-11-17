Limitations
===========

* scCS is supervised and does not discover topology or terminal states.
* Results depend on RNA-velocity model quality and transition-graph construction.
* Ordering choice affects anchors, display placement, and signed progression.
* Effective horizon is a graph scale, not a calibrated physical time unit.
* Endpoint anchors are annotation- and ordering-defined and may not be natural sinks.
* Explicit competing outcomes require biological justification.
* Hard dominant-fate labels are unstable for low-specificity cells.
* Future-fate mode does not provide a scientific velocity vector in star space.
* Instantaneous and future-fate modes answer different questions.
* Condition inference requires independent biological replicates.
* Gene associations are hypothesis-generating and not causal evidence.
* Very large future-fate solves remain more demanding than the instantaneous transform.

Undefined conditional quantities
----------------------------------

Below ``min_reach``, CFA, future-fate entropy and FFS are undefined (NaN).
Conditional summaries exclude those cells; per-replicate counts and the fraction
with defined affinity are reported separately. RC is conservatively zero while
discounted fate contributions retain their actual probabilities. Interpret
conditional preference with DFR, UFP and defined-affinity coverage; missing CFA
is not a uniform preference. Gene associations use only cells with the target
outcome defined and enforce the minimum cell count for each replicate.

Ordering ranks use average ranks for ties. Thus permuting cells tied at the
same ordering cannot create SOF. SOF remains conditional on transitions retained
within the selected furcation. DFFP horizons count discounted graph steps, not
physical elapsed time, and these associations alone do not prove irreversible
biological commitment.

Instantaneous mode remains the default for backward compatibility. In two-fate
instantaneous geometry, cosine affinity measures the sign of a collinear
displacement and cannot distinguish mixture strength along that ray. The
biological tutorials explicitly choose future-fate mode.

Export and reload validate unique cell IDs. New saves retain the full graph
axis and can be loaded against a reordered copy of the same full cell set.
Reordering the scorer's live AnnData requires reload or rebuilding. Partial
legacy saves lacking the full cell axis must be rebuilt.

Discounting and relative fate preference
----------------------------------------

DFFP weights a path of length ``t`` by ``gamma**t``. Thus an equally probable
nearer fate receives more weight than a farther fate; normalizing to CFA does
not remove this distance effect. A horizon is measured in graph steps, not
physical time. Report horizon and anchor sensitivity before interpreting CFA
changes as altered biological fate preference.

Large-input allocation scope
----------------------------

The scalability examples measure numerical transforms and synthetic sparse
solves. They exclude AnnData duplication, identifier arrays, score export and
RNA-velocity preprocessing. They do not establish an end-to-end 200M-cell
biological workflow. Build with ``cache_subset=False`` and use
``score(write_to_adata=False)`` to avoid subset/export copies. To export while
reducing metadata duplication, use ``score(cache_subset=False,
compact_metadata=True)``; full-length score arrays still require memory.
