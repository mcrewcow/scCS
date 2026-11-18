Non-trivial questions
=====================

After learning the standard scoring workflow, the next questions concern what
a score means when sampling, geometry or biological dynamics become difficult.
Does an expanded branch indicate a changed fate decision? Can cells move against
an ordering or return to an earlier state? When should two fate-mapping methods
agree? These notebooks turn those questions into explicit experiments.

Each notebook introduces the motivation, states the assumptions and reference
quantity, shows the executed result, and explains the limits of the conclusion.
Read CFA together with discounted fate reach (DFR), unresolved probability (UFP)
and defined-affinity coverage. A conditional preference among a few reachable
cells should not be interpreted as a prediction for the entire root population.

Questions explored
------------------

* **Known answers:** does the solver recover analytical first-hit probabilities?
* **Abundance:** how do branch expansion and sampling density affect root affinity?
* **Horizon and anchors:** which outcome, at what graph distance, is being measured?
* **Complex dynamics:** can identity, progression, return and competing outcomes differ?
* **Uncertainty:** what requires graph resampling, and what requires biological replicates?
* **Method comparison:** how does discounted reach relate to eventual absorption?

The analytical examples have exact graph-level references. The manifold examples
use prescribed vectors and sampled coordinates; they do not simulate splicing
counts or establish the accuracy of an RNA-velocity model. Their purpose is to
test numerical and interpretive claims under stated assumptions. For complete
workflows starting from genuine splicing data, see :doc:`tutorials_practical`.

.. toctree::
   :maxdepth: 1

   tutorials/13_validation_exact_graphs
   tutorials/14_validation_abundance
   tutorials/15_validation_horizon_anchors
   tutorials/16_validation_complex_dynamics
   tutorials/17_validation_uncertainty_inference
   tutorials/20_cellrank_matched_comparison
