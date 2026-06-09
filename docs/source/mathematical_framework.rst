Mathematical Framework
======================

This page provides full mathematical derivations for all scCS scoring
metrics and statistical tests. For practical usage, see the tutorial
notebooks.

Radial Star Embedding
---------------------

Given a progenitor cluster (root) and k terminal fate clusters (branches),
scCS constructs a 2D radial star embedding where:

- The progenitor population is placed at the origin.
- Each fate arm extends from the origin at angle:

.. math::

    \alpha_j = \frac{2\pi j}{k} \quad (j = 0, 1, \ldots, k-1)

for equal spacing, or at the centroid-derived angle for centroid-based
sector assignment.

- Cells are positioned along their assigned arm at radial distance
  proportional to their differentiation metric (pseudotime, CytoTRACE2,
  or any custom score):

.. math::

    r_i = m_i \cdot s

where :math:`m_i \in [0, 1]` is the scaled metric value and :math:`s` is
the arm scale parameter (default: 10.0).

The resulting 2D coordinates are stored in ``adata.obsm['X_sccs']``.

Velocity Projection
-------------------

Per-cell RNA velocity vectors :math:`(v_x^{\text{orig}}, v_y^{\text{orig}})`
from scVelo are projected into the star embedding coordinate system:

.. math::

    v_x^{\text{sccs}} = \sum_j v_j^{\text{orig}} \cdot w_{xj}

.. math::

    v_y^{\text{sccs}} = \sum_j v_j^{\text{orig}} \cdot w_{yj}

where :math:`w_{xj}, w_{yj}` are the projection weights derived from the
PCA loadings of the star embedding space.

Three projection strategies are available:

1. **PCA projection** — project velocity through the same PCA loadings used
   to construct the star embedding.
2. **Delta expression** — compute velocity as the difference between spliced
   and inferred full (spliced + unspliced) expression, then project.
3. **Cosine similarity** — compute per-cell velocity direction as cosine
   similarity with each arm direction.

Commitment Scores
-----------------

**Magnitude** (Eq. 1):

.. math::

    \text{magnitude}_i = \sqrt{v_{x,i}^2 + v_{y,i}^2}

**Angle** (Eq. 2–3):

.. math::

    \theta_i = \text{atan2}(v_{y,i}, v_{x,i}) \quad \in [0°, 360°)

**Angular binning** (Eq. 4–6):

Angles are binned into :math:`N` equal sectors of width :math:`360°/N`
(default :math:`N = 36`, i.e., 10° per bin). Each cell contributes its
magnitude to the corresponding bin:

.. math::

    M_{\text{bin}}(b) = \sum_{i: \theta_i \in \text{bin}_b} \text{magnitude}_i

**Sector assignment**:

Each fate arm :math:`j` is assigned a set of bins (a sector). Two methods:

- **Equal sectors**: each fate gets exactly :math:`N/k` consecutive bins.
- **Centroid sectors**: bins are assigned to the fate whose centroid direction
  is closest to the bin center.

**Sector magnitude**:

.. math::

    M_{\text{sector}}(j) = \sum_{b \in \text{sector}_j} M_{\text{bin}}(b)

**Unnormalized Commitment Score** (Eq. 8):

.. math::

    \text{unCS}(i, j) = \frac{M_{\text{sector}}(i)}{M_{\text{sector}}(j)}

Values > 1 indicate stronger commitment to fate :math:`i` than fate :math:`j`.

**Normalized Commitment Score** (Eq. 9):

.. math::

    \text{nCS}(i, j) = \text{unCS}(i, j) \times \frac{n_{\text{cells}}(j)}{n_{\text{cells}}(i)}

This corrects for differences in population size. The manuscript values
are :math:`\text{unCS}(0,1) = 9.335` and :math:`\text{nCS}(0,1) = 8.066`.

**Commitment vector**:

.. math::

    p_j = \frac{M_{\text{sector}}(j)}{\sum_{l} M_{\text{sector}}(l)}

This is a probability distribution over fates (sums to 1).

Per-Cell Fate Affinity
----------------------

Per-cell fate affinity scores are computed from the cosine similarity
between each cell's velocity vector and the unit direction toward each
fate centroid, shifted to [0, 1]:

.. math::

    s_{ij} = \frac{1}{2}\left(1 + \frac{\vec{v}_i \cdot \hat{d}_j}{|\vec{v}_i|}\right)

where :math:`\hat{d}_j` is the unit vector from the root centroid toward
fate :math:`j`'s centroid.

**Magnitude weighting**: Cells with near-zero velocity magnitude (typically
progenitors at the origin) are blended toward the uniform distribution:

.. math::

    s_{ij}^{\text{weighted}} = w_i \cdot s_{ij} + (1 - w_i) \cdot \frac{1}{k}

where :math:`w_i = 1` if :math:`\text{magnitude}_i > q_{\alpha}` (5th percentile
threshold), else :math:`w_i = \text{magnitude}_i / q_{\alpha}`.

**Row normalization**: Scores are normalized to sum to 1 per cell:

.. math::

    s_{ij}^{\text{final}} = \frac{s_{ij}^{\text{weighted}}}{\sum_l s_{il}^{\text{weighted}}}

Entropy Metrics
---------------

**Population entropy**:

.. math::

    H_{\text{pop}} = -\frac{\sum_{j=1}^{k} p_j \log(p_j)}{\log(k)}

where :math:`p_j = M_{\text{sector}}(j) / \sum_l M_{\text{sector}}(l)`.
Normalized to [0, 1] by dividing by :math:`\log(k)`.

**Mean cell entropy** (primary metric):

.. math::

    H_{\text{cell}} = \frac{1}{n}\sum_{i=1}^{n} \left[-\frac{\sum_{j=1}^{k} s_{ij} \log(s_{ij})}{\log(k)}\right]

Each cell's entropy is normalized by :math:`\log(k)` to [0, 1], then averaged.

**Per-fate cell entropy**:

For each fate :math:`j`, compute the mean binary entropy of the affinity
score treated as a Bernoulli distribution:

.. math::

    h_j = \frac{1}{n}\sum_{i=1}^{n} H_{\text{bin}}(s_{ij}, 1 - s_{ij})

where:

.. math::

    H_{\text{bin}}(p, 1-p) = -\frac{p \log(p) + (1-p) \log(1-p)}{\log(2)}

**NN-smoothed per-cell entropy**:

For each cell :math:`i`, average the cell scores over its :math:`k_{\text{nn}}`
nearest neighbors in the scCS embedding:

.. math::

    \bar{s}_{ij} = \frac{1}{k_{\text{nn}}} \sum_{n \in \text{NN}(i)} s_{nj}

Then compute k-way Shannon entropy on the smoothed scores:

.. math::

    H_{\text{nn},i} = -\frac{\sum_{j=1}^{k} \bar{s}_{ij} \log(\bar{s}_{ij})}{\log(k)}

Statistical Framework
---------------------

Pairwise comparison (PairScorer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Permutation test** (default for k=2 conditions):

For each fate arm, test whether per-cell affinity scores differ between
conditions A and B:

1. Compute observed mean difference: :math:`\Delta_0 = \bar{s}_A - \bar{s}_B`
2. For :math:`b = 1, \ldots, B` (default B=1000):
   - Shuffle condition labels
   - Recompute mean difference :math:`\Delta_b`
3. Empirical p-value: :math:`p = \frac{\#(|\Delta_b| \geq |\Delta_0|) + 1}{B + 1}`

**Delta-CS with bootstrap CI**:

.. math::

    \Delta\text{nCS}(i,j) = \text{nCS}_A(i,j) - \text{nCS}_B(i,j)

Bootstrap CI obtained by resampling cells within each condition
:math:`B` times (default 500) and computing the empirical
:math:`(1-\alpha)/2` and :math:`(1+\alpha)/2` quantiles.

Multi-comparison (MultiScorer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Omnibus tests**:

For each fate arm, test whether per-cell affinity scores differ across
ALL conditions simultaneously.

*Kruskal-Wallis H test* (non-parametric, recommended):

.. math::

    H = \frac{12}{N(N+1)} \sum_{g=1}^{G} n_g \left(\bar{R}_g - \frac{N+1}{2}\right)^2

where :math:`N` is total cells, :math:`n_g` is cells in group :math:`g`,
and :math:`\bar{R}_g` is the mean rank of group :math:`g`.

*One-way ANOVA* (parametric):

.. math::

    F = \frac{\text{SSB} / (G-1)}{\text{SSW} / (N-G)}

**Post-hoc pairwise comparisons**:

Only meaningful after an omnibus test rejects :math:`H_0`.

*Dunn's test* (non-parametric, recommended with Kruskal-Wallis):

Uses rank-based pairwise comparisons with multiple testing correction.
Implemented via ``scikit-posthocs``.

*Tukey HSD* (parametric, for balanced designs):

.. math::

    q = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{\text{MSW} / n}}

*Conover-Iman test* (more powerful than Dunn, non-parametric):

Uses rank-based pairwise t-statistics with multiple testing correction.
Implemented via ``scikit-posthocs``.

**Multiple testing correction**:

*Benjamini-Hochberg FDR*:

1. Sort p-values: :math:`p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(m)}`
2. Adjusted p-value: :math:`p_{(i)}^{\text{adj}} = \min\left(\frac{m \cdot p_{(i)}}{i}, 1\right)`
3. Enforce monotonicity from right to left.

*Bonferroni*:

.. math::

    p^{\text{adj}} = \min(m \cdot p, 1)

*Holm-Bonferroni step-down*:

1. Sort p-values ascending.
2. Adjusted p-value: :math:`p_{(i)}^{\text{adj}} = \max\left(p_{(i-1)}^{\text{adj}}, \frac{(m - i + 1) \cdot p_{(i)}}{1}\right)`
3. Enforce monotonicity from left to right.

Mixed-effects models
~~~~~~~~~~~~~~~~~~~~

**Linear mixed model** (PairScorer + MultiScorer):

.. math::

    y_{ij} = \beta_0 + \beta_1 \cdot \text{condition}_i + u_{\text{sample}(i)} + \epsilon_{ij}

where :math:`y_{ij}` is the affinity score of cell :math:`i` for fate :math:`j`,
:math:`u_{\text{sample}} \sim N(0, \sigma^2_u)` is the random intercept for
biological replicate, and :math:`\epsilon_{ij} \sim N(0, \sigma^2)`.

**Contrast testing** (MultiScorer):

For a contrast between conditions A and B:

.. math::

    \hat{\delta} = \hat{\beta}_A - \hat{\beta}_B

.. math::

    \text{SE}(\hat{\delta}) = \sqrt{\text{SE}(\hat{\beta}_A)^2 + \text{SE}(\hat{\beta}_B)^2}

.. math::

    z = \frac{\hat{\delta}}{\text{SE}(\hat{\delta})}, \quad p = 2 \cdot \Phi(-|z|)

Trajectory shift
~~~~~~~~~~~~~~~~

For each fate arm, test whether pseudotime distributions differ across
conditions:

*Kolmogorov-Smirnov test*:

.. math::

    D = \sup_x |F_A(x) - F_B(x)|

*Wasserstein distance*:

.. math::

    W_1 = \int_0^1 |F_A^{-1}(u) - F_B^{-1}(u)| du

Bootstrap CI on :math:`W_1` obtained by resampling cells within each
condition.
