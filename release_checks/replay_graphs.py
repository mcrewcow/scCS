from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import scCS

import argparse

parser = argparse.ArgumentParser(
    description="Independently validate scCS on compact full-graph audit inputs."
)
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path("release_checks"))
args = parser.parse_args()
source = args.input_dir
args.output_dir.mkdir(parents=True, exist_ok=True)
rows = []
for path in sorted(source.glob("*_graph.h5ad")):
    a = ad.read_h5ad(path)
    name = a.uns["audit_input"]["run"]
    pancreas = name.startswith("pancreas")
    root = ["Ngn3 high EP", "Pre-endocrine"] if pancreas else "Common Progenitor"
    fates = (
        ["Alpha", "Beta", "Delta", "Epsilon"]
        if pancreas
        else ["Gut", "Gut neuron", "ChC"]
    )
    ordering = (
        ("latent_time" if name == "pancreas" else "velocity_pseudotime")
        if pancreas
        else "inverse_cytotrace_pseudotime"
    )
    s = scCS.SingleScorer(
        a, root=root, branches=fates, obs_key=a.uns["audit_input"]["obs_key"]
    )
    s.build_embedding(
        ordering_metric=ordering,
        verbose=False,
        cache_subset=False,
        write_to_adata=False,
    )
    T = a.obsp["source_transition"].tocsr()
    s.fit_future_fate(
        T, effective_horizon=64, anchor_quantile=0.9, min_anchor_cells=10, verbose=False
    )
    r = s.score(write_to_adata=False, verbose=False)
    # Independent boundary-clamped fixed-point calculation on the full source graph.
    boundary = r.outcome_anchor_mask.astype(float)
    absorbing = boundary.any(axis=1)
    values = boundary.copy()
    gamma = 64 / 65
    for iteration in range(10000):
        update = gamma * (T @ values)
        update[absorbing] = boundary[absorbing]
        change = np.max(np.abs(update - values))
        values = update
        if change < 1e-12:
            break
    else:
        raise AssertionError("Independent replay did not converge")
    expected = values[r.embedding.selected_indices, : r.k]
    np.testing.assert_allclose(r.selected_probability, expected, atol=1e-9, rtol=1e-9)
    # Independent ordering rank and row-normalized induced path operator.
    pt = r.embedding.selected_ordering_values
    ranks = (
        np.zeros(len(pt))
        if np.ptp(pt) == 0
        else (rankdata(pt, method="average") - 1) / (len(pt) - 1)
    )
    selected = r.embedding.selected_indices
    induced = T[selected][:, selected].tocsr()
    mass = np.asarray(induced.sum(axis=1)).ravel()
    rowmean = np.divide(induced @ ranks, mass, out=ranks.copy(), where=mass > 0)
    np.testing.assert_allclose(r.signed_ordering_flux, rowmean - ranks, atol=1e-12)
    report = s.preflight(ordering_metric=ordering).to_frame()
    rows.append(
        dict(
            run=name,
            n_cells=a.n_obs,
            n_selected=r.n_cells,
            n_root=r.n_root,
            independent_probability_max_error=float(
                np.max(np.abs(r.selected_probability - expected))
            ),
            independent_iterations=iteration + 1,
            affinity_defined_fraction=float(r.affinity_defined[r.root_mask].mean()),
            root_mean_reach=float(r.selected_reach[r.root_mask].mean()),
            status="PASS",
        )
    )
    report.to_csv(args.output_dir / f"{name}_preflight.csv", index=False)
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(args.output_dir / "real_graph_replays.csv", index=False)
