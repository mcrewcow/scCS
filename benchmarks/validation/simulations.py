"""Prescribed-vector manifold simulations for scCS validation notebooks.

All simulators place cells on a 2-D manifold (kept in ``obsm["X_2d"]``),
embed it in D dimensions with noise, attach a velocity vector per cell, and
build an scVelo-like transition matrix in ``obsp["T_velocity"]``:

    cosine(v_i, x_j - x_i) over k nearest neighbours,
    weight expm1(scale * cos) on positive cosines,
    confidence-based self-loops (scVelo's self_transitions rule),
    row normalisation.

This mimics, but is not, scVelo. Every function is deterministic given ``seed``.
Pass ``obsp["T_velocity"]`` explicitly to ``scorer.fit(...)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from sklearn.neighbors import NearestNeighbors
import anndata as ad


# ------------------------------------------------------------------ kernel
def velocity_transition_matrix(X, V, *, k=30, scale=10.0, self_transitions=True):
    """scVelo-like row-stochastic transition matrix from positions X and velocities V."""
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    idx = idx[:, 1:]
    disp = X[idx] - X[:, None, :]
    cos = np.einsum("nkd,nd->nk", disp, V) / (
        np.linalg.norm(disp, axis=2) * np.linalg.norm(V, axis=1)[:, None] + 1e-12
    )
    G = sparse.csr_matrix(
        (
            np.expm1(scale * np.clip(cos, 0, None)).ravel(),
            (np.repeat(np.arange(len(X)), k), idx.ravel()),
        ),
        shape=(len(X), len(X)),
    )
    if self_transitions:
        rs = np.asarray(G.sum(1)).ravel()
        rs[rs == 0] = 1
        conf = np.clip(cos.max(1), 0, None)
        G = sparse.diags(1 / rs) @ G + sparse.diags(np.clip(np.percentile(conf, 98) - conf, 0, 1))
    rs = np.asarray(G.sum(1)).ravel()
    rs[rs == 0] = 1
    return (sparse.diags(1 / rs) @ G).tocsr()


def _embed(rng, X2, V2, D=20, noise=0.06, vel_noise=0.035):
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    return (
        X2 @ Q[:2] + noise * rng.normal(size=(len(X2), D)),
        V2 @ Q[:2] + vel_noise * rng.normal(size=(len(X2), D)),
    )


def _finish(
    rng,
    X2,
    V2,
    labels,
    time=None,
    *,
    k=30,
    scale=10.0,
    self_transitions=True,
    D=20,
    noise=0.06,
    vel_noise=0.035,
    extra_obs=None,
):
    X, V = _embed(rng, X2, V2, D=D, noise=noise, vel_noise=vel_noise)
    obs = pd.DataFrame(
        {"clusters": pd.Categorical(labels)},
        index=[f"c{i}" for i in range(len(labels))],
    )
    if time is not None:
        obs["true_time"] = time
        obs["latent_time"] = np.clip(time + 0.03 * rng.normal(size=len(time)), 0, None)
    for key, value in (extra_obs or {}).items():
        obs[key] = value
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs)
    adata.obsm["X_2d"] = X2
    adata.obsm["V_2d"] = V2
    adata.layers["velocity"] = V.astype(np.float32)
    adata.obsp["T_velocity"] = velocity_transition_matrix(
        X, V, k=k, scale=scale, self_transitions=self_transitions
    )
    return adata


# ------------------------------------------------------------------ simulators
def make_furcation(
    branches=(("A", 1.0, 1.0, False), ("B", 1.0, 1.0, False)),
    *,
    n_root=600,
    n_per_unit=400,
    root_len=1.0,
    outside=None,
    seed=0,
    angles=None,
    **kernel,
):
    """Root segment feeding K straight branches.

    branches: tuples (name, length, density_weight, retrograde).
      density_weight scales cells per unit length (nominal entry-density share ~ weight share);
      retrograde=True reverses the branch velocity (flow back toward the root).
    outside: optional dict(name=..., n=..., attach=0.5, length=0.8) adds an
      unselected population leaving the root mid-point (a leak / competing outcome).
    kernel: forwarded to the transition kernel (k, scale, self_transitions, vel_noise, ...).
    """
    rng = np.random.default_rng(seed)
    K = len(branches)
    angles = np.linspace(-0.6, 0.6, K) if angles is None else np.asarray(angles)
    longest = max(b[1] for b in branches)
    s = rng.uniform(0, root_len, n_root)
    pts, vel = [np.c_[s - root_len, 0 * s]], [np.tile([1.0, 0.0], (n_root, 1))]
    lab, t = ["Root"] * n_root, [0.3 * s / root_len]
    for (name, length, weight, retro), a in zip(branches, angles):
        n = int(n_per_unit * length * weight)
        u = rng.uniform(0, length, n)
        d = np.array([np.cos(a), np.sin(a)])
        pts.append(np.outer(u, d))
        vel.append(np.tile(d, (n, 1)) * (-1.0 if retro else 1.0))
        lab += [name] * n
        t.append(0.3 + 0.7 * u / longest)
    if outside:
        n = outside["n"]
        u = rng.uniform(0, outside.get("length", 0.8), n)
        base = np.array([-root_len * outside.get("attach", 0.5), 0.0])
        pts.append(base + np.outer(u, [0.0, -1.0]))
        vel.append(np.tile([0.0, -1.0], (n, 1)))
        lab += [outside["name"]] * n
        t.append(0.15 + 0.3 * u)
    return _finish(rng, np.vstack(pts), np.vstack(vel), lab, np.concatenate(t), **kernel)


def _profile(rng, n_base, length, rate, onset):
    grid = np.linspace(0, length, 4001)
    dens = np.where(grid < onset, 1.0, np.exp(rate * (grid - onset)))
    cdf = np.r_[0, np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(grid))]
    return np.interp(rng.uniform(0, cdf[-1], int(round(n_base * cdf[-1]))), cdf, grid)


def make_fork(
    rate_A=0.0,
    onset_A=0.0,
    rate_B=0.0,
    onset_B=0.0,
    density_A=1.0,
    *,
    n_root=800,
    n_entry=400,
    length=1.5,
    seed=0,
    **kernel,
):
    """Symmetric two-branch fork with post-entry proliferation (rate>0) or death (rate<0).

    Entry density at u=0 is n_entry*density_A for A and n_entry for B, so the nominal
    entry-density share toward A is density_A / (1 + density_A) regardless of rates/onsets.
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n_root)
    pts, vel = [np.c_[s - 1, 0 * s]], [np.tile([1.0, 0.0], (n_root, 1))]
    lab, t = ["Root"] * n_root, [0.3 * s]
    for name, ang, rate, onset, w in (
        ("A", -0.6, rate_A, onset_A, density_A),
        ("B", 0.6, rate_B, onset_B, 1.0),
    ):
        u = _profile(rng, n_entry * w, length, rate, onset)
        d = np.array([np.cos(ang), np.sin(ang)])
        pts.append(np.outer(u, d))
        vel.append(np.tile(d, (len(u), 1)))
        lab += [name] * len(u)
        t.append(0.3 + 0.7 * u / length)
    return _finish(rng, np.vstack(pts), np.vstack(vel), lab, np.concatenate(t), **kernel)


def make_reversible_loop(w_res=1.0, *, n_blob=800, n_arc=400, seed=0, **kernel):
    """Homeostatic and Activated steady-state blobs (random velocities) joined by an
    activation arc (Transitory, H->A, top) and a resolution arc (Resolving, A->H,
    bottom; density scaled by w_res, 0 = irreversible activation).

    obs["activation_score"] ~ x-coordinate; obs["outward"] = |activation_score|
    increases from the arcs toward both blob cores (usable as a scCS ordering).
    """
    rng = np.random.default_rng(seed)
    pts, vel, lab = [], [], []
    for name, cx in (("Homeostatic", -1.25), ("Activated", 1.25)):
        v = rng.normal(size=(n_blob, 2))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts.append(np.c_[cx + 0.28 * rng.normal(size=n_blob), 0.28 * rng.normal(size=n_blob)])
        vel.append(v)
        lab += [name] * n_blob
    for name, sign, n in (
        ("Transitory", 1, n_arc),
        ("Resolving", -1, int(round(n_arc * w_res))),
    ):
        if n:
            th = rng.uniform(0.08 * np.pi, 0.92 * np.pi, n)
            tan = np.c_[sign * 1.05 * np.sin(th), -0.75 * np.cos(th)]
            pts.append(np.c_[1.05 * np.cos(th), sign * 0.75 * np.sin(th)])
            vel.append(tan / np.linalg.norm(tan, axis=1, keepdims=True))
            lab += [name] * n
    X2 = np.vstack(pts)
    score = X2[:, 0] + 0.05 * rng.normal(size=len(X2))
    return _finish(
        rng,
        X2,
        np.vstack(vel),
        lab,
        extra_obs={"activation_score": score, "outward": np.abs(score)},
        **kernel,
    )


def make_conditions(n_reps=4, effect=0.0, *, seed=0, n_root=250, n_per_unit=160, **kw):
    """Two conditions x n_reps biological replicates of a two-fate furcation.

    'treated' multiplies A's density weight by (1 + effect); effect=0 is the null.
    Replicates are separate samples, so the pooled graph is block-diagonal.
    obs: clusters, condition (categorical control/treated), sample_id, latent_time.
    """
    parts = []
    for c_index, (cond, wA) in enumerate((("control", 1.0), ("treated", 1.0 + effect))):
        for r in range(n_reps):
            a = make_furcation(
                branches=(("A", 1.0, wA, False), ("B", 1.0, 1.0, False)),
                n_root=n_root,
                n_per_unit=n_per_unit,
                seed=seed * 10_000 + c_index * 100 + r,
                **kw,
            )
            a.obs["condition"] = cond
            a.obs["sample_id"] = f"{cond}_{r}"
            a.obs.index = [f"{cond}{r}_{i}" for i in range(a.n_obs)]
            parts.append(a)
    obs = pd.concat([p.obs for p in parts])
    obs["clusters"] = pd.Categorical(obs["clusters"].astype(str))
    obs["condition"] = pd.Categorical(obs["condition"], categories=["control", "treated"])
    out = ad.AnnData(X=np.vstack([p.X for p in parts]), obs=obs)
    out.obsm["X_2d"] = np.vstack([p.obsm["X_2d"] for p in parts])
    out.obsp["T_velocity"] = sparse.block_diag([p.obsp["T_velocity"] for p in parts]).tocsr()
    return out


# ------------------------------------------------------------------ prototypes / helpers
def dead_end_cells(T, tol=1e-9):
    """Cells whose only transition is a self-loop (no forward neighbour)."""
    return np.asarray(sparse.csr_matrix(T).diagonal()) > 1 - tol


def make_dead_ends(T, cells):
    """Return a copy of T where `cells` become pure self-loops (stress test for NaN CFA)."""
    T = sparse.lil_matrix(T, copy=True)
    for i in np.asarray(cells, int):
        T.rows[i] = [int(i)]
        T.data[i] = [1.0]
    return sparse.csr_matrix(T)


def repair_dead_ends(T, X, k=30):
    """Replace pure self-loop rows with uniform steps to the k nearest neighbours."""
    dead = np.flatnonzero(dead_end_cells(T))
    T = sparse.lil_matrix(T, copy=True)
    if dead.size:
        _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[dead])
        for i, nb in zip(dead, idx[:, 1:]):
            T.rows[i] = sorted(int(j) for j in nb)
            T.data[i] = [1.0 / k] * k
    return sparse.csr_matrix(T), dead


def discounted_occupancy(T, masks, effective_horizon=64):
    """PROTOTYPE (not in scCS): (1-g)(I-gP)^-1 1_S with g=h/(h+1).

    Expected discounted fraction of future time in each state set, with no
    absorbing anchors. Rows sum to 1 when `masks` partition all cells.
    """
    g = effective_horizon / (effective_horizon + 1.0)
    lu = splu(sparse.eye(T.shape[0], format="csc") - g * sparse.csc_matrix(T))
    B = np.column_stack([np.asarray(m, float) for m in masks.values()]) * (1 - g)
    return pd.DataFrame(lu.solve(B), columns=list(masks))
