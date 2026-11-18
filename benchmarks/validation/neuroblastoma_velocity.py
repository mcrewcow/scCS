"""Fit a matched tumor velocity graph from per-tumor looms and verified barcode mappings."""

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse

import scCS
from .velocity_inputs import audit_counts


def load_counts(expression, loom_dir, mapping):
    """Match each human cell to one loom column within the same tumor."""
    keys = ["tumor", "loom_cell_id", "cell_id", "species"]
    if not set(keys).issubset(mapping):
        raise ValueError(f"Mapping columns must include {keys}")
    wanted = mapping[mapping.cell_id.isin(expression.obs_names)].copy()
    if (
        wanted[keys].isna().any().any()
        or wanted.cell_id.duplicated().any()
        or wanted.duplicated(["tumor", "loom_cell_id"]).any()
    ):
        raise ValueError("The matched barcode mapping must be complete and one to one")
    wanted = wanted.set_index("cell_id").reindex(expression.obs_names)
    if (
        not wanted.species.eq("human").all()
        or not wanted.tumor.eq(expression.obs.tumor.astype(str)).all()
    ):
        raise ValueError(
            "Every modeled cell needs a species-verified human mapping from its own tumor"
        )
    parts = []
    for tumor, rows in wanted.groupby("tumor"):
        with h5py.File(Path(loom_dir) / (tumor + ".loom")) as loom:
            ids = pd.Index(loom["col_attrs/CellID"].asstr()[:])
            if not ids.is_unique:
                raise ValueError(f"Duplicate loom cell IDs in {tumor}")
            pos = ids.get_indexer(rows.loom_cell_id)
            if (pos < 0).any():
                raise ValueError(f"Mapped cells missing from {tumor}.loom")
            order = np.argsort(pos)
            positions = pos[order]
            genes = pd.Index(loom["row_attrs/Accession"].asstr()[:]).str.replace(
                r"\.\d+$", "", regex=True
            )
            human = genes.isin(expression.var_names)
            if genes[human].has_duplicates or not human.any():
                raise ValueError(f"Human feature IDs are absent or duplicated in {tumor}.loom")
            layers = {}
            for key in ["spliced", "unspliced"]:
                matrix = loom["layers/" + key]
                blocks = []
                for i in range(0, len(genes), 256):
                    take = human[i : i + 256]
                    if take.any():
                        block = matrix[i : i + 256][:, positions][take]
                        blocks.append(sparse.csr_matrix(block, dtype="float32"))
                layers[key] = sparse.vstack(blocks).T.tocsr()
            part = ad.AnnData(
                layers["spliced"].copy(),
                obs=expression.obs.loc[rows.index[order]].copy(),
                var=expression.var.loc[genes[human]].copy(),
                layers=layers,
            )
            audit_counts(part)
            parts.append(part)
    counts = ad.concat(parts, join="inner", merge="same")[expression.obs_names].copy()
    audit_counts(counts)
    return counts


def fit_graph(counts):
    from .real_pair_cases import model_genes

    np.random.seed(19)
    fit = model_genes(counts)
    scv.pp.filter_genes(fit, min_shared_counts=10)
    raw = fit.copy()
    scv.pp.normalize_per_cell(fit)
    sc.pp.log1p(fit)
    sc.pp.pca(fit, n_comps=30, random_state=19)
    sc.pp.neighbors(fit, n_neighbors=30, n_pcs=30, random_state=19)
    sc.tl.umap(fit, random_state=19)
    scv.pp.moments(fit, n_neighbors=None, n_pcs=None)
    scv.tl.recover_dynamics(fit, n_top_genes=250, max_iter=20, n_jobs=1, show_progress_bar=False)
    scv.tl.velocity(fit, mode="dynamical")
    scv.tl.velocity_graph(fit, n_jobs=1, show_progress_bar=False)
    scv.tl.velocity_confidence(fit)
    graph = scCS.get_scvelo_transition_matrix(fit)
    if (
        not np.isfinite(graph.data).all()
        or (graph.data < 0).any()
        or (graph - sparse.diags(graph.diagonal())).nnz == 0
    ):
        raise ValueError("Velocity graph contains invalid or only self transitions")
    np.testing.assert_allclose(np.asarray(graph.sum(axis=1)).ravel(), 1, atol=1e-6)
    fit.obsp["T_velocity"] = graph
    for key in ["spliced", "unspliced"]:
        fit.layers["raw_" + key] = raw.layers[key]
    return fit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expression", required=True)
    p.add_argument("--loom-dir", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    a = ad.read_h5ad(args.expression)
    x = load_counts(a, args.loom_dir, pd.read_csv(args.mapping, sep="\t", dtype=str))
    fit = fit_graph(x)
    print(fit.obs.groupby("tumor", observed=True).velocity_confidence.agg(["size", "median"]))
    fit.write_h5ad(args.out, compression="gzip")
