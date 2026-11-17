"""Versioned preprocessing caches shared by the public example notebooks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import tempfile

from ._version import VERSION
from .datasets import SCHWANN_SHA256


def _dependency_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def cache_signature(dataset, model="dynamical", max_iter=20):
    if dataset not in {"pancreas", "schwann"}:
        raise ValueError("The tutorial cache supports pancreas and schwann only.")
    return {
        "schema": 1,
        "scCS": VERSION,
        "dataset": dataset,
        "model": model,
        "recover_dynamics_max_iter": int(max_iter),
        "graph": (
            "pca30_neighbors30_seed20260714"
            if dataset == "pancreas"
            else "original_RegVelo"
        ),
        "source_sha256": (
            "9e3e459eca00ba06b496ec80def32941b5b2889918720e3e7aa6ffb811fbe7c6"
            if dataset == "pancreas"
            else SCHWANN_SHA256
        ),
        "dependencies": {
            name: _dependency_version(name)
            for name in [
                "numpy",
                "scipy",
                "pandas",
                "scanpy",
                "scvelo",
                "scikit-learn",
                "igraph",
                "leidenalg",
            ]
        },
    }


def load_tutorial_cache(
    dataset, model="dynamical", *, cache_dir="data/scCS_cache", max_iter=20
):
    """Return matching preprocessing, or None to execute the visible fitting code.

    Set SCCS_RECOMPUTE_VELOCITY=1 to bypass reuse deliberately. A cache is only
    for the unmodified public data, before any pseudo-condition perturbations.
    """
    if os.environ.get("SCCS_RECOMPUTE_VELOCITY") == "1":
        return None
    path = Path(cache_dir) / f"{dataset}_{model}.h5ad"
    if not path.exists():
        return None
    from anndata import read_h5ad

    adata = read_h5ad(path)
    if (
        adata.uns.get("sccs_tutorial_cache", {})
        != cache_signature(dataset, model, max_iter)
        or "velocity" not in adata.layers
        or ("velocity_graph" not in adata.uns and "velocity_graph" not in adata.obsp)
    ):
        print(
            f"[scCS] Preprocessing cache settings changed; refitting {dataset} {model}."
        )
        return None
    print(f"[scCS] Reusing {path}; set SCCS_RECOMPUTE_VELOCITY=1 to refit.")
    return adata


def save_tutorial_cache(
    adata, dataset, model="dynamical", *, cache_dir="data/scCS_cache", max_iter=20
):
    """Atomically store a completed public-data velocity fit and its signature."""
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset}_{model}.h5ad"
    if "velocity" not in adata.layers or (
        "velocity_graph" not in adata.uns and "velocity_graph" not in adata.obsp
    ):
        raise ValueError("Only completed velocity fits can be cached for tutorials.")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{dataset}_", suffix=".h5ad", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    adata.uns["sccs_tutorial_cache"] = cache_signature(dataset, model, max_iter)
    adata.write_h5ad(temporary, compression="gzip")
    temporary.replace(path)
    print(f"[scCS] Saved shared preprocessing cache: {path}")
    return path
