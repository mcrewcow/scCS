"""Observation identity checks shared by exports, plots, and persistence."""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def observation_ids(adata) -> np.ndarray:
    ids = np.asarray(adata.obs_names).astype(str)
    if not pd.Index(ids).is_unique:
        raise ValueError(
            "AnnData observation IDs must be unique; duplicate cell IDs were found."
        )
    return ids


def cell_indices(adata, cell_ids) -> np.ndarray:
    """Map result-order cells into a target, rejecting missing/duplicate IDs."""
    ids = np.asarray(cell_ids).astype(str)
    if not pd.Index(ids).is_unique:
        raise ValueError("Result cell IDs must be unique.")
    index = pd.Index(observation_ids(adata)).get_indexer(ids)
    if np.any(index < 0):
        missing = ids[index < 0][:5].tolist()
        raise ValueError(
            f"Target AnnData is missing result cell IDs, for example {missing}."
        )
    return index.astype(int)


def observation_fingerprint(cell_ids) -> str:
    """Hash order and identity without delimiter ambiguity."""
    digest = hashlib.sha256()
    for value in np.asarray(cell_ids).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def transition_fingerprint(matrix) -> str:
    """Fingerprint the exact normalized CSR operator, including its axis size."""
    from scipy import sparse

    matrix = sparse.csr_matrix(matrix)
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    for values in (matrix.indptr, matrix.indices, matrix.data):
        canonical = np.ascontiguousarray(
            values, dtype="<f8" if values.dtype.kind == "f" else "<i8"
        )
        digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()
