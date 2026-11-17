"""Cache reuse and dataset integrity contracts for public notebooks."""

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
from scCS import datasets
from scCS import tutorial_cache as cache


def test_tutorial_cache_roundtrip_invalidates_versions_and_incomplete_graph(
    tmp_path, monkeypatch
):
    signature = {"schema": 1, "dataset": "pancreas", "dependencies": {"scvelo": "test"}}
    monkeypatch.setattr(cache, "cache_signature", lambda *args: signature)
    a = ad.AnnData(np.ones((3, 2)))
    with pytest.raises(ValueError, match="completed velocity"):
        cache.save_tutorial_cache(a, "pancreas", cache_dir=tmp_path)
    a.layers["velocity"] = np.ones(a.shape)
    a.uns["velocity_graph"] = sparse.eye(3, format="csr")
    cache.save_tutorial_cache(a, "pancreas", cache_dir=tmp_path)
    assert cache.load_tutorial_cache("pancreas", cache_dir=tmp_path).shape == (3, 2)
    monkeypatch.setenv("SCCS_RECOMPUTE_VELOCITY", "1")
    assert cache.load_tutorial_cache("pancreas", cache_dir=tmp_path) is None
    monkeypatch.delenv("SCCS_RECOMPUTE_VELOCITY")
    monkeypatch.setattr(
        cache, "cache_signature", lambda *args: {**signature, "schema": 2}
    )
    assert cache.load_tutorial_cache("pancreas", cache_dir=tmp_path) is None


def test_schwann_loader_detects_unexpected_input_bytes(tmp_path):
    import hashlib

    a = ad.AnnData(np.ones((2, 2)))
    path = tmp_path / "different.h5ad"
    a.write_h5ad(path)
    with pytest.raises(ValueError, match="checksum"):
        datasets.schwann(path)
    assert datasets.schwann(path, verify_checksum=False).shape == a.shape
    assert len(datasets.SCHWANN_SHA256) == hashlib.sha256().digest_size * 2
