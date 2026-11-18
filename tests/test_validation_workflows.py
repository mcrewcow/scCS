"""Scientific and input-integrity contracts for the new benchmark workflows."""

import hashlib

import numpy as np
import pytest
from scipy import sparse

from benchmarks.validation import data, experiments, simulations
from benchmarks.validation.neuroblastoma_inputs import audit_bam, public_manifest


def test_analytical_graph_and_independent_walk_calibration():
    table = experiments.exact_calibration(n_walks=20_000, seed=231)
    assert table.solver_max_error.max() < 1e-10
    assert table.mass_error.max() < 1e-10
    assert table.simulation_max_error.max() < 0.03


def test_manifold_simulations_are_reproducible():
    a = simulations.make_fork(n_root=40, n_entry=30, seed=92)
    b = simulations.make_fork(n_root=40, n_entry=30, seed=92)
    assert np.array_equal(a.X, b.X)
    assert (a.obsp["T_velocity"] != b.obsp["T_velocity"]).nnz == 0


def test_versioned_download_rejects_existing_corrupt_bytes(tmp_path, monkeypatch):
    expected = b"versioned dataset"
    (tmp_path / "fixture.dat").write_bytes(expected)
    monkeypatch.setitem(
        data.SOURCES,
        "fixture",
        (
            "https://invalid.example/no-network",
            "fixture.dat",
            "sha256",
            hashlib.sha256(expected).hexdigest(),
        ),
    )
    assert data.fetch("fixture", tmp_path).read_bytes() == expected
    (tmp_path / "fixture.dat").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.fetch("fixture", tmp_path)


def test_neuroblastoma_expression_matrix_is_not_a_velocity_input():
    with pytest.raises(ValueError, match="aligned SAM/BAM"):
        audit_bam("counts.mtx", "barcodes.tsv", "annotation.gtf")
    manifest = public_manifest()
    assert manifest.groupby("condition").size().eq(2).all()
    assert not manifest.verified_public_velocity_layers.any()
    assert manifest.status.eq("NEEDS_TAGGED_BAM_OR_AUTHOR_VELOCITY_COUNTS").all()


def test_larry_excludes_multibarcode_future_test_cells():
    pytest.importorskip("cellrank")
    from benchmarks.validation.larry import exclude_heldout_future

    clones = sparse.csr_matrix([[1, 0], [1, 0], [0, 1], [1, 1]])
    time = np.array([2, 4, 6, 6])
    keep = exclude_heldout_future(time, clones, [0])
    assert keep.tolist() == [True, False, True, False]
    assert exclude_heldout_future(time, clones, []).all()
    with pytest.raises(ValueError, match="Time labels"):
        exclude_heldout_future(time[:-1], clones, [0])


def test_velocity_count_audit_rejects_expression_instead_of_raw_molecules():
    import anndata as ad
    from benchmarks.validation.velocity_inputs import audit_counts

    a = ad.AnnData(sparse.csr_matrix([[2, 0], [0, 4]], dtype=float))
    a.layers["spliced"] = a.X.copy()
    with pytest.raises(ValueError, match="Missing genuine"):
        audit_counts(a)
    a.layers["unspliced"] = sparse.csr_matrix([[1, 0], [0, 1]], dtype=float)
    assert audit_counts(a).total_counts.tolist() == [6.0, 2.0]
    a.layers["unspliced"][0, 0] = 0.25
    with pytest.raises(ValueError, match="raw molecule counts"):
        audit_counts(a)


def test_real_splicing_loom_roundtrip_and_cached_misalignment(tmp_path):
    pytest.importorskip("loompy")
    import anndata as ad
    from benchmarks.validation.velocity_inputs import write_verified_loom

    a = ad.AnnData(sparse.csr_matrix([[2, 0], [0, 4]], dtype=np.float32))
    a.obs_names = ["sample1:cellA", "sample2:cellA"]
    a.var_names = ["GeneA", "GeneB"]
    a.obs["condition"] = ["day2", "day4"]
    a.layers["spliced"] = a.X.copy()
    a.layers["unspliced"] = sparse.csr_matrix([[1, 0], [0, 2]], dtype=np.float32)
    path = write_verified_loom(a, tmp_path / "measured.loom")
    result = ad.read_loom(path, obs_names="CellID", var_names="Gene")
    assert result.obs_names.equals(a.obs_names)
    assert (result.layers["unspliced"] != a.layers["unspliced"]).nnz == 0
    assert result.obs.condition.tolist() == ["day2", "day4"]
    changed = a.copy()
    changed.obs_names = ["different:cellA", "sample2:cellA"]
    with pytest.raises(ValueError, match="different cell/gene identifiers"):
        write_verified_loom(changed, path)
