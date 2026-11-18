"""Input contracts for the GSE288099 matched velocity comparison."""

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from benchmarks.validation.neuroblastoma_velocity import load_counts


def test_velocity_crosswalk_matches_exact_tumor_cell_and_gene_ids(tmp_path: Path):
    expr = ad.AnnData(
        sparse.eye(2, format="csr"),
        obs=pd.DataFrame({"tumor": ["C1", "C1"]}, index=["x", "y"]),
        var=pd.DataFrame({"symbol": ["A", "B"]}, index=["ENSG1", "ENSG2"]),
    )
    mapping = pd.DataFrame(
        {
            "tumor": ["C1", "C1"],
            "loom_cell_id": ["b", "a"],
            "cell_id": ["x", "y"],
            "species": "human",
        }
    )
    with h5py.File(tmp_path / "C1.loom", "w") as f:
        f["col_attrs/CellID"] = [b"a", b"b"]
        f["row_attrs/Accession"] = [b"ENSG2.3", b"ENSMUSG3", b"ENSG1.1"]
        f["layers/spliced"] = [[1, 2], [8, 9], [3, 4]]
        f["layers/unspliced"] = [[5, 1], [6, 9], [2, 3]]
    counts = load_counts(expr, tmp_path, mapping)
    assert counts.obs_names.tolist() == ["x", "y"]
    assert set(counts.var_names) == {"ENSG1", "ENSG2"}
    np.testing.assert_array_equal(
        counts[:, expr.var_names].layers["spliced"].toarray(), [[4, 2], [3, 1]]
    )
    np.testing.assert_array_equal(
        counts[:, expr.var_names].layers["unspliced"].toarray(), [[3, 1], [2, 5]]
    )
    with pytest.raises(ValueError, match="species-verified"):
        load_counts(expr, tmp_path, mapping.assign(species=["human", "mouse"]))
    with pytest.raises(ValueError, match="one to one"):
        load_counts(expr, tmp_path, mapping.assign(loom_cell_id=["a", "a"]))
