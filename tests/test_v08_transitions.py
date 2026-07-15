import numpy as np
import pytest
from scipy import sparse

ad = pytest.importorskip("anndata")
scv = pytest.importorskip("scvelo")

from scCS.transitions import get_scvelo_transition_matrix


def test_scvelo_adapter_returns_row_normalized_sparse_matrix():
    adata = ad.AnnData(X=np.zeros((3, 1)))
    adata.uns["velocity_graph"] = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.8, 0.0],
                [0.1, 0.0, 0.7],
                [0.0, 0.2, 0.0],
            ]
        )
    )
    transition = get_scvelo_transition_matrix(
        adata,
        self_transitions=False,
        scale=1.0,
    )
    assert sparse.issparse(transition)
    row_sums = np.asarray(transition.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
