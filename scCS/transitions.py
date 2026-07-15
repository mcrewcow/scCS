"""Transition-matrix adapters for scCS v0.8."""

from __future__ import annotations

from typing import Optional

from scipy import sparse


def get_scvelo_transition_matrix(
    adata,
    *,
    vkey: str = "velocity",
    self_transitions: bool = True,
    scale: float = 10.0,
    use_negative_cosines: bool = False,
    weight_diffusion: float = 0.0,
    n_neighbors: Optional[int] = None,
):
    """Return scVelo's row-normalized directed transition matrix.

    ``scv.tl.velocity_graph`` must already have been run.  The adapter does
    not project into a display embedding; it only obtains cell-to-cell
    transition probabilities for the direct scCS projector.
    """
    try:
        import scvelo as scv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("scvelo is required to build a transition matrix from velocity.") from exc

    key = f"{vkey}_graph"
    if key not in adata.uns and key not in getattr(adata, "obsp", {}):
        raise KeyError(f"Missing {key!r}; run scv.tl.velocity_graph(adata, vkey={vkey!r}) first.")

    # scVelo 0.3.x checks adata.uns even when the graph is held in obsp.
    temporary_graph = False
    temporary_neg = False
    if key not in adata.uns and key in adata.obsp:
        adata.uns[key] = adata.obsp[key]
        temporary_graph = True
    neg_key = f"{vkey}_graph_neg"
    if neg_key not in adata.uns and neg_key in getattr(adata, "obsp", {}):
        adata.uns[neg_key] = adata.obsp[neg_key]
        temporary_neg = True

    try:
        matrix = scv.utils.get_transition_matrix(
            adata,
            vkey=vkey,
            basis=None,
            self_transitions=self_transitions,
            scale=scale,
            use_negative_cosines=use_negative_cosines,
            weight_diffusion=weight_diffusion,
            n_neighbors=n_neighbors,
        )
    finally:
        if temporary_graph:
            del adata.uns[key]
        if temporary_neg:
            del adata.uns[neg_key]

    return sparse.csr_matrix(matrix)
