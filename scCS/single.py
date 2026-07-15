"""Single-condition supervised commitment scoring for scCS v0.8.

The public workflow intentionally keeps the established method names::

    scorer = scCS.SingleScorer(...)
    scorer.build_embedding(...)
    scorer.fit()
    result = scorer.score()

The internals support two explicit scientific questions: instantaneous
transition pushforward in the supervised geometry, and discounted future-fate
hitting probabilities on the original RNA-velocity graph. The star remains a
standardized supervised display in both modes.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from .affinity import MagnitudeScaler, cosine_softmax_affinity
from .furcation import Furcation, LabelSpec, TerminalSpec
from .future_fate import FutureFateScoreResult, score_future_fate
from .ordering import FurcationOrderingScaler
from .pipeline import (
    FurcationScoreResult,
    score_projected_furcation,
)
from .population import PopulationCommitmentSummary
from .projection import (
    ProjectionResult,
    RootProjectionGeometryDiagnostics,
    RootProgressionDirectionDiagnostics,
    project_transition_velocity,
)
from .scoring_embedding import ScoringEmbeddingResult, build_scoring_embedding
from .transitions import get_scvelo_transition_matrix


class SingleScorer:
    """Quantify commitment at one manually annotated furcation.

    Parameters
    ----------
    adata
        AnnData containing the root and terminal populations.
    root
        One annotation label, or a group of labels representing one root
        population.  Retained for the familiar pre-v0.8 constructor.
    branches
        Ordered terminal labels, or a mapping from terminal name to one or
        more annotation labels.
    obs_key
        Annotation column in ``adata.obs``.
    furcation
        Optional pre-built :class:`scCS.Furcation`.  Supply either
        ``furcation`` or ``root``/``branches``, not both.
    copy
        Work on a copy of ``adata``.

    Notes
    -----
    scCS is supervised.  The root and terminal populations are supplied by
    annotation; the scorer does not infer topology or discover terminal states.
    """

    def __init__(
        self,
        adata,
        root: Optional[LabelSpec] = None,
        branches: Optional[TerminalSpec] = None,
        obs_key: str = "leiden",
        copy: bool = False,
        *,
        furcation: Optional[Furcation] = None,
    ) -> None:
        if furcation is not None:
            if root is not None or branches is not None:
                raise ValueError("Supply either furcation or root/branches, not both.")
            self.furcation = furcation
        else:
            if root is None or branches is None:
                raise ValueError("root and branches are required when furcation is not supplied.")
            self.furcation = Furcation(
                obs_key=obs_key,
                root=root,
                terminals=branches,
            )

        self.adata = adata.copy() if copy else adata
        self.obs_key = self.furcation.obs_key
        self.root = self.furcation.root_name
        self.branches = list(self.furcation.terminal_names)

        self._ordering_argument = None
        self._ordering_scaler: Optional[FurcationOrderingScaler] = None
        self._arm_scale = 1.0
        self._embedding_result: Optional[ScoringEmbeddingResult] = None
        self._projection_result: Optional[ProjectionResult] = None
        self._transition_matrix = None
        self._result: Optional[Union[FurcationScoreResult, FutureFateScoreResult]] = None
        self._scoring_mode = "instantaneous"
        self._fitted = False
        self.adata_sub = None
        self.population_summary: Optional[PopulationCommitmentSummary] = None

    def preflight(
        self,
        *,
        ordering_metric="pseudotime",
        check_velocity: bool = True,
        raise_on_error: bool = False,
    ):
        """Run annotation, ordering, velocity, and fitted-result diagnostics."""
        from .preflight import single_preflight

        report = single_preflight(
            self,
            ordering_metric=ordering_metric,
            check_velocity=check_velocity,
        )
        if raise_on_error:
            report.raise_for_errors()
        return report

    # ------------------------------------------------------------------
    # Optional RNA-velocity preprocessing
    # ------------------------------------------------------------------

    def compute_velocity(
        self,
        mode: str = "deterministic",
        n_pcs: int = 30,
        n_neighbors: int = 30,
        min_shared_counts: int = 20,
        random_state: int = 0,
        n_jobs: Optional[int] = None,
        pseudotime_root_key: Optional[Union[int, str]] = None,
        compute_pseudotime: bool = True,
        pseudotime_scope: str = "furcation",
        pseudotime_key_added: str = "velocity_pseudotime",
        pseudotime_n_dcs: int = 10,
        verbose: bool = True,
    ) -> "SingleScorer":
        """Compute a current-scVelo velocity graph on the full AnnData object.

        The method uses Scanpy for PCA and neighbors and does not pass the
        removed ``n_top_genes`` argument to scVelo.  No model fallback is
        performed silently: the requested velocity model either succeeds or
        raises its original error.
        """
        try:
            import scanpy as sc
            import scvelo as scv
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("scanpy and scvelo are required for compute_velocity().") from exc

        required = [name for name in ("spliced", "unspliced") if name not in self.adata.layers]
        if required:
            raise ValueError(f"Missing required layers: {required}.")

        if verbose:
            print(f"[scCS] Computing scVelo velocity (mode={mode!r}) on full data...")

        scv.pp.filter_and_normalize(
            self.adata,
            min_shared_counts=min_shared_counts,
        )

        if "X_pca" not in self.adata.obsm:
            max_pcs = min(n_pcs, self.adata.n_obs - 1, self.adata.n_vars - 1)
            if max_pcs < 1:
                raise ValueError("Not enough cells or genes to compute PCA.")
            sc.pp.pca(self.adata, n_comps=max_pcs)

        sc.pp.neighbors(
            self.adata,
            n_neighbors=min(n_neighbors, self.adata.n_obs - 1),
            n_pcs=min(n_pcs, self.adata.obsm["X_pca"].shape[1]),
            use_rep="X_pca",
            random_state=random_state,
        )
        scv.pp.moments(self.adata, n_neighbors=None, n_pcs=None)

        for key in ("velocity", "variance_velocity"):
            if key in self.adata.layers:
                del self.adata.layers[key]
        for key in ("velocity_graph", "velocity_graph_neg", "velocity_params"):
            if key in self.adata.uns:
                del self.adata.uns[key]

        if mode == "dynamical":
            kwargs = {} if n_jobs is None else {"n_jobs": n_jobs}
            scv.tl.recover_dynamics(self.adata, **kwargs)
            scv.tl.velocity(self.adata, mode="dynamical")
        else:
            scv.tl.velocity(self.adata, mode=mode)

        graph_kwargs = {} if n_jobs is None else {"n_jobs": n_jobs}
        scv.tl.velocity_graph(self.adata, **graph_kwargs)

        if compute_pseudotime:
            try:
                self.compute_velocity_pseudotime(
                    root_key=pseudotime_root_key,
                    scope=pseudotime_scope,
                    key_added=pseudotime_key_added,
                    n_dcs=pseudotime_n_dcs,
                    verbose=verbose,
                )
            except Exception as exc:
                warnings.warn(
                    "Velocity was computed, but velocity pseudotime was not. "
                    f"Provide another ordering metric or resolve the scVelo error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._transition_matrix = None
        self._projection_result = None
        self._result = None
        self._fitted = False

        if verbose:
            print("[scCS] Velocity graph computed.")
        return self

    def compute_velocity_pseudotime(
        self,
        root_key: Optional[Union[int, str]] = None,
        *,
        scope: str = "furcation",
        key_added: str = "velocity_pseudotime",
        vkey: str = "velocity",
        n_dcs: int = 10,
        use_velocity_graph: bool = True,
        verbose: bool = True,
    ) -> "SingleScorer":
        """Compute root-guided velocity pseudotime globally or on the furcation.

        Parameters
        ----------
        root_key
            Root cell index, observation name, or scVelo-compatible root-prior
            column.  A full-data integer index is translated automatically when
            ``scope='furcation'``.
        scope
            ``'full'`` computes pseudotime on the complete velocity graph.
            ``'furcation'`` induces the graph on the manually selected root and
            terminal populations before computing pseudotime, then transfers the
            values back to the full AnnData object.  The latter is usually the
            appropriate ordering for a supervised scCS trajectory when unrelated
            lineages are present in the same object.
        key_added
            Destination column in ``adata.obs``.  Cells outside a furcation-scoped
            calculation receive ``NaN``.
        vkey, n_dcs, use_velocity_graph
            Passed to :func:`scvelo.tl.velocity_pseudotime`.

        Notes
        -----
        This method does not recompute velocity.  Furcation-scoped pseudotime uses
        the induced subgraph of the already fitted full-data velocity graph, so
        RNA-velocity estimation and scCS transition projection remain unchanged.
        """
        try:
            import scvelo as scv
            from scipy import sparse
            from scipy.sparse.csgraph import connected_components
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "scvelo and scipy are required for compute_velocity_pseudotime()."
            ) from exc

        scope = str(scope).lower()
        if scope not in {"full", "furcation"}:
            raise ValueError("scope must be 'full' or 'furcation'.")
        if not isinstance(key_added, str) or not key_added:
            raise ValueError("key_added must be a non-empty string.")
        if not isinstance(n_dcs, int) or n_dcs < 1:
            raise ValueError("n_dcs must be a positive integer.")

        output_key = f"{vkey}_pseudotime"
        selected_indices = np.arange(self.adata.n_obs, dtype=int)
        local_root_key = root_key

        if scope == "full":
            if verbose:
                print("[scCS] Computing velocity pseudotime on the full graph...")
            scv.tl.velocity_pseudotime(
                self.adata,
                vkey=vkey,
                root_key=root_key,
                n_dcs=n_dcs,
                use_velocity_graph=use_velocity_graph,
            )
            values = self.adata.obs[output_key].to_numpy(dtype=float, copy=True)
        else:
            validation = self.furcation.validate_adata(self.adata)
            selected_indices = np.flatnonzero(validation.selected_mask)
            if len(selected_indices) < 3:
                raise ValueError(
                    "At least three selected furcation cells are required for velocity pseudotime."
                )
            local = self.adata[selected_indices].copy()

            if use_velocity_graph:
                for graph_key in (f"{vkey}_graph", f"{vkey}_graph_neg"):
                    if graph_key not in self.adata.uns:
                        raise KeyError(
                            f"Missing adata.uns[{graph_key!r}]. Compute the velocity graph first."
                        )
                    graph = self.adata.uns[graph_key]
                    if graph.shape != (self.adata.n_obs, self.adata.n_obs):
                        raise ValueError(
                            f"adata.uns[{graph_key!r}] has shape {graph.shape}; expected "
                            f"{(self.adata.n_obs, self.adata.n_obs)}."
                        )
                    if sparse.issparse(graph):
                        local.uns[graph_key] = graph.tocsr()[selected_indices][
                            :, selected_indices
                        ].copy()
                    else:
                        graph_array = np.asarray(graph)
                        local.uns[graph_key] = graph_array[
                            np.ix_(selected_indices, selected_indices)
                        ].copy()

            if isinstance(root_key, (int, np.integer)):
                matches = np.flatnonzero(selected_indices == int(root_key))
                if len(matches) != 1:
                    raise ValueError(
                        "The requested root index is not inside the selected furcation."
                    )
                local_root_key = int(matches[0])
            elif isinstance(root_key, str) and root_key in self.adata.obs_names:
                if root_key not in local.obs_names:
                    raise ValueError(
                        "The requested root observation is not inside the selected furcation."
                    )
                local_root_key = root_key

            if verbose:
                print(
                    "[scCS] Computing velocity pseudotime on the induced "
                    f"furcation graph ({len(selected_indices)} cells)..."
                )
            scv.tl.velocity_pseudotime(
                local,
                vkey=vkey,
                root_key=local_root_key,
                n_dcs=min(n_dcs, len(selected_indices) - 1),
                use_velocity_graph=use_velocity_graph,
            )
            local_values = local.obs[output_key].to_numpy(dtype=float, copy=True)
            values = np.full(self.adata.n_obs, np.nan, dtype=float)
            values[selected_indices] = local_values

        n_components = 1
        largest_component_fraction = 1.0
        if use_velocity_graph and f"{vkey}_graph" in self.adata.uns:
            positive = self.adata.uns[f"{vkey}_graph"]
            negative = self.adata.uns.get(f"{vkey}_graph_neg")

            def graph_support(graph):
                if sparse.issparse(graph):
                    selected = graph.tocsr()[selected_indices][:, selected_indices].copy()
                    selected.data = np.ones_like(selected.data, dtype=np.int8)
                    return selected
                selected = np.asarray(graph)[np.ix_(selected_indices, selected_indices)]
                return sparse.csr_matrix(selected != 0, dtype=np.int8)

            support = graph_support(positive)
            if negative is not None:
                support = support + graph_support(negative)
            support = support + support.T
            support.data = np.ones_like(support.data, dtype=np.int8)
            n_components, component_labels = connected_components(
                support,
                directed=False,
                return_labels=True,
            )
            component_sizes = np.bincount(component_labels, minlength=n_components)
            largest_component_fraction = float(component_sizes.max() / len(selected_indices))
            if scope == "furcation" and n_components > 1:
                warnings.warn(
                    "The induced furcation velocity graph contains "
                    f"{n_components} disconnected components; the largest contains "
                    f"{largest_component_fraction:.1%} of selected cells. Velocity "
                    "pseudotime may form disconnected bands and should be inspected.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.adata.obs[key_added] = values
        metadata = dict(self.adata.uns.get("sccs_v08", {}))
        pseudotime_metadata = dict(metadata.get("pseudotime", {}))
        pseudotime_metadata[key_added] = {
            "scope": scope,
            "vkey": str(vkey),
            "n_dcs": int(min(n_dcs, len(selected_indices) - 1)),
            "root_key": None if root_key is None else str(root_key),
            "n_cells": int(len(selected_indices)),
            "finite_fraction": float(np.mean(np.isfinite(values[selected_indices]))),
            "n_connected_components": int(n_components),
            "largest_component_fraction": largest_component_fraction,
        }
        metadata["pseudotime"] = pseudotime_metadata
        self.adata.uns["sccs_v08"] = metadata

        if verbose:
            finite = int(np.isfinite(values[selected_indices]).sum())
            print(
                f"[scCS] Stored {key_added!r}: {finite}/{len(selected_indices)} "
                "selected cells have finite pseudotime."
            )
        return self

    # ------------------------------------------------------------------
    # Scientific star embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ordering_alias(adata, ordering_metric):
        if isinstance(ordering_metric, str) and ordering_metric == "pseudotime":
            if "velocity_pseudotime" not in adata.obs:
                raise KeyError(
                    "ordering_metric='pseudotime' requires "
                    "adata.obs['velocity_pseudotime']. Run compute_velocity(), "
                    "compute it with scVelo, or pass an explicit ordering column."
                )
            return "velocity_pseudotime"
        return ordering_metric

    def build_embedding(
        self,
        ordering_metric: Union[str, Sequence[float], np.ndarray] = "pseudotime",
        invert_ordering: bool = False,
        arm_scale: float = 1.0,
        ordering_scaler: Optional[FurcationOrderingScaler] = None,
        write_to_adata: bool = True,
        cache_subset: bool = True,
        verbose: bool = True,
    ) -> "SingleScorer":
        """Build the deterministic root-plus-simplex scientific star.

        Plot jitter, display angles, and random seeds are intentionally absent
        because display choices must not alter scientific scores.

        Set ``cache_subset=False`` for large datasets to avoid materializing an
        additional AnnData copy of the selected furcation. The scientific
        embedding and all scoring results remain available on the scorer.
        """
        ordering = self._resolve_ordering_alias(self.adata, ordering_metric)
        if isinstance(ordering, str):
            if ordering not in self.adata.obs:
                raise KeyError(f"Ordering column {ordering!r} is missing from adata.obs.")
            if invert_ordering:
                ordering = -self.adata.obs[ordering].to_numpy(dtype=float)
        else:
            ordering = np.asarray(ordering, dtype=float)
            if ordering.ndim != 1 or len(ordering) != self.adata.n_obs:
                raise ValueError("Array ordering must be one-dimensional and match adata.n_obs.")
            if invert_ordering:
                ordering = -ordering

        scaler = ordering_scaler or FurcationOrderingScaler(
            higher_is_later=True,
        )

        embedding = build_scoring_embedding(
            self.adata,
            self.furcation,
            ordering=ordering,
            ordering_scaler=scaler,
            arm_scale=arm_scale,
            write_to_adata=write_to_adata,
        )

        self._ordering_argument = ordering
        self._ordering_scaler = scaler
        self._arm_scale = float(arm_scale)
        self._embedding_result = embedding
        self._projection_result = None
        self._result = None
        self._fitted = False

        if cache_subset:
            self.adata_sub = self.adata[embedding.selected_indices].copy()
            self.adata_sub.obsm["X_sccs_score"] = embedding.coordinates.copy()
        else:
            self.adata_sub = None

        if verbose:
            diagnostics = embedding.ordering.diagnostics
            print(
                f"[scCS] Scientific star built for {embedding.n_selected} cells "
                f"in {embedding.dimension} dimensions."
            )
            print(
                f"       Root radial clipping: "
                f"{diagnostics.root_clipped_low_fraction:.3f} low / "
                f"{diagnostics.root_clipped_high_fraction:.3f} high"
            )
            print(
                f"       Terminal scientific coordinates: fixed equal-radius "
                f"simplex vertices (radius={embedding.arm_scale:.3f})."
            )
            if (
                diagnostics.root_unique_values < 20
                or diagnostics.root_unique_fraction < 0.05
                or diagnostics.root_largest_tie_fraction > 0.20
            ):
                warnings.warn(
                    (
                        f"Root ordering has only {diagnostics.root_unique_values} unique "
                        f"values and a largest tied group of "
                        f"{diagnostics.root_largest_tie_fraction:.1%}. Root cells will "
                        "form radial bands. Use a biologically justified continuous "
                        "ordering; scCS does not add artificial tie-breaking jitter."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
        return self

    # ------------------------------------------------------------------
    # Direct transition projection
    # ------------------------------------------------------------------

    def project_velocity(
        self,
        transition_matrix=None,
        renormalize_retained: bool = True,
        min_transition_coverage: float = 0.05,
        verbose: bool = True,
    ) -> "SingleScorer":
        """Project full-graph transitions into the scientific star."""
        self._check_embedding()
        matrix = transition_matrix
        if matrix is None:
            matrix = self._transition_matrix
        if matrix is None:
            matrix = get_scvelo_transition_matrix(self.adata)

        self._transition_matrix = matrix
        self._projection_result = project_transition_velocity(
            matrix,
            self._embedding_result.coordinates,
            selected_indices=self._embedding_result.selected_indices,
            renormalize_retained=renormalize_retained,
            min_coverage=min_transition_coverage,
        )
        self._result = None
        self._fitted = False

        if verbose:
            projection = self._projection_result
            print(
                f"[scCS] Projected velocity defined for "
                f"{int(projection.velocity_defined.sum())}/{len(projection.velocity_defined)} "
                "furcation cells."
            )
            root = self._embedding_result.root_mask
            print(
                f"       Root median transition coverage: "
                f"{np.median(projection.transition_coverage[root]):.3f}"
            )
        return self

    def projection_geometry_diagnostics(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        tie_tolerance: float = 1e-12,
    ) -> RootProjectionGeometryDiagnostics:
        """Verify root-cell fate direction directly from transition destinations.

        The scientific root arm lies entirely on the axis orthogonal to the
        regular-simplex fate subspace.  Consequently, for every root cell, the
        fate-directed component of the projected velocity must equal the
        retained transition mass entering each annotated terminal population
        multiplied by that terminal's ideal simplex direction.

        This method reconstructs the branch velocity from the transition
        matrix without using the stored projected vector, then recomputes the
        cosine-softmax affinity.  Near-zero reconstruction errors demonstrate
        that the geometry conversion has not rotated or relabeled the RNA-
        velocity direction.  The check is available only when scCS projected a
        transition matrix; it is undefined for externally supplied projected
        vectors.
        """
        self._check_fitted()
        self._check_instantaneous_mode("projection_geometry_diagnostics()")
        output = self._result if result is None else result
        assert output is not None
        if self._transition_matrix is None:
            raise RuntimeError(
                "Projection-geometry diagnostics require the transition matrix "
                "used by project_velocity(). They are unavailable after "
                "load_velocity_vectors()."
            )
        if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
            raise ValueError("tie_tolerance must be finite and non-negative.")

        matrix = self._transition_matrix
        selected_indices = np.asarray(output.embedding.selected_indices, dtype=int)
        n_selected = output.n_cells
        if matrix.shape[0] == n_selected and matrix.shape[1] == n_selected:
            retained = matrix
        else:
            if sparse.issparse(matrix):
                retained = matrix.tocsr()[selected_indices][:, selected_indices]
            else:
                dense = np.asarray(matrix, dtype=float)
                retained = dense[np.ix_(selected_indices, selected_indices)]

        root_local = np.flatnonzero(output.root_mask)
        if sparse.issparse(retained):
            root_weights = retained.tocsr()[root_local].astype(float, copy=True)
        else:
            root_weights = np.asarray(retained[root_local], dtype=float).copy()

        retained_mass = np.asarray(
            output.projection.retained_transition_mass[root_local],
            dtype=float,
        )
        if output.projection.renormalized:
            inverse = np.zeros_like(retained_mass)
            positive = retained_mass > 0
            inverse[positive] = 1.0 / retained_mass[positive]
            if sparse.issparse(root_weights):
                root_weights = sparse.diags(inverse) @ root_weights
            else:
                root_weights[positive] *= inverse[positive, None]

        terminal_mass = np.zeros((len(root_local), output.k), dtype=float)
        terminal_names = np.asarray(output.embedding.terminal_names).astype(str)
        for fate_index, fate_name in enumerate(output.fate_names):
            destination_mask = output.terminal_mask & (terminal_names == fate_name)
            if sparse.issparse(root_weights):
                terminal_mass[:, fate_index] = np.asarray(
                    root_weights[:, destination_mask].sum(axis=1)
                ).ravel()
            else:
                terminal_mass[:, fate_index] = root_weights[:, destination_mask].sum(axis=1)

        directions = output.embedding.geometry.terminal_directions
        reconstructed_branch = terminal_mass @ directions
        stored_branch = np.asarray(output.branch_velocity[root_local], dtype=float)
        defined = np.asarray(output.projection.velocity_defined[root_local], dtype=bool)

        finite_defined = defined & np.all(np.isfinite(stored_branch), axis=1)
        branch_error = reconstructed_branch - np.nan_to_num(stored_branch, nan=0.0)
        if np.any(finite_defined):
            finite_error = branch_error[finite_defined]
            max_abs_branch_error = float(np.max(np.abs(finite_error)))
            branch_rmse = float(np.sqrt(np.mean(np.square(finite_error))))
        else:
            max_abs_branch_error = float("nan")
            branch_rmse = float("nan")

        reconstructed_affinity = cosine_softmax_affinity(
            reconstructed_branch,
            directions,
            aligned_probability=output.aligned_probability,
        )
        stored_affinity = np.asarray(output.directional_affinity[root_local], dtype=float)
        if np.any(finite_defined):
            max_abs_affinity_error = float(
                np.max(
                    np.abs(reconstructed_affinity[finite_defined] - stored_affinity[finite_defined])
                )
            )
        else:
            max_abs_affinity_error = float("nan")

        reconstructed_norm = np.linalg.norm(reconstructed_branch, axis=1)
        stored_norm = np.linalg.norm(np.nan_to_num(stored_branch, nan=0.0), axis=1)
        informative = (
            finite_defined
            & (terminal_mass.sum(axis=1) > np.finfo(float).eps)
            & (reconstructed_norm > np.finfo(float).eps)
            & (stored_norm > np.finfo(float).eps)
        )
        direction_cosine = np.full(len(root_local), np.nan, dtype=float)
        if np.any(informative):
            direction_cosine[informative] = np.sum(
                reconstructed_branch[informative] * stored_branch[informative],
                axis=1,
            ) / (reconstructed_norm[informative] * stored_norm[informative])
            median_direction_cosine = float(np.nanmedian(direction_cosine[informative]))
        else:
            median_direction_cosine = float("nan")

        if output.k == 2:
            sorted_mass = np.sort(terminal_mass, axis=1)
            margin = sorted_mass[:, -1] - sorted_mass[:, -2]
        else:
            partitioned = np.partition(terminal_mass, kth=output.k - 2, axis=1)
            margin = partitioned[:, -1] - partitioned[:, -2]
        decisive = informative & (margin > tie_tolerance)
        if np.any(decisive):
            direct_dominant = np.argmax(terminal_mass[decisive], axis=1)
            affinity_dominant = np.argmax(stored_affinity[decisive], axis=1)
            dominant_agreement = float(np.mean(direct_dominant == affinity_dominant))
        else:
            dominant_agreement = float("nan")

        return RootProjectionGeometryDiagnostics(
            root_local_indices=root_local,
            terminal_transition_mass=terminal_mass,
            reconstructed_branch_velocity=reconstructed_branch,
            stored_branch_velocity=stored_branch,
            reconstructed_directional_affinity=reconstructed_affinity,
            stored_directional_affinity=stored_affinity,
            defined_mask=finite_defined,
            informative_mask=informative,
            decisive_mask=decisive,
            max_abs_branch_error=max_abs_branch_error,
            branch_rmse=branch_rmse,
            max_abs_affinity_error=max_abs_affinity_error,
            median_direction_cosine=median_direction_cosine,
            dominant_fate_agreement=dominant_agreement,
        )

    def plot_projection_geometry_diagnostics(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        max_cells: int = 500,
        sort_by: str = "ordering",
        cmap: str = "viridis",
        figsize: tuple[float, float] = (13.0, 7.0),
        show_branch_strip: bool = True,
        show_fate_strip: bool = True,
        branch_tie_tolerance: float = 1e-12,
    ):
        """Plot direct terminal-transition mass beside reconstructed affinity.

        Rows are root cells only.  The left panel is computed directly from
        retained transition destinations; the right panel is the stored
        cosine-softmax directional affinity.  They need not have identical
        numerical scales, but their dominant fate must agree whenever direct
        terminal-transition mass has a unique maximum.  The optional left
        strip identifies the dominant directly reached terminal branch for
        each root cell, while the optional top strips preserve the fate-color
        mapping across both matrices.
        """
        output = self._result if result is None else result
        self._check_fitted()
        self._check_instantaneous_mode("plot_projection_geometry_diagnostics()")
        assert output is not None
        if not isinstance(max_cells, int) or max_cells < 1:
            raise ValueError("max_cells must be a positive integer.")
        if not np.isfinite(branch_tie_tolerance) or branch_tie_tolerance < 0:
            raise ValueError("branch_tie_tolerance must be finite and non-negative.")
        diagnostics = self.projection_geometry_diagnostics(output)

        root_local = diagnostics.root_local_indices
        if sort_by == "ordering":
            values = output.embedding.selected_ordering_values[root_local]
            order = np.argsort(values, kind="stable")
        elif sort_by == "terminal_mass":
            order = np.argsort(
                -diagnostics.terminal_transition_mass.max(axis=1),
                kind="stable",
            )
        elif sort_by == "none":
            order = np.arange(len(root_local))
        else:
            raise ValueError("sort_by must be 'ordering', 'terminal_mass', or 'none'.")

        if len(order) > max_cells:
            take = np.linspace(0, len(order) - 1, max_cells).round().astype(int)
            order = order[take]

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        color_cycle = plt.get_cmap("tab10").colors
        fate_colors = {
            str(name): color_cycle[(index + 1) % len(color_cycle)]
            for index, name in enumerate(output.fate_names)
        }
        panels = (
            (
                diagnostics.terminal_transition_mass[order],
                "Direct retained transition mass to terminal fates",
                "Transition mass",
            ),
            (
                diagnostics.stored_directional_affinity[order],
                "Directional affinity after cosine-softmax",
                "Directional affinity",
            ),
        )
        for axis, (values, title, label) in zip(axes, panels):
            image = axis.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
            axis.set_xticks(np.arange(output.k))
            axis.set_xticklabels(output.fate_names, rotation=30, ha="right")
            axis.set_xlabel("Annotated terminal fate")
            axis.set_title(title)
            fig.colorbar(image, ax=axis, label=label)
            if show_fate_strip:
                divider = make_axes_locatable(axis)
                fate_axis = divider.append_axes("top", size="3%", pad=0.05, sharex=axis)
                fate_axis.imshow(
                    np.arange(output.k, dtype=int)[None, :],
                    aspect="auto",
                    interpolation="nearest",
                    cmap=ListedColormap([fate_colors[str(name)] for name in output.fate_names]),
                    vmin=-0.5,
                    vmax=max(output.k - 0.5, 0.5),
                )
                fate_axis.set_xticks([])
                fate_axis.set_yticks([])
                fate_axis.set_ylabel("Fate", rotation=0, ha="right", va="center")
                fate_axis.tick_params(left=False, bottom=False)

        if show_branch_strip:
            displayed_mass = diagnostics.terminal_transition_mass[order]
            row_total = displayed_mass.sum(axis=1)
            if output.k == 2:
                sorted_mass = np.sort(displayed_mass, axis=1)
                margin = sorted_mass[:, -1] - sorted_mass[:, -2]
            else:
                partitioned = np.partition(
                    displayed_mass,
                    kth=output.k - 2,
                    axis=1,
                )
                margin = partitioned[:, -1] - partitioned[:, -2]
            dominant = np.asarray(output.fate_names, dtype=object)[
                np.argmax(displayed_mass, axis=1)
            ].astype(object)
            dominant[row_total <= np.finfo(float).eps] = "no terminal mass"
            dominant[(row_total > np.finfo(float).eps) & (margin <= branch_tie_tolerance)] = (
                "ambiguous"
            )
            categories = [str(name) for name in output.fate_names]
            for fallback in ("ambiguous", "no terminal mass"):
                if np.any(dominant == fallback):
                    categories.append(fallback)
            branch_colors = {
                **fate_colors,
                "ambiguous": "0.62",
                "no terminal mass": "0.86",
            }
            category_index = {name: index for index, name in enumerate(categories)}
            codes = np.asarray([category_index[str(name)] for name in dominant], dtype=int)
            divider = make_axes_locatable(axes[0])
            branch_axis = divider.append_axes("left", size="3.5%", pad=0.08)
            branch_axis.imshow(
                codes[:, None],
                aspect="auto",
                interpolation="nearest",
                cmap=ListedColormap([branch_colors[name] for name in categories]),
                vmin=-0.5,
                vmax=max(len(categories) - 0.5, 0.5),
            )
            branch_axis.set_xticks([])
            branch_axis.set_yticks([])
            branch_axis.set_ylim(axes[0].get_ylim())
            branch_axis.set_title("Direct\nbranch", fontsize=9, pad=4)
            branch_axis.set_ylabel(
                f"Root cells ({len(order)} shown; ordered by {sort_by.replace('_', ' ')})",
                labelpad=32,
            )
            fallback = [name for name in ("ambiguous", "no terminal mass") if name in categories]
            if fallback:
                handles = [
                    Patch(facecolor=branch_colors[name], edgecolor="none", label=name)
                    for name in fallback
                ]
                branch_axis.legend(
                    handles=handles,
                    frameon=False,
                    fontsize=8,
                    loc="upper right",
                    bbox_to_anchor=(-0.15, 1.0),
                )
            axes[0].set_ylabel("")
        else:
            axes[0].set_ylabel(
                f"Root cells ({len(order)} shown; ordered by {sort_by.replace('_', ' ')})"
            )
        fig.tight_layout()
        return fig

    def load_velocity_vectors(
        self,
        velocity,
        vy: Optional[np.ndarray] = None,
    ) -> "SingleScorer":
        """Supply projected velocity directly in scientific star coordinates.

        ``velocity`` should normally have shape ``(n_selected, k)``.  For a
        two-fate furcation only, the familiar ``vx, vy`` form remains accepted.
        Directly supplied vectors are treated as fully covered projections.
        """
        self._check_embedding()
        if vy is not None:
            if self._embedding_result.dimension != 2:
                raise ValueError(
                    "Separate vx/vy arrays are only valid for a two-fate geometry. "
                    "Supply a full (n_selected, k) velocity matrix instead."
                )
            vectors = np.column_stack(
                [
                    np.asarray(velocity, dtype=float),
                    np.asarray(vy, dtype=float),
                ]
            )
        else:
            vectors = np.asarray(velocity, dtype=float)

        expected = (
            self._embedding_result.n_selected,
            self._embedding_result.dimension,
        )
        if vectors.shape != expected:
            raise ValueError(f"Projected velocity has shape {vectors.shape}; expected {expected}.")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Projected velocity contains non-finite values.")

        n = len(vectors)
        self._projection_result = ProjectionResult(
            velocity=vectors.copy(),
            retained_transition_mass=np.ones(n),
            external_transition_mass=np.zeros(n),
            transition_coverage=np.ones(n),
            velocity_defined=np.ones(n, dtype=bool),
            selected_indices=self._embedding_result.selected_indices.copy(),
            renormalized=True,
            min_coverage=0.0,
        )
        self._result = None
        self._fitted = False
        return self

    # ------------------------------------------------------------------
    # Fit and score
    # ------------------------------------------------------------------

    def fit_future_fate(
        self,
        transition_matrix=None,
        *,
        effective_horizon: int = 64,
        anchor_quantile: float = 0.90,
        min_anchor_cells: int = 10,
        competing_outcomes: Optional[Mapping[str, Sequence[int] | np.ndarray]] = None,
        min_reach: float = 1e-6,
        progression_values: Optional[Sequence[float]] = None,
        progression_scale: str = "rank",
        solver: str = "auto",
        direct_max_states: int = 50_000,
        tolerance: float = 1e-10,
        max_iter: int = 20_000,
        committed_reach_threshold: float = 0.25,
        committed_specificity_threshold: float = 0.25,
        verbose: bool = True,
    ) -> "SingleScorer":
        """Fit discounted future-fate scoring on the original velocity graph.

        This mode does not transfer a scientific velocity vector into the star.
        It estimates discounted hitting probabilities for late cells in each
        supervised fate, while signed progression is calculated separately from
        expected changes in the supplied ordering.

        ``competing_outcomes`` is optional and explicit. Each mapping value may
        be a full-length Boolean mask or full-data cell indices. Unmodelled
        futures remain unresolved rather than being automatically assigned to a
        guessed competing terminal state.
        """
        self._check_embedding()
        matrix = transition_matrix
        if matrix is None:
            matrix = get_scvelo_transition_matrix(self.adata)
        result = score_future_fate(
            self.furcation,
            self._embedding_result,
            matrix,
            effective_horizon=effective_horizon,
            anchor_quantile=anchor_quantile,
            min_anchor_cells=min_anchor_cells,
            competing_outcomes=competing_outcomes,
            min_reach=min_reach,
            progression_values=progression_values,
            progression_scale=progression_scale,
            solver=solver,
            direct_max_states=direct_max_states,
            tolerance=tolerance,
            max_iter=max_iter,
            committed_reach_threshold=committed_reach_threshold,
            committed_specificity_threshold=committed_specificity_threshold,
        )
        self._result = result
        self._projection_result = result.projection
        self._transition_matrix = matrix
        self._fitted = True
        self._scoring_mode = "future_fate"
        self.population_summary = result.root_population_summary
        if verbose:
            print(result.summary())
        return self

    def fit(
        self,
        transition_matrix=None,
        *,
        scoring_mode: str = "instantaneous",
        future_fate_options: Optional[Mapping[str, object]] = None,
        aligned_probability: float = 0.90,
        magnitude_scaler: Optional[MagnitudeScaler] = None,
        magnitude_fit_population: str = "root",
        renormalize_retained: bool = True,
        min_transition_coverage: float = 0.05,
        committed_strength_threshold: float = 0.25,
        committed_specificity_threshold: float = 0.25,
        verbose: bool = True,
    ) -> "SingleScorer":
        """Fit one of the two explicit scCS scoring modes.

        Parameters
        ----------
        scoring_mode
            ``"instantaneous"`` retains the transition-pushforward and
            cosine-softmax model. ``"future_fate"`` uses discounted hitting
            probabilities on the original velocity graph.
        future_fate_options
            Keyword arguments forwarded to :meth:`fit_future_fate`. This keeps
            mode-specific parameters out of the established instantaneous API.
        """
        mode = str(scoring_mode)
        if mode not in {"instantaneous", "future_fate"}:
            raise ValueError("scoring_mode must be 'instantaneous' or 'future_fate'.")
        if mode == "future_fate":
            options = dict(future_fate_options or {})
            options.setdefault("verbose", verbose)
            return self.fit_future_fate(transition_matrix=transition_matrix, **options)

        if future_fate_options:
            raise ValueError(
                "future_fate_options may only be supplied when scoring_mode='future_fate'."
            )
        self._check_embedding()
        needs_projection = (
            self._projection_result is None
            or transition_matrix is not None
            or self._scoring_mode != "instantaneous"
        )
        if needs_projection:
            self.project_velocity(
                transition_matrix=transition_matrix,
                renormalize_retained=renormalize_retained,
                min_transition_coverage=min_transition_coverage,
                verbose=verbose,
            )

        scaler = magnitude_scaler or MagnitudeScaler(
            scale_quantile=0.75,
            power=1.0,
        )
        self._result = score_projected_furcation(
            self.furcation,
            self._embedding_result,
            self._projection_result,
            aligned_probability=aligned_probability,
            magnitude_scaler=scaler,
            magnitude_fit_population=magnitude_fit_population,
            committed_strength_threshold=committed_strength_threshold,
            committed_specificity_threshold=committed_specificity_threshold,
        )
        self._fitted = True
        self._scoring_mode = "instantaneous"
        self.population_summary = self._result.root_population_summary

        if verbose:
            print(self._result.summary())
        return self

    def score(
        self,
        cell_mask: Optional[np.ndarray] = None,
        *,
        write_to_adata: bool = True,
        verbose: bool = True,
    ) -> Union[FurcationScoreResult, FutureFateScoreResult]:
        """Return cell-level scores and summarize an explicit population.

        The fitted cell-level result is common to all summaries.  If
        ``cell_mask`` is supplied, it must be aligned to the selected
        furcation cells and only changes ``scorer.population_summary``.
        """
        self._check_fitted()
        assert self._result is not None

        if cell_mask is None:
            self.population_summary = self._result.root_population_summary
        else:
            mask = np.asarray(cell_mask)
            if mask.dtype != bool or mask.ndim != 1 or len(mask) != len(self._result.cell_ids):
                raise ValueError(
                    "cell_mask must be a Boolean array aligned to selected furcation cells."
                )
            self.population_summary = self._result.summarize(mask)

        if write_to_adata:
            self._result.write_to_adata(self.adata)
            self.adata_sub = self.adata[self._result.embedding.selected_indices].copy()

        if verbose:
            print(self._result.summary())
            if cell_mask is not None:
                print(
                    f"[scCS] Explicit population summary: "
                    f"n={self.population_summary.n_cells}, "
                    f"total_mass={self.population_summary.total_mass:.4g}"
                )
        return self._result

    def score_per_subset(
        self,
        split_by: str,
        *,
        population: str = "root",
        min_cells: int = 5,
        verbose: bool = False,
    ) -> dict[object, PopulationCommitmentSummary]:
        """Summarize fitted commitment separately for annotated subsets.

        This method is descriptive.  Formal condition inference will be
        provided by the replicate-aware PairScorer/MultiScorer redesign.
        """
        self._check_fitted()
        assert self._result is not None

        if self.adata_sub is not None and split_by in self.adata_sub.obs:
            values = self.adata_sub.obs[split_by]
        elif split_by in self.adata.obs:
            values = self.adata.obs.iloc[self._result.embedding.selected_indices][split_by]
        else:
            raise KeyError(f"{split_by!r} is missing from adata.obs.")

        base_mask = np.ones(len(self._result.cell_ids), dtype=bool)
        if population == "root":
            base_mask &= self._result.root_mask
        elif population != "all":
            raise ValueError("population must be 'root' or 'all'.")

        summaries = {}
        for value in values.dropna().unique():
            equal_mask = np.asarray(values.eq(value).fillna(False), dtype=bool)
            mask = base_mask & equal_mask
            if int(mask.sum()) < min_cells:
                warnings.warn(
                    f"Subset {value!r} contains {int(mask.sum())} eligible cells; skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            summaries[value] = self._result.summarize(mask)
            if verbose:
                print(
                    f"[scCS] {value!r}: n={summaries[value].n_cells}, "
                    f"total_mass={summaries[value].total_mass:.4g}"
                )
        return summaries

    # ------------------------------------------------------------------
    # AnnData transfer, persistence, and display
    # ------------------------------------------------------------------

    def transfer_labels(
        self,
        adata=None,
        result: Optional[FurcationScoreResult] = None,
    ) -> None:
        """Write the fitted v0.8 result to an AnnData object."""
        self._check_fitted()
        target = self.adata if adata is None else adata
        output = self._result if result is None else result
        assert output is not None
        output.write_to_adata(target)

    def save(self, path: str) -> None:
        """Serialize the scorer state.  Only load trusted files."""
        state = {
            "furcation": self.furcation,
            "ordering_argument": self._ordering_argument,
            "ordering_scaler": self._ordering_scaler,
            "arm_scale": self._arm_scale,
            "embedding_result": self._embedding_result,
            "projection_result": self._projection_result,
            "result": self._result,
            "fitted": self._fitted,
        }
        Path(path).write_bytes(pickle.dumps(state))

    @classmethod
    def load(cls, path: str, adata) -> "SingleScorer":
        """Load a scorer saved by :meth:`save`.  Only load trusted files."""
        state = pickle.loads(Path(path).read_bytes())
        scorer = cls(adata, furcation=state["furcation"])
        scorer._ordering_argument = state["ordering_argument"]
        scorer._ordering_scaler = state["ordering_scaler"]
        scorer._arm_scale = state["arm_scale"]
        scorer._embedding_result = state["embedding_result"]
        scorer._projection_result = state["projection_result"]
        scorer._result = state["result"]
        scorer._fitted = bool(state["fitted"])
        if scorer._embedding_result is not None:
            scorer.adata_sub = adata[scorer._embedding_result.selected_indices].copy()
        if scorer._result is not None:
            scorer.population_summary = scorer._result.root_population_summary
        return scorer

    def _selected_display_ordering(
        self,
        output: FurcationScoreResult,
    ) -> np.ndarray:
        """Return the fitted, oriented ordering values for selected cells.

        These values are used only by the two-dimensional display layer. They
        never replace the fixed equal-radius terminal vertices in the
        scientific scoring embedding.
        """
        stored = getattr(output.embedding, "selected_ordering_values", None)
        if stored is not None:
            values = np.asarray(stored, dtype=float)
        else:
            # Compatibility fallback for development pickles created before
            # selected ordering values were stored with the embedding.
            if self._ordering_argument is None:
                raise RuntimeError("No fitted ordering is available for display.")
            if isinstance(self._ordering_argument, str):
                if self._ordering_argument not in self.adata.obs:
                    raise KeyError(
                        f"Ordering column {self._ordering_argument!r} is missing from adata.obs."
                    )
                full = self.adata.obs[self._ordering_argument].to_numpy(dtype=float)
            else:
                full = np.asarray(self._ordering_argument, dtype=float)
                if full.ndim != 1 or len(full) != self.adata.n_obs:
                    raise RuntimeError(
                        "Stored ordering is not one-dimensional or no longer matches adata.n_obs."
                    )
            values = np.asarray(full[output.embedding.selected_indices], dtype=float)
            if self._ordering_scaler is not None and not self._ordering_scaler.higher_is_later:
                values = -values

        if values.ndim != 1 or len(values) != len(output.cell_ids):
            raise RuntimeError("Stored display ordering is not aligned to result cells.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Selected display ordering contains non-finite values.")
        return values

    @staticmethod
    def _within_group_progress(values: np.ndarray) -> np.ndarray:
        """Map values to deterministic within-group fractional ranks."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Display ordering values must be a non-empty 1D array.")
        if len(values) == 1:
            return np.ones(1, dtype=float)
        ranks = rankdata(values, method="average")
        return np.asarray((ranks - 1.0) / (len(values) - 1.0), dtype=float)

    @staticmethod
    def _global_ordering_progress(values: np.ndarray) -> np.ndarray:
        """Map fitted ordering values to one shared 0--1 display scale.

        Unlike within-fate ranks, this transformation preserves the relative
        spacing and timing between terminal populations.  It is used only by
        the two-dimensional display layer; the scientific terminal vertices
        remain fixed at equal radius.
        """
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Display ordering values must be a non-empty 1D array.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Display ordering values contain non-finite values.")
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper <= lower:
            return np.ones(len(values), dtype=float)
        return np.clip((values - lower) / (upper - lower), 0.0, 1.0)

    @staticmethod
    def _display_arm_directions(
        fate_names: Sequence[str],
        *,
        terminal_span_degrees: float = 150.0,
    ) -> dict[str, np.ndarray]:
        """Return deterministic 2D display directions for root and fate arms."""
        if not 0.0 < terminal_span_degrees < 180.0:
            raise ValueError("terminal_span_degrees must lie in (0, 180).")
        names = tuple(map(str, fate_names))
        if not names:
            raise ValueError("At least one fate name is required.")
        directions: dict[str, np.ndarray] = {"__root__": np.array([-1.0, 0.0])}
        angles = np.linspace(
            -terminal_span_degrees / 2.0,
            terminal_span_degrees / 2.0,
            len(names),
        )
        for name, angle_degrees in zip(names, angles):
            angle = np.deg2rad(angle_degrees)
            directions[name] = np.array([np.cos(angle), np.sin(angle)])
        return directions

    def _display_coordinates(
        self,
        output: FurcationScoreResult,
        *,
        jitter: float = 0.025,
        terminal_radial_jitter: Optional[float] = None,
        seed: int = 0,
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Construct a two-dimensional display-only star.

        Root cells preserve their scientific radial ordering along the incoming
        arm. With ``terminal_layout="ordering"`` (default), cells in each
        annotated terminal population are spread along its display ray using
        the fitted ordering values on one shared scale.  When the fitted
        ordering is velocity pseudotime, radial display position therefore
        represents that pseudotime directly rather than a within-fate rank.
        ``terminal_layout="rank"`` remains available when equal visual filling
        of every branch is desired.

        ``terminal_layout="endpoint"`` reproduces compact endpoint clouds.
        Both layouts are display-only and therefore cannot alter projected
        velocity, affinity, commitment, or population summaries.
        """
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError("jitter must be non-negative and finite.")
        if terminal_radial_jitter is None:
            terminal_radial_jitter = 0.5 * jitter
        if not np.isfinite(terminal_radial_jitter) or terminal_radial_jitter < 0:
            raise ValueError("terminal_radial_jitter must be non-negative and finite.")
        if not 0.0 < terminal_span_degrees < 180.0:
            raise ValueError("terminal_span_degrees must lie in (0, 180).")
        # ``branch`` was the development name in dev15 and ``pseudotime``
        # was the dev16 name. Preserve both as aliases for the generic fitted-
        # ordering display. The fitted ordering is often velocity pseudotime,
        # but may instead be latent time, diffusion pseudotime, CytoTRACE-
        # derived progression, or another biologically justified metric.
        if terminal_layout in {"branch", "pseudotime"}:
            terminal_layout = "ordering"
        if terminal_layout not in {"ordering", "rank", "endpoint"}:
            raise ValueError(
                "terminal_layout must be 'ordering', 'rank', or 'endpoint'. "
                "The legacy aliases 'pseudotime' and 'branch' are also accepted."
            )
        if not np.isfinite(terminal_inner_radius) or not 0.0 <= terminal_inner_radius < 1.0:
            raise ValueError("terminal_inner_radius must lie in [0, 1).")

        ordering = output.embedding.ordering
        n = len(output.cell_ids)
        coords = np.zeros((n, 2), dtype=float)
        arm_directions = self._display_arm_directions(
            output.fate_names,
            terminal_span_degrees=terminal_span_degrees,
        )

        root = output.root_mask
        terminal = output.terminal_mask
        arm_scale = float(output.embedding.arm_scale)

        root_direction = arm_directions["__root__"]
        root_radius = arm_scale * (1.0 - ordering.root_progress[root])
        coords[root] = root_radius[:, None] * root_direction[None, :]

        selected_display_ordering = None
        if terminal_layout in {"ordering", "rank"}:
            selected_display_ordering = self._selected_display_ordering(output)
        selected_global_progress = None
        if terminal_layout == "ordering":
            assert selected_display_ordering is not None
            selected_global_progress = self._global_ordering_progress(selected_display_ordering)

        terminal_base_radius = np.zeros(n, dtype=float)
        for name in output.fate_names:
            direction = arm_directions[name]
            mask = terminal & (ordering.terminal_names == name)

            if terminal_layout == "endpoint":
                radius = np.full(mask.sum(), arm_scale, dtype=float)
            elif terminal_layout == "rank":
                assert selected_display_ordering is not None
                progress = self._within_group_progress(
                    selected_display_ordering[mask],
                )
                radius = arm_scale * (
                    terminal_inner_radius + (1.0 - terminal_inner_radius) * progress
                )
            else:
                assert selected_global_progress is not None
                progress = selected_global_progress[mask]
                radius = arm_scale * (
                    terminal_inner_radius + (1.0 - terminal_inner_radius) * progress
                )

            terminal_base_radius[mask] = radius
            coords[mask] = radius[:, None] * direction[None, :]

        if jitter > 0 or terminal_radial_jitter > 0:
            if max(jitter, terminal_radial_jitter) > 0.10 * arm_scale:
                warnings.warn(
                    "Display jitter is large relative to arm_scale and may obscure "
                    "the star geometry. Values around 0.01-0.04 are recommended.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            rng = np.random.default_rng(seed)

            if jitter > 0:
                root_perpendicular = np.array([0.0, 1.0])
                root_taper = 0.20 + 0.80 * (root_radius / arm_scale)
                root_noise = rng.normal(0.0, jitter, size=root.sum()) * root_taper
                coords[root] += root_noise[:, None] * root_perpendicular[None, :]

            for name in output.fate_names:
                mask = terminal & (ordering.terminal_names == name)
                direction = arm_directions[name]
                perpendicular = np.array([-direction[1], direction[0]])
                radius = terminal_base_radius[mask]
                tangential_taper = 0.20 + 0.80 * (radius / arm_scale)
                tangential_noise = rng.normal(0.0, jitter, size=mask.sum()) * tangential_taper
                radial_noise = rng.normal(
                    0.0,
                    terminal_radial_jitter,
                    size=mask.sum(),
                )
                coords[mask] += (
                    tangential_noise[:, None] * perpendicular[None, :]
                    + radial_noise[:, None] * direction[None, :]
                )

        return coords, arm_directions

    def plot_star(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        color_by: str = "specific_commitment",
        color_values: Optional[Sequence[float]] = None,
        color_label: Optional[str] = None,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        sort_by_color: bool = False,
        jitter: float = 0.025,
        terminal_radial_jitter: Optional[float] = None,
        seed: int = 0,
        point_size: float = 10.0,
        alpha: float = 0.75,
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        show_guides: bool = True,
        label_arms: bool = True,
        cell_mask: Optional[np.ndarray] = None,
        color_mask: Optional[np.ndarray] = None,
        background_color: str = "0.82",
        background_alpha: float = 0.45,
        background_point_size: Optional[float] = None,
        title: Optional[str] = None,
        ordering_label: Optional[str] = None,
        ax=None,
    ):
        """Plot a readable 2D display-only star.

        The scientific embedding is the high-dimensional simplex stored in
        ``X_sccs_score``. Root cells preserve radial ordering. By default,
        terminal cells are spread along each display branch using the fitted
        ordering metric on one shared 0--1 scale. In the common velocity-
        pseudotime workflow, this is direct pseudotime-based positioning; the
        same display also supports latent time or any other fitted continuous
        ordering. ``terminal_layout="rank"``
        equalizes visual coverage within each fate, and
        ``terminal_layout="endpoint"`` draws compact endpoint clouds.  None of
        these display layouts enters scientific scoring.

        Parameters
        ----------
        terminal_layout
            ``"ordering"`` (default) spreads terminal cells using one shared
            scale of the fitted ordering values. ``"pseudotime"`` and
            ``"branch"`` are compatibility aliases. ``"rank"`` uses within-fate
            fractional ranks. ``"endpoint"`` places compact endpoint clouds.
        terminal_inner_radius
            Inner radial fraction for ``terminal_layout="ordering"`` or
            ``terminal_layout="rank"``. A value of
            0.15 fills most of each branch while keeping terminal cells
            visually separate from the furcation origin.
        color_mask
            Optional Boolean mask selecting cells that receive the requested
            coloring. Other displayed cells remain as neutral gray context.
            This is useful for root-focused commitment plots that retain the
            annotated terminal branches as visual anchors.
        """
        self._check_fitted()
        output = self._result if result is None else result
        assert output is not None
        future_mode = getattr(output, "scoring_mode", "instantaneous") == "future_fate"
        color_aliases = {
            "future_fate_reach": "commitment_strength",
            "discounted_fate_reach": "commitment_strength",
            "future_fate_specificity": "directional_specificity",
            "future_fate_entropy": "directional_entropy",
            "reach_supported_specificity": "specific_commitment",
            "resolved_commitment": "specific_commitment",
            "signed_progression": "progression_velocity",
            "signed_ordering_flux": "progression_velocity",
            "selected_path_coverage": "transition_coverage",
        }
        if ":" in color_by:
            prefix, fate = color_by.split(":", 1)
            prefix = {
                "future_fate_affinity": "affinity",
                "conditional_fate_affinity": "affinity",
                "future_fate_contribution": "commitment_contribution",
            }.get(prefix, prefix)
            color_by = f"{prefix}:{fate}"
        else:
            color_by = color_aliases.get(color_by, color_by)

        if cell_mask is None:
            display_mask = np.ones(output.n_cells, dtype=bool)
        else:
            display_mask = np.asarray(cell_mask)
            if (
                display_mask.dtype != bool
                or display_mask.ndim != 1
                or len(display_mask) != output.n_cells
            ):
                raise ValueError("cell_mask must be a Boolean array aligned to result cells.")
            if not np.any(display_mask):
                raise ValueError("cell_mask selects zero cells.")

        if color_mask is None:
            colored_mask = display_mask.copy()
        else:
            supplied_color_mask = np.asarray(color_mask)
            if (
                supplied_color_mask.dtype != bool
                or supplied_color_mask.ndim != 1
                or len(supplied_color_mask) != output.n_cells
            ):
                raise ValueError("color_mask must be a Boolean array aligned to result cells.")
            colored_mask = display_mask & supplied_color_mask
            if not np.any(colored_mask):
                raise ValueError("color_mask selects zero displayed cells.")
        neutral_mask = display_mask & ~colored_mask
        if not np.isfinite(background_alpha) or not 0.0 <= background_alpha <= 1.0:
            raise ValueError("background_alpha must lie in [0, 1].")
        if background_point_size is None:
            background_point_size = point_size
        if not np.isfinite(background_point_size) or background_point_size <= 0:
            raise ValueError("background_point_size must be positive and finite.")

        import matplotlib.pyplot as plt

        coords, arm_directions = self._display_coordinates(
            output,
            jitter=jitter,
            terminal_radial_jitter=terminal_radial_jitter,
            seed=seed,
            terminal_span_degrees=terminal_span_degrees,
            terminal_layout=terminal_layout,
            terminal_inner_radius=terminal_inner_radius,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        arm_scale = float(output.embedding.arm_scale)
        if show_guides:
            ax.plot(
                [-arm_scale, 0.0],
                [0.0, 0.0],
                linewidth=1.0,
                color="0.75",
                zorder=0,
            )
            for name in output.fate_names:
                direction = arm_directions[name]
                endpoint = arm_scale * direction
                ax.plot(
                    [0.0, endpoint[0]],
                    [0.0, endpoint[1]],
                    linewidth=1.0,
                    color="0.75",
                    zorder=0,
                )

        if np.any(neutral_mask):
            ax.scatter(
                coords[neutral_mask, 0],
                coords[neutral_mask, 1],
                s=background_point_size,
                alpha=background_alpha,
                linewidths=0.0,
                color=background_color,
                zorder=1,
            )

        continuous = {
            "specific_commitment": output.specific_commitment,
            "commitment_strength": output.commitment_strength,
            "directional_specificity": output.directional_specificity,
            "directional_entropy": output.directional_entropy,
            "commitment_entropy": output.commitment_entropy,
            "nearest_fate_angle_degrees": output.nearest_fate_angle_degrees,
            "transition_coverage": output.projection.transition_coverage,
            "progression_velocity": output.progression_velocity,
        }
        fate_specific = None
        fate_specific_label = None
        if ":" in color_by:
            prefix, fate = color_by.split(":", 1)
            if fate not in output.fate_names:
                raise ValueError(
                    f"Unknown fate {fate!r}; expected one of {list(output.fate_names)}."
                )
            fate_index = output.fate_names.index(fate)
            if prefix in {"affinity", "directional_affinity"}:
                fate_specific = output.directional_affinity[:, fate_index]
                fate_specific_label = (
                    f"Future-fate affinity: {fate}"
                    if future_mode
                    else f"Directional affinity: {fate}"
                )
            elif prefix == "commitment_affinity":
                fate_specific = output.commitment.commitment_affinity[:, fate_index]
                fate_specific_label = (
                    f"Reach-adjusted future affinity: {fate}"
                    if future_mode
                    else f"Commitment affinity: {fate}"
                )
            elif prefix in {"contribution", "commitment_contribution"}:
                fate_specific = output.commitment_contribution[:, fate_index]
                fate_specific_label = (
                    f"Future-fate contribution: {fate}"
                    if future_mode
                    else f"Commitment contribution: {fate}"
                )
            else:
                raise ValueError(
                    "Fate-specific color_by prefixes are 'affinity', "
                    "'directional_affinity', 'commitment_affinity', "
                    "'contribution', or 'commitment_contribution'."
                )

        external_values = None
        if color_values is not None:
            external_values = np.asarray(color_values, dtype=float)
            if external_values.ndim != 1 or len(external_values) != output.n_cells:
                raise ValueError(
                    "color_values must be one-dimensional and aligned to result cells."
                )
            if not np.all(np.isfinite(external_values[colored_mask])):
                raise ValueError("color_values contains non-finite colored values.")

        default_title = None
        if external_values is not None or color_by in continuous or fate_specific is not None:
            if external_values is not None:
                values = external_values
                label = "value" if color_label is None else str(color_label)
            else:
                values = continuous[color_by] if fate_specific is None else fate_specific
                continuous_labels = (
                    {
                        "specific_commitment": "Reach-supported specificity",
                        "commitment_strength": "Future-fate reach",
                        "directional_specificity": "Future-fate specificity",
                        "directional_entropy": "Future-fate entropy",
                        "commitment_entropy": "Reach-adjusted future entropy",
                        "nearest_fate_angle_degrees": "Not defined in future-fate mode",
                        "transition_coverage": "One-step selected-path coverage",
                        "progression_velocity": "Signed ordering progression",
                    }
                    if future_mode
                    else {
                        "specific_commitment": "Specific commitment",
                        "commitment_strength": "Commitment strength",
                        "directional_specificity": "Directional specificity",
                        "directional_entropy": "Directional entropy",
                        "commitment_entropy": "Commitment entropy",
                        "nearest_fate_angle_degrees": "Nearest fate angle (degrees)",
                        "transition_coverage": "Transition coverage",
                        "progression_velocity": "Progression velocity",
                    }
                )
                label = (
                    continuous_labels.get(color_by, color_by)
                    if fate_specific_label is None
                    else fate_specific_label
                )
            default_title = label
            values = np.asarray(values, dtype=float)
            indices = np.flatnonzero(colored_mask)
            if sort_by_color:
                indices = indices[np.argsort(values[indices], kind="stable")]
            scatter = ax.scatter(
                coords[indices, 0],
                coords[indices, 1],
                c=values[indices],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=point_size,
                alpha=alpha,
                linewidths=0.0,
            )
            if colorbar:
                ax.figure.colorbar(scatter, ax=ax, label=label)
        elif color_by == "population":
            default_title = "Cell annotation"
            groups = [(str(self.root), output.root_mask)]
            groups.extend(
                (str(name), output.terminal_mask & (output.embedding.terminal_names == name))
                for name in output.fate_names
            )
            color_cycle = plt.get_cmap("tab10").colors
            for index, (label, mask) in enumerate(groups):
                mask = mask & colored_mask
                if not np.any(mask):
                    continue
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=point_size,
                    alpha=alpha,
                    linewidths=0.0,
                    label=label,
                    color=color_cycle[index % len(color_cycle)],
                )
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        elif color_by in {
            "status",
            "dominant_fate",
            "dominant_affinity",
            "dominant_direction",
            "root_dominant_affinity",
            "dominant_affinity_root",
        }:
            if color_by == "status":
                default_title = "Commitment status"
                values = output.status
                unique = list(dict.fromkeys(np.asarray(values).astype(str).tolist()))
                color_cycle = plt.get_cmap("tab10").colors
                color_map = {
                    label: color_cycle[index % len(color_cycle)]
                    for index, label in enumerate(unique)
                }
            elif color_by == "dominant_fate":
                default_title = "Fate-committed direction"
                dominant = np.asarray(output.dominant_fate).astype(str)
                values = np.where(
                    np.isin(dominant, output.fate_names),
                    dominant,
                    "not fate-committed",
                )
                unique = list(output.fate_names)
                if np.any(values == "not fate-committed"):
                    unique.append("not fate-committed")
            else:
                default_title = (
                    "Dominant future-fate affinity"
                    if future_mode
                    else "Dominant directional affinity"
                )
                dominant_indices = np.argmax(output.directional_affinity, axis=1)
                dominant = np.asarray(output.fate_names, dtype=object)[dominant_indices]
                values = dominant.astype(object)
                if color_by in {
                    "root_dominant_affinity",
                    "dominant_affinity_root",
                }:
                    default_title = "Dominant directional affinity in root cells"
                    values[output.terminal_mask] = np.asarray(
                        output.embedding.terminal_names[output.terminal_mask],
                        dtype=object,
                    )
                    undefined = output.root_mask & ~output.projection.velocity_defined
                else:
                    undefined = ~output.projection.velocity_defined
                values[undefined] = "undefined"
                unique = list(output.fate_names)
                if np.any(values == "undefined"):
                    unique.append("undefined")

            if color_by != "status":
                color_cycle = plt.get_cmap("tab10").colors
                color_map = {
                    name: color_cycle[(index + 1) % len(color_cycle)]
                    for index, name in enumerate(output.fate_names)
                }
                color_map["not fate-committed"] = "0.70"
                color_map["undefined"] = "0.70"

            for label in unique:
                mask = (np.asarray(values).astype(str) == label) & colored_mask
                if not np.any(mask):
                    continue
                if color_by == "status":
                    legend_label = label.replace("_", " ").capitalize()
                elif label in {"not fate-committed", "undefined"}:
                    legend_label = label.capitalize()
                else:
                    legend_label = label
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=point_size,
                    alpha=alpha,
                    linewidths=0.0,
                    label=legend_label,
                    color=color_map[label],
                )
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            raise ValueError(
                "color_by must be a continuous metric, 'population', 'status', "
                "'dominant_fate', 'dominant_affinity', "
                "'root_dominant_affinity', or a fate-specific "
                "specification such as "
                "'affinity:Beta' or 'commitment_contribution:Beta'."
            )

        if label_arms:
            ax.text(
                -1.08 * arm_scale,
                0.075 * arm_scale,
                str(self.root),
                ha="right",
                va="bottom",
                fontsize=10,
                clip_on=False,
            )
            for name in output.fate_names:
                endpoint = 1.06 * arm_scale * arm_directions[name]
                ax.text(
                    endpoint[0],
                    endpoint[1],
                    str(name),
                    ha="left" if endpoint[0] >= 0 else "right",
                    va="center",
                    fontsize=10,
                    clip_on=False,
                )

        margin = 0.28 * arm_scale
        ax.set_xlim(-arm_scale - margin, arm_scale + margin)
        ax.set_ylim(-arm_scale - margin, arm_scale + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("display axis 1")
        ax.set_ylabel("display axis 2")
        if title is None:
            title = default_title or "scCS supervised furcation"
            if terminal_layout in {"ordering", "pseudotime", "branch"}:
                ordering_key = output.embedding.ordering_key
                display_ordering = ordering_key if ordering_label is None else str(ordering_label)
                if display_ordering:
                    title += f"\nTerminal placement: {display_ordering}"
        ax.set_title(title)
        return ax.figure

    def _selected_transition_weights(
        self,
        output: FurcationScoreResult,
        *,
        renormalize: Optional[bool] = None,
    ):
        """Return the induced selected-cell transition matrix used for display QC."""
        if self._transition_matrix is None:
            raise RuntimeError(
                "A transition matrix is required for velocity embedding plots. "
                "It is unavailable after load_velocity_vectors()."
            )

        matrix = self._transition_matrix
        indices = np.asarray(output.embedding.selected_indices, dtype=int)
        if matrix.shape == (output.n_cells, output.n_cells):
            retained = matrix.copy()
        elif sparse.issparse(matrix):
            retained = matrix.tocsr()[indices][:, indices].copy()
        else:
            dense = np.asarray(matrix, dtype=float)
            retained = dense[np.ix_(indices, indices)].copy()

        if not sparse.issparse(retained):
            retained = sparse.csr_matrix(retained)
        else:
            retained = retained.tocsr()

        # Preserve diagonal/self-transition mass exactly.  The scientific
        # projection includes self transitions in the retained row sum before
        # optional renormalization.  A self transition contributes zero
        # displacement, but it still scales all non-self displacements through
        # that normalization.  Removing the diagonal here would therefore
        # replay a different transition operator and can make otherwise exact
        # progression-identity diagnostics fail on real scVelo graphs.
        retained.eliminate_zeros()

        do_renormalize = (
            output.projection.renormalized if renormalize is None else bool(renormalize)
        )
        if do_renormalize:
            row_sum = np.asarray(retained.sum(axis=1)).ravel().astype(float)
            inverse = np.zeros_like(row_sum)
            positive = row_sum > 0
            inverse[positive] = 1.0 / row_sum[positive]
            retained = sparse.diags(inverse) @ retained

        invalid = ~np.asarray(output.projection.velocity_defined, dtype=bool)
        if np.any(invalid):
            keep = np.ones(output.n_cells, dtype=float)
            keep[invalid] = 0.0
            retained = sparse.diags(keep) @ retained
        return retained.tocsr()

    def _transition_expected_display_velocity(
        self,
        output: FurcationScoreResult,
        coordinates: np.ndarray,
    ) -> np.ndarray:
        """Project the selected transition matrix as direct display displacement.

        This is the display-space analogue of the scientific projection:
        ``E_T[X_j] - X_i``.  Unlike :func:`scvelo.tl.velocity_embedding`, it
        does not normalize every neighbor displacement to unit length or
        subtract a uniform-neighbor baseline.  It is therefore the preferred
        velocity field on the highly constrained star display.
        """
        transition = self._selected_transition_weights(output)
        row_sum = np.asarray(transition.sum(axis=1)).ravel().astype(float)
        velocity = np.asarray(transition @ coordinates, dtype=float)
        velocity -= row_sum[:, None] * np.asarray(coordinates, dtype=float)
        invalid = ~np.asarray(output.projection.velocity_defined, dtype=bool)
        velocity[invalid] = np.nan
        return velocity

    def prepare_scvelo_star_embedding(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        basis: str = "sccs",
        vkey: str = "velocity",
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        projection_mode: str = "transition",
        autoscale: bool = True,
        write_to_adata: bool = True,
    ):
        """Prepare a selected-cell AnnData for native scVelo star visualization.

        The method writes deterministic two-dimensional display coordinates to
        ``X_<basis>``.  ``projection_mode='transition'`` stores the direct
        transition-expected displacement ``E_T[X_j] - X_i`` and is the
        recommended star field.  ``projection_mode='scvelo'`` reproduces
        :func:`scvelo.tl.velocity_embedding`, whose normalized and
        baseline-centered formula can reverse arrows on a nearly collinear
        star and is retained as an explicit sensitivity view.  Both are
        visualization-only; scientific scCS scores continue to use
        ``X_sccs_score`` and the full regular-simplex projection.
        """
        self._check_fitted()
        self._check_instantaneous_mode("prepare_scvelo_star_embedding()")
        output = self._result if result is None else result
        assert output is not None
        projection_mode = str(projection_mode).lower()
        aliases = {
            "expected": "transition",
            "expected_displacement": "transition",
            "direct": "transition",
            "native": "scvelo",
        }
        projection_mode = aliases.get(projection_mode, projection_mode)
        if projection_mode not in {"transition", "scvelo"}:
            raise ValueError("projection_mode must be 'transition' or 'scvelo'.")
        if projection_mode == "scvelo":
            try:
                import scvelo as scv
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "scvelo is required for projection_mode='scvelo'. Install scCS-py[velocity]."
                ) from exc
            if vkey not in self.adata.layers:
                raise KeyError(
                    f"Missing adata.layers[{vkey!r}]. Native scVelo embedding requires "
                    "the velocity layer used to construct the transition graph."
                )
        basis = str(basis)
        if not basis:
            raise ValueError("basis must be a non-empty string.")

        coordinates, _ = self._display_coordinates(
            output,
            jitter=0.0,
            terminal_radial_jitter=0.0,
            seed=0,
            terminal_span_degrees=terminal_span_degrees,
            terminal_layout=terminal_layout,
            terminal_inner_radius=terminal_inner_radius,
        )
        selected_indices = np.asarray(output.embedding.selected_indices, dtype=int)
        adata_star = self.adata[selected_indices].copy()
        adata_star.obsm[f"X_{basis}"] = coordinates.copy()
        if self.obs_key in adata_star.obs:
            series = adata_star.obs[self.obs_key]
            if hasattr(series, "cat"):
                try:
                    adata_star.obs[self.obs_key] = series.cat.remove_unused_categories()
                except AttributeError:
                    pass

        velocity_key = f"{vkey}_{basis}"
        if projection_mode == "transition":
            adata_star.obsm[velocity_key] = self._transition_expected_display_velocity(
                output,
                coordinates,
            )
        else:
            transition = self._selected_transition_weights(output)
            scv.tl.velocity_embedding(
                adata_star,
                basis=basis,
                vkey=vkey,
                T=transition,
                use_negative_cosines=False,
                retain_scale=False,
                autoscale=autoscale,
                all_comps=True,
            )

        if write_to_adata:
            full_coordinates = np.full((self.adata.n_obs, 2), np.nan, dtype=float)
            full_velocity = np.full((self.adata.n_obs, 2), np.nan, dtype=float)
            full_coordinates[selected_indices] = coordinates
            full_velocity[selected_indices] = np.asarray(
                adata_star.obsm[velocity_key],
                dtype=float,
            )
            self.adata.obsm[f"X_{basis}"] = full_coordinates
            self.adata.obsm[velocity_key] = full_velocity
            self.adata.uns.setdefault("sccs", {})["display_velocity_embedding"] = {
                "basis": basis,
                "vkey": vkey,
                "terminal_layout": terminal_layout,
                "terminal_span_degrees": float(terminal_span_degrees),
                "projection_mode": projection_mode,
                "display_only": True,
            }
            if self.adata_sub is not None:
                self.adata_sub.obsm[f"X_{basis}"] = coordinates.copy()
                self.adata_sub.obsm[velocity_key] = np.asarray(
                    adata_star.obsm[velocity_key],
                    dtype=float,
                ).copy()
        return adata_star

    def display_velocity_projection(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        vkey: str = "velocity",
        projection_mode: str = "transition",
    ) -> tuple[np.ndarray, ProjectionResult]:
        """Return display-space velocities in the two-dimensional star.

        ``projection_mode='transition'`` returns direct transition-expected
        displacement; ``'scvelo'`` returns scVelo's centered embedding
        projection.  Both remain display-only and are not used by the
        scientific scorer.
        """
        self._check_fitted()
        self._check_instantaneous_mode("display_velocity_projection()")
        output = self._result if result is None else result
        assert output is not None
        adata_star = self.prepare_scvelo_star_embedding(
            output,
            basis="sccs",
            vkey=vkey,
            terminal_span_degrees=terminal_span_degrees,
            terminal_layout=terminal_layout,
            terminal_inner_radius=terminal_inner_radius,
            projection_mode=projection_mode,
            write_to_adata=False,
        )
        coordinates = np.asarray(adata_star.obsm["X_sccs"], dtype=float)
        velocity = np.asarray(adata_star.obsm[f"{vkey}_sccs"], dtype=float)
        defined = np.isfinite(velocity).all(axis=1) & (
            np.linalg.norm(np.nan_to_num(velocity, nan=0.0), axis=1) > np.finfo(float).eps
        )
        return coordinates, ProjectionResult(
            velocity=velocity,
            retained_transition_mass=output.projection.retained_transition_mass.copy(),
            external_transition_mass=output.projection.external_transition_mass.copy(),
            transition_coverage=output.projection.transition_coverage.copy(),
            velocity_defined=defined,
            selected_indices=np.asarray(output.embedding.selected_indices, dtype=int).copy(),
            renormalized=output.projection.renormalized,
            min_coverage=output.projection.min_coverage,
        )

    @staticmethod
    def _finite_positive_fraction(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return float("nan")
        return float(np.mean(values[finite] > 0.0))

    def root_progression_direction_diagnostics(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        vkey: str = "velocity",
    ) -> RootProgressionDirectionDiagnostics:
        """Diagnose whether root velocity follows increasing fitted ordering.

        For root cells, terminal destinations are assigned progression 1.  The
        direct transition-weighted change in this coordinate must equal the
        scientific progression component divided by ``arm_scale``.  The method
        also contrasts the recommended transition-expected star displacement
        with scVelo's normalized, baseline-centered embedding projection.
        """
        self._check_fitted()
        self._check_instantaneous_mode("root_progression_direction_diagnostics()")
        output = self._result if result is None else result
        assert output is not None

        transition = self._selected_transition_weights(output)
        row_sum = np.asarray(transition.sum(axis=1)).ravel().astype(float)
        root = np.asarray(output.root_mask, dtype=bool)
        root_local = np.flatnonzero(root)
        progress = np.ones(output.n_cells, dtype=float)
        progress[root] = np.asarray(
            output.embedding.ordering.root_progress[root],
            dtype=float,
        )
        expected_destination = np.asarray(transition @ progress, dtype=float).ravel()
        expected_change = expected_destination - row_sum * progress
        expected_root = expected_change[root]

        arm_scale = float(output.embedding.arm_scale)
        scientific = np.asarray(output.progression_velocity[root], dtype=float)
        identity_error = np.abs(arm_scale * expected_root - scientific)
        finite_identity = np.isfinite(identity_error)
        max_identity_error = (
            float(np.max(identity_error[finite_identity]))
            if np.any(finite_identity)
            else float("nan")
        )

        coordinates, directions = self._display_coordinates(
            output,
            jitter=0.0,
            terminal_radial_jitter=0.0,
            seed=0,
            terminal_span_degrees=terminal_span_degrees,
            terminal_layout=terminal_layout,
            terminal_inner_radius=terminal_inner_radius,
        )
        transition_display = self._transition_expected_display_velocity(
            output,
            coordinates,
        )
        centerward = -np.asarray(directions["__root__"], dtype=float)
        transition_centerward = transition_display[root] @ centerward

        try:
            native = self.prepare_scvelo_star_embedding(
                output,
                basis="sccs_direction_qc",
                vkey=vkey,
                terminal_span_degrees=terminal_span_degrees,
                terminal_layout=terminal_layout,
                terminal_inner_radius=terminal_inner_radius,
                projection_mode="scvelo",
                autoscale=False,
                write_to_adata=False,
            )
            native_velocity = np.asarray(
                native.obsm[f"{vkey}_sccs_direction_qc"],
                dtype=float,
            )
            native_centerward = native_velocity[root] @ centerward
        except (ImportError, KeyError):
            native_centerward = np.full(root.sum(), np.nan, dtype=float)

        metadata = self.adata.uns.get("sccs_v08", {}).get("pseudotime", {})
        ordering_key = str(output.embedding.ordering_key)
        root_key = metadata.get(ordering_key, {}).get("root_key")
        selected_indices = np.asarray(output.embedding.selected_indices, dtype=int)
        selected_root_local_index = int(root_local[np.argmin(progress[root])])
        if root_key is not None:
            root_key_text = str(root_key)
            candidate_local = None
            if root_key_text in self.adata.obs_names:
                matches = np.flatnonzero(
                    np.asarray(self.adata.obs_names[selected_indices]).astype(str) == root_key_text
                )
                if len(matches) == 1:
                    candidate_local = int(matches[0])
            else:
                try:
                    full_index = int(root_key_text)
                except ValueError:
                    full_index = -1
                matches = np.flatnonzero(selected_indices == full_index)
                if len(matches) == 1:
                    candidate_local = int(matches[0])
            if candidate_local is not None and root[candidate_local]:
                selected_root_local_index = candidate_local

        selected_root_progress = float(progress[selected_root_local_index])
        selected_root_radius = float(np.linalg.norm(coordinates[selected_root_local_index]))

        return RootProgressionDirectionDiagnostics(
            root_local_indices=root_local,
            root_progress=progress[root].copy(),
            expected_progress_change=expected_root.copy(),
            scientific_progression_velocity=scientific.copy(),
            transition_display_centerward_velocity=np.asarray(
                transition_centerward, dtype=float
            ).copy(),
            scvelo_display_centerward_velocity=np.asarray(native_centerward, dtype=float).copy(),
            selected_root_local_index=int(selected_root_local_index),
            selected_root_progress=selected_root_progress,
            selected_root_radius=selected_root_radius,
            max_abs_progression_identity_error=max_identity_error,
            forward_expected_progress_fraction=self._finite_positive_fraction(expected_root),
            forward_scientific_fraction=self._finite_positive_fraction(scientific),
            forward_transition_display_fraction=self._finite_positive_fraction(
                transition_centerward
            ),
            forward_scvelo_display_fraction=self._finite_positive_fraction(native_centerward),
        )

    def plot_root_progression_direction_diagnostics(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        diagnostics: Optional[RootProgressionDirectionDiagnostics] = None,
        figsize: tuple[float, float] = (15.0, 4.5),
    ):
        """Plot root ordering and forward-direction consistency diagnostics."""
        self._check_fitted()
        self._check_instantaneous_mode("plot_root_progression_direction_diagnostics()")
        output = self._result if result is None else result
        assert output is not None
        diagnostic = (
            self.root_progression_direction_diagnostics(output)
            if diagnostics is None
            else diagnostics
        )
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].scatter(
            diagnostic.root_progress,
            diagnostic.expected_progress_change,
            s=14,
            alpha=0.65,
        )
        axes[0].axhline(0.0, color="0.4", linewidth=1.0)
        axes[0].set_xlabel("Root progress (0 = earliest, 1 = furcation)")
        axes[0].set_ylabel("Expected change in root progress")
        axes[0].set_title("Direct transition ordering change")

        axes[1].scatter(
            diagnostic.expected_progress_change,
            diagnostic.scientific_progression_velocity,
            s=14,
            alpha=0.65,
        )
        finite = np.isfinite(diagnostic.expected_progress_change) & np.isfinite(
            diagnostic.scientific_progression_velocity
        )
        if np.any(finite):
            values = diagnostic.expected_progress_change[finite]
            lo, hi = float(np.min(values)), float(np.max(values))
            axes[1].plot(
                [lo, hi],
                [
                    output.embedding.arm_scale * lo,
                    output.embedding.arm_scale * hi,
                ],
                color="0.25",
                linestyle="--",
                linewidth=1.2,
            )
        axes[1].set_xlabel("Expected change in root progress")
        axes[1].set_ylabel("Scientific progression velocity")
        axes[1].set_title("Exact scientific orientation identity")

        labels = ["Transition expected", "Scientific", "Native scVelo"]
        values = [
            diagnostic.forward_transition_display_fraction,
            diagnostic.forward_scientific_fraction,
            diagnostic.forward_scvelo_display_fraction,
        ]
        axes[2].bar(labels, values)
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_ylabel("Fraction pointing toward furcation")
        axes[2].set_title("Root direction by projection")
        axes[2].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        return fig

    def _add_display_star_guides(
        self,
        ax,
        output: FurcationScoreResult,
        *,
        terminal_span_degrees: float,
    ) -> None:
        """Overlay deterministic star arms and labels on an existing axis."""
        arm_directions = self._display_arm_directions(
            output.fate_names,
            terminal_span_degrees=terminal_span_degrees,
        )
        arm_scale = float(output.embedding.arm_scale)
        ax.plot([-arm_scale, 0.0], [0.0, 0.0], color="0.78", linewidth=1.0, zorder=0)
        for name in output.fate_names:
            direction = arm_directions[name]
            endpoint = arm_scale * direction
            ax.plot(
                [0.0, endpoint[0]],
                [0.0, endpoint[1]],
                color="0.78",
                linewidth=1.0,
                zorder=0,
            )
            label_position = 1.06 * endpoint
            ax.text(
                label_position[0],
                label_position[1],
                str(name),
                ha="left" if label_position[0] >= 0 else "right",
                va="center",
                fontsize=10,
                clip_on=False,
            )
        ax.text(
            -1.08 * arm_scale,
            0.075 * arm_scale,
            str(self.root),
            ha="right",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )
        margin = 0.28 * arm_scale
        ax.set_xlim(-arm_scale - margin, arm_scale + margin)
        ax.set_ylim(-arm_scale - margin, arm_scale + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("display axis 1")
        ax.set_ylabel("display axis 2")

    def plot_velocity_embedding_grid(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        population: str = "all",
        mask: Optional[np.ndarray] = None,
        basis: str = "sccs",
        vkey: str = "velocity",
        density: float = 1.0,
        smooth: float = 0.5,
        min_mass: float = 1.0,
        n_neighbors: Optional[int] = None,
        arrow_size: float = 1.4,
        arrow_length: float = 3.0,
        scale: Optional[float] = None,
        autoscale: bool = True,
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        projection_mode: str = "transition",
        point_size: float = 22.0,
        alpha: float = 0.45,
        title: Optional[str] = None,
        write_to_adata: bool = True,
        ax=None,
    ):
        """Call ``scv.pl.velocity_embedding_grid`` on the selected star cells.

        Root and terminal cells are included by default.  The arrows are
        smoothed with :func:`scvelo.pl.velocity_embedding_grid`.  By default,
        the vectors are direct transition-expected displacements in ``X_sccs``;
        use ``projection_mode='scvelo'`` only to inspect scVelo's normalized,
        baseline-centered embedding projection.  This plot is visual QC; the
        scientific scCS projection remains the high-dimensional result.
        """
        self._check_fitted()
        self._check_instantaneous_mode("plot_velocity_embedding_grid()")
        output = self._result if result is None else result
        assert output is not None
        try:
            import scvelo as scv
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scvelo is required for plot_velocity_embedding_grid(). Install scCS-py[velocity]."
            ) from exc

        adata_star = self.prepare_scvelo_star_embedding(
            output,
            basis=basis,
            vkey=vkey,
            terminal_span_degrees=terminal_span_degrees,
            terminal_layout=terminal_layout,
            terminal_inner_radius=terminal_inner_radius,
            projection_mode=projection_mode,
            autoscale=autoscale,
            write_to_adata=write_to_adata,
        )
        if mask is not None:
            selected = np.asarray(mask)
            if selected.dtype != bool or selected.ndim != 1 or len(selected) != output.n_cells:
                raise ValueError("mask must be a Boolean array aligned to result cells.")
        elif population == "root":
            selected = output.root_mask.copy()
        elif population == "terminal":
            selected = output.terminal_mask.copy()
        elif population == "all":
            selected = np.ones(output.n_cells, dtype=bool)
        else:
            raise ValueError("population must be 'root', 'terminal', or 'all'.")
        if not np.any(selected):
            raise ValueError("No cells were selected for the velocity grid.")

        plot_data = adata_star[selected].copy()
        X = np.asarray(plot_data.obsm[f"X_{basis}"], dtype=float)
        V = np.asarray(plot_data.obsm[f"{vkey}_{basis}"], dtype=float)
        if n_neighbors is None:
            n_neighbors = min(plot_data.n_obs, max(1, int(plot_data.n_obs / 50)))
        else:
            n_neighbors = min(plot_data.n_obs, max(1, int(n_neighbors)))
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9.0, 6.8))

        palette = list(plt.get_cmap("tab10").colors)
        if title is None:
            title = (
                "Transition-expected RNA-velocity grid on the scCS star"
                if str(projection_mode).lower() not in {"scvelo", "native"}
                else "Native scVelo centered velocity grid on the scCS star"
            )
        scv.pl.velocity_embedding_grid(
            plot_data,
            basis=basis,
            vkey=vkey,
            X=X,
            V=V,
            color=self.obs_key,
            density=density,
            smooth=smooth,
            min_mass=min_mass,
            n_neighbors=n_neighbors,
            arrow_size=arrow_size,
            arrow_length=arrow_length,
            scale=scale,
            autoscale=autoscale,
            palette=palette,
            size=point_size,
            alpha=alpha,
            legend_loc="right margin",
            colorbar=False,
            frameon=True,
            title=title,
            xlabel="display axis 1",
            ylabel="display axis 2",
            show=False,
            ax=ax,
        )
        self._add_display_star_guides(
            ax,
            output,
            terminal_span_degrees=terminal_span_degrees,
        )
        return ax.figure

    def plot_velocity_star(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        population: str = "all",
        mask: Optional[np.ndarray] = None,
        mode: str = "grid",
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        projection_mode: str = "transition",
        title: Optional[str] = None,
        ax=None,
        **kwargs,
    ):
        """Compatibility wrapper for velocity plots on the star.

        ``mode='grid'`` and ``mode='cell'`` use scVelo's plotting functions.
        The default vectors are direct transition-expected displacements; set
        ``projection_mode='scvelo'`` for the native centered projection.  All
        selected root and terminal cells are included by default.
        """
        self._check_fitted()
        self._check_instantaneous_mode("plot_velocity_star()")
        output = self._result if result is None else result
        assert output is not None
        mode = str(mode).lower()
        if mode == "grid":
            grid_size = kwargs.pop("grid_size", None)
            if grid_size is not None and "density" not in kwargs:
                kwargs["density"] = max(float(grid_size) / 50.0, 0.05)
            # Retained for compatibility with the former custom grid. scVelo
            # uses kernel mass rather than a hard cells-per-bin threshold.
            kwargs.pop("min_cells_per_bin", None)
            kwargs.pop("max_arrows", None)
            kwargs.pop("cmap", None)
            kwargs.pop("colorbar", None)
            return self.plot_velocity_embedding_grid(
                output,
                population=population,
                mask=mask,
                terminal_span_degrees=terminal_span_degrees,
                terminal_layout=terminal_layout,
                terminal_inner_radius=terminal_inner_radius,
                projection_mode=projection_mode,
                title=title,
                ax=ax,
                **kwargs,
            )
        if mode != "cell":
            raise ValueError("mode must be 'cell' or 'grid'.")
        try:
            import scvelo as scv
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scvelo is required for plot_velocity_star(). Install scCS-py[velocity]."
            ) from exc
        basis = str(kwargs.pop("basis", "sccs"))
        vkey = str(kwargs.pop("vkey", "velocity"))
        max_arrows = kwargs.pop("max_arrows", None)
        density = kwargs.pop("density", 0.35)
        if max_arrows is not None:
            density = min(1.0, max(float(max_arrows) / max(output.n_cells, 1), 0.01))
        kwargs.pop("cmap", None)
        kwargs.pop("colorbar", None)
        arrow_size = kwargs.pop("arrow_size", 1.2)
        arrow_length = kwargs.pop("arrow_length", 2.5)
        scale = kwargs.pop("scale", None)
        point_size = kwargs.pop("point_size", 18.0)
        alpha = kwargs.pop("alpha", 0.40)
        write_to_adata = kwargs.pop("write_to_adata", True)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments for mode='cell': {unexpected}.")

        adata_star = self.prepare_scvelo_star_embedding(
            output,
            basis=basis,
            vkey=vkey,
            terminal_span_degrees=terminal_span_degrees,
            terminal_layout=terminal_layout,
            terminal_inner_radius=terminal_inner_radius,
            projection_mode=projection_mode,
            write_to_adata=write_to_adata,
        )
        if mask is not None:
            selected = np.asarray(mask)
            if selected.dtype != bool or selected.ndim != 1 or len(selected) != output.n_cells:
                raise ValueError("mask must be a Boolean array aligned to result cells.")
        elif population == "root":
            selected = output.root_mask.copy()
        elif population == "terminal":
            selected = output.terminal_mask.copy()
        elif population == "all":
            selected = np.ones(output.n_cells, dtype=bool)
        else:
            raise ValueError("population must be 'root', 'terminal', or 'all'.")
        plot_data = adata_star[selected].copy()
        X = np.asarray(plot_data.obsm[f"X_{basis}"], dtype=float)
        V = np.asarray(plot_data.obsm[f"{vkey}_{basis}"], dtype=float)
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9.0, 6.8))
        palette = list(plt.get_cmap("tab10").colors)
        scv.pl.velocity_embedding(
            plot_data,
            basis=basis,
            vkey=vkey,
            X=X,
            V=V,
            color=self.obs_key,
            density=density,
            arrow_size=arrow_size,
            arrow_length=arrow_length,
            scale=scale,
            palette=palette,
            size=point_size,
            alpha=alpha,
            legend_loc="right margin",
            colorbar=False,
            frameon=True,
            title=title or "RNA-velocity vectors on the scCS star",
            xlabel="display axis 1",
            ylabel="display axis 2",
            show=False,
            ax=ax,
        )
        self._add_display_star_guides(
            ax,
            output,
            terminal_span_degrees=terminal_span_degrees,
        )
        return ax.figure

    def _gene_expression_values(
        self,
        output: FurcationScoreResult,
        gene: str,
        *,
        layer: Optional[str] = None,
        use_raw: bool = False,
        gene_symbols: Optional[str] = None,
        log1p: bool = False,
    ) -> tuple[np.ndarray, str]:
        """Return one gene's expression aligned to selected result cells."""
        if use_raw and layer is not None:
            raise ValueError("layer and use_raw=True cannot be used together.")

        if use_raw:
            if self.adata.raw is None:
                raise ValueError("use_raw=True requires adata.raw.")
            source = self.adata.raw
            matrix = source.X
            var = source.var
            var_names = np.asarray(source.var_names).astype(str)
            source_label = "raw"
        else:
            source = self.adata
            if layer is None:
                matrix = source.X
                source_label = "X"
            else:
                if layer not in source.layers:
                    raise KeyError(f"Layer {layer!r} is missing from adata.layers.")
                matrix = source.layers[layer]
                source_label = f"layer={layer}"
            var = source.var
            var_names = np.asarray(source.var_names).astype(str)

        requested = str(gene)
        if gene_symbols is None:
            matches = np.flatnonzero(var_names == requested)
        else:
            if gene_symbols not in var.columns:
                raise KeyError(
                    f"gene_symbols={gene_symbols!r} is missing from the "
                    "expression source var table."
                )
            symbols = var[gene_symbols].astype(str).to_numpy()
            matches = np.flatnonzero(symbols == requested)

        if len(matches) == 0:
            source_name = "adata.raw" if use_raw else "adata"
            qualifier = f" column {gene_symbols!r}" if gene_symbols is not None else " var_names"
            raise KeyError(f"Gene {requested!r} was not found in {source_name}{qualifier}.")
        if len(matches) > 1:
            raise ValueError(
                f"Gene {requested!r} maps to {len(matches)} variables; "
                "gene identifiers must be unique."
            )

        selected_indices = np.asarray(output.embedding.selected_indices, dtype=int)
        if matrix.shape[0] != self.adata.n_obs:
            raise ValueError(
                "Expression source does not have the same observation axis as "
                "the fitted AnnData object."
            )
        values = matrix[selected_indices, int(matches[0])]
        try:
            from scipy import sparse
        except ImportError:  # pragma: no cover - scipy is a core dependency
            sparse = None
        if sparse is not None and sparse.issparse(values):
            values = values.toarray()
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != output.n_cells:
            raise RuntimeError("Gene expression values are not aligned to result cells.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Expression values for {requested!r} contain non-finite values.")
        if log1p:
            if np.any(values < -1.0):
                raise ValueError("log1p=True requires all expression values to be at least -1.")
            values = np.log1p(values)
            source_label = f"log1p({source_label})"
        return values, source_label

    @staticmethod
    def _continuous_color_limits(
        values: np.ndarray,
        mask: np.ndarray,
        *,
        vmin: Optional[float],
        vmax: Optional[float],
        percentile_range: Optional[tuple[float, float]],
    ) -> tuple[float, float]:
        """Resolve stable finite color limits for a displayed continuous value."""
        displayed = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
        displayed = displayed[np.isfinite(displayed)]
        if len(displayed) == 0:
            raise ValueError("No finite values are available for color scaling.")

        if percentile_range is None:
            lower = float(np.min(displayed))
            upper = float(np.max(displayed))
        else:
            if len(percentile_range) != 2:
                raise ValueError("percentile_range must contain exactly two values.")
            low_q, high_q = map(float, percentile_range)
            if not 0.0 <= low_q < high_q <= 100.0:
                raise ValueError("percentile_range must satisfy 0 <= low < high <= 100.")
            lower, upper = map(
                float,
                np.percentile(displayed, [low_q, high_q]),
            )

        if vmin is not None:
            lower = float(vmin)
        if vmax is not None:
            upper = float(vmax)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("Color limits must be finite.")
        if upper < lower:
            raise ValueError("vmax must be greater than or equal to vmin.")
        if upper <= lower + np.finfo(float).eps:
            delta = max(abs(lower) * 1e-6, 1e-12)
            lower -= delta
            upper += delta
        return lower, upper

    def plot_gene_expression_star(
        self,
        genes: Union[str, Sequence[str]],
        result: Optional[FurcationScoreResult] = None,
        *,
        layer: Optional[str] = None,
        use_raw: bool = False,
        gene_symbols: Optional[str] = None,
        log1p: bool = False,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        percentile_range: Optional[tuple[float, float]] = (1.0, 99.0),
        shared_scale: bool = False,
        sort_cells: bool = True,
        cell_mask: Optional[np.ndarray] = None,
        ncols: int = 3,
        figsize_per_panel: tuple[float, float] = (6.2, 5.3),
        **plot_star_kwargs,
    ):
        """Plot one or more genes directly on the display-only scCS star.

        Expression is read from ``adata.X`` by default, an AnnData ``layer``
        when provided, or ``adata.raw.X`` with ``use_raw=True``. Coordinates
        come from the already fitted star display and therefore preserve the
        selected ordering metric. Gene expression never enters scientific
        projection or commitment scoring.

        Parameters
        ----------
        genes
            One gene identifier or a sequence of identifiers.
        gene_symbols
            Optional column in ``adata.var`` (or ``adata.raw.var``) containing
            display gene symbols when ``var_names`` uses another identifier.
        percentile_range
            Robust display limits calculated independently per gene by default.
            Set to ``None`` to use the full range.
        shared_scale
            Use one color scale across all requested genes. This is most useful
            for genes measured on a directly comparable scale.
        sort_cells
            Draw low-expression cells first so high-expression cells remain
            visible on top.
        """
        self._check_fitted()
        output = self._result if result is None else result
        assert output is not None

        if isinstance(genes, str):
            gene_list = [genes]
        else:
            gene_list = [str(gene) for gene in genes]
        if len(gene_list) == 0:
            raise ValueError("genes must contain at least one gene identifier.")
        if len(set(gene_list)) != len(gene_list):
            raise ValueError("genes contains duplicate identifiers.")
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")
        prohibited = {
            "color_values",
            "color_label",
            "cell_mask",
            "title",
            "ax",
            "cmap",
            "vmin",
            "vmax",
            "sort_by_color",
        }
        overlap = prohibited.intersection(plot_star_kwargs)
        if overlap:
            raise ValueError(
                "plot_star_kwargs contains arguments controlled by "
                f"plot_gene_expression_star: {sorted(overlap)}."
            )

        if cell_mask is None:
            display_mask = np.ones(output.n_cells, dtype=bool)
        else:
            display_mask = np.asarray(cell_mask)
            if (
                display_mask.dtype != bool
                or display_mask.ndim != 1
                or len(display_mask) != output.n_cells
            ):
                raise ValueError("cell_mask must be a Boolean array aligned to result cells.")
            if not np.any(display_mask):
                raise ValueError("cell_mask selects zero cells.")

        expression = []
        source_labels = []
        for gene in gene_list:
            values, source_label = self._gene_expression_values(
                output,
                gene,
                layer=layer,
                use_raw=use_raw,
                gene_symbols=gene_symbols,
                log1p=log1p,
            )
            expression.append(values)
            source_labels.append(source_label)

        if shared_scale:
            combined = np.concatenate([values[display_mask] for values in expression])
            shared_mask = np.ones(len(combined), dtype=bool)
            shared_limits = self._continuous_color_limits(
                combined,
                shared_mask,
                vmin=vmin,
                vmax=vmax,
                percentile_range=percentile_range,
            )
            limits = [shared_limits] * len(gene_list)
        else:
            limits = [
                self._continuous_color_limits(
                    values,
                    display_mask,
                    vmin=vmin,
                    vmax=vmax,
                    percentile_range=percentile_range,
                )
                for values in expression
            ]

        import matplotlib.pyplot as plt

        ncols = min(ncols, len(gene_list))
        nrows = int(np.ceil(len(gene_list) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )
        for gene, values, source_label, limits_pair, axis in zip(
            gene_list,
            expression,
            source_labels,
            limits,
            axes.ravel(),
        ):
            lower, upper = limits_pair
            self.plot_star(
                output,
                color_values=values,
                color_label=f"{gene} expression ({source_label})",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                sort_by_color=sort_cells,
                cell_mask=display_mask,
                title=str(gene),
                ax=axis,
                **plot_star_kwargs,
            )
        for axis in axes.ravel()[len(gene_list) :]:
            axis.set_visible(False)
        fig.suptitle("Gene expression on the scCS star", y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return fig

    def plot_direction_strength_map(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        population: str = "root",
        point_size: float = 14.0,
        alpha: float = 0.7,
        show_thresholds: bool = True,
        ax=None,
    ):
        """Plot directional specificity against commitment strength.

        This separates the two ingredients of specific commitment. Points are
        colored by dominant annotated fate; cells without a defined dominant
        fate are shown as ``unassigned``.
        """
        self._check_fitted()
        self._check_instantaneous_mode("plot_direction_strength_map()")
        output = self._result if result is None else result
        assert output is not None
        if population not in {"root", "all"}:
            raise ValueError("population must be 'root' or 'all'.")
        mask = (
            output.root_mask if population == "root" else np.ones(len(output.cell_ids), dtype=bool)
        )

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 5.4))
        groups = list(output.fate_names) + ["unassigned"]
        dominant = np.asarray(output.dominant_fate).astype(str)
        for group in groups:
            group_mask = mask & (
                (dominant == group)
                if group != "unassigned"
                else ~np.isin(dominant, output.fate_names)
            )
            if not np.any(group_mask):
                continue
            ax.scatter(
                output.directional_specificity[group_mask],
                output.commitment_strength[group_mask],
                s=point_size,
                alpha=alpha,
                linewidths=0.0,
                label=group,
            )
        if show_thresholds:
            ax.axvline(0.25, color="0.55", linestyle="--", linewidth=1.0)
            ax.axhline(0.25, color="0.55", linestyle="--", linewidth=1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Directional specificity")
        ax.set_ylabel("Commitment strength")
        ax.set_title(f"scCS direction–strength map ({population})")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        return ax.figure

    def plot_population_commitment(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        population: str = "root",
        mask: Optional[np.ndarray] = None,
        metric: str = "mean_commitment_contribution",
        ax=None,
    ):
        """Plot one population-level commitment summary.

        Parameters
        ----------
        population
            ``"root"`` (default) or ``"all"``. Ignored when ``mask`` is
            supplied.
        mask
            Optional Boolean mask aligned to ``result.cell_ids``.
        metric
            One of ``"total_commitment_mass"``,
            ``"mean_commitment_contribution"``, ``"commitment_composition"``,
            or ``"pairwise_log_commitment_ratio"``.
        """
        self._check_fitted()
        output = self._result if result is None else result
        assert output is not None

        if mask is not None:
            values_mask = np.asarray(mask)
            if (
                values_mask.dtype != bool
                or values_mask.ndim != 1
                or len(values_mask) != output.n_cells
            ):
                raise ValueError("mask must be a Boolean array aligned to result.cell_ids.")
            summary = output.summarize(values_mask)
            population_label = "custom population"
        elif population == "root":
            summary = output.root_population_summary
            population_label = "root"
        elif population == "all":
            summary = output.summarize(np.ones(output.n_cells, dtype=bool))
            population_label = "all furcation cells"
        else:
            raise ValueError("population must be 'root' or 'all'.")

        allowed = {
            "total_commitment_mass",
            "mean_commitment_contribution",
            "commitment_composition",
            "pairwise_log_commitment_ratio",
        }
        if metric not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}.")

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 4.8))

        if metric == "pairwise_log_commitment_ratio":
            matrix = np.asarray(summary.pairwise_log_commitment_ratio, dtype=float)
            image = ax.imshow(matrix, aspect="equal")
            ticks = np.arange(len(output.fate_names))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(output.fate_names, rotation=30, ha="right")
            ax.set_yticklabels(output.fate_names)
            ax.figure.colorbar(image, ax=ax, label="Pairwise log commitment ratio")
            if matrix.size <= 100:
                for row in range(matrix.shape[0]):
                    for column in range(matrix.shape[1]):
                        if np.isfinite(matrix[row, column]):
                            ax.text(
                                column,
                                row,
                                f"{matrix[row, column]:.2f}",
                                ha="center",
                                va="center",
                            )
        else:
            metric_values = np.asarray(getattr(summary, metric), dtype=float)
            x = np.arange(len(output.fate_names))
            ax.bar(x, metric_values)
            ax.set_xticks(x)
            ax.set_xticklabels(output.fate_names, rotation=30, ha="right")
            label_map = {
                "total_commitment_mass": "Total commitment mass",
                "mean_commitment_contribution": "Mean commitment contribution",
                "commitment_composition": "Commitment composition",
            }
            ax.set_ylabel(label_map[metric])
            if metric == "commitment_composition":
                ax.set_ylim(0.0, 1.0)
        ax.set_title(f"scCS population commitment ({population_label})")
        return ax.figure

    def plot_star_panels(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        panels: Optional[Sequence[str]] = None,
        ncols: int = 3,
        figsize_per_panel: tuple[float, float] = (6.2, 5.3),
        **plot_star_kwargs,
    ):
        """Plot a grid of star views using the current v0.8 outputs."""
        self._check_fitted()
        output = self._result if result is None else result
        assert output is not None
        if panels is None:
            panels = (
                "population",
                "dominant_fate",
                "specific_commitment",
                "commitment_strength",
                "directional_specificity",
                "directional_entropy",
            )
        panels = tuple(str(panel) for panel in panels)
        if not panels:
            raise ValueError("panels must contain at least one color_by specification.")
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")

        import matplotlib.pyplot as plt

        ncols = min(ncols, len(panels))
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )
        for panel, axis in zip(panels, axes.ravel()):
            self.plot_star(output, color_by=panel, ax=axis, **plot_star_kwargs)
        for axis in axes.ravel()[len(panels) :]:
            axis.set_visible(False)
        fig.tight_layout()
        return fig

    def plot_rose(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        population: str = "root",
        mask: Optional[np.ndarray] = None,
        mode: str = "auto",
        n_bins: int = 24,
        normalization: str = "auto",
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        vkey: str = "velocity",
        title: Optional[str] = None,
        ax=None,
    ):
        """Plot a binned polar summary of fate-directed branch velocity.

        ``mode='angular'`` is an exact global angular histogram for a
        three-fate simplex, whose branch subspace is two-dimensional.

        For four or more fates, the scientific branch subspace has dimension
        greater than two and therefore has no distortion-free global polar
        angle.  ``mode='deviation'`` provides a permutation-equivariant binned
        alternative: cells are assigned to their nearest ideal fate axis, and
        each fate-centered polar sector is subdivided by the unsigned angular
        deviation from that axis.  Bars are mirrored around the fate-axis
        label so that no arbitrary left/right direction is invented.

        ``mode='branch'`` is the recommended public rose plot.  It removes
        incoming-root progression by using only the scientific branch component,
        maps the fate-affinity composition onto an evenly spaced 2D fate polygon,
        and bins specific fate-directed velocity mass by angle.  No incoming-root
        direction is forced into a terminal-fate label.

        ``mode='display'`` summarizes the two-dimensional star-display velocity.
        Use ``population='terminal'`` when comparing terminal branches so that
        incoming-root progression cannot be assigned to the nearest terminal
        display sector.  Display mode is visual QC and does not define scCS
        future-fate scores.

        ``mode='fate_mass'`` is the compact one-bar-per-fate summary.
        ``mode='auto'`` uses ``'fate_mass'`` for two fates, the exact ``'angular'``
        plot for three fates, and ``'branch'`` for four or more fates.

        ``normalization='auto'`` uses absolute velocity mass for two- and
        three-fate plots and a within-fate distribution for four-or-more-fate
        angular-deviation plots.  The latter prevents a high-mass fate from
        visually compressing the angular profiles of the other fates.  Use
        ``normalization='mass'`` for absolute mass or
        ``normalization='global_fraction'`` for fractions of total mass.
        """
        self._check_fitted()
        self._check_instantaneous_mode("plot_rose()")
        output = self._result if result is None else result
        assert output is not None

        if mask is not None:
            selected = np.asarray(mask)
            if selected.dtype != bool or selected.ndim != 1 or len(selected) != output.n_cells:
                raise ValueError("mask must be a Boolean array aligned to result cells.")
        elif population == "root":
            selected = output.root_mask.copy()
        elif population == "terminal":
            selected = output.terminal_mask.copy()
        elif population == "all":
            selected = np.ones(output.n_cells, dtype=bool)
        else:
            raise ValueError("population must be 'root', 'terminal', or 'all'.")
        selected &= output.projection.velocity_defined
        if not np.any(selected):
            raise ValueError("No cells with defined projected velocity were selected.")
        if not isinstance(n_bins, int) or n_bins < 6:
            raise ValueError("n_bins must be an integer of at least 6.")

        mode = str(mode).lower()
        if mode == "auto":
            if output.k == 2:
                mode = "fate_mass"
            elif output.k == 3:
                mode = "angular"
            else:
                mode = "branch"
        if mode not in {"angular", "branch", "deviation", "display", "fate_mass"}:
            raise ValueError(
                "mode must be 'auto', 'angular', 'branch', 'deviation', 'display', or 'fate_mass'."
            )
        if mode == "angular" and output.k != 3:
            raise ValueError(
                "An exact global angular rose is available only for k=3. "
                "Use mode='deviation' for a binned fate-centered rose, or "
                "mode='fate_mass' for one bar per fate."
            )

        normalization = str(normalization).lower()
        if normalization == "auto":
            normalization = "within_fate" if mode == "deviation" else "mass"
        if normalization not in {"mass", "global_fraction", "within_fate"}:
            raise ValueError(
                "normalization must be 'auto', 'mass', 'global_fraction', or 'within_fate'."
            )
        if mode == "fate_mass" and normalization == "within_fate":
            raise ValueError(
                "normalization='within_fate' is not meaningful for one-bar-per-fate "
                "mode. Use 'mass' or 'global_fraction'."
            )

        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        if ax is None:
            figure = plt.figure(figsize=(7.0, 7.0))
            ax = figure.add_subplot(111, projection="polar")
        elif getattr(ax, "name", "") != "polar":
            raise ValueError("ax must be a polar Matplotlib axis.")
        figure = ax.figure
        colors = plt.get_cmap("tab10").colors
        vectors = np.asarray(output.branch_velocity[selected], dtype=float)
        magnitudes = np.linalg.norm(vectors, axis=1)
        affinities = np.asarray(output.directional_affinity[selected], dtype=float)
        soft_fate_mass = np.sum(magnitudes[:, None] * affinities, axis=0)
        legend_mass = soft_fate_mass.copy()
        total_magnitude = float(np.sum(magnitudes))

        def normalize_mass(values: np.ndarray, fate_total: Optional[float] = None):
            values = np.asarray(values, dtype=float)
            if normalization == "mass":
                return values
            if normalization == "global_fraction":
                if total_magnitude <= np.finfo(float).eps:
                    return np.zeros_like(values)
                return values / total_magnitude
            if fate_total is None or fate_total <= np.finfo(float).eps:
                return np.zeros_like(values)
            return values / float(fate_total)

        if normalization == "mass":
            radial_label = "Fate-directed velocity mass"
        elif normalization == "global_fraction":
            radial_label = "Fraction of total fate-directed velocity mass"
        else:
            radial_label = "Within-fate velocity-mass fraction"

        if mode == "branch":
            fate_angles = np.linspace(0.0, 2.0 * np.pi, output.k, endpoint=False)
            fate_polygon = np.column_stack([np.cos(fate_angles), np.sin(fate_angles)])
            centered_affinity = affinities - 1.0 / output.k
            composition_vectors = centered_affinity @ fate_polygon
            composition_norm = np.linalg.norm(composition_vectors, axis=1)
            specificity = np.asarray(
                output.directional_specificity[selected],
                dtype=float,
            )
            branch_weights = magnitudes * specificity
            finite = (
                np.isfinite(composition_vectors).all(axis=1)
                & np.isfinite(branch_weights)
                & (composition_norm > np.finfo(float).eps)
                & (branch_weights > np.finfo(float).eps)
            )
            if not np.any(finite):
                raise ValueError(
                    "No nonzero fate-directed branch velocity was selected for the rose plot."
                )
            branch_angles = np.mod(
                np.arctan2(
                    composition_vectors[finite, 1],
                    composition_vectors[finite, 0],
                ),
                2.0 * np.pi,
            )
            weights = branch_weights[finite]
            edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
            mass, _ = np.histogram(branch_angles, bins=edges, weights=weights)
            centers = 0.5 * (edges[:-1] + edges[1:])
            distance = np.abs(np.angle(np.exp(1j * (centers[:, None] - fate_angles[None, :]))))
            nearest_bin_fate = np.argmin(distance, axis=1)
            if normalization == "within_fate":
                plotted_mass = np.zeros_like(mass, dtype=float)
                for fate_index in range(output.k):
                    fate_bins = nearest_bin_fate == fate_index
                    fate_total = float(mass[fate_bins].sum())
                    if fate_total > np.finfo(float).eps:
                        plotted_mass[fate_bins] = mass[fate_bins] / fate_total
            elif normalization == "global_fraction":
                total = float(mass.sum())
                plotted_mass = mass / total if total > np.finfo(float).eps else mass
            else:
                plotted_mass = mass
            ax.bar(
                centers,
                plotted_mass,
                width=0.92 * (2.0 * np.pi / n_bins),
                color=[colors[index % len(colors)] for index in nearest_bin_fate],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(fate_angles)
            ax.set_xticklabels(output.fate_names)
            legend_mass = np.sum(
                branch_weights[:, None] * affinities,
                axis=0,
            )
            default_title = (
                "Specific fate-directed velocity mass by branch direction"
                if normalization == "mass"
                else "Fate-directed branch-direction profile"
            )
            if normalization == "mass":
                radial_label = "Specific fate-directed velocity mass"
            elif normalization == "global_fraction":
                radial_label = "Fraction of specific fate-directed velocity mass"
            else:
                radial_label = "Within-fate branch-direction fraction"

        elif mode == "display":
            adata_star = self.prepare_scvelo_star_embedding(
                output,
                basis="sccs",
                vkey=vkey,
                terminal_span_degrees=terminal_span_degrees,
                terminal_layout=terminal_layout,
                terminal_inner_radius=terminal_inner_radius,
                projection_mode="scvelo",
                autoscale=False,
                write_to_adata=False,
            )
            display_velocity = np.asarray(adata_star.obsm[f"{vkey}_sccs"], dtype=float)
            display_vectors = display_velocity[selected]
            finite = np.isfinite(display_vectors).all(axis=1)
            display_vectors = display_vectors[finite]
            if display_vectors.size == 0:
                raise ValueError("No finite scVelo star-embedding velocities were selected.")
            display_magnitudes = np.linalg.norm(display_vectors, axis=1)
            positive = display_magnitudes > np.finfo(float).eps
            display_vectors = display_vectors[positive]
            display_magnitudes = display_magnitudes[positive]
            if display_vectors.size == 0:
                raise ValueError("No nonzero scVelo star-embedding velocities were selected.")
            display_angles = np.mod(
                np.arctan2(display_vectors[:, 1], display_vectors[:, 0]),
                2.0 * np.pi,
            )
            edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
            mass, _ = np.histogram(
                display_angles,
                bins=edges,
                weights=display_magnitudes,
            )
            centers = 0.5 * (edges[:-1] + edges[1:])
            arm_directions = self._display_arm_directions(
                output.fate_names,
                terminal_span_degrees=terminal_span_degrees,
            )
            fate_angles = np.asarray(
                [
                    np.mod(
                        np.arctan2(arm_directions[name][1], arm_directions[name][0]),
                        2.0 * np.pi,
                    )
                    for name in output.fate_names
                ],
                dtype=float,
            )
            distance = np.abs(np.angle(np.exp(1j * (centers[:, None] - fate_angles[None, :]))))
            nearest_bin_fate = np.argmin(distance, axis=1)
            display_total_magnitude = float(display_magnitudes.sum())
            if normalization == "within_fate":
                plotted_mass = np.zeros_like(mass, dtype=float)
                for fate_index in range(output.k):
                    fate_bins = nearest_bin_fate == fate_index
                    fate_total = float(mass[fate_bins].sum())
                    if fate_total > np.finfo(float).eps:
                        plotted_mass[fate_bins] = mass[fate_bins] / fate_total
            elif normalization == "global_fraction":
                plotted_mass = (
                    mass / display_total_magnitude
                    if display_total_magnitude > np.finfo(float).eps
                    else np.zeros_like(mass, dtype=float)
                )
            else:
                plotted_mass = mass
            ax.bar(
                centers,
                plotted_mass,
                width=0.92 * (2.0 * np.pi / n_bins),
                color=[colors[index % len(colors)] for index in nearest_bin_fate],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(fate_angles)
            ax.set_xticklabels(output.fate_names)
            legend_mass = np.asarray(
                [mass[nearest_bin_fate == index].sum() for index in range(output.k)],
                dtype=float,
            )
            default_title = (
                "Cumulative scVelo velocity magnitude by star direction"
                if normalization == "mass"
                else "scVelo velocity-direction profile on the star"
            )
            if normalization == "mass":
                radial_label = "Display-embedded velocity magnitude"
            elif normalization == "global_fraction":
                radial_label = "Fraction of total display-embedded velocity magnitude"
            else:
                radial_label = "Within-sector display-velocity fraction"

        elif mode == "fate_mass":
            angles = np.linspace(0.0, 2.0 * np.pi, output.k, endpoint=False)
            width = 0.82 * 2.0 * np.pi / output.k
            plotted_mass = normalize_mass(soft_fate_mass)
            ax.bar(
                angles,
                plotted_mass,
                width=width,
                color=[colors[index % len(colors)] for index in range(output.k)],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.7,
            )
            ax.set_xticks(angles)
            ax.set_xticklabels(output.fate_names)
            default_title = (
                "Cumulative fate-directed velocity mass"
                if normalization == "mass"
                else "Fate-directed velocity-mass composition"
            )

        elif mode == "angular":
            directions = output.embedding.geometry.terminal_directions
            _, _, vh = np.linalg.svd(directions, full_matrices=False)
            basis = vh[:2].T
            fate_xy = directions @ basis
            vectors_xy = vectors @ basis

            first_angle = float(np.arctan2(fate_xy[0, 1], fate_xy[0, 0]))
            cosine = np.cos(-first_angle)
            sine = np.sin(-first_angle)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            fate_xy = fate_xy @ rotation.T
            vectors_xy = vectors_xy @ rotation.T
            if np.arctan2(fate_xy[1, 1], fate_xy[1, 0]) < 0:
                fate_xy[:, 1] *= -1.0
                vectors_xy[:, 1] *= -1.0

            angles = np.mod(np.arctan2(vectors_xy[:, 1], vectors_xy[:, 0]), 2.0 * np.pi)
            edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
            mass, _ = np.histogram(angles, bins=edges, weights=magnitudes)
            centers = 0.5 * (edges[:-1] + edges[1:])
            fate_angles = np.mod(np.arctan2(fate_xy[:, 1], fate_xy[:, 0]), 2.0 * np.pi)
            distance = np.abs(np.angle(np.exp(1j * (centers[:, None] - fate_angles[None, :]))))
            nearest_bin_fate = np.argmin(distance, axis=1)
            if normalization == "within_fate":
                plotted_mass = np.zeros_like(mass, dtype=float)
                for fate_index in range(output.k):
                    fate_bins = nearest_bin_fate == fate_index
                    fate_total = float(mass[fate_bins].sum())
                    plotted_mass[fate_bins] = normalize_mass(
                        mass[fate_bins],
                        fate_total=fate_total,
                    )
            else:
                plotted_mass = normalize_mass(mass)
            ax.bar(
                centers,
                plotted_mass,
                width=0.92 * (2.0 * np.pi / n_bins),
                color=[colors[index % len(colors)] for index in nearest_bin_fate],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(fate_angles)
            ax.set_xticklabels(output.fate_names)
            legend_mass = np.asarray(
                [mass[nearest_bin_fate == index].sum() for index in range(output.k)],
                dtype=float,
            )
            default_title = (
                "Cumulative velocity magnitude by exact branch angle"
                if normalization == "mass"
                else "Velocity-direction profile by exact branch angle"
            )

        else:
            cosine = np.asarray(output.fate_cosine_similarity[selected], dtype=float)
            nearest_fate = np.argmax(cosine, axis=1)
            nearest_cosine = np.clip(
                cosine[np.arange(len(cosine)), nearest_fate],
                -1.0,
                1.0,
            )
            deviation_degrees = np.degrees(np.arccos(nearest_cosine))
            # Regular-simplex directions sum to zero, so every nonzero vector
            # has at least one non-negative cosine and nearest deviation <= 90°.
            deviation_degrees = np.clip(deviation_degrees, 0.0, 90.0)

            bins_per_fate = max(2, int(np.ceil(n_bins / output.k)))
            deviation_edges = np.linspace(0.0, 90.0, bins_per_fate + 1)
            sector_width = 2.0 * np.pi / output.k
            half_usable_sector = 0.44 * sector_width
            offset_edges = deviation_edges / 90.0 * half_usable_sector
            offset_centers = 0.5 * (offset_edges[:-1] + offset_edges[1:])
            widths = 0.90 * np.diff(offset_edges)
            fate_angles = np.linspace(0.0, 2.0 * np.pi, output.k, endpoint=False)
            legend_mass = np.zeros(output.k, dtype=float)

            for fate_index, fate_angle in enumerate(fate_angles):
                fate_mask = nearest_fate == fate_index
                mass_by_deviation, _ = np.histogram(
                    deviation_degrees[fate_mask],
                    bins=deviation_edges,
                    weights=magnitudes[fate_mask],
                )
                legend_mass[fate_index] = float(mass_by_deviation.sum())
                plotted_mass = normalize_mass(
                    mass_by_deviation,
                    fate_total=legend_mass[fate_index],
                )
                # Deviation is unsigned in >2D. Mirror half the mass on either
                # side of the fate axis instead of inventing an azimuth. For
                # within-fate fractions, mirror the complete profile because
                # the duplication is a display convention rather than a mass
                # partition; absolute/global mass remains split in half.
                multiplier = 1.0 if normalization == "within_fate" else 0.5
                for sign in (-1.0, 1.0):
                    ax.bar(
                        fate_angle + sign * offset_centers,
                        multiplier * plotted_mass,
                        width=widths,
                        color=colors[fate_index % len(colors)],
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.45,
                        align="center",
                    )
            ax.set_xticks(fate_angles)
            ax.set_xticklabels(output.fate_names)
            default_title = (
                "Fate-directed velocity mass by angular-deviation bin"
                if normalization == "mass"
                else "Within-fate angular-deviation bin profile"
            )

        legend = [
            mpatches.Patch(
                color=colors[index % len(colors)],
                label=f"{name} (M={legend_mass[index]:.2f})",
            )
            for index, name in enumerate(output.fate_names)
        ]
        ax.legend(
            handles=legend,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.04, 1.02),
            borderaxespad=0.0,
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(title or default_title, pad=18)
        ax.text(
            1.02,
            0.02,
            radial_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
        return figure

    def plot_branch_velocity_profiles(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        n_bins: int = 24,
        normalization: str = "within_branch",
        terminal_span_degrees: float = 150.0,
        terminal_layout: str = "ordering",
        terminal_inner_radius: float = 0.15,
        projection_mode: str = "scvelo",
        vkey: str = "velocity",
        ncols: Optional[int] = None,
        title: Optional[str] = None,
    ):
        """Plot branch-relative terminal velocity directions in separate roses.

        Each terminal population is kept attached to its own annotated branch.
        The angle is measured relative to that branch's outward display axis:
        ``0°`` is outward, ``±90°`` is transverse, and ``180°`` is inward.
        This avoids the misleading behavior of a pooled display rose, where an
        inward or curved vector from one annotated branch can be relabeled as
        the nearest *other* branch sector.

        This figure is a display-space RNA-velocity diagnostic.  It does not
        define discounted future-fate affinity, reach, specificity, or signed
        progression.

        Parameters
        ----------
        normalization
            ``"within_branch"`` (default) shows the fraction of each branch's
            own display-embedded velocity magnitude in every angular bin.
            ``"mass"`` shows raw display-embedded velocity magnitude, and
            ``"global_fraction"`` divides by total terminal velocity magnitude.
        projection_mode
            Passed to :meth:`prepare_scvelo_star_embedding`.  ``"scvelo"`` is
            recommended for comparison with scVelo's embedding projection.
        """
        self._check_fitted()
        self._check_instantaneous_mode("plot_branch_velocity_profiles()")
        output = self._result if result is None else result
        assert output is not None

        if not isinstance(n_bins, int) or n_bins < 8:
            raise ValueError("n_bins must be an integer of at least 8.")
        normalization = str(normalization).lower()
        if normalization not in {"within_branch", "mass", "global_fraction"}:
            raise ValueError("normalization must be 'within_branch', 'mass', or 'global_fraction'.")

        import matplotlib.pyplot as plt

        star = self.prepare_scvelo_star_embedding(
            output,
            basis="sccs",
            vkey=vkey,
            terminal_span_degrees=terminal_span_degrees,
            terminal_layout=terminal_layout,
            terminal_inner_radius=terminal_inner_radius,
            projection_mode=projection_mode,
            autoscale=False,
            write_to_adata=False,
        )
        display_velocity = np.asarray(star.obsm[f"{vkey}_sccs"], dtype=float)
        labels = np.asarray(output.embedding.selected_labels).astype(str)
        arm_directions = self._display_arm_directions(
            output.fate_names,
            terminal_span_degrees=terminal_span_degrees,
        )
        colors = plt.get_cmap("tab10").colors

        terminal_mask = np.asarray(output.terminal_mask, dtype=bool)
        terminal_magnitude = np.linalg.norm(display_velocity[terminal_mask], axis=1)
        terminal_magnitude = terminal_magnitude[np.isfinite(terminal_magnitude)]
        global_total = float(np.sum(terminal_magnitude))

        ncols = min(output.k, 4) if ncols is None else int(ncols)
        if ncols < 1:
            raise ValueError("ncols must be at least 1.")
        nrows = int(np.ceil(output.k / ncols))
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.6 * ncols, 4.4 * nrows),
            subplot_kw={"projection": "polar"},
            squeeze=False,
        )
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = 0.92 * (2.0 * np.pi / n_bins)
        plotted_max = 0.0

        for fate_index, fate in enumerate(output.fate_names):
            axis = axes.ravel()[fate_index]
            fate_mask = terminal_mask & (labels == fate)
            vectors = display_velocity[fate_mask]
            finite = np.isfinite(vectors).all(axis=1)
            vectors = vectors[finite]
            magnitudes = np.linalg.norm(vectors, axis=1)
            positive = magnitudes > np.finfo(float).eps
            vectors = vectors[positive]
            magnitudes = magnitudes[positive]

            if vectors.size == 0:
                axis.set_theta_zero_location("E")
                axis.set_theta_direction(1)
                axis.set_xticks([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
                axis.set_xticklabels(["outward", "+90°", "inward", "−90°"])
                axis.set_title(f"{fate}\n{int(fate_mask.sum())} cells", pad=18)
                axis.text(
                    0.5,
                    0.5,
                    "no nonzero\ndisplay velocity",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                continue

            velocity_angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            arm = np.asarray(arm_directions[fate], dtype=float)
            arm_angle = float(np.arctan2(arm[1], arm[0]))
            relative_angles = np.angle(np.exp(1j * (velocity_angles - arm_angle)))
            mass, _ = np.histogram(relative_angles, bins=edges, weights=magnitudes)
            raw_total = float(mass.sum())

            if normalization == "within_branch":
                plotted = mass / raw_total if raw_total > 0.0 else np.zeros_like(mass)
            elif normalization == "global_fraction":
                plotted = mass / global_total if global_total > 0.0 else np.zeros_like(mass)
            else:
                plotted = mass
            plotted_max = max(plotted_max, float(np.max(plotted, initial=0.0)))

            axis.bar(
                centers,
                plotted,
                width=width,
                color=colors[fate_index % len(colors)],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            axis.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
            axis.set_theta_zero_location("E")
            axis.set_theta_direction(1)
            axis.set_xticks([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
            axis.set_xticklabels(["outward", "+90°", "inward", "−90°"])
            axis.set_title(
                f"{fate}\n{int(fate_mask.sum())} cells; raw M={raw_total:.2f}",
                pad=18,
            )

        for axis in axes.ravel()[output.k :]:
            axis.set_visible(False)
        if normalization != "mass" and plotted_max > 0.0:
            for axis in axes.ravel()[: output.k]:
                if axis.get_visible():
                    axis.set_ylim(0.0, plotted_max * 1.05)

        if normalization == "within_branch":
            radial_label = "Fraction of each branch's display-velocity magnitude"
        elif normalization == "global_fraction":
            radial_label = "Fraction of total terminal display-velocity magnitude"
        else:
            radial_label = "Display-embedded velocity magnitude"
        figure.suptitle(
            title or "Branch-relative terminal RNA-velocity directions",
            y=1.02,
        )
        figure.text(0.5, 0.01, radial_label, ha="center", va="bottom", fontsize=9)
        figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
        return figure

    def plot_pairwise_cs(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        population: str = "root",
        mask: Optional[np.ndarray] = None,
        ax=None,
    ):
        """Plot the v0.8 pairwise log commitment-ratio matrix.

        The familiar method name is retained, but the plotted quantity is not
        the pre-v0.8 angular-sector nCS.  Entry ``[i, j]`` is
        ``log(mean_commitment_i / mean_commitment_j)`` for the selected
        population.
        """
        return self.plot_population_commitment(
            result=result,
            population=population,
            mask=mask,
            metric="pairwise_log_commitment_ratio",
            ax=ax,
        )

    def plot_commitment_bar(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        population: str = "root",
        mask: Optional[np.ndarray] = None,
        metric: str = "commitment_composition",
        ax=None,
    ):
        """Compatibility wrapper for a v0.8 population commitment bar chart."""
        if metric == "pairwise_log_commitment_ratio":
            raise ValueError("Use plot_pairwise_cs() for pairwise log commitment ratios.")
        return self.plot_population_commitment(
            result=result,
            population=population,
            mask=mask,
            metric=metric,
            ax=ax,
        )

    def plot_commitment_heatmap(
        self,
        result: Optional[FurcationScoreResult] = None,
        *,
        metric: str = "directional_affinity",
        population: str = "root",
        mask: Optional[np.ndarray] = None,
        sort_by: str = "ordering",
        max_cells: int = 500,
        row_annotation: Optional[Union[str, Sequence[str]]] = None,
        show_annotation_labels: bool = True,
        show_annotation_legends: bool = False,
        show_fate_strip: bool = True,
        ax=None,
    ):
        """Plot per-cell fate outputs with aligned row and fate annotations.

        Parameters
        ----------
        row_annotation
            Optional categorical strip, or sequence of strips, shown to the
            left of the heatmap.
            Supported values are ``"population"``, ``"status"``,
            ``"dominant_affinity"``, and ``"dominant_fate"``.
        show_fate_strip
            Draw a color strip above the heatmap columns using the same fate
            colors as the star plots.
        show_annotation_legends
            Draw legends for non-contiguous row annotations. This is off by
            default because the fate strip already identifies branch colors.
        sort_by
            In addition to continuous scCS metrics, use
            ``"population_then_ordering"`` to group the annotated root and
            terminal populations while preserving fitted ordering within each
            population.
        """
        self._check_fitted()
        output = self._result if result is None else result
        assert output is not None
        metric_aliases = {
            "future_fate_affinity": "directional_affinity",
            "conditional_fate_affinity": "directional_affinity",
            "future_fate_contribution": "commitment_contribution",
            "reach_adjusted_future_affinity": "commitment_affinity",
        }
        metric = metric_aliases.get(str(metric), str(metric))
        matrices = {
            "directional_affinity": output.directional_affinity,
            "commitment_affinity": output.commitment.commitment_affinity,
            "commitment_contribution": output.commitment_contribution,
        }
        if metric not in matrices:
            valid = sorted(set(matrices) | set(metric_aliases))
            raise ValueError(f"metric must be one of {valid}.")
        if not isinstance(max_cells, int) or max_cells < 2:
            raise ValueError("max_cells must be an integer of at least 2.")

        if row_annotation is None:
            row_annotations: list[str] = []
        elif isinstance(row_annotation, str):
            row_annotations = [row_annotation]
        else:
            row_annotations = [str(value) for value in row_annotation]
            if not row_annotations:
                raise ValueError("row_annotation sequence must not be empty.")
        if len(set(row_annotations)) != len(row_annotations):
            raise ValueError("row_annotation contains duplicate strip names.")

        if mask is not None:
            selected = np.asarray(mask)
            if selected.dtype != bool or selected.ndim != 1 or len(selected) != output.n_cells:
                raise ValueError("mask must be a Boolean array aligned to result cells.")
        elif population == "root":
            selected = output.root_mask.copy()
        elif population == "terminal":
            selected = output.terminal_mask.copy()
        elif population == "all":
            selected = np.ones(output.n_cells, dtype=bool)
        else:
            raise ValueError("population must be 'root', 'terminal', or 'all'.")
        rows = np.flatnonzero(selected)
        if len(rows) == 0:
            raise ValueError("No cells were selected for the heatmap.")

        import matplotlib.pyplot as plt

        def annotation_values(
            annotation: str,
        ) -> tuple[np.ndarray, list[str], dict[str, object], str]:
            annotation = str(annotation)
            color_cycle = plt.get_cmap("tab10").colors
            fate_colors = {
                str(name): color_cycle[(index + 1) % len(color_cycle)]
                for index, name in enumerate(output.fate_names)
            }
            if annotation == "population":
                values = np.full(output.n_cells, "", dtype=object)
                values[output.root_mask] = str(self.root)
                values[output.terminal_mask] = np.asarray(
                    output.embedding.terminal_names[output.terminal_mask],
                    dtype=object,
                )
                categories = [str(self.root), *map(str, output.fate_names)]
                colors = {str(self.root): color_cycle[0], **fate_colors}
                label = "Cell annotation"
            elif annotation == "status":
                values = np.asarray(output.status).astype(str)
                categories = list(dict.fromkeys(values.tolist()))
                colors = {
                    name: color_cycle[index % len(color_cycle)]
                    for index, name in enumerate(categories)
                }
                label = "Commitment status"
            elif annotation in {"dominant_affinity", "dominant_direction"}:
                dominant_indices = np.argmax(output.directional_affinity, axis=1)
                values = np.asarray(output.fate_names, dtype=object)[dominant_indices]
                values = values.astype(object)
                values[~output.projection.velocity_defined] = "undefined"
                categories = list(map(str, output.fate_names))
                if np.any(values == "undefined"):
                    categories.append("undefined")
                colors = {**fate_colors, "undefined": "0.70"}
                label = "Dominant directional affinity"
            elif annotation == "dominant_fate":
                dominant = np.asarray(output.dominant_fate).astype(str)
                values = np.where(
                    np.isin(dominant, output.fate_names),
                    dominant,
                    "not fate-committed",
                )
                categories = list(map(str, output.fate_names))
                if np.any(values == "not fate-committed"):
                    categories.append("not fate-committed")
                colors = {**fate_colors, "not fate-committed": "0.70"}
                label = "Fate-committed direction"
            else:
                raise ValueError(
                    "row_annotation must be 'population', 'status', "
                    "'dominant_affinity', or 'dominant_fate'."
                )
            return values.astype(str), categories, colors, label

        if sort_by in {"pseudotime", "ordering"}:
            ordering_values = self._selected_display_ordering(output)
            rows = rows[np.argsort(np.asarray(ordering_values)[rows], kind="stable")]
        elif sort_by == "population_then_ordering":
            ordering_values = self._selected_display_ordering(output)
            population_values, categories, _, _ = annotation_values("population")
            category_index = {name: index for index, name in enumerate(categories)}
            population_codes = np.asarray(
                [category_index.get(name, len(categories)) for name in population_values],
                dtype=int,
            )
            local_order = np.lexsort(
                (
                    np.asarray(ordering_values, dtype=float)[rows],
                    population_codes[rows],
                )
            )
            rows = rows[local_order]
        elif sort_by == "specific_commitment":
            ordering_values = output.specific_commitment
            rows = rows[np.argsort(np.asarray(ordering_values)[rows], kind="stable")]
        elif sort_by == "commitment_strength":
            ordering_values = output.commitment_strength
            rows = rows[np.argsort(np.asarray(ordering_values)[rows], kind="stable")]
        elif sort_by == "directional_specificity":
            ordering_values = output.directional_specificity
            rows = rows[np.argsort(np.asarray(ordering_values)[rows], kind="stable")]
        else:
            raise ValueError(
                "sort_by must be 'ordering' (or legacy alias 'pseudotime'), "
                "'population_then_ordering', "
                "'specific_commitment', 'commitment_strength', or "
                "'directional_specificity'."
            )
        if len(rows) > max_cells:
            keep = np.linspace(0, len(rows) - 1, max_cells).round().astype(int)
            rows = rows[keep]

        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 6.0))
        image = ax.imshow(matrices[metric][rows], aspect="auto", interpolation="nearest")

        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)

        annotation_axes = []
        for annotation in reversed(row_annotations):
            values, categories, colors, annotation_label = annotation_values(annotation)
            present = [name for name in categories if np.any(values[rows] == name)]
            category_index = {name: index for index, name in enumerate(present)}
            codes = np.asarray([category_index[name] for name in values[rows]], dtype=int)
            annotation_ax = divider.append_axes(
                "left",
                size="4%",
                pad=0.08,
            )
            annotation_ax.imshow(
                codes[:, None],
                aspect="auto",
                interpolation="nearest",
                cmap=ListedColormap([colors[name] for name in present]),
                vmin=-0.5,
                vmax=max(len(present) - 0.5, 0.5),
            )
            annotation_ax.set_xticks([])
            annotation_ax.set_ylim(ax.get_ylim())
            short_titles = {
                "population": "Type",
                "status": "Status",
                "dominant_affinity": "Branch",
                "dominant_direction": "Branch",
                "dominant_fate": "Committed fate",
            }
            annotation_ax.set_title(
                short_titles.get(annotation, annotation_label),
                fontsize=8,
                pad=3,
            )
            annotation_axes.append(annotation_ax)

            displayed_values = values[rows]
            contiguous = True
            centers = []
            for name in present:
                positions = np.flatnonzero(displayed_values == name)
                if len(positions) == 0:
                    continue
                if positions[-1] - positions[0] + 1 != len(positions):
                    contiguous = False
                    break
                centers.append(float(positions.mean()))
            if show_annotation_labels and contiguous:
                annotation_ax.set_yticks(centers)
                annotation_ax.set_yticklabels(present, fontsize=8)
                annotation_ax.tick_params(axis="y", length=0, pad=3)
            else:
                annotation_ax.set_yticks([])
                if show_annotation_labels and show_annotation_legends:
                    handles = [
                        Patch(facecolor=colors[name], edgecolor="none", label=name)
                        for name in present
                    ]
                    annotation_ax.legend(
                        handles=handles,
                        title=annotation_label,
                        frameon=False,
                        loc="upper right",
                        bbox_to_anchor=(-0.15, 1.0),
                    )

        if show_fate_strip:
            color_cycle = plt.get_cmap("tab10").colors
            fate_colors = [color_cycle[(index + 1) % len(color_cycle)] for index in range(output.k)]
            fate_ax = divider.append_axes("top", size="3%", pad=0.05, sharex=ax)
            fate_ax.imshow(
                np.arange(output.k, dtype=int)[None, :],
                aspect="auto",
                interpolation="nearest",
                cmap=ListedColormap(fate_colors),
                vmin=-0.5,
                vmax=max(output.k - 0.5, 0.5),
            )
            fate_ax.set_xticks([])
            fate_ax.set_yticks([])
            fate_ax.set_title("Fate", fontsize=8, loc="left", pad=2)
            fate_ax.tick_params(left=False, bottom=False)

        ax.set_xticks(np.arange(output.k))
        ax.set_xticklabels(output.fate_names, rotation=30, ha="right")
        ax.set_yticks([])
        ax.set_xlabel("Annotated fate")
        row_label = f"Cells ({len(rows)} shown; ordered by {sort_by})"
        if annotation_axes:
            ax.set_ylabel("")
            annotation_axes[-1].set_ylabel(row_label, labelpad=34)
        else:
            ax.set_ylabel(row_label)
        title_pad = 28 if show_fate_strip else 6
        ax.set_title(metric.replace("_", " ").title(), pad=title_pad)
        ax.figure.colorbar(image, ax=ax, label=metric.replace("_", " "))
        return ax.figure

    def plot_subset_comparison(
        self,
        subset_results: Mapping[object, PopulationCommitmentSummary],
        *,
        metric: str = "commitment_composition",
        ax=None,
    ):
        """Compare descriptive population summaries across named subsets."""
        if not subset_results:
            raise ValueError("subset_results is empty.")
        allowed = {
            "total_commitment_mass",
            "mean_commitment_contribution",
            "commitment_composition",
        }
        if metric not in allowed:
            raise ValueError(f"metric must be one of {sorted(allowed)}.")
        names = [str(name) for name in subset_results]
        matrix = np.vstack(
            [
                np.asarray(getattr(summary, metric), dtype=float)
                for summary in subset_results.values()
            ]
        )
        if matrix.shape[1] != len(self.branches):
            raise ValueError("Subset summaries do not match the scorer's fate count.")

        import matplotlib.pyplot as plt

        if ax is None:
            width = max(7.0, 0.9 * len(names) * len(self.branches))
            _, ax = plt.subplots(figsize=(width, 4.8))
        x = np.arange(len(names))
        bar_width = 0.82 / len(self.branches)
        for fate_index, fate in enumerate(self.branches):
            offset = (fate_index - (len(self.branches) - 1) / 2.0) * bar_width
            ax.bar(x + offset, matrix[:, fate_index], bar_width * 0.92, label=fate)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title("scCS descriptive subset comparison")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        if metric == "commitment_composition":
            ax.set_ylim(0.0, 1.0)
        return ax.figure

    def plot_expression_trends(
        self,
        genes: Sequence[str],
        result: Optional[FurcationScoreResult] = None,
        *,
        fate: Optional[str] = None,
        x_axis: str = "ordering",
        population: str = "root",
        layer: Optional[str] = None,
        n_bins: int = 20,
        ncols: int = 3,
        figsize_per_panel: tuple[float, float] = (4.4, 3.5),
    ):
        """Plot binned mean expression along a v0.8 commitment axis."""
        self._check_fitted()
        output = self._result if result is None else result
        assert output is not None
        genes = tuple(str(gene) for gene in genes)
        if not genes:
            raise ValueError("genes must contain at least one gene name.")
        missing = [gene for gene in genes if gene not in self.adata.var_names]
        if missing:
            raise KeyError(f"Genes missing from adata.var_names: {missing!r}.")
        if not isinstance(n_bins, int) or n_bins < 4:
            raise ValueError("n_bins must be an integer of at least 4.")
        if not isinstance(ncols, int) or ncols < 1:
            raise ValueError("ncols must be a positive integer.")

        if population == "root":
            selected = output.root_mask.copy()
        elif population == "all":
            selected = np.ones(output.n_cells, dtype=bool)
        else:
            raise ValueError("population must be 'root' or 'all'.")

        if fate is None:
            fate = output.fate_names[
                int(np.nanargmax(output.root_population_summary.commitment_composition))
            ]
        if fate not in output.fate_names:
            raise ValueError(f"Unknown fate {fate!r}; expected one of {output.fate_names!r}.")
        fate_index = output.fate_names.index(fate)

        x_axis = str(x_axis).lower()
        if x_axis in {"pseudotime", "ordering"}:
            x_values = self._selected_display_ordering(output)
            x_label = output.embedding.ordering_key or "Fitted ordering"
        elif x_axis in {"affinity", "directional_affinity"}:
            x_values = output.directional_affinity[:, fate_index]
            x_label = f"Directional affinity toward {fate}"
        elif x_axis in {"contribution", "commitment_contribution"}:
            x_values = output.commitment_contribution[:, fate_index]
            x_label = f"Commitment contribution toward {fate}"
        elif x_axis == "specific_commitment":
            x_values = output.specific_commitment
            x_label = "Specific commitment"
        else:
            raise ValueError(
                "x_axis must be 'ordering' (or legacy alias 'pseudotime'), "
                "'affinity', 'commitment_contribution', or "
                "'specific_commitment'."
            )

        rows = np.flatnonzero(selected & np.isfinite(x_values))
        if len(rows) < n_bins:
            raise ValueError("Too few finite selected cells for the requested number of bins.")
        x_selected = np.asarray(x_values)[rows]
        edges = np.unique(np.quantile(x_selected, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 4:
            raise ValueError("The selected x-axis has too few distinct values for trend plots.")
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_index = np.digitize(x_selected, edges[1:-1], right=False)

        matrix = self.adata.layers[layer] if layer is not None else self.adata.X
        selected_indices = output.embedding.selected_indices[rows]
        gene_indices = [int(self.adata.var_names.get_loc(gene)) for gene in genes]
        expression = matrix[selected_indices][:, gene_indices]
        try:
            from scipy import sparse

            if sparse.issparse(expression):
                expression = expression.toarray()
        except ImportError:  # pragma: no cover - scipy is a core dependency
            pass
        expression = np.asarray(expression, dtype=float)

        import matplotlib.pyplot as plt

        ncols = min(ncols, len(genes))
        nrows = int(np.ceil(len(genes) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False,
        )
        for gene_index, (gene, axis) in enumerate(zip(genes, axes.ravel())):
            means = np.full(len(centers), np.nan, dtype=float)
            for index in range(len(centers)):
                values = expression[bin_index == index, gene_index]
                finite = values[np.isfinite(values)]
                if len(finite):
                    means[index] = float(np.mean(finite))
            axis.plot(centers, means, marker="o", linewidth=2.0)
            axis.set_title(gene)
            axis.set_xlabel(x_label)
            axis.set_ylabel("Mean expression")
        for axis in axes.ravel()[len(genes) :]:
            axis.set_visible(False)
        fig.suptitle(f"Expression trends ({population}; reference fate: {fate})", y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return fig

    # ------------------------------------------------------------------
    # Candidate commitment-associated genes
    # ------------------------------------------------------------------

    def get_commitment_associated_genes(self, **kwargs):
        """Associate genes with fitted v0.8 commitment outcomes.

        This is a convenience wrapper around
        :func:`scCS.get_commitment_associated_genes`.  Cell-level inference is
        explicitly exploratory; use ``inference_unit="replicate"`` with a
        biological ``replicate_key`` for formal gene-level inference.
        """
        self._check_fitted()
        assert self._result is not None
        from .drivers import get_commitment_associated_genes

        return get_commitment_associated_genes(self.adata, self._result, **kwargs)

    def get_fate_markers(self, **kwargs):
        """Find terminal annotation markers versus the supplied root.

        Fate markers describe annotated cell identity and are distinct from
        commitment-associated genes.
        """
        self._check_fitted()
        assert self._result is not None
        from .drivers import get_fate_markers

        return get_fate_markers(self.adata, self._result, **kwargs)

    # ------------------------------------------------------------------
    # Properties and checks
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self._embedding_result is None:
            return None
        return self._embedding_result.coordinates.copy()

    @property
    def projected_velocity(self) -> Optional[np.ndarray]:
        if self._projection_result is None or self._scoring_mode == "future_fate":
            return None
        return self._projection_result.velocity.copy()

    @property
    def scoring_mode(self) -> str:
        """Currently fitted scoring mode."""
        return self._scoring_mode

    @property
    def result(self) -> Optional[Union[FurcationScoreResult, FutureFateScoreResult]]:
        return self._result

    def _check_instantaneous_mode(self, operation: str) -> None:
        if self._scoring_mode != "instantaneous":
            raise RuntimeError(
                f"{operation} is only defined for scoring_mode='instantaneous'. "
                "future_fate mode intentionally does not transfer a scientific "
                "velocity vector into the star."
            )

    def _check_embedding(self) -> None:
        if self._embedding_result is None:
            raise RuntimeError("Scientific embedding not built. Call build_embedding() first.")

    def _check_fitted(self) -> None:
        if not self._fitted or self._result is None:
            raise RuntimeError("Scorer is not fitted. Call build_embedding() and fit() first.")
