"""Permutation-equivariant star geometry for scCS v0.8.

The scientific star contains one incoming root axis and ``k`` symmetric
outgoing terminal directions.  Terminal directions form a regular simplex in
the subspace orthogonal to the root axis.  A separate 2D radial layout may be
used for visualization, but must never define scientific scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


def regular_simplex_directions(k: int) -> np.ndarray:
    """Return ``k`` unit regular-simplex directions in ``R^k``.

    The rows have unit norm, sum to zero, and have pairwise dot product
    ``-1 / (k - 1)``.
    """
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer greater than or equal to 2.")
    identity = np.eye(k, dtype=float)
    centered = identity - np.ones((k, k), dtype=float) / k
    return np.sqrt(k / (k - 1.0)) * centered


@dataclass(frozen=True)
class SimplexStarGeometry:
    """Scientific root-plus-simplex star geometry."""

    fate_names: Tuple[str, ...]
    root_direction: np.ndarray
    terminal_directions: np.ndarray

    def __init__(self, fate_names: Sequence[str]) -> None:
        names = tuple(str(name) for name in fate_names)
        if len(names) < 2:
            raise ValueError("At least two terminal fates are required.")
        if any(not name for name in names):
            raise ValueError("Fate names must be non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate fate names: {names!r}.")

        k = len(names)
        root_direction = np.ones(k, dtype=float) / np.sqrt(k)
        terminal_directions = regular_simplex_directions(k)

        object.__setattr__(self, "fate_names", names)
        object.__setattr__(self, "root_direction", root_direction)
        object.__setattr__(self, "terminal_directions", terminal_directions)

        self._validate_internal_geometry()

    @property
    def k(self) -> int:
        return len(self.fate_names)

    @property
    def dimension(self) -> int:
        return self.k

    @property
    def incoming_root_direction(self) -> np.ndarray:
        return -self.root_direction

    def _validate_internal_geometry(self, atol: float = 1e-12) -> None:
        directions = self.terminal_directions
        gram = directions @ directions.T
        expected = np.full((self.k, self.k), -1.0 / (self.k - 1))
        np.fill_diagonal(expected, 1.0)
        if not np.allclose(gram, expected, atol=atol, rtol=0.0):
            raise RuntimeError("Invalid regular-simplex terminal geometry.")
        if not np.allclose(
            directions @ self.root_direction,
            0.0,
            atol=atol,
            rtol=0.0,
        ):
            raise RuntimeError("Terminal directions are not orthogonal to root axis.")

    def direction_for(self, fate_name: str) -> np.ndarray:
        try:
            index = self.fate_names.index(str(fate_name))
        except ValueError as exc:
            raise KeyError(
                f"Unknown fate {fate_name!r}; expected one of {self.fate_names!r}."
            ) from exc
        return self.terminal_directions[index].copy()

    def direction_map(self) -> Dict[str, np.ndarray]:
        return {
            name: self.terminal_directions[index].copy()
            for index, name in enumerate(self.fate_names)
        }

    @staticmethod
    def _validate_progress(progress: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(progress, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1].")
        return values

    def root_coordinates(
        self,
        progress: np.ndarray,
        *,
        arm_scale: float = 1.0,
    ) -> np.ndarray:
        """Place root cells on the incoming arm.

        ``progress=0`` is farthest before the furcation and ``progress=1`` is
        at the furcation point.
        """
        values = self._validate_progress(progress, "root progress")
        if not np.isfinite(arm_scale) or arm_scale <= 0:
            raise ValueError("arm_scale must be positive and finite.")
        radius = arm_scale * (1.0 - values)
        return radius[:, None] * self.incoming_root_direction[None, :]

    def terminal_coordinates(
        self,
        fate_labels: Sequence[str],
        *,
        arm_scale: float = 1.0,
    ) -> np.ndarray:
        """Place terminal cells at equal-radius annotated fate vertices.

        Terminal populations are supervised endpoint anchors.  Their
        scientific radius is therefore fixed and does not depend on terminal
        pseudotime, population abundance, or branch-list order.
        """
        labels = np.asarray(fate_labels).astype(str)
        if labels.ndim != 1:
            raise ValueError("fate_labels must be one-dimensional.")
        if not np.isfinite(arm_scale) or arm_scale <= 0:
            raise ValueError("arm_scale must be positive and finite.")

        coordinates = np.empty((len(labels), self.dimension), dtype=float)
        for index, label in enumerate(labels):
            coordinates[index] = arm_scale * self.direction_for(label)
        return coordinates

    def decompose_velocity(
        self,
        velocity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Separate root-axis progression from fate-directed branch velocity.

        Returns
        -------
        progression
            Signed scalar component along the outgoing root-axis direction.
        branch_velocity
            Component orthogonal to the root axis.
        """
        vectors = np.asarray(velocity, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"velocity must have shape (n_cells, {self.dimension}).")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("velocity contains non-finite values.")
        progression = vectors @ self.root_direction
        branch = vectors - progression[:, None] * self.root_direction[None, :]
        return progression, branch
