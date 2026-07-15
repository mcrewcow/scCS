"""Root-arm ordering transform for the supervised v0.8 furcation star.

The user supplies one annotated root population and two or more annotated
terminal populations.  Annotation determines which side of the furcation each
cell occupies.  The ordering metric is therefore used only to arrange root
cells along the incoming arm; terminal populations are fixed scientific
anchors at equal-radius simplex vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .furcation import Furcation


def _as_1d(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array


def _validate_quantile_pair(low: float, high: float, *, name: str) -> None:
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f"{name} quantiles must satisfy 0 <= low < high <= 1; found low={low}, high={high}."
        )


def _safe_bounds(values: np.ndarray, low_q: float, high_q: float) -> tuple[float, float]:
    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Ordering quantiles are not finite.")
    if high <= low + np.finfo(float).eps:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low + np.finfo(float).eps:
        midpoint = float(values[0])
        return midpoint, midpoint
    return low, high


def _scale_to_unit(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low + np.finfo(float).eps:
        # A constant metric contains no radial information.  Use one common
        # mid-arm location rather than manufacturing an ordering from row order.
        return np.full(len(values), 0.5, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


@dataclass(frozen=True)
class OrderingDiagnostics:
    """Diagnostics for root-only radial scaling."""

    root_lower_bound: float
    root_upper_bound: float
    root_clipped_low_fraction: float
    root_clipped_high_fraction: float
    root_progress_min: float
    root_progress_max: float
    root_progress_mean: float
    root_progress_sd: float
    root_unique_values: int
    root_unique_fraction: float
    root_largest_tie_fraction: float


@dataclass(frozen=True)
class FurcationOrderingResult:
    """Root-arm progress and annotation-derived population masks."""

    root_progress: np.ndarray
    terminal_names: np.ndarray
    root_mask: np.ndarray
    terminal_mask: np.ndarray
    diagnostics: OrderingDiagnostics


@dataclass
class FurcationOrderingScaler:
    """Map an ordering metric onto the incoming annotated root arm.

    Parameters
    ----------
    root_lower_quantile, root_upper_quantile
        Robust bounds for root radial placement.  Earlier root cells map
        farther from the furcation; later root cells map closer to the origin.
    higher_is_later
        Whether larger ordering values indicate later developmental progress.

    Notes
    -----
    Terminal ordering values are intentionally not fitted or used for the
    scientific geometry.  Every manually annotated terminal population is an
    equal-radius endpoint anchor.  Terminal ordering is used only by the
    display layer to spread overplotted cells along a 2D ray.
    """

    root_lower_quantile: float = 0.05
    root_upper_quantile: float = 0.95
    higher_is_later: bool = True

    root_lower_: Optional[float] = None
    root_upper_: Optional[float] = None
    fitted_: bool = False

    def __post_init__(self) -> None:
        _validate_quantile_pair(
            self.root_lower_quantile,
            self.root_upper_quantile,
            name="root",
        )

    @property
    def fitted(self) -> bool:
        return self.fitted_

    def _orient(self, values: np.ndarray) -> np.ndarray:
        return values if self.higher_is_later else -values

    def fit(
        self,
        ordering: Sequence[float],
        labels: Sequence[object],
        furcation: Furcation,
    ) -> "FurcationOrderingScaler":
        values = _as_1d(ordering, name="ordering")
        label_values = np.asarray(labels).astype(str)
        if label_values.ndim != 1 or len(label_values) != len(values):
            raise ValueError("labels must be one-dimensional and match ordering.")

        validation = furcation.validate_labels(label_values)
        selected = validation.selected_mask
        selected_values_raw = values[selected]
        if not np.all(np.isfinite(selected_values_raw)):
            raise ValueError("ordering contains non-finite values among selected furcation cells.")
        selected_values = self._orient(selected_values_raw)
        selected_labels = label_values[selected]
        root_mask = furcation.masks(selected_labels)["root"]
        root_values = selected_values[root_mask]

        self.root_lower_, self.root_upper_ = _safe_bounds(
            root_values,
            self.root_lower_quantile,
            self.root_upper_quantile,
        )
        self.fitted_ = True
        return self

    def transform(
        self,
        ordering: Sequence[float],
        labels: Sequence[object],
        furcation: Furcation,
    ) -> FurcationOrderingResult:
        if not self.fitted_:
            raise RuntimeError("FurcationOrderingScaler must be fit before transform().")

        values = _as_1d(ordering, name="ordering")
        label_values = np.asarray(labels).astype(str)
        if label_values.ndim != 1 or len(label_values) != len(values):
            raise ValueError("labels must be one-dimensional and match ordering.")

        validation = furcation.validate_labels(label_values)
        selected = validation.selected_mask
        selected_values_raw = values[selected]
        if not np.all(np.isfinite(selected_values_raw)):
            raise ValueError("ordering contains non-finite values among selected furcation cells.")
        selected_values = self._orient(selected_values_raw)
        selected_labels = label_values[selected]
        masks = furcation.masks(selected_labels)
        root_mask = masks["root"]
        terminal_mask = ~root_mask

        assert self.root_lower_ is not None and self.root_upper_ is not None
        root_values = selected_values[root_mask]

        root_progress = np.zeros(len(selected_values), dtype=float)
        root_progress[root_mask] = _scale_to_unit(
            root_values,
            self.root_lower_,
            self.root_upper_,
        )

        terminal_names = np.full(len(selected_values), "", dtype=object)
        for name, group_labels in furcation.terminal_groups:
            group_mask = np.isin(selected_labels, group_labels)
            terminal_names[group_mask] = name
        if np.any(terminal_mask & (terminal_names == "")):
            raise RuntimeError("Failed to assign a terminal name to selected cells.")

        root_scaled = root_progress[root_mask]
        unique_values, unique_counts = np.unique(root_values, return_counts=True)
        diagnostics = OrderingDiagnostics(
            root_lower_bound=float(self.root_lower_),
            root_upper_bound=float(self.root_upper_),
            root_clipped_low_fraction=float(np.mean(root_values <= self.root_lower_)),
            root_clipped_high_fraction=float(np.mean(root_values >= self.root_upper_)),
            root_progress_min=float(root_scaled.min()),
            root_progress_max=float(root_scaled.max()),
            root_progress_mean=float(root_scaled.mean()),
            root_progress_sd=float(root_scaled.std(ddof=0)),
            root_unique_values=int(len(unique_values)),
            root_unique_fraction=float(len(unique_values) / len(root_values)),
            root_largest_tie_fraction=float(unique_counts.max() / len(root_values)),
        )
        return FurcationOrderingResult(
            root_progress=root_progress,
            terminal_names=terminal_names,
            root_mask=root_mask,
            terminal_mask=terminal_mask,
            diagnostics=diagnostics,
        )

    def fit_transform(
        self,
        ordering: Sequence[float],
        labels: Sequence[object],
        furcation: Furcation,
    ) -> FurcationOrderingResult:
        return self.fit(ordering, labels, furcation).transform(
            ordering,
            labels,
            furcation,
        )
