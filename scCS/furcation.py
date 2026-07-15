"""Annotation-defined furcations for scCS v0.8.

A :class:`Furcation` is deliberately simple and supervised: one manually
annotated root population gives rise to two or more manually annotated
terminal populations.  scCS does not infer intermediate states, topology, or
terminal identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple, Union

import numpy as np

LabelSpec = Union[str, Sequence[str]]
TerminalSpec = Union[Sequence[str], Mapping[str, LabelSpec]]


def _normalize_labels(value: LabelSpec, *, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        labels = (value,)
    else:
        labels = tuple(str(item) for item in value)

    labels = tuple(label for label in labels if label != "")
    if not labels:
        raise ValueError(f"{field_name} must contain at least one non-empty label.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{field_name} contains duplicate labels: {labels!r}.")
    return labels


@dataclass(frozen=True)
class FurcationValidation:
    """Validation result for a furcation against an annotation vector."""

    root_count: int
    terminal_counts: Dict[str, int]
    selected_count: int
    selected_mask: np.ndarray


@dataclass(frozen=True, init=False)
class Furcation:
    """A manually annotated root-to-terminal furcation.

    Parameters
    ----------
    obs_key
        Column containing annotation labels.
    root
        One label, or a group of labels representing one biological root
        population.
    terminals
        Either an ordered sequence of terminal labels, or an ordered mapping
        from biological terminal name to one or more annotation labels.
    min_cells
        Minimum number of cells required in the root and in every terminal
        population during validation.

    Notes
    -----
    Grouping multiple annotation labels under one root or terminal is only a
    naming convenience.  It does not introduce intermediate-state modelling.
    """

    obs_key: str
    root_labels: Tuple[str, ...]
    terminal_groups: Tuple[Tuple[str, Tuple[str, ...]], ...]
    min_cells: int

    def __init__(
        self,
        *,
        obs_key: str,
        root: LabelSpec,
        terminals: TerminalSpec,
        min_cells: int = 2,
    ) -> None:
        if not isinstance(obs_key, str) or not obs_key:
            raise ValueError("obs_key must be a non-empty string.")
        if not isinstance(min_cells, int) or min_cells < 1:
            raise ValueError("min_cells must be a positive integer.")

        root_labels = _normalize_labels(root, field_name="root")

        if isinstance(terminals, Mapping):
            items = tuple(terminals.items())
        else:
            if isinstance(terminals, str):
                raise TypeError(
                    "terminals must be a sequence of labels or a mapping, not a string."
                )
            items = tuple((str(label), str(label)) for label in terminals)

        if len(items) < 2:
            raise ValueError("A furcation requires at least two terminal populations.")

        terminal_groups = []
        terminal_names = []
        terminal_labels_flat = []
        for name, labels in items:
            name = str(name)
            if not name:
                raise ValueError("Terminal names must be non-empty strings.")
            normalized = _normalize_labels(labels, field_name=f"terminal {name!r}")
            terminal_names.append(name)
            terminal_labels_flat.extend(normalized)
            terminal_groups.append((name, normalized))

        if len(set(terminal_names)) != len(terminal_names):
            raise ValueError(f"Duplicate terminal names: {terminal_names!r}.")
        if len(set(terminal_labels_flat)) != len(terminal_labels_flat):
            raise ValueError(
                "Annotation labels cannot belong to more than one terminal population."
            )

        overlap = set(root_labels).intersection(terminal_labels_flat)
        if overlap:
            raise ValueError(
                f"Root and terminal annotations must be disjoint; overlap: {sorted(overlap)!r}."
            )

        object.__setattr__(self, "obs_key", obs_key)
        object.__setattr__(self, "root_labels", root_labels)
        object.__setattr__(self, "terminal_groups", tuple(terminal_groups))
        object.__setattr__(self, "min_cells", min_cells)

    @property
    def root_name(self) -> str:
        """Human-readable name of the grouped root population."""
        return self.root_labels[0] if len(self.root_labels) == 1 else "root"

    @property
    def terminal_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.terminal_groups)

    @property
    def terminal_labels(self) -> Dict[str, Tuple[str, ...]]:
        return {name: labels for name, labels in self.terminal_groups}

    @property
    def k(self) -> int:
        return len(self.terminal_groups)

    @property
    def all_annotation_labels(self) -> Tuple[str, ...]:
        labels = list(self.root_labels)
        for _, group in self.terminal_groups:
            labels.extend(group)
        return tuple(labels)

    def validate_labels(self, labels: Sequence[object]) -> FurcationValidation:
        """Validate the furcation against a one-dimensional annotation vector."""
        values = np.asarray(labels).astype(str)
        if values.ndim != 1:
            raise ValueError("Annotation labels must be one-dimensional.")

        root_mask = np.isin(values, self.root_labels)
        root_count = int(root_mask.sum())
        if root_count < self.min_cells:
            raise ValueError(
                f"Root population has {root_count} cells; at least "
                f"{self.min_cells} are required. Root labels: {self.root_labels!r}."
            )

        selected_mask = root_mask.copy()
        terminal_counts: Dict[str, int] = {}
        for name, group in self.terminal_groups:
            mask = np.isin(values, group)
            count = int(mask.sum())
            if count < self.min_cells:
                raise ValueError(
                    f"Terminal population {name!r} has {count} cells; at least "
                    f"{self.min_cells} are required. Labels: {group!r}."
                )
            terminal_counts[name] = count
            selected_mask |= mask

        return FurcationValidation(
            root_count=root_count,
            terminal_counts=terminal_counts,
            selected_count=int(selected_mask.sum()),
            selected_mask=selected_mask,
        )

    def validate_adata(self, adata) -> FurcationValidation:
        """Validate against an AnnData-like object with an ``obs`` table."""
        if not hasattr(adata, "obs"):
            raise TypeError("adata must expose an .obs table.")
        if self.obs_key not in adata.obs:
            raise KeyError(f"Annotation column {self.obs_key!r} is missing from adata.obs.")
        return self.validate_labels(adata.obs[self.obs_key].to_numpy())

    def masks(self, labels: Sequence[object]) -> Dict[str, np.ndarray]:
        """Return Boolean masks for the root and each terminal population."""
        values = np.asarray(labels).astype(str)
        self.validate_labels(values)
        result: Dict[str, np.ndarray] = {"root": np.isin(values, self.root_labels)}
        for name, group in self.terminal_groups:
            result[name] = np.isin(values, group)
        return result

    def summary(self) -> str:
        terminal_text = ", ".join(
            f"{name}={list(labels)!r}" for name, labels in self.terminal_groups
        )
        return (
            f"Furcation(obs_key={self.obs_key!r}, root={list(self.root_labels)!r}, "
            f"terminals={{ {terminal_text} }}, k={self.k})"
        )
