#!/usr/bin/env python
"""Execute one complete no-chunk DFFP graph solve at a requested scale."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import scCS


def graph_solve_bytes(n_cells: int, degree: int, n_outcomes: int) -> int:
    csr = n_cells * degree * (8 + 4) + (n_cells + 1) * 8
    probabilities = n_cells * n_outcomes * 8 * 3
    vectors = n_cells * 8 * 5
    anchors = n_cells * n_outcomes
    return int(csr + probabilities + vectors + anchors)

def available_memory() -> int:
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

def current_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9

def build_transition(n_cells: int, degree: int, n_outcomes: int):
    if degree < n_outcomes + 1:
        raise ValueError("degree must be at least n_outcomes + 1")
    row = np.arange(n_cells, dtype=np.int64)
    indices_2d = np.empty((n_cells, degree), dtype=np.int32)
    indices_2d[:, 0] = ((row + 1) % n_cells).astype(np.int32)
    for outcome in range(n_outcomes):
        indices_2d[:, outcome + 1] = np.int32(outcome)
    for column in range(n_outcomes + 1, degree):
        indices_2d[:, column] = ((row + column) % n_cells).astype(np.int32)
    for outcome in range(n_outcomes):
        indices_2d[outcome, :] = np.int32(outcome)
    indices = indices_2d.ravel()
    indptr = np.arange(0, n_cells * degree + 1, degree, dtype=np.int64)
    data = np.full(n_cells * degree, 1.0 / degree, dtype=np.float64)
    return sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=200_000_000)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--outcomes", type=int, default=3)
    parser.add_argument("--effective-horizon", type=float, default=64.0)
    parser.add_argument("--output", type=Path, default=Path("dffp_200m_result.json"))
    parser.add_argument("--memory-headroom", type=float, default=1.20)
    args = parser.parse_args()

    estimate = graph_solve_bytes(args.cells, args.degree, args.outcomes)
    available = available_memory()
    required = int(estimate * args.memory_headroom)
    if available < required:
        raise MemoryError(
            f"Need at least {required / 1e9:.1f} GB free with headroom; "
            f"only {available / 1e9:.1f} GB is available."
        )

    started = time.perf_counter()
    transition = build_transition(args.cells, args.degree, args.outcomes)
    anchors = np.zeros((args.cells, args.outcomes), dtype=bool)
    for outcome in range(args.outcomes):
        anchors[outcome, outcome] = True
    solution = scCS.solve_discounted_outcomes(
        transition,
        anchors,
        tuple(f"fate_{i + 1}" for i in range(args.outcomes)),
        effective_horizon=args.effective_horizon,
        solver="iterative",
        tolerance=1e-6,
        max_iter=500,
    )
    elapsed = time.perf_counter() - started
    checksum = float(solution.probability.mean() + solution.unresolved_probability.mean())
    result = {
        "scCS_version": scCS.__version__,
        "cells": args.cells,
        "degree": args.degree,
        "outcomes": args.outcomes,
        "effective_horizon": args.effective_horizon,
        "scientific_scoring_chunked": False,
        "elapsed_seconds": elapsed,
        "cells_per_second": args.cells / elapsed,
        "estimated_bytes": estimate,
        "peak_rss_gb": current_rss_gb(),
        "iterations": solution.iterations,
        "converged": solution.converged,
        "residual": solution.residual,
        "checksum": checksum,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
