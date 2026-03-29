"""
╔══════════════════════════════════════════════════════════╗
║  preprocessing « build Markov transition matrices »      ║
╚══════════════════════════════════════════════════════════╝

Converts raw experimental data (prototypical-movement index
sequences and velocity tables) into the cumulative transition-
probability matrices consumed by :class:`~satyre.simulation.MarkovWalker`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def build_transition_matrices(
    idx_path: str | Path,
    velo_path: str | Path,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build Markov transition matrices from raw text files.

    Args:
        idx_path: Path to a text file containing one PM
            index per line (the recorded sequence of
            prototypical movements).
        velo_path: Path to a CSV-like text file where each
            line holds ``thrust, slip, yaw`` velocities for
            one PM.

    Returns:
        A tuple ``(cumsum_matrix, pm_index, velocity_array)``
        where:

        - **cumsum_matrix** — cumulative transition-probability
          matrix of shape ``(N+1, N+1)`` (row 0 / col 0 are
          header indices).
        - **pm_index** — companion index matrix of same shape
          mapping columns back to PM identifiers.
        - **velocity_array** — ``(N+1, 3)`` array of
          ``[thrust, slip, yaw]`` velocities.

    Example:
        >>> from satyre.utils import build_transition_matrices
        >>> trans, idx, velos = build_transition_matrices(
        ...     'data/ORL_IDX.txt', 'data/ORL_C.txt'
        ... )
    """
    idx_path = Path(idx_path)
    velo_path = Path(velo_path)

    # ── load PM index sequence ──────────────────────────
    idx_arr = np.loadtxt(idx_path)
    n_pm = int(idx_arr.max())

    # ── load velocity table ─────────────────────────────
    velo_raw = np.loadtxt(velo_path, delimiter=",")
    velocity_array = np.full((n_pm + 1, 3), np.nan)
    velocity_array[1 : len(velo_raw) + 1, :] = velo_raw

    # ── build frequency matrix ──────────────────────────
    freq = np.zeros((n_pm + 1, n_pm + 1))
    for i in range(len(idx_arr) - 1):
        src = int(idx_arr[i])
        tgt = int(idx_arr[i + 1])
        freq[src, tgt] += 1

    # ── frequency → probability ─────────────────────────
    prob = np.full_like(freq, np.nan)
    for row in range(1, n_pm + 1):
        row_sum = freq[row, 1:].sum()
        if row_sum > 0:
            prob[row, 1:] = freq[row, 1:] / row_sum

    # ── sort by probability & build cumsum ──────────────
    sorted_prob = np.full_like(prob, np.nan)
    pm_index = np.full_like(prob, np.nan)
    cumsum = np.full_like(prob, np.nan)

    for row in range(1, n_pm + 1):
        order = np.argsort(prob[row, 1:])
        sorted_prob[row, 1:] = prob[row, 1:][order]
        pm_index[row, 1:] = order + 1
        cumsum[row, 1:] = np.cumsum(sorted_prob[row, 1:])

    return cumsum, pm_index, velocity_array


def main() -> None:
    """CLI entry point for ``satyre-preprocess``.

    Usage::

        satyre-preprocess --idx data/ORL_IDX.txt \\
                          --velo data/ORL_C.txt   \\
                          --out  data/preprocessed
    """
    parser = argparse.ArgumentParser(
        description="Build Markov transition matrices from raw PM data."
    )
    parser.add_argument(
        "--idx",
        type=Path,
        required=True,
        help="Path to the PM index sequence (.txt).",
    )
    parser.add_argument(
        "--velo",
        type=Path,
        required=True,
        help="Path to the PM velocity table (.txt / .csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("preprocessed"),
        help="Output directory for .npy matrices (default: ./preprocessed).",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cumsum, pm_index, velocities = build_transition_matrices(
        args.idx, args.velo
    )
    np.save(args.out / "cumsum_transition_matrix.npy", cumsum)
    np.save(args.out / "pm_index_matrix.npy", pm_index)
    np.save(args.out / "velocity_array.npy", velocities)
    print(f"Saved matrices to {args.out.resolve()}")


if __name__ == "__main__":
    main()
