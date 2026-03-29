"""
╔══════════════════════════════════════════════════════════╗
║  step_size « step-length distribution analysis »        ║
╚══════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def step_size_distribution(
    step_sizes: NDArray,
) -> dict[str, NDArray | float]:
    """Compute descriptive statistics of step-size arrays.

    Args:
        step_sizes: Array of shape ``(n_trials, n_steps)``
            containing Euclidean step lengths.

    Returns:
        Dictionary with ``'mean_per_trial'``, ``'median_per_trial'``,
        ``'grand_mean'``, ``'grand_median'``, and the flattened
        ``'all_steps'`` (NaN-stripped).
    """
    means = np.nanmean(step_sizes, axis=1)
    medians = np.nanmedian(step_sizes, axis=1)
    flat = step_sizes.ravel()
    flat = flat[~np.isnan(flat)]
    return {
        "mean_per_trial": means,
        "median_per_trial": medians,
        "grand_mean": float(np.nanmean(flat)),
        "grand_median": float(np.nanmedian(flat)),
        "all_steps": flat,
    }
