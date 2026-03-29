"""
╔══════════════════════════════════════════════════════════╗
║  food_found « foraging-efficiency summary »              ║
╚══════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def summarise_food_collected(
    results: dict[str, NDArray],
) -> dict[str, float]:
    """Compute summary statistics for food-collection data.

    Args:
        results: Dictionary returned by
            :meth:`~satyre.simulation.MarkovWalker.simulate_multiple`
            (must contain key ``'food_found'``).

    Returns:
        Dictionary with keys ``'mean'``, ``'median'``,
        ``'std'``, ``'ci_lower'``, ``'ci_upper'`` (95 %
        bootstrap CI), and ``'n'``.
    """
    ff = results["food_found"]
    ff = ff[~np.isnan(ff)]
    n = len(ff)
    mean = float(np.mean(ff))
    ci = stats.bootstrap(
        (ff,), np.mean, confidence_level=0.95, n_resamples=9_999
    )
    return {
        "mean": mean,
        "median": float(np.median(ff)),
        "std": float(np.std(ff, ddof=1)) if n > 1 else 0.0,
        "ci_lower": float(ci.confidence_interval.low),
        "ci_upper": float(ci.confidence_interval.high),
        "n": n,
    }
