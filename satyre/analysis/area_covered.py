"""
╔══════════════════════════════════════════════════════════╗
║  area_covered « exploration-rate computation »           ║
╚══════════════════════════════════════════════════════════╝

Compute the unique area swept by the fly's body (or by a
disc of equal area) per unit time, following the analytical
method described in the manuscript.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon


def exploration_rate_ellipse(
    head_x: NDArray,
    head_y: NDArray,
    tail_x: NDArray,
    tail_y: NDArray,
    dt: float = 1.0,
) -> NDArray:
    """Compute orientation-dependent exploration rate.

    Superimposes the fly body (modelled as the quadrilateral
    between successive head/tail positions) and sums the
    non-overlapping area per time step.

    Args:
        head_x: Head x-coordinates, shape ``(n_trials, n_steps)``.
        head_y: Head y-coordinates, same shape.
        tail_x: Tail x-coordinates, same shape.
        tail_y: Tail y-coordinates, same shape.
        dt: Time interval between steps (seconds).

    Returns:
        Array of shape ``(n_trials,)`` giving the total area
        covered per trial (mm²).
    """
    n_trials, n_steps = head_x.shape
    area_totals = np.full(n_trials, np.nan)

    for t in range(n_trials):
        trial_area = 0.0
        for s in range(n_steps - 1):
            poly = Polygon(
                [
                    (head_x[t, s], head_y[t, s]),
                    (tail_x[t, s], tail_y[t, s]),
                    (tail_x[t, s + 1], tail_y[t, s + 1]),
                    (head_x[t, s + 1], head_y[t, s + 1]),
                ]
            )
            trial_area += poly.area
        area_totals[t] = trial_area / dt

    return area_totals


def exploration_rate_disc(
    step_sizes: NDArray,
    body_width: float = 2.0,
    dt: float = 1.0,
) -> NDArray:
    """Compute orientation-independent exploration rate.

    Replaces the elliptical fly body with a circle of equal
    surface area, so each step sweeps a rectangle of width
    *body_width* × step length.

    Args:
        step_sizes: Step-size array, shape ``(n_trials, n_steps)``.
        body_width: Diameter of the equivalent disc (mm).
        dt: Time interval between steps (seconds).

    Returns:
        Array of shape ``(n_trials,)`` giving the total area
        covered per trial (mm²).
    """
    return np.nansum(step_sizes * body_width, axis=1) / dt
