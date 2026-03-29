"""
╔══════════════════════════════════════════════════════════╗
║  movement_types « stationary / translational / rot »    ║
╚══════════════════════════════════════════════════════════╝

Classify each prototypical-movement step into stationary,
translational, or rotational based on velocity thresholds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def classify_movements(
    performed_pms: NDArray,
    velocity_array: NDArray,
    *,
    trans_threshold: float = 0.5,
    rot_threshold: float = 1.0,
) -> dict[str, NDArray | list[float]]:
    """Classify each PM step as stationary, translational,
    or rotational.

    Args:
        performed_pms: 1-D integer array of PM indices visited
            during a trial.
        velocity_array: ``(N, 3)`` velocity look-up table
            ``[thrust, slip, yaw]``.
        trans_threshold: Combined thrust + slip speed below
            which the step is considered stationary (mm s⁻¹).
        rot_threshold: Yaw speed above which the step is
            rotational (rad s⁻¹).

    Returns:
        Dictionary with keys:

        - ``'labels'`` — list of strings (``'stationary'``,
          ``'translational'``, ``'rotational'``).
        - ``'proportions'`` — ``[% stationary, % translational,
          % rotational]``.
    """
    labels: list[str] = []
    for pm in performed_pms:
        velo = velocity_array[int(pm)]
        trans_speed = abs(velo[0]) + abs(velo[1])
        rot_speed = abs(velo[2])
        if trans_speed <= trans_threshold and rot_speed <= rot_threshold:
            labels.append("stationary")
        elif rot_speed > rot_threshold:
            labels.append("rotational")
        else:
            labels.append("translational")

    n = len(labels)
    proportions = [
        labels.count("stationary") / n * 100,
        labels.count("translational") / n * 100,
        labels.count("rotational") / n * 100,
    ]
    return {"labels": labels, "proportions": proportions}
