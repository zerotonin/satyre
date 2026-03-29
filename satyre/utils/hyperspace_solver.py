"""
╔══════════════════════════════════════════════════════════╗
║  hyperspace_solver « toroidal boundary wrap »            ║
╚══════════════════════════════════════════════════════════╝

Implements a line-segment intersection solver for a square
toroidal arena.  When an agent's displacement vector exits
one side, it re-enters from the opposite side, preserving
direction and remaining distance.

Based on the vector-intersection method described in
*Computer Graphics* by F. S. Hill.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString


class HyperspaceSolver:
    """Resolve boundary crossings in a square toroidal arena.

    Given a displacement vector that may extend beyond the
    arena borders, the solver decomposes it into a sequence
    of sub-steps that wrap around the edges.

    Args:
        vector: A ``(2, 2)`` array where ``vector[0]`` is
            the start position and ``vector[1]`` is the
            (possibly out-of-bounds) end position.
        border: Half-width of the square arena.  The arena
            spans ``[-border, border]`` on both axes.

    Example:
        >>> import numpy as np
        >>> from satyre.utils import HyperspaceSolver
        >>> vec = np.array([[-9000., -9000.], [-14000., -16000.]])
        >>> solver = HyperspaceSolver(vector=vec, border=10000)
        >>> sub_steps = solver.solve()
        >>> sub_steps[-1]  # final in-bounds position
        array([...])
    """

    def __init__(
        self,
        vector: NDArray[np.floating] = None,
        border: float = 10_000.0,
    ) -> None:
        if vector is None:
            vector = np.array(
                [[-9000.0, -9000.0], [-14000.0, -16000.0]]
            )
        self.border = border
        self.vector = vector.copy()
        self.sub_steps: list[NDArray] = []

        # ── arena boundary segments ─────────────────────
        b = border
        self._left = np.array([[-b, -b], [-b, b]])
        self._right = np.array([[b, -b], [b, b]])
        self._top = np.array([[-b, b], [b, b]])
        self._bottom = np.array([[-b, -b], [b, -b]])

        # ── direction preserved across wraps ────────────
        rel = self.vector[1] - self.vector[0]
        self._theta = np.arctan2(rel[0], rel[1])
        ct, st = np.cos(-self._theta), np.sin(-self._theta)
        self._rot = np.array([[ct, -st], [st, ct]])
        self._vec_norm = float(np.linalg.norm(rel))

    # ── public interface ────────────────────────────────

    def solve(self) -> NDArray:
        """Decompose the displacement into wrapped sub-steps.

        Returns:
            Array of shape ``(2*K, 2)`` where rows alternate
            between entry points and exit / final points for
            each sub-step.  The last row is the agent's final
            in-bounds position.
        """
        intersection, entry = self._find_exit()

        # no crossing — vector stays inside the arena
        if np.isnan(intersection[0]):
            return np.vstack((self.vector[0], self.vector[1]))

        self.sub_steps = [self.vector[0], np.array(intersection)]
        self._update_norm()
        self._rebuild_vector(np.array(entry))

        while not np.isnan(intersection[0]):
            self.sub_steps.append(self.vector[0].copy())
            intersection, entry = self._find_exit()
            if np.isnan(intersection[0]):
                self.sub_steps.append(self.vector[1].copy())
            else:
                self.sub_steps.append(np.array(intersection))
                self._update_norm()
                self._rebuild_vector(np.array(entry))

        return np.vstack(self.sub_steps)

    # ── segment intersection ────────────────────────────

    def _seg_intersect(
        self, seg_a: NDArray, seg_b: NDArray
    ) -> tuple[float, float]:
        """Find the intersection of two line segments.

        Args:
            seg_a: First segment ``(2, 2)``.
            seg_b: Second segment ``(2, 2)``.

        Returns:
            ``(x, y)`` intersection point, or ``(nan, nan)``
            if the segments do not intersect or if the
            intersection coincides with the vector origin.
        """
        line1 = LineString(
            [(seg_a[0, 0], seg_a[0, 1]), (seg_a[1, 0], seg_a[1, 1])]
        )
        line2 = LineString(
            [(seg_b[0, 0], seg_b[0, 1]), (seg_b[1, 0], seg_b[1, 1])]
        )
        result = line1.intersection(line2)
        if result.is_empty:
            return (np.nan, np.nan)
        # Point intersection
        if result.geom_type == "Point":
            ix, iy = result.x, result.y
        else:
            try:
                ix = result.coords.xy[0][0]
                iy = result.coords.xy[1][0]
            except (IndexError, NotImplementedError):
                return (np.nan, np.nan)
        # reject if intersection is at the start of vector
        if ix == self.vector[0, 0] and iy == self.vector[0, 1]:
            return (np.nan, np.nan)
        return (ix, iy)

    # ── find which wall is crossed ──────────────────────

    def _find_exit(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Determine which arena wall the vector crosses and
        compute the corresponding re-entry point.

        Returns:
            Tuple of ``(intersection, entry_point)`` where
            *intersection* is the wall crossing point and
            *entry_point* is the mirrored re-entry position.
            Both are ``(nan, nan)`` if no crossing occurs.
        """
        b = self.border
        intersection: tuple[float, float] = (np.nan, np.nan)
        entry: tuple[float, float] = (np.nan, np.nan)

        # left wall
        ix = self._seg_intersect(self.vector, self._left)
        if not np.isnan(ix[0]):
            intersection = ix
            if self._is_corner(ix):
                entry = (-ix[0], -ix[1])
            else:
                entry = (-ix[0], ix[1])
            return intersection, entry

        # right wall
        ix = self._seg_intersect(self.vector, self._right)
        if not np.isnan(ix[0]):
            intersection = ix
            if self._is_corner(ix):
                entry = (-ix[0], -ix[1])
            else:
                entry = (-ix[0], ix[1])
            return intersection, entry

        # top wall
        ix = self._seg_intersect(self.vector, self._top)
        if not np.isnan(ix[0]):
            intersection = ix
            entry = (ix[0], -ix[1])
            return intersection, entry

        # bottom wall
        ix = self._seg_intersect(self.vector, self._bottom)
        if not np.isnan(ix[0]):
            intersection = ix
            entry = (ix[0], -ix[1])
            return intersection, entry

        return intersection, entry

    def _is_corner(self, point: tuple[float, float]) -> bool:
        """Check whether *point* lies on a corner of the arena.

        Args:
            point: ``(x, y)`` coordinates.

        Returns:
            ``True`` if the point is at a corner.
        """
        b = self.border
        return (abs(point[0]) == b) and (abs(point[1]) == b)

    # ── vector reconstruction ───────────────────────────

    def _rebuild_vector(self, entry: NDArray) -> None:
        """Reconstruct the displacement vector from the
        re-entry point, preserving direction and remaining
        distance.

        Args:
            entry: Re-entry ``(2,)`` position.
        """
        new_end = self._rot @ np.array([0.0, self._vec_norm])
        self.vector = np.vstack((entry, new_end + entry))

    def _update_norm(self) -> None:
        """Subtract the last sub-step length from the
        remaining vector norm."""
        rel = self.sub_steps[-2] - self.sub_steps[-1]
        self._vec_norm -= float(np.linalg.norm(rel))
