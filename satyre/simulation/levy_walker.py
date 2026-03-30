"""
╔══════════════════════════════════════════════════════════╗
║  levy_walker « Lévy-flight / random-walk simulator »     ║
╚══════════════════════════════════════════════════════════╝

Implements Lévy-flight and uniform random-walk foraging
agents that move through a toroidal 2-D arena collecting
food items by tactile body-overlap.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from satyre.utils.hyperspace_solver import HyperspaceSolver


# ──────────────────────────────────────────────────────────
#  LevyWalker
# ──────────────────────────────────────────────────────────
class LevyWalker:
    """Simulate Lévy-flight or random-walk foraging in a
    toroidal arena.

    The agent draws step sizes from either a Cauchy-derived
    heavy-tailed distribution (Lévy flight) or a Gaussian
    (uniform random walk), chooses a uniformly random
    direction, and checks for food overlap after each step.

    Args:
        cauchy_alpha: Shape parameter for the Cauchy step-size
            distribution.  Higher values produce shorter tails.
            Ignored when *mode* is ``'uniform'``.
        n_trials: Number of independent simulation trials
            executed by :meth:`simulate_multiple`.
        border: Half-width of the square toroidal arena (mm).
        mode: Step-size distribution — ``'cauchy'`` for Lévy
            flight or ``'uniform'`` for Gaussian random walk.
        max_steps: Maximum number of steps per trial.
        total_food: Number of food items placed at the start
            of each trial.
        food_mode: Food distribution — ``'random'`` or
            ``'clustered'``.

    Attributes:
        food_found_total: Items collected in the most recent
            single trial.
        step_size_arr: Step-size array from the most recent
            trial.

    Example:
        >>> from satyre import LevyWalker
        >>> walker = LevyWalker(cauchy_alpha=1.5, mode='cauchy',
        ...                     max_steps=500, n_trials=10)
        >>> walker.simulate()
        >>> print(walker.food_found_total)
    """

    # ── body geometry (mm) ──────────────────────────────
    BODY_WIDTH: float = 2.0
    BODY_LENGTH: float = 7.0

    def __init__(
        self,
        cauchy_alpha: float = 1.0,
        *,
        n_trials: int = 1_000,
        border: float = 10_000.0,
        mode: str = "cauchy",
        max_steps: int = 1_000,
        total_food: int = 10_000,
        food_mode: str = "random",
    ) -> None:
        self.cauchy_alpha = cauchy_alpha
        self.n_trials = n_trials
        self.border = border
        self.mode = mode
        self.max_steps = max_steps
        self.total_food = total_food
        self.food_mode = food_mode

        self._reset_trial()

    # ── public interface ────────────────────────────────

    def simulate(self) -> None:
        """Run a single foraging trial.

        Resets the agent, scatters food, steps until
        *max_steps*, and populates :attr:`food_found_total`.
        """
        self._reset_trial()
        self._scatter_food()
        while self._count <= self.max_steps:
            self._step_levy()
        self.food_found_total = self.total_food - self._total_food_updated
        self._calc_step_sizes()

    def simulate_multiple(self) -> dict[str, NDArray]:
        """Run *n_trials* independent trials.

        Returns:
            Dictionary keyed by:

            - ``food_found`` — shape ``(n_trials,)``
            - ``step_sizes`` — shape ``(n_trials, n_cols)``

            Column count *n_cols* is determined by the longest
            trial (hyperspace wrapping can add sub-steps beyond
            *max_steps*).  Shorter trials are NaN-padded.
        """
        n = self.n_trials
        food_list: list[float] = []
        step_list: list[NDArray] = []
        for trial in range(n):
            self.simulate()
            food_list.append(float(self.food_found_total))
            step_list.append(self.step_size_arr.copy())
        max_cols = max(len(s) for s in step_list)
        results: dict[str, NDArray] = {
            "food_found": np.array(food_list),
            "step_sizes": np.full((n, max_cols), np.nan),
        }
        for i, s in enumerate(step_list):
            results["step_sizes"][i, : len(s)] = s
        return results

    # ── food placement ──────────────────────────────────

    def _scatter_food(self) -> None:
        """Place food items according to *food_mode*."""
        if self.food_mode == "random":
            self._scatter_random()
        elif self.food_mode == "clustered":
            self._scatter_clustered()
        else:
            raise ValueError(
                f"Unknown food_mode '{self.food_mode}'; "
                "use 'random' or 'clustered'."
            )

    def _scatter_random(self) -> None:
        """Scatter food items uniformly at random inside the arena."""
        self._food_pos = np.random.randint(
            -10_000, 10_000, size=(self.total_food, 2)
        ).astype(float)

    def _scatter_clustered(self) -> None:
        """Scatter food items in Gaussian clusters.

        Ten items per cluster; cluster centres drawn uniformly,
        items drawn from unit-variance Gaussian around each
        centre.
        """
        items_per_cluster = 10
        n_clusters = self.total_food // items_per_cluster
        centres = np.random.randint(
            -10_000, 10_000, size=(n_clusters, 2)
        ).astype(float)
        positions = []
        for cx, cy in centres:
            xs = np.random.normal(cx, 1.0, items_per_cluster)
            ys = np.random.normal(cy, 1.0, items_per_cluster)
            positions.extend(zip(xs, ys))
        self._food_pos = np.array(positions)

    # ── step logic ──────────────────────────────────────

    def _draw_step_size(self) -> float:
        """Draw a step size from the configured distribution.

        Returns:
            Scalar step length.
        """
        if self.mode == "cauchy":
            return float(np.random.uniform() ** (-1.0 / self.cauchy_alpha))
        elif self.mode == "uniform":
            return float(np.random.normal())
        raise ValueError(f"Unknown mode '{self.mode}'.")

    def _step_levy(self) -> None:
        """Execute one Lévy / random-walk step.

        Draws a random direction and step size, displaces the
        agent, handles toroidal wrapping, and collects food.
        """
        angle = np.random.uniform() * 2.0 * np.pi
        step = self._draw_step_size()
        self._body_angles.append(angle)

        new_x = self._start_x[-1] + step * np.cos(angle)
        new_y = self._start_y[-1] + step * np.sin(angle)
        self._step_counter += 1

        current = np.array([[self._start_x[-1], self._start_y[-1]]])
        end = np.array([[new_x, new_y]])

        # ── toroidal wrap ───────────────────────────────
        if abs(new_x) > self.border or abs(new_y) > self.border:
            solver = HyperspaceSolver(
                vector=np.vstack((current, end)),
                border=self.border,
            )
            sub_steps = solver.solve()
            starts = sub_steps[::2]
            stops = sub_steps[1::2]
            for i in range(1, len(starts)):
                self._start_x.append(starts[i, 0])
                self._start_y.append(starts[i, 1])
            for i in range(len(stops)):
                self._end_x.append(stops[i, 0])
                self._end_y.append(stops[i, 1])
            for i in range(len(starts)):
                c = np.array([[starts[i, 0], starts[i, 1]]])
                e = np.array([[stops[i, 0], stops[i, 1]]])
                self._food_routine(c, e)
            self._start_x.append(stops[-1, 0])
            self._start_y.append(stops[-1, 1])
        else:
            self._end_x.append(new_x)
            self._end_y.append(new_y)
            self._start_x.append(new_x)
            self._start_y.append(new_y)
            self._food_routine(current, end)

        self._count += 1

    # ── food collection ─────────────────────────────────

    def _food_routine(self, current: NDArray, end: NDArray) -> None:
        """Check for food inside the rectangular body corridor
        between *current* and *end* and remove collected items.

        Args:
            current: Start position ``(1, 2)``.
            end: End position ``(1, 2)``.
        """
        dx = end[0, 0] - current[0, 0]
        dy = end[0, 1] - current[0, 1]
        if dx == 0.0 and dy == 0.0:
            return

        try:
            slope_h = dy / dx
            slope_w = -1.0 / slope_h
        except (ZeroDivisionError, FloatingPointError):
            return

        hw = self.BODY_WIDTH / 2
        ddx = np.sqrt(hw ** 2 / (1 + slope_w ** 2))
        ddy = slope_w * ddx
        corners = np.array(
            [
                [current[0, 0] + ddx, current[0, 1] + ddy],
                [current[0, 0] - ddx, current[0, 1] - ddy],
                [end[0, 0] + ddx, end[0, 1] + ddy],
                [end[0, 0] - ddx, end[0, 1] - ddy],
            ]
        )
        # pre-filter by distance
        dist_sq = np.sum((self._food_pos - current) ** 2, axis=1)
        near = dist_sq <= self.BODY_LENGTH ** 2
        if not np.any(near):
            return
        candidates = self._food_pos[near]
        try:
            hull = Delaunay(corners)
            inside = hull.find_simplex(candidates) >= 0
        except Exception:
            inside = np.zeros(len(candidates), dtype=bool)
        near_idx = np.where(near)[0]
        collected = near_idx[inside]
        keep = np.ones(len(self._food_pos), dtype=bool)
        keep[collected] = False
        self._food_pos = self._food_pos[keep]
        self._total_food_updated = len(self._food_pos)

    # ── utilities ───────────────────────────────────────

    def _calc_step_sizes(self) -> None:
        """Compute Euclidean step sizes from the position lists."""
        x = np.array(self._start_x)
        y = np.array(self._start_y)
        self.step_size_arr = np.zeros(len(x))
        self.step_size_arr[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

    def _reset_trial(self) -> None:
        """Clear all per-trial state."""
        self._count = 0
        self._step_counter = 0
        self._start_x: list[float] = [0.0]
        self._start_y: list[float] = [0.0]
        self._end_x: list[float] = []
        self._end_y: list[float] = []
        self._body_angles: list[float] = [0.0]
        self.food_found_total: int = 0
        self._total_food_updated: int = self.total_food
        self._food_pos = np.empty((0, 2))
        self.step_size_arr = np.array([])
