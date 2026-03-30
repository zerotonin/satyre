"""
╔══════════════════════════════════════════════════════════╗
║  markov_walker « HMM-driven locomotion simulator »       ║
╚══════════════════════════════════════════════════════════╝

Replays empirical prototypical-movement (PM) sequences via a
first-order Markov chain built from experimental velocity
clusters.  The agent moves through a toroidal 2-D arena and
collects food items by tactile or visual contact.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from satyre.utils.hyperspace_solver import HyperspaceSolver


# ──────────────────────────────────────────────────────────
#  MarkovWalker
# ──────────────────────────────────────────────────────────
class MarkovWalker:
    """Simulate fly locomotion using a Hidden-Markov-Model of
    prototypical movements derived from experimental data.

    The transition-probability matrix drives the selection of
    successive PMs.  Each PM carries thrust, slip, and yaw
    velocities that determine displacement and body rotation.

    Args:
        transition_matrix: Cumulative transition-probability
            matrix of shape ``(N, N)`` where ``N`` is the
            number of prototypical movements.
        pm_index: Row-sorted PM index matrix of shape
            ``(N, N)``.  Each row maps cumulative-probability
            columns back to PM identifiers.
        velocity_array: Array of shape ``(N, 3)`` holding
            ``[thrust, slip, yaw]`` velocities for each PM
            (units: mm s⁻¹ / rad s⁻¹).
        n_trials: Number of independent simulation trials
            executed by :meth:`simulate_multiple`.
        border: Half-width of the square toroidal arena
            (mm).
        max_steps: Maximum number of PM steps per trial.
        total_food: Number of food items scattered in the
            arena at the start of each trial.
        food_mode: Food-distribution mode — ``'random'`` or
            ``'clustered'``.
        condition: Sensory condition — ``'dark'`` (tactile
            body-overlap detection) or ``'light'`` (visual
            detection within a 20 mm radius).

    Attributes:
        food_found_total: Number of food items collected in
            the most recent single trial.
        all_performed_pms: List of PM indices visited during
            the most recent trial.
        psi_angles: Drift (ψ) angles between body heading
            and trajectory direction.

    Example:
        >>> import numpy as np
        >>> from satyre import MarkovWalker
        >>> trans = np.eye(5, dtype=float)        # dummy identity
        >>> trans = np.cumsum(trans, axis=1)
        >>> pm_idx = np.tile(np.arange(5), (5, 1)).astype(float)
        >>> velos  = np.random.randn(5, 3) * 0.5
        >>> walker = MarkovWalker(trans, pm_idx, velos,
        ...                       max_steps=100, n_trials=2)
        >>> walker.simulate()
    """

    # ── body geometry (mm) ──────────────────────────────
    BODY_WIDTH: float = 2.0
    BODY_LENGTH: float = 3.0
    SIM_RESOLUTION: int = 500  # frames per second

    def __init__(
        self,
        transition_matrix: NDArray[np.floating],
        pm_index: NDArray[np.floating],
        velocity_array: NDArray[np.floating],
        *,
        n_trials: int = 1_000,
        border: float = 10_000.0,
        max_steps: int = 10_000,
        total_food: int = 1_000,
        food_mode: str = "random",
        condition: str = "dark",
    ) -> None:
        self.transition_matrix = transition_matrix
        self.pm_index = pm_index
        self.velocity_array = velocity_array
        self.n_trials = n_trials
        self.border = border
        self.max_steps = max_steps
        self.total_food = total_food
        self.food_mode = food_mode
        self.condition = condition

        self._body_length_sq = self.BODY_LENGTH ** 2
        self._body_length_vec = np.array([[0.0, self.BODY_LENGTH]])
        self._first_pm = self._find_first_pm()
        self._reset_trial()

    # ── public interface ────────────────────────────────

    def simulate(self) -> None:
        """Run a single foraging trial.

        Resets the agent, scatters food, then steps through
        PMs until *max_steps* is reached.  Populates
        :attr:`food_found_total` and :attr:`psi_angles`.
        """
        self._reset_trial()
        self._scatter_food()
        while self._count <= self.max_steps:
            self._step_markov()
        self._calc_step_sizes()
        self.food_found_total = self.total_food - self._total_food_updated

    def simulate_multiple(self) -> dict[str, NDArray]:
        """Run *n_trials* independent trials and collect results.

        Returns:
            Dictionary with the following keys (each value is
            an ``ndarray`` indexed by trial number).  Column
            count is determined by the longest trial
            (hyperspace wrapping can add sub-steps); shorter
            trials are NaN-padded.

            - ``food_found`` — shape ``(n_trials,)``
            - ``step_sizes`` — shape ``(n_trials, n_cols)``
            - ``psi_angles`` — shape ``(n_trials, n_psi)``
            - ``position_x`` — shape ``(n_trials, n_cols)``
            - ``position_y`` — shape ``(n_trials, n_cols)``
            - ``head_x``, ``head_y``, ``tail_x``, ``tail_y``
        """
        n = self.n_trials

        # ── collect variable-length per-trial data ──────
        food_list: list[float] = []
        steps_list: list[NDArray] = []
        psi_list: list[list[float]] = []
        pos_x_list: list[list[float]] = []
        pos_y_list: list[list[float]] = []
        hx_list: list[list[float]] = []
        hy_list: list[list[float]] = []
        tx_list: list[list[float]] = []
        ty_list: list[list[float]] = []

        for _ in range(n):
            self.simulate()
            food_list.append(float(self.food_found_total))
            steps_list.append(self._step_size_arr.copy())
            psi_list.append(list(self.psi_angles))
            pos_x_list.append(list(self._start_x))
            pos_y_list.append(list(self._start_y))
            hx_list.append(list(self._head_x))
            hy_list.append(list(self._head_y))
            tx_list.append(list(self._tail_x))
            ty_list.append(list(self._tail_y))

        # ── determine padded dimensions ─────────────────
        max_pos = max(len(v) for v in pos_x_list)
        max_psi = max(len(v) for v in psi_list)
        max_step = max(len(v) for v in steps_list)
        max_body = max(len(v) for v in hx_list)

        def _padded(src: list, rows: int, cols: int) -> NDArray:
            out = np.full((rows, cols), np.nan)
            for i, v in enumerate(src):
                length = min(len(v), cols)
                out[i, :length] = np.asarray(v)[:length]
            return out

        results: dict[str, NDArray] = {
            "food_found": np.array(food_list),
            "step_sizes": _padded(steps_list, n, max_step),
            "psi_angles": _padded(psi_list, n, max_psi),
            "position_x": _padded(pos_x_list, n, max_pos),
            "position_y": _padded(pos_y_list, n, max_pos),
            "head_x": _padded(hx_list, n, max_body),
            "head_y": _padded(hy_list, n, max_body),
            "tail_x": _padded(tx_list, n, max_body),
            "tail_y": _padded(ty_list, n, max_body),
        }
        return results

    # ── food placement ──────────────────────────────────

    def _scatter_food(self) -> None:
        """Place food items in the arena according to *food_mode*.

        Dispatches to :meth:`_scatter_random` or
        :meth:`_scatter_clustered`.
        """
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
        self._food_border = 60.0
        self._food_pos = np.random.randint(
            int(-self._food_border),
            int(self._food_border),
            size=(self.total_food, 2),
        ).astype(float)

    def _scatter_clustered(self) -> None:
        """Scatter food items in Gaussian clusters.

        Ten items per cluster, with cluster centres drawn
        uniformly and items drawn from a unit-variance
        Gaussian around each centre.
        """
        self._food_border = 60.0
        items_per_cluster = 10
        n_clusters = self.total_food // items_per_cluster
        centres = np.random.randint(
            int(-self._food_border),
            int(self._food_border),
            size=(n_clusters, 2),
        ).astype(float)
        positions = []
        for cx, cy in centres:
            xs = np.random.normal(cx, 1.0, items_per_cluster)
            ys = np.random.normal(cy, 1.0, items_per_cluster)
            positions.extend(zip(xs, ys))
        self._food_pos = np.array(positions)

    # ── Markov step logic ───────────────────────────────

    def _find_first_pm(self) -> int:
        """Identify the PM with the lowest median velocity.

        Returns:
            Index of the stationary (or near-stationary) PM.
        """
        medians = np.nanmedian(self.velocity_array, axis=1)
        return int(np.nanargmin(np.abs(medians)))

    def _step_markov(self) -> None:
        """Execute one Markov step: select the next PM,
        compute displacement, handle toroidal wrapping, and
        collect food.
        """
        # ── dice roll → next PM ─────────────────────────
        dice = np.random.uniform()
        row = self.transition_matrix[self.all_performed_pms[-1], :]
        target_col = int(np.searchsorted(row, dice, side="left"))
        target_col = min(target_col, len(row) - 1)
        target_pm = int(self.pm_index[self.all_performed_pms[-1], target_col])
        self.all_performed_pms.append(target_pm)

        # ── velocities → displacements ──────────────────
        velo = self.velocity_array[target_pm]
        thrust_dist = velo[0] / self.SIM_RESOLUTION
        slip_dist = velo[1] / self.SIM_RESOLUTION
        traj_angle = velo[2] / self.SIM_RESOLUTION
        self._cum_yaw += traj_angle
        body_angle = self._cum_yaw
        self._body_angles.append(traj_angle)

        current = np.array([[self._start_x[-1], self._start_y[-1]]])
        end = self._calc_new_position(
            current, body_angle, traj_angle, thrust_dist, slip_dist
        )
        self._step_counter += 1

        # ── toroidal wrap ───────────────────────────────
        if (
            abs(end[0, 0]) > self.border
            or abs(end[0, 1]) > self.border
        ):
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
                self._food_routine_rect(c, e)
            final = np.array([[stops[-1, 0], stops[-1, 1]]])
            self._start_x.append(final[0, 0])
            self._start_y.append(final[0, 1])
        else:
            self._end_x.append(end[0, 0])
            self._end_y.append(end[0, 1])
            self._start_x.append(end[0, 0])
            self._start_y.append(end[0, 1])
            self._food_routine_rect(current, end)

        # ── body-part coordinates ───────────────────────
        mid = np.array([[self._start_x[-1], self._start_y[-1]]])
        head = self._calc_body_endpoint(
            mid, body_angle, traj_angle, self.BODY_LENGTH / 2
        )
        tail = self._calc_body_endpoint(
            mid,
            body_angle + np.pi,
            traj_angle,
            self.BODY_LENGTH / 2,
        )
        self._head_x.append(head[0, 0])
        self._head_y.append(head[0, 1])
        self._tail_x.append(tail[0, 0])
        self._tail_y.append(tail[0, 1])

        # ── psi angle ──────────────────────────────────
        if len(self._start_x) >= 2:
            vec_body = np.array(
                [head[0, 0] - tail[0, 0], head[0, 1] - tail[0, 1]]
            )
            vec_traj = np.array(
                [
                    self._start_x[-1] - self._start_x[-2],
                    self._start_y[-1] - self._start_y[-2],
                ]
            )
            denom = np.linalg.norm(vec_body) * np.linalg.norm(vec_traj)
            if denom > 0:
                cos_psi = np.clip(np.dot(vec_body, vec_traj) / denom, -1, 1)
                self.psi_angles.append(np.arccos(cos_psi))
            else:
                self.psi_angles.append(0.0)
        else:
            self.psi_angles.append(0.0)

        self._count += 1

    # ── coordinate transforms ───────────────────────────

    @staticmethod
    def _calc_new_position(
        current: NDArray,
        body_angle: float,
        traj_angle: float,
        thrust: float,
        slip: float,
    ) -> NDArray:
        """Compute end position via a Fick rotation.

        Args:
            current: Current ``(1, 2)`` position.
            body_angle: Cumulative yaw angle (rad).
            traj_angle: Incremental trajectory angle (rad).
            thrust: Forward displacement (mm).
            slip: Sideward displacement (mm).

        Returns:
            New ``(1, 2)`` position array.
        """
        theta = body_angle + traj_angle
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        disp = np.array([[slip, thrust]])
        return current + (rot @ disp.T).T

    @staticmethod
    def _calc_body_endpoint(
        mid: NDArray,
        body_angle: float,
        traj_angle: float,
        half_length: float,
    ) -> NDArray:
        """Project a body endpoint (head or tail) from mid.

        Args:
            mid: Midpoint ``(1, 2)`` position.
            body_angle: Cumulative yaw angle (rad).
            traj_angle: Incremental trajectory angle (rad).
            half_length: Half body-length (mm).

        Returns:
            Endpoint ``(1, 2)`` position array.
        """
        theta = body_angle + traj_angle
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        vec = np.array([[0.0, half_length]])
        return mid + (rot @ vec.T).T

    # ── food collection ─────────────────────────────────

    def _food_routine_rect(
        self, current: NDArray, end: NDArray
    ) -> None:
        """Check for food items inside the rectangular body
        corridor swept between *current* and *end*, and
        remove collected items from the arena.

        Args:
            current: Start position ``(1, 2)``.
            end: End position ``(1, 2)``.
        """
        dx_pos = end[0, 0] - current[0, 0]
        dy_pos = end[0, 1] - current[0, 1]
        if dx_pos == 0 and dy_pos == 0:
            return
        # quick distance pre-filter
        dist_sq = np.sum(
            (self._food_pos - current) ** 2, axis=1
        )
        near = dist_sq <= self._body_length_sq * 4
        if not np.any(near):
            return
        candidates = self._food_pos[near]
        # build rectangle corners
        try:
            slope_h = dy_pos / dx_pos
            slope_w = -1.0 / slope_h
        except (ZeroDivisionError, FloatingPointError):
            # vertical or horizontal movement
            self._food_pos = self._food_pos[~near]
            self._total_food_updated = len(self._food_pos)
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
        try:
            hull = Delaunay(corners)
            inside = hull.find_simplex(candidates) >= 0
        except Exception:
            inside = np.zeros(len(candidates), dtype=bool)
        # update near mask
        near_indices = np.where(near)[0]
        collected = near_indices[inside]
        keep = np.ones(len(self._food_pos), dtype=bool)
        keep[collected] = False
        self._food_pos = self._food_pos[keep]
        self._total_food_updated = len(self._food_pos)

    # ── utilities ───────────────────────────────────────

    def _calc_step_sizes(self) -> None:
        """Compute Euclidean step sizes from the position lists."""
        x = np.array(self._start_x)
        y = np.array(self._start_y)
        self._step_size_arr = np.zeros(len(x))
        self._step_size_arr[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

    def _reset_trial(self) -> None:
        """Clear all per-trial state to prepare for a fresh
        simulation."""
        self._count = 0
        self._step_counter = 0
        self._cum_yaw = 0.0
        self._start_x: list[float] = [0.0]
        self._start_y: list[float] = [0.0]
        self._end_x: list[float] = []
        self._end_y: list[float] = []
        self._body_angles: list[float] = [0.0]
        half = self._body_length_vec / 2
        self._head_x: list[float] = [half[0, 0]]
        self._head_y: list[float] = [half[0, 1]]
        self._tail_x: list[float] = [-half[0, 0]]
        self._tail_y: list[float] = [-half[0, 1]]
        self.all_performed_pms: list[int] = [self._first_pm]
        self.psi_angles: list[float] = []
        self.food_found_total: int = 0
        self._total_food_updated: int = self.total_food
        self._food_pos = np.empty((0, 2))
