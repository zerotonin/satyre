"""
╔══════════════════════════════════════════════════════════╗
║  tests « satyre test suite »                            ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import pytest

from satyre.utils.hyperspace_solver import HyperspaceSolver
from satyre.simulation.levy_walker import LevyWalker
from satyre.simulation.markov_walker import MarkovWalker
from satyre.utils.preprocessing import build_transition_matrices
from satyre.analysis.movement_types import classify_movements
from satyre.analysis.step_size import step_size_distribution


# ══════════════════════════════════════════════════════════
#  HyperspaceSolver
# ══════════════════════════════════════════════════════════

class TestHyperspaceSolver:
    """Tests for toroidal boundary wrapping."""

    def test_no_crossing(self):
        """Vector inside the arena should return start and end."""
        vec = np.array([[0.0, 0.0], [5.0, 5.0]])
        solver = HyperspaceSolver(vector=vec, border=10_000)
        steps = solver.solve()
        # last point should match the original endpoint
        np.testing.assert_allclose(steps[-1], vec[1], atol=1e-6)

    def test_single_crossing(self):
        """Vector crossing one wall should produce ≥4 rows."""
        vec = np.array([[9000.0, 0.0], [11000.0, 0.0]])
        solver = HyperspaceSolver(vector=vec, border=10_000)
        steps = solver.solve()
        assert steps.shape[0] >= 4
        # final point must be inside the arena
        assert np.all(np.abs(steps[-1]) <= 10_000 + 1e-6)

    def test_preserves_distance(self):
        """Total sub-step distance should approximate the
        original vector length."""
        vec = np.array([[-9000.0, -9000.0], [-14000.0, -16000.0]])
        solver = HyperspaceSolver(vector=vec, border=10_000)
        steps = solver.solve()
        total = 0.0
        for i in range(0, len(steps) - 1, 2):
            total += np.linalg.norm(steps[i + 1] - steps[i])
        expected = np.linalg.norm(vec[1] - vec[0])
        assert abs(total - expected) / expected < 0.05


# ══════════════════════════════════════════════════════════
#  LevyWalker
# ══════════════════════════════════════════════════════════

class TestLevyWalker:
    """Tests for the Lévy-flight / random-walk agent."""

    def test_simulate_runs(self):
        """A short simulation should complete without error."""
        walker = LevyWalker(
            cauchy_alpha=1.5, mode="cauchy", max_steps=50, n_trials=1
        )
        walker.simulate()
        assert walker.food_found_total >= 0

    def test_uniform_mode(self):
        """Uniform mode should also run."""
        walker = LevyWalker(
            mode="uniform", max_steps=50, n_trials=1
        )
        walker.simulate()
        assert walker.step_size_arr is not None

    def test_multiple_trials(self):
        """simulate_multiple should return correct shapes."""
        walker = LevyWalker(
            cauchy_alpha=1.0, max_steps=20, n_trials=3
        )
        results = walker.simulate_multiple()
        assert results["food_found"].shape == (3,)
        assert results["step_sizes"].shape[0] == 3

    def test_clustered_food(self):
        """Clustered food mode should not crash."""
        walker = LevyWalker(
            max_steps=30, n_trials=1, food_mode="clustered"
        )
        walker.simulate()

    def test_invalid_food_mode(self):
        """Invalid food mode should raise ValueError."""
        walker = LevyWalker(max_steps=10, n_trials=1, food_mode="bogus")
        with pytest.raises(ValueError, match="Unknown food_mode"):
            walker.simulate()


# ══════════════════════════════════════════════════════════
#  MarkovWalker
# ══════════════════════════════════════════════════════════

class TestMarkovWalker:
    """Tests for the HMM-driven agent."""

    @staticmethod
    def _dummy_matrices(n: int = 10):
        """Create minimal valid transition matrices."""
        prob = np.random.dirichlet(np.ones(n), size=n)
        cumsum = np.cumsum(prob, axis=1)
        pm_index = np.tile(np.arange(n, dtype=float), (n, 1))
        velos = np.random.randn(n, 3) * 0.5
        return cumsum, pm_index, velos

    def test_simulate_runs(self):
        """A short Markov simulation should complete."""
        trans, idx, velos = self._dummy_matrices()
        walker = MarkovWalker(
            trans, idx, velos, max_steps=30, n_trials=1
        )
        walker.simulate()
        assert walker.food_found_total >= 0

    def test_multiple_returns_dict(self):
        """simulate_multiple should return a dict with the
        expected keys."""
        trans, idx, velos = self._dummy_matrices()
        walker = MarkovWalker(
            trans, idx, velos, max_steps=20, n_trials=2
        )
        results = walker.simulate_multiple()
        assert "food_found" in results
        assert "psi_angles" in results
        assert results["food_found"].shape == (2,)

    def test_psi_angles_populated(self):
        """Drift angles should be populated after simulation."""
        trans, idx, velos = self._dummy_matrices()
        walker = MarkovWalker(
            trans, idx, velos, max_steps=50, n_trials=1
        )
        walker.simulate()
        assert len(walker.psi_angles) > 0


# ══════════════════════════════════════════════════════════
#  Analysis helpers
# ══════════════════════════════════════════════════════════

class TestAnalysis:
    """Tests for analysis utilities."""

    def test_classify_movements(self):
        """Classification should return three categories."""
        velos = np.array([
            [0.0, 0.0, 0.0],   # stationary
            [2.0, 0.0, 0.0],   # translational
            [0.0, 0.0, 5.0],   # rotational
        ])
        result = classify_movements(np.array([0, 1, 2]), velos)
        assert result["labels"] == [
            "stationary", "translational", "rotational"
        ]
        assert pytest.approx(sum(result["proportions"]), abs=1e-6) == 100.0

    def test_step_size_distribution(self):
        """Step-size stats should have correct keys."""
        data = np.random.rand(5, 20)
        result = step_size_distribution(data)
        assert "grand_mean" in result
        assert "all_steps" in result
        assert len(result["mean_per_trial"]) == 5
