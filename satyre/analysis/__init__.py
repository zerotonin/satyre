"""
╔══════════════════════════════════════════════════════════╗
║  analysis « post-simulation metrics & plotting »         ║
╚══════════════════════════════════════════════════════════╝

Functions for computing exploration rate, food-collection
efficiency, step-size distributions, movement-type
classification, and drift-angle statistics from simulation
results.
"""

from satyre.analysis.area_covered import (
    exploration_rate_ellipse,
    exploration_rate_disc,
)
from satyre.analysis.food_found import summarise_food_collected
from satyre.analysis.step_size import step_size_distribution
from satyre.analysis.movement_types import classify_movements

__all__ = [
    "exploration_rate_ellipse",
    "exploration_rate_disc",
    "summarise_food_collected",
    "step_size_distribution",
    "classify_movements",
]
