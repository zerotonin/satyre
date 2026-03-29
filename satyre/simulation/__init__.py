"""
╔══════════════════════════════════════════════════════════╗
║  simulation « walking strategy simulators »             ║
╚══════════════════════════════════════════════════════════╝

Provides two agent classes that move through a toroidal 2-D
arena and collect food items:

* :class:`MarkovWalker` — HMM-driven replay of empirical
  prototypical-movement sequences (dark-fly or OregonR).
* :class:`LevyWalker` — Lévy-flight / random-walk agent
  with configurable step-size distributions.
"""

from satyre.simulation.markov_walker import MarkovWalker
from satyre.simulation.levy_walker import LevyWalker

__all__ = ["MarkovWalker", "LevyWalker"]
