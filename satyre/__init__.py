"""
╔═══════════════════════════════════════════════════════════════════════╗
║  ███████╗ █████╗ ████████╗██╗   ██╗██████╗ ███████╗                 ║
║  ██╔════╝██╔══██╗╚══██╔══╝╚██╗ ██╔╝██╔══██╗██╔════╝                ║
║  ███████╗███████║   ██║    ╚████╔╝ ██████╔╝█████╗                  ║
║  ╚════██║██╔══██║   ██║     ╚██╔╝  ██╔══██╗██╔══╝                  ║
║  ███████║██║  ██║   ██║      ██║   ██║  ██║███████╗                 ║
║  ╚══════╝╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚══════╝                 ║
║                                                                       ║
║  « Sensory locomotion strATegY fRamEwork »                            ║
║                                                                       ║
║  Simulating and analysing fly exploratory behaviour under              ║
║  visual deprivation.                                                   ║
║                                                                       ║
║  Authors: Irene M. Aji & Bart R.H. Geurten                           ║
║  Licence: MIT                                                          ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

__version__ = "1.0.0"
__author__ = "Irene M. Aji & Bart R.H. Geurten"

from satyre.simulation.markov_walker import MarkovWalker
from satyre.simulation.levy_walker import LevyWalker
from satyre.utils.hyperspace_solver import HyperspaceSolver

__all__ = [
    "MarkovWalker",
    "LevyWalker",
    "HyperspaceSolver",
]
