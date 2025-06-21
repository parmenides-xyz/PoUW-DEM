"""
Grid Simulation Module

Provides tools for simulating and coordinating with electrical grids:
- MATPOWER integration for power flow analysis
- Mosaik framework coordination for co-simulation
"""

from .matpower_integration import MATPOWERGrid, GridState, GridOptimizationTasks
from .mosaik_coordinator import MosaikCoordinator

# For backwards compatibility
MatpowerIntegration = MATPOWERGrid

__all__ = [
    "MATPOWERGrid",
    "GridState",
    "GridOptimizationTasks",
    "MatpowerIntegration",  # Backwards compatibility alias
    "MosaikCoordinator",
]