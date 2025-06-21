"""
Core Module

Contains the core functionality of the PoUW-DEM system including:
- Intelligent agents for mining optimization
- Grid simulation and coordination
- Task scheduling and prioritization
- Security features including SGX support
"""

# Import main components from submodules with lazy loading
import warnings

def _safe_import(module_name, item_name):
    """Safely import an item from a module."""
    try:
        module = __import__(f"pouw_dem.core.{module_name}", fromlist=[item_name])
        return getattr(module, item_name)
    except (ImportError, AttributeError) as e:
        warnings.warn(f"Could not import {item_name} from {module_name}: {e}")
        return None

# Agent imports
PoUWAgent = _safe_import("agents", "PoUWAgent")
IntelligentMiningAgent = _safe_import("agents", "IntelligentMiningAgent")
Multi_Agent = _safe_import("agents", "Multi_Agent")
DSAC = _safe_import("agents", "DSAC")
ProgressusEnv = _safe_import("agents", "ProgressusEnv")

# Scheduling imports
DynamicTaskScheduler = _safe_import("scheduling", "DynamicTaskScheduler")
TaskType = _safe_import("scheduling", "TaskType")
Priority = _safe_import("scheduling", "Priority")
ScheduledTask = _safe_import("scheduling", "ScheduledTask")

# Security imports
SGXInterface = _safe_import("security", "SGXInterface")
AttestationStatus = _safe_import("security", "AttestationStatus")
SGXQuote = _safe_import("security", "SGXQuote")

# Grid simulation imports
MosaikCoordinator = _safe_import("grid_simulation", "MosaikCoordinator")
matpower_integration = _safe_import("grid_simulation", "matpower_integration")

__all__ = [
    # Agents
    "PoUWAgent",
    "IntelligentMiningAgent",
    "Multi_Agent",
    "DSAC",
    "ProgressusEnv",
    
    # Scheduling
    "DynamicTaskScheduler",
    "TaskType",
    "Priority",
    "ScheduledTask",
    
    # Security
    "SGXInterface",
    "AttestationStatus",
    "SGXQuote",
    
    # Grid simulation
    "MosaikCoordinator",
    "matpower_integration",
]