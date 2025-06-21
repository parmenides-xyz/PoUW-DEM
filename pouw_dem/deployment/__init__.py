"""
Deployment Module

Provides tools for deploying the PoUW-DEM system:
- Smart contract deployment to mainnet and test networks
- Polygon network deployment
- Trained model deployment
- Full system orchestration
"""

# Import with error handling for optional dependencies
import warnings

def _safe_import(module_name, items):
    """Safely import items from a module."""
    imported = {}
    try:
        module = __import__(f"pouw_dem.deployment.{module_name}", fromlist=items if isinstance(items, list) else [items])
        if isinstance(items, list):
            for item in items:
                try:
                    imported[item] = getattr(module, item)
                except AttributeError:
                    warnings.warn(f"Could not import {item} from {module_name}")
                    imported[item] = None
        else:
            imported[items] = getattr(module, items)
    except ImportError as e:
        warnings.warn(f"Could not import from {module_name}: {e}")
        if isinstance(items, list):
            for item in items:
                imported[item] = None
        else:
            imported[items] = None
    return imported

# Import deployment functions
deploy_contracts = _safe_import("deploy_mainnet", "deploy_contracts").get("deploy_contracts")
polygon_imports = _safe_import("deploy_to_polygon", ["PolygonDeployer", "main"])
PolygonDeployer = polygon_imports.get("PolygonDeployer")
deploy_to_polygon = polygon_imports.get("main")
TrainedAgentDeployer = _safe_import("deploy_trained_models", "TrainedAgentDeployer").get("TrainedAgentDeployer")
run_full_system = _safe_import("run_full_system", "run_full_system").get("run_full_system")

__all__ = [
    # Contract deployment
    "deploy_contracts",
    "PolygonDeployer",
    "deploy_to_polygon",
    
    # Model deployment
    "TrainedAgentDeployer",
    
    # System orchestration
    "run_full_system",
]