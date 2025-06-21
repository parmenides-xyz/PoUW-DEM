"""
PoUW-DEM Integration Package

A comprehensive system integrating Proof of Useful Work (PoUW) with Dynamic Energy Management (DEM)
for Bitcoin mining operations. This package provides tools for optimizing mining operations based
on grid signals, renewable energy availability, and economic incentives.

Main components:
- API integrations for grid operators and mining pools
- Blockchain integration for recording energy profiles
- Core agents for intelligent mining decisions
- Grid simulation and coordination
- Security features including SGX support
- Training and deployment utilities
"""

__version__ = "1.0.0"

# Import with graceful fallback for optional dependencies
import warnings

def _safe_import(module_path, class_name):
    """Safely import a class, returning None if import fails."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        warnings.warn(f"Could not import {class_name} from {module_path}: {e}")
        return None

# Core components
PoUWAgent = _safe_import("pouw_dem.core.agents.PoUWAgent", "PoUWAgent")
IntelligentMiningAgent = _safe_import("pouw_dem.core.agents.intelligent_mining_agent", "IntelligentMiningAgent")
DynamicTaskScheduler = _safe_import("pouw_dem.core.scheduling.task_scheduler", "DynamicTaskScheduler")
MosaikCoordinator = _safe_import("pouw_dem.core.grid_simulation.mosaik_coordinator", "MosaikCoordinator")
SGXInterface = _safe_import("pouw_dem.core.security.sgx_interface", "SGXInterface")

# API integrations
ERCOTRealAPI = _safe_import("pouw_dem.api.ercot_real_api", "ERCOTRealAPI")
GridAPIConnector = _safe_import("pouw_dem.api.grid_api_integration", "GridAPIConnector")
GridSignalProcessor = _safe_import("pouw_dem.api.grid_api_integration", "GridSignalProcessor")
RealMiningPoolIntegration = _safe_import("pouw_dem.api.real_mining_pool_integration", "RealMiningPoolIntegration")

# Blockchain integration
BlockchainRecorder = _safe_import("pouw_dem.blockchain.blockchain_integration", "BlockchainRecorder")

# Analytics and tracking
AnalyticsTracker = _safe_import("pouw_dem.utils.analytics_tracker", "AnalyticsTracker")

# Training utilities
FederatedTrainer = _safe_import("pouw_dem.training.implement_full_training", "FederatedTrainer")

# Deployment utilities
TrainedAgentDeployer = _safe_import("pouw_dem.deployment.deploy_trained_models", "TrainedAgentDeployer")

__all__ = [
    # Core agents
    "PoUWAgent",
    "IntelligentMiningAgent",
    
    # Scheduling
    "DynamicTaskScheduler",
    
    # Grid simulation
    "MosaikCoordinator",
    
    # Security
    "SGXInterface",
    
    # API integrations
    "ERCOTRealAPI",
    "GridAPIConnector",
    "GridSignalProcessor",
    "RealMiningPoolIntegration",
    
    # Blockchain
    "BlockchainRecorder",
    
    # Analytics
    "AnalyticsTracker",
    
    # Training
    "FederatedTrainer",
    
    # Deployment
    "TrainedAgentDeployer",
]