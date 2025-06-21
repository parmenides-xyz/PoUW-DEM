"""
API Integration Module

Provides interfaces for connecting to external services including:
- Grid operators (ERCOT, CAISO, PJM, MISO)
- Mining pools (Foundry USA, AntPool, F2Pool, ViaBTC, Luxor)
- Real-time energy and mining data APIs
"""

# Grid API integrations
from .ercot_real_api import (
    ERCOTRealAPI,
    integrate_with_pouw_system,
    get_ercot_ecrs,
    get_ercot_recommendation
)

from .grid_api_integration import (
    GridAPIConnector,
    ERCOTConnector,
    CAISOConnector,
    PJMConnector,
    MISOConnector,
    GridSignalProcessor,
    create_grid_integration_service,
    get_grid_status,
    get_operator_status,
    calculate_grid_reward
)

# Mining pool integrations
from .mining_pool_integration import (
    MiningPoolConnector,
    FoundryUSAPool,
    AntPoolConnector,
    F2PoolConnector,
    ViaBTCConnector,
    LuxorConnector
)

from .real_mining_pool_integration import (
    RealMiningPoolConnector,
    SlushPoolAPI,
    F2PoolAPI,
    ViaBTCAPI,
    LuxorAPI,
    BTCcomAPI,
    MempoolSpaceAPI
)

# For backwards compatibility
RealMiningPoolIntegration = RealMiningPoolConnector

__all__ = [
    # ERCOT API
    "ERCOTRealAPI",
    "integrate_with_pouw_system",
    "get_ercot_ecrs",
    "get_ercot_recommendation",
    
    # Grid API connectors
    "GridAPIConnector",
    "ERCOTConnector",
    "CAISOConnector",
    "PJMConnector",
    "MISOConnector",
    "GridSignalProcessor",
    "create_grid_integration_service",
    "get_grid_status",
    "get_operator_status",
    "calculate_grid_reward",
    
    # Mining pool connectors
    "MiningPoolConnector",
    "FoundryUSAPool",
    "AntPoolConnector",
    "F2PoolConnector",
    "ViaBTCConnector",
    "LuxorConnector",
    
    # Real mining pool APIs
    "RealMiningPoolConnector",
    "SlushPoolAPI",
    "F2PoolAPI",
    "ViaBTCAPI",
    "LuxorAPI",
    "BTCcomAPI",
    "MempoolSpaceAPI",
    "RealMiningPoolIntegration",  # Backwards compatibility alias
]