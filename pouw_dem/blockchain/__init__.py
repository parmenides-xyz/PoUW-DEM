"""
Blockchain Integration Module

Provides blockchain integration capabilities for recording energy profiles,
managing smart contracts, and interfacing with the Ethereum/Polygon networks.
"""

from .blockchain_integration import (
    BlockchainRecorder,
    integrate_blockchain_recorder
)

__all__ = [
    "BlockchainRecorder",
    "integrate_blockchain_recorder",
]