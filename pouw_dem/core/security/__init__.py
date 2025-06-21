"""
Security Module

Provides security features for the PoUW-DEM system:
- SGX (Software Guard Extensions) interface for trusted execution
- Attestation and verification mechanisms
- Secure enclave support
"""

from .sgx_interface import (
    SGXInterface,
    AttestationStatus,
    SGXQuote,
    TaskExecutionProof
)

# Mock enclave for testing
from .enclaves.mock_enclave import MockEnclave

__all__ = [
    # SGX interface
    "SGXInterface",
    "AttestationStatus",
    "SGXQuote",
    "TaskExecutionProof",
    
    # Testing
    "MockEnclave",
]