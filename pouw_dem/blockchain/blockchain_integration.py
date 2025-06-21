#!/usr/bin/env python3
"""
Blockchain Integration for PoUW-DEM System
Connects to Polygon mainnet and interacts with deployed smart contracts
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from web3 import Web3
try:
    from web3.middleware import geth_poa_middleware
except ImportError:
    # For web3 v6+
    try:
        from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
    except ImportError:
        # For web3 v7+
        geth_poa_middleware = None
try:
    from web3.exceptions import ContractLogicError, TimeExhausted
except ImportError:
    # For web3 v7+
    from web3.exceptions import ContractLogicError
    TimeExhausted = Exception
from eth_account import Account
import logging
from flask import jsonify, request

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainRecorder:
    """
    Main class for blockchain interactions with the PoUW-DEM system
    Handles all on-chain operations including allocation recording, NFT minting,
    proof submission, and reward distribution
    """
    
    def __init__(self, private_key: Optional[str] = None, rpc_url: Optional[str] = None):
        """
        Initialize blockchain connection and load contracts
        
        Args:
            private_key: Private key for transaction signing (optional, can use env var)
            rpc_url: Polygon RPC endpoint (optional, defaults to public RPC)
        """
        # Load deployment data
        self.deployment_data = self._load_deployment_data()
        
        # Setup Web3 connection
        self.rpc_url = rpc_url or os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com')
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # Add PoA middleware for Polygon (if available)
        if geth_poa_middleware:
            try:
                if callable(geth_poa_middleware):
                    # For newer versions that return a constructor
                    self.w3.middleware_onion.inject(geth_poa_middleware(), layer=0)
                else:
                    # For older versions
                    self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            except Exception as e:
                logger.warning(f"Could not inject PoA middleware: {e}")
        else:
            logger.info("PoA middleware not available in this web3 version, continuing without it")
        
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Polygon at {self.rpc_url}")
        
        logger.info(f"Connected to Polygon. Chain ID: {self.w3.eth.chain_id}")
        
        # Setup account
        self.private_key = private_key or os.getenv('PRIVATE_KEY')
        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.address = self.account.address
            logger.info(f"Using account: {self.address}")
        else:
            self.account = None
            self.address = None
            logger.warning("No private key provided - running in read-only mode")
        
        # Load contract ABIs
        self.contracts = self._load_contracts()
        
        # Initialize event listeners
        self.event_filters = {}
        self._setup_event_listeners()
        
    def _load_deployment_data(self) -> Dict:
        """Load deployment data from JSON file"""
        deployment_file = os.path.join(
            os.path.dirname(__file__), 
            'full_deployment.json'
        )
        
        with open(deployment_file, 'r') as f:
            return json.load(f)
    
    def _load_contracts(self) -> Dict:
        """Load contract instances with their ABIs"""
        contracts = {}
        
        # Contract ABIs (simplified for main functions)
        abis = {
            'ProofOfUsefulWork': [
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "computePower", "type": "uint256"},
                        {"internalType": "address", "name": "sgxKey", "type": "address"}
                    ],
                    "name": "registerMiner",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "taskId", "type": "uint256"},
                        {"internalType": "bytes32", "name": "outputHash", "type": "bytes32"},
                        {"internalType": "bytes", "name": "sgxAttestation", "type": "bytes"},
                        {"internalType": "uint256", "name": "computationTime", "type": "uint256"},
                        {"internalType": "uint256", "name": "energyUsed", "type": "uint256"},
                        {"internalType": "uint256", "name": "gridImpact", "type": "uint256"}
                    ],
                    "name": "submitTaskProof",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "name": "taskId", "type": "uint256"},
                        {"indexed": True, "name": "miner", "type": "address"},
                        {"indexed": False, "name": "gridImpact", "type": "uint256"}
                    ],
                    "name": "TaskCompleted",
                    "type": "event"
                }
            ],
            'EnergyMarket': [
                {
                    "inputs": [
                        {"internalType": "uint256", "name": "energyAmount", "type": "uint256"},
                        {"internalType": "uint256", "name": "price", "type": "uint256"},
                        {"internalType": "address", "name": "buyer", "type": "address"}
                    ],
                    "name": "recordEnergyTrade",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "name": "seller", "type": "address"},
                        {"indexed": True, "name": "buyer", "type": "address"},
                        {"indexed": False, "name": "amount", "type": "uint256"},
                        {"indexed": False, "name": "price", "type": "uint256"}
                    ],
                    "name": "EnergyTraded",
                    "type": "event"
                }
            ],
            'EnergyProfile': [
                {
                    "inputs": [
                        {"internalType": "address", "name": "miner", "type": "address"},
                        {"internalType": "uint256", "name": "taskId", "type": "uint256"},
                        {"internalType": "uint256", "name": "impact", "type": "uint256"},
                        {"internalType": "bytes32", "name": "proofHash", "type": "bytes32"}
                    ],
                    "name": "mintStabilityNFT",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"internalType": "address", "name": "owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ],
            'EnergyPool': [
                {
                    "inputs": [
                        {"internalType": "address", "name": "recipient", "type": "address"},
                        {"internalType": "uint256", "name": "amount", "type": "uint256"}
                    ],
                    "name": "distributeRewards",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
                    "name": "getBalance",
                    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        }
        
        # Create contract instances
        for contract_name, abi in abis.items():
            if contract_name in self.deployment_data:
                address = self.deployment_data[contract_name]['address']
                contracts[contract_name] = self.w3.eth.contract(
                    address=Web3.to_checksum_address(address),
                    abi=abi
                )
                logger.info(f"Loaded {contract_name} at {address}")
        
        return contracts
    
    def _setup_event_listeners(self):
        """Setup event listeners for important contract events"""
        if 'ProofOfUsefulWork' in self.contracts:
            # Listen for TaskCompleted events
            event_filter = self.contracts['ProofOfUsefulWork'].events.TaskCompleted.create_filter(
                fromBlock='latest'
            )
            self.event_filters['TaskCompleted'] = event_filter
            
        if 'EnergyMarket' in self.contracts:
            # Listen for EnergyTraded events
            event_filter = self.contracts['EnergyMarket'].events.EnergyTraded.create_filter(
                fromBlock='latest'
            )
            self.event_filters['EnergyTraded'] = event_filter
    
    def _estimate_gas_and_send_tx(self, func, value: int = 0) -> str:
        """
        Estimate gas and send transaction
        
        Args:
            func: Contract function to call
            value: ETH value to send (in wei)
            
        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("No account configured for sending transactions")
        
        # Get current gas price
        gas_price = self.w3.eth.gas_price
        
        # Estimate gas
        try:
            gas_estimate = func.estimate_gas({
                'from': self.address,
                'value': value
            })
            # Add 20% buffer
            gas_limit = int(gas_estimate * 1.2)
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}, using default")
            gas_limit = 300000
        
        # Build transaction
        tx = func.build_transaction({
            'from': self.address,
            'value': value,
            'gas': gas_limit,
            'gasPrice': gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.address)
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Transaction sent: {tx_hash.hex()}")
        return tx_hash.hex()
    
    def _wait_for_tx_receipt(self, tx_hash: str, timeout: int = 120) -> Dict:
        """Wait for transaction receipt with timeout"""
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash, 
                timeout=timeout
            )
            if receipt['status'] == 1:
                logger.info(f"Transaction successful. Gas used: {receipt['gasUsed']}")
            else:
                logger.error(f"Transaction failed: {tx_hash}")
            return receipt
        except TimeExhausted:
            logger.error(f"Transaction timeout: {tx_hash}")
            return None
    
    def record_allocation_decision(self, 
                                 allocation_data: Dict,
                                 grid_conditions: Dict) -> Optional[str]:
        """
        Record an energy allocation decision on-chain
        
        Args:
            allocation_data: Dictionary with allocation details
            grid_conditions: Current grid conditions
            
        Returns:
            Transaction hash if successful
        """
        try:
            # For now, we'll record this as an energy trade in the market
            contract = self.contracts['EnergyMarket']
            
            # Extract relevant data
            energy_amount = int(allocation_data.get('energy_mw', 0) * 1e6)  # Convert MW to W
            price = int(grid_conditions.get('price', 50) * 1e18)  # Convert to wei
            buyer = allocation_data.get('buyer_address', '0x0000000000000000000000000000000000000000')
            
            # Send transaction
            func = contract.functions.recordEnergyTrade(
                energy_amount,
                price,
                Web3.to_checksum_address(buyer)
            )
            
            tx_hash = self._estimate_gas_and_send_tx(func)
            
            # Wait for confirmation
            receipt = self._wait_for_tx_receipt(tx_hash)
            
            if receipt and receipt['status'] == 1:
                logger.info(f"Allocation recorded: {energy_amount} W at {price} wei")
                return tx_hash
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to record allocation: {e}")
            return None
    
    def mint_dynamic_nft(self, 
                        miner_address: str,
                        task_id: int,
                        grid_impact: int,
                        proof_hash: str) -> Optional[str]:
        """
        Mint a Dynamic NFT for grid support contribution
        
        Args:
            miner_address: Address of the contributing miner
            task_id: ID of the completed task
            grid_impact: Measured impact on grid stability (0-100)
            proof_hash: Hash of the proof data
            
        Returns:
            Transaction hash if successful
        """
        try:
            contract = self.contracts['EnergyProfile']
            
            # Convert proof hash to bytes32
            if proof_hash.startswith('0x'):
                proof_bytes32 = proof_hash
            else:
                proof_bytes32 = '0x' + proof_hash
            
            # Send transaction
            func = contract.functions.mintStabilityNFT(
                Web3.to_checksum_address(miner_address),
                task_id,
                grid_impact,
                proof_bytes32
            )
            
            tx_hash = self._estimate_gas_and_send_tx(func)
            
            # Wait for confirmation
            receipt = self._wait_for_tx_receipt(tx_hash)
            
            if receipt and receipt['status'] == 1:
                logger.info(f"NFT minted for {miner_address}, impact: {grid_impact}")
                return tx_hash
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to mint NFT: {e}")
            return None
    
    def submit_pouw_validation_proof(self,
                                   task_id: int,
                                   output_hash: str,
                                   sgx_attestation: bytes,
                                   computation_time: int,
                                   energy_used: int,
                                   grid_impact: int) -> Optional[str]:
        """
        Submit a PoUW validation proof on-chain
        
        Args:
            task_id: ID of the completed task
            output_hash: Hash of computation output
            sgx_attestation: SGX attestation data
            computation_time: Time taken in seconds
            energy_used: Energy consumed in Wh
            grid_impact: Grid stability impact (0-100)
            
        Returns:
            Transaction hash if successful
        """
        try:
            contract = self.contracts['ProofOfUsefulWork']
            
            # Convert output hash to bytes32
            if output_hash.startswith('0x'):
                output_bytes32 = output_hash
            else:
                output_bytes32 = '0x' + output_hash
            
            # Send transaction
            func = contract.functions.submitTaskProof(
                task_id,
                output_bytes32,
                sgx_attestation,
                computation_time,
                energy_used,
                grid_impact
            )
            
            tx_hash = self._estimate_gas_and_send_tx(func)
            
            # Wait for confirmation
            receipt = self._wait_for_tx_receipt(tx_hash)
            
            if receipt and receipt['status'] == 1:
                logger.info(f"Proof submitted for task {task_id}")
                return tx_hash
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to submit proof: {e}")
            return None
    
    def distribute_rewards(self, 
                         recipient: str,
                         amount: int) -> Optional[str]:
        """
        Distribute rewards through the RewardDistribution contract
        
        Args:
            recipient: Address to receive rewards
            amount: Amount in wei
            
        Returns:
            Transaction hash if successful
        """
        try:
            contract = self.contracts['EnergyPool']
            
            # Send transaction
            func = contract.functions.distributeRewards(
                Web3.to_checksum_address(recipient),
                amount
            )
            
            tx_hash = self._estimate_gas_and_send_tx(func)
            
            # Wait for confirmation
            receipt = self._wait_for_tx_receipt(tx_hash)
            
            if receipt and receipt['status'] == 1:
                logger.info(f"Rewards distributed: {amount} wei to {recipient}")
                return tx_hash
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to distribute rewards: {e}")
            return None
    
    def get_resource_token_balance(self, address: str) -> int:
        """
        Get ResourceToken balance for an address
        
        Args:
            address: Address to check
            
        Returns:
            Balance in wei
        """
        try:
            contract = self.contracts['EnergyPool']
            balance = contract.functions.getBalance(
                Web3.to_checksum_address(address)
            ).call()
            return balance
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    def get_nft_balance(self, address: str) -> int:
        """
        Get number of stability NFTs owned by address
        
        Args:
            address: Address to check
            
        Returns:
            Number of NFTs
        """
        try:
            contract = self.contracts['EnergyProfile']
            balance = contract.functions.balanceOf(
                Web3.to_checksum_address(address)
            ).call()
            return balance
        except Exception as e:
            logger.error(f"Failed to get NFT balance: {e}")
            return 0
    
    def register_miner(self, compute_power: int, sgx_key: str) -> Optional[str]:
        """
        Register a new miner in the PoUW system
        
        Args:
            compute_power: Miner's compute power in TFLOPS
            sgx_key: SGX attestation key address
            
        Returns:
            Transaction hash if successful
        """
        try:
            contract = self.contracts['ProofOfUsefulWork']
            
            # Send transaction
            func = contract.functions.registerMiner(
                compute_power,
                Web3.to_checksum_address(sgx_key)
            )
            
            tx_hash = self._estimate_gas_and_send_tx(func)
            
            # Wait for confirmation
            receipt = self._wait_for_tx_receipt(tx_hash)
            
            if receipt and receipt['status'] == 1:
                logger.info(f"Miner registered with {compute_power} TFLOPS")
                return tx_hash
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to register miner: {e}")
            return None
    
    def poll_events(self) -> List[Dict]:
        """
        Poll for new events from all event filters
        
        Returns:
            List of new events
        """
        all_events = []
        
        for event_name, event_filter in self.event_filters.items():
            try:
                new_events = event_filter.get_new_entries()
                for event in new_events:
                    event_data = {
                        'event': event_name,
                        'args': dict(event['args']),
                        'blockNumber': event['blockNumber'],
                        'transactionHash': event['transactionHash'].hex()
                    }
                    all_events.append(event_data)
                    logger.info(f"New event: {event_name} - {event_data['args']}")
            except Exception as e:
                logger.error(f"Error polling {event_name} events: {e}")
        
        return all_events
    
    def get_connection_status(self) -> Dict:
        """Get current connection and contract status"""
        status = {
            'connected': self.w3.is_connected(),
            'chain_id': self.w3.eth.chain_id if self.w3.is_connected() else None,
            'latest_block': self.w3.eth.block_number if self.w3.is_connected() else None,
            'account': self.address,
            'contracts_loaded': list(self.contracts.keys()),
            'event_listeners': list(self.event_filters.keys())
        }
        
        # Add balances if account is configured
        if self.address and self.w3.is_connected():
            try:
                status['eth_balance'] = self.w3.eth.get_balance(self.address)
                status['resource_token_balance'] = self.get_resource_token_balance(self.address)
                status['nft_balance'] = self.get_nft_balance(self.address)
            except:
                pass
        
        return status


# Example integration function for run_full_system.py
def integrate_blockchain_recorder(app, analytics_tracker=None):
    """
    Integration function to be called from run_full_system.py
    
    Args:
        app: Flask application instance
        analytics_tracker: Optional analytics tracker instance
    """
    # Initialize blockchain recorder
    blockchain = BlockchainRecorder()
    
    # Add blockchain status endpoint
    @app.route('/api/blockchain/status')
    def blockchain_status():
        return jsonify(blockchain.get_connection_status())
    
    # Add allocation recording endpoint
    @app.route('/api/blockchain/record_allocation', methods=['POST'])
    def record_allocation():
        try:
            data = request.json
            tx_hash = blockchain.record_allocation_decision(
                data['allocation'],
                data['grid_conditions']
            )
            
            if tx_hash and analytics_tracker:
                # Record in analytics
                analytics_tracker.record_blockchain_transaction({
                    'type': 'allocation',
                    'tx_hash': tx_hash,
                    'timestamp': datetime.now()
                })
            
            return jsonify({
                'success': bool(tx_hash),
                'tx_hash': tx_hash
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
    
    # Add NFT minting endpoint
    @app.route('/api/blockchain/mint_nft', methods=['POST'])
    def mint_nft():
        try:
            data = request.json
            tx_hash = blockchain.mint_dynamic_nft(
                data['miner_address'],
                data['task_id'],
                data['grid_impact'],
                data['proof_hash']
            )
            
            return jsonify({
                'success': bool(tx_hash),
                'tx_hash': tx_hash
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
    
    # Add proof submission endpoint
    @app.route('/api/blockchain/submit_proof', methods=['POST'])
    def submit_proof():
        try:
            data = request.json
            tx_hash = blockchain.submit_pouw_validation_proof(
                data['task_id'],
                data['output_hash'],
                bytes.fromhex(data['sgx_attestation']),
                data['computation_time'],
                data['energy_used'],
                data['grid_impact']
            )
            
            return jsonify({
                'success': bool(tx_hash),
                'tx_hash': tx_hash
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
    
    # Poll events periodically
    def poll_blockchain_events():
        while True:
            try:
                events = blockchain.poll_events()
                if events and analytics_tracker:
                    for event in events:
                        analytics_tracker.record_blockchain_event(event)
            except Exception as e:
                logger.error(f"Event polling error: {e}")
            
            time.sleep(10)  # Poll every 10 seconds
    
    # Start event polling in background
    import threading
    event_thread = threading.Thread(target=poll_blockchain_events, daemon=True)
    event_thread.start()
    
    logger.info("Blockchain integration initialized")
    return blockchain


if __name__ == "__main__":
    # Test the blockchain recorder
    print("Testing Blockchain Recorder...")
    
    # Initialize in read-only mode for testing
    recorder = BlockchainRecorder()
    
    # Check status
    status = recorder.get_connection_status()
    print(f"\nConnection Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test balance checking (replace with actual address)
    test_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD9e"
    balance = recorder.get_resource_token_balance(test_address)
    print(f"\nResource Token Balance for {test_address}: {balance} wei")
    
    nft_balance = recorder.get_nft_balance(test_address)
    print(f"NFT Balance: {nft_balance}")
    
    print("\nBlockchain recorder test complete!")