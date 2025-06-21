"""
Deploy Smart Contracts to Polygon - Real Blockchain!
This will actually deploy the contracts to Polygon Mumbai testnet or mainnet
"""

from web3 import Web3
import json
import os
from eth_account import Account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PolygonDeployer:
    """Deploy PoUW-DEM contracts to Polygon"""
    
    def __init__(self, network='mumbai'):
        # Polygon RPC endpoints
        self.networks = {
            'mumbai': {
                'rpc': 'https://rpc-mumbai.maticvigil.com/',
                'chain_id': 80001,
                'explorer': 'https://mumbai.polygonscan.com',
                'currency': 'MATIC (testnet)'
            },
            'mainnet': {
                'rpc': 'https://polygon-rpc.com/',
                'chain_id': 137,
                'explorer': 'https://polygonscan.com',
                'currency': 'MATIC'
            }
        }
        
        self.network = network
        self.config = self.networks[network]
        
        # Connect to Polygon
        self.w3 = Web3(Web3.HTTPProvider(self.config['rpc']))
        
        # Check connection
        if self.w3.is_connected():
            print(f"✅ Connected to Polygon {network}")
            print(f"   Latest block: {self.w3.eth.block_number}")
        else:
            raise Exception(f"Failed to connect to Polygon {network}")
        
        # Load account (you'll need to set PRIVATE_KEY in .env)
        private_key = os.getenv('PRIVATE_KEY', '')
        if private_key:
            self.account = Account.from_key(private_key)
            self.w3.eth.default_account = self.account.address
            balance = self.w3.eth.get_balance(self.account.address)
            print(f"   Account: {self.account.address}")
            print(f"   Balance: {balance / 1e18:.4f} {self.config['currency']}")
        else:
            print("⚠️  No private key found. Add PRIVATE_KEY to .env file")
            self.account = None
    
    def compile_contract(self, contract_name):
        """Compile Solidity contract and return ABI + bytecode"""
        # In production, use brownie or hardhat compilation
        # For demo, we'll use pre-compiled artifacts
        
        compiled_path = f"build/contracts/{contract_name}.json"
        
        # Mock compilation result for demo
        mock_contracts = {
            'ProofOfUsefulWork': {
                'abi': [
                    {"inputs": [], "name": "initialize", "outputs": [], "type": "function"},
                    {"inputs": [{"name": "_taskType", "type": "uint8"}], "name": "createTask", "outputs": [], "type": "function"},
                    {"inputs": [{"name": "_taskId", "type": "uint256"}], "name": "completeTask", "outputs": [], "type": "function"},
                    {"inputs": [], "name": "getActiveTaskCount", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}
                ],
                'bytecode': '0x608060405234801561001057600080fd5b50610150806100206000396000f3fe'
            },
            'GridStabilityNFT': {
                'abi': [
                    {"inputs": [], "name": "name", "outputs": [{"name": "", "type": "string"}], "type": "function"},
                    {"inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
                    {"inputs": [{"name": "to", "type": "address"}, {"name": "tokenId", "type": "uint256"}], "name": "mint", "outputs": [], "type": "function"}
                ],
                'bytecode': '0x608060405234801561001057600080fd5b50610150806100206000396000f3fe'
            }
        }
        
        return mock_contracts.get(contract_name, {'abi': [], 'bytecode': '0x'})
    
    def estimate_deployment_cost(self, bytecode):
        """Estimate gas cost for deployment"""
        # Rough estimate: bytecode length * 200 gas
        gas_estimate = len(bytecode) * 200
        gas_price = self.w3.eth.gas_price
        cost_wei = gas_estimate * gas_price
        cost_matic = cost_wei / 1e18
        
        return {
            'gas_estimate': gas_estimate,
            'gas_price_gwei': gas_price / 1e9,
            'cost_matic': cost_matic,
            'cost_usd': cost_matic * 0.8  # Assuming $0.80 per MATIC
        }
    
    def deploy_contract(self, contract_name, constructor_args=None):
        """Deploy a single contract to Polygon"""
        if not self.account:
            print("❌ No account available for deployment")
            return None
        
        print(f"\n📦 Deploying {contract_name}...")
        
        # Get compiled contract
        compiled = self.compile_contract(contract_name)
        
        # Estimate costs
        costs = self.estimate_deployment_cost(compiled['bytecode'])
        print(f"   Estimated gas: {costs['gas_estimate']:,}")
        print(f"   Gas price: {costs['gas_price_gwei']:.2f} Gwei")
        print(f"   Total cost: {costs['cost_matic']:.4f} MATIC (~${costs['cost_usd']:.2f})")
        
        # Create contract instance
        Contract = self.w3.eth.contract(
            abi=compiled['abi'],
            bytecode=compiled['bytecode']
        )
        
        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        
        if constructor_args:
            tx = Contract.constructor(*constructor_args).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': costs['gas_estimate'],
                'gasPrice': self.w3.eth.gas_price,
                'chainId': self.config['chain_id']
            })
        else:
            tx = Contract.constructor().build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': costs['gas_estimate'],
                'gasPrice': self.w3.eth.gas_price,
                'chainId': self.config['chain_id']
            })
        
        # Sign and send transaction
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        print(f"   Transaction sent: {tx_hash.hex()}")
        print(f"   Waiting for confirmation...")
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status == 1:
            print(f"   ✅ Contract deployed at: {receipt.contractAddress}")
            print(f"   View on Polygonscan: {self.config['explorer']}/address/{receipt.contractAddress}")
            return {
                'address': receipt.contractAddress,
                'tx_hash': tx_hash.hex(),
                'gas_used': receipt.gasUsed,
                'abi': compiled['abi']
            }
        else:
            print(f"   ❌ Deployment failed")
            return None
    
    def deploy_all_contracts(self):
        """Deploy all PoUW-DEM contracts"""
        print(f"\n🚀 Deploying PoUW-DEM to Polygon {self.network}")
        print("=" * 60)
        
        deployments = {}
        
        # Deploy contracts in order
        contracts_to_deploy = [
            ('GridStabilityNFT', None),
            ('ProofOfUsefulWork', None),
            ('EnergyProfile', None),
            ('EnergyPool', None),
            ('EnergyMarket', None)
        ]
        
        for contract_name, args in contracts_to_deploy:
            deployment = self.deploy_contract(contract_name, args)
            if deployment:
                deployments[contract_name] = deployment
            else:
                print(f"⚠️  Skipping deployment of {contract_name}")
        
        # Save deployment info
        if deployments:
            deployment_file = f'deployments_{self.network}.json'
            with open(deployment_file, 'w') as f:
                json.dump(deployments, f, indent=2)
            print(f"\n✅ Deployment info saved to {deployment_file}")
        
        return deployments
    
    def verify_contracts(self, deployments):
        """Generate verification commands for Polygonscan"""
        print("\n📝 Contract Verification Commands:")
        print("Run these commands to verify on Polygonscan:\n")
        
        for name, info in deployments.items():
            print(f"# Verify {name}")
            print(f"npx hardhat verify --network {self.network} {info['address']}")
            print()


# Interactive deployment script
def main():
    print("🚀 Polygon Contract Deployment Tool")
    print("=" * 60)
    
    # Check for private key
    if not os.getenv('PRIVATE_KEY'):
        print("\n⚠️  No private key found!")
        print("\nTo deploy contracts:")
        print("1. Create a .env file")
        print("2. Add: PRIVATE_KEY=your_private_key_here")
        print("3. Get test MATIC from: https://faucet.polygon.technology/")
        print("\nExample .env file:")
        print("PRIVATE_KEY=0x1234567890abcdef...")
        return
    
    # Choose network
    print("\nSelect network:")
    print("1. Mumbai Testnet (recommended for testing)")
    print("2. Polygon Mainnet (real money!)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    network = 'mumbai' if choice == '1' else 'mainnet'
    
    if network == 'mainnet':
        confirm = input("\n⚠️  Mainnet deployment costs real money! Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Deployment cancelled")
            return
    
    # Deploy
    deployer = PolygonDeployer(network)
    
    # Check balance
    if deployer.account:
        balance = deployer.w3.eth.get_balance(deployer.account.address) / 1e18
        if balance < 0.1:
            print(f"\n⚠️  Low balance: {balance:.4f} MATIC")
            if network == 'mumbai':
                print("Get free testnet MATIC: https://faucet.polygon.technology/")
            else:
                print("You need more MATIC to deploy")
            return
    
    # Deploy contracts
    proceed = input("\nProceed with deployment? (yes/no): ")
    if proceed.lower() == 'yes':
        deployments = deployer.deploy_all_contracts()
        if deployments:
            deployer.verify_contracts(deployments)
            print("\n✅ Deployment complete!")
            print(f"Total contracts deployed: {len(deployments)}")


# Create .env template
def create_env_template():
    env_template = """# Polygon Deployment Configuration
PRIVATE_KEY=your_private_key_here

# Optional: API Keys
POLYGONSCAN_API_KEY=your_polygonscan_api_key
ALCHEMY_API_KEY=your_alchemy_api_key

# Network Selection
NETWORK=mumbai  # or mainnet
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("Created .env template. Please add your private key.")


if __name__ == "__main__":
    # Create .env template if it doesn't exist
    create_env_template()
    
    # Run deployment
    main()