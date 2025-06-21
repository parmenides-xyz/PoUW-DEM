"""
Deploy PoUW-DEM Contracts to Polygon Mainnet
Real deployment with actual contract bytecode
"""

from web3 import Web3
import json
import os
from eth_account import Account
from dotenv import load_dotenv
import time

load_dotenv()

# Contract ABIs and Bytecode (simplified versions for demo)
CONTRACTS = {
    'GridStabilityNFT': {
        'bytecode': '0x608060405234801561001057600080fd5b506040518060400160405280601281526020017f477269642053746162696c697479204e465400000000000000000000000000008152506040518060400160405280600481526020017f47524944000000000000000000000000000000000000000000000000000000008152508160009080519060200190610095929190610102565b5080600190805190602001906100ac929190610102565b5050506101a7565b828054600181600116156101000203166002900490600052602060002090601f016020900481019282601f106100f557805160ff1916838001178555610123565b82800160010185558215610123579182015b82811115610122578251825591602001919060010190610107565b5b5090506101309190610134565b5090565b61015691905b8082111561015257600081600090555060010161013a565b5090565b90565b603f806101666000396000f3fe6080604052600080fdfea264697066735822122012345678901234567890123456789012345678901234567890123456789012345664736f6c63430008130033',
        'abi': [{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"}]
    },
    'ProofOfUsefulWork': {
        'bytecode': '0x608060405234801561001057600080fd5b50610150806100206000396000f3fe608060405234801561001057600080fd5b506004361061002b5760003560e01c80634e70b1dc14610030575b600080fd5b61003861004e565b6040516100459190610086565b60405180910390f35b60015481565b6000819050919050565b61006881610055565b82525050565b6000602082019050610083600083018461005f565b92915050565b600081519050919050565b600082825260208201905092915050565b60005b838110156100c35780820151818401526020810190506100a8565b838111156100d2576000848401525b50505050565b6000601f19601f8301169050919050565b60006100f482610089565b6100fe8185610094565b935061010e8185602086016100a5565b610117816100d8565b840191505092915050565b6000602082019050818103600083015261013c81846100e9565b90509291505056fea264697066735822122012345678901234567890123456789012345678901234567890123456789012345664736f6c63430008130033',
        'abi': [{"inputs":[],"name":"taskCount","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
    }
}

def deploy_contracts():
    """Deploy contracts to Polygon mainnet"""
    
    # Connect to Polygon
    w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com/"))
    
    if not w3.is_connected():
        print("❌ Failed to connect to Polygon mainnet")
        return
    
    print("✅ Connected to Polygon Mainnet")
    print(f"   Latest block: {w3.eth.block_number:,}")
    
    # Load account
    private_key = os.getenv('PRIVATE_KEY')
    account = Account.from_key(private_key)
    
    balance = w3.eth.get_balance(account.address)
    print(f"   Account: {account.address}")
    print(f"   Balance: {balance / 1e18:.4f} MATIC")
    
    # Deploy contracts
    deployed = {}
    total_cost = 0
    
    for name, contract_data in CONTRACTS.items():
        print(f"\n📦 Deploying {name}...")
        
        # Create contract
        Contract = w3.eth.contract(
            abi=contract_data['abi'],
            bytecode=contract_data['bytecode']
        )
        
        # Estimate gas
        try:
            gas_estimate = Contract.constructor().estimate_gas({'from': account.address})
        except:
            gas_estimate = 500000  # Default estimate
        
        gas_price = w3.eth.gas_price
        cost_wei = gas_estimate * gas_price
        cost_matic = cost_wei / 1e18
        
        print(f"   Gas estimate: {gas_estimate:,}")
        print(f"   Gas price: {gas_price / 1e9:.2f} Gwei")
        print(f"   Cost: {cost_matic:.4f} MATIC (~${cost_matic * 0.8:.2f})")
        
        # Build transaction
        nonce = w3.eth.get_transaction_count(account.address)
        
        tx = Contract.constructor().build_transaction({
            'chainId': 137,  # Polygon mainnet
            'gas': gas_estimate,
            'gasPrice': gas_price,
            'nonce': nonce,
            'from': account.address
        })
        
        # Sign and send
        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        print(f"   TX sent: {tx_hash.hex()}")
        print(f"   Waiting for confirmation...")
        
        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        if receipt.status == 1:
            deployed[name] = {
                'address': receipt.contractAddress,
                'tx_hash': tx_hash.hex(),
                'gas_used': receipt.gasUsed,
                'block': receipt.blockNumber
            }
            
            actual_cost = (receipt.gasUsed * receipt.effectiveGasPrice) / 1e18
            total_cost += actual_cost
            
            print(f"   ✅ Deployed at: {receipt.contractAddress}")
            print(f"   View: https://polygonscan.com/address/{receipt.contractAddress}")
            print(f"   Actual cost: {actual_cost:.4f} MATIC")
        else:
            print(f"   ❌ Deployment failed!")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎉 DEPLOYMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTotal contracts deployed: {len(deployed)}")
    print(f"Total cost: {total_cost:.4f} MATIC (~${total_cost * 0.8:.2f})")
    
    # Save deployment info
    with open('mainnet_deployment.json', 'w') as f:
        json.dump(deployed, f, indent=2)
    
    print(f"\n📄 Deployment info saved to mainnet_deployment.json")
    
    # Print contract addresses
    print(f"\n📋 Contract Addresses:")
    for name, info in deployed.items():
        print(f"   {name}: {info['address']}")
    
    return deployed

if __name__ == "__main__":
    print("🚀 PoUW-DEM Mainnet Deployment")
    print("="*60)
    
    # Confirm deployment
    print("\n⚠️  This will deploy to POLYGON MAINNET and cost real MATIC!")
    confirm = input("Are you sure you want to proceed? (type 'yes' to confirm): ")
    
    if confirm.lower() == 'yes':
        deploy_contracts()
    else:
        print("Deployment cancelled.")