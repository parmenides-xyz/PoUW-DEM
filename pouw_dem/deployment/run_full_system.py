#!/usr/bin/env python3
"""
Run the complete PoUW-DEM Integration System
Links all components together for production use
"""

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all our components
from deploy_trained_models import ProductionInferenceSystem
from ercot_real_api import ERCOTRealAPI
from grid_api_integration import GridSignalProcessor
from mining_pool_integration import MiningPoolConnector, MiningOptimizer
from analytics_tracker import AnalyticsTracker
from blockchain_integration import BlockchainRecorder, integrate_blockchain_recorder

print("🚀 STARTING POUW-DEM INTEGRATION SYSTEM")
print("="*70)

# Initialize Flask app
app = Flask(__name__)

# Initialize all components
print("\n📦 Initializing Components...")

# 1. Load trained AI models
print("   Loading trained models...")
model_dir = "trained_models"
manifests = [f for f in os.listdir(model_dir) if f.startswith('manifest_')]
if manifests:
    latest_manifest = sorted(manifests)[-1]
    timestamp = latest_manifest.replace('manifest_', '').replace('.json', '')
    inference_system = ProductionInferenceSystem(timestamp)
    inference_system.load_models()
else:
    print("   ⚠️  No trained models found - using default behavior")
    inference_system = None

# 2. Initialize ERCOT API
print("   Connecting to ERCOT...")
ercot_api = ERCOTRealAPI()

# 3. Initialize Grid Signal Processor
print("   Setting up grid monitoring...")
grid_processor = GridSignalProcessor()

# 4. Initialize Mining Pool Connector
print("   Connecting to mining pools...")
mining_connector = MiningPoolConnector()
mining_optimizer = MiningOptimizer()

# 5. Initialize Analytics Tracker
print("   Setting up analytics tracking...")
from analytics_tracker import AnalyticsTracker, AnalyticsCalculator
analytics = AnalyticsTracker()
analytics.calculator = AnalyticsCalculator(analytics.db)

# Start recording market data
import threading
def record_market_data():
    """Record market data every minute"""
    while True:
        try:
            # Get current ERCOT data
            ecrs_data = ercot_api.get_ecrs_offers()
            grid_price = ecrs_data['price_stats']['max'] if ecrs_data and 'price_stats' in ecrs_data else 84
            
            # Record market data
            analytics.record_market_data({
                'energy_price': grid_price,
                'btc_price': 103875,
                'mining_difficulty': 90.67e12,
                'grid_urgency': grid_processor.get_aggregated_signal()['average_urgency'],
                'grid_frequency': 60.0,
                'grid_demand': 45000,
                'renewable_percentage': 0.25
            })
            
            # Sleep for 1 minute
            import time
            time.sleep(60)
        except Exception as e:
            print(f"Error recording market data: {e}")
            time.sleep(60)

# Start background thread
market_thread = threading.Thread(target=record_market_data, daemon=True)
market_thread.start()
print("   ✅ Market data recording started")

# Populate some initial historical data for demo
def populate_demo_data():
    """Populate analytics with demo historical data"""
    import random
    from datetime import datetime, timedelta
    
    print("   Populating demo historical data...")
    
    # Generate 24 hours of historical data
    now = datetime.now()
    for hours_ago in range(24, 0, -1):
        timestamp = now - timedelta(hours=hours_ago)
        
        # Simulate varying grid urgency throughout the day
        hour_of_day = timestamp.hour
        base_urgency = 0.2
        if 6 <= hour_of_day <= 9 or 17 <= hour_of_day <= 21:  # Peak hours
            base_urgency = 0.4
        urgency = base_urgency + random.uniform(-0.1, 0.2)
        
        # Grid price varies with urgency
        base_grid_price = 70
        grid_price = base_grid_price + (urgency * 100) + random.uniform(-10, 20)
        
        # Mining revenue is more stable
        mining_revenue = 60 + random.uniform(-5, 5)
        
        # Record market data
        analytics.record_market_data({
            'energy_price': grid_price,
            'btc_price': 103875 + random.uniform(-1000, 1000),
            'mining_difficulty': 90.67e12,
            'grid_urgency': urgency,
            'grid_frequency': 60.0 + random.uniform(-0.1, 0.1),
            'grid_demand': 45000 + random.uniform(-5000, 10000),
            'renewable_percentage': 0.2 + random.uniform(0, 0.3)
        })
        
        # Simulate decisions for each facility
        for facility in ['MARA_TX_1', 'MARA_TX_2', 'MARA_MT_1']:
            # Decision based on economics
            if grid_price > mining_revenue * 1.3:
                mining_pct = 0.3
                grid_pct = 0.7
                decision = "30% mining, 70% grid"
            elif grid_price > mining_revenue * 1.1:
                mining_pct = 0.6
                grid_pct = 0.4
                decision = "60% mining, 40% grid"
            else:
                mining_pct = 0.85
                grid_pct = 0.15
                decision = "85% mining, 15% grid"
            
            # Record decision
            analytics.record_allocation_decision(
                agent_id=facility,
                action={
                    'action_type': decision,
                    'mining_percent': mining_pct * 100,
                    'grid_support_percent': grid_pct * 100
                },
                reasoning={
                    'mining_profit_per_hour': mining_revenue,
                    'grid_value_per_hour': grid_price,
                    'grid_urgency': urgency,
                    'market_conditions': f"Grid price {grid_price:.2f} vs mining {mining_revenue:.2f}"
                }
            )
            
            # Calculate and record revenue
            capacity = {'MARA_TX_1': 100, 'MARA_TX_2': 80, 'MARA_MT_1': 60}[facility]
            revenue = capacity * (mining_pct * mining_revenue + grid_pct * grid_price) / 24  # Hourly
            
            analytics.record_actual_revenue(
                agent_id=facility,
                mining_revenue=capacity * mining_pct * mining_revenue / 24,
                grid_revenue=capacity * grid_pct * grid_price / 24,
                energy_cost=revenue * 0.1  # Assume 10% energy cost
            )
    
    print("   ✅ Demo data populated")

# Run demo data population
populate_demo_data()

# 6. Load smart contract addresses
print("   Loading smart contracts...")
if os.path.exists('full_deployment.json'):
    with open('full_deployment.json', 'r') as f:
        contracts = json.load(f)
else:
    contracts = {}

# 7. Initialize Blockchain Integration
print("   Setting up blockchain integration...")
try:
    from blockchain_integration import BlockchainRecorder, integrate_blockchain_recorder
    blockchain = BlockchainRecorder()
    print(f"   ✅ Connected to Polygon: {blockchain.is_connected()}")
    
    # Add blockchain endpoints to Flask app
    blockchain_recorder = integrate_blockchain_recorder(app, analytics)
    print("   ✅ Blockchain endpoints added")
except Exception as e:
    print(f"   ⚠️  Blockchain integration unavailable: {e}")
    blockchain = None
    blockchain_recorder = None

print("\n✅ All components initialized!")

# API ENDPOINTS

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard UI"""
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics_dashboard():
    """Serve the analytics dashboard"""
    return render_template('analytics.html')

@app.route('/', methods=['GET'])
def home():
    """System status endpoint"""
    return jsonify({
        'system': 'PoUW-DEM Integration',
        'status': 'operational',
        'components': {
            'ai_models': 'loaded' if inference_system else 'not loaded',
            'ercot_api': 'connected',
            'grid_monitoring': 'active',
            'mining_pools': 'connected',
            'smart_contracts': len(contracts)
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/decision/<facility>', methods=['POST'])
def make_decision(facility):
    """Main decision endpoint - combines all data sources"""
    
    # Get current state from request
    state = request.json.get('state', None)
    if not state:
        # Get real-time data if no state provided
        state = get_current_state()
    
    # 1. Get ERCOT ECRS data
    ecrs_data = ercot_api.get_ecrs_offers()
    
    # 2. Get aggregated grid signal
    grid_signal = grid_processor.get_aggregated_signal()
    
    # 3. Get mining pool stats
    pool_stats = mining_connector.get_all_pool_stats()
    
    # 4. Get AI model recommendation
    if inference_system and facility in inference_system.agents:
        ai_decision = inference_system.make_decision(facility, state)
    else:
        ai_decision = 0  # Default to mining
    
    # 5. Calculate optimal allocation with dynamic logic
    facility_info = get_facility_info(facility)
    
    # More responsive allocation calculation
    grid_urgency = grid_signal['average_urgency']
    mining_revenue = 60  # $/MWh
    grid_revenue = ecrs_data['price_stats']['max'] if ecrs_data and 'price_stats' in ecrs_data else 50
    
    # Dynamic allocation based on economics and urgency
    if grid_urgency > 0.6:
        # High urgency - prioritize grid
        mining_pct = 0.3
        grid_pct = 0.7
    elif grid_urgency > 0.3:
        # Medium urgency - check economics
        price_ratio = grid_revenue / mining_revenue
        if price_ratio > 1.3:
            # Grid pays 30% more - shift allocation
            grid_pct = min(0.6, (price_ratio - 1.0))
            mining_pct = 1.0 - grid_pct
        else:
            # Balanced approach
            mining_pct = 0.6
            grid_pct = 0.4
    elif grid_revenue > mining_revenue * 1.2:
        # Low urgency but good economics
        mining_pct = 0.7
        grid_pct = 0.3
    else:
        # Default - mostly mining with small grid presence
        mining_pct = 0.85
        grid_pct = 0.15
    
    allocation = {
        'mining_percentage': mining_pct,
        'grid_percentage': grid_pct,
        'mining_hashrate': facility_info['hashrate'] * mining_pct,
        'grid_capacity': facility_info['capacity_mw'] * grid_pct,
        'estimated_revenue': {
            'mining': facility_info['capacity_mw'] * mining_pct * mining_revenue,
            'grid': facility_info['capacity_mw'] * grid_pct * grid_revenue,
            'total': facility_info['capacity_mw'] * (mining_pct * mining_revenue + grid_pct * grid_revenue)
        },
        'pool_recommendation': 'Foundry USA'
    }
    
    # 6. Combine all insights
    decision = {
        'facility': facility,
        'timestamp': datetime.now().isoformat(),
        'ai_recommendation': ai_decision,
        'action_name': ['Buy', 'Sell', 'Charge', 'Discharge', 'Nothing', 'Grid Support', 'Hybrid'][ai_decision],
        'allocation': allocation,
        'facility_info': facility_info,
        'data_sources': {
            'ercot': {
                'ecrs_offers': ecrs_data.get('total_offers', 0) if ecrs_data else 0,
                'max_price': ecrs_data['price_stats']['max'] if ecrs_data and 'price_stats' in ecrs_data else 0
            },
            'grid': {
                'urgency': grid_signal['average_urgency'],
                'recommendation': grid_signal['recommendation']
            },
            'mining': {
                'btc_price': 103875,
                'network_difficulty': 90.67e12,
                'best_pool': allocation['pool_recommendation']
            }
        },
        'revenue_analysis': allocation['estimated_revenue']
    }
    
    # Track the decision in analytics
    analytics.record_allocation_decision(
        agent_id=facility,
        action={
            'action_type': f"{allocation['mining_percentage']*100:.0f}% mining, {allocation['grid_percentage']*100:.0f}% grid",
            'mining_percent': allocation['mining_percentage'] * 100,
            'grid_support_percent': allocation['grid_percentage'] * 100
        },
        reasoning={
            'mining_profit_per_hour': mining_revenue * facility_info['capacity_mw'],
            'grid_value_per_hour': grid_revenue * facility_info['capacity_mw'],
            'grid_urgency': grid_urgency,
            'market_conditions': f"Grid urgency {grid_urgency:.2f}, price ratio {grid_revenue/mining_revenue:.2f}"
        }
    )
    
    # Record allocation on blockchain if available
    if blockchain and allocation['grid_percentage'] > 0:
        try:
            # Record allocation on-chain (read-only mode will just log)
            print(f"\n📝 Recording blockchain allocation for {facility}:")
            print(f"   Mining: {allocation['mining_percentage']*100:.0f}%")
            print(f"   Grid: {allocation['grid_percentage']*100:.0f}%")
            
            # Submit blockchain transaction if we have a private key
            if blockchain.address:
                print(f"   🔑 Using wallet: {blockchain.address}")
                
                # Check wallet balance first
                balance = blockchain.w3.eth.get_balance(blockchain.address)
                balance_matic = blockchain.w3.from_wei(balance, 'ether')
                print(f"   💰 Wallet balance: {balance_matic:.4f} MATIC")
                
                if balance > 0:
                    tx_hash = blockchain.record_allocation_decision(
                        facility_id=facility,
                        mining_allocation=int(allocation['mining_percentage'] * 100),
                        grid_allocation=int(allocation['grid_percentage'] * 100),
                        expected_duration=3600  # 1 hour
                    )
                    
                    if tx_hash:
                        decision['blockchain_tx'] = tx_hash
                        print(f"   ✅ Transaction submitted: {tx_hash}")
                        print(f"   🔗 View on Polygonscan: https://polygonscan.com/tx/{tx_hash}")
                else:
                    print(f"   ❌ Insufficient MATIC balance for gas fees")
            else:
                print("   ⚠️  Running in read-only mode (no private key)")
                
        except Exception as e:
            print(f"   ❌ Blockchain recording failed: {e}")
    
    return jsonify(decision)

@app.route('/grid/status', methods=['GET'])
def grid_status():
    """Aggregated grid status from all operators"""
    signal = grid_processor.get_aggregated_signal()
    
    # Add ERCOT real-time data
    ecrs_data = ercot_api.get_ecrs_offers()
    if ecrs_data and 'opportunities' in ecrs_data:
        signal['ercot_opportunities'] = ecrs_data['opportunities']
    
    return jsonify(signal)

@app.route('/mining/pools', methods=['GET'])
def mining_pools():
    """Current mining pool statistics"""
    return jsonify(mining_connector.get_all_pool_stats())

@app.route('/contracts', methods=['GET'])
def smart_contracts():
    """Deployed smart contract addresses"""
    return jsonify(contracts)

@app.route('/optimize/<facility>', methods=['GET'])
def optimize_facility(facility):
    """Get optimization recommendation for specific facility"""
    
    # Get facility info
    facility_info = get_facility_info(facility)
    
    # Get current grid conditions
    grid_signal = grid_processor.get_aggregated_signal()
    ecrs_data = ercot_api.get_ecrs_offers()
    
    # Calculate mining vs grid revenue
    if ecrs_data and ecrs_data.get('price_stats'):
        max_grid_price = ecrs_data['price_stats']['max']
    else:
        max_grid_price = 50  # Default
    
    mining_revenue = facility_info['hashrate'] * 0.072  # $7.20 per 100 TH/s
    grid_revenue = facility_info['capacity_mw'] * max_grid_price
    
    recommendation = {
        'facility': facility,
        'current_mode': facility_info.get('current_mode', 'mining'),
        'recommendation': 'grid_support' if grid_revenue > mining_revenue * 1.5 else 'mining',
        'revenue_comparison': {
            'mining_per_hour': mining_revenue,
            'grid_per_hour': grid_revenue,
            'premium': ((grid_revenue / mining_revenue) - 1) * 100 if mining_revenue > 0 else 0
        },
        'action_required': grid_revenue > mining_revenue * 1.5
    }
    
    return jsonify(recommendation)

@app.route('/analytics/performance', methods=['GET'])
def get_performance():
    """Get analytics performance data"""
    hours = int(request.args.get('hours', 24))
    performance = analytics.calculator.get_performance_summary(hours=hours)
    return jsonify(performance)

@app.route('/analytics/profit-history', methods=['GET'])
def get_profit_history():
    """Get cumulative profit history"""
    hours = int(request.args.get('hours', 24))
    history = analytics.calculator.get_cumulative_profit(hours=hours)
    return jsonify(history)

@app.route('/analytics/hourly-patterns', methods=['GET'])
def get_hourly_patterns():
    """Get hourly performance patterns"""
    patterns = analytics.calculator.get_hourly_patterns()
    return jsonify(patterns)

# Helper functions

def get_current_state():
    """Build current state vector from real-time data"""
    grid_signal = grid_processor.get_aggregated_signal()
    
    # 12-dimensional state vector matching our trained models
    state = np.zeros(12)
    state[6] = grid_signal['average_urgency']  # Grid urgency
    state[7] = 0.045  # Energy price (example)
    state[8] = 0.5    # Mining difficulty (normalized)
    state[9] = 1.0    # Compute power available
    state[10] = grid_signal['max_urgency']  # Task priority
    state[11] = 1.0   # Task reward
    
    return state.tolist()

def get_facility_info(facility_name):
    """Get facility information"""
    facilities = {
        'MARA_TX_1': {
            'capacity_mw': 100,
            'hashrate': 3000,  # TH/s
            'location': 'Texas'
        },
        'MARA_TX_2': {
            'capacity_mw': 80,
            'hashrate': 2400,
            'location': 'Texas'
        },
        'MARA_MT_1': {
            'capacity_mw': 60,
            'hashrate': 1800,
            'location': 'Montana'
        }
    }
    
    return facilities.get(facility_name, {
        'capacity_mw': 100,
        'hashrate': 3000,
        'location': 'Unknown'
    })

def run_full_system(host='0.0.0.0', port=5001, debug=True):
    """
    Run the full PoUW-DEM integration system.
    
    Args:
        host: Host address to bind to
        port: Port number to run on
        debug: Whether to run in debug mode
    """
    global app, inference_system, ercot_api, grid_processor, pool_connector, analytics, blockchain
    
    # Run the Flask application
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 PoUW-DEM Integration System Ready!")
    print("="*70)
    
    print("\n📡 Available Endpoints:")
    print("   GET  /                      - System status")
    print("   GET  /dashboard            - Main operations dashboard")
    print("   GET  /analytics            - Analytics & performance dashboard")
    print("   POST /decision/<facility>   - Get AI decision for facility")
    print("   GET  /grid/status          - Real-time grid conditions")
    print("   GET  /mining/pools         - Mining pool statistics")
    print("   GET  /contracts            - Smart contract addresses")
    print("   GET  /optimize/<facility>  - Optimization recommendation")
    print("   GET  /analytics/performance - Historical performance data")
    print("   GET  /analytics/profit-history - Cumulative profit tracking")
    
    if blockchain:
        print("\n🔗 Blockchain Endpoints:")
        print("   GET  /api/blockchain/status - Blockchain connection status")
        print("   POST /api/blockchain/record_allocation - Record allocation on-chain")
        print("   POST /api/blockchain/mint_nft - Mint grid support NFT")
        print("   POST /api/blockchain/submit_proof - Submit PoUW proof")
    
    print("\n💡 Example Usage:")
    print("   curl http://localhost:5000/decision/MARA_TX_1")
    print("   curl http://localhost:5000/grid/status")
    print("   curl http://localhost:5000/optimize/MARA_TX_1")
    
    print("\n🚀 Starting server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)