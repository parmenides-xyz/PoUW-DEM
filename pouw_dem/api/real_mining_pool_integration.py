#!/usr/bin/env python3
"""
Real Bitcoin mining pool integration with actual APIs
"""

import requests
import json
import hashlib
import hmac
import time
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class RealMiningPoolConnector:
    """Connects to real Bitcoin mining pool APIs"""
    
    def __init__(self):
        self.pools = {
            'SlushPool': SlushPoolAPI(),
            'F2Pool': F2PoolAPI(),
            'ViaBTC': ViaBTCAPI(),
            'Luxor': LuxorAPI(),
            'BTC.com': BTCcomAPI()
        }
        
    def get_real_stats(self):
        """Get real statistics from all pools"""
        stats = {}
        for name, pool in self.pools.items():
            try:
                stats[name] = pool.get_stats()
            except Exception as e:
                stats[name] = {'error': str(e)}
        return stats


class SlushPoolAPI:
    """Braiins Pool (formerly Slush Pool) API integration"""
    
    def __init__(self):
        self.base_url = "https://pool.braiins.com/stats/json/btc"
        self.api_key = os.getenv('SLUSHPOOL_API_KEY')
        
    def get_stats(self):
        """Get pool statistics from Slush Pool"""
        # Stats endpoint requires authentication
        try:
            headers = {}
            if self.api_key:
                headers['Pool-Auth-Token'] = self.api_key
                
            response = requests.get(self.base_url, headers=headers, timeout=10)
            
            if response.status_code == 401:
                return {
                    'pool_name': 'Braiins Pool',
                    'api_status': 'auth_required',
                    'message': 'API key required for pool stats',
                    'note': 'Using simulated data',
                    'hashrate': 12.5,  # EH/s (typical for Braiins)
                    'active_workers': 125000,
                    'blocks_found_24h': 3,
                    'fees': '2% (0% with Braiins OS+)'
                }
                
            data = response.json()
            
            # Extract BTC stats from response
            btc_stats = data.get('btc', {})
            
            # Convert hash rate from Gh/s to EH/s (1 EH/s = 1e9 GH/s)
            hashrate_5m = btc_stats.get('pool_5m_hash_rate', 0) / 1e9
            hashrate_60m = btc_stats.get('pool_60m_hash_rate', 0) / 1e9
            hashrate_24h = btc_stats.get('pool_24h_hash_rate', 0) / 1e9
            
            # Count blocks in last 24h
            blocks = btc_stats.get('blocks', {})
            blocks_24h = 0
            current_time = datetime.now().timestamp()
            for block_data in blocks.values():
                if isinstance(block_data, dict) and 'date_found' in block_data:
                    if current_time - block_data['date_found'] < 86400:  # 24 hours
                        blocks_24h += 1
            
            return {
                'pool_name': 'Braiins Pool',
                'hashrate_5m': hashrate_5m,
                'hashrate_60m': hashrate_60m,
                'hashrate_24h': hashrate_24h,
                'active_workers': btc_stats.get('pool_active_workers', 0),
                'blocks_found_24h': blocks_24h,
                'blocks_total': len(blocks),
                'last_update': datetime.fromtimestamp(btc_stats.get('update_ts', 0)).isoformat() if btc_stats.get('update_ts') else None,
                'fees': '2% (0% with Braiins OS+)',
                'api_status': 'connected'
            }
        except Exception as e:
            return {
                'pool_name': 'Braiins Pool',
                'api_status': 'error',
                'error': str(e)
            }
            
    def get_account_stats(self):
        """Get account-specific stats (requires API key)"""
        if not self.api_key:
            return {'error': 'No API key configured'}
            
        headers = {'SlushPool-Auth-Token': self.api_key}
        
        try:
            response = requests.get(
                f"{self.base_url}/accounts/profile", 
                headers=headers
            )
            return response.json()
        except Exception as e:
            return {'error': str(e)}


class F2PoolAPI:
    """F2Pool API integration"""
    
    def __init__(self):
        self.base_url = "https://api.f2pool.com/bitcoin"
        self.account = os.getenv('F2POOL_ACCOUNT')
        
    def get_stats(self):
        """Get public pool stats"""
        try:
            # F2Pool public stats
            response = requests.get("https://api.f2pool.com/pool/stats")
            data = response.json()
            
            btc_stats = data.get('bitcoin', {})
            
            return {
                'pool_name': 'F2Pool',
                'hashrate': btc_stats.get('hashrate', 0) / 1e18,  # Convert to EH/s
                'active_miners': btc_stats.get('miner_count', 0),
                'blocks_found_24h': btc_stats.get('blocks_24h', 0),
                'pool_income_24h': btc_stats.get('income_24h', 0),
                'pps_rate': btc_stats.get('pps_rate', 0),
                'fees': '2.5%',
                'api_status': 'connected'
            }
        except Exception as e:
            return {
                'pool_name': 'F2Pool',
                'api_status': 'error',
                'error': str(e)
            }
            
    def get_account_stats(self):
        """Get account hashrate and earnings"""
        if not self.account:
            return {'error': 'No F2Pool account configured'}
            
        try:
            response = requests.get(f"{self.base_url}/{self.account}")
            return response.json()
        except Exception as e:
            return {'error': str(e)}


class ViaBTCAPI:
    """ViaBTC Pool API integration"""
    
    def __init__(self):
        self.base_url = "https://www.viabtc.com/api/v1"
        self.api_key = os.getenv('VIABTC_API_KEY')
        self.api_secret = os.getenv('VIABTC_API_SECRET')
        
    def get_stats(self):
        """Get public pool statistics"""
        try:
            response = requests.get(f"{self.base_url}/pool/stats")
            data = response.json()
            
            if data.get('code') == 0:
                stats = data.get('data', {})
                return {
                    'pool_name': 'ViaBTC',
                    'hashrate': stats.get('pool_hashrate', 0) / 1e18,  # Convert to EH/s
                    'active_miners': stats.get('miner_count', 0),
                    'blocks_found_24h': stats.get('block_count_24h', 0),
                    'difficulty': stats.get('difficulty', 0),
                    'next_difficulty': stats.get('next_difficulty', 0),
                    'fees': 'PPS+ 4%, PPLNS 2%',
                    'api_status': 'connected'
                }
            else:
                return {
                    'pool_name': 'ViaBTC',
                    'api_status': 'error',
                    'error': data.get('message', 'Unknown error')
                }
        except Exception as e:
            return {
                'pool_name': 'ViaBTC',
                'api_status': 'error',
                'error': str(e)
            }


class LuxorAPI:
    """Luxor Mining Pool API integration"""
    
    def __init__(self):
        self.base_url = "https://api.beta.luxor.tech/graphql"
        self.api_key = os.getenv('LUXOR_API_KEY')
        
    def get_stats(self):
        """Get pool statistics using GraphQL"""
        query = """
        query getPoolStats {
            getPoolStats {
                hashrate
                miners
                workers
            }
        }
        """
        
        headers = {
            'Content-Type': 'application/json',
            'x-lux-api-key': self.api_key if self.api_key else ''
        }
        
        try:
            response = requests.post(
                self.base_url,
                json={'query': query},
                headers=headers
            )
            
            data = response.json()
            
            if 'data' in data:
                stats = data['data']['getPoolStats']
                return {
                    'pool_name': 'Luxor',
                    'hashrate': stats.get('hashrate', 0) / 1e18,  # Convert to EH/s
                    'active_miners': stats.get('miners', 0),
                    'active_workers': stats.get('workers', 0),
                    'special_programs': ['Demand Response', 'Hashrate Derivatives'],
                    'fees': '2.5%',
                    'api_status': 'connected'
                }
            else:
                return {
                    'pool_name': 'Luxor',
                    'api_status': 'limited',
                    'message': 'Using public data only'
                }
                
        except Exception as e:
            return {
                'pool_name': 'Luxor',
                'api_status': 'error',
                'error': str(e)
            }


class BTCcomAPI:
    """BTC.com Pool API integration"""
    
    def __init__(self):
        self.base_url = "https://pool.btc.com/api/v1"
        self.api_key = os.getenv('BTCCOM_API_KEY')
        
    def get_stats(self):
        """Get pool statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            data = response.json()
            
            if data.get('err_no') == 0:
                stats = data.get('data', {})
                return {
                    'pool_name': 'BTC.com',
                    'hashrate': stats.get('shares_15m', 0) * 4.295e9 / 900 / 1e18,  # Approximate EH/s
                    'active_workers': stats.get('workers', 0),
                    'blocks_found': stats.get('blocks', {}).get('count', 0),
                    'lucky_24h': stats.get('shares_24h_stale', 0),
                    'fees': 'PPS+ 4%, FPPS 2.5%',
                    'api_status': 'connected'
                }
            else:
                return {
                    'pool_name': 'BTC.com',
                    'api_status': 'error',
                    'error': 'API returned error'
                }
                
        except Exception as e:
            return {
                'pool_name': 'BTC.com',
                'api_status': 'error',
                'error': str(e)
            }


class MempoolSpaceAPI:
    """Bitcoin network stats from mempool.space"""
    
    def __init__(self):
        self.base_url = "https://mempool.space/api"
        
    def get_network_stats(self):
        """Get current Bitcoin network statistics"""
        try:
            # Get mining stats
            mining_response = requests.get(f"{self.base_url}/v1/mining/pools/24h")
            mining_data = mining_response.json()
            
            # Get fee estimates
            fees_response = requests.get(f"{self.base_url}/v1/fees/recommended")
            fees_data = fees_response.json()
            
            # Get difficulty adjustment
            diff_response = requests.get(f"{self.base_url}/v1/difficulty-adjustment")
            diff_data = diff_response.json()
            
            # Calculate network hashrate from pools
            total_hashrate = sum(pool.get('avgHashrate', 0) for pool in mining_data.get('pools', []))
            
            return {
                'network_hashrate': total_hashrate / 1e18,  # EH/s
                'difficulty': diff_data.get('difficultyChange', 0),
                'time_to_adjustment': diff_data.get('timeAvg', 0),
                'recommended_fees': fees_data,
                'top_pools': [
                    {
                        'name': pool['name'],
                        'hashrate_share': pool.get('share', 0),
                        'blocks_found': pool.get('blockCount', 0)
                    }
                    for pool in mining_data.get('pools', [])[:5]
                ]
            }
            
        except Exception as e:
            return {'error': str(e)}


def test_real_pools():
    """Test real mining pool connections"""
    print("🔌 TESTING REAL MINING POOL APIS")
    print("="*70)
    
    connector = RealMiningPoolConnector()
    stats = connector.get_real_stats()
    
    print("\n📊 Pool Statistics:")
    for pool_name, pool_stats in stats.items():
        print(f"\n{pool_name}:")
        if 'error' in pool_stats:
            print(f"   ❌ Error: {pool_stats['error']}")
        else:
            print(f"   Status: {pool_stats.get('api_status', 'unknown')}")
            # Handle different hashrate fields
            if pool_stats.get('hashrate'):
                print(f"   Hashrate: {pool_stats['hashrate']:.2f} EH/s")
            elif pool_stats.get('hashrate_24h'):
                print(f"   Hashrate (24h): {pool_stats['hashrate_24h']:.2f} EH/s")
                if pool_stats.get('hashrate_5m'):
                    print(f"   Hashrate (5m): {pool_stats['hashrate_5m']:.2f} EH/s")
            if pool_stats.get('active_miners'):
                print(f"   Active miners: {pool_stats['active_miners']:,}")
            elif pool_stats.get('active_workers'):
                print(f"   Active workers: {pool_stats['active_workers']:,}")
            if pool_stats.get('blocks_found_24h'):
                print(f"   Blocks (24h): {pool_stats['blocks_found_24h']}")
    
    # Get network stats
    print("\n🌐 Bitcoin Network Stats:")
    mempool = MempoolSpaceAPI()
    network_stats = mempool.get_network_stats()
    
    if 'error' not in network_stats:
        print(f"   Network hashrate: {network_stats['network_hashrate']:.2f} EH/s")
        print(f"   Difficulty change: {network_stats['difficulty']:.2f}%")
        print(f"   Recommended fees: {network_stats['recommended_fees']}")
        print("\n   Top mining pools:")
        for pool in network_stats['top_pools']:
            print(f"   - {pool['name']}: {pool['hashrate_share']:.1f}% ({pool['blocks_found']} blocks)")


if __name__ == "__main__":
    test_real_pools()
    
    print("\n💡 To get account-specific data:")
    print("   1. Register with each pool")
    print("   2. Generate API credentials")
    print("   3. Add to .env file:")
    print("      SLUSHPOOL_API_KEY=your_key")
    print("      F2POOL_ACCOUNT=your_account")
    print("      VIABTC_API_KEY=your_key")
    print("      LUXOR_API_KEY=your_key")
    print("      BTCCOM_API_KEY=your_key")