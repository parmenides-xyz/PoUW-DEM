#!/usr/bin/env python3
"""
Analytics Tracking System for PoUW-DEM Integration
Tracks historical data, calculates performance metrics, and provides API endpoints
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from flask import Flask, jsonify, request
import threading
import time
from contextlib import contextmanager

class AnalyticsDatabase:
    """Manages SQLite database for historical data storage"""
    
    def __init__(self, db_path: str = "pouw_dem_analytics.db"):
        self.db_path = db_path
        self.init_database()
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Hourly price data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hourly_prices (
                    timestamp INTEGER PRIMARY KEY,
                    energy_price REAL NOT NULL,
                    btc_price REAL NOT NULL,
                    mining_difficulty REAL NOT NULL,
                    grid_stress REAL NOT NULL,
                    renewable_percent REAL NOT NULL,
                    frequency REAL NOT NULL
                )
            """)
            
            # Allocation decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS allocation_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    mining_percent REAL NOT NULL,
                    grid_support_percent REAL NOT NULL,
                    expected_profit REAL NOT NULL,
                    mining_profit REAL NOT NULL,
                    grid_value REAL NOT NULL,
                    decision_factors TEXT
                )
            """)
            
            # Actual revenue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actual_revenue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    mining_revenue REAL NOT NULL,
                    grid_revenue REAL NOT NULL,
                    total_revenue REAL NOT NULL,
                    energy_cost REAL NOT NULL,
                    net_profit REAL NOT NULL
                )
            """)
            
            # Grid urgency events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS grid_urgency_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    urgency_level TEXT NOT NULL,
                    grid_stress REAL NOT NULL,
                    response_capacity_mw REAL NOT NULL,
                    participating_agents TEXT NOT NULL
                )
            """)
            
            # Create indices for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON hourly_prices(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON allocation_decisions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_revenue_timestamp ON actual_revenue(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_urgency_timestamp ON grid_urgency_events(timestamp)")


class AnalyticsTracker:
    """Tracks and records analytics data from the PoUW-DEM system"""
    
    def __init__(self, db_path: str = "pouw_dem_analytics.db"):
        self.db = AnalyticsDatabase(db_path)
        self.is_recording = False
        self.recording_thread = None
        
    def start_recording(self):
        """Start automatic data recording"""
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
    def stop_recording(self):
        """Stop automatic data recording"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
    def _recording_loop(self):
        """Background thread for periodic data recording"""
        while self.is_recording:
            try:
                # Record current market data every 5 minutes
                self.record_market_data()
                time.sleep(300)  # 5 minutes
            except Exception as e:
                print(f"Error in recording loop: {e}")
                time.sleep(60)  # Wait a minute before retry
    
    def record_market_data(self, market_data: Optional[Dict] = None) -> int:
        """Record current market conditions"""
        if market_data is None:
            # Get real market data (would come from intelligent_mining_agent)
            from agents.intelligent_mining_agent import IntelligentMiningAgent
            agent = IntelligentMiningAgent("temp", 100)
            market_data = agent.get_market_data()
            
        timestamp = int(datetime.now().timestamp())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO hourly_prices 
                (timestamp, energy_price, btc_price, mining_difficulty, 
                 grid_stress, renewable_percent, frequency)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                market_data.get('energy_price', 0.05),
                market_data.get('btc_price', 40000),
                market_data.get('mining_difficulty', 65e12),
                market_data.get('grid_stress', 0.5),
                market_data.get('renewable_percent', 0.2),
                market_data.get('frequency', 60.0)
            ))
            
        return timestamp
    
    def record_allocation_decision(self, agent_id: str, action: Dict, 
                                 reasoning: Dict) -> int:
        """Record an allocation decision made by an agent"""
        timestamp = int(datetime.now().timestamp())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO allocation_decisions
                (timestamp, agent_id, action_type, mining_percent, 
                 grid_support_percent, expected_profit, mining_profit,
                 grid_value, decision_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                agent_id,
                action.get('action_type', 'UNKNOWN'),
                action.get('mining_percent', 0),
                action.get('grid_support_percent', 0),
                reasoning.get('mining_profit_per_hour', 0) * (action.get('mining_percent', 0) / 100) +
                reasoning.get('grid_value_per_hour', 0) * (action.get('grid_support_percent', 0) / 100),
                reasoning.get('mining_profit_per_hour', 0),
                reasoning.get('grid_value_per_hour', 0),
                json.dumps(reasoning.get('decision_factors', []))
            ))
            
            return cursor.lastrowid
    
    def record_actual_revenue(self, agent_id: str, mining_revenue: float,
                            grid_revenue: float, energy_cost: float) -> int:
        """Record actual revenue earned"""
        timestamp = int(datetime.now().timestamp())
        total_revenue = mining_revenue + grid_revenue
        net_profit = total_revenue - energy_cost
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO actual_revenue
                (timestamp, agent_id, mining_revenue, grid_revenue,
                 total_revenue, energy_cost, net_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, agent_id, mining_revenue, grid_revenue,
                total_revenue, energy_cost, net_profit
            ))
            
            return cursor.lastrowid
    
    def record_grid_urgency_event(self, urgency_level: str, grid_stress: float,
                                response_capacity_mw: float, participating_agents: List[str]) -> int:
        """Record a grid urgency event"""
        timestamp = int(datetime.now().timestamp())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO grid_urgency_events
                (timestamp, urgency_level, grid_stress, response_capacity_mw,
                 participating_agents)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp, urgency_level, grid_stress, response_capacity_mw,
                json.dumps(participating_agents)
            ))
            
            return cursor.lastrowid


class AnalyticsCalculator:
    """Calculates analytics and performance metrics"""
    
    def __init__(self, db: AnalyticsDatabase):
        self.db = db
        
    def calculate_win_rate(self, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Dict:
        """Calculate win rate (how often grid revenue > mining revenue)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total_decisions,
                    SUM(CASE WHEN grid_value > mining_profit THEN 1 ELSE 0 END) as grid_wins,
                    AVG(mining_profit) as avg_mining_profit,
                    AVG(grid_value) as avg_grid_value
                FROM allocation_decisions
            """
            
            params = []
            if start_time:
                query += " WHERE timestamp >= ?"
                params.append(int(start_time.timestamp()))
            if end_time:
                query += " AND timestamp <= ?" if start_time else " WHERE timestamp <= ?"
                params.append(int(end_time.timestamp()))
                
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result['total_decisions'] == 0:
                return {
                    'win_rate': 0,
                    'total_decisions': 0,
                    'grid_wins': 0,
                    'avg_mining_profit': 0,
                    'avg_grid_value': 0
                }
                
            return {
                'win_rate': result['grid_wins'] / result['total_decisions'],
                'total_decisions': result['total_decisions'],
                'grid_wins': result['grid_wins'],
                'avg_mining_profit': result['avg_mining_profit'],
                'avg_grid_value': result['avg_grid_value']
            }
    
    def calculate_cumulative_profits(self, agent_id: Optional[str] = None) -> Dict:
        """Calculate cumulative arbitrage profits"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    agent_id,
                    SUM(net_profit) as total_profit,
                    SUM(mining_revenue) as total_mining_revenue,
                    SUM(grid_revenue) as total_grid_revenue,
                    SUM(energy_cost) as total_energy_cost,
                    COUNT(*) as revenue_records
                FROM actual_revenue
            """
            
            if agent_id:
                query += " WHERE agent_id = ?"
                query += " GROUP BY agent_id"
                cursor.execute(query, (agent_id,))
            else:
                query += " GROUP BY agent_id"
                cursor.execute(query)
                
            results = cursor.fetchall()
            
            return {
                'by_agent': {
                    row['agent_id']: {
                        'total_profit': row['total_profit'],
                        'total_mining_revenue': row['total_mining_revenue'],
                        'total_grid_revenue': row['total_grid_revenue'],
                        'total_energy_cost': row['total_energy_cost'],
                        'revenue_records': row['revenue_records']
                    }
                    for row in results
                },
                'total_system_profit': sum(row['total_profit'] for row in results)
            }
    
    def calculate_market_spreads(self, window_hours: int = 24) -> Dict:
        """Calculate average spread between mining and grid markets"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    AVG(ABS(mining_profit - grid_value)) as avg_spread,
                    MAX(ABS(mining_profit - grid_value)) as max_spread,
                    MIN(ABS(mining_profit - grid_value)) as min_spread,
                    AVG(ABS(mining_profit - grid_value) * ABS(mining_profit - grid_value)) - 
                        AVG(ABS(mining_profit - grid_value)) * AVG(ABS(mining_profit - grid_value)) as spread_volatility
                FROM allocation_decisions
                WHERE timestamp >= ? AND timestamp <= ?
            """, (int(start_time.timestamp()), int(end_time.timestamp())))
            
            result = cursor.fetchone()
            
            return {
                'avg_spread': result['avg_spread'] or 0,
                'max_spread': result['max_spread'] or 0,
                'min_spread': result['min_spread'] or 0,
                'spread_volatility': result['spread_volatility'] or 0,
                'window_hours': window_hours
            }
    
    def get_best_worst_hours(self, limit: int = 10) -> Dict:
        """Get best and worst performing hours"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Best performing hours
            cursor.execute("""
                SELECT 
                    timestamp,
                    agent_id,
                    net_profit,
                    mining_revenue,
                    grid_revenue
                FROM actual_revenue
                ORDER BY net_profit DESC
                LIMIT ?
            """, (limit,))
            
            best_hours = [
                {
                    'timestamp': datetime.fromtimestamp(row['timestamp']).isoformat(),
                    'agent_id': row['agent_id'],
                    'profit_per_mwh': row['net_profit'],
                    'spread': row['grid_revenue'] - row['mining_revenue'],
                    'net_profit': row['net_profit'],
                    'mining_revenue': row['mining_revenue'],
                    'grid_revenue': row['grid_revenue']
                }
                for row in cursor.fetchall()
            ]
            
            # Worst performing hours
            cursor.execute("""
                SELECT 
                    timestamp,
                    agent_id,
                    net_profit,
                    mining_revenue,
                    grid_revenue
                FROM actual_revenue
                ORDER BY net_profit ASC
                LIMIT ?
            """, (limit,))
            
            worst_hours = [
                {
                    'timestamp': datetime.fromtimestamp(row['timestamp']).isoformat(),
                    'agent_id': row['agent_id'],
                    'profit_per_mwh': row['net_profit'],
                    'spread': row['grid_revenue'] - row['mining_revenue'],
                    'net_profit': row['net_profit'],
                    'mining_revenue': row['mining_revenue'],
                    'grid_revenue': row['grid_revenue']
                }
                for row in cursor.fetchall()
            ]
            
            return {
                'best_performing_hours': best_hours,
                'worst_performing_hours': worst_hours
            }
    
    def get_hourly_statistics(self) -> Dict:
        """Get statistics by hour of day"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    CAST(strftime('%H', timestamp, 'unixepoch', 'localtime') AS INTEGER) as hour,
                    AVG(net_profit) as avg_profit,
                    AVG(mining_revenue) as avg_mining,
                    AVG(grid_revenue) as avg_grid,
                    COUNT(*) as sample_count
                FROM actual_revenue
                GROUP BY hour
                ORDER BY hour
            """)
            
            hourly_stats = {}
            for row in cursor.fetchall():
                hourly_stats[row['hour']] = {
                    'avg_profit': row['avg_profit'],
                    'avg_mining_revenue': row['avg_mining'],
                    'avg_grid_revenue': row['avg_grid'],
                    'sample_count': row['sample_count']
                }
                
            return hourly_stats
    
    def get_grid_response_statistics(self) -> Dict:
        """Get statistics on grid urgency responses"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    urgency_level,
                    COUNT(*) as event_count,
                    AVG(grid_stress) as avg_stress,
                    SUM(response_capacity_mw) as total_response_mw
                FROM grid_urgency_events
                GROUP BY urgency_level
            """)
            
            response_stats = {}
            for row in cursor.fetchall():
                response_stats[row['urgency_level']] = {
                    'event_count': row['event_count'],
                    'avg_stress': row['avg_stress'],
                    'total_response_mw': row['total_response_mw']
                }
                
            return response_stats
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get comprehensive performance summary"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get win rate
        win_rate_data = self.calculate_win_rate(start_time, end_time)
        
        # Get cumulative profits
        profit_data = self.calculate_cumulative_profits()
        
        # Get market spreads
        spread_data = self.calculate_market_spreads(hours)
        
        # Get best/worst hours
        performance_hours = self.get_best_worst_hours(5)
        
        # Count grid events
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM grid_urgency_events 
                WHERE timestamp >= ? AND urgency_level IN ('HIGH', 'CRITICAL')
            """, (int(start_time.timestamp()),))
            grid_events = cursor.fetchone()['count']
        
        return {
            'win_rate': win_rate_data.get('win_rate', 0),
            'cumulative_profit': profit_data.get('total_system_profit', 0),
            'avg_market_spread': spread_data.get('avg_spread', 0),
            'grid_events': grid_events,
            'best_hours': performance_hours.get('best_performing_hours', []),
            'worst_hours': performance_hours.get('worst_performing_hours', []),
            'total_decisions': win_rate_data.get('total_decisions', 0)
        }
    
    def get_cumulative_profit(self, hours: int = 24) -> List[Dict]:
        """Get cumulative profit history for charting"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    timestamp,
                    SUM(net_profit) OVER (ORDER BY timestamp) as cumulative_profit
                FROM actual_revenue
                WHERE timestamp >= ?
                ORDER BY timestamp
            """, (int(start_time.timestamp()),))
            
            return [
                {
                    'timestamp': datetime.fromtimestamp(row['timestamp']).isoformat(),
                    'cumulative_profit': row['cumulative_profit']
                }
                for row in cursor.fetchall()
            ]
    
    def get_hourly_patterns(self) -> Dict:
        """Get hourly performance patterns"""
        hourly_stats = self.get_hourly_statistics()
        
        # Extract hourly average profits
        hourly_avg_profit = [0] * 24
        for hour, stats in hourly_stats.items():
            if hour < 24:
                hourly_avg_profit[hour] = stats.get('avg_profit', 0)
        
        return {
            'hourly_avg_profit': hourly_avg_profit,
            'hourly_stats': hourly_stats
        }


# Flask API for serving analytics data
app = Flask(__name__)
analytics_tracker = AnalyticsTracker()
analytics_calc = AnalyticsCalculator(analytics_tracker.db)

@app.route('/api/analytics/status', methods=['GET'])
def get_analytics_status():
    """Get current analytics system status"""
    return jsonify({
        'status': 'active',
        'recording': analytics_tracker.is_recording,
        'database': analytics_tracker.db.db_path
    })

@app.route('/api/analytics/win-rate', methods=['GET'])
def get_win_rate():
    """Get win rate statistics"""
    # Optional time range parameters
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    if start_time:
        start_time = datetime.fromisoformat(start_time)
    if end_time:
        end_time = datetime.fromisoformat(end_time)
        
    win_rate_data = analytics_calc.calculate_win_rate(start_time, end_time)
    return jsonify(win_rate_data)

@app.route('/api/analytics/cumulative-profits', methods=['GET'])
def get_cumulative_profits():
    """Get cumulative profit data"""
    agent_id = request.args.get('agent_id')
    profit_data = analytics_calc.calculate_cumulative_profits(agent_id)
    return jsonify(profit_data)

@app.route('/api/analytics/market-spreads', methods=['GET'])
def get_market_spreads():
    """Get market spread analytics"""
    window_hours = int(request.args.get('window_hours', 24))
    spread_data = analytics_calc.calculate_market_spreads(window_hours)
    return jsonify(spread_data)

@app.route('/api/analytics/best-worst-hours', methods=['GET'])
def get_best_worst_hours():
    """Get best and worst performing hours"""
    limit = int(request.args.get('limit', 10))
    performance_data = analytics_calc.get_best_worst_hours(limit)
    return jsonify(performance_data)

@app.route('/api/analytics/hourly-statistics', methods=['GET'])
def get_hourly_statistics():
    """Get statistics by hour of day"""
    hourly_stats = analytics_calc.get_hourly_statistics()
    return jsonify(hourly_stats)

@app.route('/api/analytics/grid-response', methods=['GET'])
def get_grid_response_stats():
    """Get grid response statistics"""
    response_stats = analytics_calc.get_grid_response_statistics()
    return jsonify(response_stats)

@app.route('/api/analytics/record-decision', methods=['POST'])
def record_decision():
    """Record an allocation decision"""
    data = request.json
    agent_id = data.get('agent_id')
    action = data.get('action')
    reasoning = data.get('reasoning')
    
    if not all([agent_id, action, reasoning]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    decision_id = analytics_tracker.record_allocation_decision(
        agent_id, action, reasoning
    )
    
    return jsonify({'decision_id': decision_id})

@app.route('/api/analytics/record-revenue', methods=['POST'])
def record_revenue():
    """Record actual revenue data"""
    data = request.json
    agent_id = data.get('agent_id')
    mining_revenue = data.get('mining_revenue', 0)
    grid_revenue = data.get('grid_revenue', 0)
    energy_cost = data.get('energy_cost', 0)
    
    if not agent_id:
        return jsonify({'error': 'Missing agent_id'}), 400
        
    revenue_id = analytics_tracker.record_actual_revenue(
        agent_id, mining_revenue, grid_revenue, energy_cost
    )
    
    return jsonify({'revenue_id': revenue_id})

@app.route('/api/analytics/record-grid-urgency', methods=['POST'])
def record_grid_urgency():
    """Record a grid urgency event"""
    data = request.json
    urgency_level = data.get('urgency_level')
    grid_stress = data.get('grid_stress')
    response_capacity_mw = data.get('response_capacity_mw')
    participating_agents = data.get('participating_agents', [])
    
    if not all([urgency_level, grid_stress is not None, response_capacity_mw is not None]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    event_id = analytics_tracker.record_grid_urgency_event(
        urgency_level, grid_stress, response_capacity_mw, participating_agents
    )
    
    return jsonify({'event_id': event_id})

@app.route('/api/analytics/start-recording', methods=['POST'])
def start_recording():
    """Start automatic data recording"""
    analytics_tracker.start_recording()
    return jsonify({'status': 'recording started'})

@app.route('/api/analytics/stop-recording', methods=['POST'])
def stop_recording():
    """Stop automatic data recording"""
    analytics_tracker.stop_recording()
    return jsonify({'status': 'recording stopped'})


# Integration helper functions
def integrate_with_agent(agent):
    """Helper to integrate analytics with an IntelligentMiningAgent"""
    original_decide = agent.decide_action
    original_execute = agent.execute_action
    
    def wrapped_decide():
        action, reasoning = original_decide()
        # Record the decision
        analytics_tracker.record_allocation_decision(
            agent.agent_id, action, reasoning
        )
        return action, reasoning
    
    def wrapped_execute(action):
        result = original_execute(action)
        # You would calculate actual revenue here based on execution results
        # This is a placeholder - actual implementation would integrate with real revenue tracking
        return result
    
    agent.decide_action = wrapped_decide
    agent.execute_action = wrapped_execute
    
    return agent


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Run test mode - populate with sample data
        print("Populating analytics database with test data...")
        
        # Create some sample market data
        for i in range(100):
            timestamp = int((datetime.now() - timedelta(hours=i)).timestamp())
            
            # Simulate market conditions
            hour = (24 - i) % 24
            grid_stress = 0.4 + 0.3 * np.sin((hour - 14) * np.pi / 12)
            energy_price = 0.05 * (1 + grid_stress)
            
            with analytics_tracker.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO hourly_prices 
                    (timestamp, energy_price, btc_price, mining_difficulty,
                     grid_stress, renewable_percent, frequency)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    energy_price,
                    40000 + np.random.normal(0, 1000),
                    65e12,
                    grid_stress,
                    max(0, np.sin((hour - 6) * np.pi / 12)) * 0.4,
                    60.0 + np.random.normal(0, 0.01)
                ))
                
                # Simulate decisions and revenue
                for agent_id in ['MARA_TX_1', 'MARA_TX_2', 'MARA_MT_1']:
                    mining_profit = np.random.uniform(50, 150)
                    grid_value = np.random.uniform(30, 200)
                    
                    if grid_stress > 0.8 or grid_value > mining_profit:
                        action_type = 'GRID_PRIORITY'
                        mining_percent = 20
                        grid_percent = 80
                    else:
                        action_type = 'MINE_MAX'
                        mining_percent = 100
                        grid_percent = 0
                        
                    cursor.execute("""
                        INSERT INTO allocation_decisions
                        (timestamp, agent_id, action_type, mining_percent,
                         grid_support_percent, expected_profit, mining_profit,
                         grid_value, decision_factors)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, agent_id, action_type, mining_percent,
                        grid_percent, 
                        mining_profit * (mining_percent/100) + grid_value * (grid_percent/100),
                        mining_profit, grid_value,
                        json.dumps(['Test decision'])
                    ))
                    
                    # Record revenue
                    mining_revenue = mining_profit * (mining_percent/100) * np.random.uniform(0.9, 1.1)
                    grid_revenue = grid_value * (grid_percent/100) * np.random.uniform(0.95, 1.05)
                    energy_cost = energy_price * 100 * np.random.uniform(0.9, 1.1)
                    
                    cursor.execute("""
                        INSERT INTO actual_revenue
                        (timestamp, agent_id, mining_revenue, grid_revenue,
                         total_revenue, energy_cost, net_profit)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, agent_id, mining_revenue, grid_revenue,
                        mining_revenue + grid_revenue, energy_cost,
                        mining_revenue + grid_revenue - energy_cost
                    ))
        
        print("Test data populated successfully!")
        
        # Print some sample analytics
        print("\nSample Analytics:")
        print("-" * 50)
        
        win_rate = analytics_calc.calculate_win_rate()
        print(f"Win Rate: {win_rate['win_rate']:.2%}")
        print(f"Total Decisions: {win_rate['total_decisions']}")
        
        profits = analytics_calc.calculate_cumulative_profits()
        print(f"\nTotal System Profit: ${profits['total_system_profit']:,.2f}")
        
        spreads = analytics_calc.calculate_market_spreads()
        print(f"\nAverage Market Spread: ${spreads['avg_spread']:,.2f}")
        
    else:
        # Run the API server
        print("Starting PoUW-DEM Analytics API on http://localhost:5001")
        print("API Endpoints:")
        print("  GET  /api/analytics/status")
        print("  GET  /api/analytics/win-rate")
        print("  GET  /api/analytics/cumulative-profits")
        print("  GET  /api/analytics/market-spreads")
        print("  GET  /api/analytics/best-worst-hours")
        print("  GET  /api/analytics/hourly-statistics")
        print("  GET  /api/analytics/grid-response")
        print("  POST /api/analytics/record-decision")
        print("  POST /api/analytics/record-revenue")
        print("  POST /api/analytics/record-grid-urgency")
        print("  POST /api/analytics/start-recording")
        print("  POST /api/analytics/stop-recording")
        
        app.run(host='0.0.0.0', port=5001, debug=False)