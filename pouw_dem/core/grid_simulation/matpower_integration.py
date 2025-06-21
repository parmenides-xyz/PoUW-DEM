"""
MATPOWER Integration for Realistic Grid Simulation
Connects our PoUW system to actual power flow simulations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import subprocess
import json
import asyncio
from dataclasses import dataclass
from datetime import datetime

# For MATPOWER interface via oct2py
try:
    from oct2py import octave
    MATPOWER_AVAILABLE = True
except ImportError:
    MATPOWER_AVAILABLE = False
    print("oct2py not installed. Install with: pip install oct2py")


@dataclass
class GridState:
    """Real-time grid state from MATPOWER simulation"""
    timestamp: datetime
    bus_voltages: np.ndarray
    line_flows: np.ndarray
    frequency_deviation: float
    total_generation: float
    total_load: float
    renewable_generation: float
    system_losses: float
    stability_margin: float
    congestion_level: float


class MATPOWERGrid:
    """
    Interface to MATPOWER for realistic grid simulation
    Uses IEEE test cases or custom grid models
    """
    
    def __init__(self, case_name: str = "case30"):
        """
        Initialize MATPOWER grid simulation
        
        Args:
            case_name: MATPOWER case to use (e.g., 'case30', 'case118', 'case300')
        """
        self.case_name = case_name
        self.grid_data = None
        self.current_state = None
        
        # Mining farms as flexible loads
        self.mining_loads = {}
        
        # Renewable generation buses
        self.renewable_buses = []
        
        if MATPOWER_AVAILABLE:
            # Add MATPOWER to Octave path
            octave.addpath('/usr/local/MATPOWER/lib')
            octave.addpath('/usr/local/MATPOWER/data')
            
            # Load the case
            self.load_case()
    
    def load_case(self):
        """Load MATPOWER case data"""
        if not MATPOWER_AVAILABLE:
            # Use synthetic data if MATPOWER not available
            self.grid_data = self._create_synthetic_grid()
            return
        
        # Load MATPOWER case
        octave.eval(f"mpc = loadcase('{self.case_name}');")
        
        # Extract grid data
        self.grid_data = {
            'bus': octave.eval('mpc.bus'),
            'gen': octave.eval('mpc.gen'),
            'branch': octave.eval('mpc.branch'),
            'baseMVA': octave.eval('mpc.baseMVA')
        }
        
        # Identify renewable generation buses (simplified - mark some as renewable)
        num_gens = self.grid_data['gen'].shape[0]
        self.renewable_buses = np.random.choice(
            range(num_gens), 
            size=max(1, num_gens // 3), 
            replace=False
        ).tolist()
    
    def add_mining_load(self, bus_id: int, capacity_mw: float, miner_id: str):
        """
        Add a Bitcoin mining farm as a flexible load
        
        Args:
            bus_id: Bus number to connect the mining load
            capacity_mw: Maximum mining capacity in MW
            miner_id: Unique identifier for the miner
        """
        self.mining_loads[miner_id] = {
            'bus_id': bus_id,
            'capacity_mw': capacity_mw,
            'current_load': capacity_mw,  # Start at full capacity
            'is_active': True,
            'grid_task_mode': False
        }
        
        print(f"Added {capacity_mw} MW mining load at bus {bus_id}")
    
    def run_power_flow(self) -> GridState:
        """
        Run AC power flow analysis and return grid state
        """
        if not MATPOWER_AVAILABLE:
            return self._simulate_power_flow()
        
        # Update loads with current mining status
        self._update_mining_loads()
        
        # Run power flow
        octave.eval("results = runpf(mpc);")
        success = octave.eval("results.success")
        
        if not success:
            print("Power flow did not converge!")
            return self._get_emergency_state()
        
        # Extract results
        bus_results = octave.eval("results.bus")
        branch_results = octave.eval("results.branch")
        gen_results = octave.eval("results.gen")
        
        # Calculate grid metrics
        state = GridState(
            timestamp=datetime.now(),
            bus_voltages=bus_results[:, 7],  # Voltage magnitudes
            line_flows=branch_results[:, 13],  # Real power flows
            frequency_deviation=self._calculate_frequency_deviation(gen_results),
            total_generation=np.sum(gen_results[:, 1]),
            total_load=self._calculate_total_load(bus_results),
            renewable_generation=self._calculate_renewable_generation(gen_results),
            system_losses=self._calculate_losses(branch_results),
            stability_margin=self._calculate_stability_margin(bus_results),
            congestion_level=self._calculate_congestion(branch_results)
        )
        
        self.current_state = state
        return state
    
    def _simulate_power_flow(self) -> GridState:
        """Synthetic power flow for when MATPOWER isn't available"""
        # Create realistic-looking grid state
        num_buses = 30  # IEEE 30-bus system
        
        # Simulate voltage deviations
        base_voltage = 1.0
        voltage_deviation = np.random.normal(0, 0.02, num_buses)
        bus_voltages = base_voltage + voltage_deviation
        
        # Simulate line flows (% of capacity)
        num_lines = 41  # Typical for 30-bus system
        line_flows = np.random.beta(2, 5, num_lines) * 100  # Skewed towards lower flows
        
        # Other metrics
        total_gen = 250 + np.random.normal(0, 10)
        total_load = 240 + np.random.normal(0, 8)
        
        # Account for mining loads
        mining_load = sum(
            load['current_load'] 
            for load in self.mining_loads.values() 
            if load['is_active']
        )
        total_load += mining_load
        
        # Renewable generation (varies with time)
        hour = datetime.now().hour
        solar_factor = max(0, np.sin((hour - 6) * np.pi / 12)) if 6 <= hour <= 18 else 0
        renewable_gen = 50 * solar_factor + np.random.normal(0, 5)
        
        return GridState(
            timestamp=datetime.now(),
            bus_voltages=bus_voltages,
            line_flows=line_flows,
            frequency_deviation=np.random.normal(0, 0.1),
            total_generation=total_gen,
            total_load=total_load,
            renewable_generation=max(0, renewable_gen),
            system_losses=total_gen - total_load,
            stability_margin=self._calculate_synthetic_stability(bus_voltages),
            congestion_level=np.max(line_flows) / 100
        )
    
    def adjust_mining_load(self, miner_id: str, new_load_mw: float):
        """
        Adjust mining load (e.g., when switching to grid tasks)
        
        Args:
            miner_id: Miner identifier
            new_load_mw: New load in MW (0 for complete shutdown)
        """
        if miner_id in self.mining_loads:
            old_load = self.mining_loads[miner_id]['current_load']
            self.mining_loads[miner_id]['current_load'] = new_load_mw
            self.mining_loads[miner_id]['grid_task_mode'] = new_load_mw < old_load * 0.5
            
            print(f"Miner {miner_id} load adjusted from {old_load} MW to {new_load_mw} MW")
    
    def get_grid_urgency(self) -> float:
        """
        Calculate grid urgency score (0-1) based on current state
        """
        if not self.current_state:
            return 0.5
        
        state = self.current_state
        
        # Factors contributing to urgency
        voltage_stress = np.mean(np.abs(state.bus_voltages - 1.0)) * 10
        congestion_stress = state.congestion_level
        frequency_stress = abs(state.frequency_deviation) / 0.5
        margin_stress = max(0, 1 - state.stability_margin)
        
        # Supply-demand imbalance
        imbalance = abs(state.total_generation - state.total_load) / state.total_generation
        
        # Weighted urgency score
        urgency = (
            0.25 * voltage_stress +
            0.25 * congestion_stress +
            0.2 * frequency_stress +
            0.2 * margin_stress +
            0.1 * imbalance
        )
        
        return min(1.0, urgency)
    
    def simulate_contingency(self, line_id: int) -> Dict[str, float]:
        """
        Simulate N-1 contingency (line outage) and assess impact
        
        Args:
            line_id: Line to take out of service
            
        Returns:
            Impact metrics
        """
        if not MATPOWER_AVAILABLE:
            # Synthetic contingency analysis
            return {
                'voltage_violations': np.random.randint(0, 5),
                'overloaded_lines': np.random.randint(0, 3),
                'load_shed_mw': np.random.uniform(0, 50),
                'can_mining_help': np.random.random() > 0.3
            }
        
        # Save current state
        octave.eval("mpc_backup = mpc;")
        
        # Remove line
        octave.eval(f"mpc.branch({line_id}, 11) = 0;")  # Set line status to 0
        
        # Run contingency power flow
        octave.eval("results_cont = runpf(mpc);")
        
        # Analyze impact
        impact = self._analyze_contingency_impact()
        
        # Restore original state
        octave.eval("mpc = mpc_backup;")
        
        return impact
    
    def _calculate_frequency_deviation(self, gen_results: np.ndarray) -> float:
        """Calculate system frequency deviation based on generation/load imbalance"""
        # Simplified: frequency deviation proportional to power imbalance
        total_gen = np.sum(gen_results[:, 1])
        total_load = self._calculate_total_load(None)
        
        imbalance = (total_gen - total_load) / total_gen
        # Typical frequency sensitivity: 0.1 Hz per 1% imbalance
        return imbalance * 10  # Returns deviation in 0.01 Hz units
    
    def _calculate_total_load(self, bus_results: np.ndarray) -> float:
        """Calculate total system load including mining"""
        if bus_results is not None:
            base_load = np.sum(bus_results[:, 2])  # Real power load
        else:
            base_load = 240  # Default for synthetic case
        
        mining_load = sum(
            load['current_load'] 
            for load in self.mining_loads.values()
        )
        
        return base_load + mining_load
    
    def _calculate_renewable_generation(self, gen_results: np.ndarray) -> float:
        """Calculate total renewable generation"""
        renewable_gen = 0
        for i in self.renewable_buses:
            if i < len(gen_results):
                renewable_gen += gen_results[i, 1]
        return renewable_gen
    
    def _calculate_losses(self, branch_results: np.ndarray) -> float:
        """Calculate system losses"""
        # Real power losses = sum of (Pf + Pt) for all branches
        return np.sum(branch_results[:, 13] + branch_results[:, 15])
    
    def _calculate_stability_margin(self, bus_results: np.ndarray) -> float:
        """
        Calculate voltage stability margin
        Returns value between 0 (unstable) and 1 (very stable)
        """
        voltages = bus_results[:, 7] if bus_results is not None else np.ones(30)
        
        # Check voltage violations (typically ±5% of nominal)
        violations = np.sum((voltages < 0.95) | (voltages > 1.05))
        
        # Average voltage deviation
        avg_deviation = np.mean(np.abs(voltages - 1.0))
        
        # Stability margin (higher is better)
        margin = 1.0 - (violations / len(voltages)) - avg_deviation * 2
        
        return max(0, margin)
    
    def _calculate_congestion(self, branch_results: np.ndarray) -> float:
        """
        Calculate transmission congestion level
        Returns value between 0 (no congestion) and 1 (severe congestion)
        """
        if branch_results is None:
            return 0.3  # Default moderate congestion
        
        # Line flows as percentage of limits
        flow_percentages = branch_results[:, 13] / branch_results[:, 5]  # Pf / rateA
        
        # Count heavily loaded lines (>80% capacity)
        heavy_lines = np.sum(flow_percentages > 0.8)
        
        # Maximum line loading
        max_loading = np.max(flow_percentages)
        
        # Congestion metric
        congestion = (heavy_lines / len(flow_percentages)) * 0.5 + min(1.0, max_loading) * 0.5
        
        return congestion
    
    def _calculate_synthetic_stability(self, voltages: np.ndarray) -> float:
        """Synthetic stability calculation when MATPOWER not available"""
        violations = np.sum((voltages < 0.95) | (voltages > 1.05))
        avg_deviation = np.mean(np.abs(voltages - 1.0))
        return max(0, 1.0 - (violations / len(voltages)) - avg_deviation * 2)
    
    def _get_emergency_state(self) -> GridState:
        """Return emergency grid state when power flow fails"""
        return GridState(
            timestamp=datetime.now(),
            bus_voltages=np.ones(30) * 0.9,  # Low voltages
            line_flows=np.ones(41) * 90,  # High flows
            frequency_deviation=-0.5,  # Frequency drop
            total_generation=200,
            total_load=250,
            renewable_generation=10,
            system_losses=20,
            stability_margin=0.1,
            congestion_level=0.9
        )
    
    async def monitor_grid(self, callback, interval_seconds: int = 5):
        """
        Continuously monitor grid state and trigger callbacks
        
        Args:
            callback: Async function to call with grid state
            interval_seconds: Monitoring interval
        """
        while True:
            state = self.run_power_flow()
            await callback(state)
            await asyncio.sleep(interval_seconds)


class GridOptimizationTasks:
    """
    Generate realistic grid optimization tasks based on MATPOWER state
    """
    
    def __init__(self, grid: MATPOWERGrid):
        self.grid = grid
    
    def generate_stability_task(self, state: GridState) -> Dict:
        """Generate voltage stability improvement task"""
        weak_buses = np.where(state.bus_voltages < 0.95)[0]
        
        return {
            'type': 'STABILITY_SIMULATION',
            'description': f"Optimize reactive power for {len(weak_buses)} weak buses",
            'input_data': {
                'weak_buses': weak_buses.tolist(),
                'current_voltages': state.bus_voltages[weak_buses].tolist(),
                'target_voltage': 1.0
            },
            'expected_impact': len(weak_buses) * 0.5,  # % improvement
            'compute_requirement': len(weak_buses) * 0.1,  # GHz
            'deadline_minutes': 30 if state.stability_margin < 0.3 else 60
        }
    
    def generate_congestion_task(self, state: GridState) -> Dict:
        """Generate transmission congestion relief task"""
        congested_lines = np.where(state.line_flows > 80)[0]
        
        return {
            'type': 'LOAD_BALANCE',
            'description': f"Optimize power flow for {len(congested_lines)} congested lines",
            'input_data': {
                'congested_lines': congested_lines.tolist(),
                'current_flows': state.line_flows[congested_lines].tolist(),
                'line_limits': [100] * len(congested_lines)  # MW limits
            },
            'expected_impact': state.congestion_level * 10,
            'compute_requirement': len(congested_lines) * 0.2,
            'deadline_minutes': 15 if state.congestion_level > 0.8 else 45
        }
    
    def generate_renewable_forecast_task(self, state: GridState) -> Dict:
        """Generate renewable energy forecasting task"""
        return {
            'type': 'RENEWABLE_FORECAST',
            'description': "24-hour ahead solar/wind forecast",
            'input_data': {
                'current_renewable_mw': state.renewable_generation,
                'total_capacity_mw': 100,  # Assumed capacity
                'weather_data': self._get_synthetic_weather()
            },
            'expected_impact': 2.0,  # Better forecasts improve dispatch
            'compute_requirement': 1.0,  # Standard ML workload
            'deadline_minutes': 120
        }
    
    def _get_synthetic_weather(self) -> Dict:
        """Generate synthetic weather data"""
        hour = datetime.now().hour
        return {
            'temperature': 20 + 10 * np.sin((hour - 6) * np.pi / 12),
            'cloud_cover': np.random.beta(2, 5),  # 0-1, skewed towards clear
            'wind_speed': np.random.gamma(2, 2),  # m/s
            'humidity': np.random.uniform(0.3, 0.8)
        }