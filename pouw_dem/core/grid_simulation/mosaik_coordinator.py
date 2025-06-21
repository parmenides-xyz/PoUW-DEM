"""
Mosaik Co-Simulation Coordinator
Integrates MATPOWER grid, blockchain, and AI agents
"""

import mosaik
import mosaik.util
import asyncio
import numpy as np
from typing import Dict, List
import json
from datetime import datetime
import pandas as pd

# Import our components
from matpower_integration import MATPOWERGrid, GridOptimizationTasks
from ..agents.PoUWAgent import PoUWAgent, MinerState, TaskType, Priority
from ..task_scheduler.task_scheduler import DynamicTaskScheduler


class GridSimulator(mosaik.Simulator):
    """
    Mosaik adapter for MATPOWER grid simulation
    """
    
    def __init__(self):
        super().__init__(mosaik.SimConfig(
            'Grid',
            time_resolution=1,  # 1 second resolution
            sim_start='01-01-2024 00:00:00',
        ))
        self.grid = None
        self.entities = {}
        
    def init(self, sid, time_resolution=1, case_name='case30'):
        self.grid = MATPOWERGrid(case_name)
        # Add example mining loads
        self.grid.add_mining_load(bus_id=10, capacity_mw=50, miner_id='miner_1')
        self.grid.add_mining_load(bus_id=15, capacity_mw=30, miner_id='miner_2')
        self.grid.add_mining_load(bus_id=20, capacity_mw=40, miner_id='miner_3')
        return self.meta
        
    def create(self, num, model, **params):
        entities = []
        for i in range(num):
            eid = f'{model}_{i}'
            self.entities[eid] = {
                'type': model,
                'params': params
            }
            entities.append({'eid': eid, 'type': model})
        return entities
    
    def step(self, time, inputs, max_advance):
        # Process mining load adjustments from inputs
        for eid, attrs in inputs.items():
            if 'mining_load_mw' in attrs:
                miner_id = eid.split('_')[1]
                new_load = attrs['mining_load_mw'][0]
                self.grid.adjust_mining_load(f'miner_{miner_id}', new_load)
        
        # Run power flow
        state = self.grid.run_power_flow()
        
        # Update entity outputs
        for eid in self.entities:
            self.entities[eid]['grid_urgency'] = self.grid.get_grid_urgency()
            self.entities[eid]['frequency'] = 50 + state.frequency_deviation * 0.01
            self.entities[eid]['voltage'] = np.mean(state.bus_voltages)
            self.entities[eid]['congestion'] = state.congestion_level
            self.entities[eid]['renewable_mw'] = state.renewable_generation
        
        return time + 60  # Advance 1 minute
    
    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            data[eid] = {}
            for attr in attrs:
                if attr in self.entities[eid]:
                    data[eid][attr] = self.entities[eid][attr]
        return data
    
    @property
    def meta(self):
        return {
            'type': 'time-based',
            'models': {
                'GridBus': {
                    'public': True,
                    'params': ['bus_id'],
                    'attrs': [
                        'grid_urgency', 'frequency', 'voltage', 
                        'congestion', 'renewable_mw', 'mining_load_mw'
                    ]
                }
            }
        }


class MinerSimulator(mosaik.Simulator):
    """
    Mosaik adapter for PoUW mining agents
    """
    
    def __init__(self):
        super().__init__(mosaik.SimConfig(
            'Miner',
            time_resolution=1,
        ))
        self.agents = {}
        self.scheduler = DynamicTaskScheduler(
            energy_pool_address="0x" + "0" * 40,  # Mock address
            pouw_contract_address="0x" + "1" * 40
        )
        
    def init(self, sid, time_resolution=1):
        return self.meta
    
    def create(self, num, model, **params):
        entities = []
        for i in range(num):
            eid = f'{model}_{i}'
            
            # Create PoUW agent
            agent_config = {
                'agent_id': i,
                'n_agents': num,
                'lr_actor': 1e-4,
                'lr_critic': 1e-3,
                'device': 'cpu'
            }
            agent = PoUWAgent(agent_config)
            
            self.agents[eid] = {
                'agent': agent,
                'capacity_mw': params.get('capacity_mw', 50),
                'current_action': 0,
                'grid_task_active': False
            }
            
            entities.append({'eid': eid, 'type': model})
        return entities
    
    def step(self, time, inputs, max_advance):
        # Process grid state inputs
        for eid, attrs in inputs.items():
            if eid in self.agents:
                # Create miner state from grid inputs
                grid_urgency = attrs.get('grid_urgency', [0.5])[0]
                frequency = attrs.get('frequency', [50])[0]
                voltage = attrs.get('voltage', [1.0])[0]
                
                # Calculate energy price based on grid conditions
                energy_price = 0.1 + grid_urgency * 0.2
                
                state = MinerState(
                    battery_level=1.5,
                    solar_production=attrs.get('renewable_mw', [0])[0] / 100,
                    house_consumption=1.0,
                    hour_of_day=(time // 3600) % 24,
                    grid_urgency=grid_urgency,
                    energy_price=energy_price,
                    mining_difficulty=1000000,
                    pending_tasks=[],
                    compute_power=self.agents[eid]['capacity_mw'] / 100
                )
                
                # Get action from agent
                nn_state = self.agents[eid]['agent'].process_state(state)
                action, _ = self.agents[eid]['agent'].select_action(nn_state)
                
                # Update mining load based on action
                if action == 5:  # Grid task
                    # Reduce mining by 80% when doing grid tasks
                    self.agents[eid]['mining_load_mw'] = self.agents[eid]['capacity_mw'] * 0.2
                    self.agents[eid]['grid_task_active'] = True
                else:
                    # Full mining load
                    self.agents[eid]['mining_load_mw'] = self.agents[eid]['capacity_mw']
                    self.agents[eid]['grid_task_active'] = False
                
                self.agents[eid]['current_action'] = action
        
        return time + 60
    
    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid in self.agents:
                data[eid] = {}
                for attr in attrs:
                    if attr in self.agents[eid]:
                        data[eid][attr] = self.agents[eid][attr]
        return data
    
    @property
    def meta(self):
        return {
            'type': 'time-based',
            'models': {
                'MiningAgent': {
                    'public': True,
                    'params': ['capacity_mw'],
                    'attrs': [
                        'mining_load_mw', 'current_action', 'grid_task_active',
                        'grid_urgency', 'frequency', 'voltage', 'renewable_mw'
                    ]
                }
            }
        }


class BlockchainSimulator(mosaik.Simulator):
    """
    Mosaik adapter for blockchain coordination
    """
    
    def __init__(self):
        super().__init__(mosaik.SimConfig(
            'Blockchain',
            time_resolution=1,
        ))
        self.stability_tokens = {}
        self.total_tokens_minted = 0
        self.task_queue = []
        
    def init(self, sid, time_resolution=1):
        return self.meta
    
    def create(self, num, model, **params):
        return [{'eid': 'blockchain_0', 'type': model}]
    
    def step(self, time, inputs, max_advance):
        # Process grid task completions
        for eid, attrs in inputs.items():
            if 'grid_task_active' in attrs and attrs['grid_task_active'][0]:
                # Mint stability token
                miner_id = eid.split('_')[1]
                if miner_id not in self.stability_tokens:
                    self.stability_tokens[miner_id] = 0
                
                # Calculate reward based on grid urgency
                urgency = attrs.get('grid_urgency', [0.5])[0]
                token_reward = urgency * 10  # More tokens for higher urgency
                
                self.stability_tokens[miner_id] += token_reward
                self.total_tokens_minted += token_reward
        
        return time + 60
    
    def get_data(self, outputs):
        return {
            'blockchain_0': {
                'total_tokens': self.total_tokens_minted,
                'miner_tokens': json.dumps(self.stability_tokens)
            }
        }
    
    @property
    def meta(self):
        return {
            'type': 'time-based',
            'models': {
                'StabilityToken': {
                    'public': True,
                    'params': [],
                    'attrs': ['total_tokens', 'miner_tokens', 'grid_task_active']
                }
            }
        }


def run_cosimulation(duration_hours: int = 24):
    """
    Run the full co-simulation with mosaik
    """
    # Create world
    world = mosaik.World(sim_config={
        'Grid': {
            'python': 'grid_simulation.mosaik_coordinator:GridSimulator',
        },
        'Miner': {
            'python': 'grid_simulation.mosaik_coordinator:MinerSimulator',
        },
        'Blockchain': {
            'python': 'grid_simulation.mosaik_coordinator:BlockchainSimulator',
        },
        'DB': {
            'cmd': 'mosaik-csv %(addr)s',
            'cwd': '.',
        }
    })
    
    # Start simulators
    grid_sim = world.start('Grid', case_name='case30')
    miner_sim = world.start('Miner')
    blockchain_sim = world.start('Blockchain')
    db = world.start('DB', step_size=60)
    
    # Create entities
    grid = grid_sim.GridBus(bus_id=1)
    miners = miner_sim.MiningAgent.create(3, capacity_mw=[50, 30, 40])
    blockchain = blockchain_sim.StabilityToken()
    database = db.Database(filename='simulation_results.csv')
    
    # Connect entities
    for i, miner in enumerate(miners):
        # Grid -> Miner (state information)
        world.connect(grid, miner, 
                     ('grid_urgency', 'frequency', 'voltage', 'renewable_mw'))
        
        # Miner -> Grid (load adjustments)
        world.connect(miner, grid, 'mining_load_mw')
        
        # Miner -> Blockchain (task completion)
        world.connect(miner, blockchain, ('grid_task_active', 'grid_urgency'))
    
    # Connect to database for logging
    world.connect(grid, database, 
                 ('grid_urgency', 'frequency', 'voltage', 'congestion'))
    world.connect(blockchain, database, 'total_tokens')
    
    # Run simulation
    duration_seconds = duration_hours * 3600
    world.run(until=duration_seconds, print_progress=True)
    
    print(f"\nSimulation completed for {duration_hours} hours")
    print("Results saved to simulation_results.csv")
    
    return world


def analyze_results(csv_file: str = 'simulation_results.csv'):
    """
    Analyze and visualize co-simulation results
    """
    import matplotlib.pyplot as plt
    
    # Load results
    df = pd.read_csv(csv_file, index_col=0)
    
    # Convert time to hours
    df['hour'] = df.index / 3600
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mosaik Co-Simulation Results: Grid-Aware Bitcoin Mining', fontsize=16)
    
    # Plot 1: Grid conditions
    ax1 = axes[0, 0]
    ax1.plot(df['hour'], df['Grid-0.grid_urgency'], 'r-', label='Grid Urgency')
    ax1.plot(df['hour'], df['Grid-0.congestion'], 'b--', label='Congestion Level')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Level (0-1)')
    ax1.set_title('Grid Stress Indicators')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency stability
    ax2 = axes[0, 1]
    ax2.plot(df['hour'], df['Grid-0.frequency'], 'g-')
    ax2.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Nominal')
    ax2.fill_between(df['hour'], 49.9, 50.1, alpha=0.2, color='green', label='Safe band')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Grid Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mining load adjustments
    ax3 = axes[1, 0]
    for i in range(3):
        if f'MiningAgent_{i}.mining_load_mw' in df.columns:
            ax3.plot(df['hour'], df[f'MiningAgent_{i}.mining_load_mw'], 
                    label=f'Miner {i+1}')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Mining Load (MW)')
    ax3.set_title('Dynamic Mining Load Adjustments')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stability tokens earned
    ax4 = axes[1, 1]
    if 'StabilityToken-0.total_tokens' in df.columns:
        ax4.plot(df['hour'], df['StabilityToken-0.total_tokens'].cumsum(), 'purple')
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Cumulative Tokens')
    ax4.set_title('Stability Tokens Minted')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mosaik_simulation_results.png', dpi=300)
    plt.show()
    
    # Print summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Average Grid Urgency: {df['Grid-0.grid_urgency'].mean():.3f}")
    print(f"Peak Grid Urgency: {df['Grid-0.grid_urgency'].max():.3f}")
    print(f"Frequency Deviations > 0.2 Hz: {sum(abs(df['Grid-0.frequency'] - 50) > 0.2)}")
    print(f"Total Stability Tokens Minted: {df['StabilityToken-0.total_tokens'].iloc[-1]:.0f}")
    
    # Calculate mining flexibility
    total_capacity = 120  # MW (50 + 30 + 40)
    avg_reduction = total_capacity - df[[c for c in df.columns if 'mining_load_mw' in c]].sum(axis=1).mean()
    print(f"Average Mining Load Reduction: {avg_reduction:.1f} MW ({avg_reduction/total_capacity*100:.1f}%)")


if __name__ == "__main__":
    # Run co-simulation
    world = run_cosimulation(duration_hours=24)
    
    # Analyze results
    analyze_results()