"""
Enhanced FDRL Agent with Proof of Useful Work capabilities
Combines mining optimization with grid stability tasks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import base DEM agent components (simplified for demo)
# In production, would import from actual DEM modules


class TaskType(Enum):
    """Grid optimization task types"""
    STABILITY_SIMULATION = 0
    RENEWABLE_FORECAST = 1
    LOAD_BALANCE = 2
    FREQUENCY_REGULATION = 3
    VOLTAGE_OPTIMIZATION = 4
    BITCOIN_MINING = 5


class Priority(Enum):
    """Task priority levels"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class GridTask:
    """Grid optimization task structure"""
    task_id: int
    task_type: TaskType
    priority: Priority
    reward: float
    compute_requirement: float
    deadline: int
    energy_estimate: float


@dataclass
class MinerState:
    """Enhanced state including grid and mining information"""
    battery_level: float
    solar_production: float
    house_consumption: float
    hour_of_day: int
    grid_urgency: float
    energy_price: float
    mining_difficulty: float
    pending_tasks: List[GridTask]
    compute_power: float
    current_task: Optional[GridTask] = None


class PoUWAgent:
    """
    Enhanced DSAC agent with Proof of Useful Work capabilities
    Extends the base DEM agent to handle both mining and grid tasks
    """
    
    def __init__(self, config: Dict):
        # Extended state and action dimensions
        state_dim = 10  # Original 4 + grid_urgency, energy_price, mining_difficulty, compute_power, task_priority, task_reward
        action_dim = 7  # Original 5 + grid_task, hybrid_mode
        
        # Initialize agent parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = config.get('n_agents', 1)
        self.agent_id = config.get('agent_id', 0)
        self.device = config.get('device', 'cpu')
        
        # Initialize memory buffer
        self.memory = deque(maxlen=10000)
        
        self.config = config
        self.task_history = deque(maxlen=100)
        self.stability_score = 0
        self.grid_tasks_completed = 0
        self.mining_efficiency = 1.0
        
        # Task scheduling parameters
        self.task_threshold = config.get('task_threshold', 0.7)
        self.energy_threshold = config.get('energy_threshold', 0.3)
        self.hybrid_ratio = config.get('hybrid_ratio', 0.5)
        
        # Privacy parameters for federated learning
        self.privacy_budget = config.get('privacy_epsilon', 1.0)
        self.noise_scale = config.get('noise_scale', 0.1)
        
    def process_state(self, miner_state: MinerState) -> np.ndarray:
        """Convert MinerState to neural network input"""
        # Basic state features
        basic_features = np.array([
            miner_state.battery_level / 2.0,  # Normalize to [0, 1]
            miner_state.solar_production,
            miner_state.house_consumption,
            miner_state.hour_of_day / 24.0,
            miner_state.grid_urgency,
            miner_state.energy_price,
            miner_state.mining_difficulty,
            miner_state.compute_power
        ])
        
        # Task features (highest priority task)
        if miner_state.pending_tasks:
            highest_priority_task = max(miner_state.pending_tasks, 
                                      key=lambda t: (t.priority.value, t.reward))
            task_features = np.array([
                highest_priority_task.priority.value / 3.0,  # Normalize priority
                highest_priority_task.reward / 10.0  # Normalize reward
            ])
        else:
            task_features = np.array([0.0, 0.0])
        
        return np.concatenate([basic_features, task_features])
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, Dict]:
        """
        Select action with enhanced action space
        Actions: 0-4 (original DEM actions), 5 (grid task), 6 (hybrid mode)
        """
        # Simplified action selection for demo
        action_probs = np.ones(self.action_dim) / self.action_dim
        action = np.random.choice(self.action_dim, p=action_probs)
        action_info = {'method': 'random' if not evaluate else 'greedy'}
        
        # Extract state components
        grid_urgency = state[4]
        energy_price = state[5]
        task_priority = state[8]
        
        # Override action selection for critical grid conditions
        if grid_urgency > 0.9 and task_priority > 0.8:
            action = 5  # Force grid task
            action_info['override'] = 'critical_grid'
        elif energy_price < self.energy_threshold and grid_urgency < 0.3:
            # Prefer mining when energy is cheap and grid is stable
            if action > 4:
                action = np.random.choice([0, 1, 2, 3, 4])
                action_info['override'] = 'cheap_energy_mining'
        
        return action, action_info
    
    def execute_grid_task(self, task: GridTask, state: MinerState) -> Dict:
        """
        Execute a grid optimization task
        Returns execution results including proof generation
        """
        result = {
            'task_id': task.task_id,
            'task_type': task.task_type.name,
            'energy_used': 0.0,
            'computation_time': 0.0,
            'grid_impact': 0.0,
            'proof_hash': None,
            'success': False
        }
        
        # Simulate task execution based on type
        if task.task_type == TaskType.STABILITY_SIMULATION:
            result['grid_impact'] = self._run_stability_simulation(state)
            result['energy_used'] = task.energy_estimate * 0.9
        elif task.task_type == TaskType.RENEWABLE_FORECAST:
            result['grid_impact'] = self._run_renewable_forecast(state)
            result['energy_used'] = task.energy_estimate * 0.7
        elif task.task_type == TaskType.LOAD_BALANCE:
            result['grid_impact'] = self._run_load_balance(state)
            result['energy_used'] = task.energy_estimate * 0.8
        
        # Generate proof (simplified - would use SGX in production)
        result['proof_hash'] = self._generate_task_proof(result)
        result['success'] = result['grid_impact'] > 0
        
        # Update agent metrics
        if result['success']:
            self.grid_tasks_completed += 1
            self.stability_score += result['grid_impact']
            self.task_history.append(task)
        
        return result
    
    def _run_stability_simulation(self, state: MinerState) -> float:
        """Run grid stability simulation"""
        # Simplified simulation - in production would run actual grid models
        base_impact = np.random.uniform(0.5, 2.0)  # 0.5-2% improvement
        
        # Adjust based on current grid conditions
        if state.grid_urgency > 0.8:
            base_impact *= 1.5
        
        return base_impact
    
    def _run_renewable_forecast(self, state: MinerState) -> float:
        """Run renewable energy forecast"""
        # Forecast accuracy improves grid planning
        accuracy = np.random.uniform(0.85, 0.95)
        impact = (accuracy - 0.8) * 10  # Convert to percentage impact
        
        # Better forecasts during high solar production hours
        if 8 <= state.hour_of_day <= 16:
            impact *= 1.2
        
        return max(0, impact)
    
    def _run_load_balance(self, state: MinerState) -> float:
        """Run load balancing optimization"""
        # Impact based on consumption vs production balance
        imbalance = abs(state.solar_production - state.house_consumption)
        impact = imbalance * 0.5
        
        return min(impact, 3.0)  # Cap at 3% improvement
    
    def _generate_task_proof(self, result: Dict) -> str:
        """Generate cryptographic proof of task completion"""
        # Simplified proof generation - would use SGX attestation in production
        import hashlib
        
        proof_data = f"{result['task_id']}:{result['grid_impact']}:{result['energy_used']}"
        return hashlib.sha256(proof_data.encode()).hexdigest()
    
    def calculate_reward(self, action: int, state: MinerState, 
                        next_state: MinerState, task_result: Optional[Dict] = None) -> float:
        """
        Calculate reward considering both energy and grid optimization objectives
        """
        reward = 0.0
        
        # Original DEM rewards for actions 0-4
        if action < 5:
            # Energy cost/benefit calculation
            energy_diff = state.solar_production - state.house_consumption
            
            if action == 0:  # Charge battery
                if state.battery_level < 1.8:
                    reward = 0.1 * (2.0 - state.battery_level)
                else:
                    reward = -0.5  # Penalty for overcharging
            elif action == 1:  # Charge and sell
                if energy_diff > 0 and state.battery_level < 1.8:
                    reward = 0.1 + state.energy_price * min(energy_diff, 0.5)
                else:
                    reward = -0.3
            elif action == 2:  # Discharge battery
                if state.battery_level > 0.2 and energy_diff < 0:
                    reward = 0.1 * state.battery_level
                else:
                    reward = -1.0  # Heavy penalty for over-discharge
            elif action == 3:  # Sell energy
                if energy_diff > 0:
                    reward = state.energy_price * energy_diff
                else:
                    reward = -0.5
            elif action == 4:  # Buy energy
                if energy_diff < 0:
                    reward = -state.energy_price * abs(energy_diff)
                else:
                    reward = -1.0
        
        # Grid task rewards
        elif action == 5 and task_result:
            if task_result['success']:
                # Base reward from task
                reward = task_result.get('reward', 1.0)
                
                # Bonus for grid impact
                reward += task_result['grid_impact'] * 0.5
                
                # Efficiency bonus
                energy_efficiency = 1.0 - (task_result['energy_used'] / 
                                         (state.compute_power * 0.1))
                reward += energy_efficiency * 0.2
            else:
                reward = -0.5  # Penalty for failed task
        
        # Hybrid mode reward
        elif action == 6:
            # Balance between mining profit and grid support
            mining_reward = -state.energy_price * state.compute_power * 0.5
            grid_support_reward = state.grid_urgency * 0.3
            reward = mining_reward + grid_support_reward
        
        return reward
    
    def update_federated(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Update model with federated learning while preserving privacy
        """
        # Get current weights
        local_weights = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }
        
        # Calculate weight updates with differential privacy
        weight_updates = {}
        for model_name, model_weights in local_weights.items():
            weight_updates[model_name] = {}
            for param_name, param in model_weights.items():
                if param_name in global_weights.get(model_name, {}):
                    # Calculate update
                    update = param - global_weights[model_name][param_name]
                    
                    # Add Gaussian noise for differential privacy
                    noise = torch.randn_like(update) * self.noise_scale / self.privacy_budget
                    noisy_update = update + noise
                    
                    weight_updates[model_name][param_name] = noisy_update
        
        return weight_updates
    
    def share_experience_federated(self, num_samples: int = 100) -> List[Tuple]:
        """
        Share experience samples for federated learning
        Privacy-preserving: only share state-action pairs, not raw sensor data
        """
        if len(self.memory) < num_samples:
            return []
        
        # Sample experiences
        samples = random.sample(self.memory, num_samples)
        
        # Filter sensitive information
        filtered_samples = []
        for state, action, reward, next_state, done in samples:
            # Anonymize location-specific features
            anon_state = state.copy()
            anon_state[2] = np.clip(anon_state[2], 0, 1)  # Normalize consumption
            anon_state[1] = np.clip(anon_state[1], 0, 1)  # Normalize production
            
            anon_next_state = next_state.copy()
            anon_next_state[2] = np.clip(anon_next_state[2], 0, 1)
            anon_next_state[1] = np.clip(anon_next_state[1], 0, 1)
            
            filtered_samples.append((anon_state, action, reward, anon_next_state, done))
        
        return filtered_samples
    
    def get_metrics(self) -> Dict:
        """Get agent performance metrics"""
        return {
            'stability_score': self.stability_score,
            'grid_tasks_completed': self.grid_tasks_completed,
            'mining_efficiency': self.mining_efficiency,
            'avg_reward': np.mean([r for _, _, r, _, _ in list(self.memory)[-100:]]) if self.memory else 0,
            'task_success_rate': self.grid_tasks_completed / max(len(self.task_history), 1)
        }