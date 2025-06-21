import numpy as np
from collections import deque
import random
import torch
import torch as th
from .gumbel_softmax import gumbel_softmax
from .model import Critics, Actors

import os
import numpy as np
import sys
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from .networks import Critic, Actor, DDQN_Net
import math
import copy
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum





# Removed TensorFlow logging configuration as we're using PyTorch
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# --------------------------------------------------DSAC------------------------------------------------------------
class DSAC:
    def __init__(self, env, config, agent_id):
        self.env = env
        self.agent_id_ = agent_id
        self.avg_reward_per_house = []
        self.code_model = config.get('DRL', 'code_model')
        self.best_score = -np.inf
        self.best_diffpro = -np.inf
        self.best_battery = -np.inf

        # Original state and action sizes
        self.original_state_size = self.env.observation_space.shape[0]
        self.original_action_size = self.env.action_space.n
        
        # Extended state and action sizes for PoUW
        self.pouw_enabled = config.getboolean('PoUW', 'enabled', fallback=False)
        if self.pouw_enabled:
            # Check if environment already has extended state space
            if self.original_state_size == 12:
                # Environment already includes PoUW features
                self.state_size = self.original_state_size
                self.action_size = self.original_action_size
            else:
                # Need to extend state for PoUW features
                # Extended state: original + grid_urgency, energy_price, mining_difficulty, compute_power, task_priority, task_reward
                self.state_size = self.original_state_size + 6
                # Extended actions: original + grid_task (5), hybrid_mode (6)
                self.action_size = self.original_action_size + 2
        else:
            self.state_size = self.original_state_size
            self.action_size = self.original_action_size
            
        state_size = self.state_size
        action_size = self.action_size
        
        # PoUW-specific attributes
        self.grid_tasks_completed = 0
        self.stability_score = 0.0
        self.sgx_interface = None
        self.task_scheduler = None
        self.current_grid_task = None
        self.hybrid_mode_active = False
        self.nft_count = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("-----------------------",device)

        self.gamma = 0.99
        self.tau = 1e-2
        hidden_size = 256
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # Actor Network

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize PoUW components if enabled
        if self.pouw_enabled:
            self._initialize_pouw_components(config)

    def _initialize_pouw_components(self, config):
        """Initialize PoUW-specific components"""
        try:
            # Import PoUW modules
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from sgx.sgx_interface import SGXInterface
            from task_scheduler.task_scheduler import DynamicTaskScheduler
            
            # Initialize SGX interface
            self.sgx_interface = SGXInterface()
            self.sgx_interface.create_enclave()
            
            # Initialize task scheduler
            energy_pool_address = config.get('PoUW', 'energy_pool_address', fallback='')
            pouw_contract_address = config.get('PoUW', 'pouw_contract_address', fallback='')
            
            if energy_pool_address and pouw_contract_address:
                self.task_scheduler = DynamicTaskScheduler(
                    energy_pool_address=energy_pool_address,
                    pouw_contract_address=pouw_contract_address
                )
            
            # PoUW configuration
            self.task_threshold = config.getfloat('PoUW', 'task_threshold', fallback=0.7)
            self.energy_threshold = config.getfloat('PoUW', 'energy_threshold', fallback=0.3)
            self.hybrid_ratio = config.getfloat('PoUW', 'hybrid_ratio', fallback=0.5)
            
        except Exception as e:
            print(f"Warning: Failed to initialize PoUW components: {e}")
            self.pouw_enabled = False
    
    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        # Handle extended state for PoUW
        if self.pouw_enabled and len(state) == self.original_state_size:
            # Extend state with PoUW features
            state = self._extend_state_for_pouw(state)
        
        state_tensor = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            action = self.actor_local.get_det_action(state_tensor)
        
        action_value = action.numpy()
        
        # Handle PoUW action selection
        if self.pouw_enabled:
            action_value = self._process_pouw_action(action_value, state)
            
        return action_value

    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_Q = torch.min(q1, q2)
        actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q)).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (
                        torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))

            # Compute critic loss
        q1 = self.critic1(states)
        q2 = self.critic2(states)

        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())

        critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)

        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()

        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    ##############################################################################
    def update_agent_network(self, weights):
        self.train_network.set_weights(weights)
    
    # PoUW-specific methods
    def _extend_state_for_pouw(self, state):
        """Extend state with PoUW features"""
        # Default PoUW features
        grid_urgency = 0.5
        energy_price = 0.3
        mining_difficulty = 1.0
        compute_power = 1.0
        task_priority = 0.0
        task_reward = 0.0
        
        # Get real values if task scheduler is available
        if self.task_scheduler:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                grid_status = loop.run_until_complete(self.task_scheduler.fetch_grid_status())
                grid_urgency = grid_status.get('urgency', 0.5)
                energy_price = grid_status.get('energy_price', 0.3)
                loop.close()
            except:
                pass
        
        # Get current task info
        if self.current_grid_task:
            task_priority = self.current_grid_task.priority.value / 3.0
            task_reward = min(self.current_grid_task.reward / 10.0, 1.0)
        
        # Extend state
        pouw_features = np.array([
            grid_urgency,
            energy_price,
            mining_difficulty,
            compute_power,
            task_priority,
            task_reward
        ])
        
        return np.concatenate([state, pouw_features])
    
    def _process_pouw_action(self, action, state):
        """Process action for PoUW mode"""
        # Extract PoUW state features
        grid_urgency = state[self.original_state_size] if len(state) > self.original_state_size else 0.5
        
        # Override action for critical grid conditions
        if grid_urgency > 0.9 and self.current_grid_task:
            return 5  # Force grid task
        
        # Convert continuous action to discrete if needed
        if isinstance(action, np.ndarray) and action.shape:
            action = int(action[0] * self.action_size)
        
        return np.clip(action, 0, self.action_size - 1)
    
    def execute_grid_task(self, task_data):
        """Execute a grid optimization task using SGX"""
        if not self.pouw_enabled or not self.sgx_interface:
            return None
        
        try:
            # Execute task in SGX enclave
            proof = self.sgx_interface.execute_grid_task(
                task_type=task_data['task_type'],
                task_data=task_data['data'],
                task_id=task_data['task_id']
            )
            
            # Update metrics
            self.grid_tasks_completed += 1
            self.stability_score += proof.grid_impact
            
            # Check for NFT reward (every 10 tasks)
            if self.grid_tasks_completed % 10 == 0:
                self.nft_count += 1
                print(f"Agent {self.agent_id_} earned NFT #{self.nft_count}!")
            
            return {
                'success': True,
                'proof': proof,
                'grid_impact': proof.grid_impact,
                'energy_used': proof.energy_used
            }
            
        except Exception as e:
            print(f"Grid task execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def set_hybrid_mode(self, enabled, ratio=0.5):
        """Enable/disable hybrid mining-grid mode"""
        self.hybrid_mode_active = enabled
        self.hybrid_ratio = ratio
    
    def get_pouw_metrics(self):
        """Get PoUW-specific metrics"""
        return {
            'grid_tasks_completed': self.grid_tasks_completed,
            'stability_score': self.stability_score,
            'nft_count': self.nft_count,
            'hybrid_mode': self.hybrid_mode_active,
            'sgx_status': self.sgx_interface.status.name if self.sgx_interface else 'DISABLED'
        }
    
    def calculate_pouw_reward(self, action, state, next_state, original_reward, task_result=None):
        """Calculate enhanced reward for PoUW mode"""
        if not self.pouw_enabled:
            return original_reward
        
        reward = original_reward
        
        # Extract PoUW state features
        if len(state) > self.original_state_size:
            grid_urgency = state[self.original_state_size]
            energy_price = state[self.original_state_size + 1]
        else:
            grid_urgency = 0.5
            energy_price = 0.3
        
        # Grid task reward (action 5)
        if action == 5 and task_result:
            if task_result.get('success', False):
                # Base reward for successful grid task
                reward += task_result.get('grid_impact', 0) * 0.5
                
                # Bonus for high urgency situations
                if grid_urgency > 0.8:
                    reward += 0.2
                
                # Energy efficiency bonus
                energy_used = task_result.get('energy_used', 0)
                if energy_used < 0.05:  # Very efficient
                    reward += 0.1
            else:
                # Penalty for failed grid task
                reward -= 0.5
        
        # Hybrid mode reward (action 6)
        elif action == 6:
            # Balance between mining profit and grid support
            mining_component = -energy_price * 0.5  # Energy cost for mining
            grid_component = grid_urgency * 0.3     # Grid support value
            reward += mining_component + grid_component
        
        # Penalty for ignoring critical grid situations
        if grid_urgency > 0.9 and action < 5:
            reward -= 0.3
        
        return reward
    
    def step_with_pouw(self, env, action):
        """Enhanced step function that handles PoUW actions"""
        task_result = None
        
        # Handle grid task action
        if action == 5 and self.pouw_enabled:
            # Execute current grid task if available
            if self.current_grid_task:
                task_data = {
                    'task_id': self.current_grid_task.task_id,
                    'task_type': self.current_grid_task.task_type.name.lower(),
                    'data': {
                        'voltage': 230 + np.random.normal(0, 5),
                        'frequency': 50 + np.random.normal(0, 0.2),
                        'load': 0.8 + np.random.normal(0, 0.1),
                        'generation': 0.85 + np.random.normal(0, 0.1)
                    }
                }
                task_result = self.execute_grid_task(task_data)
                
                # Clear current task after execution
                self.current_grid_task = None
            
            # Map to a safe DEM action (e.g., maintain current state)
            dem_action = 0  # Default to charging battery
        
        # Handle hybrid mode action
        elif action == 6 and self.pouw_enabled:
            # In hybrid mode, split resources between mining and grid support
            self.set_hybrid_mode(True, self.hybrid_ratio)
            dem_action = 3  # Sell excess energy (simplified)
        
        else:
            # Regular DEM action
            dem_action = min(action, self.original_action_size - 1)
        
        # Execute action in environment
        next_state, reward, done, info = env.step(dem_action)
        
        # Calculate enhanced reward
        if self.pouw_enabled:
            reward = self.calculate_pouw_reward(action, env.state, next_state, reward, task_result)
        
        # Add task result to info if available
        if task_result:
            info['task_result'] = task_result
        
        return next_state, reward, done, info
