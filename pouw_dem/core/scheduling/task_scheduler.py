"""
Dynamic Task Scheduler for PoUW-DEM Integration
Manages task allocation between mining and grid optimization
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import asyncio
from web3 import Web3
import json


class TaskType(Enum):
    """Task types matching smart contract"""
    STABILITY_SIMULATION = 0
    RENEWABLE_FORECAST = 1
    LOAD_BALANCE = 2
    FREQUENCY_REGULATION = 3
    VOLTAGE_OPTIMIZATION = 4
    BITCOIN_MINING = 5


class Priority(Enum):
    """Priority levels matching smart contract"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(order=True)
class ScheduledTask:
    """Task with scheduling metadata"""
    priority_score: float = field(compare=True)
    task_id: int = field(compare=False)
    task_type: TaskType = field(compare=False)
    priority: Priority = field(compare=False)
    reward: float = field(compare=False)
    compute_requirement: float = field(compare=False)
    deadline: datetime = field(compare=False)
    energy_estimate: float = field(compare=False)
    created_at: datetime = field(default_factory=datetime.now, compare=False)
    assigned_to: Optional[str] = field(default=None, compare=False)
    contract_address: Optional[str] = field(default=None, compare=False)


class DynamicTaskScheduler:
    """
    Intelligent task scheduler that allocates resources between
    Bitcoin mining and grid optimization tasks
    """
    
    def __init__(self, 
                 energy_pool_address: str,
                 pouw_contract_address: str,
                 web3_provider: str = "https://eth-mainnet.g.alchemy.com/v2/exAp0m_LKHnmcM2Uni2BbYH5cLgBYaV2"):
        
        self.energy_pool_address = energy_pool_address
        self.pouw_contract_address = pouw_contract_address
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        
        # Task queues by priority
        self.task_queues = {
            Priority.CRITICAL: [],
            Priority.HIGH: [],
            Priority.MEDIUM: [],
            Priority.LOW: []
        }
        
        # Mining pool
        self.mining_queue = []
        
        # Scheduler configuration
        self.config = {
            'grid_urgency_threshold': 0.7,
            'energy_surplus_threshold': 0.3,
            'min_mining_profitability': 0.1,
            'task_timeout_minutes': 60,
            'max_concurrent_tasks': 10
        }
        
        # Performance metrics
        self.metrics = {
            'total_tasks_scheduled': 0,
            'grid_tasks_completed': 0,
            'mining_blocks_found': 0,
            'total_energy_used': 0.0,
            'total_rewards_earned': 0.0,
            'avg_task_completion_time': 0.0
        }
        
        # Active tasks tracking
        self.active_tasks: Dict[int, ScheduledTask] = {}
        
        # Load contract ABIs (simplified for demo)
        self.energy_pool_abi = self._load_energy_pool_abi()
        self.pouw_abi = self._load_pouw_abi()
        
    def _load_energy_pool_abi(self) -> List:
        """Load EnergyPool contract ABI"""
        # Simplified ABI for demo
        return [{
            "name": "intervals",
            "type": "function",
            "inputs": [{"name": "index", "type": "uint256"}],
            "outputs": [
                {"name": "energyCommittedProduction", "type": "uint256"},
                {"name": "energyCommittedConsumption", "type": "uint256"},
                {"name": "current_interval_block_number", "type": "uint256"},
                {"name": "energyProduced", "type": "uint256"},
                {"name": "energyConsumed", "type": "uint256"}
            ]
        }]
    
    def _load_pouw_abi(self) -> List:
        """Load ProofOfUsefulWork contract ABI"""
        # Simplified ABI for demo
        return [{
            "name": "getPendingTasksByPriority",
            "type": "function",
            "inputs": [{"name": "priority", "type": "uint8"}],
            "outputs": [{"name": "taskIds", "type": "uint256[]"}]
        }]
    
    async def fetch_grid_status(self) -> Dict:
        """Fetch current grid status from EnergyPool contract"""
        try:
            contract = self.w3.eth.contract(
                address=self.energy_pool_address,
                abi=self.energy_pool_abi
            )
            
            # Get current interval data
            interval_data = contract.functions.intervals(0).call()
            
            production = interval_data[3]
            consumption = interval_data[4]
            surplus = production - consumption
            
            # Calculate grid urgency based on supply-demand balance
            if consumption > 0:
                urgency = max(0, min(1, (consumption - production) / consumption))
            else:
                urgency = 0
            
            return {
                'production': production,
                'consumption': consumption,
                'surplus': surplus,
                'urgency': urgency,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching grid status: {e}")
            return {
                'production': 0,
                'consumption': 0,
                'surplus': 0,
                'urgency': 0.5,
                'timestamp': datetime.now()
            }
    
    async def fetch_pending_tasks(self) -> List[ScheduledTask]:
        """Fetch pending tasks from ProofOfUsefulWork contract"""
        tasks = []
        
        try:
            contract = self.w3.eth.contract(
                address=self.pouw_contract_address,
                abi=self.pouw_abi
            )
            
            # Fetch tasks by priority
            for priority in Priority:
                task_ids = contract.functions.getPendingTasksByPriority(priority.value).call()
                
                for task_id in task_ids:
                    # In production, would fetch full task details
                    task = ScheduledTask(
                        priority_score=self._calculate_priority_score(priority, 1.0),
                        task_id=task_id,
                        task_type=TaskType.STABILITY_SIMULATION,  # Simplified
                        priority=priority,
                        reward=1.0 * (priority.value + 1),
                        compute_requirement=0.5,
                        deadline=datetime.now() + timedelta(hours=1),
                        energy_estimate=0.1,
                        contract_address=self.pouw_contract_address
                    )
                    tasks.append(task)
                    
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            
        return tasks
    
    def _calculate_priority_score(self, 
                                 priority: Priority, 
                                 reward: float,
                                 deadline_hours: float = 1.0) -> float:
        """Calculate composite priority score for task scheduling"""
        # Base score from priority level
        base_score = (priority.value + 1) * 100
        
        # Reward factor
        reward_factor = reward * 10
        
        # Urgency factor based on deadline
        urgency_factor = 50 / max(deadline_hours, 0.1)
        
        return base_score + reward_factor + urgency_factor
    
    def add_task(self, task: ScheduledTask) -> None:
        """Add a task to the appropriate queue"""
        heapq.heappush(self.task_queues[task.priority], 
                      (-task.priority_score, task))
        self.metrics['total_tasks_scheduled'] += 1
    
    def add_mining_task(self, 
                       difficulty: float,
                       block_reward: float,
                       energy_price: float) -> None:
        """Add a mining task to the queue"""
        # Calculate mining profitability
        energy_cost = energy_price * 0.5  # Assume 0.5 kW for mining
        profitability = block_reward - energy_cost
        
        if profitability > self.config['min_mining_profitability']:
            mining_task = ScheduledTask(
                priority_score=profitability * 100,
                task_id=-1,  # Negative ID for mining tasks
                task_type=TaskType.BITCOIN_MINING,
                priority=Priority.LOW,
                reward=block_reward,
                compute_requirement=1.0,
                deadline=datetime.now() + timedelta(hours=24),
                energy_estimate=0.5
            )
            heapq.heappush(self.mining_queue, (-mining_task.priority_score, mining_task))
    
    async def schedule_next_task(self, 
                               miner_capabilities: Dict) -> Optional[ScheduledTask]:
        """
        Schedule the next task based on current conditions and miner capabilities
        """
        # Fetch current grid status
        grid_status = await self.fetch_grid_status()
        
        # Fetch pending blockchain tasks
        pending_tasks = await self.fetch_pending_tasks()
        for task in pending_tasks:
            self.add_task(task)
        
        # Decision logic
        if grid_status['urgency'] > self.config['grid_urgency_threshold']:
            # High grid urgency - prioritize grid tasks
            return self._get_next_grid_task(miner_capabilities)
            
        elif grid_status['surplus'] > self.config['energy_surplus_threshold']:
            # Energy surplus - can do either mining or grid tasks
            grid_task = self._get_next_grid_task(miner_capabilities)
            mining_task = self._get_next_mining_task(miner_capabilities)
            
            if grid_task and mining_task:
                # Compare profitability
                if grid_task.reward > mining_task.reward:
                    return grid_task
                else:
                    return mining_task
            elif grid_task:
                return grid_task
            else:
                return mining_task
                
        else:
            # Default to mining if available
            return self._get_next_mining_task(miner_capabilities)
    
    def _get_next_grid_task(self, 
                           miner_capabilities: Dict) -> Optional[ScheduledTask]:
        """Get the next suitable grid task"""
        # Check queues from highest to lowest priority
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            queue = self.task_queues[priority]
            
            while queue:
                _, task = heapq.heappop(queue)
                
                # Check if task matches miner capabilities
                if (task.compute_requirement <= miner_capabilities.get('compute_power', 1.0) and
                    task.deadline > datetime.now()):
                    
                    # Assign task
                    task.assigned_to = miner_capabilities.get('address')
                    self.active_tasks[task.task_id] = task
                    return task
                    
        return None
    
    def _get_next_mining_task(self, 
                             miner_capabilities: Dict) -> Optional[ScheduledTask]:
        """Get the next mining task"""
        while self.mining_queue:
            _, task = heapq.heappop(self.mining_queue)
            
            if task.compute_requirement <= miner_capabilities.get('compute_power', 1.0):
                task.assigned_to = miner_capabilities.get('address')
                self.active_tasks[task.task_id] = task
                return task
                
        return None
    
    def complete_task(self, 
                     task_id: int, 
                     success: bool,
                     energy_used: float,
                     reward_earned: float) -> None:
        """Mark a task as completed and update metrics"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            if success:
                if task.task_type == TaskType.BITCOIN_MINING:
                    self.metrics['mining_blocks_found'] += 1
                else:
                    self.metrics['grid_tasks_completed'] += 1
                
                self.metrics['total_rewards_earned'] += reward_earned
            
            self.metrics['total_energy_used'] += energy_used
            
            # Update average completion time
            completion_time = (datetime.now() - task.created_at).total_seconds() / 60
            current_avg = self.metrics['avg_task_completion_time']
            total_completed = self.metrics['grid_tasks_completed'] + self.metrics['mining_blocks_found']
            
            if total_completed > 0:
                self.metrics['avg_task_completion_time'] = (
                    (current_avg * (total_completed - 1) + completion_time) / total_completed
                )
            
            # Remove from active tasks
            del self.active_tasks[task_id]
    
    def get_scheduler_metrics(self) -> Dict:
        """Get scheduler performance metrics"""
        return {
            **self.metrics,
            'active_tasks': len(self.active_tasks),
            'queued_grid_tasks': sum(len(q) for q in self.task_queues.values()),
            'queued_mining_tasks': len(self.mining_queue),
            'efficiency_ratio': self.metrics['total_rewards_earned'] / 
                               max(self.metrics['total_energy_used'], 1)
        }
    
    def optimize_scheduling_params(self, 
                                  historical_data: List[Dict]) -> None:
        """
        Use historical data to optimize scheduling parameters
        This could use reinforcement learning in production
        """
        if not historical_data:
            return
        
        # Analyze historical performance
        grid_task_success_rate = np.mean([
            d.get('grid_success', 0) for d in historical_data
        ])
        
        avg_energy_price = np.mean([
            d.get('energy_price', 0.3) for d in historical_data
        ])
        
        # Adjust thresholds based on performance
        if grid_task_success_rate > 0.8:
            # High success rate - can be more aggressive with grid tasks
            self.config['grid_urgency_threshold'] *= 0.95
        elif grid_task_success_rate < 0.5:
            # Low success rate - be more conservative
            self.config['grid_urgency_threshold'] *= 1.05
        
        # Adjust mining profitability threshold based on energy prices
        self.config['min_mining_profitability'] = avg_energy_price * 0.5
    
    async def run_scheduler_loop(self, 
                                interval_seconds: int = 60) -> None:
        """Main scheduler loop"""
        while True:
            try:
                # Clean up expired tasks
                self._cleanup_expired_tasks()
                
                # Re-balance queues if needed
                self._rebalance_queues()
                
                # Log metrics
                metrics = self.get_scheduler_metrics()
                print(f"Scheduler metrics: {metrics}")
                
            except Exception as e:
                print(f"Scheduler loop error: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    def _cleanup_expired_tasks(self) -> None:
        """Remove expired tasks from queues"""
        current_time = datetime.now()
        
        for priority in Priority:
            queue = self.task_queues[priority]
            valid_tasks = []
            
            while queue:
                _, task = heapq.heappop(queue)
                if task.deadline > current_time:
                    valid_tasks.append((-task.priority_score, task))
            
            # Re-build queue with valid tasks
            self.task_queues[priority] = valid_tasks
            heapq.heapify(self.task_queues[priority])
    
    def _rebalance_queues(self) -> None:
        """Rebalance task queues based on current conditions"""
        # Move critical tasks that are close to deadline
        for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            queue = self.task_queues[priority]
            escalated = []
            
            for i, (_, task) in enumerate(queue):
                time_remaining = (task.deadline - datetime.now()).total_seconds() / 3600
                
                # Escalate if less than 30 minutes remaining
                if time_remaining < 0.5 and priority != Priority.CRITICAL:
                    task.priority = Priority.CRITICAL
                    task.priority_score = self._calculate_priority_score(
                        Priority.CRITICAL, task.reward, time_remaining
                    )
                    escalated.append(i)
                    self.add_task(task)
            
            # Remove escalated tasks from original queue
            for i in reversed(escalated):
                queue.pop(i)
            
            heapq.heapify(queue)