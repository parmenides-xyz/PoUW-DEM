// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "./interfaces/IEnergyProfile.sol";
import "./interfaces/IEnergyPool.sol";

/**
 * @title ProofOfUsefulWork
 * @dev Core contract for managing grid optimization tasks and miner rewards
 */
contract ProofOfUsefulWork is Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    // Task types for grid optimization
    enum TaskType {
        STABILITY_SIMULATION,
        RENEWABLE_FORECAST,
        LOAD_BALANCE,
        FREQUENCY_REGULATION,
        VOLTAGE_OPTIMIZATION
    }
    
    // Task priority levels
    enum Priority {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }
    
    struct GridTask {
        uint256 taskId;
        TaskType taskType;
        Priority priority;
        uint256 reward;
        bytes32 inputDataHash;
        bytes32 expectedOutputHash;
        uint256 deadline;
        uint256 minComputePower;
        bool completed;
        address assignedMiner;
        uint256 createdAt;
    }
    
    struct MinerProfile {
        uint256 computePower;
        uint256 gridTasksCompleted;
        uint256 totalRewardsEarned;
        uint256 stabilityScore;
        uint256 lastTaskTimestamp;
        address sgxAttestationKey;
        bool isActive;
        uint256 successRate; // Percentage * 100 (e.g., 9500 = 95%)
    }
    
    struct TaskProof {
        bytes32 outputHash;
        bytes sgxAttestation;
        uint256 computationTime;
        uint256 energyUsed;
        uint256 gridImpact; // Measured improvement in grid stability
    }
    
    // State variables
    Counters.Counter private _taskIdCounter;
    IEnergyProfile public energyProfileContract;
    IEnergyPool public energyPoolContract;
    
    mapping(uint256 => GridTask) public gridTasks;
    mapping(address => MinerProfile) public miners;
    mapping(uint256 => TaskProof) public taskProofs;
    mapping(address => uint256[]) public minerTaskHistory;
    
    // Task queue by priority
    mapping(Priority => uint256[]) public taskQueue;
    
    // Rewards and penalties
    uint256 public constant BASE_REWARD = 1 ether;
    uint256 public constant CRITICAL_MULTIPLIER = 5;
    uint256 public constant HIGH_MULTIPLIER = 3;
    uint256 public constant MEDIUM_MULTIPLIER = 2;
    uint256 public constant FAILURE_PENALTY = 0.1 ether;
    
    // Events
    event TaskCreated(uint256 indexed taskId, TaskType taskType, Priority priority, uint256 reward);
    event TaskAssigned(uint256 indexed taskId, address indexed miner);
    event TaskCompleted(uint256 indexed taskId, address indexed miner, uint256 gridImpact);
    event MinerRegistered(address indexed miner, uint256 computePower);
    event StabilityNFTMinted(address indexed miner, uint256 tokenId, uint256 stabilityScore);
    event TaskFailed(uint256 indexed taskId, address indexed miner, string reason);
    
    constructor(address _energyProfileAddress, address _energyPoolAddress) {
        energyProfileContract = IEnergyProfile(_energyProfileAddress);
        energyPoolContract = IEnergyPool(_energyPoolAddress);
    }
    
    /**
     * @dev Register a new miner with SGX attestation
     */
    function registerMiner(
        uint256 computePower,
        address sgxAttestationKey
    ) external {
        require(!miners[msg.sender].isActive, "Miner already registered");
        require(computePower > 0, "Invalid compute power");
        require(sgxAttestationKey != address(0), "Invalid SGX key");
        
        miners[msg.sender] = MinerProfile({
            computePower: computePower,
            gridTasksCompleted: 0,
            totalRewardsEarned: 0,
            stabilityScore: 0,
            lastTaskTimestamp: block.timestamp,
            sgxAttestationKey: sgxAttestationKey,
            isActive: true,
            successRate: 10000 // Start at 100%
        });
        
        emit MinerRegistered(msg.sender, computePower);
    }
    
    /**
     * @dev Create a new grid optimization task
     */
    function createTask(
        TaskType taskType,
        Priority priority,
        bytes32 inputDataHash,
        bytes32 expectedOutputHash,
        uint256 duration,
        uint256 minComputePower
    ) external onlyOwner returns (uint256) {
        _taskIdCounter.increment();
        uint256 taskId = _taskIdCounter.current();
        
        uint256 reward = calculateReward(priority);
        
        gridTasks[taskId] = GridTask({
            taskId: taskId,
            taskType: taskType,
            priority: priority,
            reward: reward,
            inputDataHash: inputDataHash,
            expectedOutputHash: expectedOutputHash,
            deadline: block.timestamp + duration,
            minComputePower: minComputePower,
            completed: false,
            assignedMiner: address(0),
            createdAt: block.timestamp
        });
        
        // Add to priority queue
        taskQueue[priority].push(taskId);
        
        emit TaskCreated(taskId, taskType, priority, reward);
        return taskId;
    }
    
    /**
     * @dev Miner requests task assignment based on capabilities
     */
    function requestTask() external returns (uint256) {
        require(miners[msg.sender].isActive, "Miner not registered");
        MinerProfile storage miner = miners[msg.sender];
        
        // Find highest priority task matching miner capabilities
        uint256 assignedTaskId = 0;
        
        // Check from CRITICAL to LOW priority
        for (uint i = uint(Priority.CRITICAL); i >= uint(Priority.LOW); i--) {
            Priority p = Priority(i);
            uint256[] storage queue = taskQueue[p];
            
            for (uint j = 0; j < queue.length; j++) {
                uint256 taskId = queue[j];
                GridTask storage task = gridTasks[taskId];
                
                if (!task.completed && 
                    task.assignedMiner == address(0) &&
                    miner.computePower >= task.minComputePower &&
                    block.timestamp < task.deadline) {
                    
                    // Assign task
                    task.assignedMiner = msg.sender;
                    assignedTaskId = taskId;
                    
                    // Remove from queue
                    queue[j] = queue[queue.length - 1];
                    queue.pop();
                    
                    emit TaskAssigned(taskId, msg.sender);
                    return taskId;
                }
            }
            
            if (i == 0) break; // Prevent underflow
        }
        
        revert("No suitable tasks available");
    }
    
    /**
     * @dev Submit proof of task completion
     */
    function submitTaskProof(
        uint256 taskId,
        bytes32 outputHash,
        bytes calldata sgxAttestation,
        uint256 computationTime,
        uint256 energyUsed,
        uint256 gridImpact
    ) external nonReentrant {
        GridTask storage task = gridTasks[taskId];
        require(task.assignedMiner == msg.sender, "Not assigned to this task");
        require(!task.completed, "Task already completed");
        require(block.timestamp <= task.deadline, "Task deadline passed");
        
        // Verify SGX attestation
        require(verifySGXAttestation(msg.sender, sgxAttestation), "Invalid SGX attestation");
        
        // Store proof
        taskProofs[taskId] = TaskProof({
            outputHash: outputHash,
            sgxAttestation: sgxAttestation,
            computationTime: computationTime,
            energyUsed: energyUsed,
            gridImpact: gridImpact
        });
        
        // Validate output (simplified - in production would verify actual computation)
        if (outputHash == task.expectedOutputHash || gridImpact > 0) {
            // Mark task completed
            task.completed = true;
            
            // Update miner profile
            MinerProfile storage miner = miners[msg.sender];
            miner.gridTasksCompleted++;
            miner.totalRewardsEarned += task.reward;
            miner.stabilityScore += gridImpact;
            miner.lastTaskTimestamp = block.timestamp;
            
            // Update success rate
            uint256 totalTasks = minerTaskHistory[msg.sender].length + 1;
            miner.successRate = (miner.gridTasksCompleted * 10000) / totalTasks;
            
            // Record task in history
            minerTaskHistory[msg.sender].push(taskId);
            
            // Mint stability NFT through EnergyProfile
            mintStabilityNFT(msg.sender, taskId, gridImpact);
            
            // Update energy pool with computation metrics
            energyPoolContract.updateIntervalEnergyByMarket(false, energyUsed);
            
            // Transfer reward
            payable(msg.sender).transfer(task.reward);
            
            emit TaskCompleted(taskId, msg.sender, gridImpact);
        } else {
            // Task failed
            task.assignedMiner = address(0); // Reset assignment
            taskQueue[task.priority].push(taskId); // Re-queue task
            
            // Penalize miner
            MinerProfile storage miner = miners[msg.sender];
            if (miner.stabilityScore >= 10) {
                miner.stabilityScore -= 10;
            }
            
            emit TaskFailed(taskId, msg.sender, "Invalid output");
        }
    }
    
    /**
     * @dev Calculate reward based on priority
     */
    function calculateReward(Priority priority) private pure returns (uint256) {
        if (priority == Priority.CRITICAL) {
            return BASE_REWARD * CRITICAL_MULTIPLIER;
        } else if (priority == Priority.HIGH) {
            return BASE_REWARD * HIGH_MULTIPLIER;
        } else if (priority == Priority.MEDIUM) {
            return BASE_REWARD * MEDIUM_MULTIPLIER;
        } else {
            return BASE_REWARD;
        }
    }
    
    /**
     * @dev Verify SGX attestation (simplified for demo)
     */
    function verifySGXAttestation(
        address miner,
        bytes calldata attestation
    ) private view returns (bool) {
        // In production, this would verify the SGX remote attestation
        // For now, just check if attestation is non-empty and miner has SGX key
        return attestation.length > 0 && miners[miner].sgxAttestationKey != address(0);
    }
    
    /**
     * @dev Mint stability NFT through EnergyProfile contract
     */
    function mintStabilityNFT(
        address miner,
        uint256 taskId,
        uint256 gridImpact
    ) private {
        // This would call the modified EnergyProfile contract
        // For now, emit an event
        emit StabilityNFTMinted(miner, taskId, gridImpact);
    }
    
    /**
     * @dev Get pending tasks by priority
     */
    function getPendingTasksByPriority(Priority priority) external view returns (uint256[] memory) {
        return taskQueue[priority];
    }
    
    /**
     * @dev Get miner's task history
     */
    function getMinerTaskHistory(address miner) external view returns (uint256[] memory) {
        return minerTaskHistory[miner];
    }
    
    /**
     * @dev Emergency pause for critical grid events
     */
    function emergencyPause() external onlyOwner {
        // In production, would implement OpenZeppelin Pausable
        // This would halt all non-critical operations
    }
    
    /**
     * @dev Withdraw accumulated fees (for contract maintenance)
     */
    function withdrawFees() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No fees to withdraw");
        payable(owner()).transfer(balance);
    }
    
    // Receive function to accept payments
    receive() external payable {}
}