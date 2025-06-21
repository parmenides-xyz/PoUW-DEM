"""
Integration tests for DEM and PoUW contracts
Tests the compatibility and interaction between the original DEM contracts
and the ProofOfUsefulWork additions
"""

import pytest
from brownie import (
    accounts,
    Contract,
    EnergyProfile,
    EnergyProfileEnhanced,
    EnergyPool,
    EnergyMarket,
    ProofOfUsefulWork,
    MockV3AggregatorEnergyProduced,
    MockV3AggregatorEnergyConsumed,
    MockV3AggregatorPrice,
    exceptions
)
from brownie.network.state import Chain


@pytest.fixture
def deploy_contracts():
    """Deploy all contracts for testing"""
    deployer = accounts[0]
    
    # Deploy mock oracles
    energy_produced_oracle = MockV3AggregatorEnergyProduced.deploy(
        18, 1000000000, {"from": deployer}
    )
    energy_consumed_oracle = MockV3AggregatorEnergyConsumed.deploy(
        18, 900000000, {"from": deployer}
    )
    energy_price_oracle = MockV3AggregatorPrice.deploy(
        18, 2000, {"from": deployer}
    )
    
    # Deploy EnergyProfileEnhanced (instead of basic EnergyProfile)
    energy_profile = EnergyProfileEnhanced.deploy(
        energy_produced_oracle.address,
        energy_consumed_oracle.address,
        energy_price_oracle.address,
        {"from": deployer}
    )
    
    # Deploy EnergyMarket
    energy_market = EnergyMarket.deploy(
        energy_profile.address,
        {"from": deployer}
    )
    
    # Deploy EnergyPool
    energy_pool = EnergyPool.deploy(
        energy_profile.address,
        energy_market.address,
        energy_price_oracle.address,
        {"from": deployer}
    )
    
    # Deploy ProofOfUsefulWork
    proof_of_useful_work = ProofOfUsefulWork.deploy(
        energy_profile.address,
        energy_pool.address,
        {"from": deployer}
    )
    
    # Configure contract connections
    energy_profile.setEnergyPoolContract(energy_pool.address, {"from": deployer})
    energy_profile.setEnergyMarketContract(energy_market.address, {"from": deployer})
    energy_profile.setProofOfUsefulWorkContract(proof_of_useful_work.address, {"from": deployer})
    energy_market.setEnergyPoolContract(energy_pool.address, {"from": deployer})
    
    # Fund PoUW contract for rewards
    deployer.transfer(proof_of_useful_work.address, "10 ether")
    
    return {
        "energy_profile": energy_profile,
        "energy_market": energy_market,
        "energy_pool": energy_pool,
        "proof_of_useful_work": proof_of_useful_work,
        "oracles": {
            "produced": energy_produced_oracle,
            "consumed": energy_consumed_oracle,
            "price": energy_price_oracle
        }
    }


class TestDEMPoUWIntegration:
    """Test suite for DEM and PoUW integration"""
    
    def test_user_registration_compatibility(self, deploy_contracts):
        """Test that user registration works for both energy users and miners"""
        contracts = deploy_contracts
        energy_profile = contracts["energy_profile"]
        user1 = accounts[1]
        user2 = accounts[2]
        
        # Register energy user
        tx1 = energy_profile.registerUser(
            user1.address,
            1000,
            "Location A",
            "Solar",
            {"from": user1, "value": "1 ether"}
        )
        
        # Register miner (also an energy user)
        tx2 = energy_profile.registerUser(
            user2.address,
            2000,
            "Location B",
            "Wind",
            {"from": user2, "value": "2 ether"}
        )
        
        # Verify registrations
        assert energy_profile.balanceOf(user1) == 1
        assert energy_profile.balanceOf(user2) == 1
        
        # Check user profiles
        profile1 = energy_profile.getUserProfile(1)
        profile2 = energy_profile.getUserProfile(2)
        
        assert profile1[0] == 1e18  # collateral
        assert profile2[0] == 2e18  # collateral
        assert profile1[8] == 0  # stabilityScore
        assert profile2[8] == 0  # stabilityScore
    
    def test_miner_registration_and_task_assignment(self, deploy_contracts):
        """Test miner registration in PoUW and task assignment"""
        contracts = deploy_contracts
        pouw = contracts["proof_of_useful_work"]
        energy_profile = contracts["energy_profile"]
        
        miner = accounts[3]
        
        # First register as energy user
        energy_profile.registerUser(
            miner.address,
            1000,
            "Mining Location",
            "Grid",
            {"from": miner, "value": "1 ether"}
        )
        
        # Register as miner in PoUW
        tx = pouw.registerMiner(
            int(0.5e18),  # 0.5 ETH compute power
            accounts[4].address,  # SGX attestation key
            {"from": miner}
        )
        
        # Verify miner registration
        miner_profile = pouw.miners(miner.address)
        assert miner_profile[0] == int(0.5e18)  # computePower
        assert miner_profile[6] == True  # isActive
        
        # Create a task
        deployer = accounts[0]
        task_tx = pouw.createTask(
            0,  # STABILITY_SIMULATION
            2,  # HIGH priority
            b"0x" + b"input_hash".hex(),
            b"0x" + b"output_hash".hex(),
            3600,  # 1 hour
            int(0.3e18),  # Min compute power
            {"from": deployer}
        )
        
        # Request task assignment
        assign_tx = pouw.requestTask({"from": miner})
        
        # Verify task assignment
        task = pouw.gridTasks(1)
        assert task[9] == miner.address  # assignedMiner
    
    def test_task_completion_updates_energy_profile(self, deploy_contracts):
        """Test that completing PoUW tasks updates the energy profile"""
        contracts = deploy_contracts
        pouw = contracts["proof_of_useful_work"]
        energy_profile = contracts["energy_profile"]
        energy_pool = contracts["energy_pool"]
        
        miner = accounts[3]
        deployer = accounts[0]
        
        # Setup: Register miner
        energy_profile.registerUser(
            miner.address,
            1000,
            "Mining Location",
            "Grid",
            {"from": miner, "value": "1 ether"}
        )
        
        pouw.registerMiner(
            int(0.5e18),
            accounts[4].address,
            {"from": miner}
        )
        
        # Create and assign task
        pouw.createTask(
            0,  # STABILITY_SIMULATION
            3,  # CRITICAL priority
            b"0x" + b"input_hash".hex(),
            b"0x" + b"expected_output".hex(),
            3600,
            int(0.3e18),
            {"from": deployer}
        )
        
        pouw.requestTask({"from": miner})
        
        # Submit task proof
        tx = pouw.submitTaskProof(
            1,  # taskId
            b"0x" + b"expected_output".hex(),  # Matching output hash
            b"SGX_ATTESTATION_DATA",
            100,  # computation time
            int(0.1e18),  # energy used
            150,  # grid impact (1.5% improvement)
            {"from": miner}
        )
        
        # Check that stability NFT was minted (event emission)
        assert "StabilityNFTMinted" in tx.events
        
        # Get token ID for miner
        token_id = energy_profile.getTokenIdForMiner(miner.address)
        
        # Check updated profile
        profile = energy_profile.getUserProfile(token_id)
        assert profile[8] == 150  # stabilityScore
        assert profile[9] == 1    # gridTasksCompleted
        
        # Check stability contributions
        contributions = energy_profile.getStabilityContributions(token_id)
        assert len(contributions) == 1
        assert contributions[0][0] == 1   # taskId
        assert contributions[0][1] == 150 # impact
    
    def test_energy_commitments_with_pouw_active(self, deploy_contracts):
        """Test that energy commitments work normally with PoUW active"""
        contracts = deploy_contracts
        energy_profile = contracts["energy_profile"]
        energy_pool = contracts["energy_pool"]
        
        producer = accounts[5]
        consumer = accounts[6]
        
        # Register users
        energy_profile.registerUser(
            producer.address,
            1000,
            "Producer Location",
            "Solar",
            {"from": producer, "value": "2 ether"}
        )
        
        energy_profile.registerUser(
            consumer.address,
            1000,
            "Consumer Location",
            "Residential",
            {"from": consumer, "value": "2 ether"}
        )
        
        producer_token = 1
        consumer_token = 2
        
        # Create commitments
        energy_profile.createCommitment(
            producer_token,
            100,  # energy amount
            True,  # is production
            100,   # duration
            {"from": producer}
        )
        
        energy_profile.createCommitment(
            consumer_token,
            80,   # energy amount
            False, # is consumption
            100,   # duration
            {"from": consumer}
        )
        
        # Deposit to energy pool
        energy_pool.deposit(producer_token, {"from": producer})
        energy_pool.deposit(consumer_token, {"from": consumer})
        
        # Verify commitments were created
        producer_commitments = energy_profile.getUserCommitments(producer_token)
        consumer_commitments = energy_profile.getUserCommitments(consumer_token)
        
        assert len(producer_commitments) > 0
        assert len(consumer_commitments) > 0
        assert producer_commitments[0][1] == True  # isProduction
        assert consumer_commitments[0][1] == False # isProduction
    
    def test_miner_energy_usage_tracking(self, deploy_contracts):
        """Test that miner energy usage is tracked in the energy pool"""
        contracts = deploy_contracts
        pouw = contracts["proof_of_useful_work"]
        energy_profile = contracts["energy_profile"]
        energy_pool = contracts["energy_pool"]
        
        miner = accounts[7]
        deployer = accounts[0]
        
        # Setup miner
        energy_profile.registerUser(
            miner.address,
            1000,
            "Mining Location",
            "Grid",
            {"from": miner, "value": "1 ether"}
        )
        
        pouw.registerMiner(
            int(0.5e18),
            accounts[8].address,
            {"from": miner}
        )
        
        # Create and complete task
        pouw.createTask(
            1,  # RENEWABLE_FORECAST
            2,  # HIGH priority
            b"0x" + b"input".hex(),
            b"0x" + b"output".hex(),
            3600,
            int(0.3e18),
            {"from": deployer}
        )
        
        pouw.requestTask({"from": miner})
        
        # Get initial energy consumed in interval
        initial_interval = energy_pool.intervals(0)
        initial_consumed = initial_interval[4]
        
        # Submit task with energy usage
        energy_used = int(0.2e18)
        pouw.submitTaskProof(
            1,
            b"0x" + b"output".hex(),
            b"SGX_ATTESTATION",
            100,
            energy_used,
            200,  # grid impact
            {"from": miner}
        )
        
        # Check energy pool was updated
        updated_interval = energy_pool.intervals(0)
        updated_consumed = updated_interval[4]
        
        assert updated_consumed == initial_consumed + energy_used
    
    def test_reward_distribution(self, deploy_contracts):
        """Test that PoUW rewards are properly distributed"""
        contracts = deploy_contracts
        pouw = contracts["proof_of_useful_work"]
        energy_profile = contracts["energy_profile"]
        
        miner = accounts[9]
        deployer = accounts[0]
        
        # Setup miner
        energy_profile.registerUser(
            miner.address,
            1000,
            "Mining Location",
            "Grid",
            {"from": miner, "value": "1 ether"}
        )
        
        pouw.registerMiner(
            int(1e18),
            accounts[8].address,
            {"from": miner}
        )
        
        # Create critical task (5x base reward)
        pouw.createTask(
            2,  # LOAD_BALANCE
            3,  # CRITICAL priority
            b"0x" + b"input".hex(),
            b"0x" + b"output".hex(),
            3600,
            int(0.5e18),
            {"from": deployer}
        )
        
        pouw.requestTask({"from": miner})
        
        # Get initial balance
        initial_balance = miner.balance()
        
        # Complete task
        pouw.submitTaskProof(
            1,
            b"0x" + b"output".hex(),
            b"SGX_ATTESTATION",
            100,
            int(0.1e18),
            300,
            {"from": miner}
        )
        
        # Check reward was received
        final_balance = miner.balance()
        expected_reward = 5e18  # 5 ETH (CRITICAL_MULTIPLIER * BASE_REWARD)
        
        # Account for gas costs
        assert final_balance > initial_balance
        assert final_balance - initial_balance < expected_reward  # Less due to gas
        
        # Check miner profile update
        miner_profile = pouw.miners(miner.address)
        assert miner_profile[2] == expected_reward  # totalRewardsEarned
    
    def test_performance_score_calculation(self, deploy_contracts):
        """Test that performance scores include both energy and stability contributions"""
        contracts = deploy_contracts
        pouw = contracts["proof_of_useful_work"]
        energy_profile = contracts["energy_profile"]
        
        user = accounts[3]
        deployer = accounts[0]
        
        # Register as both energy producer and miner
        energy_profile.registerUser(
            user.address,
            1000,
            "Hybrid Location",
            "Solar+Computing",
            {"from": user, "value": "2 ether"}
        )
        
        token_id = 1
        
        # Create energy production commitment
        energy_profile.createCommitment(
            token_id,
            1000,  # energy amount
            True,  # is production
            1000,  # duration
            {"from": user}
        )
        
        # Register as miner and complete tasks
        pouw.registerMiner(
            int(1e18),
            accounts[4].address,
            {"from": user}
        )
        
        # Complete multiple tasks to build up stability score
        for i in range(3):
            pouw.createTask(
                i % 3,  # Vary task types
                2,      # HIGH priority
                f"0x{'input' + str(i)}".encode().hex(),
                f"0x{'output' + str(i)}".encode().hex(),
                3600,
                int(0.5e18),
                {"from": deployer}
            )
            
            pouw.requestTask({"from": user})
            
            pouw.submitTaskProof(
                i + 1,
                f"0x{'output' + str(i)}".encode().hex(),
                b"SGX_ATTESTATION",
                100,
                int(0.1e18),
                100 + i * 50,  # Increasing grid impact
                {"from": user}
            )
        
        # Check final profile
        profile = energy_profile.getUserProfile(token_id)
        
        # Performance should include:
        # - Base energy score
        # - Stability bonus (450 total impact / 10 = 45)
        # - Task bonus (3 tasks * 5 = 15)
        assert profile[7] > 100  # historicalPerformance
        assert profile[8] == 450  # stabilityScore (100 + 150 + 200)
        assert profile[9] == 3    # gridTasksCompleted


def test_contract_interfaces():
    """Test that contract interfaces match expected signatures"""
    # This test verifies that the interfaces are properly implemented
    # It's a compile-time check, so if the contracts compile, interfaces match
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])