// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IEnergyProfile {
    struct UserData {
        uint256 collateral;
        uint256 energyProduced;
        uint256 energyConsumed;
        uint256 maxCommitment;
        uint256 energyPrice;
        string location;
        string energyType;
        uint256 historicalPerformance;
    }
    
    struct StabilityContribution {
        uint256 taskId;
        uint256 impact;
        uint256 timestamp;
        bytes32 proofHash;
    }
    
    function mintStabilityNFT(
        address miner,
        uint256 taskId,
        uint256 impact,
        bytes32 proofHash
    ) external;
    
    function getUserProfile(uint256 tokenId) external view returns (UserData memory);
    function tokenOfOwnerByIndex(address owner, uint256 index) external view returns (uint256);
}