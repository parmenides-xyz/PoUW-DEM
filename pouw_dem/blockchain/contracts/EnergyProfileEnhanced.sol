// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV2V3Interface.sol";

/**
 * @title EnergyProfileEnhanced
 * @dev Extended EnergyProfile contract with Stability NFT support for PoUW integration
 */
contract EnergyProfileEnhanced is ERC721, ERC721Enumerable, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIdCounter;
    
    // Oracle interfaces
    AggregatorV2V3Interface private _energyProducedOracle;
    AggregatorV2V3Interface private _energyConsumedOracle;
    AggregatorV2V3Interface private _energyPriceOracle;
    
    // Contract addresses
    address private _energyPoolContract;
    address private _energyMarketContract;
    address private _proofOfUsefulWorkContract;
    
    // User profile structure
    struct UserData {
        uint256 collateral;
        uint256 energyProduced;
        uint256 energyConsumed;
        uint256 maxCommitment;
        uint256 energyPrice;
        string location;
        string energyType;
        uint256 historicalPerformance;
        uint256 stabilityScore; // New: Accumulated stability contributions
        uint256 gridTasksCompleted; // New: Number of grid tasks completed
    }
    
    struct Commitment {
        uint256 energyAmount;
        bool isProduction;
        uint256 startBlock;
        uint256 duration;
        bool settled;
        bool energy_pool_processed;
        bool energy_market_processed;
        uint256 energy_price;
    }
    
    // New: Stability contribution structure
    struct StabilityContribution {
        uint256 taskId;
        uint256 impact; // Grid stability improvement (basis points, e.g., 150 = 1.5%)
        uint256 timestamp;
        bytes32 proofHash;
        string taskType; // "stability_sim", "renewable_forecast", etc.
        uint256 energyUsed; // Energy consumed for the task
        uint256 reward; // Reward earned from the task
    }
    
    // Mappings
    mapping(uint256 => UserData) private _userProfiles;
    mapping(uint256 => Commitment[]) public _userCommitments;
    mapping(uint256 => StabilityContribution[]) public _stabilityContributions;
    mapping(address => uint256) private _minerToTokenId; // New: Quick lookup for miners
    
    // Events
    event UserRegistered(address indexed user, uint256 tokenId);
    event CommitmentCreated(uint256 indexed tokenId, Commitment commitment);
    event EnergyDataUpdated(uint256 indexed tokenId, uint256 energyProduced, uint256 energyConsumed);
    event EnergyPriceUpdated(uint256 indexed tokenId, uint256 newEnergyPrice);
    event StabilityNFTMinted(uint256 indexed tokenId, uint256 taskId, uint256 impact);
    event StabilityScoreUpdated(uint256 indexed tokenId, uint256 newScore);
    
    constructor(
        address energyProducedOracleAddress,
        address energyConsumedOracleAddress,
        address energyPriceOracleAddress
    ) ERC721("EnergyProfileEnhanced", "EPE") {
        _energyProducedOracle = AggregatorV2V3Interface(energyProducedOracleAddress);
        _energyConsumedOracle = AggregatorV2V3Interface(energyConsumedOracleAddress);
        _energyPriceOracle = AggregatorV2V3Interface(energyPriceOracleAddress);
    }
    
    /**
     * @dev Register a new user (modified to support miners)
     */
    function registerUser(
        address user,
        uint256 collateral,
        string memory location,
        string memory energyType
    ) public payable {
        _tokenIdCounter.increment();
        uint256 tokenId = _tokenIdCounter.current();
        _mint(user, tokenId);
        
        _userProfiles[tokenId] = UserData(
            msg.value, // Collateral from transaction value
            0, // energyProduced
            0, // energyConsumed
            1000, // maxCommitment
            0, // energyPrice
            location,
            energyType,
            100, // historicalPerformance
            0, // stabilityScore (new)
            0  // gridTasksCompleted (new)
        );
        
        // Store miner mapping for quick lookup
        _minerToTokenId[user] = tokenId;
        
        emit UserRegistered(user, tokenId);
    }
    
    /**
     * @dev Mint a Stability NFT for completed grid optimization task
     */
    function mintStabilityNFT(
        address miner,
        uint256 taskId,
        uint256 impact,
        bytes32 proofHash
    ) external {
        require(
            msg.sender == _proofOfUsefulWorkContract,
            "Only PoUW contract can mint stability NFTs"
        );
        
        uint256 tokenId = _minerToTokenId[miner];
        require(tokenId != 0, "Miner not registered");
        
        // Create stability contribution record
        StabilityContribution memory contribution = StabilityContribution({
            taskId: taskId,
            impact: impact,
            timestamp: block.timestamp,
            proofHash: proofHash,
            taskType: "grid_optimization", // Could be more specific based on task
            energyUsed: 0, // To be updated by PoUW contract
            reward: 0 // To be updated by PoUW contract
        });
        
        // Add to user's stability contributions
        _stabilityContributions[tokenId].push(contribution);
        
        // Update user profile
        UserData storage userData = _userProfiles[tokenId];
        userData.stabilityScore += impact;
        userData.gridTasksCompleted++;
        userData.historicalPerformance = calculatePerformanceScore(tokenId);
        
        emit StabilityNFTMinted(tokenId, taskId, impact);
        emit StabilityScoreUpdated(tokenId, userData.stabilityScore);
    }
    
    /**
     * @dev Get stability contributions for a token
     */
    function getStabilityContributions(uint256 tokenId) 
        external 
        view 
        returns (StabilityContribution[] memory) 
    {
        return _stabilityContributions[tokenId];
    }
    
    /**
     * @dev Calculate performance score based on energy and stability contributions
     */
    function calculatePerformanceScore(uint256 tokenId) private view returns (uint256) {
        UserData storage userData = _userProfiles[tokenId];
        uint256 energyScore = userData.energyProduced > 0 
            ? (userData.energyProduced * 100) / (userData.energyProduced + userData.energyConsumed)
            : 50;
        
        uint256 stabilityBonus = userData.stabilityScore / 10; // 1 point per 10 stability points
        uint256 taskBonus = userData.gridTasksCompleted * 5; // 5 points per completed task
        
        return energyScore + stabilityBonus + taskBonus;
    }
    
    /**
     * @dev Get token ID for a miner address
     */
    function getTokenIdForMiner(address miner) external view returns (uint256) {
        return _minerToTokenId[miner];
    }
    
    /**
     * @dev Set ProofOfUsefulWork contract address
     */
    function setProofOfUsefulWorkContract(address pouvContract) external onlyOwner {
        _proofOfUsefulWorkContract = pouvContract;
    }
    
    /**
     * @dev Override transfer to update miner mapping
     */
    function _transfer(
        address from,
        address to,
        uint256 tokenId
    ) internal virtual override {
        super._transfer(from, to, tokenId);
        
        // Update miner mapping
        if (_minerToTokenId[from] == tokenId) {
            delete _minerToTokenId[from];
            _minerToTokenId[to] = tokenId;
        }
    }
    
    // Existing functions remain the same...
    function getUserProfile(uint256 tokenId) public view returns (UserData memory) {
        return _userProfiles[tokenId];
    }
    
    function createCommitment(
        uint256 tokenId,
        uint256 energyAmount,
        bool isProduction,
        uint256 duration
    ) public returns (uint256) {
        require(
            ownerOf(tokenId) == msg.sender || msg.sender == _energyPoolContract,
            "Only the owner of the token or the EnergyPool contract can add a commitment"
        );
        
        UserData storage userData = _userProfiles[tokenId];
        (, int256 energyPrice, , ,) = _energyPriceOracle.latestRoundData();
        
        Commitment memory newCommitment = Commitment({
            energyAmount: energyAmount,
            isProduction: isProduction,
            startBlock: block.number,
            duration: duration,
            settled: false,
            energy_pool_processed: false,
            energy_market_processed: false,
            energy_price: uint256(energyPrice)
        });
        
        uint256 newCommitmentIndex;
        
        if (_userCommitments[tokenId].length > 2) {
            bool replaced = false;
            for (uint i = 0; i < _userCommitments[tokenId].length; i++) {
                if (_userCommitments[tokenId][i].settled || 
                    block.number > _userCommitments[tokenId][i].startBlock + _userCommitments[tokenId][i].duration) {
                    _userCommitments[tokenId][i] = newCommitment;
                    replaced = true;
                    newCommitmentIndex = i;
                    break;
                }
            }
            require(replaced, "No settled commitment found to replace");
        } else {
            _userCommitments[tokenId].push(newCommitment);
            newCommitmentIndex = _userCommitments[tokenId].length - 1;
        }
        
        emit CommitmentCreated(tokenId, newCommitment);
        return newCommitmentIndex;
    }
    
    function setEnergyPoolContract(address energyPoolContract) external onlyOwner {
        _energyPoolContract = energyPoolContract;
    }
    
    function setEnergyMarketContract(address energyMarketContractAddress) external onlyOwner {
        _energyMarketContract = energyMarketContractAddress;
    }
    
    function _beforeTokenTransfer(
        address from, 
        address to, 
        uint256 tokenId
    ) internal virtual override(ERC721, ERC721Enumerable) {
        super._beforeTokenTransfer(from, to, tokenId);
    }
    
    function supportsInterface(bytes4 interfaceId) 
        public 
        view 
        virtual 
        override(ERC721, ERC721Enumerable) 
        returns (bool) 
    {
        return super.supportsInterface(interfaceId);
    }
}