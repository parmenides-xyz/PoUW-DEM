// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title Grid Stability NFT
 * @dev Real NFT contract for tracking miner contributions to grid stability
 * Deployable on Sepolia/Polygon testnet in minutes
 */
contract GridStabilityNFT is ERC721, ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIdCounter;
    
    // Miner contribution tracking
    struct Contribution {
        uint256 timestamp;
        uint256 gridStressLevel; // 0-100%
        uint256 mwReduced;       // MW taken offline for grid
        uint256 duration;        // Minutes of support
        string taskType;         // "EMERGENCY", "REGULATION", "FORECAST"
        uint256 carbonSaved;     // Estimated tons CO2 avoided
    }
    
    // NFT metadata
    struct StabilityToken {
        address miner;
        uint256 totalContributions;
        uint256 totalMWh;
        uint256 totalCarbonSaved;
        uint256 reputationScore;
    }
    
    mapping(uint256 => StabilityToken) public tokens;
    mapping(uint256 => Contribution[]) public tokenContributions;
    mapping(address => uint256[]) public minerTokens;
    
    // Leaderboard
    address[] public topContributors;
    mapping(address => uint256) public minerScores;
    
    event StabilityContribution(
        uint256 indexed tokenId,
        address indexed miner,
        uint256 mwReduced,
        string taskType
    );
    
    constructor() ERC721("Grid Stability NFT", "GRID") {}
    
    /**
     * @dev Mint NFT for grid stability contribution
     * Called when miner reduces load for grid support
     */
    function mintStabilityNFT(
        address miner,
        uint256 gridStressLevel,
        uint256 mwReduced,
        uint256 duration,
        string memory taskType
    ) public onlyOwner returns (uint256) {
        require(miner != address(0), "Invalid miner address");
        require(mwReduced > 0, "Must reduce load");
        
        _tokenIdCounter.increment();
        uint256 tokenId = _tokenIdCounter.current();
        
        _safeMint(miner, tokenId);
        
        // Calculate impact metrics
        uint256 mwhContributed = (mwReduced * duration) / 60;
        uint256 carbonSaved = (mwhContributed * 500) / 1000; // 0.5 tons CO2/MWh
        
        // Initialize token data
        tokens[tokenId] = StabilityToken({
            miner: miner,
            totalContributions: 1,
            totalMWh: mwhContributed,
            totalCarbonSaved: carbonSaved,
            reputationScore: calculateReputation(gridStressLevel, mwReduced)
        });
        
        // Record contribution
        tokenContributions[tokenId].push(Contribution({
            timestamp: block.timestamp,
            gridStressLevel: gridStressLevel,
            mwReduced: mwReduced,
            duration: duration,
            taskType: taskType,
            carbonSaved: carbonSaved
        }));
        
        // Update miner records
        minerTokens[miner].push(tokenId);
        minerScores[miner] += tokens[tokenId].reputationScore;
        
        // Set token URI (would point to IPFS in production)
        string memory uri = generateTokenURI(tokenId);
        _setTokenURI(tokenId, uri);
        
        emit StabilityContribution(tokenId, miner, mwReduced, taskType);
        
        return tokenId;
    }
    
    /**
     * @dev Add contribution to existing NFT
     */
    function addContribution(
        uint256 tokenId,
        uint256 gridStressLevel,
        uint256 mwReduced,
        uint256 duration,
        string memory taskType
    ) public onlyOwner {
        require(_exists(tokenId), "Token does not exist");
        
        uint256 mwhContributed = (mwReduced * duration) / 60;
        uint256 carbonSaved = (mwhContributed * 500) / 1000;
        
        // Update token stats
        tokens[tokenId].totalContributions++;
        tokens[tokenId].totalMWh += mwhContributed;
        tokens[tokenId].totalCarbonSaved += carbonSaved;
        tokens[tokenId].reputationScore += calculateReputation(gridStressLevel, mwReduced);
        
        // Record contribution
        tokenContributions[tokenId].push(Contribution({
            timestamp: block.timestamp,
            gridStressLevel: gridStressLevel,
            mwReduced: mwReduced,
            duration: duration,
            taskType: taskType,
            carbonSaved: carbonSaved
        }));
        
        // Update miner score
        address miner = tokens[tokenId].miner;
        minerScores[miner] += calculateReputation(gridStressLevel, mwReduced);
        
        emit StabilityContribution(tokenId, miner, mwReduced, taskType);
    }
    
    /**
     * @dev Calculate reputation points based on contribution impact
     */
    function calculateReputation(uint256 gridStress, uint256 mwReduced) 
        private 
        pure 
        returns (uint256) 
    {
        // Higher rewards for helping during critical times
        uint256 stressMultiplier = gridStress > 80 ? 5 : 
                                   gridStress > 60 ? 3 : 1;
        
        return mwReduced * stressMultiplier;
    }
    
    /**
     * @dev Generate token metadata (simplified - use IPFS in production)
     */
    function generateTokenURI(uint256 tokenId) 
        private 
        view 
        returns (string memory) 
    {
        StabilityToken memory token = tokens[tokenId];
        
        // In production, this would generate and upload to IPFS
        return string(abi.encodePacked(
            "data:application/json;base64,",
            "eyJuYW1lIjoiR3JpZCBTdGFiaWxpdHkgTkZUIiwi",
            "ZGVzY3JpcHRpb24iOiJQcm9vZiBvZiBncmlkIHN0YWJpbGl0eSBjb250cmlidXRpb24iLCJ",
            "hdHRyaWJ1dGVzIjpbeyJ0cmFpdF90eXBlIjoiVG90YWwgTVdoIiwidmFsdWUiOiI",
            uint2str(token.totalMWh),
            "In0seyJ0cmFpdF90eXBlIjoiQ2FyYm9uIFNhdmVkIiwidmFsdWUiOiI",
            uint2str(token.totalCarbonSaved),
            "In1dfQ=="
        ));
    }
    
    /**
     * @dev Get miner statistics
     */
    function getMinerStats(address miner) 
        public 
        view 
        returns (
            uint256 totalTokens,
            uint256 totalScore,
            uint256 totalMWh,
            uint256 totalCarbon
        ) 
    {
        uint256[] memory tokenIds = minerTokens[miner];
        totalTokens = tokenIds.length;
        totalScore = minerScores[miner];
        
        for (uint i = 0; i < tokenIds.length; i++) {
            totalMWh += tokens[tokenIds[i]].totalMWh;
            totalCarbon += tokens[tokenIds[i]].totalCarbonSaved;
        }
    }
    
    /**
     * @dev Get top contributors (for leaderboard)
     */
    function getTopContributors(uint256 limit) 
        public 
        view 
        returns (address[] memory, uint256[] memory) 
    {
        // In production, maintain sorted list
        // For demo, return all miners with scores
        uint256 count = limit < topContributors.length ? limit : topContributors.length;
        address[] memory miners = new address[](count);
        uint256[] memory scores = new uint256[](count);
        
        for (uint i = 0; i < count; i++) {
            miners[i] = topContributors[i];
            scores[i] = minerScores[miners[i]];
        }
        
        return (miners, scores);
    }
    
    // Utility functions
    function uint2str(uint256 _i) internal pure returns (string memory) {
        if (_i == 0) return "0";
        uint256 j = _i;
        uint256 len;
        while (j != 0) {
            len++;
            j /= 10;
        }
        bytes memory bstr = new bytes(len);
        uint256 k = len;
        while (_i != 0) {
            k = k-1;
            uint8 temp = (48 + uint8(_i - _i / 10 * 10));
            bytes1 b1 = bytes1(temp);
            bstr[k] = b1;
            _i /= 10;
        }
        return string(bstr);
    }
    
    // Required overrides
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }
    
    function tokenURI(uint256 tokenId) 
        public 
        view 
        override(ERC721, ERC721URIStorage) 
        returns (string memory) 
    {
        return super.tokenURI(tokenId);
    }
}