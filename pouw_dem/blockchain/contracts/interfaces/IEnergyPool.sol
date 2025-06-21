// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IEnergyPool {
    function updateIntervalEnergyByMarket(bool isProduction, uint256 energyAmount) external;
    function intervals(uint256 index) external view returns (
        uint256 energyCommittedProduction,
        uint256 energyCommittedConsumption,
        uint256 current_interval_block_number,
        uint256 energyProduced,
        uint256 energyConsumed
    );
}