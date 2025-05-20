// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title IUncertaintyAnalytics
 * @notice Interface for the UncertaintyAnalytics contract
 */
interface IUncertaintyAnalytics {
    /**
     * @notice Submit a new request
     * @return requestId The ID of the newly created request
     */
    function submitRequest() external payable returns (uint256);
    
    /**
     * @notice Submit a response to a request
     * @param requestId The ID of the request to respond to
     */
    function submitResponse(uint256 requestId) external;
    
    /**
     * @notice Calculate the unavailability cost for a request
     * @param requestId The ID of the request
     */
    function calculateUnavailabilityCost(uint256 requestId) external;
    
    /**
     * @notice Get metrics about the system
     * @return avgProcessingTime Average processing time of all responses
     * @return successRate Percentage of successful requests
     * @return totalCost Total cost of all transactions
     * @return disruptionCount Number of failed transactions
     */
    function getMetrics() external view returns (
        uint256 avgProcessingTime,
        uint256 successRate,
        uint256 totalCost,
        uint256 disruptionCount
    );
}