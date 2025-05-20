// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./IUncertaintyAnalytics.sol";

/**
 * @title ResponseManager
 * @notice Manages response submission to the UncertaintyAnalytics contract
 * @dev Uses string instead of bytes32 for error messages as requested
 */
contract ResponseManager {
    // State variables
    IUncertaintyAnalytics private immutable analytics;
    string public constant VERSION = "1.0.0";
    address public owner;
    
    // Mapping to track responder activity
    mapping(address => uint256) public responderCount;
    mapping(uint256 => bool) public processedRequests;
    
    // Events
    event ResponseSubmitted(uint256 indexed requestId, address indexed responder);
    event FundsWithdrawn(address indexed recipient, uint256 amount);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "ResponseManager: caller is not the owner");
        _;
    }
    
    /**
     * @notice Contract constructor
     * @param _analytics Address of the UncertaintyAnalytics contract
     */
    constructor(address _analytics) {
        require(_analytics != address(0), "ResponseManager: analytics address cannot be zero");
        // Check if the address contains code (is a contract)
        // This could fail if deployed in the same transaction, consider removing if needed
        uint256 codeSize;
        assembly {
            codeSize := extcodesize(_analytics)
        }
        require(codeSize > 0, "ResponseManager: analytics must be a contract");
        
        analytics = IUncertaintyAnalytics(_analytics);
        owner = msg.sender;
    }
    
    /**
     * @notice Submit a response to a request
     * @param _requestId The ID of the request to respond to
     */
    function submitResponse(uint256 _requestId) external {
        require(_requestId > 0, "ResponseManager: request ID must be greater than zero");
        require(!processedRequests[_requestId], "ResponseManager: request already processed");
        
        // Mark request as processed to prevent duplicate responses
        processedRequests[_requestId] = true;
        
        // Update responder stats
        responderCount[msg.sender] += 1;
        
        // Submit response to analytics contract
        analytics.submitResponse(_requestId);
        
        emit ResponseSubmitted(_requestId, msg.sender);
    }
    
    /**
     * @notice Submit a response with additional calculation
     * @param _requestId The ID of the request to respond to
     */
    function submitResponseWithCalculation(uint256 _requestId) external {
        require(_requestId > 0, "ResponseManager: request ID must be greater than zero");
        require(!processedRequests[_requestId], "ResponseManager: request already processed");
        
        // Mark request as processed to prevent duplicate responses
        processedRequests[_requestId] = true;
        
        // Update responder stats
        responderCount[msg.sender] += 1;
        
        // Submit response and calculate unavailability cost
        analytics.submitResponse(_requestId);
        analytics.calculateUnavailabilityCost(_requestId);
        
        emit ResponseSubmitted(_requestId, msg.sender);
    }
    
    /**
     * @notice Get the number of responses submitted by an address
     * @param _responder The address to check
     * @return count The number of responses
     */
    function getResponderCount(address _responder) external view returns (uint256 count) {
        return responderCount[_responder];
    }
    
   
    function getAnalyticsMetrics() external view returns (
        uint256 avgProcessingTime,
        uint256 successRate,
        uint256 totalCost,
        uint256 disruptionCount
    ) {
        return analytics.getMetrics();
    }
    
    /**
     * @notice Withdraw any ETH that might be stuck in this contract
     * @dev Only callable by the owner
     */
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "ResponseManager: no funds to withdraw");
        
        (bool success, ) = payable(owner).call{value: balance}("");
        require(success, "ResponseManager: withdrawal failed");
        
        emit FundsWithdrawn(owner, balance);
    }
    
    /**
     * @notice Change the owner of the contract
     * @param _newOwner The address of the new owner
     */
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "ResponseManager: new owner cannot be zero address");
        owner = _newOwner;
    }
    
    /**
     * @notice Allow the contract to receive ETH
     */
    receive() external payable {}
}