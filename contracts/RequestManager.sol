// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./IUncertaintyAnalytics.sol";

/**
 * @title RequestManager
 * @notice Manages request submission to the UncertaintyAnalytics contract
 * @dev Uses string instead of bytes32 for error messages as requested
 */
contract RequestManager {
    // State variables
    IUncertaintyAnalytics private immutable analytics;
    string public constant VERSION = "1.0.0";
    address public owner;
    
    // Events
    event RequestSubmitted(uint256 indexed requestId, address indexed requester, uint256 value);
    event FundsWithdrawn(address indexed recipient, uint256 amount);
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "RequestManager: caller is not the owner");
        _;
    }
    
    /**
     * @notice Contract constructor
     * @param _analytics Address of the UncertaintyAnalytics contract
     */
    constructor(address _analytics) {
        require(_analytics != address(0), "RequestManager: analytics address cannot be zero");
        // Check if the address contains code (is a contract)
        // This could fail if deployed in the same transaction, consider removing if needed
        uint256 codeSize;
        assembly {
            codeSize := extcodesize(_analytics)
        }
        require(codeSize > 0, "RequestManager: analytics must be a contract");
        
        analytics = IUncertaintyAnalytics(_analytics);
        owner = msg.sender;
    }
    
    /**
     * @notice Submit a request to the analytics contract
     * @dev Forwards any value sent with the transaction
     * @return requestId The ID of the newly created request
     */
    function submitRequest() external payable returns (uint256) {
        require(msg.value > 0, "RequestManager: value must be greater than zero");
        
        uint256 requestId = analytics.submitRequest{value: msg.value}();
        
        emit RequestSubmitted(requestId, msg.sender, msg.value);
        return requestId;
    }
    
    /**
     * @notice Submit a request with a specified value
     * @dev For users who want to contribute more than the base cost
     * @param _additionalInfo Additional information about the request (unused currently)
     * @return requestId The ID of the newly created request
     */
    function submitRequestWithInfo(string calldata _additionalInfo) external payable returns (uint256) {
        require(msg.value > 0, "RequestManager: value must be greater than zero");
        require(bytes(_additionalInfo).length > 0, "RequestManager: additional info cannot be empty");
        
        uint256 requestId = analytics.submitRequest{value: msg.value}();
        
        emit RequestSubmitted(requestId, msg.sender, msg.value);
        return requestId;
    }
    
   
    function getAnalyticsMetrics() external view returns (
        uint256 avgProcessingTime,
        uint256 successRate,
        uint256 totalCost,
        uint256 disruptionCount
    ) {
        return analytics.getMetrics();
    }
    
   
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "RequestManager: no funds to withdraw");
        
        (bool success, ) = payable(owner).call{value: balance}("");
        require(success, "RequestManager: withdrawal failed");
        
        emit FundsWithdrawn(owner, balance);
    }
    
    /**
     * @notice Change the owner of the contract
     * @param _newOwner The address of the new owner
     */
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "RequestManager: new owner cannot be zero address");
        owner = _newOwner;
    }
    
    /**
     * @notice Allow the contract to receive ETH
     */
    receive() external payable {}
}