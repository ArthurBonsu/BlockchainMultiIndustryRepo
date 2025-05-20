// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title UncertaintyBase
 * @notice Base contract with shared components for Uncertainty contracts
 */
contract UncertaintyBase {
    address public owner;
    bool private locked;
    
    enum Status { Pending, Processing, Completed, Failed }
    
    uint256 public constant MAX_PROCESSING_TIME = 1 days;
    uint256 public constant BASE_COST = 0.001 ether;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier noReentrant() {
        require(!locked, "No reentrancy");
        locked = true;
        _;
        locked = false;
    }
    
    constructor() {
        owner = msg.sender;
        locked = false;
    }
    
    // Simple receive function
    receive() external payable {}
    
    // Safe withdrawal function
    function withdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        (bool success, ) = payable(owner).call{value: balance}("");
        require(success, "Withdrawal failed");
    }
}

/**
 * @title RequestManager
 * @notice Handles request/response management functionality
 */
contract RequestManager {
    UncertaintyBase private base;
    
    struct Request {
        uint256 id;
        address requester;
        uint256 timestamp;
        uint256 confirmationTime;
        uint256 executionTime;
        uint256 cost;
        bool isValid;
        bool isProcessed;
        UncertaintyBase.Status status;
    }
    
    struct Response {
        uint256 requestId;
        address responder;
        uint256 timestamp;
        uint256 processingTime;
        uint256 cost;
        bool isValid;
        UncertaintyBase.Status status;
    }
    
    uint256 public requestCount;
    uint256 public responseCount;
    uint256 public totalTransactionCost;
    uint256 public failedTransactionCount;
    
    mapping(uint256 => Request) public requests;
    mapping(uint256 => Response) public responses;
    mapping(address => uint256) public requesterStats;
    mapping(address => uint256) public responderStats;
    
    event RequestSubmitted(uint256 indexed requestId, address requester);
    event ResponseReceived(uint256 indexed requestId, address responder);
    event TransactionFailed(uint256 indexed requestId, string reason);
    
    constructor(UncertaintyBase _base) {
        base = _base;
        requestCount = 0;
        responseCount = 0;
        totalTransactionCost = 0;
        failedTransactionCount = 0;
    }
    
    // Function to check if sender is owner (using base contract)
    modifier onlyOwner() {
        require(msg.sender == base.owner(), "Not owner");
        _;
    }
    
    function submitRequest() external payable returns (uint256) {
        require(msg.value >= base.BASE_COST(), "Insufficient payment");
        
        requestCount += 1;
        
        requests[requestCount] = Request({
            id: requestCount,
            requester: msg.sender,
            timestamp: block.timestamp,
            confirmationTime: 0,
            executionTime: 0,
            cost: msg.value,
            isValid: true,
            isProcessed: false,
            status: UncertaintyBase.Status.Pending
        });
        
        totalTransactionCost += msg.value;
        requesterStats[msg.sender] += 1;
        
        emit RequestSubmitted(requestCount, msg.sender);
        return requestCount;
    }
    
    function submitResponse(uint256 _requestId) external {
        require(_requestId > 0 && _requestId <= requestCount, "Invalid request ID");
        require(requests[_requestId].isValid, "Request not valid");
        require(requests[_requestId].status == UncertaintyBase.Status.Pending, "Request not pending");
        
        responseCount += 1;
        
        requests[_requestId].status = UncertaintyBase.Status.Completed;
        requests[_requestId].isProcessed = true;
        
        responses[_requestId] = Response({
            requestId: _requestId,
            responder: msg.sender,
            timestamp: block.timestamp,
            processingTime: block.timestamp - requests[_requestId].timestamp,
            cost: 0,
            isValid: true,
            status: UncertaintyBase.Status.Completed
        });
        
        responderStats[msg.sender] += 1;
        
        emit ResponseReceived(_requestId, msg.sender);
    }
    
    function recordFailedTransaction(uint256 _requestId, string calldata _reason) external onlyOwner {
        require(_requestId > 0 && _requestId <= requestCount, "Invalid request ID");
        Request storage request = requests[_requestId];
        require(request.isValid, "Request not valid");
        require(request.status != UncertaintyBase.Status.Failed, "Already marked as failed");
        
        request.status = UncertaintyBase.Status.Failed;
        failedTransactionCount += 1;
        
        emit TransactionFailed(_requestId, _reason);
    }
    
    // Helper function to get request validity
    function isRequestValid(uint256 _requestId) external view returns (bool) {
        if (_requestId == 0 || _requestId > requestCount) {
            return false;
        }
        return requests[_requestId].isValid;
    }
    
    // Helper function to get response processing time
    function getResponseProcessingTime(uint256 _requestId) external view returns (uint256, bool) {
        if (_requestId == 0 || _requestId > requestCount) {
            return (0, false);
        }
        
        if (responses[_requestId].requestId == 0) {
            return (0, false);
        }
        
        return (responses[_requestId].processingTime, true);
    }
    
    // Function to get total processing time for all valid responses
    function getTotalProcessingStats() external view returns (uint256 totalTime, uint256 validResponses) {
        totalTime = 0;
        validResponses = 0;
        
        for (uint256 i = 1; i <= requestCount; i++) {
            if (responses[i].requestId != 0 && responses[i].isValid) {
                totalTime += responses[i].processingTime;
                validResponses++;
            }
        }
        
        return (totalTime, validResponses);
    }
}

/**
 * @title CostAnalytics
 * @notice Handles cost tracking and analytics
 */
contract CostAnalytics {
    UncertaintyBase private base;
    
    uint256 public dataHoldingCost;
    uint256 public unavailabilityCost;
    uint256 public disruptionLevel;
    uint256 public escalationLevel;
    
    event CostRecorded(uint256 indexed requestId, uint256 cost, string costType);
    
    constructor(UncertaintyBase _base) {
        base = _base;
        dataHoldingCost = 0;
        unavailabilityCost = 0;
        disruptionLevel = 0;
        escalationLevel = 0;
    }
    
    // Function to check if sender is owner (using base contract)
    modifier onlyOwner() {
        require(msg.sender == base.owner(), "Not owner");
        _;
    }
    
    function calculateUnavailabilityCost(uint256 _processingTime) external onlyOwner {
        if (_processingTime > base.MAX_PROCESSING_TIME()) {
            uint256 penalty = (_processingTime - base.MAX_PROCESSING_TIME()) * base.BASE_COST() / 86400;
            unavailabilityCost += penalty;
            emit CostRecorded(0, penalty, "Unavailability");
        }
    }
    
    function updateDataHoldingCost(uint256 _cost) external onlyOwner {
        dataHoldingCost += _cost;
        emit CostRecorded(0, _cost, "DataHolding");
    }
    
    function updateDisruptionLevel(uint256 _level) external onlyOwner {
        disruptionLevel = _level;
    }
    
    function updateEscalationLevel(uint256 _level) external onlyOwner {
        escalationLevel = _level;
    }
}

/**
 * @title UncertaintyAnalytics
 * @notice Main contract that combines all functionality
 */
contract UncertaintyAnalytics {
    UncertaintyBase public base;
    RequestManager public requestManager;
    CostAnalytics public costAnalytics;
    
    constructor() {
        // Create the base contract first
        base = new UncertaintyBase();
        
        // Then create the components with the base contract
        requestManager = new RequestManager(base);
        costAnalytics = new CostAnalytics(base);
    }
    
    // Forward request functions to RequestManager
    function submitRequest() external payable returns (uint256) {
        return requestManager.submitRequest{value: msg.value}();
    }
    
    function submitResponse(uint256 _requestId) external {
        requestManager.submitResponse(_requestId);
    }
    
    function recordFailedTransaction(uint256 _requestId, string calldata _reason) external {
        requestManager.recordFailedTransaction(_requestId, _reason);
    }
    
    // Forward analytics functions to CostAnalytics
    function calculateUnavailabilityCost(uint256 _processingTime) external {
        costAnalytics.calculateUnavailabilityCost(_processingTime);
    }
    
    function updateDataHoldingCost(uint256 _cost) external {
        costAnalytics.updateDataHoldingCost(_cost);
    }
    
    function updateDisruptionLevel(uint256 _level) external {
        costAnalytics.updateDisruptionLevel(_level);
    }
    
    function updateEscalationLevel(uint256 _level) external {
        costAnalytics.updateEscalationLevel(_level);
    }
    
    // Combined metrics function
    function getMetrics() external view returns (
        uint256 avgProcessingTime,
        uint256 successRate,
        uint256 totalCost,
        uint256 disruptionCount
    ) {
        // Use the RequestManager's helper function to get processing stats
        (uint256 totalTime, uint256 validResponses) = requestManager.getTotalProcessingStats();
        
        // Calculate average processing time if there are valid responses
        if (validResponses > 0) {
            avgProcessingTime = totalTime / validResponses;
        }
        
        // Calculate success rate if there are any requests
        if (requestManager.requestCount() > 0) {
            successRate = ((requestManager.requestCount() - requestManager.failedTransactionCount()) * 100) / requestManager.requestCount();
        }
        
        // Calculate total cost across all components
        totalCost = requestManager.totalTransactionCost() + costAnalytics.dataHoldingCost() + costAnalytics.unavailabilityCost();
        
        // Get disruption count
        disruptionCount = requestManager.failedTransactionCount();
    }
    
    // Forward withdraw function to base
    function withdraw() external {
        base.withdraw();
    }
    
    // Allow the contract to receive ETH
    receive() external payable {}
}