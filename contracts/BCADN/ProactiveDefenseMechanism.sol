// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title ProactiveDefenseMechanism
 * @dev Implementation of BCADN's proactive defense system for anomaly detection
 */
contract ProactiveDefenseMechanism is Ownable, ReentrancyGuard {
    // Constants
    uint256 public constant MIN_ANOMALY_SCORE = 0;
    uint256 public constant MAX_ANOMALY_SCORE = 100;
    
    // Structures
    struct NodeMetrics {
        uint256 latency;         // ms
        uint256 throughput;      // txns/sec
        uint256 uptime;          // scaled 0-1000 (0.1% precision)
        uint256 errorRate;       // scaled 0-1000 (0.1% precision)
        uint256 cpuUsage;        // 0-100%
        uint256 memoryUsage;     // 0-100%
        uint256 connectionCount; // Number of connections
        uint256 lastUpdated;     // Timestamp
    }
    
    struct AttackDetection {
        address nodeAddress;
        uint256 timestamp;
        uint256 anomalyScore;
        string attackType;
        bool resolved;
        mapping(uint256 => uint256) featureContributions; // Feature index to contribution score
    }
    
    // State variables
    mapping(address => NodeMetrics) public nodeMetrics;
    mapping(address => bool) public registeredNodes;
    address[] public nodeList;
    
    // Attack history management
    uint256 public attackDetectionCount;
    mapping(uint256 => address) public attackToNode;
    mapping(uint256 => uint256) public attackToTimestamp;
    mapping(uint256 => uint256) public attackToAnomalyScore;
    mapping(uint256 => string) public attackToType;
    mapping(uint256 => bool) public attackResolved;
    mapping(uint256 => mapping(uint256 => uint256)) public attackFeatureContributions;
    
    // Node attack history
    mapping(address => uint256[]) public nodeAttackHistory;
    
    // Thresholds
    uint256 public anomalyThreshold;
    uint256 public baselineUpdateInterval;
    
    // Events
    event NodeRegistered(address indexed node);
    event MetricsUpdated(address indexed node, uint256 latency, uint256 throughput);
    event AnomalyDetected(address indexed node, uint256 indexed attackId, uint256 anomalyScore, string attackType);
    event AnomalyResolved(address indexed node, uint256 indexed attackId);
    event ThresholdUpdated(uint256 newThreshold);
    
    constructor(uint256 _anomalyThreshold, uint256 _baselineUpdateInterval) {
        anomalyThreshold = _anomalyThreshold;
        baselineUpdateInterval = _baselineUpdateInterval;
        attackDetectionCount = 0;
    }
    
    /**
     * @dev Registers a new node in the defense system
     */
    function registerNode(address _nodeAddress) external onlyOwner {
        require(!registeredNodes[_nodeAddress], "Node already registered");
        
        registeredNodes[_nodeAddress] = true;
        nodeList.push(_nodeAddress);
        
        // Initialize node metrics with defaults
        NodeMetrics storage metrics = nodeMetrics[_nodeAddress];
        metrics.latency = 100;         // 100ms default
        metrics.throughput = 100;      // 100 txns/sec default
        metrics.uptime = 990;          // 99.0% default
        metrics.errorRate = 10;        // 1.0% default
        metrics.cpuUsage = 50;         // 50% default
        metrics.memoryUsage = 50;      // 50% default
        metrics.connectionCount = 10;  // 10 connections default
        metrics.lastUpdated = block.timestamp;
        
        emit NodeRegistered(_nodeAddress);
    }
    
    /**
     * @dev Updates node metrics
     */
    function updateNodeMetrics(
        address _nodeAddress,
        uint256 _latency,
        uint256 _throughput,
        uint256 _uptime,
        uint256 _errorRate,
        uint256 _cpuUsage,
        uint256 _memoryUsage,
        uint256 _connectionCount
    ) external {
        require(registeredNodes[_nodeAddress], "Node not registered");
        
        NodeMetrics storage metrics = nodeMetrics[_nodeAddress];
        metrics.latency = _latency;
        metrics.throughput = _throughput;
        metrics.uptime = _uptime;
        metrics.errorRate = _errorRate;
        metrics.cpuUsage = _cpuUsage;
        metrics.memoryUsage = _memoryUsage;
        metrics.connectionCount = _connectionCount;
        metrics.lastUpdated = block.timestamp;
        
        emit MetricsUpdated(_nodeAddress, _latency, _throughput);
    }
    
    /**
     * @dev Records detection of an anomaly for a node
     */
    function recordAnomaly(
        address _nodeAddress,
        uint256 _anomalyScore,
        string calldata _attackType,
        uint256[] calldata _featureIndices,
        uint256[] calldata _featureContributions
    ) external onlyOwner {
        require(registeredNodes[_nodeAddress], "Node not registered");
        require(_anomalyScore >= MIN_ANOMALY_SCORE && _anomalyScore <= MAX_ANOMALY_SCORE, "Invalid anomaly score");
        require(_featureIndices.length == _featureContributions.length, "Feature arrays length mismatch");
        
        uint256 attackId = attackDetectionCount;
        attackToNode[attackId] = _nodeAddress;
        attackToTimestamp[attackId] = block.timestamp;
        attackToAnomalyScore[attackId] = _anomalyScore;
        attackToType[attackId] = _attackType;
        attackResolved[attackId] = false;
        
        // Record feature contributions
        for (uint256 i = 0; i < _featureIndices.length; i++) {
            attackFeatureContributions[attackId][_featureIndices[i]] = _featureContributions[i];
        }
        
        // Add to node's attack history
        nodeAttackHistory[_nodeAddress].push(attackId);
        
        // Increment counter
        attackDetectionCount++;
        
        emit AnomalyDetected(_nodeAddress, attackId, _anomalyScore, _attackType);
    }
    
    /**
     * @dev Resolves a previously detected anomaly
     */
    function resolveAnomaly(uint256 _attackId) external onlyOwner {
        require(_attackId < attackDetectionCount, "Invalid attack ID");
        require(!attackResolved[_attackId], "Already resolved");
        
        attackResolved[_attackId] = true;
        
        emit AnomalyResolved(attackToNode[_attackId], _attackId);
    }
    
    /**
     * @dev Updates the anomaly threshold
     */
    function updateAnomalyThreshold(uint256 _newThreshold) external onlyOwner {
        require(_newThreshold > 0, "Threshold must be positive");
        anomalyThreshold = _newThreshold;
        
        emit ThresholdUpdated(_newThreshold);
    }
    
    /**
     * @dev Updates the baseline update interval
     */
    function updateBaselineInterval(uint256 _newInterval) external onlyOwner {
        baselineUpdateInterval = _newInterval;
    }
    
    /**
     * @dev Gets all registered nodes
     */
    function getAllNodes() external view returns (address[] memory) {
        return nodeList;
    }
    
    /**
     * @dev Gets node metrics
     */
    function getNodeMetrics(address _nodeAddress) external view returns (
        uint256 latency,
        uint256 throughput,
        uint256 uptime,
        uint256 errorRate,
        uint256 cpuUsage,
        uint256 memoryUsage,
        uint256 connectionCount,
        uint256 lastUpdated
    ) {
        require(registeredNodes[_nodeAddress], "Node not registered");
        
        NodeMetrics storage metrics = nodeMetrics[_nodeAddress];
        return (
            metrics.latency,
            metrics.throughput,
            metrics.uptime,
            metrics.errorRate,
            metrics.cpuUsage,
            metrics.memoryUsage,
            metrics.connectionCount,
            metrics.lastUpdated
        );
    }
    
    /**
     * @dev Gets a node's attack history
     */
    function getNodeAttackHistory(address _nodeAddress) external view returns (
        uint256[] memory attackIds,
        uint256[] memory timestamps,
        uint256[] memory anomalyScores,
        bool[] memory resolved
    ) {
        uint256[] memory attacks = nodeAttackHistory[_nodeAddress];
        uint256 length = attacks.length;
        
        timestamps = new uint256[](length);
        anomalyScores = new uint256[](length);
        resolved = new bool[](length);
        
        for (uint256 i = 0; i < length; i++) {
            uint256 attackId = attacks[i];
            timestamps[i] = attackToTimestamp[attackId];
            anomalyScores[i] = attackToAnomalyScore[attackId];
            resolved[i] = attackResolved[attackId];
        }
        
        return (attacks, timestamps, anomalyScores, resolved);
    }
    
    /**
     * @dev Gets attack details
     */
    function getAttackDetails(uint256 _attackId) external view returns (
        address nodeAddress,
        uint256 timestamp,
        uint256 anomalyScore,
        string memory attackType,
        bool resolved
    ) {
        require(_attackId < attackDetectionCount, "Invalid attack ID");
        
        return (
            attackToNode[_attackId],
            attackToTimestamp[_attackId],
            attackToAnomalyScore[_attackId],
            attackToType[_attackId],
            attackResolved[_attackId]
        );
    }
    
    /**
     * @dev Gets feature contributions for an attack
     */
    function getAttackFeatureContributions(uint256 _attackId, uint256[] calldata _featureIndices) 
        external view returns (uint256[] memory contributions) 
    {
        require(_attackId < attackDetectionCount, "Invalid attack ID");
        
        contributions = new uint256[](_featureIndices.length);
        
        for (uint256 i = 0; i < _featureIndices.length; i++) {
            contributions[i] = attackFeatureContributions[_attackId][_featureIndices[i]];
        }
        
        return contributions;
    }
}