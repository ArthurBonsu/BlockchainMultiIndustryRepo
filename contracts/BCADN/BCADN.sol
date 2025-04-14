// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title BCADN - Blockchain-Cellular Adaptive Decentralized Network
 * @dev Implementation of the BCADN architecture with dynamic node weighting, 
 * probability gap, and proactive defense mechanism
 */
contract BCADN is Ownable, ReentrancyGuard {
    // Node status enum
    enum NodeStatus {
        Active,
        Probation,
        Excluded,
        Pending
    }

    // Node structure
    struct Node {
        address nodeAddress;
        uint256 performance;    // Performance metric
        uint256 reliability;    // Reliability score
        uint256 anomalyScore;   // Anomaly detection score
        uint256 weight;         // Dynamic weight
        uint256 isolationTime;  // Time when node was isolated (if applicable)
        NodeStatus status;      // Current status of the node
    }

    // Shard structure
    struct Shard {
        uint256 id;
        address[] nodes;
        uint256 capacity;
        uint256 currentLoad;
        bool active;
    }

    // Transaction structure
    struct Transaction {
        bytes32 txHash;
        address sender;
        address receiver;
        uint256 amount;
        uint256 fee;
        uint256 timestamp;
        uint256 processingTime;
        bool completed;
    }

    // Attack data structure
    struct AttackData {
        address node;
        uint256 timestamp;
        uint256 anomalyScore;
        string attackType;
        bool resolved;
    }

    // Network parameters
    uint256 public alpha = 10;  // Fee weighting factor
    uint256 public beta = 20;   // Performance weighting factor
    uint256 public gamma = 30;  // Anomaly weighting factor
    uint256 public delta = 50;  // Congestion responsiveness parameter
    uint256 public mu = 5;      // Transaction age influence parameter
    
    uint256 public anomalyThreshold = 30;
    uint256 public probationPeriod = 3600;  // 1 hour in seconds
    uint256 public minProbability = 20;     // Probability gap minimum
    uint256 public maxProbability = 80;     // Probability gap maximum
    uint256 public currentGap;              // Probability gap size

    // Network metrics
    uint256 public pendingTransactions;
    uint256 public networkCapacity = 1000;  // Default capacity
    uint256 public congestionIndex;         // Calculated as pendingTransactions / networkCapacity

    // Node registry
    mapping(address => Node) public nodes;
    address[] public nodesList;
    
    // Shards registry
    mapping(uint256 => Shard) public shards;
    uint256[] public shardIds;
    uint256 public nextShardId = 1;
    
    // Transaction assignment
    mapping(bytes32 => uint256) public transactionToShard;
    
    // Transactions registry
    mapping(bytes32 => Transaction) public transactions;
    bytes32[] public transactionHashes;
    
    // Attack history
    AttackData[] public attackHistory;
    mapping(address => uint256[]) public nodeAttackIndices;

    // Events
    event NodeRegistered(address indexed nodeAddress);
    event NodeWeightUpdated(address indexed nodeAddress, uint256 newWeight);
    event NodeStatusChanged(address indexed nodeAddress, NodeStatus newStatus);
    event ShardCreated(uint256 indexed shardId);
    event NodeAddedToShard(uint256 indexed shardId, address indexed node);
    event NodeRemovedFromShard(uint256 indexed shardId, address indexed node);
    event TransactionAssigned(bytes32 indexed txHash, uint256 indexed shardId);
    event TransactionSubmitted(bytes32 indexed txHash, address indexed sender, address indexed receiver, uint256 amount, uint256 fee);
    event TransactionCompleted(bytes32 indexed txHash, uint256 processingTime);
    event AnomalyDetected(address indexed node, uint256 anomalyScore, string attackType);
    event AnomalyResolved(address indexed node, uint256 indexed attackIndex);
    event CongestionUpdated(uint256 indexed congestionIndex, uint256 pendingTransactions, uint256 networkCapacity);
    event GapUpdated(uint256 newGap, uint256 minProbability, uint256 maxProbability);

    constructor() {
        // Initialize probability gap
        currentGap = maxProbability - minProbability;
    }

    /**
     * @dev Updates network parameters
     */
    function updateNetworkParams(
        uint256 _alpha,
        uint256 _beta,
        uint256 _gamma,
        uint256 _delta,
        uint256 _mu
    ) external onlyOwner {
        alpha = _alpha;
        beta = _beta;
        gamma = _gamma;
        delta = _delta;
        mu = _mu;
    }
    
    /**
     * @dev Updates the anomaly threshold and probation period
     */
    function updateThresholds(
        uint256 _anomalyThreshold,
        uint256 _probationPeriod
    ) external onlyOwner {
        anomalyThreshold = _anomalyThreshold;
        probationPeriod = _probationPeriod;
    }
    
    /**
     * @dev Updates the probability range for the gap mechanism
     */
    function updateProbabilityRange(
        uint256 _minProbability,
        uint256 _maxProbability
    ) external onlyOwner {
        require(_minProbability < _maxProbability, "Invalid probability range");
        
        minProbability = _minProbability;
        maxProbability = _maxProbability;
        currentGap = _maxProbability - _minProbability;
        
        emit GapUpdated(currentGap, minProbability, maxProbability);
    }
    
    /**
     * @dev Adjusts a probability to fall within the gap if needed
     */
    function adjustToProbabilityGap(uint256 _probability) public view returns (uint256) {
        if (_probability < minProbability) {
            return minProbability;
        } else if (_probability > maxProbability) {
            return maxProbability;
        }
        return _probability;
    }
    
    /**
     * @dev Checks if a given probability falls within the allowed gap
     */
    function isWithinGap(uint256 _probability) public view returns (bool) {
        return _probability >= minProbability && _probability <= maxProbability;
    }
    
    /**
     * @dev Updates network capacity
     */
    function setNetworkCapacity(uint256 _capacity) external onlyOwner {
        networkCapacity = _capacity;
        _updateCongestionIndex();
    }
    
    /**
     * @dev Updates congestion index based on pending transactions and network capacity
     */
    function _updateCongestionIndex() internal {
        if (networkCapacity > 0) {
            congestionIndex = (pendingTransactions * 1e18) / networkCapacity;
        } else {
            congestionIndex = 0;
        }
        
        emit CongestionUpdated(congestionIndex, pendingTransactions, networkCapacity);
    }
    
    /**
     * @dev Calculates dynamic fee based on current congestion
     * @param _baseFee Base transaction fee
     * @return dynamicFee The adjusted transaction fee
     */
    function calculateDynamicFee(uint256 _baseFee) public view returns (uint256) {
        if (networkCapacity == 0) return _baseFee;
        
        // Formula: F = Fb * (1 + Tcurrent/Tmax)
        return _baseFee * (1e18 + congestionIndex) / 1e18;
    }
    
    /**
     * @dev Registers a new node in the network
     */
    function registerNode(
        address _nodeAddress,
        uint256 _performance,
        uint256 _reliability
    ) external onlyOwner {
        require(nodes[_nodeAddress].nodeAddress == address(0), "Node already exists");
        
        // Initial weight calculation: W(0) = alpha * Fee + beta * Performance
        uint256 initialWeight = (alpha * 100) + (beta * _performance);
        
        Node memory newNode = Node({
            nodeAddress: _nodeAddress,
            performance: _performance,
            reliability: _reliability,
            anomalyScore: 0,
            weight: initialWeight,
            isolationTime: 0,
            status: NodeStatus.Active
        });
        
        nodes[_nodeAddress] = newNode;
        nodesList.push(_nodeAddress);
        
        emit NodeRegistered(_nodeAddress);
    }
    
    /**
     * @dev Updates a node's performance metrics
     */
    function updateNodeMetrics(
        address _nodeAddress,
        uint256 _performance,
        uint256 _reliability,
        uint256 _anomalyScore
    ) external {
        require(nodes[_nodeAddress].nodeAddress != address(0), "Node does not exist");
        
        Node storage node = nodes[_nodeAddress];
        node.performance = _performance;
        node.reliability = _reliability;
        node.anomalyScore = _anomalyScore;
        
        // Check anomaly threshold
        if (_anomalyScore > anomalyThreshold && node.status == NodeStatus.Active) {
            node.status = NodeStatus.Probation;
            node.isolationTime = block.timestamp;
            emit NodeStatusChanged(_nodeAddress, NodeStatus.Probation);
        }
        
        // Dynamic weight adjustment
        _updateNodeWeight(_nodeAddress);
    }
    
    /**
     * @dev Updates the node's weight according to the dynamic weighting formula
     */
    function _updateNodeWeight(address _nodeAddress) internal {
        Node storage node = nodes[_nodeAddress];
        
        // Calculate dynamic weight based on Algorithm 1
        // W(t) = (alpha * Fee + beta * Performance - gamma * AnomalyScore)
        uint256 newWeight = (alpha * 100) + (beta * node.performance) - (gamma * node.anomalyScore);
        
        // Apply probability gap to ensure fairness
        newWeight = adjustToProbabilityGap(newWeight);
        
        node.weight = newWeight;
        emit NodeWeightUpdated(_nodeAddress, newWeight);
    }
    
    /**
     * @dev Checks probation status and potentially reintegrates nodes
     */
    function checkProbationStatus(address _nodeAddress) external {
        Node storage node = nodes[_nodeAddress];
        require(node.nodeAddress != address(0), "Node does not exist");
        
        if (node.status == NodeStatus.Probation) {
            if (
                block.timestamp - node.isolationTime >= probationPeriod &&
                node.anomalyScore <= anomalyThreshold
            ) {
                node.status = NodeStatus.Active;
                emit NodeStatusChanged(_nodeAddress, NodeStatus.Active);
            }
        }
    }
    
    /**
     * @dev Gets all nodes with their current weights and status
     */
    function getAllNodes() external view returns (address[] memory, uint256[] memory, NodeStatus[] memory) {
        uint256 length = nodesList.length;
        uint256[] memory weights = new uint256[](length);
        NodeStatus[] memory statuses = new NodeStatus[](length);
        
        for (uint256 i = 0; i < length; i++) {
            weights[i] = nodes[nodesList[i]].weight;
            statuses[i] = nodes[nodesList[i]].status;
        }
        
        return (nodesList, weights, statuses);
    }
    
    /**
     * @dev Creates a new shard
     */
    function createShard(uint256 _capacity) external onlyOwner {
        Shard memory newShard = Shard({
            id: nextShardId,
            nodes: new address[](0),
            capacity: _capacity,
            currentLoad: 0,
            active: true
        });
        
        shards[nextShardId] = newShard;
        shardIds.push(nextShardId);
        
        emit ShardCreated(nextShardId);
        nextShardId++;
    }
    
    /**
     * @dev Adds a node to a shard
     */
    function addNodeToShard(uint256 _shardId, address _node) external onlyOwner {
        require(shards[_shardId].id == _shardId, "Shard does not exist");
        require(nodes[_node].nodeAddress != address(0), "Node does not exist");
        
        shards[_shardId].nodes.push(_node);
        emit NodeAddedToShard(_shardId, _node);
    }
    
    /**
     * @dev Removes a node from a shard
     */
    function removeNodeFromShard(uint256 _shardId, address _node) external onlyOwner {
        require(shards[_shardId].id == _shardId, "Shard does not exist");
        
        Shard storage shard = shards[_shardId];
        uint256 length = shard.nodes.length;
        bool found = false;
        uint256 index;
        
        for (uint256 i = 0; i < length; i++) {
            if (shard.nodes[i] == _node) {
                found = true;
                index = i;
                break;
            }
        }
        
        require(found, "Node not in shard");
        
        // Remove node by replacing with the last element and popping
        if (index < length - 1) {
            shard.nodes[index] = shard.nodes[length - 1];
        }
        shard.nodes.pop();
        
        emit NodeRemovedFromShard(_shardId, _node);
    }
    
    /**
     * @dev Returns all shard IDs
     */
    function getAllShards() external view returns (uint256[] memory) {
        return shardIds;
    }
    
    /**
     * @dev Assigns a transaction to a shard using randomized selection
     */
    function assignTransactionToShard(bytes32 _txHash) public returns (uint256) {
        require(shardIds.length > 0, "No shards available");
        
        // Implement randomized transaction selection (from Random Selector)
        uint256 randomShardIndex = uint256(keccak256(abi.encodePacked(block.timestamp, _txHash))) % shardIds.length;
        uint256 shardId = shardIds[randomShardIndex];
        
        // Assign transaction to shard
        transactionToShard[_txHash] = shardId;
        shards[shardId].currentLoad++;
        
        emit TransactionAssigned(_txHash, shardId);
        return shardId;
    }
    
    /**
     * @dev Records a detected anomaly
     */
    function recordAnomaly(
        address _node,
        uint256 _anomalyScore,
        string calldata _attackType
    ) external onlyOwner {
        require(nodes[_node].nodeAddress != address(0), "Node does not exist");
        
        AttackData memory newAttack = AttackData({
            node: _node,
            timestamp: block.timestamp,
            anomalyScore: _anomalyScore,
            attackType: _attackType,
            resolved: false
        });
        
        uint256 newIndex = attackHistory.length;
        attackHistory.push(newAttack);
        nodeAttackIndices[_node].push(newIndex);
        
        emit AnomalyDetected(_node, _anomalyScore, _attackType);
    }
    
    /**
     * @dev Marks an anomaly as resolved
     */
    function resolveAnomaly(uint256 _attackIndex) external onlyOwner {
        require(_attackIndex < attackHistory.length, "Invalid attack index");
        require(!attackHistory[_attackIndex].resolved, "Already resolved");
        
        attackHistory[_attackIndex].resolved = true;
        
        emit AnomalyResolved(attackHistory[_attackIndex].node, _attackIndex);
    }
    
    /**
     * @dev Gets all attack history
     */
     function getAttackHistory() external view returns (
        address[] memory nodeAddresses,
        uint256[] memory timestamps,
        uint256[] memory anomalyScores,
        bool[] memory resolved
    ) {
        uint256 length = attackHistory.length;
        
        nodeAddresses = new address[](length);
        timestamps = new uint256[](length);
        anomalyScores = new uint256[](length);
        resolved = new bool[](length);
        
        for (uint256 i = 0; i < length; i++) {
            AttackData storage attack = attackHistory[i];
            nodeAddresses[i] = attack.node;
            timestamps[i] = attack.timestamp;
            anomalyScores[i] = attack.anomalyScore;
            resolved[i] = attack.resolved;
        }
        
        return (nodeAddresses, timestamps, anomalyScores, resolved);
    }
    /**
     * @dev Submits a new transaction to the network
     */
    function submitTransaction(
        address _receiver,
        uint256 _amount,
        uint256 _baseFee
    ) external payable nonReentrant returns (bytes32) {
        // Calculate dynamic fee
        uint256 dynamicFee = calculateDynamicFee(_baseFee);
        require(msg.value >= dynamicFee, "Insufficient fee");
        
        // Generate transaction hash
        bytes32 txHash = keccak256(abi.encodePacked(
            msg.sender,
            _receiver,
            _amount,
            dynamicFee,
            block.timestamp
        ));
        
        // Create transaction record
        Transaction memory newTx = Transaction({
            txHash: txHash,
            sender: msg.sender,
            receiver: _receiver,
            amount: _amount,
            fee: dynamicFee,
            timestamp: block.timestamp,
            processingTime: 0,
            completed: false
        });
        
        transactions[txHash] = newTx;
        transactionHashes.push(txHash);
        
        // Update network congestion
        pendingTransactions++;
        _updateCongestionIndex();
        
        // Assign to shard using randomized selection
        assignTransactionToShard(txHash);
        
        emit TransactionSubmitted(txHash, msg.sender, _receiver, _amount, dynamicFee);
        
        // Refund excess fee
        if (msg.value > dynamicFee) {
            payable(msg.sender).transfer(msg.value - dynamicFee);
        }
        
        return txHash;
    }
    
    /**
     * @dev Processes and completes a transaction
     * @notice This would typically be called by validators/miners
     */
    function processTransaction(bytes32 _txHash) external onlyOwner nonReentrant {
        Transaction storage transaction = transactions[_txHash];
        require(transaction.txHash == _txHash, "Transaction does not exist");
        require(!transaction.completed, "Transaction already completed");
        
        // Update transaction status
        transaction.completed = true;
        transaction.processingTime = block.timestamp - transaction.timestamp;
        
        // Update network and shard status
        pendingTransactions--;
        _updateCongestionIndex();
        
        // Complete transaction in shard
        uint256 shardId = transactionToShard[_txHash];
        if (shardId > 0 && shards[shardId].currentLoad > 0) {
            shards[shardId].currentLoad--;
        }
        delete transactionToShard[_txHash];
        
        emit TransactionCompleted(_txHash, transaction.processingTime);
    }
    /**
     * @dev Gets all transaction hashes
     */
    function getAllTransactionHashes() external view returns (bytes32[] memory) {
        return transactionHashes;
    }
    
    /**
     * @dev Withdraws accumulated fees
     */
    function withdrawFees() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
}




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