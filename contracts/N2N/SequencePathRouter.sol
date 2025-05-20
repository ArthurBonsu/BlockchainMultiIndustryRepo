// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title SequencePathRouter
 * @dev Contract for managing routing sequences between nodes
 */
contract SequencePathRouter {
    // Path record structure
    struct PathRecord {
        string pathId;                  // Unique path identifier
        string sourceNode;              // Source node
        string destinationNode;         // Destination node
        string[] pathSequence;          // Complete node sequence for the path
        string[] originalPath;          // Original path (for record-keeping)
        bool isActive;                  // Whether the path is active
        uint256 creationTime;           // When the path was created
        uint256 lastUpdated;            // Last time the path was updated
        uint8 pathStatus;               // 0 = Pending, 1 = In Progress, 2 = Completed, 3 = Failed
        string serviceClass;            // Type of service (e.g., "VoIP", "Streaming", "Standard")
    }
    
    // Path status tracking
    struct PathStatus {
        uint256 startTime;              // When transmission started
        uint256 endTime;                // When transmission ended or expected to end
        uint256 packetsTotal;           // Total number of packets in transmission
        uint256 packetsLost;            // Number of packets lost in transmission
        uint16 measuredLatency;         // Measured latency in ms
        uint8 securityLevel;            // Security level of the path
        bool complianceCheck;           // Whether the path meets QoS requirements
    }

    // Disjoint path structure 
    struct DisjointPath {
        string pathId;                  // Reference to main path ID
        string[] pathSequence;          // Alternative sequence of nodes
        uint256 creationTime;           // When the disjoint path was created
        bool isActive;                  // Whether the disjoint path is active
    }
    
    // Path storage
    mapping(string => PathRecord) public paths;               // Path ID to path record
    mapping(string => PathStatus) public pathStatus;            // Path ID to path status
    mapping(string => DisjointPath[]) public disjointPaths;     // Path ID to disjoint paths
    mapping(string => string[]) public nodeToActivePaths;       // Node ID to active paths it's part of
    mapping(string => uint256) public nodeSuccessRate;          // Node ID to success rate (0-100)
    mapping(string => uint256) public nodePacketCount;          // Node ID to packet count processed
    
    // Path sequences by service class
    mapping(string => string[]) public pathsByService;
    
    // All path IDs
    string[] public allPathIds;
    
    // Events
    event PathCreated(string indexed pathId, string sourceNode, string destinationNode);
    event PathUpdated(string indexed pathId, string[] pathSequence);
    event PathStatusChanged(string indexed pathId, uint8 status);
    event DisjointPathCreated(string indexed pathId, uint256 disjointPathIndex);
    event NodePerformanceUpdated(string indexed nodeId, uint256 successRate);
    event PathRerouted(string indexed pathId, string failedNode, string[] newSequence);
    event TransmissionStarted(string indexed pathId, uint256 startTime);
    event TransmissionCompleted(string indexed pathId, uint256 endTime, uint256 packetsLost);
    
    /**
     * @dev Create a new path
     * @param _pathId Unique path identifier
     * @param _sourceNode Source node
     * @param _destinationNode Destination node
     * @param _pathSequence Complete node sequence for the path
     * @param _serviceClass Type of service
     */
    function createPath(
        string memory _pathId,
        string memory _sourceNode,
        string memory _destinationNode,
        string[] memory _pathSequence,
        string memory _serviceClass
    ) public {
        // Verify path doesn't already exist
        require(bytes(paths[_pathId].pathId).length == 0, "Path already exists");
        
        // Verify path sequence starts with source node and ends with destination node
        require(_pathSequence.length >= 2, "Path must have at least source and destination");
        require(keccak256(bytes(_pathSequence[0])) == keccak256(bytes(_sourceNode)), "Path must start with source node");
        require(keccak256(bytes(_pathSequence[_pathSequence.length - 1])) == keccak256(bytes(_destinationNode)), "Path must end with destination node");
        
        // Create path record
        paths[_pathId] = PathRecord({
            pathId: _pathId,
            sourceNode: _sourceNode,
            destinationNode: _destinationNode,
            pathSequence: _pathSequence,
            originalPath: _pathSequence,
            isActive: true,
            creationTime: block.timestamp,
            lastUpdated: block.timestamp,
            pathStatus: 0, // Pending
            serviceClass: _serviceClass
        });
        
        // Initialize path status
        pathStatus[_pathId] = PathStatus({
            startTime: 0,
            endTime: 0,
            packetsTotal: 0,
            packetsLost: 0,
            measuredLatency: 0,
            securityLevel: 0,
            complianceCheck: false
        });
        
        // Add path ID to lists
        allPathIds.push(_pathId);
        pathsByService[_serviceClass].push(_pathId);
        
        // Add path to node's active paths
        for (uint i = 0; i < _pathSequence.length; i++) {
            nodeToActivePaths[_pathSequence[i]].push(_pathId);
        }
        
        emit PathCreated(_pathId, _sourceNode, _destinationNode);
    }

    /**
     * @dev Create a disjoint path as an alternative route
     * @param _pathId Path ID to create disjoint path for
     * @param _disjointSequence Disjoint path sequence
     */
    function createDisjointPath(string memory _pathId, string[] memory _disjointSequence) public {
        // Verify path exists and is active
        require(bytes(paths[_pathId].pathId).length != 0, "Path does not exist");
        require(paths[_pathId].isActive, "Path is not active");
        
        // Verify disjoint sequence starts with source node and ends with destination node
        string memory sourceNode = paths[_pathId].sourceNode;
        string memory destinationNode = paths[_pathId].destinationNode;
        
        require(_disjointSequence.length >= 2, "Disjoint path must have at least source and destination");
        require(keccak256(bytes(_disjointSequence[0])) == keccak256(bytes(sourceNode)), "Disjoint path must start with source node");
        require(keccak256(bytes(_disjointSequence[_disjointSequence.length - 1])) == keccak256(bytes(destinationNode)), "Disjoint path must end with destination node");
        
        // Verify disjointness - no intermediate nodes overlap with original path
        string[] memory originalPath = paths[_pathId].originalPath;
        for (uint i = 1; i < _disjointSequence.length - 1; i++) {
            for (uint j = 1; j < originalPath.length - 1; j++) {
                require(keccak256(bytes(_disjointSequence[i])) != keccak256(bytes(originalPath[j])), "Disjoint path cannot share intermediate nodes with original path");
            }
        }
        
        // Create disjoint path
        DisjointPath memory disjointPath = DisjointPath({
            pathId: _pathId,
            pathSequence: _disjointSequence,
            creationTime: block.timestamp,
            isActive: true
        });
        
        // Add disjoint path to list
        disjointPaths[_pathId].push(disjointPath);
        
        // Add disjoint path to node's active paths
        for (uint i = 0; i < _disjointSequence.length; i++) {
            nodeToActivePaths[_disjointSequence[i]].push(_pathId);
        }
        
        emit DisjointPathCreated(_pathId, disjointPaths[_pathId].length - 1);
    }
    
    /**
     * @dev Start transmission on a path
     * @param _pathId Path ID
     * @param _packetsTotal Total number of packets in transmission
     * @param _securityLevel Security level of the path
     */
    function startTransmission(string memory _pathId, uint256 _packetsTotal, uint8 _securityLevel) public {
        // Verify path exists and is active
        require(bytes(paths[_pathId].pathId).length != 0, "Path does not exist");
        require(paths[_pathId].isActive, "Path is not active");
        require(paths[_pathId].pathStatus == 0, "Path is not in pending status");
        
        // Update path status
        paths[_pathId].pathStatus = 1; // In Progress
        pathStatus[_pathId].startTime = block.timestamp;
        pathStatus[_pathId].packetsTotal = _packetsTotal;
        pathStatus[_pathId].securityLevel = _securityLevel;
        
        emit TransmissionStarted(_pathId, block.timestamp);
        emit PathStatusChanged(_pathId, 1);
    }
    
    /**
     * @dev Complete transmission on a path
     * @param _pathId Path ID
     * @param _packetsLost Number of packets lost
     * @param _measuredLatency Measured latency in ms
     * @param _complianceCheck Whether the path meets QoS requirements
     */
    function completeTransmission(
        string memory _pathId,
        uint256 _packetsLost,
        uint16 _measuredLatency,
        bool _complianceCheck
    ) public {
        // Verify path exists and is in progress
        require(bytes(paths[_pathId].pathId).length != 0, "Path does not exist");
        require(paths[_pathId].pathStatus == 1, "Path is not in progress");
        
        // Update path status
        paths[_pathId].pathStatus = _complianceCheck ? 2 : 3; // Completed or Failed
        
        // Update transmission details
        PathStatus storage status = pathStatus[_pathId];
        status.endTime = block.timestamp;
        status.packetsLost = _packetsLost;
        status.measuredLatency = _measuredLatency;
        status.complianceCheck = _complianceCheck;
        
        // Update node performance metrics
        string[] memory pathSequence = paths[_pathId].pathSequence;
        for (uint i = 0; i < pathSequence.length; i++) {
            updateNodePerformance(pathSequence[i], _complianceCheck);
        }
        
        emit TransmissionCompleted(_pathId, block.timestamp, _packetsLost);
        emit PathStatusChanged(_pathId, paths[_pathId].pathStatus);
    }
    
    /**
     * @dev Update node performance metrics
     * @param _nodeId Node identifier
     * @param _pathSuccess Whether the path was successful
     */
    function updateNodePerformance(string memory _nodeId, bool _pathSuccess) internal {
        // Increment total packet count for the node
        nodePacketCount[_nodeId]++;
        
        // Calculate and update success rate
        uint256 currentSuccessRate = nodeSuccessRate[_nodeId];
        uint256 newSuccessRate;
        
        if (currentSuccessRate == 0) {
            // First measurement
            newSuccessRate = _pathSuccess ? 100 : 0;
        } else {
            // Weighted average
            newSuccessRate = (currentSuccessRate * (nodePacketCount[_nodeId] - 1) + 
                              (_pathSuccess ? 100 : 0)) / nodePacketCount[_nodeId];
        }
        
        nodeSuccessRate[_nodeId] = newSuccessRate;
        
        emit NodePerformanceUpdated(_nodeId, newSuccessRate);
    }
    
    /**
     * @dev Reroute a path if a node fails
     * @param _pathId Path ID
     * @param _failedNodeId Node that failed
     */
    function reroutePath(string memory _pathId, string memory _failedNodeId) public {
        // Verify path exists and is active
        require(bytes(paths[_pathId].pathId).length != 0, "Path does not exist");
        require(paths[_pathId].isActive, "Path is not active");
        
        // Find a disjoint path
        DisjointPath memory alternativePath = findDisjointPath(_pathId, _failedNodeId);
        
        // Update path sequence
        paths[_pathId].pathSequence = alternativePath.pathSequence;
        paths[_pathId].lastUpdated = block.timestamp;
        
        emit PathRerouted(_pathId, _failedNodeId, alternativePath.pathSequence);
    }
    
    /**
     * @dev Find a disjoint path avoiding a failed node
     * @param _pathId Path ID
     * @param _failedNodeId Node that failed
     * @return DisjointPath An alternative path
     */
    function findDisjointPath(string memory _pathId, string memory _failedNodeId) internal view returns (DisjointPath memory) {
        DisjointPath[] storage paths2 = disjointPaths[_pathId];
        
        for (uint i = 0; i < paths2.length; i++) {
            // Check if this disjoint path does not include the failed node
            bool isValidAlternative = true;
            for (uint j = 0; j < paths2[i].pathSequence.length; j++) {
                if (keccak256(bytes(paths2[i].pathSequence[j])) == keccak256(bytes(_failedNodeId))) {
                    isValidAlternative = false;
                    break;
                }
            }
            
            // Return the first valid alternative path
            if (isValidAlternative && paths2[i].isActive) {
                return paths2[i];
            }
        }
        
        // Revert if no alternative path found
        revert("No alternative path available");
    }
    
    /**
     * @dev Get performance metrics for a specific node
     * @param _nodeId Node identifier
     * @return successRate Node's success rate
     * @return packetCount Total packets processed by the node
     */
    function getNodePerformance(string memory _nodeId) public view returns (uint256 successRate, uint256 packetCount) {
        return (nodeSuccessRate[_nodeId], nodePacketCount[_nodeId]);
    }
    
    /**
     * @dev Get all paths for a specific service class
     * @param _serviceClass Service class to query
     * @return string[] Array of path IDs
     */
    function getPathsByServiceClass(string memory _serviceClass) public view returns (string[] memory) {
        return pathsByService[_serviceClass];
    }
}