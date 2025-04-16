// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "./NIDRegistry.sol";
import "./NIASRegistry.sol";
import "./ABATLTranslation.sol";

/**
 * @title SequencePathRouter
 * @dev Contract for managing routing sequences between NIDs and NIAS
 */
contract SequencePathRouter {
    // Path record structure
    struct PathRecord {
        bytes32 pathId;                  // Unique path identifier
        bytes32 sourceNID;               // Source NID hash
        bytes32 destinationNIAS;         // Destination NIAS hash
        bytes32[] pathSequence;          // Complete node sequence for the path
        bytes32[] originalPath;          // Original path (for record-keeping)
        bool isActive;                   // Whether the path is active
        uint256 creationTime;            // When the path was created
        uint256 lastUpdated;             // Last time the path was updated
        uint8 pathStatus;                // 0 = Pending, 1 = In Progress, 2 = Completed, 3 = Failed
        string serviceClass;             // Type of service (e.g., "VoIP", "Streaming", "Standard")
    }
    
    // Path status tracking
    struct PathStatus {
        uint256 startTime;               // When transmission started
        uint256 endTime;                 // When transmission ended or expected to end
        uint256 packetsTotal;            // Total number of packets in transmission
        uint256 packetsLost;             // Number of packets lost in transmission
        uint16 measuredLatency;          // Measured latency in ms
        uint8 securityLevel;             // Security level of the path
        bool complianceCheck;            // Whether the path meets QoS requirements
    }

    // Disjoint path structure 
    struct DisjointPath {
        bytes32 pathId;                  // Reference to main path ID
        bytes32[] pathSequence;          // Alternative sequence of nodes
        uint256 creationTime;            // When the disjoint path was created
        bool isActive;                   // Whether the disjoint path is active
    }
    
    // Core registries
    NIDRegistry public nidRegistry;
    NIASRegistry public niasRegistry;
    ABATLTranslation public abatlTranslation;
    
    // Path storage
    mapping(bytes32 => PathRecord) public paths;                 // Path ID to path record
    mapping(bytes32 => PathStatus) public pathStatus;            // Path ID to path status
    mapping(bytes32 => DisjointPath[]) public disjointPaths;     // Path ID to disjoint paths
    mapping(bytes32 => bytes32[]) public nodeToActivePaths;      // Node ID to active paths it's part of
    mapping(bytes32 => uint256) public nodeSuccessRate;          // Node ID to success rate (0-100)
    mapping(bytes32 => uint256) public nodePacketCount;          // Node ID to packet count processed
    
    // Path sequences by service class
    mapping(string => bytes32[]) public pathsByService;
    
    // All path IDs
    bytes32[] public allPathIds;
    
    // Events
    event PathCreated(bytes32 indexed pathId, bytes32 sourceNID, bytes32 destinationNIAS);
    event PathUpdated(bytes32 indexed pathId, bytes32[] pathSequence);
    event PathStatusChanged(bytes32 indexed pathId, uint8 status);
    event DisjointPathCreated(bytes32 indexed pathId, uint256 disjointPathIndex);
    event NodePerformanceUpdated(bytes32 indexed nodeId, uint256 successRate);
    event PathRerouted(bytes32 indexed pathId, bytes32 failedNode, bytes32[] newSequence);
    event TransmissionStarted(bytes32 indexed pathId, uint256 startTime);
    event TransmissionCompleted(bytes32 indexed pathId, uint256 endTime, uint256 packetsLost);
    
    /**
     * @dev Constructor
     * @param _nidRegistryAddress Address of NIDRegistry contract
     * @param _niasRegistryAddress Address of NIASRegistry contract
     * @param _abatlTranslationAddress Address of ABATLTranslation contract
     */
    constructor(
        address _nidRegistryAddress,
        address _niasRegistryAddress,
        address _abatlTranslationAddress
    ) {
        nidRegistry = NIDRegistry(_nidRegistryAddress);
        niasRegistry = NIASRegistry(_niasRegistryAddress);
        abatlTranslation = ABATLTranslation(_abatlTranslationAddress);
    }
    
    /**
     * @dev Create a new path
     * @param _pathId Unique path identifier
     * @param _sourceNID Source NID hash
     * @param _destinationNIAS Destination NIAS hash
     * @param _pathSequence Complete node sequence for the path
     * @param _serviceClass Type of service
     */
    function createPath(
        bytes32 _pathId,
        bytes32 _sourceNID,
        bytes32 _destinationNIAS,
        bytes32[] memory _pathSequence,
        string memory _serviceClass
    ) public {
        // Verify path doesn't already exist
        require(paths[_pathId].creationTime == 0, "Path already exists");
        
        // Verify source and destination exist and are active
        require(nidRegistry.nodeExists(_sourceNID), "Source NID does not exist");
        require(nidRegistry.isNodeActive(_sourceNID), "Source NID is not active");
        require(niasRegistry.niasExists(_destinationNIAS), "Destination NIAS does not exist");
        require(niasRegistry.isNIASActive(_destinationNIAS), "Destination NIAS is not active");
        
        // Verify path sequence starts with source NID and ends with destination NIAS
        require(_pathSequence.length >= 2, "Path must have at least source and destination");
        require(_pathSequence[0] == _sourceNID, "Path must start with source NID");
        require(_pathSequence[_pathSequence.length - 1] == _destinationNIAS, "Path must end with destination NIAS");
        
        // Verify all nodes in path exist and are active
        for (uint i = 1; i < _pathSequence.length - 1; i++) {
            require(nidRegistry.nodeExists(_pathSequence[i]), "Intermediate node does not exist");
            require(nidRegistry.isNodeActive(_pathSequence[i]), "Intermediate node is not active");
        }
        
        // Create path record
        paths[_pathId] = PathRecord({
            pathId: _pathId,
            sourceNID: _sourceNID,
            destinationNIAS: _destinationNIAS,
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
        
        emit PathCreated(_pathId, _sourceNID, _destinationNIAS);
    }
    
    /**
     * @dev Create a disjoint path as an alternative route
     * @param _pathId Path ID to create disjoint path for
     * @param _disjointSequence Disjoint path sequence
     */
    function createDisjointPath(bytes32 _pathId, bytes32[] memory _disjointSequence) public {
        // Verify path exists and is active
        require(paths[_pathId].creationTime != 0, "Path does not exist");
        require(paths[_pathId].isActive, "Path is not active");
        
        // Verify disjoint sequence starts with source NID and ends with destination NIAS
        bytes32 sourceNID = paths[_pathId].sourceNID;
        bytes32 destinationNIAS = paths[_pathId].destinationNIAS;
        
        require(_disjointSequence.length >= 2, "Disjoint path must have at least source and destination");
        require(_disjointSequence[0] == sourceNID, "Disjoint path must start with source NID");
        require(_disjointSequence[_disjointSequence.length - 1] == destinationNIAS, "Disjoint path must end with destination NIAS");
        
        // Verify all nodes in path exist and are active
        for (uint i = 1; i < _disjointSequence.length - 1; i++) {
            require(nidRegistry.nodeExists(_disjointSequence[i]), "Intermediate node does not exist");
            require(nidRegistry.isNodeActive(_disjointSequence[i]), "Intermediate node is not active");
        }
        
        // Verify disjointness - no intermediate nodes overlap with original path
        bytes32[] memory originalPath = paths[_pathId].originalPath;
        for (uint i = 1; i < _disjointSequence.length - 1; i++) {
            for (uint j = 1; j < originalPath.length - 1; j++) {
                require(_disjointSequence[i] != originalPath[j], "Disjoint path cannot share intermediate nodes with original path");
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
    function startTransmission(bytes32 _pathId, uint256 _packetsTotal, uint8 _securityLevel) public {
        // Verify path exists and is active
        require(paths[_pathId].creationTime != 0, "Path does not exist");
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
        bytes32 _pathId,
        uint256 _packetsLost,
        uint16 _measuredLatency,
        bool _complianceCheck
    ) public {
        // Verify path exists and is in progress
        require(paths[_pathId].creationTime != 0, "Path does not exist");
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
        bytes32[] memory pathSequence = paths[_pathId].pathSequence;
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
    function updateNodePerformance(bytes32 _nodeId, bool _pathSuccess) internal {
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
    function reroutePath(bytes32 _pathId, bytes32 _failedNodeId) public {
        // Verify path exists and is active
        require(paths[_pathId].creationTime != 0, "Path does not exist");
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
    function findDisjointPath(bytes32 _pathId, bytes32 _failedNodeId) internal view returns (DisjointPath memory) {
        DisjointPath[] storage paths = disjointPaths[_pathId];
        
        for (uint i = 0; i < paths.length; i++) {
            // Check if this disjoint path does not include the failed node
            bool isValidAlternative = true;
            for (uint j = 0; j < paths[i].pathSequence.length; j++) {
                if (paths[i].pathSequence[j] == _failedNodeId) {
                    isValidAlternative = false;
                    break;
                }
            }
            
            // Return the first valid alternative path
            if (isValidAlternative && paths[i].isActive) {
                return paths[i];
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
    function getNodePerformance(bytes32 _nodeId) public view returns (uint256 successRate, uint256 packetCount) {
        return (nodeSuccessRate[_nodeId], nodePacketCount[_nodeId]);
    }
    
    /**
     * @dev Get all paths for a specific service class
     * @param _serviceClass Service class to query
     * @return bytes32[] Array of path IDs
     */
    function getPathsByServiceClass(string memory _serviceClass) public view returns (bytes32[] memory) {
        return pathsByService[_serviceClass];
    }
}