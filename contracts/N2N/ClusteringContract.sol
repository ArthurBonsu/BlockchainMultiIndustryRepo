// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "./NIDRegistry.sol";
import "./NIASRegistry.sol";
import "./ABATLTranslation.sol";

/**
 * @title ClusteringContract
 * @dev Contract for clustering nodes based on attributes and managing cluster dynamics
 */
contract ClusteringContract {
    // Cluster structure
    struct Cluster {
        uint8 clusterId;              // Unique cluster identifier
        string clusterName;           // Human-readable name
        uint8 clusterType;            // 0 = NAP cluster, 1 = B-BGP cluster
        uint256 creationTime;         // When the cluster was created
        uint256 validUntil;           // Expiration time for the cluster
        uint8 securityLevel;          // Required security level for members
        uint16 maxLatency;            // Maximum acceptable latency in ms
        uint16 minBandwidth;          // Minimum required bandwidth in Mbps
        bool isActive;                // Whether the cluster is active
        uint256 nodeCount;            // Number of nodes in the cluster
    }
    
    // ClusterMembership record
    struct ClusterMembership {
        bytes32 nodeId;              // Node ID (NID or NIAS)
        uint8 clusterId;             // Cluster ID
        uint256 joinTime;            // When the node joined the cluster
        bool isActive;               // Whether the membership is active
    }
    
    // Metrics structure for cluster
    struct ClusterMetrics {
        uint16 avgLatency;           // Average latency across all nodes
        uint16 avgBandwidth;         // Average bandwidth across all nodes
        uint8 avgSecurityLevel;      // Average security level across all nodes
        uint256 successfulTransmissions; // Number of successful transmissions
        uint256 failedTransmissions;  // Number of failed transmissions
        uint256 lastUpdated;          // When metrics were last updated
    }
    
    // Core registries
    NIDRegistry public nidRegistry;
    NIASRegistry public niasRegistry;
    ABATLTranslation public abatlTranslation;
    
    // Cluster storage
    mapping(uint8 => Cluster) public clusters;                    // Cluster ID to Cluster
    mapping(uint8 => bytes32[]) public clusterMembers;            // Cluster ID to member node IDs
    mapping(bytes32 => uint8[]) public nodeClusters;              // Node ID to array of cluster IDs
    mapping(bytes32 => mapping(uint8 => ClusterMembership)) public memberships; // Node ID and Cluster ID to ClusterMembership
    mapping(uint8 => ClusterMetrics) public clusterMetrics;       // Cluster ID to ClusterMetrics
    
    // All cluster IDs
    uint8[] public allClusterIds;
    
    // Events
    event ClusterCreated(uint8 indexed clusterId, string clusterName, uint8 clusterType);
    event ClusterUpdated(uint8 indexed clusterId, uint256 validUntil, bool isActive);
    event NodeAddedToCluster(bytes32 indexed nodeId, uint8 indexed clusterId);
    event NodeRemovedFromCluster(bytes32 indexed nodeId, uint8 indexed clusterId);
    event ClusterMetricsUpdated(uint8 indexed clusterId, uint16 avgLatency, uint16 avgBandwidth);
    event ClusterReclustered(uint8 indexed oldClusterId, uint8 indexed newClusterId);
    
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
     * @dev Create a new cluster
     * @param _clusterId Unique cluster identifier
     * @param _clusterName Human-readable name
     * @param _clusterType Cluster type (0 = NAP, 1 = B-BGP)
     * @param _validUntil Expiration time
     * @param _securityLevel Required security level
     * @param _maxLatency Maximum acceptable latency
     * @param _minBandwidth Minimum required bandwidth
     */
    function createCluster(
        uint8 _clusterId,
        string memory _clusterName,
        uint8 _clusterType,
        uint256 _validUntil,
        uint8 _securityLevel,
        uint16 _maxLatency,
        uint16 _minBandwidth
    ) public {
        // Verify cluster doesn't already exist
        require(clusters[_clusterId].creationTime == 0, "Cluster already exists");
        require(_clusterType <= 1, "Invalid cluster type");
        require(_validUntil > block.timestamp, "Valid until must be in the future");
        
        // Create cluster
        clusters[_clusterId] = Cluster({
            clusterId: _clusterId,
            clusterName: _clusterName,
            clusterType: _clusterType,
            creationTime: block.timestamp,
            validUntil: _validUntil,
            securityLevel: _securityLevel,
            maxLatency: _maxLatency,
            minBandwidth: _minBandwidth,
            isActive: true,
            nodeCount: 0
        });
        
        // Initialize cluster metrics
        clusterMetrics[_clusterId] = ClusterMetrics({
            avgLatency: 0,
            avgBandwidth: 0,
            avgSecurityLevel: 0,
            successfulTransmissions: 0,
            failedTransmissions: 0,
            lastUpdated: block.timestamp
        });
        
        // Add to list of all cluster IDs
        allClusterIds.push(_clusterId);
        
        emit ClusterCreated(_clusterId, _clusterName, _clusterType);
    }
    
    /**
     * @dev Update cluster parameters
     * @param _clusterId Cluster ID
     * @param _validUntil New expiration time
     * @param _securityLevel New security level
     * @param _maxLatency New maximum latency
     * @param _minBandwidth New minimum bandwidth
     * @param _isActive New active status
     */
    function updateCluster(
        uint8 _clusterId,
        uint256 _validUntil,
        uint8 _securityLevel,
        uint16 _maxLatency,
        uint16 _minBandwidth,
        bool _isActive
    ) public {
        // Verify cluster exists
        require(clusters[_clusterId].creationTime != 0, "Cluster does not exist");
        require(_validUntil > block.timestamp, "Valid until must be in the future");
        
        // Update cluster
        clusters[_clusterId].validUntil = _validUntil;
        clusters[_clusterId].securityLevel = _securityLevel;
        clusters[_clusterId].maxLatency = _maxLatency;
        clusters[_clusterId].minBandwidth = _minBandwidth;
        clusters[_clusterId].isActive = _isActive;
        
        emit ClusterUpdated(_clusterId, _validUntil, _isActive);
    }
    
    /**
     * @dev Add a node to a cluster
     * @param _nodeId Node ID
     * @param _clusterId Cluster ID
     */
    function addNodeToCluster(bytes32 _nodeId, uint8 _clusterId) public {
        // Verify cluster exists and is active
        require(clusters[_clusterId].creationTime != 0, "Cluster does not exist");
        require(clusters[_clusterId].isActive, "Cluster is not active");
        require(clusters[_clusterId].validUntil > block.timestamp, "Cluster has expired");
        
        // Verify node exists and is active based on cluster type
        uint8 clusterType = clusters[_clusterId].clusterType;
        if (clusterType == 0) {
            // NAP cluster, check NID
            require(nidRegistry.nodeExists(_nodeId), "Node does not exist");
            require(nidRegistry.isNodeActive(_nodeId), "Node is not active");
        } else {
            // B-BGP cluster, check NIAS
            require(niasRegistry.niasExists(_nodeId), "NIAS does not exist");
            require(niasRegistry.isNIASActive(_nodeId), "NIAS is not active");
        }
        
        // Verify node is not already in the cluster
        require(memberships[_nodeId][_clusterId].joinTime == 0, "Node is already in the cluster");
        
        // Add node to cluster
        memberships[_nodeId][_clusterId] = ClusterMembership({
            nodeId: _nodeId,
            clusterId: _clusterId,
            joinTime: block.timestamp,
            isActive: true
        });
        
        // Update cluster members and node clusters
        clusterMembers[_clusterId].push(_nodeId);
        nodeClusters[_nodeId].push(_clusterId);
        
        // Update cluster node count
        clusters[_clusterId].nodeCount++;
        
        emit NodeAddedToCluster(_nodeId, _clusterId);
    }
    
    /**
     * @dev Remove a node from a cluster
     * @param _nodeId Node ID
     * @param _clusterId Cluster ID
     */
    function removeNodeFromCluster(bytes32 _nodeId, uint8 _clusterId) public {
        // Verify node is in the cluster
        require(memberships[_nodeId][_clusterId].joinTime != 0, "Node is not in the cluster");
        require(memberships[_nodeId][_clusterId].isActive, "Node is not active in the cluster");
        
        // Update membership
        memberships[_nodeId][_clusterId].isActive = false;
        
        // Remove node from cluster members
        removeFromClusterMembers(_nodeId, _clusterId);
        
        // Remove cluster from node clusters
        removeFromNodeClusters(_clusterId, _nodeId);
        
        // Update cluster node count
        clusters[_clusterId].nodeCount--;
        
        emit NodeRemovedFromCluster(_nodeId, _clusterId);
    }
    
    /**
     * @dev Helper function to remove node from cluster members
     * @param _nodeId Node ID
     * @param _clusterId Cluster ID
     */
    function removeFromClusterMembers(bytes32 _nodeId, uint8 _clusterId) internal {
        bytes32[] storage members = clusterMembers[_clusterId];
        
        for (uint i = 0; i < members.length; i++) {
            if (members[i] == _nodeId) {
                // Swap with the last element and then remove the last element
                members[i] = members[members.length - 1];
                members.pop();
                break;
            }
        }
    }
    
    /**
     * @dev Helper function to remove cluster from node clusters
     * @param _clusterId Cluster ID
     * @param _nodeId Node ID
     */
    function removeFromNodeClusters(uint8 _clusterId, bytes32 _nodeId) internal {
        uint8[] storage clusters = nodeClusters[_nodeId];
        
        for (uint i = 0; i < clusters.length; i++) {
            if (clusters[i] == _clusterId) {
                // Swap with the last element and then remove the last element
                clusters[i] = clusters[clusters.length - 1];
                clusters.pop();
                break;
            }
        }
    }
    
    /**
     * @dev Update cluster metrics
     * @param _clusterId Cluster ID
     * @param _avgLatency Average latency
     * @param _avgBandwidth Average bandwidth
     * @param _avgSecurityLevel Average security level
     * @param _successfulTransmissions Number of successful transmissions
     * @param _failedTransmissions Number of failed transmissions
     */
    function updateClusterMetrics(
        uint8 _clusterId,
        uint16 _avgLatency,
        uint16 _avgBandwidth,
        uint8 _avgSecurityLevel,
        uint256 _successfulTransmissions,
        uint256 _failedTransmissions
    ) public {
        // Verify cluster exists
        require(clusters[_clusterId].creationTime != 0, "Cluster does not exist");
        
        // Update metrics
        clusterMetrics[_clusterId].avgLatency = _avgLatency;
        clusterMetrics[_clusterId].avgBandwidth = _avgBandwidth;
        clusterMetrics[_clusterId].avgSecurityLevel = _avgSecurityLevel;
        clusterMetrics[_clusterId].successfulTransmissions = _successfulTransmissions;
        clusterMetrics[_clusterId].failedTransmissions = _failedTransmissions;
        clusterMetrics[_clusterId].lastUpdated = block.timestamp;
        
        emit ClusterMetricsUpdated(_clusterId, _avgLatency, _avgBandwidth);
    }
    
    /**
     * @dev Recluster nodes from an old cluster to a new cluster
     * @param _oldClusterId Old cluster ID
     * @param _newClusterId New cluster ID
     */
    function reclusterNodes(uint8 _oldClusterId, uint8 _newClusterId) public {
        // Verify both clusters exist
        require(clusters[_oldClusterId].creationTime != 0, "Old cluster does not exist");
        require(clusters[_newClusterId].creationTime != 0, "New cluster does not exist");
        require(clusters[_newClusterId].isActive, "New cluster is not active");
        require(clusters[_newClusterId].validUntil > block.timestamp, "New cluster has expired");
        
        // Get nodes from old cluster
        bytes32[] memory nodesToMove = clusterMembers[_oldClusterId];
        
        // Add nodes to new cluster
        for (uint i = 0; i < nodesToMove.length; i++) {
            bytes32 nodeId = nodesToMove[i];
            
            // Skip if node is already in the new cluster
            if (memberships[nodeId][_newClusterId].joinTime != 0) {
                continue;
            }
            
            // Add node to new cluster
            memberships[nodeId][_newClusterId] = ClusterMembership({
                nodeId: nodeId,
                clusterId: _newClusterId,
                joinTime: block.timestamp,
                isActive: true
            });
            
            // Update node clusters
            nodeClusters[nodeId].push(_newClusterId);
            
            // Update cluster members
            clusterMembers[_newClusterId].push(nodeId);
            
            // Increment new cluster node count
            clusters[_newClusterId].nodeCount++;
            
            emit NodeAddedToCluster(nodeId, _newClusterId);
        }
        
        // Update old cluster status
        clusters[_oldClusterId].isActive = false;
        
        emit ClusterReclustered(_oldClusterId, _newClusterId);
    }
    
    /**
     * @dev Get cluster details
     * @param _clusterId Cluster ID
     * @return Cluster details (clusterId, clusterName, clusterType, creationTime, validUntil, securityLevel, maxLatency, minBandwidth, isActive, nodeCount)
     */
    function getClusterDetails(uint8 _clusterId) public view returns (
        uint8, string memory, uint8, uint256, uint256, uint8, uint16, uint16, bool, uint256
    ) {
        require(clusters[_clusterId].creationTime != 0, "Cluster does not exist");
        
        Cluster memory cluster = clusters[_clusterId];
        
        return (
            cluster.clusterId,
            cluster.clusterName,
            cluster.clusterType,
            cluster.creationTime,
            cluster.validUntil,
            cluster.securityLevel,
            cluster.maxLatency,
            cluster.minBandwidth,
            cluster.isActive,
            cluster.nodeCount
        );
    }
    
    /**
     * @dev Get cluster metrics
     * @param _clusterId Cluster ID
     * @return Cluster metrics (avgLatency, avgBandwidth, avgSecurityLevel, successfulTransmissions, failedTransmissions, lastUpdated)
     */
    function getClusterMetrics(uint8 _clusterId) public view returns (
        uint16, uint16, uint8, uint256, uint256, uint256
    ) {
        require(clusters[_clusterId].creationTime != 0, "Cluster does not exist");
        
        ClusterMetrics memory metrics = clusterMetrics[_clusterId];
        
        return (
            metrics.avgLatency,
            metrics.avgBandwidth,
            metrics.avgSecurityLevel,
            metrics.successfulTransmissions,
            metrics.failedTransmissions,
            metrics.lastUpdated
        );
    }
    
    /**
     * @dev Get cluster members
     * @param _clusterId Cluster ID
     * @return Array of node IDs in the cluster
     */
    function getClusterMembers(uint8 _clusterId) public view returns (bytes32[] memory) {
        require(clusters[_clusterId].creationTime != 0, "Cluster does not exist");
        
        return clusterMembers[_clusterId];
    }
    
    /**
     * @dev Get clusters for a node
     * @param _nodeId Node ID
     * @return Array of cluster IDs the node belongs to
     */
    function getNodeClusters(bytes32 _nodeId) public view returns (uint8[] memory) {
        return nodeClusters[_nodeId];
    }
    
    /**
     * @dev Check if a node is in a cluster
     * @param _nodeId Node ID
     * @param _clusterId Cluster ID
     * @return Whether the node is in the cluster
     */
    function isNodeInCluster(bytes32 _nodeId, uint8 _clusterId) public view returns (bool) {
        return memberships[_nodeId][_clusterId].joinTime != 0 && memberships[_nodeId][_clusterId].isActive;
    }
    
    /**
     * @dev Get count of all clusters
     * @return Number of clusters
     */
    function getClusterCount() public view returns (uint256) {
        return allClusterIds.length;
    }
}