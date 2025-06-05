// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "./NIDRegistry.sol";
import "./NIASRegistry.sol";
import "./ABATLTranslation.sol";

contract ClusteringContract {
    // Cluster structure
    struct Cluster {
        uint8 clusterId;              
        string clusterName;           
        uint8 clusterType;            
        uint256 creationTime;         
        uint256 validUntil;           
        uint8 securityLevel;          
        uint16 maxLatency;            
        uint16 minBandwidth;          
        bool isActive;                
        uint256 nodeCount;            
    }
    
    // ClusterMembership record
    struct ClusterMembership {
        string nodeId;                
        uint8 clusterId;              
        uint256 joinTime;             
        bool isActive;                
    }
    
    // Metrics structure for cluster
    struct ClusterMetrics {
        uint16 avgLatency;            
        uint16 avgBandwidth;          
        uint8 avgSecurityLevel;       
        uint256 successfulTransmissions; 
        uint256 failedTransmissions;  
        uint256 lastUpdated;          
    }
    
    // Core registries
    NIDRegistry public nidRegistry;
    NIASRegistry public niasRegistry;
    ABATLTranslation public abatlTranslation;
    
    // Cluster storage
    mapping(uint8 => Cluster) public clusters;                    
    mapping(uint8 => string[]) public clusterMembers;             
    mapping(string => uint8[]) public nodeClusters;               
    mapping(string => mapping(uint8 => ClusterMembership)) public memberships; 
    mapping(uint8 => ClusterMetrics) public clusterMetrics;       
    
    // All cluster IDs
    uint8[] public allClusterIds;
    
    // Events
    event ClusterCreated(uint8 indexed clusterId, string clusterName, uint8 clusterType);
    event ClusterUpdated(uint8 indexed clusterId, uint256 validUntil, bool isActive);
    event NodeAddedToCluster(string indexed nodeId, uint8 indexed clusterId);
    event NodeRemovedFromCluster(string indexed nodeId, uint8 indexed clusterId);
    event ClusterMetricsUpdated(uint8 indexed clusterId, uint16 avgLatency, uint16 avgBandwidth);
    event ClusterReclustered(uint8 indexed oldClusterId, uint8 indexed newClusterId);
    
    constructor(
        address _nidRegistryAddress,
        address _niasRegistryAddress,
        address _abatlTranslationAddress
    ) {
        nidRegistry = NIDRegistry(_nidRegistryAddress);
        niasRegistry = NIASRegistry(_niasRegistryAddress);
        abatlTranslation = ABATLTranslation(_abatlTranslationAddress);
    }
    
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
    
    function addNodeToCluster(string memory _nodeId, uint8 _clusterId) public {
      
        // Add node to cluster membership
        memberships[_nodeId][_clusterId] = ClusterMembership({
            nodeId: _nodeId,
            clusterId: _clusterId,
            joinTime: block.timestamp,
            isActive: true
        });
        
        // Verify node is not already in cluster members
        bool nodeExists = false;
        for (uint i = 0; i < clusterMembers[_clusterId].length; i++) {
            if (keccak256(bytes(clusterMembers[_clusterId][i])) == keccak256(bytes(_nodeId))) {
                nodeExists = true;
                break;
            }
        }
        require(!nodeExists, "Node already in cluster members");
        
        // Add to cluster members
        clusterMembers[_clusterId].push(_nodeId);
        
        // Add cluster to node's clusters
        nodeClusters[_nodeId].push(_clusterId);
        
        // Update cluster node count
        clusters[_clusterId].nodeCount++;
        
        emit NodeAddedToCluster(_nodeId, _clusterId);
    }
    
    function removeNodeFromCluster(string memory _nodeId, uint8 _clusterId) public {
        // Verify node is in the cluster
        require(memberships[_nodeId][_clusterId].joinTime != 0, "Node is not in the cluster");
        require(memberships[_nodeId][_clusterId].isActive, "Node is not active in the cluster");
        
        // Update membership
        memberships[_nodeId][_clusterId].isActive = false;
        
        // Remove node from cluster members
        _removeFromClusterMembers(_nodeId, _clusterId);
        
        // Remove cluster from node clusters
        _removeFromNodeClusters(_clusterId, _nodeId);
        
        // Update cluster node count
        clusters[_clusterId].nodeCount--;
        
        emit NodeRemovedFromCluster(_nodeId, _clusterId);
    }
    
    function _removeFromClusterMembers(string memory _nodeId, uint8 _clusterId) internal {
        string[] storage members = clusterMembers[_clusterId];
        
        for (uint i = 0; i < members.length; i++) {
            if (keccak256(bytes(members[i])) == keccak256(bytes(_nodeId))) {
                // Swap with the last element and then remove the last element
                members[i] = members[members.length - 1];
                members.pop();
                break;
            }
        }
    }
    
    function _removeFromNodeClusters(uint8 _clusterId, string memory _nodeId) internal {
        uint8[] storage nodeClusterList = nodeClusters[_nodeId];
        
        for (uint i = 0; i < nodeClusterList.length; i++) {
            if (nodeClusterList[i] == _clusterId) {
                // Swap with the last element and then remove the last element
                nodeClusterList[i] = nodeClusterList[nodeClusterList.length - 1];
                nodeClusterList.pop();
                break;
            }
        }
    }
    
    function reclusterNodes(uint8 _oldClusterId, uint8 _newClusterId) public {
        // Verify both clusters exist
        require(clusters[_oldClusterId].creationTime != 0, "Old cluster does not exist");
        require(clusters[_newClusterId].creationTime != 0, "New cluster does not exist");
        require(clusters[_newClusterId].isActive, "New cluster is not active");
        require(clusters[_newClusterId].validUntil > block.timestamp, "New cluster has expired");
        
        // Get nodes from old cluster
        string[] memory nodesToMove = clusterMembers[_oldClusterId];
        
        // Add nodes to new cluster
        for (uint i = 0; i < nodesToMove.length; i++) {
            string memory nodeId = nodesToMove[i];
            
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
    
    function getClusterMembers(uint8 _clusterId) public view returns (string[] memory) {
        require(clusters[_clusterId].creationTime != 0, "Cluster does not exist");
        return clusterMembers[_clusterId];
    }
    
    function getNodeClusters(string memory _nodeId) public view returns (uint8[] memory) {
        return nodeClusters[_nodeId];
    }
    
    function isNodeInCluster(string memory _nodeId, uint8 _clusterId) public view returns (bool) {
        return memberships[_nodeId][_clusterId].joinTime != 0 && 
               memberships[_nodeId][_clusterId].isActive;
    }
    
    function getClusterCount() public view returns (uint256) {
        return allClusterIds.length;
    }
}