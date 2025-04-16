// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title NIDRegistry
 * @dev Contract for registering and managing Node IDs (NIDs) on the NAP Layer
 */
contract NIDRegistry {
    // Struct to store NID information
    struct NodeID {
        bytes32 primaryId;       // Primary ID hash (static)
        bytes32 secondaryId;     // Secondary ID hash (dynamic)
        uint256 registrationTime;// Time when the node was registered
        bool isActive;           // Whether the node is currently active
        uint8 securityLevel;     // Security level (0-255)
        uint8 clusterID;         // Cluster ID the node belongs to
        string nodeType;         // Type of node (e.g., "VALIDATOR", "RELAY", "EDGE")
    }
    
    // Mapping from NID hash to NodeID struct
    mapping(bytes32 => NodeID) public nodes;
    
    // Array to store all registered NID primary hashes
    bytes32[] public allNodeIds;
    
    // Mapping from cluster ID to array of node IDs in that cluster
    mapping(uint8 => bytes32[]) public clusterNodes;
    
    // Events
    event NodeRegistered(bytes32 indexed primaryId, uint8 clusterID, string nodeType);
    event NodeUpdated(bytes32 indexed primaryId, bytes32 secondaryId);
    event NodeStatusChanged(bytes32 indexed primaryId, bool isActive);
    event NodeSecurityLevelChanged(bytes32 indexed primaryId, uint8 securityLevel);
    event NodeClusterChanged(bytes32 indexed primaryId, uint8 oldClusterID, uint8 newClusterID);
    
    /**
     * @dev Register a new node
     * @param _primaryId Primary ID hash
     * @param _secondaryId Secondary ID hash
     * @param _securityLevel Security level
     * @param _clusterID Cluster ID
     * @param _nodeType Type of node
     */
    function registerNode(
        bytes32 _primaryId,
        bytes32 _secondaryId,
        uint8 _securityLevel,
        uint8 _clusterID,
        string memory _nodeType
    ) public {
        require(nodes[_primaryId].registrationTime == 0, "Node already registered");
        
        nodes[_primaryId] = NodeID({
            primaryId: _primaryId,
            secondaryId: _secondaryId,
            registrationTime: block.timestamp,
            isActive: true,
            securityLevel: _securityLevel,
            clusterID: _clusterID,
            nodeType: _nodeType
        });
        
        allNodeIds.push(_primaryId);
        clusterNodes[_clusterID].push(_primaryId);
        
        emit NodeRegistered(_primaryId, _clusterID, _nodeType);
    }
    
    /**
     * @dev Update node's secondary ID (for dynamic attributes)
     * @param _primaryId Primary ID hash
     * @param _newSecondaryId New secondary ID hash
     */
    function updateSecondaryId(bytes32 _primaryId, bytes32 _newSecondaryId) public {
        require(nodes[_primaryId].registrationTime != 0, "Node not registered");
        
        nodes[_primaryId].secondaryId = _newSecondaryId;
        
        emit NodeUpdated(_primaryId, _newSecondaryId);
    }
    
    /**
     * @dev Change node's active status
     * @param _primaryId Primary ID hash
     * @param _isActive New active status
     */
    function setNodeStatus(bytes32 _primaryId, bool _isActive) public {
        require(nodes[_primaryId].registrationTime != 0, "Node not registered");
        
        nodes[_primaryId].isActive = _isActive;
        
        emit NodeStatusChanged(_primaryId, _isActive);
    }
    
    /**
     * @dev Change node's security level
     * @param _primaryId Primary ID hash
     * @param _securityLevel New security level
     */
    function setSecurityLevel(bytes32 _primaryId, uint8 _securityLevel) public {
        require(nodes[_primaryId].registrationTime != 0, "Node not registered");
        
        nodes[_primaryId].securityLevel = _securityLevel;
        
        emit NodeSecurityLevelChanged(_primaryId, _securityLevel);
    }
    
    /**
     * @dev Move node to different cluster
     * @param _primaryId Primary ID hash
     * @param _newClusterID New cluster ID
     */
    function changeNodeCluster(bytes32 _primaryId, uint8 _newClusterID) public {
        require(nodes[_primaryId].registrationTime != 0, "Node not registered");
        
        uint8 oldClusterID = nodes[_primaryId].clusterID;
        require(oldClusterID != _newClusterID, "Node already in this cluster");
        
        // Remove from old cluster
        removeFromCluster(_primaryId, oldClusterID);
        
        // Add to new cluster
        nodes[_primaryId].clusterID = _newClusterID;
        clusterNodes[_newClusterID].push(_primaryId);
        
        emit NodeClusterChanged(_primaryId, oldClusterID, _newClusterID);
    }
    
    /**
     * @dev Helper function to remove node from cluster
     * @param _primaryId Primary ID hash
     * @param _clusterID Cluster ID
     */
    function removeFromCluster(bytes32 _primaryId, uint8 _clusterID) internal {
        bytes32[] storage nodesInCluster = clusterNodes[_clusterID];
        
        for (uint i = 0; i < nodesInCluster.length; i++) {
            if (nodesInCluster[i] == _primaryId) {
                // Swap with the last element and then remove the last element
                nodesInCluster[i] = nodesInCluster[nodesInCluster.length - 1];
                nodesInCluster.pop();
                break;
            }
        }
    }
    
    /**
     * @dev Get all nodes in a specific cluster
     * @param _clusterID Cluster ID
     * @return Array of node primary IDs in the cluster
     */
    function getNodesInCluster(uint8 _clusterID) public view returns (bytes32[] memory) {
        return clusterNodes[_clusterID];
    }
    
    /**
     * @dev Get count of all registered nodes
     * @return Number of registered nodes
     */
    function getNodeCount() public view returns (uint256) {
        return allNodeIds.length;
    }
    
    /**
     * @dev Get node details
     * @param _primaryId Primary ID hash
     * @return Node details (primaryId, secondaryId, registrationTime, isActive, securityLevel, clusterID, nodeType)
     */
    function getNodeDetails(bytes32 _primaryId) public view returns (
        bytes32, bytes32, uint256, bool, uint8, uint8, string memory
    ) {
        NodeID memory node = nodes[_primaryId];
        require(node.registrationTime != 0, "Node not registered");
        
        return (
            node.primaryId,
            node.secondaryId,
            node.registrationTime,
            node.isActive,
            node.securityLevel,
            node.clusterID,
            node.nodeType
        );
    }
    
    /**
     * @dev Check if a node exists
     * @param _primaryId Primary ID hash
     * @return Whether the node exists
     */
    function nodeExists(bytes32 _primaryId) public view returns (bool) {
        return nodes[_primaryId].registrationTime != 0;
    }
    
    /**
     * @dev Check if a node is active
     * @param _primaryId Primary ID hash
     * @return Whether the node is active
     */
    function isNodeActive(bytes32 _primaryId) public view returns (bool) {
        require(nodes[_primaryId].registrationTime != 0, "Node not registered");
        return nodes[_primaryId].isActive;
    }
}