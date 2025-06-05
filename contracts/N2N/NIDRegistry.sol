// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title NIDRegistry
 * @dev Contract for registering and managing Node IDs (NIDs) on the NAP Layer
 */
contract NIDRegistry {
    // Struct to store NID information
    struct NodeID {
        string primaryId;       // Primary ID (static)
        string secondaryId;     // Secondary ID (dynamic)
        uint256 registrationTime;// Time when the node was registered
        bool isActive;           // Whether the node is currently active
        uint8 securityLevel;     // Security level (0-255)
        uint8 clusterID;         // Cluster ID the node belongs to
        string nodeType;         // Type of node (e.g., "VALIDATOR", "RELAY", "EDGE")
    }
    
    // Mapping from NID to NodeID struct
    mapping(string => NodeID) public nodes;
    
    // Array to store all registered NID primary IDs
    string[] public allNodeIds;
    
    // Mapping from cluster ID to array of node IDs in that cluster
    mapping(uint8 => string[]) public clusterNodes;
    
    // Events
    event NodeRegistered(string indexed primaryId, uint8 clusterID, string nodeType);
    event NodeUpdated(string indexed primaryId, string secondaryId);
    event NodeStatusChanged(string indexed primaryId, bool isActive);
    event NodeSecurityLevelChanged(string indexed primaryId, uint8 securityLevel);
    event NodeClusterChanged(string indexed primaryId, uint8 oldClusterID, uint8 newClusterID);
    
    /**
     * @dev Register a new node
     * @param _primaryId Primary ID
     * @param _secondaryId Secondary ID
     * @param _securityLevel Security level
     * @param _clusterID Cluster ID
     * @param _nodeType Type of node
     */
    function registerNode(
        string memory _primaryId,
        string memory _secondaryId,
        uint8 _securityLevel,
        uint8 _clusterID,
        string memory _nodeType
    ) public {
        require(bytes(nodes[_primaryId].primaryId).length == 0, "Node already registered");
        
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
     * @param _primaryId Primary ID
     * @param _newSecondaryId New secondary ID
     */
    function updateSecondaryId(string memory _primaryId, string memory _newSecondaryId) public {
        require(bytes(nodes[_primaryId].primaryId).length > 0, "Node not registered");
        
        nodes[_primaryId].secondaryId = _newSecondaryId;
        
        emit NodeUpdated(_primaryId, _newSecondaryId);
    }
    
    /**
     * @dev Change node's active status
     * @param _primaryId Primary ID
     * @param _isActive New active status
     */
    function setNodeStatus(string memory _primaryId, bool _isActive) public {
        require(bytes(nodes[_primaryId].primaryId).length > 0, "Node not registered");
        
        nodes[_primaryId].isActive = _isActive;
        
        emit NodeStatusChanged(_primaryId, _isActive);
    }
    
    /**
     * @dev Change node's security level
     * @param _primaryId Primary ID
     * @param _securityLevel New security level
     */
    function setSecurityLevel(string memory _primaryId, uint8 _securityLevel) public {
        require(bytes(nodes[_primaryId].primaryId).length > 0, "Node not registered");
        
        nodes[_primaryId].securityLevel = _securityLevel;
        
        emit NodeSecurityLevelChanged(_primaryId, _securityLevel);
    }
    
    /**
     * @dev Move node to different cluster
     * @param _primaryId Primary ID
     * @param _newClusterID New cluster ID
     */
    function changeNodeCluster(string memory _primaryId, uint8 _newClusterID) public {
        require(bytes(nodes[_primaryId].primaryId).length > 0, "Node not registered");
        
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
     * @param _primaryId Primary ID
     * @param _clusterID Cluster ID
     */
    function removeFromCluster(string memory _primaryId, uint8 _clusterID) internal {
        string[] storage nodesInCluster = clusterNodes[_clusterID];
        
        for (uint i = 0; i < nodesInCluster.length; i++) {
            if (keccak256(bytes(nodesInCluster[i])) == keccak256(bytes(_primaryId))) {
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
    function getNodesInCluster(uint8 _clusterID) public view returns (string[] memory) {
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
     * @param _primaryId Primary ID
     * @return Node details (primaryId, secondaryId, registrationTime, isActive, securityLevel, clusterID, nodeType)
     */
    function getNodeDetails(string memory _primaryId) public view returns (
        string memory, string memory, uint256, bool, uint8, uint8, string memory
    ) {
        NodeID memory node = nodes[_primaryId];
        require(bytes(node.primaryId).length > 0, "Node not registered");
        
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
     * @param _primaryId Primary ID
     * @return Whether the node exists
     */
    function nodeExists(string memory _primaryId) public view returns (bool) {
        return bytes(nodes[_primaryId].primaryId).length > 0;
    }
    
    /**
     * @dev Check if a node is active
     * @param _primaryId Primary ID
     * @return Whether the node is active
     */
    function isNodeActive(string memory _primaryId) public view returns (bool) {
        require(bytes(nodes[_primaryId].primaryId).length > 0, "Node not registered");
        return nodes[_primaryId].isActive;
    }
}