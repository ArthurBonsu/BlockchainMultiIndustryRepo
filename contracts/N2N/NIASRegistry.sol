// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title NIASRegistry
 * @dev Contract for registering and managing Node Identifiable Autonomous Systems (NIAS) on the B-BGP Layer
 */
contract NIASRegistry {
    // Struct to store NIAS information
    struct NodeAS {
        bytes32 primaryId;        // Primary ID hash (static)
        bytes32 secondaryId;      // Secondary ID hash (dynamic)
        uint256 registrationTime; // Time when the NIAS was registered
        bool isActive;            // Whether the NIAS is currently active
        uint8 securityLevel;      // Security level (0-255)
        uint16 routingWeight;     // Routing weight for path selection
        uint16 loadBalancingFactor; // Load balancing factor
        uint8 clusterID;          // Cluster ID the NIAS belongs to
        string niasType;          // Type of NIAS (e.g., "EDGE", "RELAY", "VALIDATOR")
    }
    
    // Mapping from NIAS hash to NodeAS struct
    mapping(bytes32 => NodeAS) public autonomousSystems;
    
    // Array to store all registered NIAS primary hashes
    bytes32[] public allNIASIds;
    
    // Mapping from cluster ID to array of NIAS IDs in that cluster
    mapping(uint8 => bytes32[]) public clusterNIAS;
    
    // Events
    event NIASRegistered(bytes32 indexed primaryId, uint8 clusterID, string niasType);
    event NIASUpdated(bytes32 indexed primaryId, bytes32 secondaryId);
    event NIASStatusChanged(bytes32 indexed primaryId, bool isActive);
    event NIASSecurityLevelChanged(bytes32 indexed primaryId, uint8 securityLevel);
    event NIASRoutingWeightChanged(bytes32 indexed primaryId, uint16 routingWeight);
    event NIASLoadBalancingFactorChanged(bytes32 indexed primaryId, uint16 loadBalancingFactor);
    event NIASClusterChanged(bytes32 indexed primaryId, uint8 oldClusterID, uint8 newClusterID);
    
    /**
     * @dev Register a new NIAS
     * @param _primaryId Primary ID hash
     * @param _secondaryId Secondary ID hash
     * @param _securityLevel Security level
     * @param _routingWeight Routing weight
     * @param _loadBalancingFactor Load balancing factor
     * @param _clusterID Cluster ID
     * @param _niasType Type of NIAS
     */
    function registerNIAS(
        bytes32 _primaryId,
        bytes32 _secondaryId,
        uint8 _securityLevel,
        uint16 _routingWeight,
        uint16 _loadBalancingFactor,
        uint8 _clusterID,
        string memory _niasType
    ) public {
        require(autonomousSystems[_primaryId].registrationTime == 0, "NIAS already registered");
        
        autonomousSystems[_primaryId] = NodeAS({
            primaryId: _primaryId,
            secondaryId: _secondaryId,
            registrationTime: block.timestamp,
            isActive: true,
            securityLevel: _securityLevel,
            routingWeight: _routingWeight,
            loadBalancingFactor: _loadBalancingFactor,
            clusterID: _clusterID,
            niasType: _niasType
        });
        
        allNIASIds.push(_primaryId);
        clusterNIAS[_clusterID].push(_primaryId);
        
        emit NIASRegistered(_primaryId, _clusterID, _niasType);
    }
    
    /**
     * @dev Update NIAS's secondary ID (for dynamic attributes)
     * @param _primaryId Primary ID hash
     * @param _newSecondaryId New secondary ID hash
     */
    function updateSecondaryId(bytes32 _primaryId, bytes32 _newSecondaryId) public {
        require(autonomousSystems[_primaryId].registrationTime != 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].secondaryId = _newSecondaryId;
        
        emit NIASUpdated(_primaryId, _newSecondaryId);
    }
    
    /**
     * @dev Change NIAS's active status
     * @param _primaryId Primary ID hash
     * @param _isActive New active status
     */
    function setNIASStatus(bytes32 _primaryId, bool _isActive) public {
        require(autonomousSystems[_primaryId].registrationTime != 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].isActive = _isActive;
        
        emit NIASStatusChanged(_primaryId, _isActive);
    }
    
    /**
     * @dev Change NIAS's security level
     * @param _primaryId Primary ID hash
     * @param _securityLevel New security level
     */
    function setSecurityLevel(bytes32 _primaryId, uint8 _securityLevel) public {
        require(autonomousSystems[_primaryId].registrationTime != 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].securityLevel = _securityLevel;
        
        emit NIASSecurityLevelChanged(_primaryId, _securityLevel);
    }
    
    /**
     * @dev Change NIAS's routing weight
     * @param _primaryId Primary ID hash
     * @param _routingWeight New routing weight
     */
    function setRoutingWeight(bytes32 _primaryId, uint16 _routingWeight) public {
        require(autonomousSystems[_primaryId].registrationTime != 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].routingWeight = _routingWeight;
        
        emit NIASRoutingWeightChanged(_primaryId, _routingWeight);
    }
    
    /**
     * @dev Change NIAS's load balancing factor
     * @param _primaryId Primary ID hash
     * @param _loadBalancingFactor New load balancing factor
     */
    function setLoadBalancingFactor(bytes32 _primaryId, uint16 _loadBalancingFactor) public {
        require(autonomousSystems[_primaryId].registrationTime != 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].loadBalancingFactor = _loadBalancingFactor;
        
        emit NIASLoadBalancingFactorChanged(_primaryId, _loadBalancingFactor);
    }
    
    /**
     * @dev Move NIAS to different cluster
     * @param _primaryId Primary ID hash
     * @param _newClusterID New cluster ID
     */
    function changeNIASCluster(bytes32 _primaryId, uint8 _newClusterID) public {
        require(autonomousSystems[_primaryId].registrationTime != 0, "NIAS not registered");
        
        uint8 oldClusterID = autonomousSystems[_primaryId].clusterID;
        require(oldClusterID != _newClusterID, "NIAS already in this cluster");
        
        // Remove from old cluster
        removeFromCluster(_primaryId, oldClusterID);
        
        // Add to new cluster
        autonomousSystems[_primaryId].clusterID = _newClusterID;
        clusterNIAS[_newClusterID].push(_primaryId);
        
        emit NIASClusterChanged(_primaryId, oldClusterID, _newClusterID);
    }
    
    /**
     * @dev Helper function to remove NIAS from cluster
     * @param _primaryId Primary ID hash
     * @param _clusterID Cluster ID
     */
    function removeFromCluster(bytes32 _primaryId, uint8 _clusterID) internal {
        bytes32[] storage niasInCluster = clusterNIAS[_clusterID];
        
        for (uint i = 0; i < niasInCluster.length; i++) {
            if (niasInCluster[i] == _primaryId) {
                // Swap with the last element and then remove the last element
                niasInCluster[i] = niasInCluster[niasInCluster.length - 1];
                niasInCluster.pop();
                break;
            }
        }
    }
    
    /**
     * @dev Get all NIAS in a specific cluster
     * @param _clusterID Cluster ID
     * @return Array of NIAS primary IDs in the cluster
     */
    function getNIASInCluster(uint8 _clusterID) public view returns (bytes32[] memory) {
        return clusterNIAS[_clusterID];
    }
    
    /**
     * @dev Get count of all registered NIAS
     * @return Number of registered NIAS
     */
    function getNIASCount() public view returns (uint256) {
        return allNIASIds.length;
    }
    
    /**
     * @dev Get NIAS details
     * @param _primaryId Primary ID hash
     * @return NIAS details (primaryId, secondaryId, registrationTime, isActive, securityLevel, routingWeight, loadBalancingFactor, clusterID, niasType)
     */
    function getNIASDetails(bytes32 _primaryId) public view returns (
        bytes32, bytes32, uint256, bool, uint8, uint16, uint16, uint8, string memory
    ) {
        NodeAS memory nias = autonomousSystems[_primaryId];
        require(nias.registrationTime != 0, "NIAS not registered");
        
        return (
            nias.primaryId,
            nias.secondaryId,
            nias.registrationTime,
            nias.isActive,
            nias.securityLevel,
            nias.routingWeight,
            nias.loadBalancingFactor,
            nias.clusterID,
            nias.niasType
        );
    }
    
    /**
     * @dev Check if a NIAS exists
     * @param _primaryId Primary ID hash
     * @return Whether the NIAS exists
     */
    function niasExists(bytes32 _primaryId) public view returns (bool) {
        return autonomousSystems[_primaryId].registrationTime != 0;
    }
    
    /**
     * @dev Check if a NIAS is active
     * @param _primaryId Primary ID hash
     * @return Whether the NIAS is active
     */
    function isNIASActive(bytes32 _primaryId) public view returns (bool) {
        require(autonomousSystems[_primaryId].registrationTime != 0, "NIAS not registered");
        return autonomousSystems[_primaryId].isActive;
    }
}