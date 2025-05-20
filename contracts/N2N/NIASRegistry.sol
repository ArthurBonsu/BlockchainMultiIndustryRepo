// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title NIASRegistry
 * @dev Contract for registering and managing Node Identifiable Autonomous Systems (NIAS) on the B-BGP Layer
 */
contract NIASRegistry {
    // Struct to store NIAS information
    struct NodeAS {
        string primaryId;        // Primary ID (static)
        string secondaryId;      // Secondary ID (dynamic)
        uint256 registrationTime; // Time when the NIAS was registered
        bool isActive;            // Whether the NIAS is currently active
        uint8 securityLevel;      // Security level (0-255)
        uint16 routingWeight;     // Routing weight for path selection
        uint16 loadBalancingFactor; // Load balancing factor
        uint8 clusterID;          // Cluster ID the NIAS belongs to
        string niasType;          // Type of NIAS (e.g., "EDGE", "RELAY", "VALIDATOR")
    }
    
    // Mapping from NIAS ID to NodeAS struct
    mapping(string => NodeAS) public autonomousSystems;
    
    // Array to store all registered NIAS primary IDs
    string[] public allNIASIds;
    
    // Mapping from cluster ID to array of NIAS IDs in that cluster
    mapping(uint8 => string[]) public clusterNIAS;
    
    // Events
    event NIASRegistered(string indexed primaryId, uint8 clusterID, string niasType);
    event NIASUpdated(string indexed primaryId, string secondaryId);
    event NIASStatusChanged(string indexed primaryId, bool isActive);
    event NIASSecurityLevelChanged(string indexed primaryId, uint8 securityLevel);
    event NIASRoutingWeightChanged(string indexed primaryId, uint16 routingWeight);
    event NIASLoadBalancingFactorChanged(string indexed primaryId, uint16 loadBalancingFactor);
    event NIASClusterChanged(string indexed primaryId, uint8 oldClusterID, uint8 newClusterID);
    
    /**
     * @dev Register a new NIAS
     * @param _primaryId Primary ID
     * @param _secondaryId Secondary ID
     * @param _securityLevel Security level
     * @param _routingWeight Routing weight
     * @param _loadBalancingFactor Load balancing factor
     * @param _clusterID Cluster ID
     * @param _niasType Type of NIAS
     */
    function registerNIAS(
        string memory _primaryId,
        string memory _secondaryId,
        uint8 _securityLevel,
        uint16 _routingWeight,
        uint16 _loadBalancingFactor,
        uint8 _clusterID,
        string memory _niasType
    ) public {
        require(bytes(autonomousSystems[_primaryId].primaryId).length == 0, "NIAS already registered");
        
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
     * @param _primaryId Primary ID
     * @param _newSecondaryId New secondary ID
     */
    function updateSecondaryId(string memory _primaryId, string memory _newSecondaryId) public {
        require(bytes(autonomousSystems[_primaryId].primaryId).length > 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].secondaryId = _newSecondaryId;
        
        emit NIASUpdated(_primaryId, _newSecondaryId);
    }
    
    /**
     * @dev Change NIAS's active status
     * @param _primaryId Primary ID
     * @param _isActive New active status
     */
    function setNIASStatus(string memory _primaryId, bool _isActive) public {
        require(bytes(autonomousSystems[_primaryId].primaryId).length > 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].isActive = _isActive;
        
        emit NIASStatusChanged(_primaryId, _isActive);
    }
    
    /**
     * @dev Change NIAS's security level
     * @param _primaryId Primary ID
     * @param _securityLevel New security level
     */
    function setSecurityLevel(string memory _primaryId, uint8 _securityLevel) public {
        require(bytes(autonomousSystems[_primaryId].primaryId).length > 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].securityLevel = _securityLevel;
        
        emit NIASSecurityLevelChanged(_primaryId, _securityLevel);
    }
    
    /**
     * @dev Change NIAS's routing weight
     * @param _primaryId Primary ID
     * @param _routingWeight New routing weight
     */
    function setRoutingWeight(string memory _primaryId, uint16 _routingWeight) public {
        require(bytes(autonomousSystems[_primaryId].primaryId).length > 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].routingWeight = _routingWeight;
        
        emit NIASRoutingWeightChanged(_primaryId, _routingWeight);
    }
    
    /**
     * @dev Change NIAS's load balancing factor
     * @param _primaryId Primary ID
     * @param _loadBalancingFactor New load balancing factor
     */
    function setLoadBalancingFactor(string memory _primaryId, uint16 _loadBalancingFactor) public {
        require(bytes(autonomousSystems[_primaryId].primaryId).length > 0, "NIAS not registered");
        
        autonomousSystems[_primaryId].loadBalancingFactor = _loadBalancingFactor;
        
        emit NIASLoadBalancingFactorChanged(_primaryId, _loadBalancingFactor);
    }
    
}
    