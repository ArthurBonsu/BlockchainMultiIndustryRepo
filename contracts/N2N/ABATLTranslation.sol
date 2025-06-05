// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "./NIDRegistry.sol";
import "./NIASRegistry.sol";

/**
 * @title ABATLTranslation
 * @dev Contract for facilitating translation between NAP layer and B-BGP layer
 */
contract ABATLTranslation {
    // ABATL Record structure
    struct ABATLRecord {
        string abatlId;             // ABATL ID
        string nidId;               // NID ID
        string niasId;              // NIAS ID
        uint8 clusterID;             // Cluster ID
        uint8 abatlType;             // Type of ABATL (0-255)
        uint256 timestamp;           // Registration timestamp
        uint8 senderType;            // 0 = NID_SENDER, 1 = NIAS_SENDER
        bool isActive;               // Whether the ABATL is active
    }
    
    // Secondary attributes for ABATL (dynamic attributes)
    struct ABATLSecondaryAttributes {
        string abatlId;             // ABATL ID
        uint8 qosLevel;              // QoS level (0-100)
        uint16 latency;              // Latency in ms
        uint16 bandwidth;            // Bandwidth in Mbps
        uint8 securityLevel;         // Security level (0-255)
        uint256 lastUpdated;         // Last time attributes were updated
    }
    
    // Mapping from ABATL ID to ABATLRecord
    mapping(string => ABATLRecord) public abatlRecords;
    
    // Mapping from ABATL ID to ABATLSecondaryAttributes
    mapping(string => ABATLSecondaryAttributes) public abatlSecondaryAttributes;
    
    // Mapping from NID to ABATL IDs
    mapping(string => string[]) public nidToAbatl;
    
    // Mapping from NIAS to ABATL IDs
    mapping(string => string[]) public niasToAbatl;
    
    // Array to store all ABATL IDs
    string[] public allAbatlIds;
    
    // NID and NIAS registry contracts
    NIDRegistry public nidRegistry;
    NIASRegistry public niasRegistry;
    
    // Events
    event ABATLRegistered(string indexed abatlId, string nidId, string niasId, uint8 abatlType);
    event ABATLSecondaryAttributesUpdated(string indexed abatlId, uint8 qosLevel, uint16 latency, uint16 bandwidth, uint8 securityLevel);
    event ABATLStatusChanged(string indexed abatlId, bool isActive);
    
    /**
     * @dev Constructor
     * @param _nidRegistryAddress Address of NIDRegistry contract
     * @param _niasRegistryAddress Address of NIASRegistry contract
     */
    constructor(address _nidRegistryAddress, address _niasRegistryAddress) {
        nidRegistry = NIDRegistry(_nidRegistryAddress);
        niasRegistry = NIASRegistry(_niasRegistryAddress);
    }
    
    /**
     * @dev Register a new ABATL record
     * @param _abatlId ABATL ID
     * @param _nidId NID ID
     * @param _niasId NIAS ID
     * @param _clusterID Cluster ID
     * @param _abatlType Type of ABATL
     * @param _senderType Sender type (0 = NID_SENDER, 1 = NIAS_SENDER)
     */
    function registerABATL(
        string memory _abatlId,
        string memory _nidId,
        string memory _niasId,
        uint8 _clusterID,
        uint8 _abatlType,
        uint8 _senderType
    ) public {
        /*
        // Verify that NID and NIAS exist and are active
        require(nidRegistry.nodeExists(_nidId), "NID does not exist");
        require(nidRegistry.isNodeActive(_nidId), "NID is not active");
        require(niasRegistry.niasExists(_niasId), "NIAS does not exist");
        require(niasRegistry.isNIASActive(_niasId), "NIAS is not active");
        require(_senderType <= 1, "Invalid sender type");
       
        // Verify ABATL doesn't already exist
        require(bytes(abatlRecords[_abatlId].abatlId).length == 0, "ABATL already registered");
         */
        // Create the ABATL record
        abatlRecords[_abatlId] = ABATLRecord({
            abatlId: _abatlId,
            nidId: _nidId,
            niasId: _niasId,
            clusterID: _clusterID,
            abatlType: _abatlType,
            timestamp: block.timestamp,
            senderType: _senderType,
            isActive: true
        });
        
        // Add to mappings
        nidToAbatl[_nidId].push(_abatlId);
        niasToAbatl[_niasId].push(_abatlId);
        allAbatlIds.push(_abatlId);
        
        emit ABATLRegistered(_abatlId, _nidId, _niasId, _abatlType);
    }
    
    /**
     * @dev Update ABATL secondary attributes
     * @param _abatlId ABATL ID
     * @param _qosLevel QoS level
     * @param _latency Latency in ms
     * @param _bandwidth Bandwidth in Mbps
     * @param _securityLevel Security level
     */
    function updateABATLSecondaryAttributes(
        string memory _abatlId,
        uint8 _qosLevel,
        uint16 _latency,
        uint16 _bandwidth,
        uint8 _securityLevel
    ) public {
        require(bytes(abatlRecords[_abatlId].abatlId).length > 0, "ABATL does not exist");
        require(abatlRecords[_abatlId].isActive, "ABATL is not active");
        
        abatlSecondaryAttributes[_abatlId] = ABATLSecondaryAttributes({
            abatlId: _abatlId,
            qosLevel: _qosLevel,
            latency: _latency,
            bandwidth: _bandwidth,
            securityLevel: _securityLevel,
            lastUpdated: block.timestamp
        });
        
        emit ABATLSecondaryAttributesUpdated(_abatlId, _qosLevel, _latency, _bandwidth, _securityLevel);
    }
    
    /**
     * @dev Set ABATL active status
     * @param _abatlId ABATL ID
     * @param _isActive New active status
     */
    function setABATLStatus(string memory _abatlId, bool _isActive) public {
        require(bytes(abatlRecords[_abatlId].abatlId).length > 0, "ABATL does not exist");
        
        abatlRecords[_abatlId].isActive = _isActive;
        
        emit ABATLStatusChanged(_abatlId, _isActive);
    }
    
    /**
     * @dev Get ABATL primary details
     * @param _abatlId ABATL ID
     * @return ABATL primary details (abatlId, nidId, niasId, clusterID, abatlType, timestamp, senderType, isActive)
     */
    function getABATLPrimaryDetails(string memory _abatlId) public view returns (
        string memory, string memory, string memory, uint8, uint8, uint256, uint8, bool
    ) {
        require(bytes(abatlRecords[_abatlId].abatlId).length > 0, "ABATL does not exist");
        
        ABATLRecord memory record = abatlRecords[_abatlId];
        
        return (
            record.abatlId,
            record.nidId,
            record.niasId,
            record.clusterID,
            record.abatlType,
            record.timestamp,
            record.senderType,
            record.isActive
        );
    }
    
    /**
     * @dev Get ABATL secondary attributes
     * @param _abatlId ABATL ID
     * @return ABATL secondary attributes (abatlId, qosLevel, latency, bandwidth, securityLevel, lastUpdated)
     */
    function getABATLSecondaryAttributes(string memory _abatlId) public view returns (
        string memory, uint8, uint16, uint16, uint8, uint256
    ) {
        require(bytes(abatlRecords[_abatlId].abatlId).length > 0, "ABATL does not exist");
        
        ABATLSecondaryAttributes memory attrs = abatlSecondaryAttributes[_abatlId];
        
        return (
            attrs.abatlId,
            attrs.qosLevel,
            attrs.latency,
            attrs.bandwidth,
            attrs.securityLevel,
            attrs.lastUpdated
        );
    }
    
    /**
     * @dev Get all ABATL IDs for a NID
     * @param _nidId NID ID
     * @return Array of ABATL IDs
     */
    function getABATLsByNID(string memory _nidId) public view returns (string[] memory) {
        return nidToAbatl[_nidId];
    }
    
    /**
     * @dev Get all ABATL IDs for a NIAS
     * @param _niasId NIAS ID
     * @return Array of ABATL IDs
     */
    function getABATLsByNIAS(string memory _niasId) public view returns (string[] memory) {
        return niasToAbatl[_niasId];
    }
    
    /**
     * @dev Get count of all ABATL records
     * @return Number of ABATL records
     */
    function getABATLCount() public view returns (uint256) {
        return allAbatlIds.length;
    }
}