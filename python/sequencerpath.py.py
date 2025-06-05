/**
 * Enhanced Sequencer Precomputation Algorithm
 * 
 * This improved algorithm enhances the path calculation with:
 * - Dynamic weight adjustment based on traffic type
 * - Historical performance tracking
 * - Predictive congestion modeling
 * - Multi-objective optimization
 */

class EnhancedSequencer {
  constructor() {
    // Initialize system parameters
    this.nodes = new Map(); // NID -> NodeData mapping
    this.historicalPerformance = new Map(); // Connection -> performance history
    this.trafficPatterns = new Map(); // Traffic patterns to optimize for
    this.blockchainInterface = null; // Interface to read/write to blockchain
  }

  /**
   * Calculate optimal path from source NID to destination NIAS
   * @param {string} sourceNID - Source Node ID
   * @param {string} destNIAS - Destination NIAS ID
   * @param {object} trafficRequirements - Requirements for this transmission
   * @returns {object} - Optimal path sequence and metadata
   */
  computeOptimalPath(sourceNID, destNIAS, trafficRequirements) {
    // Validate nodes exist and are reachable
    if (!this.validateEndpoints(sourceNID, destNIAS)) {
      throw new Error(`Invalid endpoints: ${sourceNID} -> ${destNIAS}`);
    }

    // Get weights based on traffic requirements (VoIP, video, general data, etc.)
    const weights = this.dynamicWeightSelection(trafficRequirements);
    
    // Get all active nodes from blockchain
    const activeNodes = this.getActiveNodesFromBlockchain();
    
    // Build connectivity graph with weighted edges
    const graph = this.buildWeightedGraph(activeNodes, weights, trafficRequirements);
    
    // Calculate the optimal path using enhanced algorithm
    const paths = this.multiObjectivePathFinding(graph, sourceNID, destNIAS, weights);
    
    // Select primary and backup paths
    const { primaryPath, backupPaths } = this.selectPrimaryAndBackupPaths(paths);
    
    // Prepare path sequence for blockchain recording
    const pathSequence = this.preparePathSequence(primaryPath, backupPaths);
    
    // Record path to blockchain for validation
    this.recordPathToBlockchain(pathSequence);
    
    return pathSequence;
  }

  /**
   * Dynamically select weights based on traffic requirements
   * @param {object} trafficRequirements - Type of traffic and QoS needs
   * @returns {object} - Weights for different factors
   */
  dynamicWeightSelection(trafficRequirements) {
    const weights = {
      latency: 0.5,
      bandwidth: 0.2,
      security: 0.2,
      reliability: 0.1
    };
    
    // Adjust weights based on traffic type
    switch (trafficRequirements.type) {
      case 'VoIP':
        weights.latency = 0.7;
        weights.bandwidth = 0.1;
        weights.security = 0.1;
        weights.reliability = 0.1;
        break;
      case 'Video':
        weights.latency = 0.3;
        weights.bandwidth = 0.5;
        weights.security = 0.1;
        weights.reliability = 0.1;
        break;
      case 'CriticalData':
        weights.latency = 0.2;
        weights.bandwidth = 0.1;
        weights.security = 0.5;
        weights.reliability = 0.2;
        break;
      // Add more traffic types as needed
    }
    
    return weights;
  }

  /**
   * Build a weighted graph representation of the network
   * @param {Array} activeNodes - Array of active NIDs
   * @param {object} weights - Weights for different factors
   * @param {object} trafficRequirements - Traffic requirements
   * @returns {object} - Graph with weighted edges
   */
  buildWeightedGraph(activeNodes, weights, trafficRequirements) {
    const graph = {};
    
    // Initialize graph nodes
    activeNodes.forEach(node => {
      graph[node.id] = { edges: [] };
    });
    
    // Build edges between nodes
    activeNodes.forEach(nodeA => {
      activeNodes.forEach(nodeB => {
        if (nodeA.id !== nodeB.id && this.nodesCanConnect(nodeA, nodeB)) {
          // Get real-time metrics for this connection
          const metrics = this.getConnectionMetrics(nodeA.id, nodeB.id);
          
          // Get historical performance data
          const history = this.getHistoricalPerformance(nodeA.id, nodeB.id);
          
          // Predict future congestion
          const predictedCongestion = this.predictCongestion(nodeA.id, nodeB.id, history);
          
          // Calculate edge cost based on multiple factors
          const cost = this.calculateEdgeCost(metrics, history, predictedCongestion, weights, trafficRequirements);
          
          // Add edge to graph
          graph[nodeA.id].edges.push({
            target: nodeB.id,
            cost: cost,
            metrics: metrics,
            predictedCongestion: predictedCongestion
          });
        }
      });
    });
    
    return graph;
  }

  /**
   * Calculate the cost of an edge based on multiple factors
   * @param {object} metrics - Current connection metrics
   * @param {object} history - Historical performance
   * @param {number} predictedCongestion - Predicted congestion level
   * @param {object} weights - Weights for different factors
   * @param {object} trafficRequirements - Traffic requirements
   * @returns {number} - Weighted cost of the edge
   */
  calculateEdgeCost(metrics, history, predictedCongestion, weights, trafficRequirements) {
    // Normalize metrics to 0-1 range
    const normalizedLatency = metrics.latency / 100; // Assuming 100ms is worst acceptable
    const normalizedBandwidth = 1 - (metrics.availableBandwidth / metrics.maxBandwidth);
    const normalizedSecurity = 1 - metrics.securityScore; // Higher security score is better
    const normalizedReliability = 1 - metrics.reliability; // Higher reliability is better
    
    // Apply congestion prediction (0-1 range)
    const congestionFactor = predictedCongestion * 0.5; // Weight of prediction
    
    // Historical performance factor (0-1 range)
    const historyFactor = this.calculateHistoryFactor(history);
    
    // Calculate weighted cost
    let cost = 
      weights.latency * normalizedLatency +
      weights.bandwidth * normalizedBandwidth +
      weights.security * normalizedSecurity +
      weights.reliability * normalizedReliability;
    
    // Add congestion prediction and history factors
    cost = cost * (1 + congestionFactor) * (1 + historyFactor);
    
    // Add traffic-specific adjustments
    if (trafficRequirements.type === 'VoIP' && normalizedLatency > 0.5) {
      // Penalize high latency routes for VoIP traffic
      cost *= 2;
    }
    
    return cost;
  }

  /**
   * Multi-objective path finding algorithm
   * Enhanced version of A* that considers multiple optimization objectives
   * @param {object} graph - Network graph
   * @param {string} start - Start node ID
   * @param {string} goal - Goal node ID
   * @param {object} weights - Weights for different factors
   * @returns {Array} - Multiple candidate paths
   */
  multiObjectivePathFinding(graph, start, goal, weights) {
    // Initialize data structures
    const openSet = new PriorityQueue();
    const cameFrom = {};
    const gScore = {};
    const fScore = {};
    const evaluatedPaths = [];
    
    // Initialize starting node
    gScore[start] = 0;
    fScore[start] = this.heuristic(start, goal, weights);
    openSet.enqueue(start, fScore[start]);
    
    while (!openSet.isEmpty()) {
      const current = openSet.dequeue().element;
      
      // If we've reached the goal, reconstruct and return the path
      if (current === goal) {
        const path = this.reconstructPath(cameFrom, current);
        evaluatedPaths.push({
          path: path,
          score: gScore[current]
        });
        
        // If we have enough paths, return them
        if (evaluatedPaths.length >= 3) {
          return evaluatedPaths;
        }
        // Continue searching for alternative paths
        continue;
      }
      
      // Explore neighbors
      graph[current].edges.forEach(edge => {
        const neighbor = edge.target;
        const tentativeGScore = gScore[current] + edge.cost;
        
        if (!(neighbor in gScore) || tentativeGScore < gScore[neighbor]) {
          // This path is better than any previous one
          cameFrom[neighbor] = current;
          gScore[neighbor] = tentativeGScore;
          fScore[neighbor] = gScore[neighbor] + this.heuristic(neighbor, goal, weights);
          
          if (!openSet.contains(neighbor)) {
            openSet.enqueue(neighbor, fScore[neighbor]);
          }
        }
      });
    }
    
    // If no path was found, return empty array
    return evaluatedPaths;
  }

  /**
   * Heuristic function for A* algorithm
   * @param {string} node - Current node ID
   * @param {string} goal - Goal node ID
   * @param {object} weights - Weights for different factors
   * @returns {number} - Heuristic value
   */
  heuristic(node, goal, weights) {
    // Get node data
    const nodeData = this.nodes.get(node);
    const goalData = this.nodes.get(goal);
    
    if (!nodeData || !goalData) {
      return 0;
    }
    
    // Calculate geographical distance using Haversine formula
    const distance = this.haversineDistance(
      nodeData.latitude, nodeData.longitude,
      goalData.latitude, goalData.longitude
    );
    
    // Normalize distance (assuming max distance is 20,000 km)
    const normalizedDistance = distance / 20000;
    
    // Return weighted heuristic
    return normalizedDistance * weights.latency;
  }

  /**
   * Calculate geographical distance using Haversine formula
   * @param {number} lat1 - Latitude of first point
   * @param {number} lon1 - Longitude of first point
   * @param {number} lat2 - Latitude of second point
   * @param {number} lon2 - Longitude of second point
   * @returns {number} - Distance in kilometers
   */
  haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth radius in kilometers
    const dLat = this.deg2rad(lat2 - lat1);
    const dLon = this.deg2rad(lon2 - lon1);
    const a = 
      Math.sin(dLat/2) * Math.sin(dLat/2) +
      Math.cos(this.deg2rad(lat1)) * Math.cos(this.deg2rad(lat2)) * 
      Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  }

  deg2rad(deg) {
    return deg * (Math.PI/180);
  }

  /**
   * Predict congestion based on historical data and current trends
   * @param {string} nodeA - First node ID
   * @param {string} nodeB - Second node ID
   * @param {object} history - Historical performance data
   * @returns {number} - Predicted congestion level (0-1)
   */
  predictCongestion(nodeA, nodeB, history) {
    if (!history || !history.congestionSamples || history.congestionSamples.length === 0) {
      return 0;
    }
    
    // Get current time and day of week
    const now = new Date();
    const hour = now.getHours();
    const dayOfWeek = now.getDay();
    
    // Filter historical samples for similar time and day
    const relevantSamples = history.congestionSamples.filter(sample => {
      const sampleTime = new Date(sample.timestamp);
      return Math.abs(sampleTime.getHours() - hour) <= 1 && 
             (sampleTime.getDay() === dayOfWeek);
    });
    
    if (relevantSamples.length === 0) {
      return 0;
    }
    
    // Calculate average congestion for similar times
    const avgCongestion = relevantSamples.reduce((sum, sample) => sum + sample.congestion, 0) / relevantSamples.length;
    
    // Current trend (increasing/decreasing congestion)
    const recentSamples = history.congestionSamples.slice(-5);
    const trend = this.calculateCongestionTrend(recentSamples);
    
    // Combine historical average with current trend
    return Math.min(1, Math.max(0, avgCongestion + trend));
  }

  /**
   * Calculate congestion trend from recent samples
   * @param {Array} samples - Recent congestion samples
   * @returns {number} - Trend value (-0.2 to 0.2)
   */
  calculateCongestionTrend(samples) {
    if (samples.length < 2) {
      return 0;
    }
    
    // Calculate linear regression slope
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    const n = samples.length;
    
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += samples[i].congestion;
      sumXY += i * samples[i].congestion;
      sumXX += i * i;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    
    // Normalize slope to -0.2 to 0.2 range
    return Math.max(-0.2, Math.min(0.2, slope));
  }

  /**
   * Select primary and backup paths from candidate paths
   * @param {Array} candidatePaths - Array of candidate paths
   * @returns {object} - Selected primary and backup paths
   */
  selectPrimaryAndBackupPaths(candidatePaths) {
    if (candidatePaths.length === 0) {
      throw new Error("No valid paths found");
    }
    
    // Sort paths by score (lower is better)
    candidatePaths.sort((a, b) => a.score - b.score);
    
    // Select primary path (best score)
    const primaryPath = candidatePaths[0].path;
    
    // Select backup paths (next best, prioritizing path diversity)
    const backupPaths = [];
    
    for (let i = 1; i < candidatePaths.length && backupPaths.length < 2; i++) {
      const path = candidatePaths[i].path;
      
      // Check path diversity (how different is this path from the primary)
      const diversity = this.calculatePathDiversity(primaryPath, path);
      
      // Only select paths with sufficient diversity
      if (diversity > 0.6) {
        backupPaths.push(path);
      }
    }
    
    return { primaryPath, backupPaths };
  }

  /**
   * Calculate diversity between two paths (0-1, higher means more diverse)
   * @param {Array} path1 - First path
   * @param {Array} path2 - Second path
   * @returns {number} - Path diversity score
   */
  calculatePathDiversity(path1, path2) {
    // Convert paths to sets for easy comparison
    const set1 = new Set(path1);
    const set2 = new Set(path2);
    
    // Count shared nodes
    let sharedNodes = 0;
    set1.forEach(node => {
      if (set2.has(node)) {
        sharedNodes++;
      }
    });
    
    // Calculate diversity
    const totalUniqueNodes = set1.size + set2.size - sharedNodes;
    return 1 - (sharedNodes / totalUniqueNodes);
  }

  /**
   * Prepare path sequence for blockchain recording
   * @param {Array} primaryPath - Primary path
   * @param {Array} backupPaths - Backup paths
   * @returns {object} - Path sequence object
   */
  preparePathSequence(primaryPath, backupPaths) {
    const pathId = "SEQ_" + this.generateUniqueId();
    
    return {
      path_id: pathId,
      source_nid: primaryPath[0],
      destination_nias: primaryPath[primaryPath.length - 1],
      path_sequence: primaryPath,
      backup_paths: backupPaths,
      timestamp: new Date().toISOString(),
      metrics: this.calculatePathMetrics(primaryPath)
    };
  }

  /**
   * Calculate aggregate metrics for entire path
   * @param {Array} path - Path to calculate metrics for
   * @returns {object} - Path metrics
   */
  calculatePathMetrics(path) {
    let totalLatency = 0;
    let minBandwidth = Infinity;
    let securityScore = 1;
    
    // Calculate metrics for each segment and aggregate
    for (let i = 0; i < path.length - 1; i++) {
      const metrics = this.getConnectionMetrics(path[i], path[i + 1]);
      
      totalLatency += metrics.latency;
      minBandwidth = Math.min(minBandwidth, metrics.availableBandwidth);
      securityScore *= metrics.securityScore; // Multiply security scores (weakest link principle)
    }
    
    return {
      totalLatency,
      minBandwidth,
      securityScore,
      hopCount: path.length - 1
    };
  }

  /**
   * Record path to blockchain for validation
   * @param {object} pathSequence - Path sequence object
   */
  recordPathToBlockchain(pathSequence) {
    // Format for blockchain storage
    const blockchainRecord = {
      type: "PATH_SEQUENCE",
      data: pathSequence,
      hash: this.hashPathSequence(pathSequence),
      timestamp: pathSequence.timestamp
    };
    
    // Write to blockchain (implementation depends on blockchain interface)
    this.blockchainInterface.writeRecord(blockchainRecord);
  }

  /**
   * Hash path sequence for blockchain integrity
   * @param {object} pathSequence - Path sequence object
   * @returns {string} - Hash of path sequence
   */
  hashPathSequence(pathSequence) {
    // Simple hash function for example purposes
    // In production, use a cryptographic hash function
    return "0x" + this.simpleHash(JSON.stringify(pathSequence));
  }

  /**
   * Simple hash function (for demonstration only)
   * @param {string} str - String to hash
   * @returns {string} - Hashed string
   */
  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(16);
  }

  /**
   * Generate unique ID for path sequence
   * @returns {string} - Unique ID
   */
  generateUniqueId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
  }

  /**
   * Helper method implementations would go here
   * These are referenced above but not fully implemented for brevity
   */
  validateEndpoints(sourceNID, destNIAS) {
    // Implementation would check if endpoints exist and are reachable
    return true;
  }
  
  getActiveNodesFromBlockchain() {
    // Implementation would query blockchain for active nodes
    return [];
  }
  
  nodesCanConnect(nodeA, nodeB) {
    // Implementation would check if nodes can connect
    return true;
  }
  
  getConnectionMetrics(nodeA, nodeB) {
    // Implementation would get real-time metrics for connection
    return {
      latency: 10,
      availableBandwidth: 100,
      maxBandwidth: 1000,
      securityScore: 0.9,
      reliability: 0.95
    };
  }
  
  getHistoricalPerformance(nodeA, nodeB) {
    // Implementation would get historical performance data
    return {
      congestionSamples: []
    };
  }
  
  calculateHistoryFactor(history) {
    // Implementation would calculate factor based on historical performance
    return 0;
  }
  
  reconstructPath(cameFrom, current) {
    // Implementation would reconstruct path from cameFrom map
    return [];
  }
}

/**
 * Priority Queue implementation for A* algorithm
 */
class PriorityQueue {
  constructor() {
    this.elements = [];
  }
  
  enqueue(element, priority) {
    this.elements.push({ element, priority });
    this.elements.sort((a, b) => a.priority - b.priority);
  }
  
  dequeue() {
    return this.elements.shift();
  }
  
  isEmpty() {
    return this.elements.length === 0;
  }
  
  contains(element) {
    return this.elements.some(item => item.element === element);
  }
}

// Example usage:
/*
const sequencer = new EnhancedSequencer();
const path = sequencer.computeOptimalPath('NID-1', 'NIAS-12', { type: 'VoIP' });
console.log(path);
*/