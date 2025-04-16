# crosschainproject
 
# Node Addressed SDN Intelligent Blockchain Network

## Overview

This repository implements the Node-to-Node (N2N) multi-layer communication scheme as described in the research paper. The system enables direct communication between Application Nodes and BGP networks through a blockchain-based approach.

Key components of this implementation:

1. **NID (Node ID) Registry** - For registering and managing nodes on the Application Layer
2. **NIAS (Node Identifiable Autonomous Systems) Registry** - For managing BGP-layer nodes
3. **ABATL (Attribute-Based Translation Layer)** - For bridging NID and NIAS nodes
4. **Sequence Path Router** - For managing routing sequences and path optimization
5. **Clustering Contract** - For efficient node clustering and management

## Architecture

![N2N System Architecture](./docs/architecture.png)

The system consists of three primary layers:

1. **Node Application Layer (NAP)** - Contains application nodes with NIDs
2. **Attribute-Based Translation Layer (ABATL)** - Translates between NAP and B-BGP layers
3. **Blockchain-based BGP Layer (B-BGP)** - Contains NIAS nodes for routing

## Prerequisites

- Node.js (v14+)
- Truffle Suite
- Ganache-CLI
- Python 3.8+
- Web3.py
- Pandas
- Matplotlib

## Setup

### 1. Install Dependencies

```bash
# Install Truffle and Ganache
npm install -g truffle ganache-cli

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start Local Blockchain

```bash
ganache-cli -a 20
```

### 3. Compile and Deploy Contracts

```bash
truffle compile
truffle migrate --reset
```

This will deploy all the contracts and save their addresses to `contract_addresses.json`.

### 4. Run Tests

```bash
# Run smart contract tests
truffle test

# Run Python evaluation script
python test_n2n_routing.py
```

## Smart Contracts

### NIDRegistry.sol

Manages Node IDs (NIDs) on the Application Layer. Handles:
- Registration of nodes with primary and secondary IDs
- Node status management (active/inactive)
- Cluster membership

### NIASRegistry.sol

Manages Node Identifiable Autonomous Systems on the BGP Layer. Handles:
- Registration of NIAS nodes with primary and secondary IDs
- Routing weight and load balancing parameters
- Security levels and cluster membership

### ABATLTranslation.sol

Provides translation between NAP and B-BGP layers. Handles:
- Attributes mapping between NIDs and NIAS
- QoS translations
- Security policy enforcement

### SequencePathRouter.sol

Implements the sequence-based routing mechanism. Handles:
- Path creation and management
- Disjoint path generation for fault tolerance
- Transmission monitoring
- Rerouting on node failures

### ClusteringContract.sol

Manages node clustering for efficient network organization. Handles:
- Cluster creation and management
- Node assignment to clusters
- Metrics collection and analysis
- Dynamic reclustering

## Performance Evaluation

The Python evaluation script (`test_n2n_routing.py`) assesses the performance of the N2N system across several dimensions:

1. **Transmission Efficiency** - Measures latency, throughput, and success rate
2. **Node Failure Recovery** - Tests the system's ability to reroute on node failures
3. **Clustering Efficiency** - Evaluates the clustering approach's impact on performance
4. **Comparison with Traditional BGP** - Compares with traditional BGP approaches

Results are saved to the `results` directory as visualizations and a summary report.

## Key Innovations

The implemented system demonstrates several key innovations:

1. **Direct Node-to-Node Communication** - Eliminates the dependency on IP-based naming
2. **Precomputed Path Sequences** - Optimizes routing decisions ahead of transmission time
3. **Attribute-Based Translation** - Seamlessly bridges application and BGP layers
4. **Blockchain-Based Verification** - Ensures secure and tamper-proof routing
5. **Dynamic Clustering** - Optimizes network organization for efficiency
6. **Disjoint Path Routing** - Provides robust failure recovery

## License

MIT

## Acknowledgments

This implementation is based on the research paper "Node To Node (N2N) Addressing Protocol For Blockchain Attribute-Based Translation (ABATL) BGP Networks".