import os
import sys
import json
import logging
from typing import Dict, Any, Optional
import time
import random

from web3 import Web3, HTTPProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BCADNAnalyzer:
    def __init__(
        self, 
        web3: Web3, 
        contract_address: str,
        build_contracts_dir: Optional[str] = None,
        project_root: Optional[str] = None
    ):
        """
        Initialize BCADN Analyzer
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Determine project root and default paths
        self.project_root = project_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # Build contracts directory
        self.build_contracts_dir = build_contracts_dir or os.path.join(
            self.project_root, "build", "contracts"
        )

        # Blockchain connection
        self.w3 = web3

        # Contract details
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract = None
        self.abi = None
        
        # Mock data for testing
        self.mock_nodes = {}
        self.mock_shards = {}
        self.mock_transactions = {}
        self.mock_anomalies = []

    def _load_contract_abi(self, contract_name: str = "BCADN") -> Optional[list]:
        """
        Load ABI for the BCADN contract
        """
        abi_paths = [
            os.path.join(self.build_contracts_dir, f"{contract_name}.json"),
            os.path.join(self.build_contracts_dir, "N2N", f"{contract_name}.json"),
        ]

        for abi_path in abi_paths:
            try:
                if os.path.exists(abi_path):
                    with open(abi_path, "r") as f:
                        content = f.read()
                        
                        try:
                            contract_data = json.loads(content)

                            possible_abi_keys = [
                                "abi", "contractName", "compilerOutput", "output"
                            ]

                            for key in possible_abi_keys:
                                if isinstance(contract_data, dict) and key in contract_data:
                                    abi = contract_data[key]
                                    
                                    if isinstance(abi, list):
                                        print(f"Successfully loaded ABI from {abi_path}")
                                        
                                        function_names = [
                                            func.get("name", "unnamed")
                                            for func in abi
                                            if func.get("type") == "function"
                                        ]
                                        print(f"Available functions: {function_names}")
                                        
                                        return abi

                            if isinstance(contract_data, list):
                                return contract_data

                        except json.JSONDecodeError:
                            self.logger.warning(f"Could not parse JSON from {abi_path}")

            except Exception as e:
                self.logger.warning(f"Error reading ABI at {abi_path}: {e}")

        self.logger.error(f"No ABI found for contract")
        return None

    def load_contract(self) -> Dict[str, Any]:
        """
        Load BCADN contract with comprehensive verification
        """
        if not self.w3.is_connected():
            raise ConnectionError("No Web3 connection available")

        self.abi = self._load_contract_abi()
        if not self.abi:
            raise ValueError("Could not load contract ABI")

        try:
            self.contract = self.w3.eth.contract(
                address=self.contract_address, 
                abi=self.abi
            )

            print("\n--- BCADN Contract Verification ---")
            print(f"Contract Address: {self.contract_address}")
            print(f"Network: {self.w3.eth.chain_id}")

            contract_bytecode = self.w3.eth.get_code(self.contract_address)
            bytecode_length = len(contract_bytecode)
            print(f"Contract Bytecode Length: {bytecode_length}")

            function_names = [
                func.get("name", "unnamed")
                for func in self.abi
                if func.get("type") == "function"
            ]

            view_functions = []
            write_functions = []

            for func_name in function_names:
                try:
                    if not hasattr(self.contract.functions, func_name):
                        continue

                    func = getattr(self.contract.functions, func_name)
                    print(f"\n- {func_name}")

                    try:
                        func_object = self.contract.get_function_by_name(func_name)
                        if func_object.get("stateMutability") in ["view", "pure"]:
                            view_functions.append(func_name)
                        else:
                            write_functions.append(func_name)
                    except:
                        write_functions.append(func_name)

                except Exception as func_error:
                    self.logger.error(f"Error analyzing function {func_name}: {func_error}")

            print(f"\nFunction Statistics:")
            print(f"Total Functions: {len(function_names)}")
            print(f"View/Pure Functions: {len(view_functions)}")
            print(f"State-Changing Functions: {len(write_functions)}")

            return {
                "contract": self.contract,
                "address": self.contract_address,
                "bytecode_size": bytecode_length,
                "view_functions": view_functions,
                "write_functions": write_functions,
            }

        except Exception as e:
            self.logger.error(f"Error loading contract: {e}")
            raise
    
    # Simulation functions for BCADN
    def simulate_register_node(self, node_id, performance, reliability):
        """
        Simulate node registration
        """
        print(f"\nRegistering node {node_id}")
        
        # Calculate initial weight
        alpha = 10  # Default alpha from contract
        beta = 20   # Default beta from contract
        initial_weight = (alpha * 100) + (beta * performance)
        
        self.mock_nodes[node_id] = {
            "nodeId": node_id,
            "performance": performance,
            "reliability": reliability,
            "anomalyScore": 0,
            "weight": initial_weight,
            "isolationTime": 0,
            "status": "Active"  # Enum: Active = 0
        }
        
        print(f"Node registered with initial weight: {initial_weight}")
        return True
    
    def simulate_create_shard(self, shard_id, capacity):
        """
        Simulate shard creation
        """
        print(f"\nCreating shard {shard_id}")
        
        self.mock_shards[shard_id] = {
            "id": shard_id,
            "nodes": [],
            "capacity": capacity,
            "currentLoad": 0,
            "active": True
        }
        
        print(f"Shard created with capacity: {capacity}")
        return True
    
    def simulate_add_node_to_shard(self, shard_id, node_id):
        """
        Simulate adding node to shard
        """
        if shard_id not in self.mock_shards:
            print(f"Error: Shard {shard_id} does not exist")
            return False
            
        if node_id not in self.mock_nodes:
            print(f"Error: Node {node_id} does not exist")
            return False
        
        self.mock_shards[shard_id]["nodes"].append(node_id)
        print(f"Added node {node_id} to shard {shard_id}")
        return True
    
    def simulate_update_node_metrics(self, node_id, performance, reliability, anomaly_score):
        """
        Simulate updating node metrics
        """
        if node_id not in self.mock_nodes:
            print(f"Error: Node {node_id} does not exist")
            return False
            
        node = self.mock_nodes[node_id]
        node["performance"] = performance
        node["reliability"] = reliability
        node["anomalyScore"] = anomaly_score
        
        # Check anomaly threshold
        anomaly_threshold = 30  # Default from contract
        
        if anomaly_score > anomaly_threshold and node["status"] == "Active":
            node["status"] = "Probation"
            node["isolationTime"] = int(time.time())
            print(f"Node {node_id} placed on probation due to high anomaly score")
        
        # Dynamic weight adjustment
        self._simulate_update_node_weight(node_id)
        
        print(f"Updated metrics for node {node_id}")
        return True
    
    def _simulate_update_node_weight(self, node_id):
        """
        Simulate dynamic weight adjustment
        """
        node = self.mock_nodes[node_id]
        
        # Default parameters from contract
        alpha = 10
        beta = 20
        gamma = 30
        
        # Calculate dynamic weight
        # W(t) = (alpha * Fee + beta * Performance - gamma * AnomalyScore)
        new_weight = (alpha * 100) + (beta * node["performance"]) - (gamma * node["anomalyScore"])
        
        # Apply probability gap
        min_probability = 20  # Default from contract
        max_probability = 80  # Default from contract
        
        if new_weight < min_probability:
            new_weight = min_probability
        elif new_weight > max_probability:
            new_weight = max_probability
            
        node["weight"] = new_weight
        print(f"Updated weight for node {node_id}: {new_weight}")
    
    def simulate_submit_transaction(self, sender, receiver, amount):
        """
        Simulate transaction submission
        """
        # Generate transaction hash
        tx_hash = f"0x{random.getrandbits(256):064x}"
        
        # Calculate dynamic fee
        base_fee = 100  # Example base fee
        dynamic_fee = self._simulate_calculate_dynamic_fee(base_fee)
        
        self.mock_transactions[tx_hash] = {
            "txHash": tx_hash,
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "fee": dynamic_fee,
            "timestamp": int(time.time()),
            "processingTime": 0,
            "completed": False
        }
        
        # Assign to shard
        assigned_shard = self._simulate_assign_transaction(tx_hash)
        
        print(f"\nTransaction submitted: {tx_hash}")
        print(f"Sender: {sender}")
        print(f"Receiver: {receiver}")
        print(f"Amount: {amount}")
        print(f"Fee: {dynamic_fee}")
        print(f"Assigned to shard: {assigned_shard}")
        
        return tx_hash
    
    def _simulate_calculate_dynamic_fee(self, base_fee):
        """
        Simulate dynamic fee calculation based on congestion
        """
        # Simple implementation - in real contract, would depend on congestion index
        pending_tx_count = len([tx for tx in self.mock_transactions.values() if not tx["completed"]])
        network_capacity = 1000  # Default capacity
        
        if network_capacity == 0:
            return base_fee
            
        congestion_index = (pending_tx_count * 1e18) / network_capacity
        dynamic_fee = int(base_fee * (1e18 + congestion_index) / 1e18)
        
        return dynamic_fee
    
    def _simulate_assign_transaction(self, tx_hash):
        """
        Simulate transaction assignment to a shard
        """
        if not self.mock_shards:
            # Auto-create a shard if none exists
            self.simulate_create_shard(1, 1000)
            
        # Randomized shard selection
        shard_ids = list(self.mock_shards.keys())
        selected_shard = random.choice(shard_ids)
        
        # Update shard load
        self.mock_shards[selected_shard]["currentLoad"] += 1
        
        return selected_shard
    
    def simulate_process_transaction(self, tx_hash):
        """
        Simulate transaction processing
        """
        if tx_hash not in self.mock_transactions:
            print(f"Error: Transaction {tx_hash} does not exist")
            return False
            
        tx = self.mock_transactions[tx_hash]
        if tx["completed"]:
            print(f"Error: Transaction {tx_hash} already completed")
            return False
            
        # Mark as completed and calculate processing time
        tx["completed"] = True
        tx["processingTime"] = int(time.time()) - tx["timestamp"]
        
        print(f"\nTransaction {tx_hash} processed")
        print(f"Processing time: {tx['processingTime']} seconds")
        
        return True
    
    def simulate_record_anomaly(self, node_id, anomaly_score, attack_type):
        """
        Simulate anomaly recording
        """
        if node_id not in self.mock_nodes:
            print(f"Error: Node {node_id} does not exist")
            return False
            
        anomaly_id = len(self.mock_anomalies)
        
        anomaly = {
            "id": anomaly_id,
            "node": node_id,
            "timestamp": int(time.time()),
            "anomalyScore": anomaly_score,
            "attackType": attack_type,
            "resolved": False
        }
        
        self.mock_anomalies.append(anomaly)
        
        print(f"\nAnomaly recorded for node {node_id}")
        print(f"Anomaly ID: {anomaly_id}")
        print(f"Anomaly Score: {anomaly_score}")
        print(f"Attack Type: {attack_type}")
        
        # Update node metrics to reflect anomaly
        self.simulate_update_node_metrics(
            node_id,
            self.mock_nodes[node_id]["performance"],
            self.mock_nodes[node_id]["reliability"],
            anomaly_score
        )
        
        return anomaly_id
    
    def get_node_details(self, node_id):
        """
        Get node details
        """
        if node_id not in self.mock_nodes:
            print(f"Error: Node {node_id} does not exist")
            return None
            
        return self.mock_nodes[node_id]
    
    def get_shard_details(self, shard_id):
        """
        Get shard details
        """
        if shard_id not in self.mock_shards:
            print(f"Error: Shard {shard_id} does not exist")
            return None
            
        return self.mock_shards[shard_id]
    
    def get_transaction_details(self, tx_hash):
        """
        Get transaction details
        """
        if tx_hash not in self.mock_transactions:
            print(f"Error: Transaction {tx_hash} does not exist")
            return None
            
        return self.mock_transactions[tx_hash]
    
    def get_all_nodes(self):
        """
        Get all nodes
        """
        return list(self.mock_nodes.values())
    
    def get_all_shards(self):
        """
        Get all shards
        """
        return list(self.mock_shards.values())
    
    def get_all_transactions(self):
        """
        Get all transactions
        """
        return list(self.mock_transactions.values())
    
    def get_all_anomalies(self):
        """
        Get all anomalies
        """
        return self.mock_anomalies
    
    def get_node_stats(self):
        """
        Get node stats summary
        """
        if not self.mock_nodes:
            return None
            
        total_nodes = len(self.mock_nodes)
        active_nodes = sum(1 for node in self.mock_nodes.values() if node["status"] == "Active")
        probation_nodes = sum(1 for node in self.mock_nodes.values() if node["status"] == "Probation")
        excluded_nodes = sum(1 for node in self.mock_nodes.values() if node["status"] == "Excluded")
        
        avg_performance = sum(node["performance"] for node in self.mock_nodes.values()) / total_nodes
        avg_reliability = sum(node["reliability"] for node in self.mock_nodes.values()) / total_nodes
        avg_anomaly = sum(node["anomalyScore"] for node in self.mock_nodes.values()) / total_nodes
        
        return {
            "totalNodes": total_nodes,
            "activeNodes": active_nodes,
            "probationNodes": probation_nodes,
            "excludedNodes": excluded_nodes,
            "avgPerformance": avg_performance,
            "avgReliability": avg_reliability,
            "avgAnomalyScore": avg_anomaly
        }
    
    def get_network_stats(self):
        """
        Get network stats summary
        """
        total_tx = len(self.mock_transactions)
        pending_tx = sum(1 for tx in self.mock_transactions.values() if not tx["completed"])
        completed_tx = total_tx - pending_tx
        
        avg_fee = 0
        avg_processing_time = 0
        
        if total_tx > 0:
            avg_fee = sum(tx["fee"] for tx in self.mock_transactions.values()) / total_tx
            
        if completed_tx > 0:
            avg_processing_time = sum(
                tx["processingTime"] for tx in self.mock_transactions.values() if tx["completed"]
            ) / completed_tx
            
        total_shards = len(self.mock_shards)
        total_capacity = sum(shard["capacity"] for shard in self.mock_shards.values())
        total_load = sum(shard["currentLoad"] for shard in self.mock_shards.values())
        
        return {
            "totalTransactions": total_tx,
            "pendingTransactions": pending_tx,
            "completedTransactions": completed_tx,
            "averageFee": avg_fee,
            "averageProcessingTime": avg_processing_time,
            "totalShards": total_shards,
            "totalCapacity": total_capacity,
            "currentLoad": total_load,
            "loadPercentage": (total_load / total_capacity * 100) if total_capacity > 0 else 0
        }


def create_web3_connection(network: str = "sepolia"):
    """
    Create a simple Web3 connection to Sepolia testnet using Infura
    """
    try:
        # Retrieve Infura Project ID from environment variables
        INFURA_PROJECT_ID = os.getenv("INFURA_PROJECT_ID")

        if not INFURA_PROJECT_ID:
            raise ValueError("INFURA_PROJECT_ID not found in environment variables")

        # Construct Infura URL for Sepolia
        sepolia_url = f"https://sepolia.infura.io/v3/{INFURA_PROJECT_ID}"

        # Create Web3 instance
        web3 = Web3(HTTPProvider(sepolia_url))

        # Check connection
        if not web3.is_connected():
            raise ConnectionError("Failed to connect to Infura Sepolia endpoint")

        # Set the default account
        web3.eth.default_account = "0x7927E739C9B0b304610D4Ae35cBf5FDD0D5ad36A"

        # Basic network information
        print("\n--- Web3 Connection ---")
        print(f"Connected to Network: Sepolia")
        print(f"Chain ID: {web3.eth.chain_id}")
        print(f"Latest Block Number: {web3.eth.block_number}")
        print(f"Default Account: {web3.eth.default_account}")

        return web3

    except Exception as e:
        logging.error(f"Blockchain Connection Error: {e}")
        print(f"Connection Failed: {e}")
        return None

def main():
    """
    Main function to interact with BCADN contract
    """
    try:
        # Contract address for BCADN on Sepolia
        BCADN_CONTRACT_ADDRESS = "0x6ad3e5e5a741a1e88602d229aa547e5e013324cf"  # Replace with actual deployed address

        # Create Web3 connection
        web3 = create_web3_connection("sepolia")
        if not web3:
            print("Could not establish blockchain connection. Exiting.")
            sys.exit(1)

        # Initialize analyzer
        analyzer = BCADNAnalyzer(
            web3=web3, 
            contract_address=BCADN_CONTRACT_ADDRESS,
        )

        # Use simulation mode for testing
        use_simulation = True

        # Set up ABI directly from contract code
        custom_abi = {"abi": [
  {
    "inputs": [],
    "stateMutability": "nonpayable",
    "type": "constructor"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "address",
        "name": "node",
        "type": "address"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "anomalyScore",
        "type": "uint256"
      },
      {
        "indexed": False,
        "internalType": "string",
        "name": "attackType",
        "type": "string"
      }
    ],
    "name": "AnomalyDetected",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "address",
        "name": "node",
        "type": "address"
      },
      {
        "indexed": True,
        "internalType": "uint256",
        "name": "attackIndex",
        "type": "uint256"
      }
    ],
    "name": "AnomalyResolved",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "uint256",
        "name": "congestionIndex",
        "type": "uint256"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "pendingTransactions",
        "type": "uint256"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "networkCapacity",
        "type": "uint256"
      }
    ],
    "name": "CongestionUpdated",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "newGap",
        "type": "uint256"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "minProbability",
        "type": "uint256"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "maxProbability",
        "type": "uint256"
      }
    ],
    "name": "GapUpdated",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "uint256",
        "name": "shardId",
        "type": "uint256"
      },
      {
        "indexed": True,
        "internalType": "address",
        "name": "node",
        "type": "address"
      }
    ],
    "name": "NodeAddedToShard",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "address",
        "name": "nodeAddress",
        "type": "address"
      }
    ],
    "name": "NodeRegistered",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "uint256",
        "name": "shardId",
        "type": "uint256"
      },
      {
        "indexed": True,
        "internalType": "address",
        "name": "node",
        "type": "address"
      }
    ],
    "name": "NodeRemovedFromShard",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "address",
        "name": "nodeAddress",
        "type": "address"
      },
      {
        "indexed": False,
        "internalType": "enum BCADN.NodeStatus",
        "name": "newStatus",
        "type": "uint8"
      }
    ],
    "name": "NodeStatusChanged",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "address",
        "name": "nodeAddress",
        "type": "address"
      },
      {
        "indexed": True,
        "internalType": "uint256",
        "name": "newWeight",
        "type": "uint256"
      }
    ],
    "name": "NodeWeightUpdated",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "address",
        "name": "previousOwner",
        "type": "address"
      },
      {
        "indexed": True,
        "internalType": "address",
        "name": "newOwner",
        "type": "address"
      }
    ],
    "name": "OwnershipTransferred",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "uint256",
        "name": "shardId",
        "type": "uint256"
      }
    ],
    "name": "ShardCreated",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "bytes32",
        "name": "txHash",
        "type": "bytes32"
      },
      {
        "indexed": True,
        "internalType": "uint256",
        "name": "shardId",
        "type": "uint256"
      }
    ],
    "name": "TransactionAssigned",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "bytes32",
        "name": "txHash",
        "type": "bytes32"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "processingTime",
        "type": "uint256"
      }
    ],
    "name": "TransactionCompleted",
    "type": "event"
  },
  {
    "anonymous": False,
    "inputs": [
      {
        "indexed": True,
        "internalType": "bytes32",
        "name": "txHash",
        "type": "bytes32"
      },
      {
        "indexed": True,
        "internalType": "address",
        "name": "sender",
        "type": "address"
      },
      {
        "indexed": True,
        "internalType": "address",
        "name": "receiver",
        "type": "address"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "amount",
        "type": "uint256"
      },
      {
        "indexed": False,
        "internalType": "uint256",
        "name": "fee",
        "type": "uint256"
      }
    ],
    "name": "TransactionSubmitted",
    "type": "event"
  },
  {
    "inputs": [],
    "name": "alpha",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "anomalyThreshold",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "name": "attackHistory",
    "outputs": [
      {
        "internalType": "address",
        "name": "node",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "timestamp",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "anomalyScore",
        "type": "uint256"
      },
      {
        "internalType": "string",
        "name": "attackType",
        "type": "string"
      },
      {
        "internalType": "bool",
        "name": "resolved",
        "type": "bool"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "beta",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "congestionIndex",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "currentGap",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "delta",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "gamma",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "maxProbability",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "minProbability",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "mu",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "networkCapacity",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "nextShardId",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "name": "nodeAttackIndices",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "name": "nodesList",
    "outputs": [
      {
        "internalType": "address",
        "name": "",
        "type": "address"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "",
        "type": "address"
      }
    ],
    "name": "nodes",
    "outputs": [
      {
        "internalType": "address",
        "name": "nodeAddress",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "performance",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "reliability",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "anomalyScore",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "weight",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "isolationTime",
        "type": "uint256"
      },
      {
        "internalType": "enum BCADN.NodeStatus",
        "name": "status",
        "type": "uint8"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "owner",
    "outputs": [
      {
        "internalType": "address",
        "name": "",
        "type": "address"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "pendingTransactions",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "probationPeriod",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "renounceOwnership",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "name": "shardIds",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "name": "shards",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "id",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "capacity",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "currentLoad",
        "type": "uint256"
      },
      {
        "internalType": "bool",
        "name": "active",
        "type": "bool"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "newOwner",
        "type": "address"
      }
    ],
    "name": "transferOwnership",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "bytes32",
        "name": "",
        "type": "bytes32"
      }
    ],
    "name": "transactionHashes",
    "outputs": [
      {
        "internalType": "bytes32",
        "name": "",
        "type": "bytes32"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "bytes32",
        "name": "",
        "type": "bytes32"
      }
    ],
    "name": "transactionToShard",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "bytes32",
        "name": "",
        "type": "bytes32"
      }
    ],
    "name": "transactions",
    "outputs": [
      {
        "internalType": "bytes32",
        "name": "txHash",
        "type": "bytes32"
      },
      {
        "internalType": "address",
        "name": "sender",
        "type": "address"
      },
      {
        "internalType": "address",
        "name": "receiver",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "amount",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "fee",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "timestamp",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "processingTime",
        "type": "uint256"
      },
      {
        "internalType": "bool",
        "name": "completed",
        "type": "bool"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_probability",
        "type": "uint256"
      }
    ],
    "name": "adjustToProbabilityGap",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_probability",
        "type": "uint256"
      }
    ],
    "name": "isWithinGap",
    "outputs": [
      {
        "internalType": "bool",
        "name": "",
        "type": "bool"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_baseFee",
        "type": "uint256"
      }
    ],
    "name": "calculateDynamicFee",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "_nodeAddress",
        "type": "address"
      }
    ],
    "name": "checkProbationStatus",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "getAllNodes",
    "outputs": [
      {
        "internalType": "address[]",
        "name": "",
        "type": "address[]"
      },
      {
        "internalType": "uint256[]",
        "name": "",
        "type": "uint256[]"
      },
      {
        "internalType": "enum BCADN.NodeStatus[]",
        "name": "",
        "type": "uint8[]"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "getAllShards",
    "outputs": [
      {
        "internalType": "uint256[]",
        "name": "",
        "type": "uint256[]"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "bytes32",
        "name": "_txHash",
        "type": "bytes32"
      }
    ],
    "name": "assignTransactionToShard",
    "outputs": [
      {
        "internalType": "uint256",
        "name": "",
        "type": "uint256"
      }
    ],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "getAttackHistory",
    "outputs": [
      {
        "internalType": "address[]",
        "name": "nodeAddresses",
        "type": "address[]"
      },
      {
        "internalType": "uint256[]",
        "name": "timestamps",
        "type": "uint256[]"
      },
      {
        "internalType": "uint256[]",
        "name": "anomalyScores",
        "type": "uint256[]"
      },
      {
        "internalType": "bool[]",
        "name": "resolved",
        "type": "bool[]"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "getAllTransactionHashes",
    "outputs": [
      {
        "internalType": "bytes32[]",
        "name": "",
        "type": "bytes32[]"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_alpha",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_beta",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_gamma",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_delta",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_mu",
        "type": "uint256"
      }
    ],
    "name": "updateNetworkParams",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_anomalyThreshold",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_probationPeriod",
        "type": "uint256"
      }
    ],
    "name": "updateThresholds",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_minProbability",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_maxProbability",
        "type": "uint256"
      }
    ],
    "name": "updateProbabilityRange",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_capacity",
        "type": "uint256"
      }
    ],
    "name": "setNetworkCapacity",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "_nodeAddress",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "_performance",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_reliability",
        "type": "uint256"
      }
    ],
    "name": "registerNode",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "_nodeAddress",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "_performance",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_reliability",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_anomalyScore",
        "type": "uint256"
      }
    ],
    "name": "updateNodeMetrics",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_capacity",
        "type": "uint256"
      }
    ],
    "name": "createShard",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_shardId",
        "type": "uint256"
      },
      {
        "internalType": "address",
        "name": "_node",
        "type": "address"
      }
    ],
    "name": "addNodeToShard",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_shardId",
        "type": "uint256"
      },
      {
        "internalType": "address",
        "name": "_node",
        "type": "address"
      }
    ],
    "name": "removeNodeFromShard",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "_node",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "_anomalyScore",
        "type": "uint256"
      },
      {
        "internalType": "string",
        "name": "_attackType",
        "type": "string"
      }
    ],
    "name": "recordAnomaly",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "uint256",
        "name": "_attackIndex",
        "type": "uint256"
      }
    ],
    "name": "resolveAnomaly",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "address",
        "name": "_receiver",
        "type": "address"
      },
      {
        "internalType": "uint256",
        "name": "_amount",
        "type": "uint256"
      },
      {
        "internalType": "uint256",
        "name": "_baseFee",
        "type": "uint256"
      }
    ],
    "name": "submitTransaction",
    "outputs": [
      {
        "internalType": "bytes32",
        "name": "",
        "type": "bytes32"
      }
    ],
    "stateMutability": "payable",
    "type": "function"
  },
  {
    "inputs": [
      {
        "internalType": "bytes32",
        "name": "_txHash",
        "type": "bytes32"
      }
    ],
    "name": "processTransaction",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "withdrawFees",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  }
]}
        
        # Load contract with custom ABI
        analyzer.abi = custom_abi["abi"]
        contract_details = analyzer.load_contract()
        contract = contract_details['contract']

        # Get the private key from environment
        private_key = os.getenv('PRIVATE_KEY')
        if not private_key:
            raise ValueError("PRIVATE_KEY not found in environment variables")
        
        # Add '0x' prefix if it's missing
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key

        # Test data for BCADN contract interaction
        print("\n=== BCADN Network Simulation ===")

        # 1. Register nodes with varying performance levels
        node_data = [
            {"id": "node_01", "performance": 95, "reliability": 97},  # High performance
            {"id": "node_02", "performance": 92, "reliability": 94},  # High performance
            {"id": "node_03", "performance": 89, "reliability": 91},  # Medium-high performance
            {"id": "node_04", "performance": 85, "reliability": 88},  # Medium performance
            {"id": "node_05", "performance": 82, "reliability": 85},  # Medium performance
            {"id": "node_06", "performance": 78, "reliability": 82},  # Medium-low performance
            {"id": "node_07", "performance": 75, "reliability": 80},  # Medium-low performance 
            {"id": "node_08", "performance": 72, "reliability": 78},  # Low performance
            {"id": "node_09", "performance": 68, "reliability": 72},  # Low performance
            {"id": "node_10", "performance": 65, "reliability": 70},  # Low performance
            {"id": "node_11", "performance": 90, "reliability": 93},  # High performance
            {"id": "node_12", "performance": 86, "reliability": 89},  # Medium-high performance
            {"id": "node_13", "performance": 80, "reliability": 85},  # Medium performance
            {"id": "node_14", "performance": 74, "reliability": 79},  # Medium-low performance
            {"id": "node_15", "performance": 69, "reliability": 75},  # Low performance
        ]

        print("\n--- Registering Nodes ---")
        for node in node_data:
            if use_simulation:
                analyzer.simulate_register_node(node["id"], node["performance"], node["reliability"])
            else:
                # This would be real blockchain transaction in non-simulation mode
                # Skipping real transaction implementation for brevity
                pass
        
        # 2. Create shards
        print("\n--- Creating Shards ---")
        shard_data = [
            {"id": 1, "capacity": 1000},  # High capacity
            {"id": 2, "capacity": 800},   # Medium-high capacity
            {"id": 3, "capacity": 600},   # Medium capacity
            {"id": 4, "capacity": 400},   # Medium-low capacity
            {"id": 5, "capacity": 200}    # Low capacity
        ]
        
        for shard in shard_data:
            if use_simulation:
                analyzer.simulate_create_shard(shard["id"], shard["capacity"])
            else:
                # This would be real blockchain transaction in non-simulation mode
                pass
                
        # 3. Assign nodes to shards
        print("\n--- Assigning Nodes to Shards ---")
        # Distribute nodes across shards
        node_assignments = [
            {"shard_id": 1, "node_ids": ["node_01", "node_02", "node_03"]},
            {"shard_id": 2, "node_ids": ["node_04", "node_05", "node_06"]},
            {"shard_id": 3, "node_ids": ["node_07", "node_08", "node_09"]},
            {"shard_id": 4, "node_ids": ["node_10", "node_11", "node_12"]},
            {"shard_id": 5, "node_ids": ["node_13", "node_14", "node_15"]}
        ]
        
        for assignment in node_assignments:
            for node_id in assignment["node_ids"]:
                if use_simulation:
                    analyzer.simulate_add_node_to_shard(assignment["shard_id"], node_id)
                else:
                    # This would be real blockchain transaction in non-simulation mode
                    pass
                    
        # 4. Submit transactions
        print("\n--- Submitting Transactions ---")
        
        # Create sample transactions with different volumes
        transaction_data = []
        for i in range(1, 51):  # 50 transactions
            sender = f"account_{random.randint(1, 100)}"
            receiver = f"account_{random.randint(1, 100)}"
            amount = random.randint(100, 10000)
            transaction_data.append({"sender": sender, "receiver": receiver, "amount": amount})
            
        # Submit transactions in batches to simulate network traffic
        tx_hashes = []
        for i, tx in enumerate(transaction_data):
            if use_simulation:
                tx_hash = analyzer.simulate_submit_transaction(tx["sender"], tx["receiver"], tx["amount"])
                tx_hashes.append(tx_hash)
                # Simulate small delay between transactions
                if i % 10 == 0:
                    time.sleep(0.1)
            else:
                # This would be real blockchain transaction in non-simulation mode
                pass
        
        # 5. Simulate anomalies on some nodes
        print("\n--- Simulating Anomalies ---")
        
        # Define sample attack scenarios with different types and severity
        attack_scenarios = [
            {"node_id": "node_03", "anomaly_score": 25, "attack_type": "Unusual Traffic Pattern"},
            {"node_id": "node_08", "anomaly_score": 35, "attack_type": "DDoS Attempt"},
            {"node_id": "node_14", "anomaly_score": 40, "attack_type": "Malicious Connection Attempt"},
            {"node_id": "node_05", "anomaly_score": 15, "attack_type": "Data Manipulation Probe"},
            {"node_id": "node_11", "anomaly_score": 28, "attack_type": "Resource Exhaustion"}
        ]
        
        anomaly_ids = []
        for attack in attack_scenarios:
            if use_simulation:
                anomaly_id = analyzer.simulate_record_anomaly(
                    attack["node_id"], 
                    attack["anomaly_score"], 
                    attack["attack_type"]
                )
                anomaly_ids.append(anomaly_id)
                # Short delay between anomalies
                time.sleep(0.1)
            else:
                # This would be real blockchain transaction in non-simulation mode
                pass
        
        # 6. Process some of the transactions
        print("\n--- Processing Transactions ---")
        
        # Process 70% of transactions (simulating network throughput)
        tx_to_process = tx_hashes[:int(len(tx_hashes) * 0.7)]
        for tx_hash in tx_to_process:
            if use_simulation:
                analyzer.simulate_process_transaction(tx_hash)
                # Small delay to simulate processing time
                time.sleep(0.05)
            else:
                # This would be real blockchain transaction in non-simulation mode
                pass
                
        # 7. Collect and display network statistics
        print("\n=== BCADN Network Analysis Results ===")
        
        # Node statistics
        if use_simulation:
            node_stats = analyzer.get_node_stats()
            if node_stats:
                print("\n--- Node Statistics ---")
                print(f"Total Nodes: {node_stats['totalNodes']}")
                print(f"Active Nodes: {node_stats['activeNodes']} ({node_stats['activeNodes']/node_stats['totalNodes']*100:.1f}%)")
                print(f"Probation Nodes: {node_stats['probationNodes']} ({node_stats['probationNodes']/node_stats['totalNodes']*100:.1f}%)")
                print(f"Excluded Nodes: {node_stats['excludedNodes']} ({node_stats['excludedNodes']/node_stats['totalNodes']*100:.1f}%)")
                print(f"Average Performance: {node_stats['avgPerformance']:.2f}")
                print(f"Average Reliability: {node_stats['avgReliability']:.2f}")
                print(f"Average Anomaly Score: {node_stats['avgAnomalyScore']:.2f}")
        
        # Network statistics
        if use_simulation:
            network_stats = analyzer.get_network_stats()
            if network_stats:
                print("\n--- Network Statistics ---")
                print(f"Total Transactions: {network_stats['totalTransactions']}")
                print(f"Completed Transactions: {network_stats['completedTransactions']} ({network_stats['completedTransactions']/network_stats['totalTransactions']*100:.1f}%)")
                print(f"Pending Transactions: {network_stats['pendingTransactions']} ({network_stats['pendingTransactions']/network_stats['totalTransactions']*100:.1f}%)")
                print(f"Average Fee: {network_stats['averageFee']:.2f}")
                print(f"Average Processing Time: {network_stats['averageProcessingTime']:.2f} seconds")
                print(f"Total Shards: {network_stats['totalShards']}")
                print(f"Total Capacity: {network_stats['totalCapacity']}")
                print(f"Current Load: {network_stats['currentLoad']} ({network_stats['loadPercentage']:.1f}%)")
        
        # Anomaly statistics
        if use_simulation:
            anomalies = analyzer.get_all_anomalies()
            if anomalies:
                print("\n--- Anomaly Statistics ---")
                print(f"Total Anomalies Detected: {len(anomalies)}")
                
                # Group by attack type
                attack_types = {}
                for anomaly in anomalies:
                    attack_type = anomaly["attackType"]
                    if attack_type in attack_types:
                        attack_types[attack_type] += 1
                    else:
                        attack_types[attack_type] = 1
                        
                print("Breakdown by Attack Type:")
                for attack_type, count in attack_types.items():
                    print(f"  - {attack_type}: {count} ({count/len(anomalies)*100:.1f}%)")
                    
                # Average anomaly score
                avg_score = sum(anomaly["anomalyScore"] for anomaly in anomalies) / len(anomalies)
                print(f"Average Anomaly Score: {avg_score:.2f}")
        
        # High-level assessment
        if use_simulation:
            print("\n--- Network Assessment ---")
            
            # Calculate overall network health
            completed_ratio = network_stats['completedTransactions'] / network_stats['totalTransactions']
            active_node_ratio = node_stats['activeNodes'] / node_stats['totalNodes']
            load_ratio = network_stats['currentLoad'] / network_stats['totalCapacity']
            anomaly_ratio = len(anomalies) / node_stats['totalNodes']
            
            health_score = (completed_ratio * 0.4) + (active_node_ratio * 0.3) + ((1 - load_ratio) * 0.2) + ((1 - anomaly_ratio) * 0.1)
            health_score = health_score * 100
            
            print(f"Overall Network Health Score: {health_score:.1f}%")
            
            if health_score >= 90:
                health_rating = "Excellent"
            elif health_score >= 80:
                health_rating = "Good"
            elif health_score >= 70:
                health_rating = "Satisfactory"
            elif health_score >= 60:
                health_rating = "Fair"
            else:
                health_rating = "Poor"
                
            print(f"Network Health Rating: {health_rating}")
            
            # Performance bottlenecks
            print("\nPotential Bottlenecks:")
            if load_ratio > 0.8:
                print("  - High network load exceeding 80% of capacity")
            if node_stats['probationNodes'] > 0:
                print(f"  - {node_stats['probationNodes']} nodes on probation due to anomalies")
            if network_stats['pendingTransactions'] / network_stats['totalTransactions'] > 0.3:
                print("  - High percentage of pending transactions (>30%)")
            
            # Recommendations
            print("\nRecommendations:")
            if load_ratio > 0.8:
                print("  - Increase network capacity or add more shards")
            if node_stats['probationNodes'] > 0:
                print("  - Investigate and resolve node anomalies")
            if anomaly_ratio > 0.2:
                print("  - Enhance security monitoring and implement countermeasures")
            if network_stats['pendingTransactions'] / network_stats['totalTransactions'] > 0.3:
                print("  - Optimize transaction processing or add more processing nodes")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()