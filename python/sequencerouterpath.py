import os
import sys
import json
import logging
from typing import Dict, Any, Optional
import time  # Add for simulated timestamps

from web3 import Web3, HTTPProvider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SequencePathRouterAnalyzer:
    def __init__(
        self, 
        web3: Web3, 
        contract_address: str,
        build_contracts_dir: Optional[str] = None,
        project_root: Optional[str] = None
    ):
        """
        Initialize SequencePathRouter Analyzer
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
        
        # Add mock data storage
        self.mock_paths = {}
        self.mock_service_paths = {"VoIP": [], "Streaming": [], "Standard": []}
        self.mock_node_performance = {}

    def _load_contract_abi(self, contract_name: str = "SequencePathRouter") -> Optional[list]:
        """
        Load ABI for the SequencePathRouter contract
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
        Load SequencePathRouter contract with comprehensive verification
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

            print("\n--- SequencePathRouter Contract Verification ---")
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
            
    # Add new functions for mock data handling
    def simulate_path_creation(self, path_data):
        """
        Simulate successful path creation and store mock data
        """
        path_id = path_data["pathId"]
        service_class = path_data["serviceClass"]
        
        # Store path data
        self.mock_paths[path_id] = {
            "pathId": path_id,
            "sourceNode": path_data["sourceNode"],
            "destinationNode": path_data["destinationNode"],
            "pathSequence": path_data["pathSequence"],
            "serviceClass": service_class,
            "isActive": True,
            "creationTime": int(time.time()),
            "lastUpdated": int(time.time()),
            "pathStatus": 1
        }
        
        # Add to service class paths
        if service_class in self.mock_service_paths:
            self.mock_service_paths[service_class].append(path_id)
        
        # Update node performance data
        for node in path_data["pathSequence"]:
            if node not in self.mock_node_performance:
                # Generate some realistic performance data
                self.mock_node_performance[node] = {
                    "successRate": 95 + (hash(node) % 5),  # 95-99% success rate
                    "packetCount": 1000 + (hash(node) % 2000)  # 1000-3000 packets
                }
        
        print(f"Transaction status: Success (Simulated)")
        return True
        
    def simulate_get_paths_by_service(self, service_class):
        """
        Return mocked paths for a service class
        """
        return self.mock_service_paths.get(service_class, [])
    
    def simulate_get_node_performance(self, node_id):
        """
        Return mocked node performance data
        """
        if node_id in self.mock_node_performance:
            data = self.mock_node_performance[node_id]
            return (data["successRate"], data["packetCount"])
        return (0, 0)
    
    def simulate_path_transmission(self, path_id, packets_total=1000, security_level=2):
        """
        Simulate a transmission on a path
        """
        if path_id not in self.mock_paths:
            print(f"Path {path_id} not found")
            return False
            
        start_time = int(time.time())
        print(f"\nSimulating transmission for path {path_id}")
        print(f"Transmission started at: {start_time}")
        print(f"Packets total: {packets_total}")
        print(f"Security level: {security_level}")
        
        # Simulate a short delay
        time.sleep(0.5)
        
        # Complete the transmission with some simulated metrics
        end_time = int(time.time())
        packets_lost = int(packets_total * 0.002)  # 0.2% packet loss
        latency = 15 + (hash(path_id) % 30)  # 15-45ms latency
        
        print(f"Transmission completed at: {end_time}")
        print(f"Transmission duration: {end_time - start_time} seconds")
        print(f"Packets lost: {packets_lost}")
        print(f"Measured latency: {latency}ms")
        print(f"Compliance check: Passed")
        
        # Update node performance based on transmission
        path_sequence = self.mock_paths[path_id]["pathSequence"]
        for node in path_sequence:
            if node in self.mock_node_performance:
                # Update packet count
                self.mock_node_performance[node]["packetCount"] += packets_total
                
                # Slightly adjust success rate based on packet loss
                old_rate = self.mock_node_performance[node]["successRate"]
                new_rate = old_rate - (packets_lost/packets_total * 0.5)
                self.mock_node_performance[node]["successRate"] = max(90, new_rate)
        
        return True

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
    Main function to interact with SequencePathRouter contract
    """
    try:
        # Updated Sepolia Contract Address from contract_addresses.json
        SEQUENCE_PATH_ROUTER_ADDRESS = "0x4ac58335090c3F31b9ECE65ce611F69e2C55435e"

        # Create Web3 connection
        web3 = create_web3_connection("sepolia")
        if not web3:
            print("Could not establish blockchain connection. Exiting.")
            sys.exit(1)

        # Initialize analyzer with custom ABI
        analyzer = SequencePathRouterAnalyzer(
            web3=web3, 
            contract_address=SEQUENCE_PATH_ROUTER_ADDRESS,
        )

        # Use simulation mode flag
        use_simulation = True

        # Custom ABI (keep this exactly as in the original script)
        custom_abi = {"abi": [
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": True,
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "uint256",
            "name": "disjointPathIndex",
            "type": "uint256"
          }
        ],
        "name": "DisjointPathCreated",
        "type": "event"
      },
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": True,
            "internalType": "string",
            "name": "nodeId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "uint256",
            "name": "successRate",
            "type": "uint256"
          }
        ],
        "name": "NodePerformanceUpdated",
        "type": "event"
      },
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": True,
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "string",
            "name": "sourceNode",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "string",
            "name": "destinationNode",
            "type": "string"
          }
        ],
        "name": "PathCreated",
        "type": "event"
      },
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": True,
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "string",
            "name": "failedNode",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "string[]",
            "name": "newSequence",
            "type": "string[]"
          }
        ],
        "name": "PathRerouted",
        "type": "event"
      },
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": False,
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "uint8",
            "name": "status",
            "type": "uint8"
          }
        ],
        "name": "PathStatusChanged",
        "type": "event"
      },
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": True,
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "string[]",
            "name": "pathSequence",
            "type": "string[]"
          }
        ],
        "name": "PathUpdated",
        "type": "event"
      },
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": True,
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "uint256",
            "name": "endTime",
            "type": "uint256"
          },
          {
            "indexed": False,
            "internalType": "uint256",
            "name": "packetsLost",
            "type": "uint256"
          }
        ],
        "name": "TransmissionCompleted",
        "type": "event"
      },
      {
        "anonymous": False,
        "inputs": [
          {
            "indexed": True,
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "indexed": False,
            "internalType": "uint256",
            "name": "startTime",
            "type": "uint256"
          }
        ],
        "name": "TransmissionStarted",
        "type": "event"
      },
      {
        "inputs": [],
        "name": "allPathIds",
        "outputs": [
          {
            "internalType": "string[]",
            "name": "",
            "type": "string[]"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "_pathId",
            "type": "string"
          },
          {
            "internalType": "uint256",
            "name": "_packetsLost",
            "type": "uint256"
          },
          {
            "internalType": "uint16",
            "name": "_measuredLatency",
            "type": "uint16"
          },
          {
            "internalType": "bool",
            "name": "_complianceCheck",
            "type": "bool"
          }
        ],
        "name": "completeTransmission",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "_pathId",
            "type": "string"
          },
          {
            "internalType": "string[]",
            "name": "_disjointSequence",
            "type": "string[]"
          }
        ],
        "name": "createDisjointPath",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "_pathId",
            "type": "string"
          },
          {
            "internalType": "string",
            "name": "_sourceNode",
            "type": "string"
          },
          {
            "internalType": "string",
            "name": "_destinationNode",
            "type": "string"
          },
          {
            "internalType": "string[]",
            "name": "_pathSequence",
            "type": "string[]"
          },
          {
            "internalType": "string",
            "name": "_serviceClass",
            "type": "string"
          }
        ],
        "name": "createPath",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "",
            "type": "string"
          },
          {
            "internalType": "uint256",
            "name": "",
            "type": "uint256"
          }
        ],
        "name": "disjointPaths",
        "outputs": [
          {
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "internalType": "uint256",
            "name": "creationTime",
            "type": "uint256"
          },
          {
            "internalType": "bool",
            "name": "isActive",
            "type": "bool"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "_nodeId",
            "type": "string"
          }
        ],
        "name": "getNodePerformance",
        "outputs": [
          {
            "internalType": "uint256",
            "name": "successRate",
            "type": "uint256"
          },
          {
            "internalType": "uint256",
            "name": "packetCount",
            "type": "uint256"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "_serviceClass",
            "type": "string"
          }
        ],
        "name": "getPathsByServiceClass",
        "outputs": [
          {
            "internalType": "string[]",
            "name": "",
            "type": "string[]"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "",
            "type": "string"
          }
        ],
        "name": "nodePacketCount",
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
            "internalType": "string",
            "name": "",
            "type": "string"
          }
        ],
        "name": "nodeSuccessRate",
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
            "internalType": "string",
            "name": "",
            "type": "string"
          },
          {
            "internalType": "uint256",
            "name": "",
            "type": "uint256"
          }
        ],
        "name": "nodeToActivePaths",
        "outputs": [
          {
            "internalType": "string",
            "name": "",
            "type": "string"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "",
            "type": "string"
          }
        ],
        "name": "paths",
        "outputs": [
          {
            "internalType": "string",
            "name": "pathId",
            "type": "string"
          },
          {
            "internalType": "string",
            "name": "sourceNode",
            "type": "string"
          },
          {
            "internalType": "string",
            "name": "destinationNode",
            "type": "string"
          },
          {
            "internalType": "bool",
            "name": "isActive",
            "type": "bool"
          },
          {
            "internalType": "uint256",
            "name": "creationTime",
            "type": "uint256"
          },
          {
            "internalType": "uint256",
            "name": "lastUpdated",
            "type": "uint256"
          },
          {
            "internalType": "uint8",
            "name": "pathStatus",
            "type": "uint8"
          },
          {
            "internalType": "string",
            "name": "serviceClass",
            "type": "string"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "",
            "type": "string"
          }
        ],
        "name": "pathStatus",
        "outputs": [
          {
            "internalType": "uint256",
            "name": "startTime",
            "type": "uint256"
          },
          {
            "internalType": "uint256",
            "name": "endTime",
            "type": "uint256"
          },
          {
            "internalType": "uint256",
            "name": "packetsTotal",
            "type": "uint256"
          },
          {
            "internalType": "uint256",
            "name": "packetsLost",
            "type": "uint256"
          },
          {
            "internalType": "uint16",
            "name": "measuredLatency",
            "type": "uint16"
          },
          {
            "internalType": "uint8",
            "name": "securityLevel",
            "type": "uint8"
          },
          {
            "internalType": "bool",
            "name": "complianceCheck",
            "type": "bool"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "",
            "type": "string"
          },
          {
            "internalType": "uint256",
            "name": "",
            "type": "uint256"
          }
        ],
        "name": "pathsByService",
        "outputs": [
          {
            "internalType": "string",
            "name": "",
            "type": "string"
          }
        ],
        "stateMutability": "view",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "_pathId",
            "type": "string"
          },
          {
            "internalType": "string",
            "name": "_failedNodeId",
            "type": "string"
          }
        ],
        "name": "reroutePath",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
      },
      {
        "inputs": [
          {
            "internalType": "string",
            "name": "_pathId",
            "type": "string"
          },
          {
            "internalType": "uint256",
            "name": "_packetsTotal",
            "type": "uint256"
          },
          {
            "internalType": "uint8",
            "name": "_securityLevel",
            "type": "uint8"
          }
        ],
        "name": "startTransmission",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
      }
    ]}
        
        # Load contract with custom ABI (maintaining original structure)
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

        # Sample data for path creation and interaction (expanded with more test data)
        sample_paths = [
            {
        "pathId": "voip_path_1",
        "sourceNode": "node1",
        "destinationNode": "node4",
        "pathSequence": ["node1", "node2", "node3", "node4"],
        "serviceClass": "VoIP"
    },
    {
        "pathId": "streaming_path_1",
        "sourceNode": "node5",
        "destinationNode": "node8",
        "pathSequence": ["node5", "node6", "node7", "node8"],
        "serviceClass": "Streaming"
    },
    {
        "pathId": "standard_path_1",
        "sourceNode": "node9",
        "destinationNode": "node12",
        "pathSequence": ["node9", "node10", "node11", "node12"],
        "serviceClass": "Standard"
    },
    {
        "pathId": "voip_path_2",
        "sourceNode": "node2",
        "destinationNode": "node7",
        "pathSequence": ["node2", "node3", "node5", "node7"],
        "serviceClass": "VoIP"
    },
    # Additional paths for expanded test data
    {
        "pathId": "voip_path_3",
        "sourceNode": "node3",
        "destinationNode": "node9",
        "pathSequence": ["node3", "node4", "node8", "node9"],
        "serviceClass": "VoIP"
    },
    {
        "pathId": "streaming_path_2",
        "sourceNode": "node6",
        "destinationNode": "node13",
        "pathSequence": ["node6", "node7", "node9", "node13"],
        "serviceClass": "Streaming"
    },
    {
        "pathId": "streaming_path_3",
        "sourceNode": "node2",
        "destinationNode": "node14",
        "pathSequence": ["node2", "node4", "node9", "node14"],
        "serviceClass": "Streaming"
    },
    {
        "pathId": "standard_path_2",
        "sourceNode": "node5",
        "destinationNode": "node15",
        "pathSequence": ["node5", "node10", "node12", "node15"],
        "serviceClass": "Standard"
    },
    {
        "pathId": "standard_path_3",
        "sourceNode": "node1",
        "destinationNode": "node16",
        "pathSequence": ["node1", "node6", "node11", "node16"],
        "serviceClass": "Standard"
    },
    {
        "pathId": "voip_path_4",
        "sourceNode": "node7",
        "destinationNode": "node17",
        "pathSequence": ["node7", "node9", "node13", "node17"],
        "serviceClass": "VoIP"
    },
    {
        "pathId": "streaming_path_4",
        "sourceNode": "node4",
        "destinationNode": "node18",
        "pathSequence": ["node4", "node8", "node14", "node18"],
        "serviceClass": "Streaming"
    },
    {
        "pathId": "standard_path_4",
        "sourceNode": "node3",
        "destinationNode": "node19",
        "pathSequence": ["node3", "node7", "node15", "node19"],
        "serviceClass": "Standard"
    },
    {
        "pathId": "voip_path_5",
        "sourceNode": "node10",
        "destinationNode": "node20",
        "pathSequence": ["node10", "node12", "node16", "node20"],
        "serviceClass": "VoIP"
    },
    {
        "pathId": "streaming_path_5",
        "sourceNode": "node8",
        "destinationNode": "node21",
        "pathSequence": ["node8", "node11", "node17", "node21"],
        "serviceClass": "Streaming"
    },
    {
        "pathId": "high_priority_path_1",
        "sourceNode": "node1",
        "destinationNode": "node22",
        "pathSequence": ["node1", "node5", "node10", "node22"],
        "serviceClass": "VoIP"
    },
    {
        "pathId": "secure_path_1",
        "sourceNode": "node2",
        "destinationNode": "node23",
        "pathSequence": ["node2", "node6", "node12", "node23"],
        "serviceClass": "Standard"
    },
    {
        "pathId": "low_latency_path_1",
        "sourceNode": "node3",
        "destinationNode": "node24",
        "pathSequence": ["node3", "node7", "node13", "node24"],
        "serviceClass": "Streaming"
    },
    {
        "pathId": "high_bandwidth_path_1",
        "sourceNode": "node4",
        "destinationNode": "node25",
        "pathSequence": ["node4", "node8", "node14", "node25"],
        "serviceClass": "Streaming"
    }

        ]

        # Create paths
        for path in sample_paths:
            try:
                print(f"\nAttempting to create path {path['pathId']}")
                
                if use_simulation:
                    # Use simulation method instead of actual blockchain transaction
                    analyzer.simulate_path_creation(path)
                else:
                    # Original blockchain transaction code
                    # Get the nonce for the transaction
                    nonce = web3.eth.get_transaction_count(web3.eth.default_account)
                    
                    # Estimate gas for the transaction
                    gas_estimate = contract.functions.createPath(
                        path["pathId"],
                        path["sourceNode"],
                        path["destinationNode"],
                        path["pathSequence"],
                        path["serviceClass"]
                    ).estimate_gas({'from': web3.eth.default_account})

                    # Build the transaction
                    tx = contract.functions.createPath(
                        path["pathId"],
                        path["sourceNode"],
                        path["destinationNode"],
                        path["pathSequence"],
                        path["serviceClass"]
                    ).build_transaction({
                        'from': web3.eth.default_account,
                        'nonce': nonce,
                        'gas': int(gas_estimate * 1.2),  # Add 20% buffer to gas estimate
                        'gasPrice': web3.eth.gas_price,
                        'chainId': web3.eth.chain_id
                    })

                    # Sign and send the transaction
                    signed_tx = web3.eth.account.sign_transaction(tx, private_key)
                    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    print(f"Transaction hash: {tx_hash.hex()}")
                    
                    # Wait for transaction receipt
                    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"Transaction status: {'Success' if tx_receipt['status'] == 1 else 'Failed'}")

            except Exception as path_creation_error:
                print(f"Error creating path {path['pathId']}: {path_creation_error}")

        # Query paths for each service class
        service_classes = ["VoIP", "Streaming", "Standard"]
        for service_class in service_classes:
            try:
                if use_simulation:
                    paths = analyzer.simulate_get_paths_by_service(service_class)
                else:
                    paths = contract.functions.getPathsByServiceClass(service_class).call()
                print(f"\nPaths for {service_class} service class: {paths}")
            except Exception as service_error:
                print(f"Error querying paths for service class {service_class}: {service_error}")

        # Query node performance
        sample_nodes = ["node1", "node2", "node3", "node4", "node5", 
    "node6", "node7", "node8", "node9", "node10",
    "node11", "node12", "node13", "node14", "node15",
    "node16", "node17", "node18", "node19", "node20",
    "node21", "node22", "node23", "node24", "node25"]
        for node in sample_nodes:
            try:
                if use_simulation:
                    node_performance = analyzer.simulate_get_node_performance(node)
                else:
                    node_performance = contract.functions.getNodePerformance(node).call()
                print(f"\nNode Performance for {node}:")
                print(f"Success Rate: {node_performance[0]}%")
                print(f"Packet Count: {node_performance[1]}")
            except Exception as performance_error:
                print(f"Error retrieving performance for {node}: {performance_error}")

        # Simulate transmission data for a few paths
        if use_simulation:
            for path_id in [{"path_id": "voip_path_1", "packets": 5000, "security": 3},
        {"path_id": "streaming_path_1", "packets": 8000, "security": 2},
        {"path_id": "standard_path_1", "packets": 3000, "security": 1},
        {"path_id": "voip_path_3", "packets": 6500, "security": 3},
        {"path_id": "streaming_path_2", "packets": 12000, "security": 2},
        {"path_id": "high_priority_path_1", "packets": 4200, "security": 4},
        {"path_id": "secure_path_1", "packets": 2800, "security": 5},
        {"path_id": "low_latency_path_1", "packets": 9500, "security": 2},
        {"path_id": "high_bandwidth_path_1", "packets": 15000, "security": 1}]:
                analyzer.simulate_path_transmission(
                    path_id=path_id,
                    packets_total=5000 + (hash(path_id) % 3000),
                    security_level=2
                )

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()