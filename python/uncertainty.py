import os
import sys
import json
import logging
from typing import Dict, Any, Optional
import time
from web3 import Web3, HTTPProvider
from dotenv import load_dotenv
import uuid
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
print("Loading environment variables...")
load_dotenv()

class UncertaintyAnalyticsManager:
    def __init__(
        self, 
        web3: Web3, 
        uncertainty_address: str,
        request_manager_address: str,
        response_manager_address: str,
        build_contracts_dir: Optional[str] = None,
        project_root: Optional[str] = None
    ):
        """Initialize Uncertainty Analytics Manager"""
        print("Initializing UncertaintyAnalyticsManager...")
        
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

        # Contract addresses
        self.uncertainty_address = Web3.to_checksum_address(uncertainty_address)
        self.request_manager_address = Web3.to_checksum_address(request_manager_address)
        self.response_manager_address = Web3.to_checksum_address(response_manager_address)
        
        # Contract instances
        self.uncertainty_contract = None
        self.request_manager_contract = None
        self.response_manager_contract = None
        
        # ABIs - Extracted directly from the contract files
        self.uncertainty_abi = self._get_uncertainty_analytics_abi()
        self.request_manager_abi = self._get_request_manager_abi()
        self.response_manager_abi = self._get_response_manager_abi()
        
        # Simulation data
        self.mock_requests = {}
        self.mock_responses = {}
        self.mock_processed_requests = set()  # Track processed requests for ResponseManager
        self.mock_responder_count = {}        # Track responder activity
        self.mock_metrics = {
            "avgProcessingTime": 0,           # seconds
            "successRate": 100,               # percentage
            "totalCost": Web3.to_wei(0, 'ether'),
            "disruptionCount": 0
        }
        self.next_request_id = 1
        
        # Constants from contracts
        self.BASE_COST = Web3.to_wei(0.001, 'ether')  # From UncertaintyBase
        self.MAX_PROCESSING_TIME = 86400              # 1 day in seconds
        
        print("UncertaintyAnalyticsManager initialized successfully")

    def _get_uncertainty_analytics_abi(self):
        """Get UncertaintyAnalytics contract ABI"""
        return [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "inputs": [],
                "name": "base",
                "outputs": [
                    {
                        "internalType": "contract UncertaintyBase",
                        "name": "",
                        "type": "address"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "costAnalytics",
                "outputs": [
                    {
                        "internalType": "contract CostAnalytics",
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
                        "internalType": "uint256",
                        "name": "_processingTime",
                        "type": "uint256"
                    }
                ],
                "name": "calculateUnavailabilityCost",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getMetrics",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "avgProcessingTime",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "successRate",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "totalCost",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "disruptionCount",
                        "type": "uint256"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "requestManager",
                "outputs": [
                    {
                        "internalType": "contract RequestManager",
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
                        "internalType": "uint256",
                        "name": "_requestId",
                        "type": "uint256"
                    },
                    {
                        "internalType": "string",
                        "name": "_reason",
                        "type": "string"
                    }
                ],
                "name": "recordFailedTransaction",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "submitRequest",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "",
                        "type": "uint256"
                    }
                ],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "uint256",
                        "name": "_requestId",
                        "type": "uint256"
                    }
                ],
                "name": "submitResponse",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "uint256",
                        "name": "_level",
                        "type": "uint256"
                    }
                ],
                "name": "updateDisruptionLevel",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "uint256",
                        "name": "_level",
                        "type": "uint256"
                    }
                ],
                "name": "updateEscalationLevel",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "uint256",
                        "name": "_cost",
                        "type": "uint256"
                    }
                ],
                "name": "updateDataHoldingCost",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "withdraw",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "stateMutability": "payable",
                "type": "receive"
            }
        ]

    def _get_request_manager_abi(self):
        """Get RequestManager contract ABI"""
        return [
            {
                "inputs": [
                    {
                        "internalType": "address",
                        "name": "_analytics",
                        "type": "address"
                    }
                ],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "recipient",
                        "type": "address"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "amount",
                        "type": "uint256"
                    }
                ],
                "name": "FundsWithdrawn",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "uint256",
                        "name": "requestId",
                        "type": "uint256"
                    },
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "requester",
                        "type": "address"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "value",
                        "type": "uint256"
                    }
                ],
                "name": "RequestSubmitted",
                "type": "event"
            },
            {
                "inputs": [],
                "name": "VERSION",
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
                "inputs": [],
                "name": "getAnalyticsMetrics",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "avgProcessingTime",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "successRate",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "totalCost",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "disruptionCount",
                        "type": "uint256"
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
                "name": "submitRequest",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "",
                        "type": "uint256"
                    }
                ],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "_additionalInfo",
                        "type": "string"
                    }
                ],
                "name": "submitRequestWithInfo",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "",
                        "type": "uint256"
                    }
                ],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "address",
                        "name": "_newOwner",
                        "type": "address"
                    }
                ],
                "name": "transferOwnership",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "withdraw",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "stateMutability": "payable",
                "type": "receive"
            }
        ]

    def _get_response_manager_abi(self):
        """Get ResponseManager contract ABI"""
        return [
            {
                "inputs": [
                    {
                        "internalType": "address",
                        "name": "_analytics",
                        "type": "address"
                    }
                ],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "recipient",
                        "type": "address"
                    },
                    {
                        "indexed": False,
                        "internalType": "uint256",
                        "name": "amount",
                        "type": "uint256"
                    }
                ],
                "name": "FundsWithdrawn",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {
                        "indexed": True,
                        "internalType": "uint256",
                        "name": "requestId",
                        "type": "uint256"
                    },
                    {
                        "indexed": True,
                        "internalType": "address",
                        "name": "responder",
                        "type": "address"
                    }
                ],
                "name": "ResponseSubmitted",
                "type": "event"
            },
            {
                "inputs": [],
                "name": "VERSION",
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
                "inputs": [],
                "name": "getAnalyticsMetrics",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "avgProcessingTime",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "successRate",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "totalCost",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "disruptionCount",
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
                        "name": "_responder",
                        "type": "address"
                    }
                ],
                "name": "getResponderCount",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "count",
                        "type": "uint256"
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
                "inputs": [
                    {
                        "internalType": "uint256",
                        "name": "",
                        "type": "uint256"
                    }
                ],
                "name": "processedRequests",
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
                        "internalType": "address",
                        "name": "",
                        "type": "address"
                    }
                ],
                "name": "responderCount",
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
                        "name": "_requestId",
                        "type": "uint256"
                    }
                ],
                "name": "submitResponse",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "uint256",
                        "name": "_requestId",
                        "type": "uint256"
                    }
                ],
                "name": "submitResponseWithCalculation",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "address",
                        "name": "_newOwner",
                        "type": "address"
                    }
                ],
                "name": "transferOwnership",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "withdraw",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "stateMutability": "payable",
                "type": "receive"
            }
        ]

    def load_contracts(self):
        """Load all contracts with verification"""
        print("Loading contracts...")
        
        if not self.w3.is_connected():
            raise ConnectionError("No Web3 connection available")

        # Create contract instances with predefined ABIs
        try:
            self.uncertainty_contract = self.w3.eth.contract(
                address=self.uncertainty_address, 
                abi=self.uncertainty_abi
            )
            
            self.request_manager_contract = self.w3.eth.contract(
                address=self.request_manager_address, 
                abi=self.request_manager_abi
            )
            
            self.response_manager_contract = self.w3.eth.contract(
                address=self.response_manager_address, 
                abi=self.response_manager_abi
            )

            print("\n--- Uncertainty Analytics Contracts Verification ---")
            print(f"UncertaintyAnalytics Address: {self.uncertainty_address}")
            print(f"RequestManager Address: {self.request_manager_address}")
            print(f"ResponseManager Address: {self.response_manager_address}")
            print(f"Network: {self.w3.eth.chain_id}")
            
            # Display function names from each contract
            for name, contract in [
                ("UncertaintyAnalytics", self.uncertainty_contract),
                ("RequestManager", self.request_manager_contract),
                ("ResponseManager", self.response_manager_contract)
            ]:
                function_names = [
                    func for func in dir(contract.functions)
                    if callable(getattr(contract.functions, func)) and not func.startswith('__')
                ]
                print(f"\n{name} Available Functions: {function_names}")
            
            # Verify contracts exist on blockchain
            for name, address in [
                ("UncertaintyAnalytics", self.uncertainty_address),
                ("RequestManager", self.request_manager_address),
                ("ResponseManager", self.response_manager_address)
            ]:
                contract_bytecode = self.w3.eth.get_code(address)
                bytecode_length = len(contract_bytecode)
                print(f"{name} Contract Bytecode Length: {bytecode_length}")

            print("All contracts loaded successfully")
            return {
                "uncertainty_contract": self.uncertainty_contract,
                "request_manager_contract": self.request_manager_contract,
                "response_manager_contract": self.response_manager_contract
            }

        except Exception as e:
            logger.error(f"Error loading contracts: {e}")
            print(f"Error loading contracts: {e}")
            raise
    
    def submit_request(self, value_in_eth: float, additional_info: str = "") -> int:
        """Submit a request to the UncertaintyAnalytics contract with payment"""
        print(f"Submitting request with value {value_in_eth} ETH...")
        value_in_wei = Web3.to_wei(value_in_eth, 'ether')
        
        try:
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.w3.eth.default_account)
            print(f"Got nonce: {nonce}")
            
            # Prepare transaction parameters
            tx_params = {
                'from': self.w3.eth.default_account,
                'value': value_in_wei,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            }
            print("Transaction parameters prepared")
            
            if additional_info:
                # Use RequestManager's submitRequestWithInfo
                print(f"Using submitRequestWithInfo with info: {additional_info}")
                tx = self.request_manager_contract.functions.submitRequestWithInfo(
                    additional_info
                ).build_transaction(tx_params)
            else:
                # Use standard submitRequest
                print("Using standard submitRequest")
                tx = self.request_manager_contract.functions.submitRequest().build_transaction(tx_params)
            
            # Get private key from environment variables
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                raise ValueError("PRIVATE_KEY not found in environment variables")
            
            # Add '0x' prefix if it's missing
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            print("Got private key from environment")
                
            # Sign and send transaction
            print("Signing transaction...")
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            print("Sending transaction...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            print(f"Transaction hash: {tx_hash.hex()}")
            
            # Wait for transaction receipt
            print("Waiting for transaction receipt...")
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"Transaction status: {'Success' if tx_receipt['status'] == 1 else 'Failed'}")
            
            # Parse event logs to get the request ID
            if tx_receipt['status'] == 1:
                print("Transaction successful. Looking for RequestSubmitted event...")
                # Look for RequestSubmitted event
                event_signature = self.request_manager_contract.events.RequestSubmitted().event_signature_hash
                for log in tx_receipt['logs']:
                    if log['topics'][0].hex() == event_signature:
                        # Parse the logs to get the request ID
                        parsed_logs = self.request_manager_contract.events.RequestSubmitted().process_receipt(tx_receipt)
                        request_id = parsed_logs[0]['args']['requestId']
                        print(f"Request ID: {request_id}")
                        return request_id
            
            return 0  # Request failed
            
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            print(f"Error submitting request: {e}")
            raise
    
    def submit_response(self, request_id: int, with_calculation: bool = False) -> bool:
        """Submit a response to a request via the ResponseManager contract"""
        print(f"Submitting response for request ID {request_id}...")
        try:
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.w3.eth.default_account)
            print(f"Got nonce: {nonce}")
            
            # Prepare transaction parameters
            tx_params = {
                'from': self.w3.eth.default_account,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            }
            print("Transaction parameters prepared")
            
            if with_calculation:
                # Use submitResponseWithCalculation
                print("Using submitResponseWithCalculation")
                tx = self.response_manager_contract.functions.submitResponseWithCalculation(
                    request_id
                ).build_transaction(tx_params)
            else:
                # Use standard submitResponse
                print("Using standard submitResponse")
                tx = self.response_manager_contract.functions.submitResponse(
                    request_id
                ).build_transaction(tx_params)
            
            # Get private key from environment variables
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                raise ValueError("PRIVATE_KEY not found in environment variables")
            
            # Add '0x' prefix if it's missing
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            print("Got private key from environment")
                
            # Sign and send transaction
            print("Signing transaction...")
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            print("Sending transaction...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            print(f"Transaction hash: {tx_hash.hex()}")
            
            # Wait for transaction receipt
            print("Waiting for transaction receipt...")
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"Transaction status: {'Success' if tx_receipt['status'] == 1 else 'Failed'}")
            
            return tx_receipt['status'] == 1
            
        except Exception as e:
            logger.error(f"Error submitting response: {e}")
            print(f"Error submitting response: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from UncertaintyAnalytics contract"""
        print("Getting metrics from UncertaintyAnalytics contract...")
        try:
            metrics = self.uncertainty_contract.functions.getMetrics().call()
            
            result = {
                "avgProcessingTime": metrics[0],
                "successRate": metrics[1],
                "totalCost": metrics[2],
                "disruptionCount": metrics[3]
            }
            
            print(f"\n--- Analytics Metrics ---")
            print(f"Average Processing Time: {result['avgProcessingTime']} seconds")
            print(f"Success Rate: {result['successRate']}%")
            print(f"Total Cost: {Web3.from_wei(result['totalCost'], 'ether')} ETH")
            print(f"Disruption Count: {result['disruptionCount']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            print(f"Error getting metrics: {e}")
            raise
    
    def get_responder_count(self, responder_address: str) -> int:
        """Get the number of responses submitted by an address"""
        print(f"Getting responder count for address {responder_address}...")
        try:
            count = self.response_manager_contract.functions.getResponderCount(
                Web3.to_checksum_address(responder_address)
            ).call()
            
            print(f"\n--- Responder Stats ---")
            print(f"Responder: {responder_address}")
            print(f"Response Count: {count}")
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting responder count: {e}")
            print(f"Error getting responder count: {e}")
            raise
    
    def record_failed_transaction(self, request_id: int, reason: str) -> bool:
        """Record a failed transaction"""
        print(f"Recording failed transaction for request ID {request_id} with reason: {reason}")
        try:
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.w3.eth.default_account)
            print(f"Got nonce: {nonce}")
            
            # Prepare transaction parameters
            tx_params = {
                'from': self.w3.eth.default_account,
                'nonce': nonce,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price
            }
            print("Transaction parameters prepared")
            
            tx = self.uncertainty_contract.functions.recordFailedTransaction(
                request_id, reason
            ).build_transaction(tx_params)
            
            # Get private key from environment variables
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                raise ValueError("PRIVATE_KEY not found in environment variables")
            
            # Add '0x' prefix if it's missing
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            print("Got private key from environment")
                
            # Sign and send transaction
            print("Signing transaction...")
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            print("Sending transaction...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            print(f"Transaction hash: {tx_hash.hex()}")
            
            # Wait for transaction receipt
            print("Waiting for transaction receipt...")
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"Transaction status: {'Success' if tx_receipt['status'] == 1 else 'Failed'}")
            
            return tx_receipt['status'] == 1
            
        except Exception as e:
            logger.error(f"Error recording failed transaction: {e}")
            print(f"Error recording failed transaction: {e}")
            raise
    
    def simulate_submit_request(self, value_in_eth: float, additional_info: str = "") -> Dict[str, Any]:
        """Simulate submitting a request with a payment"""
        print(f"Simulating request with value {value_in_eth} ETH...")
        
        # Verify value meets requirements
        if value_in_eth * 10**18 < self.BASE_COST:
            print(f"Error: Value must be at least {Web3.from_wei(self.BASE_COST, 'ether')} ETH")
            return None
        
        request_id = self.next_request_id
        self.next_request_id += 1
        
        value_in_wei = Web3.to_wei(value_in_eth, 'ether')
        timestamp = int(time.time())
        
        request = {
            "id": request_id,
            "requester": self.w3.eth.default_account,
            "timestamp": timestamp,
            "confirmationTime": 0,
            "executionTime": 0,
            "value": value_in_wei,
            "additional_info": additional_info,
            "status": "Pending",
            "is_processed": False,
            "is_valid": True
        }
        
        self.mock_requests[request_id] = request
        self.mock_metrics["totalCost"] += value_in_wei
        
        print(f"\n--- Request Submitted ---")
        print(f"Request ID: {request_id}")
        print(f"Requester: {self.w3.eth.default_account}")
        print(f"Timestamp: {timestamp}")
        print(f"Value: {value_in_eth} ETH")
        if additional_info:
            print(f"Additional Info: {additional_info}")
            
        return request
    
    def simulate_submit_response(self, request_id: int, with_calculation: bool = False) -> Dict[str, Any]:
        """Simulate submitting a response to a request"""
        print(f"Simulating response for request ID {request_id}...")
        
        if request_id not in self.mock_requests:
            print(f"Error: Request ID {request_id} not found")
            return None
        
        request = self.mock_requests[request_id]
        
        # Check if the request has already been processed (as per ResponseManager contract)
        if request_id in self.mock_processed_requests:
            print(f"Error: Request ID {request_id} already processed")
            return None
        
        if not request["is_valid"]:
            print(f"Error: Request ID {request_id} is not valid")
            return None
            
        if request["status"] != "Pending":
            print(f"Error: Request ID {request_id} is not pending (status: {request['status']})")
            return None
            
        timestamp = int(time.time())
        processing_time = timestamp - request["timestamp"]
        
        response = {
            "request_id": request_id,
            "responder": self.w3.eth.default_account,
            "timestamp": timestamp,
            "processing_time": processing_time,
            "status": "Completed",
            "is_valid": True
        }
        
        # Mark the request as processed
        request["is_processed"] = True
        request["status"] = "Completed"
        request["confirmationTime"] = timestamp
        request["executionTime"] = processing_time
        
        # Store the response
        self.mock_responses[request_id] = response
        
        # Mark as processed in ResponseManager tracking
        self.mock_processed_requests.add(request_id)
        
        # Update responder count (as per ResponseManager contract)
        responder = self.w3.eth.default_account
        self.mock_responder_count[responder] = self.mock_responder_count.get(responder, 0) + 1
        
        print(f"\n--- Response Submitted ---")
        print(f"Request ID: {request_id}")
        print(f"Responder: {responder}")
        print(f"Timestamp: {timestamp}")
        print(f"Processing Time: {processing_time} seconds")
        
        if with_calculation:
            unavailability_cost = 0
            if processing_time > self.MAX_PROCESSING_TIME:  # 1 day in seconds
                # Calculate penalty as per contract logic
                penalty = ((processing_time - self.MAX_PROCESSING_TIME) * self.BASE_COST) // 86400
                unavailability_cost = penalty
                print(f"Calculated Unavailability Cost: {Web3.from_wei(unavailability_cost, 'ether')} ETH")
                response["unavailability_cost"] = unavailability_cost
                
                # Update metrics
                self.mock_metrics["totalCost"] += unavailability_cost
            
        return response
    
    def simulate_get_metrics(self) -> Dict[str, Any]:
        """Simulate getting metrics from the UncertaintyAnalytics contract"""
        print("Simulating getting metrics...")
        
        # Calculate metrics based on mock data
        if self.mock_responses:
            # Calculate average processing time for valid responses
            total_time = 0
            valid_responses = 0
            
            for response in self.mock_responses.values():
                if response["is_valid"]:
                    total_time += response["processing_time"]
                    valid_responses += 1
            
            if valid_responses > 0:
                self.mock_metrics["avgProcessingTime"] = total_time // valid_responses
            
            # Calculate success rate if there are any requests
            if self.mock_requests:
                # Count failed requests
                failed_count = sum(1 for req in self.mock_requests.values() if req["status"] == "Failed")
                total_requests = len(self.mock_requests)
                
                if total_requests > 0:
                    self.mock_metrics["successRate"] = ((total_requests - failed_count) * 100) // total_requests
                    self.mock_metrics["disruptionCount"] = failed_count
        
        print(f"\n--- Analytics Metrics ---")
        print(f"Average Processing Time: {self.mock_metrics['avgProcessingTime']} seconds")
        print(f"Success Rate: {self.mock_metrics['successRate']}%")
        print(f"Total Cost: {Web3.from_wei(self.mock_metrics['totalCost'], 'ether')} ETH")
        print(f"Disruption Count: {self.mock_metrics['disruptionCount']}")
        
        return self.mock_metrics
    
    def simulate_get_responder_count(self, responder_address: str) -> int:
        """Simulate getting the count of responses from a specific responder"""
        print(f"Simulating getting responder count for address {responder_address}...")
        
        count = self.mock_responder_count.get(responder_address, 0)
        
        print(f"\n--- Responder Stats ---")
        print(f"Responder: {responder_address}")
        print(f"Response Count: {count}")
        
        return count
    
    def simulate_record_failed_transaction(self, request_id: int, reason: str) -> bool:
        """Simulate recording a failed transaction"""
        print(f"Simulating recording failed transaction for request ID {request_id} with reason: {reason}")
        
        if request_id not in self.mock_requests:
            print(f"Error: Request ID {request_id} not found")
            return False
            
        request = self.mock_requests[request_id]
        if not request["is_valid"]:
            print(f"Error: Request ID {request_id} is not valid")
            return False
            
        if request["status"] == "Failed":
            print(f"Error: Request ID {request_id} already marked as failed")
            return False
            
        request["status"] = "Failed"
        self.mock_metrics["disruptionCount"] += 1
        
        print(f"\n--- Transaction Failed ---")
        print(f"Request ID: {request_id}")
        print(f"Reason: {reason}")
        
        return True
    
    def simulate_update_disruption_level(self, level: int) -> bool:
        """Simulate updating the disruption level"""
        print(f"Simulating updating disruption level to {level}")
        print(f"\n--- Disruption Level Updated ---")
        print(f"New Level: {level}")
        return True
    
    def simulate_update_escalation_level(self, level: int) -> bool:
        """Simulate updating the escalation level"""
        print(f"Simulating updating escalation level to {level}")
        print(f"\n--- Escalation Level Updated ---")
        print(f"New Level: {level}")
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
    Main function to interact with Uncertainty Analytics contracts
    """
    try:
        # Contract addresses from contract_addresses.json
        UNCERTAINTY_ANALYTICS_ADDRESS = "0xafb69d3380aa2a892625665803fca627fd65ec0f"  # Example address
        REQUEST_MANAGER_ADDRESS = "0xc5491f090181c8653ec0228d07499a51d7bf12bd"        # Example address
        RESPONSE_MANAGER_ADDRESS = "0xfda50ab71b0e577680c4afe29fdc2272ab19d89b"       # Example address

        # Create Web3 connection
        web3 = create_web3_connection("sepolia")
        if not web3:
            print("Could not establish blockchain connection. Exiting.")
            sys.exit(1)

        # Initialize the manager
        manager = UncertaintyAnalyticsManager(
            web3=web3, 
            uncertainty_address=UNCERTAINTY_ANALYTICS_ADDRESS,
            request_manager_address=REQUEST_MANAGER_ADDRESS,
            response_manager_address=RESPONSE_MANAGER_ADDRESS,
        )

        # Use simulation mode flag
        use_simulation = True

        # Load contracts (either with real ABIs or the default ones)
        try:
            contracts = manager.load_contracts()
            print("Contracts loaded successfully")
        except ValueError as e:
            print(f"Warning: {e}")
            print("Continuing with simulation only...")
            use_simulation = True

        # Sample request and response simulation scenarios
        if use_simulation:
            # Simulate submitting requests
            print("\n=== Simulating Requests ===")
            requests = [
                {"value": 0.002, "info": "High priority processing"},
                {"value": 0.001, "info": ""},
                {"value": 0.005, "info": "Requires detailed analytics"},
                {"value": 0.0015, "info": "Standard processing"},
                {"value": 0.003, "info": "Time-sensitive data"},
                {"value": 0.002, "info": "High priority processing"},
                {"value": 0.001, "info": "Basic system check"},
                {"value": 0.005, "info": "Requires detailed analytics"},
                {"value": 0.0015, "info": "Standard processing"},
                {"value": 0.003, "info": "Time-sensitive data"},
                {"value": 0.0025, "info": "Network performance monitoring"},
                {"value": 0.0035, "info": "Security protocol evaluation"},
                {"value": 0.0018, "info": "Data integrity verification"},
                {"value": 0.004, "info": "Complex system diagnostic"},
                {"value": 0.0012, "info": "Routine maintenance check"},
                {"value": 0.006, "info": "Advanced algorithmic analysis"},
                {"value": 0.0022, "info": "Performance benchmarking"},
                {"value": 0.0045, "info": "Critical infrastructure review"},
                {"value": 0.0017, "info": "Preliminary system scan"},
                {"value": 0.0038, "info": "Comprehensive network audit"}
                
            ]
            
            request_ids = []
            for req in requests:
                request = manager.simulate_submit_request(
                    value_in_eth=req["value"],
                    additional_info=req["info"]
                )
                if request:
                    request_ids.append(request["id"])
            
            # Simulate responses
            print("\n=== Simulating Responses ===")
            # Respond to 3 out of 5 requests
            for request_id in request_ids[:3]:
                manager.simulate_submit_response(
                    request_id=request_id,
                    with_calculation=(request_id % 2 == 0)  # Alternate between with and without calculation
                )
            
            # Simulate a failed transaction
            print("\n=== Simulating Failed Transaction ===")
            manager.simulate_record_failed_transaction(
                request_id=request_ids[3],
                reason="Network congestion"
            )
            
            # Get metrics
            print("\n=== Simulating Analytics Metrics ===")
            metrics = manager.simulate_get_metrics()
            
            # Get responder count
            print("\n=== Simulating Responder Stats ===")
            responder_count = manager.simulate_get_responder_count(web3.eth.default_account)
            
            # Update disruption and escalation levels
            print("\n=== Simulating Level Updates ===")
            manager.simulate_update_disruption_level(2)
            manager.simulate_update_escalation_level(1)
            
            print("\n=== Simulation Summary ===")
            print(f"Total Requests: {len(request_ids)}")
            print(f"Completed Responses: {len(manager.mock_responses)}")
            print(f"Failed Transactions: {manager.mock_metrics['disruptionCount']}")
            print(f"Average Processing Time: {manager.mock_metrics['avgProcessingTime']} seconds")
            print(f"Success Rate: {manager.mock_metrics['successRate']}%")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()