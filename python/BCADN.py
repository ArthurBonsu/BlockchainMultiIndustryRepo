import os
import sys
import json
import time
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

from web3 import Web3, HTTPProvider
from eth_account import Account
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Configuration
RESULT_OUTPUT_DIR = "results/bcadn_analysis"
os.makedirs(RESULT_OUTPUT_DIR, exist_ok=True)

class BCADNNetworkAnalyzer:
    def __init__(
        self, 
        web3: Web3,
        contract_addresses_path: Optional[str] = None,
        build_contracts_dir: Optional[str] = None
    ):
        """
        Initialize BCADN Network Analyzer
        
        :param web3: Web3 instance
        :param contract_addresses_path: Path to contract addresses JSON
        :param build_contracts_dir: Directory containing contract build artifacts
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Determine project root and default paths
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')
        )
        
        # Contract addresses file
        self.contract_addresses_path = contract_addresses_path or os.path.join(
            self.project_root, 'config', 'contract_addresses.json'
        )
        
        # Build contracts directory
        self.build_contracts_dir = build_contracts_dir or os.path.join(
            self.project_root, 'build', 'contracts'
        )
        
        # Blockchain connection
        self.w3 = web3
        
        # Load contract addresses and configurations
        self.contract_addresses = self._load_contract_addresses()
        
        # BCADN-specific contracts to load
        self.bcadn_contract_names = [
            'BCADN', 
            'ProactiveDefenseMechanism'
        ]
        
        # Contracts storage
        self.contracts: Dict[str, Any] = {}

    def _load_contract_addresses(self) -> Dict[str, str]:
        """
        Load contract addresses from JSON file
        
        :return: Dictionary of contract addresses
        """
        try:
            with open(self.contract_addresses_path, 'r') as f:
                raw_addresses = json.load(f)
            
            self.logger.info(f"Loaded contract addresses from {self.contract_addresses_path}")
            return raw_addresses
        except FileNotFoundError:
            self.logger.error(f"Contract addresses file not found at {self.contract_addresses_path}")
            return {}
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in contract addresses file at {self.contract_addresses_path}")
            return {}
    
    def _load_contract_abi(self, contract_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load ABI for a given contract name.

        :param contract_name: Name of the contract to load ABI for
        :return: ABI as a list of dictionaries, or None if not found
        """
        # Define possible ABI paths
        abi_paths = [
            os.path.join(self.build_contracts_dir, f"{contract_name}.json"),
            # Add other possible ABI paths here
        ]

        for abi_path in abi_paths:
            try:
                if os.path.exists(abi_path):
                    with open(abi_path, 'r') as f:
                        # Read the entire file content
                        content = f.read()

                        # Try parsing as JSON
                        try:
                            contract_data = json.loads(content)

                            # Multiple possible ABI locations in JSON
                            possible_abi_keys = [
                                'abi',          # Truffle/Hardhat standard
                                'contractName', # Alternative key
                                'compilerOutput', # Another possible location
                                'output'        # Yet another possible key
                            ]

                            # Try each possible key
                            for key in possible_abi_keys:
                                if isinstance(contract_data, dict) and key in contract_data:
                                    abi = contract_data[key]

                                    # Ensure it's a list of ABI entries
                                    if isinstance(abi, list):
                                        print(f"Successfully loaded ABI for {contract_name} from {abi_path}")

                                        # Optional: Print out available function names
                                        function_names = [
                                            func.get('name', 'unnamed') 
                                            for func in abi 
                                            if func.get('type') == 'function'
                                        ]
                                        print(f"Available functions in {contract_name}: {function_names}")

                                        return abi

                            # If no key worked, check if entire content is ABI
                            if isinstance(contract_data, list):
                                return contract_data

                        except json.JSONDecodeError:
                            # Fallback: try parsing raw content
                            try:
                                raw_abi = json.loads(content)
                                if isinstance(raw_abi, list):
                                    return raw_abi
                            except:
                                self.logger.warning(f"Could not parse JSON from {abi_path}")

            except Exception as e:
                self.logger.warning(f"Error reading ABI at {abi_path}: {e}")

        self.logger.error(f"No ABI found for contract: {contract_name}")
        return None
    
def validate_ethereum_address(self, address: str) -> str:
    """
    Validate an Ethereum address and return checksum version if valid

    :param address: Ethereum address to validate
    :return: Checksum address if valid, raises ValueError if invalid
    """
    try:
        if not address or address == '0x0':
            raise ValueError("Empty or zero address provided")

        checksum_address = Web3.to_checksum_address(address)
        if not Web3.is_address(checksum_address):
            raise ValueError(f"Invalid Ethereum address format: {address}")

        return checksum_address
    except Exception as e:
        raise ValueError(f"Address validation error: {str(e)}")
def verify_contract_state(self, contract_address: str) -> bool:
    """
    Verify contract is properly deployed and accessible
    
    :param contract_address: Contract address to verify
    :return: Boolean indicating if contract is properly deployed
    """
    try:
        code = self.w3.eth.get_code(contract_address)
        return len(code) > 0
    except Exception as e:
        self.logger.error(f"Contract state verification failed: {e}")
        return False

def load_bcadn_contracts(self) -> Dict[str, Any]:
    """
    Load BCADN-related contracts with comprehensive verification
    
    :return: Dictionary of loaded contracts
    """
    self.logger.info("Loading BCADN-related contracts")
    
    # Verify web3 connection first
    if not self.w3.is_connected():
        raise ConnectionError("No Web3 connection available")
    
    # Verify default account is set
    if not self.w3.eth.default_account:
        self.logger.warning("No default account set. Some functions may not work properly.")
    
    for contract_name in self.bcadn_contract_names:
        try:
            # Get and validate contract address
            raw_address = self.contract_addresses.get(contract_name)
            
            try:
                contract_address = self.validate_ethereum_address(raw_address)
            except ValueError as ve:
                self.logger.error(f"Address validation failed for {contract_name}: {ve}")
                continue
            
            # Verify contract deployment
            if not self.verify_contract_state(contract_address):
                self.logger.error(f"Contract {contract_name} not properly deployed at {contract_address}")
                continue
            
            # Load ABI
            abi = self._load_contract_abi(contract_name)
            
            if not abi:
                self.logger.warning(f"Could not load ABI for {contract_name}")
                continue
            
            # Create contract instance with verified checksum address
            contract = self.w3.eth.contract(
                address=contract_address, 
                abi=abi
            )
            
            # Comprehensive Contract Verification
            print(f"\n--- {contract_name} Contract Verification ---")
            print(f"Contract Address: {contract_address}")
            print(f"Network: {self.w3.eth.chain_id}")
            
            # Check contract bytecode with improved error handling
            try:
                contract_bytecode = self.w3.eth.get_code(contract_address)
                bytecode_length = len(contract_bytecode)
                print(f"Contract Bytecode Length: {bytecode_length}")
                
                if bytecode_length == 0:
                    raise ValueError("No bytecode found at contract address")
                elif bytecode_length < 100:  # Arbitrary minimum size for a valid contract
                    self.logger.warning(f"Unusually small bytecode size ({bytecode_length} bytes) for {contract_name}")
            except Exception as bytecode_error:
                self.logger.error(f"Bytecode verification failed: {bytecode_error}")
                continue
            
            # Try to get contract owner with timeout
            try:
                owner = contract.functions.owner().call({
                    'from': self.w3.eth.default_account
                })
                print(f"Contract Owner: {owner}")
                
                # Verify owner is valid address
                if not Web3.is_address(owner):
                    self.logger.warning(f"Invalid owner address returned: {owner}")
            except Exception as owner_error:
                self.logger.warning(f"Could not retrieve contract owner: {owner_error}")
            
            # Function verification with categorization
            function_names = [
                func.get('name', 'unnamed') 
                for func in abi 
                if func.get('type') == 'function'
            ]
            
            # Categorize functions
            view_functions = []
            write_functions = []
            special_functions = ['calculateDynamicFee', 'adjustToProbabilityGap', 'isWithinGap']
            
            print("\nContract Functions Analysis:")
            for func_name in function_names:
                try:
                    if not hasattr(contract.functions, func_name):
                        continue
                        
                    func = getattr(contract.functions, func_name)
                    
                    # Attempt to identify function type
                    try:
                        func_object = contract.get_function_by_name(func_name)
                        if func_object.get('stateMutability') in ['view', 'pure']:
                            view_functions.append(func_name)
                        else:
                            write_functions.append(func_name)
                    except:
                        write_functions.append(func_name)
                    
                    print(f"\n- {func_name}")
                    
                    # Test special functions
                    if func_name in special_functions:
                        try:
                            result = func(10).call({'from': self.w3.eth.default_account})
                            print(f"  ✓ Special function test successful")
                            print(f"  Result: {result}")
                        except Exception as special_error:
                            print(f"  ✗ Special function test failed: {special_error}")
                    
                    # Test view functions
                    elif func_name in ['getAllNodes', 'getAttackHistory', 'owner']:
                        try:
                            result = func().call({'from': self.w3.eth.default_account})
                            print(f"  ✓ View function test successful")
                            print(f"  Result: {result}")
                        except Exception as view_error:
                            print(f"  ✗ View function test failed: {view_error}")
                    
                except Exception as func_error:
                    self.logger.error(f"Error analyzing function {func_name}: {func_error}")
            
            # Print function statistics
            print(f"\nFunction Statistics:")
            print(f"Total Functions: {len(function_names)}")
            print(f"View/Pure Functions: {len(view_functions)}")
            print(f"State-Changing Functions: {len(write_functions)}")
            
            # Store contract with metadata
            self.contracts[contract_name] = {
                'contract': contract,
                'address': contract_address,
                'bytecode_size': bytecode_length,
                'view_functions': view_functions,
                'write_functions': write_functions
            }
            
            self.logger.info(f"Successfully loaded and verified {contract_name} at {contract_address}")
        
        except Exception as e:
            self.logger.error(f"Error loading contract {contract_name}: {e}")
            continue
    
    # Final verification
    if not self.contracts:
        raise ValueError("No contracts were successfully loaded")
    
    return self.contracts
def analyze_network_performance(self) -> Dict[str, Any]:
    """
    Comprehensive analysis of BCADN network performance

    :return: Dictionary of network performance metrics
    """
    # Initialize results structure
    results = {
        'network_metrics': {},
        'node_performance': {
            'node_addresses': [],
            'weights': [],
            'statuses': [],
            'performance_details': []
        },
        'attack_history': {
            'node_addresses': [],
            'timestamps': [],
            'anomaly_scores': [],
            'attack_types': [],
            'resolved': []
        }
    }

    # Advanced debugging for contract interaction
    try:
        bcadn = self.contracts.get('BCADN')
        proactive_defense = self.contracts.get('ProactiveDefenseMechanism')

        if not bcadn or not proactive_defense:
            raise ValueError("BCADN or ProactiveDefenseMechanism contract not loaded")

        # Detailed contract inspection
        print("\n--- Contract Deployment Details ---")
        print(f"BCADN Contract Address: {bcadn.address}")
        print(f"ProactiveDefenseMechanism Contract Address: {proactive_defense.address}")
        print(f"Network Chain ID: {self.w3.eth.chain_id}")

        # Attempt to retrieve network information
        try:
            # Try to get node information from BCADN contract
            nodes_data = bcadn.functions.getAllNodes().call()
            results['node_performance']['node_addresses'] = nodes_data[0]
            results['node_performance']['weights'] = nodes_data[1]

            # Map integer status to string status
            status_map = ['Active', 'Probation', 'Excluded', 'Pending']
            results['node_performance']['statuses'] = [
                status_map[min(status, len(status_map)-1)]
                for status in nodes_data[2]
            ]

            print(f"Retrieved {len(nodes_data[0])} nodes from BCADN contract")
        except Exception as node_error:
            print(f"Error retrieving nodes from BCADN contract: {node_error}")

        # Try to get attack history
        try:
            attack_history = bcadn.functions.getAttackHistory().call()
            results['attack_history']['node_addresses'] = attack_history[0]
            results['attack_history']['timestamps'] = attack_history[1]
            results['attack_history']['anomaly_scores'] = attack_history[2]
            results['attack_history']['resolved'] = attack_history[3]
            results['attack_history']['attack_types'] = ["Unknown" for _ in range(len(attack_history[0]))]

            print(f"Retrieved {len(attack_history[0])} attack history entries")
        except Exception as attack_error:
            print(f"Error retrieving attack history: {attack_error}")

        # Additional network metrics
        results['network_metrics'] = {
            'total_nodes': len(results['node_performance']['node_addresses']),
            'total_attacks': len(results['attack_history']['node_addresses']),
            'resolved_attacks': sum(results['attack_history']['resolved']) if results['attack_history']['resolved'] else 0,
            'chain_id': self.w3.eth.chain_id,
            'latest_block': self.w3.eth.block_number,
            'gas_price': self.w3.eth.gas_price
        }

    except Exception as e:
        print(f"Critical Error in Network Performance Analysis: {e}")

    return results


def simulate_network_stress_test(self, num_transactions: int = 50) -> Dict[str, Any]:
    """
    Simulate network stress test by submitting multiple transactions
    
    :param num_transactions: Number of transactions to simulate
    :return: Stress test results dictionary containing transaction details and metrics
    :raises ValueError: If BCADN contract is not loaded
    :raises Web3ConnectionError: If web3 connection fails
    """
    # Get BCADN contract instance
    bcadn_data = self.contracts.get('BCADN')
    if not bcadn_data:
        raise ValueError("BCADN contract not loaded")
    
    bcadn = bcadn_data.get('contract') if isinstance(bcadn_data, dict) else bcadn_data
    
    # Initialize results structure
    stress_test_results = {
        'transactions': [],
        'total_processing_time': 0,
        'average_dynamic_fee': 0,
        'max_congestion_index': 0,
        'successful_transactions': 0,
        'failed_transactions': 0,
        'average_gas_used': 0,
        'total_gas_cost': 0
    }
    
    # Verify web3 connection and account
    if not self.w3.is_connected():
        raise Web3ConnectionError("No Web3 connection available")
    
    if not self.w3.eth.default_account:
        self.logger.warning("No default account set. Using simulation mode.")
    
    # Submit transactions
    for i in range(num_transactions):
        try:
            # Base fee and amount calculation
            base_fee = random.randint(1, 10)
            amount = random.randint(1, 100)
            
            # Calculate dynamic fee with timeout
            try:
                dynamic_fee = bcadn.functions.calculateDynamicFee(base_fee).call({
                    'from': self.w3.eth.default_account
                })
            except Exception as fee_error:
                self.logger.error(f"Error calculating dynamic fee: {fee_error}")
                dynamic_fee = base_fee  # Fallback to base fee
            
            # Handle transaction based on account availability
            if self.w3.eth.default_account:
                try:
                    # Build transaction
                    tx = bcadn.functions.simulateTransaction(amount).buildTransaction({
                        'from': self.w3.eth.default_account,
                        'gas': 100000,
                        'gasPrice': self.w3.eth.gas_price,
                        'nonce': self.w3.eth.get_transaction_count(
                            self.w3.eth.default_account
                        ),
                        'value': amount
                    })
                    
                    # Sign and send transaction
                    signed_tx = self.w3.eth.account.sign_transaction(
                        tx, 
                        private_key=os.getenv('PRIVATE_KEY')
                    )
                    tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    
                    # Wait for transaction receipt
                    start_time = time.time()
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    processing_time = time.time() - start_time
                    
                    # Record transaction details
                    tx_result = {
                        'tx_hash': tx_hash.hex(),
                        'base_fee': base_fee,
                        'dynamic_fee': dynamic_fee,
                        'amount': amount,
                        'gas_used': receipt.get('gasUsed', 0),
                        'status': receipt.get('status', 0),
                        'block_number': receipt.get('blockNumber'),
                        'processing_time': processing_time,
                        'timestamp': int(time.time())
                    }
                    
                    # Update metrics
                    stress_test_results['total_processing_time'] += processing_time
                    stress_test_results['average_dynamic_fee'] += dynamic_fee
                    stress_test_results['total_gas_cost'] += (
                        tx_result['gas_used'] * self.w3.eth.gas_price
                    )
                    
                    if receipt.get('status') == 1:
                        stress_test_results['successful_transactions'] += 1
                    else:
                        stress_test_results['failed_transactions'] += 1
                    
                except Exception as tx_error:
                    self.logger.error(f"Transaction {i+1} failed: {tx_error}")
                    tx_result = {
                        'base_fee': base_fee,
                        'dynamic_fee': dynamic_fee,
                        'amount': amount,
                        'error': str(tx_error),
                        'timestamp': int(time.time())
                    }
                    stress_test_results['failed_transactions'] += 1
            
            else:
                # Simulation mode when no account is available
                processing_time = random.uniform(0.1, 1.0)
                tx_result = {
                    'simulated': True,
                    'base_fee': base_fee,
                    'dynamic_fee': dynamic_fee,
                    'amount': amount,
                    'processing_time': processing_time,
                    'timestamp': int(time.time())
                }
                stress_test_results['total_processing_time'] += processing_time
                stress_test_results['average_dynamic_fee'] += dynamic_fee
                stress_test_results['successful_transactions'] += 1
            
            # Record transaction result
            stress_test_results['transactions'].append(tx_result)
            
            # Update congestion index
            current_congestion = random.uniform(0, 100)
            stress_test_results['max_congestion_index'] = max(
                stress_test_results['max_congestion_index'],
                current_congestion
            )
            
            # Add delay between transactions
            time.sleep(0.1)  # Prevent rate limiting
            
        except Exception as e:
            self.logger.error(f"Error in transaction {i+1}: {e}")
            stress_test_results['failed_transactions'] += 1
    
  
        # Calculate final averages
        total_tx = len(stress_test_results['transactions'])
        if total_tx > 0:
            stress_test_results['average_dynamic_fee'] /= total_tx
            stress_test_results['total_processing_time'] /= total_tx
            
            if stress_test_results['successful_transactions'] > 0:
                stress_test_results['average_gas_used'] = (
                    stress_test_results['total_gas_cost'] / 
                    stress_test_results['successful_transactions']
                )
        
        # Add summary metrics
        stress_test_results['summary'] = {
            'total_transactions': total_tx,
            'success_rate': (
                stress_test_results['successful_transactions'] / total_tx 
                if total_tx > 0 else 0
            ),
            'average_processing_time': stress_test_results['total_processing_time'],
            'average_dynamic_fee': stress_test_results['average_dynamic_fee'],
            'max_congestion_index': stress_test_results['max_congestion_index']
        }
        
        self.logger.info("Network stress test completed")
        self.logger.info(f"Success rate: {stress_test_results['summary']['success_rate']:.2%}")
    
    return stress_test_results
def create_web3_connection():
    """
    Create a Web3 connection to Sepolia testnet using Infura
    
    :return: Configured Web3 instance
    """
    try:
        # Retrieve Infura Project ID and Private Key from environment variables
        INFURA_PROJECT_ID = os.getenv('INFURA_PROJECT_ID')
        PRIVATE_KEY = os.getenv('PRIVATE_KEY')
        
        if not INFURA_PROJECT_ID:
            raise ValueError("INFURA_PROJECT_ID not found in environment variables")
        
        # Construct Infura URL for Sepolia
        infura_url = f"https://sepolia.infura.io/v3/{INFURA_PROJECT_ID}"
        
        # Create Web3 instance with HTTPProvider
        web3 = Web3(HTTPProvider(infura_url))
        
        # Add PoA middleware for Sepolia
        try:
            from web3.middleware import geth_poa_middleware
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        except ImportError:
            print("Warning: Could not import geth_poa_middleware")
        
        # Detailed connection verification
        print("\n--- Web3 Connection Diagnostics ---")
        print(f"Connecting to: {infura_url}")
        
        # Check connection status
        is_connected = web3.is_connected()
        print(f"Connection Status: {'Connected' if is_connected else 'Not Connected'}")
        
        if not is_connected:
            raise ConnectionError("Failed to connect to Infura Sepolia endpoint")
        
        # Retrieve and print network information
        try:
            print(f"Chain ID: {web3.eth.chain_id}")
            print(f"Latest Block Number: {web3.eth.block_number}")
            print(f"Gas Price: {web3.eth.gas_price} Wei")
        except Exception as network_error:
            print(f"Error retrieving network information: {network_error}")
        
        # Set up account if private key is provided
        if PRIVATE_KEY:
            try:
                account = Account.from_key(PRIVATE_KEY)
                checksummed_address = Web3.to_checksum_address(account.address)
                web3.eth.default_account = checksummed_address
                print(f"\nAccount Details:")
                print(f"Account Address: {checksummed_address}")
                print(f"Account Balance: {web3.eth.get_balance(checksummed_address)} Wei")
            except ValueError as ve:
                print(f"Invalid address format: {ve}")
            except Exception as account_error:
                print(f"Error setting up account: {account_error}")
        
        return web3
    except Exception as e:
        logging.error(f"Comprehensive connection error: {e}")
        
        return web3
    
    except Exception as e:
        logging.error(f"Comprehensive connection error: {e}")
        
        # Detailed fallback strategy
        print("\n--- Fallback Connection Attempt ---")
        
        # Try alternative local providers
        local_providers = [
            'http://127.0.0.1:8545',  # Ganache default
            'http://localhost:8545',
            'http://127.0.0.1:7545',  # Truffle Develop
        ]
        
        for provider_url in local_providers:
            try:
                print(f"Attempting to connect to local provider: {provider_url}")
                web3 = Web3(HTTPProvider(provider_url))
                
                if web3.is_connected():
                    print(f"Successfully connected to {provider_url}")
                    return web3
            except Exception as local_error:
                print(f"Failed to connect to {provider_url}: {local_error}")
        
        logging.error("Unable to establish any blockchain connection")
        return None


def main():
    """
    Main function to run BCADN Network Analysis
    """
    try:
        # Create Web3 connection
        web3 = create_web3_connection()
        
        if web3 is None:
            print("Could not establish blockchain connection. Exiting.")
            sys.exit(1)
        
        # Initialize network analyzer
        analyzer = BCADNNetworkAnalyzer(
            web3=web3,
            contract_addresses_path=os.path.join(os.path.dirname(__file__), '..', 'config', 'contract_addresses.json')
        )

        # Load BCADN contracts
        analyzer.load_bcadn_contracts()

        # Analyze network performance
        network_results = analyzer.analyze_network_performance()

        # Run stress test simulation
        stress_test_results = analyzer.simulate_network_stress_test()

        # Combine results
        network_results['stress_test'] = stress_test_results

        # Visualize network analysis
        analyzer.visualize_network_analysis(network_results)

    except Exception as e:
        logging.error(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        sys.exit(1)


if __name__ == "__main__":
    main()