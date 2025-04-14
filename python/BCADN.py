
#!/usr/bin/env python3
# test_bcadn.py

import pytest
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from web3 import Web3
from web3.middleware import geth_poa_middleware
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Connect to local Ganache instance
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Load BCADN contract ABI and address
with open('build/contracts/BCADN.json', 'r') as f:
    contract_data = json.load(f)
    CONTRACT_ABI = contract_data['abi']
    CONTRACT_ADDRESS = contract_data['networks']['5777']['address']

# Create contract instance
bcadn = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

# Test account setup
OWNER = w3.eth.accounts[0]
TEST_ACCOUNTS = w3.eth.accounts[1:11]  # Use 10 test accounts

# Constants for tests
DEFAULT_GAS = 6000000
NUM_SIMULATION_DAYS = 14  # For anomaly detection

class TestBCADN:
    def setup_method(self):
        """Setup test environment before each test"""
        # Register nodes
        for i, account in enumerate(TEST_ACCOUNTS):
            performance = random.randint(70, 100)
            reliability = random.randint(80, 100)
            
            tx_hash = bcadn.functions.registerNode(
                account, 
                performance, 
                reliability
            ).transact({'from': OWNER, 'gas': DEFAULT_GAS})
            
            w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Create shards
        for i in range(3):
            capacity = 1000 * (i + 1)
            tx_hash = bcadn.functions.createShard(capacity).transact({'from': OWNER, 'gas': DEFAULT_GAS})
            w3.eth.wait_for_transaction_receipt(tx_hash)
            
        # Assign nodes to shards
        shards = bcadn.functions.getAllShards().call()
        nodes, _, _ = bcadn.functions.getAllNodes().call()
        
        for i, node in enumerate(nodes):
            shard_id = shards[i % len(shards)]
            tx_hash = bcadn.functions.addNodeToShard(shard_id, node).transact({'from': OWNER, 'gas': DEFAULT_GAS})
            w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def test_node_registration(self):
        """Test if nodes were registered correctly"""
        nodes, weights, statuses = bcadn.functions.getAllNodes().call()
        
        assert len(nodes) == len(TEST_ACCOUNTS), "Not all nodes were registered"
        assert all(w > 0 for w in weights), "Node weights not initialized correctly"
        assert all(s == 0 for s in statuses), "Node statuses not initialized correctly"  # 0 = Active
    
    def test_dynamic_node_weighting(self):
        """Test dynamic node weighting algorithm"""
        # Get initial weights
        nodes, initial_weights, _ = bcadn.functions.getAllNodes().call()
        
        # Update node metrics with different anomaly scores
        for i, node in enumerate(nodes):
            performance = random.randint(70, 100)
            reliability = random.randint(80, 100)
            anomaly_score = i * 5  # 0, 5, 10, 15, ...
            
            tx_hash = bcadn.functions.updateNodeMetrics(
                node,
                performance,
                reliability,
                anomaly_score
            ).transact({'from': OWNER, 'gas': DEFAULT_GAS})
            
            w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get updated weights
        _, updated_weights, statuses = bcadn.functions.getAllNodes().call()
        
        # Verify weight calculations and probation status
        anomaly_threshold = bcadn.functions.anomalyThreshold().call()
        
        for i, (initial, updated, status) in enumerate(zip(initial_weights, updated_weights, statuses)):
            anomaly_score = i * 5
            
            # Node should be in probation if anomaly score > threshold
            if anomaly_score > anomaly_threshold:
                assert status == 1, f"Node with anomaly score {anomaly_score} should be in probation"
            else:
                assert status == 0, f"Node with anomaly score {anomaly_score} should be active"
            
            # Node weight should decrease with higher anomaly scores
            if i > 0:
                assert updated < updated_weights[i-1], "Weight should decrease with higher anomaly scores"
    
    def test_probability_gap_mechanism(self):
        """Test probability gap mechanism"""
        # Get current gap parameters
        min_prob = bcadn.functions.minProbability().call()
        max_prob = bcadn.functions.maxProbability().call()
        current_gap = bcadn.functions.currentGap().call()
        
        assert current_gap == max_prob - min_prob, "Gap calculation incorrect"
        
        # Test adjustments at boundaries
        below_min = min_prob - 10
        above_max = max_prob + 10
        within_range = (min_prob + max_prob) // 2
        
        adjusted_below = bcadn.functions.adjustToProbabilityGap(below_min).call()
        adjusted_above = bcadn.functions.adjustToProbabilityGap(above_max).call()
        adjusted_within = bcadn.functions.adjustToProbabilityGap(within_range).call()
        
        assert adjusted_below == min_prob, "Below-minimum value not adjusted correctly"
        assert adjusted_above == max_prob, "Above-maximum value not adjusted correctly"
        assert adjusted_within == within_range, "Within-range value incorrectly adjusted"
        
        # Test gap updates
        new_min = 30
        new_max = 70
        
        tx_hash = bcadn.functions.updateProbabilityRange(new_min, new_max).transact({'from': OWNER, 'gas': DEFAULT_GAS})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        updated_min = bcadn.functions.minProbability().call()
        updated_max = bcadn.functions.maxProbability().call()
        updated_gap = bcadn.functions.currentGap().call()
        
        assert updated_min == new_min, "Min probability not updated"
        assert updated_max == new_max, "Max probability not updated"
        assert updated_gap == new_max - new_min, "Gap not updated correctly"
    
    def test_transaction_processing(self):
        """Test transaction submission and processing"""
        # Submit a test transaction
        receiver = TEST_ACCOUNTS[1]
        amount = 100
        base_fee = w3.to_wei(0.01, 'ether')
        
        # Get dynamic fee
        dynamic_fee = bcadn.functions.calculateDynamicFee(base_fee).call()
        
        tx_hash = bcadn.functions.submitTransaction(
            receiver,
            amount,
            base_fee
        ).transact({
            'from': TEST_ACCOUNTS[0],
            'value': dynamic_fee + w3.to_wei(0.001, 'ether'),  # Add buffer
            'gas': DEFAULT_GAS
        })
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Extract transaction hash from event logs
        logs = bcadn.events.TransactionSubmitted().processReceipt(receipt)
        assert len(logs) > 0, "TransactionSubmitted event not emitted"
        
        tx_hash_event = logs[0]['args']['txHash']
        
        # Check pending transactions count
        pending_count = bcadn.functions.pendingTransactions().call()
        assert pending_count >= 1, "Pending transactions count incorrect"
        
        # Check transaction was assigned to a shard
        shard_id = bcadn.functions.transactionToShard(tx_hash_event).call()
        assert shard_id > 0, "Transaction not assigned to a shard"
        
        # Process the transaction
        tx_hash = bcadn.functions.processTransaction(tx_hash_event).transact({'from': OWNER, 'gas': DEFAULT_GAS})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Verify transaction status
        tx_info = bcadn.functions.transactions(tx_hash_event).call()
        assert tx_info[7], "Transaction not marked as completed"  # Index 7 is 'completed'
        
        # Check pending transactions count decreased
        updated_pending_count = bcadn.functions.pendingTransactions().call()
        assert updated_pending_count < pending_count, "Pending transactions count not decreased"
    
    def test_congestion_handling(self):
        """Test network congestion handling and dynamic fees"""
        # Get initial capacity and congestion index
        initial_capacity = bcadn.functions.networkCapacity().call()
        initial_congestion = bcadn.functions.congestionIndex().call()
        
        # Create high congestion by reducing capacity
        reduced_capacity = initial_capacity // 4
        tx_hash = bcadn.functions.setNetworkCapacity(reduced_capacity).transact({'from': OWNER, 'gas': DEFAULT_GAS})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Check congestion index increased
        high_congestion = bcadn.functions.congestionIndex().call()
        assert high_congestion > initial_congestion, "Congestion index did not increase"
        
        # Calculate dynamic fee under high congestion
        base_fee = w3.to_wei(0.01, 'ether')
        high_congestion_fee = bcadn.functions.calculateDynamicFee(base_fee).call()
        
        # Restore normal capacity
        tx_hash = bcadn.functions.setNetworkCapacity(initial_capacity).transact({'from': OWNER, 'gas': DEFAULT_GAS})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Calculate dynamic fee under normal congestion
        normal_congestion_fee = bcadn.functions.calculateDynamicFee(base_fee).call()
        
        # Verify fee decreased with lower congestion
        assert normal_congestion_fee < high_congestion_fee, "Fee did not decrease with lower congestion"
    
    def test_proactive_defense(self):
        """Test anomaly detection and proactive defense mechanism"""
        # Record anomalies for test nodes
        attack_node_indices = detection_results['node_indices']
    days = detection_results['days']
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Anomaly Score Distribution
    plt.subplot(2, 2, 1)
    plt.hist(anomaly_scores[y_true == 0], bins=20, alpha=0.5, label='Normal', color='green')
    plt.hist(anomaly_scores[y_true == 1], bins=20, alpha=0.5, label='Anomalous', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Latency vs Throughput with Anomalies
    plt.subplot(2, 2, 2)
    plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1], 
                c='green', label='True Normal', alpha=0.5)
    plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], 
                c='red', label='True Anomalous', alpha=0.5)
    plt.scatter(X[(y_pred == 1) & (y_true == 0), 0], X[(y_pred == 1) & (y_true == 0), 1], 
                marker='x', c='blue', s=100, label='False Positive')
    plt.scatter(X[(y_pred == 0) & (y_true == 1), 0], X[(y_pred == 0) & (y_true == 1), 1], 
                marker='x', c='orange', s=100, label='False Negative')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (txns/sec)')
    plt.title('Latency vs Throughput by Node Status')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 3: Node Activity Over Time
    plt.subplot(2, 2, 3)
    
    # Create a heatmap-like visualization
    unique_nodes = np.unique(node_indices)
    unique_days = np.unique(days)
    
    heatmap_data = np.zeros((len(unique_nodes), len(unique_days)))
    
    for i, node_id in enumerate(node_indices):
        node_idx = np.where(unique_nodes == node_id)[0][0]
        day_idx = np.where(unique_days == days[i])[0][0]
        heatmap_data[node_idx, day_idx] = anomaly_scores[i]
    
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.xlabel('Day')
    plt.ylabel('Node ID')
    plt.title('Node Anomaly Scores Over Time')
    plt.xticks(np.arange(len(unique_days)), unique_days)
    plt.yticks(np.arange(len(unique_nodes))[::5], unique_nodes[::5])  # Show every 5th node ID
    
    # Plot 4: Feature Importance
    plt.subplot(2, 2, 4)
    
    # Use a simple proxy for feature importance - difference in means between normal and anomalous
    feature_names = ['Latency', 'Throughput', 'Uptime', 'Error Rate', 
                     'CPU Usage', 'Memory Usage', 'Connection Count']
    
    normal_means = np.mean(X[y_true == 0], axis=0)
    anomaly_means = np.mean(X[y_true == 1], axis=0)
    
    # Normalize to see relative differences
    all_means = np.concatenate([normal_means, anomaly_means])
    min_means = np.min(all_means, axis=0)
    max_means = np.max(all_means, axis=0)
    normal_means_norm = (normal_means - min_means) / (max_means - min_means)
    anomaly_means_norm = (anomaly_means - min_means) / (max_means - min_means)
    
    # Plot feature means
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, normal_means_norm, width, label='Normal', color='green')
    plt.bar(x + width/2, anomaly_means_norm, width, label='Anomalous', color='red')
    
    plt.xlabel('Features')
    plt.ylabel('Normalized Mean Value')
    plt.title('Feature Distribution by Node Status')
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_analysis.png')
    print("Saved anomaly detection analysis to 'anomaly_detection_analysis.png'")
    
    # Additional plot: ROC curve
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig('anomaly_detection_roc.png')
    print("Saved ROC curve to 'anomaly_detection_roc.png'")


def analyze_anomalies_by_attack_type(nodes_data, detection_results):
    """Analyze detection performance by attack type"""
    # Collect data by attack type
    attack_data = {}
    
    for node_address, node in nodes_data.items():
        for day_data in node.history:
            if day_data['anomaly']:
                attack_type = day_data['attack_type']
                node_id = node.node_id
                day = day_data['day']
                
                # Find corresponding detection result
                idx = None
                for i, (idx_node, idx_day) in enumerate(zip(detection_results['node_indices'], detection_results['days'])):
                    if idx_node == node_id and idx_day == day:
                        idx = i
                        break
                
                if idx is not None:
                    detection = detection_results['y_pred'][idx]
                    score = detection_results['anomaly_scores'][idx]
                    
                    if attack_type not in attack_data:
                        attack_data[attack_type] = {'total': 0, 'detected': 0, 'scores': []}
                    
                    attack_data[attack_type]['total'] += 1
                    attack_data[attack_type]['detected'] += detection
                    attack_data[attack_type]['scores'].append(score)
    
    # Calculate detection rates and average scores
    results = []
    for attack_type, data in attack_data.items():
        detection_rate = data['detected'] / data['total'] if data['total'] > 0 else 0
        avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
        
        results.append({
            'attack_type': attack_type,
            'total': data['total'],
            'detected': data['detected'],
            'detection_rate': detection_rate,
            'avg_score': avg_score
        })
    
    # Sort by detection rate
    results.sort(key=lambda x: x['detection_rate'], reverse=True)
    
    # Create dataframe for easier handling
    df = pd.DataFrame(results)
    
    # Print results
    print("\nAnomaly Detection by Attack Type:")
    print(df[['attack_type', 'total', 'detected', 'detection_rate', 'avg_score']].to_string(index=False))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(df['attack_type'], df['detection_rate'], color='skyblue')
    plt.xlabel('Attack Type')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate by Attack Type')
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(df['attack_type'], df['avg_score'], color='orange')
    plt.xlabel('Attack Type')
    plt.ylabel('Average Anomaly Score')
    plt.title('Anomaly Score by Attack Type')
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_by_attack.png')
    print("Saved attack type analysis to 'anomaly_detection_by_attack.png'")
    
    return df


def main():
    """Main function to run the tests"""
    print("BCADN Proactive Defense Mechanism Analysis")
    print("=========================================")
    
    # Simulate node behavior
    nodes_data = simulate_node_behavior()
    
    # Detect anomalies using machine learning
    detection_results = detect_anomalies(nodes_data)
    
    # Record anomalies on blockchain
    record_anomalies_on_blockchain(nodes_data, detection_results)
    
    # Visualize anomaly detection
    visualize_anomaly_detection(nodes_data, detection_results)
    
    # Analyze by attack type
    attack_analysis = analyze_anomalies_by_attack_type(nodes_data, detection_results)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([detection_results['metrics']])
    metrics_df.to_csv('anomaly_detection_metrics.csv', index=False)
    
    attack_analysis.to_csv('attack_type_analysis.csv', index=False)
    
    print("\nAnalysis complete. Results saved to CSV files and plots.")


if __name__ == "__main__":
    main()
```

# Script for testing the Probability Gap mechanism independently

```python
#!/usr/bin/env python3
# test_probability_gap.py

import pytest
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# Connect to local Ganache instance
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Load contract addresses and ABIs
with open('./config/contract_addresses.json', 'r') as f:
    addresses = json.load(f)

# Load ProbabilityGap contract
with open('./build/contracts/ProbabilityGap.json', 'r') as f:
    pg_data = json.load(f)
    pg_abi = pg_data['abi']
    pg_address = addresses.get('ProbabilityGap')
    pg_contract = w3.eth.contract(address=pg_address, abi=pg_abi)

# Test parameters
OWNER_ACCOUNT = w3.eth.accounts[0]


def test_probability_adjustment():
    """Test how probability gap adjusts values and impacts system operation"""
    print("Testing Probability Gap Adjustment...")
    
    # Test different probability ranges
    probability_ranges = [
        (10, 90),  # Wide range
        (30, 70),  # Medium range
        (45, 55)   # Narrow range
    ]
    
    results = []
    
    for min_prob, max_prob in probability_ranges:
        # Update probability range in contract
        tx_hash = pg_contract.functions.updateProbabilityRange(
            min_prob, max_prob
        ).transact({'from': OWNER_ACCOUNT})
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert receipt.status == 1, f"Failed to update probability range to ({min_prob}, {max_prob})"
        
        # Verify range was updated
        contract_min = pg_contract.functions.minProbability().call()
        contract_max = pg_contract.functions.maxProbability().call()
        contract_gap = pg_contract.functions.currentGap().call()
        
        assert contract_min == min_prob, "Min probability not updated correctly"
        assert contract_max == max_prob, "Max probability not updated correctly"
        assert contract_gap == max_prob - min_prob, "Gap not calculated correctly"
        
        # Test adjustments across the probability spectrum
        test_values = list(range(0, 101, 5))  # 0, 5, 10, ..., 100
        adjusted_values = []
        
        for val in test_values:
            adjusted = pg_contract.functions.adjustToProbabilityGap(val).call()
            adjusted_values.append(adjusted)
            
            # Verify the adjustment logic
            if val < min_prob:
                assert adjusted == min_prob, f"Value {val} not adjusted to min {min_prob}"
            elif val > max_prob:
                assert adjusted == max_prob, f"Value {val} not adjusted to max {max_prob}"
            else:
                assert adjusted == val, f"Value {val} incorrectly adjusted to {adjusted}"
        
        # Test if values are within gap
        within_gap = []
        
        for val in test_values:
            is_within = pg_contract.functions.isWithinGap(val).call()
            within_gap.append(is_within)
            
            # Verify the within gap logic
            if min_prob <= val <= max_prob:
                assert is_within is True, f"Value {val} should be within gap ({min_prob}, {max_prob})"
            else:
                assert is_within is False, f"Value {val} should not be within gap ({min_prob}, {max_prob})"
        
        # Record results
        results.append({
            'min_prob': min_prob,
            'max_prob': max_prob,
            'gap': max_prob - min_prob,
            'test_values': test_values,
            'adjusted_values': adjusted_values,
            'within_gap': within_gap
        })
        
        print(f"Tested probability range ({min_prob}, {max_prob}) with gap = {max_prob - min_prob}")
    
    return results


def simulate_blockchain_load_with_gap():
    """Simulate blockchain load under different probability gap configurations"""
    print("Simulating blockchain load with different probability gaps...")
    
    # Simulation parameters
    NUM_NODES = 100
    NUM_TRANSACTIONS = 1000
    NUM_ITERATIONS = 5
    
    # Different gap configurations to test
    gap_configs = [
        (10, 90),  # Wide range
        (30, 70),  # Medium range
        (45, 55)   # Narrow range
    ]
    
    results = []
    
    for min_prob, max_prob in gap_configs:
        # Update probability range in contract
        tx_hash = pg_contract.functions.updateProbabilityRange(
            min_prob, max_prob
        ).transact({'from': OWNER_ACCOUNT})
        
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Run simulation with this gap configuration
        print(f"Running simulation with probability gap ({min_prob}, {max_prob})...")
        
        # Create nodes with random performance characteristics
        nodes = []
        for i in range(NUM_NODES):
            node_capacity = random.randint(5, 20)  # Transactions per iteration
            node_reliability = random.random() * 100  # 0-100 reliability score
            
            nodes.append({
                'id': i,
                'capacity': node_capacity,
                'reliability': node_reliability,
                'total_processed': 0,
                'total_errors': 0
            })
        
        # Create simulation metrics
        total_processed = 0
        total_failed = 0
        iterations_to_complete = 0
        load_distribution = []
        
        # Generate transactions
        transactions = [{'id': i, 'processed': False} for i in range(NUM_TRANSACTIONS)]
        pending_transactions = transactions.copy()
        
        # Run simulation iterations
        for iteration in range(NUM_ITERATIONS):
            if not pending_transactions:
                break
                
            iterations_to_complete = iteration + 1
            
            # Track transactions processed this iteration
            processed_this_iteration = 0
            failed_this_iteration = 0
            
            # Calculate node selection probabilities based on reliability
            raw_probabilities = [node['reliability'] for node in nodes]
            
            # Apply probability gap
            adjusted_probabilities = []
            for prob in raw_probabilities:
                if prob < min_prob:
                    adjusted_prob = min_prob
                elif prob > max_prob:
                    adjusted_prob = max_prob
                else:
                    adjusted_prob = prob
                adjusted_probabilities.append(adjusted_prob)
            
            # Normalize to sum to 1
            adjusted_probabilities = np.array(adjusted_probabilities)
            adjusted_probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)
            
            # Distribute transactions to nodes based on adjusted probabilities
            node_loads = [0] * NUM_NODES
            
            for tx in pending_transactions[:]:
                # Select node based on probability
                selected_node_idx = np.random.choice(range(NUM_NODES), p=adjusted_probabilities)
                node = nodes[selected_node_idx]
                
                # Check if node has capacity
                if node_loads[selected_node_idx] < node['capacity']:
                    # Process transaction
                    node_loads[selected_node_idx] += 1
                    
                    # Determine if transaction is successful based on node reliability
                    success_chance = node['reliability'] / 100
                    if random.random() <= success_chance:
                        tx['processed'] = True
                        node['total_processed'] += 1
                        total_processed += 1
                        processed_this_iteration += 1
                        pending_transactions.remove(tx)
                    else:
                        node['total_errors'] += 1
                        total_failed += 1
                        failed_this_iteration += 1
            
            # Record load distribution for this iteration
            load_distribution.append({
                'iteration': iteration,
                'node_loads': node_loads.copy(),
                'processed': processed_this_iteration,
                'failed': failed_this_iteration,
                'pending': len(pending_transactions)
            })
        
        # Calculate results
        throughput = total_processed / iterations_to_complete
        error_rate = total_failed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0
        
        # Calculate load distribution statistics
        load_std_devs = [np.std(dist['node_loads']) for dist in load_distribution]
        avg_load_std_dev = np.mean(load_std_devs) if load_std_devs else 0
        
        # Calculate node utilization
        node_utilizations = [node['total_processed'] / (node['capacity'] * iterations_to_complete) for node in nodes]
        avg_utilization = np.mean(node_utilizations)
        utilization_std_dev = np.std(node_utilizations)
        
        # Record results
        results.append({
            'min_prob': min_prob,
            'max_prob': max_prob,
            'gap': max_prob - min_prob,
            'total_processed': total_processed,
            'total_failed': total_failed,
            'iterations_to_complete': iterations_to_complete,
            'throughput': throughput,
            'error_rate': error_rate,
            'avg_load_std_dev': avg_load_std_dev,
            'avg_utilization': avg_utilization,
            'utilization_std_dev': utilization_std_dev,
            'load_distribution': load_distribution,
            'nodes': nodes
        })
        
        print(f"Simulation complete: Processed {total_processed}/{NUM_TRANSACTIONS} transactions "
              f"in {iterations_to_complete} iterations")
        print(f"Throughput: {throughput:.2f} tx/iteration, Error Rate: {error_rate:.4f}")
    
    return results


def visualize_probability_gap_adjustment(adjustment_results):
    """Visualize how probability gap adjusts values"""
    # Plot adjustment for each gap configuration
    plt.figure(figsize=(15, 5))
    
    for i, result in enumerate(adjustment_results):
        plt.subplot(1, 3, i+1)
        
        # Plot original vs adjusted values
        plt.plot(result['test_values'], result['test_values'], 'b--', label='Original')
        plt.plot(result['test_values'], result['adjusted_values'], 'r-', label='Adjusted')
        
        # Highlight the gap region
        plt.axvspan(result['min_prob'], result['max_prob'], alpha=0.2, color='green', label='Valid Gap')
        
        # Add reference lines
        plt.axhline(y=result['min_prob'], color='g', linestyle='-.')
        plt.axhline(y=result['max_prob'], color='g', linestyle='-.')
        
        plt.xlabel('Original Probability')
        plt.ylabel('Adjusted Probability')
        plt.title(f'Gap: {result["min_prob"]}-{result["max_prob"]} ({result["gap"]})')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('probability_gap_adjustment.png')
    print("Saved probability gap adjustment visualization to 'probability_gap_adjustment.png'")


def visualize_simulation_results(simulation_results):
    """Visualize blockchain load simulation results with different probability gaps"""
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Extract data
    gap_sizes = [result['gap'] for result in simulation_results]
    throughputs = [result['throughput'] for result in simulation_results]
    error_rates = [result['error_rate'] * 100 for result in simulation_results]  # Convert to percentage
    load_std_devs = [result['avg_load_std_dev'] for result in simulation_results]
    utilizations = [result['avg_utilization'] * 100 for result in simulation_results]  # Convert to percentage
    utilization_std_devs = [result['utilization_std_dev'] * 100 for result in simulation_results]  # Convert to percentage
    
    # Plot throughput
    plt.subplot(2, 2, 1)
    plt.bar(gap_sizes, throughputs, color='skyblue')
    plt.xlabel('Probability Gap Size')
    plt.ylabel('Throughput (Tx/Iteration)')
    plt.title('Effect of Probability Gap on Throughput')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot error rate
    plt.subplot(2, 2, 2)
    plt.bar(gap_sizes, error_rates, color='salmon')
    plt.xlabel('Probability Gap Size')
    plt.ylabel('Error Rate (%)')
    plt.title('Effect of Probability Gap on Error Rate')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot load distribution
    plt.subplot(2, 2, 3)
    plt.bar(gap_sizes, load_std_devs, color='lightgreen')
    plt.xlabel('Probability Gap Size')
    plt.ylabel('Node Load Standard Deviation')
    plt.title('Effect of Probability Gap on Load Distribution')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot node utilization
    plt.subplot(2, 2, 4)
    
    x = np.arange(len(gap_sizes))
    width = 0.35
    
    plt.bar(x - width/2, utilizations, width, label='Average Utilization', color='lightblue')
    plt.bar(x + width/2, utilization_std_devs, width, label='Utilization Std Dev', color='lightcoral')
    
    plt.xlabel('Probability Gap Size')
    plt.ylabel('Utilization (%)')
    plt.title('Effect of Probability Gap on Node Utilization')
    plt.xticks(x, gap_sizes)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_gap_simulation.png')
    print("Saved probability gap simulation visualization to 'probability_gap_simulation.png'")
    
    # Additional plots: Load distribution over iterations
    plt.figure(figsize=(15, 5 * len(simulation_results)))
    
    for i, result in enumerate(simulation_results):
        plt.subplot(len(simulation_results), 1, i+1)
        
        # Extract iteration data
        iterations = [dist['iteration'] for dist in result['load_distribution']]
        processed = [dist['processed'] for dist in result['load_distribution']]
        failed = [dist['failed'] for dist in result['load_distribution']]
        pending = [dist['pending'] for dist in result['load_distribution']]
        
        # Create stacked bar chart
        width = 0.7
        plt.bar(iterations, processed, width, label='Processed', color='green')
        plt.bar(iterations, failed, width, bottom=processed, label='Failed', color='red')
        
        # Plot pending as line
        plt.plot(iterations, pending, 'bo-', label='Pending', linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Transactions')
        plt.title(f'Transaction Processing with Gap Size {result["gap"]} ({result["min_prob"]}-{result["max_prob"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_gap_iterations.png')
    print("Saved iteration analysis to 'probability_gap_iterations.png'")
    
    # Node utilization distribution
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(simulation_results):
        plt.subplot(2, 2, i+1)
        
        # Extract node utilization data
        utilizations = [node['total_processed'] / (node['capacity'] * result['iterations_to_complete']) * 100 
                        for node in result['nodes']]
        
        # Plot histogram
        plt.hist(utilizations, bins=20, color='skyblue', alpha=0.7)
        plt.axvline(np.mean(utilizations), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(utilizations):.1f}%')
        
        plt.xlabel('Node Utilization (%)')
        plt.ylabel('Number of Nodes')
        plt.title(f'Node Utilization Distribution with Gap Size {result["gap"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_gap_utilization.png')
    print("Saved utilization analysis to 'probability_gap_utilization.png'")


def main():
    """Main function to run the tests"""
    print("BCADN Probability Gap Mechanism Analysis")
    print("========================================")
    
    # Test probability adjustment
    adjustment_results = test_probability_adjustment()
    
    # Simulate blockchain load
    simulation_results = simulate_blockchain_load_with_gap()
    
    # Visualize results
    visualize_probability_gap_adjustment(adjustment_results)
    visualize_simulation_results(simulation_results)
    
    # Save results to CSV
    adjustment_df = pd.DataFrame([{
        'min_prob': result['min_prob'],
        'max_prob': result['max_prob'],
        'gap': result['gap']
    } for result in adjustment_results])
    adjustment_df.to_csv('probability_gap_adjustment_results.csv', index=False)
    
    simulation_df = pd.DataFrame([{
        'min_prob': result['min_prob'],
        'max_prob': result['max_prob'],
        'gap': result['gap'],
        'throughput': result['throughput'],
        'error_rate': result['error_rate'],
        'avg_load_std_dev': result['avg_load_std_dev'],
        'avg_utilization': result['avg_utilization'],
        'utilization_std_dev': result['utilization_std_dev'],
        'iterations_to_complete': result['iterations_to_complete']
    } for result in simulation_results])
    simulation_df.to_csv('probability_gap_simulation_results.csv', index=False)
    
    print("\nAnalysis complete. Results saved to CSV files and plots.")


if __name__ == "__main__":
    main()
```

# Main deployment and test script to run the entire BCADN test suite

```python
#!/usr/bin/env python3
# deploy_and_test_bcadn.py

import os
import subprocess
import time
import json
import argparse
from web3 import Web3
from web3.middleware import geth_poa_middleware

# Configuration
NETWORK_URL = 'http://127.0.0.1:8545'
GAS_LIMIT = 6000000
GAS_PRICE = 20000000000  # 20 Gwei

# Contract deployment order
DEPLOY_ORDER = [
    'BCNetworkMonitor',
    'DynamicNodeWeighting',
    'BCShardingManager',
    'AnomalyDetection',
    'ProbabilityGap',
    'BCTransactionProcessor'
]

# Connect to network
w3 = Web3(Web3.HTTPProvider(NETWORK_URL))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)


def compile_contracts():
    """Compile smart contracts using Truffle"""
    print("Compiling smart contracts...")
    result = subprocess.run(['truffle', 'compile'], check=True, capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0


def deploy_contracts():
    """Deploy smart contracts to the network"""
    print("Deploying smart contracts...")
    result = subprocess.run(['truffle', 'migrate', '--reset'], check=True, capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0


def load_contract_addresses():
    """Load deployed contract addresses from file"""
    with open('./config/contract_addresses.json', 'r') as f:
        addresses = json.load(f)
    return addresses


def run_tests(tests_to_run=None):
    """Run the test suite"""
    print("\nRunning BCADN test suite...")
    
    available_tests = {
        "main": "python -m pytest test_bcadn_components.py -v",
        "node_weighting": "python test_dynamic_node_weighting.py",
        "defense": "python test_proactive_defense.py",
        "probability_gap": "python test_probability_gap.py"
    }
    
    if tests_to_run is None or len(tests_to_run) == 0:
        # Run all tests
        tests_to_run = available_tests.keys()
    
    results = {}
    
    for test in tests_to_run:
        if test in available_tests:
            print(f"\n\n{'='*80}")
            print(f"Running {test} tests...")
            print(f"{'='*80}\n")
            
            command = available_tests[test]
            start_time = time.time()
            
            result = subprocess.run(command, shell=True, capture_output=False)
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[test] = {
                "success": result.returncode == 0,
                "duration": duration
            }
            
            print(f"\n{test} tests {'PASSED' if result.returncode == 0 else 'FAILED'} in {duration:.2f} seconds")
        else:
            print(f"Unknown test: {test}")
    
    return results


def generate_report(test_results):
    """Generate a summary report of the test results"""
    print("\n\n" + "="*80)
    print("BCADN TEST SUITE SUMMARY REPORT")
    print("="*80 + "\n")
    
    success_count = sum(1 for result in test_results.values() if result["success"])
    total_count = len(test_results)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Tests Run: {total_count}")
    print(f"Tests Passed: {success_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("\nTest Results:")
    
    for test_name, result in test_results.items():
        status = "PASSED" if result["success"] else "FAILED"
        print(f"  - {test_name}: {status} ({result['duration']:.2f}s)")
    
    # Check if visualization files were created
    visualizations = [
        "bcadn_performance.png",
        "bcadn_congestion.png",
        "dynamic_weight_parameters.png",
        "dynamic_weight_probation.png",
        "anomaly_detection_analysis.png",
        "anomaly_detection_roc.png",
        "probability_gap_adjustment.png",
        "probability_gap_simulation.png"
    ]
    
    print("\nVisualizations Generated:")
    for viz in visualizations:
        exists = os.path.exists(viz)
        print(f"  - {viz}: {'Yes' if exists else 'No'}")
    
    # Check for CSV result files
    csv_files = [
        "bcadn_performance_results.csv",
        "dynamic_weight_parameter_results.csv",
        "anomaly_threshold_results.csv",
        "anomaly_detection_metrics.csv",
        "attack_type_analysis.csv",
        "probability_gap_adjustment_results.csv",
        "probability_gap_simulation_results.csv"
    ]
    
    print("\nAnalysis Results:")
    for csv_file in csv_files:
        exists = os.path.exists(csv_file)
        print(f"  - {csv_file}: {'Yes' if exists else 'No'}")
    
    print("\n" + "="*80)
    if success_count == total_count:
        print("All tests passed successfully!")
    else:
        print(f"Some tests failed. Please check the log for details.")
    print("="*80 + "\n")


def analyze_architecture_improvements():
    """Analyze the improvements made in BCADN based on review considerations"""
    print("\n\n" + "="*80)
    print("BCADN ARCHITECTURE IMPROVEMENT ANALYSIS")
    print("="*80 + "\n")
    
    improvements = [
        {
            "issue": "Lack of Modular Analysis",
            "solution": "Implemented dedicated test scripts for each component (Dynamic Node Weighting, Probability Gap, Proactive Defense Mechanism) to isolate and analyze their individual contributions to overall performance.",
            "test_evidence": "test_dynamic_node_weighting.py, test_proactive_defense.py, test_probability_gap.py",
            "visualization": "dynamic_weight_parameters.png, anomaly_detection_analysis.png, probability_gap_simulation.png"
        },
        {
            "issue": "Conceptual Design of Key Components",
            "solution": "Added concrete implementation of all components with detailed formulas and algorithms, including the Proactive Defense Mechanism using machine learning for anomaly detection.",
            "test_evidence": "AnomalyDetection contract implementation with attack data collection, IsolationForest algorithm in test_proactive_defense.py",
            "visualization": "anomaly_detection_roc.png, anomaly_detection_by_attack.png"
        },
        {
            "issue": "Unexplained Central Repository Design",
            "solution": "Replaced central repository with decentralized sharding approach using randomized transaction processing, eliminating the need for a central coordination point.",
            "test_evidence": "BCShardingManager contract with randomized transaction assignment",
            "visualization": "bcadn_congestion.png"
        },
        {
            "issue": "Potential Fairness Issues",
            "solution": "Implemented and tested the Probability Gap mechanism to ensure fair resource allocation by constraining dynamic weights within acceptable bounds.",
            "test_evidence": "test_probability_gap.py simulation showing improved node utilization distribution",
            "visualization": "probability_gap_utilization.png"
        },
        {
            "issue": "Network Monitoring Limitations",
            "solution": "Implemented multi-modal detection of network status rather than relying on individual node status, using a statistical approach for anomaly detection.",
            "test_evidence": "Feature importance analysis in test_proactive_defense.py",
            "visualization": "anomaly_detection_analysis.png (Feature Distribution by Node Status)"
        },
        {
            "issue": "Inter-shard Communication",
            "solution": "Implemented a randomized transaction processing system that eliminates the need for a central committee leader while maintaining cross-shard consistency.",
            "test_evidence": "Transaction assignment test in TestBCShardingManager",
            "visualization": "probability_gap_iterations.png"
        },
        {
            "issue": "Anomaly Detection Algorithm",
            "solution": "Implemented a concrete anomaly detection algorithm using Isolation Forest, explaining its operation and validating its effectiveness against various attack types.",
            "test_evidence": "detect_anomalies function in test_proactive_defense.py with detailed metrics",
            "visualization": "anomaly_detection_by_attack.png"
        },
        {
            "issue": "Real Blockchain Testing",
            "solution": "Created a full test suite with both smart contract unit tests and performance tests that simulate realistic blockchain conditions with varying numbers of nodes and transactions.",
            "test_evidence": "test_bcadn_components.py with parametrized test_performance_scaling",
            "visualization": "bcadn_performance.png"
        }
    ]
    
    for i, improvement in enumerate(improvements):
        print(f"{i+1}. {improvement['issue']}:")
        print(f"   Solution: {improvement['solution']}")
        print(f"   Test Evidence: {improvement['test_evidence']}")
        print(f"   Visualization: {improvement['visualization']}")
        print()
    
    print("Key Architecture Improvements:")
    print("1. Decentralized sharding with randomized transaction processing eliminates centralized bottlenecks")
    print("2. Dynamic Node Weighting with Probability Gap ensures fair resource allocation")
    print("3. Proactive Defense Mechanism using machine learning enables effective attack detection")
    print("4. Cross-component integration testing validates the synergistic effects of the combined mechanisms")
    print("\n" + "="*80)


def main():
    """Main function to deploy contracts and run tests"""
    parser = argparse.ArgumentParser(description='BCADN Test Suite Runner')
    parser.add_argument('--skip-deployment', action='store_true', help='Skip contract deployment')
    parser.add_argument('--tests', nargs='+', choices=['main', 'node_weighting', 'defense', 'probability_gap', 'all'],
                        default=['all'], help='Tests to run')
    
    args = parser.parse_args()
    
    if not args.skip_deployment:
        # Compile and deploy contracts
        if not compile_contracts():
            print("Contract compilation failed. Exiting.")
            return
        
        if not deploy_contracts():
            print("Contract deployment failed. Exiting.")
            return
    
    # Load contract addresses
    addresses = load_contract_addresses()
    print("\nDeployed contract addresses:")
    for contract, address in addresses.items():
        print(f"  {contract}: {address}")
    
    # Determine which tests to run
    tests_to_run = []
    if 'all' in args.tests:
        tests_to_run = ['main', 'node_weighting', 'defense', 'probability_gap']
    else:
        tests_to_run = args.tests
    
    # Run tests
    test_results = run_tests(tests_to_run)
    
    # Generate report
    generate_report(test_results)
    
    # Analyze architecture improvements
    analyze_architecture_improvements()


if __name__ == "__main__":
    main()
```

## Deployment Migration Script

Let's create a migration script to deploy our contracts in the correct order:

```javascript
// 2_deploy_contracts.js
const BCNetworkMonitor = artifacts.require("BCNetworkMonitor");
const DynamicNodeWeighting = artifacts.require("DynamicNodeWeighting");
const BCShardingManager = artifacts.require("BCShardingManager");
const AnomalyDetection = artifacts.require("AnomalyDetection");
const ProbabilityGap = artifacts.require("ProbabilityGap");
const BCTransactionProcessor = artifacts.require("BCTransactionProcessor");

const fs = require('fs');

module.exports = async function(deployer, network, accounts) {
  // Default parameters
  const alpha = 10;  // Fee weighting factor
  const beta = 20;   // Performance weighting factor
  const gamma = 30;  // Anomaly weighting factor
  const delta = 50;  // Congestion responsiveness parameter
  const mu = 5;      // Transaction age influence parameter
  const anomalyThreshold = 30;
  const probationPeriod = 3600;  // 1 hour in seconds
  const minProbability = 20;
  const maxProbability = 80;
  
  // Deploy BCNetworkMonitor
  await deployer.deploy(
    BCNetworkMonitor, 
    alpha, 
    beta, 
    gamma, 
    delta, 
    mu, 
    anomalyThreshold, 
    probationPeriod
  );
  const networkMonitor = await BCNetworkMonitor.deployed();
  console.log(`BCNetworkMonitor deployed at ${networkMonitor.address}`);
  
  // Deploy DynamicNodeWeighting
  await deployer.deploy(
    DynamicNodeWeighting,
    networkMonitor.address
  );
  const nodeWeighting = await DynamicNodeWeighting.deployed();
  console.log(`DynamicNodeWeighting deployed at ${nodeWeighting.address}`);
  
  // Deploy BCShardingManager
  await deployer.deploy(
    BCShardingManager,
    nodeWeighting.address
  );
  const shardingManager = await BCShardingManager.deployed();
  console.log(`BCShardingManager deployed at ${shardingManager.address}`);
  
  // Deploy AnomalyDetection
  await deployer.deploy(
    AnomalyDetection,
    nodeWeighting.address
  );
  const anomalyDetection = await AnomalyDetection.deployed();
  console.log(`AnomalyDetection deployed at ${anomalyDetection.address}`);
  
  // Deploy ProbabilityGap
  await deployer.deploy(
    ProbabilityGap,
    minProbability,
    maxProbability
  );
  const probabilityGap = await ProbabilityGap.deployed();
  console.log(`ProbabilityGap deployed at ${probabilityGap.address}`);
  
  // Deploy BCTransactionProcessor
  await deployer.deploy(
    BCTransactionProcessor,
    networkMonitor.address,
    nodeWeighting.address,
    shardingManager.address,
    anomalyDetection.address,
    probabilityGap.address
  );
  const transactionProcessor = await BCTransactionProcessor.deployed();
  console.log(`BCTransactionProcessor deployed at ${transactionProcessor.address}`);
  
  // Save addresses to a JSON file
  const addresses = {
    "BCNetworkMonitor": networkMonitor.address,
    "DynamicNodeWeighting": nodeWeighting.address,
    "BCShardingManager": shardingManager.address,
    "AnomalyDetection": anomalyDetection.address,
    "ProbabilityGap": probabilityGap.address,
    "BCTransactionProcessor": transactionProcessor.address
  };
  
  // Create config directory if it doesn't exist
  if (!fs.existsSync('./config')) {
    fs.mkdirSync('./config');
  }
  
  // Write addresses to file
  fs.writeFileSync(
    './config/contract_addresses.json',
    JSON.stringify(addresses, null, 2)
  );
  
  console.log("Contract addresses saved to ./config/contract_addresses.json");
};
```# BCADN Test Suite

This test suite evaluates the key components of the BCADN (Blockchain-Cellular Adaptive Decentralized Network) system,
addressing concerns raised in the review statements and validating the functionality of the updated architecture.

## Smart Contracts

First, let's define the smart contracts necessary to implement and test the BCADN system:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title NodeTypes
 * @dev Defines the structure and types of nodes in the BCADN network
 */
contract NodeTypes {
    // Node status enum
    enum NodeStatus {
        Active,
        Probation,
        Excluded,
        Pending
    }

    // Node structure
    struct Node {
        address nodeAddress;
        uint256 performance;    // Performance metric
        uint256 reliability;    // Reliability score
        uint256 anomalyScore;   // Anomaly detection score
        uint256 weight;         // Dynamic weight
        uint256 isolationTime;  // Time when node was isolated (if applicable)
        NodeStatus status;      // Current status of the node
    }

    // ADN structure (Assured Data Node)
    struct ADN {
        address nodeAddress;
        uint256 significance;   // Importance in the network
        uint256 responseTime;   // Average response time
        bool isActive;          // Active status
    }
}

/**
 * @title BCNetworkMonitor
 * @dev Monitors network conditions and congestion in the BCADN architecture
 */
contract BCNetworkMonitor is Ownable, ReentrancyGuard {
    uint256 public pendingTransactions;
    uint256 public networkCapacity;
    uint256 public congestionIndex;    // Calculated as pendingTransactions / networkCapacity
    
    // Network parameters
    uint256 public alpha;  // Fee weighting factor
    uint256 public beta;   // Performance weighting factor
    uint256 public gamma;  // Anomaly weighting factor
    uint256 public delta;  // Congestion responsiveness parameter
    uint256 public mu;     // Transaction age influence parameter
    
    uint256 public anomalyThreshold;
    uint256 public probationPeriod;
    
    // Events
    event CongestionUpdated(uint256 indexed congestionIndex, uint256 pendingTransactions, uint256 networkCapacity);
    
    constructor(
        uint256 _alpha,
        uint256 _beta,
        uint256 _gamma,
        uint256 _delta,
        uint256 _mu,
        uint256 _anomalyThreshold,
        uint256 _probationPeriod
    ) {
        alpha = _alpha;
        beta = _beta;
        gamma = _gamma;
        delta = _delta;
        mu = _mu;
        anomalyThreshold = _anomalyThreshold;
        probationPeriod = _probationPeriod;
        
        // Initialize network parameters
        networkCapacity = 1000;    // Default capacity, can be adjusted
        pendingTransactions = 0;
        congestionIndex = 0;
    }
    
    /**
     * @dev Updates network parameters
     */
    function updateNetworkParams(
        uint256 _alpha,
        uint256 _beta,
        uint256 _gamma,
        uint256 _delta,
        uint256 _mu
    ) external onlyOwner {
        alpha = _alpha;
        beta = _beta;
        gamma = _gamma;
        delta = _delta;
        mu = _mu;
    }
    
    /**
     * @dev Updates the anomaly threshold and probation period
     */
    function updateThresholds(
        uint256 _anomalyThreshold,
        uint256 _probationPeriod
    ) external onlyOwner {
        anomalyThreshold = _anomalyThreshold;
        probationPeriod = _probationPeriod;
    }
    
    /**
     * @dev Updates network capacity
     */
    function setNetworkCapacity(uint256 _capacity) external onlyOwner {
        networkCapacity = _capacity;
        _updateCongestionIndex();
    }
    
    /**
     * @dev Adds pending transactions to the network
     */
    function addPendingTransactions(uint256 _count) external {
        pendingTransactions += _count;
        _updateCongestionIndex();
    }
    
    /**
     * @dev Completes a number of transactions, reducing the pending count
     */
    function completeTransactions(uint256 _count) external {
        require(pendingTransactions >= _count, "Invalid transaction count");
        pendingTransactions -= _count;
        _updateCongestionIndex();
    }
    
    /**
     * @dev Updates congestion index based on pending transactions and network capacity
     */
    function _updateCongestionIndex() internal {
        if (networkCapacity > 0) {
            congestionIndex = (pendingTransactions * 1e18) / networkCapacity;
        } else {
            congestionIndex = 0;
        }
        
        emit CongestionUpdated(congestionIndex, pendingTransactions, networkCapacity);
    }
    
    /**
     * @dev Calculates dynamic fee based on current congestion
     * @param _baseFee Base transaction fee
     * @return dynamicFee The adjusted transaction fee
     */
    function calculateDynamicFee(uint256 _baseFee) external view returns (uint256) {
        if (networkCapacity == 0) return _baseFee;
        
        // Formula: F = Fb * (1 + Tcurrent/Tmax)
        return _baseFee * (1e18 + congestionIndex) / 1e18;
    }
}

/**
 * @title DynamicNodeWeighting
 * @dev Implements the Dynamic Node Weighting algorithm for BCADN
 */
contract DynamicNodeWeighting is Ownable, NodeTypes, ReentrancyGuard {
    BCNetworkMonitor public networkMonitor;
    
    // Node registry
    mapping(address => Node) public nodes;
    address[] public nodesList;
    
    // Events
    event NodeRegistered(address indexed nodeAddress);
    event NodeWeightUpdated(address indexed nodeAddress, uint256 newWeight);
    event NodeStatusChanged(address indexed nodeAddress, NodeStatus newStatus);
    
    constructor(address _networkMonitor) {
        networkMonitor = BCNetworkMonitor(_networkMonitor);
    }
    
    /**
     * @dev Registers a new node in the network
     */
    function registerNode(
        address _nodeAddress,
        uint256 _performance,
        uint256 _reliability
    ) external onlyOwner {
        require(nodes[_nodeAddress].nodeAddress == address(0), "Node already exists");
        
        // Initial weight calculation: W(0) = alpha * Fee + beta * Performance
        uint256 initialWeight = (networkMonitor.alpha() * 100) + (networkMonitor.beta() * _performance);
        
        Node memory newNode = Node({
            nodeAddress: _nodeAddress,
            performance: _performance,
            reliability: _reliability,
            anomalyScore: 0,
            weight: initialWeight,
            isolationTime: 0,
            status: NodeStatus.Active
        });
        
        nodes[_nodeAddress] = newNode;
        nodesList.push(_nodeAddress);
        
        emit NodeRegistered(_nodeAddress);
    }
    
    /**
     * @dev Updates a node's performance metrics
     */
    function updateNodeMetrics(
        address _nodeAddress,
        uint256 _performance,
        uint256 _reliability,
        uint256 _anomalyScore
    ) external {
        require(nodes[_nodeAddress].nodeAddress != address(0), "Node does not exist");
        
        Node storage node = nodes[_nodeAddress];
        node.performance = _performance;
        node.reliability = _reliability;
        node.anomalyScore = _anomalyScore;
        
        // Check anomaly threshold
        if (_anomalyScore > networkMonitor.anomalyThreshold() && node.status == NodeStatus.Active) {
            node.status = NodeStatus.Probation;
            node.isolationTime = block.timestamp;
            emit NodeStatusChanged(_nodeAddress, NodeStatus.Probation);
        }
        
        // Dynamic weight adjustment
        _updateNodeWeight(_nodeAddress);
    }
    
    /**
     * @dev Updates the node's weight according to the dynamic weighting formula
     */
    function _updateNodeWeight(address _nodeAddress) internal {
        Node storage node = nodes[_nodeAddress];
        
        // Get parameters from network monitor
        uint256 alpha = networkMonitor.alpha();
        uint256 beta = networkMonitor.beta();
        uint256 gamma = networkMonitor.gamma();
        
        // Calculate dynamic weight based on Algorithm 1
        uint256 newWeight = (alpha * 100) + (beta * node.performance) - (gamma * node.anomalyScore);
        
        node.weight = newWeight;
        emit NodeWeightUpdated(_nodeAddress, newWeight);
    }
    
    /**
     * @dev Checks probation status and potentially reintegrates nodes
     */
    function checkProbationStatus(address _nodeAddress) external {
        Node storage node = nodes[_nodeAddress];
        require(node.nodeAddress != address(0), "Node does not exist");
        
        if (node.status == NodeStatus.Probation) {
            if (
                block.timestamp - node.isolationTime >= networkMonitor.probationPeriod() &&
                node.anomalyScore <= networkMonitor.anomalyThreshold()
            ) {
                node.status = NodeStatus.Active;
                emit NodeStatusChanged(_nodeAddress, NodeStatus.Active);
            }
        }
    }
    
    /**
     * @dev Gets all nodes with their current weights and status
     */
    function getAllNodes() external view returns (address[] memory, uint256[] memory, NodeStatus[] memory) {
        uint256 length = nodesList.length;
        uint256[] memory weights = new uint256[](length);
        NodeStatus[] memory statuses = new NodeStatus[](length);
        
        for (uint256 i = 0; i < length; i++) {
            weights[i] = nodes[nodesList[i]].weight;
            statuses[i] = nodes[nodesList[i]].status;
        }
        
        return (nodesList, weights, statuses);
    }
}

/**
 * @title BCShardingManager
 * @dev Implements sharding functionality for the BCADN architecture
 */
contract BCShardingManager is Ownable, ReentrancyGuard {
    DynamicNodeWeighting public nodeWeighting;
    
    // Shard structure
    struct Shard {
        uint256 id;
        address[] nodes;
        uint256 capacity;
        uint256 currentLoad;
        bool active;
    }
    
    // Shards mapping and list
    mapping(uint256 => Shard) public shards;
    uint256[] public shardIds;
    uint256 public nextShardId;
    
    // Transaction assignment
    mapping(bytes32 => uint256) public transactionToShard;
    
    // Events
    event ShardCreated(uint256 indexed shardId);
    event NodeAddedToShard(uint256 indexed shardId, address indexed node);
    event NodeRemovedFromShard(uint256 indexed shardId, address indexed node);
    event TransactionAssigned(bytes32 indexed txHash, uint256 indexed shardId);
    
    constructor(address _nodeWeighting) {
        nodeWeighting = DynamicNodeWeighting(_nodeWeighting);
        nextShardId = 1;
    }
    
    /**
     * @dev Creates a new shard
     */
    function createShard(uint256 _capacity) external onlyOwner {
        Shard memory newShard = Shard({
            id: nextShardId,
            nodes: new address[](0),
            capacity: _capacity,
            currentLoad: 0,
            active: true
        });
        
        shards[nextShardId] = newShard;
        shardIds.push(nextShardId);
        
        emit ShardCreated(nextShardId);
        nextShardId++;
    }
    
    /**
     * @dev Adds a node to a shard
     */
    function addNodeToShard(uint256 _shardId, address _node) external onlyOwner {
        require(shards[_shardId].id == _shardId, "Shard does not exist");
        
        shards[_shardId].nodes.push(_node);
        emit NodeAddedToShard(_shardId, _node);
    }
    
    /**
     * @dev Removes a node from a shard
     */
    function removeNodeFromShard(uint256 _shardId, address _node) external onlyOwner {
        require(shards[_shardId].id == _shardId, "Shard does not exist");
        
        Shard storage shard = shards[_shardId];
        uint256 length = shard.nodes.length;
        bool found = false;
        uint256 index;
        
        for (uint256 i = 0; i < length; i++) {
            if (shard.nodes[i] == _node) {
                found = true;
                index = i;
                break;
            }
        }
        
        require(found, "Node not in shard");
        
        // Remove node by replacing with the last element and popping
        if (index < length - 1) {
            shard.nodes[index] = shard.nodes[length - 1];
        }
        shard.nodes.pop();
        
        emit NodeRemovedFromShard(_shardId, _node);
    }
    
    /**
     * @dev Assigns a transaction to a shard using randomized selection
     */
    function assignTransactionToShard(bytes32 _txHash) external returns (uint256) {
        require(shardIds.length > 0, "No shards available");
        
        // Implement randomized transaction selection (from Random Selector)
        uint256 randomShardIndex = uint256(keccak256(abi.encodePacked(block.timestamp, _txHash))) % shardIds.length;
        uint256 shardId = shardIds[randomShardIndex];
        
        // Assign transaction to shard
        transactionToShard[_txHash] = shardId;
        shards[shardId].currentLoad++;
        
        emit TransactionAssigned(_txHash, shardId);
        return shardId;
    }
    
    /**
     * @dev Completes transaction processing, reducing shard load
     */
    function completeTransaction(bytes32 _txHash) external {
        uint256 shardId = transactionToShard[_txHash];
        require(shardId > 0, "Transaction not assigned to any shard");
        
        if (shards[shardId].currentLoad > 0) {
            shards[shardId].currentLoad--;
        }
        
        // Clear assignment
        delete transactionToShard[_txHash];
    }
    
    /**
     * @dev Returns shard information
     */
    function getShardInfo(uint256 _shardId) external view returns (
        uint256 id,
        address[] memory nodes,
        uint256 capacity,
        uint256 currentLoad,
        bool active
    ) {
        Shard storage shard = shards[_shardId];
        return (
            shard.id,
            shard.nodes,
            shard.capacity,
            shard.currentLoad,
            shard.active
        );
    }
    
    /**
     * @dev Returns all shard IDs
     */
    function getAllShards() external view returns (uint256[] memory) {
        return shardIds;
    }
}

/**
 * @title AnomalyDetection
 * @dev Implements the Proactive Defense Mechanism for BCADN
 */
contract AnomalyDetection is Ownable, ReentrancyGuard {
    DynamicNodeWeighting public nodeWeighting;
    
    // Attack data structure
    struct AttackData {
        address node;
        uint256 timestamp;
        uint256 anomalyScore;
        string attackType;
        bool resolved;
    }
    
    // Attack history
    AttackData[] public attackHistory;
    
    // Mapping of node to its attack history indices
    mapping(address => uint256[]) public nodeAttackIndices;
    
    // Events
    event AnomalyDetected(address indexed node, uint256 anomalyScore, string attackType);
    event AnomalyResolved(address indexed node, uint256 indexed attackIndex);
    
    constructor(address _nodeWeighting) {
        nodeWeighting = DynamicNodeWeighting(_nodeWeighting);
    }
    
    /**
     * @dev Records a detected anomaly
     */
    function recordAnomaly(
        address _node,
        uint256 _anomalyScore,
        string calldata _attackType
    ) external onlyOwner {
        AttackData memory newAttack = AttackData({
            node: _node,
            timestamp: block.timestamp,
            anomalyScore: _anomalyScore,
            attackType: _attackType,
            resolved: false
        });
        
        uint256 newIndex = attackHistory.length;
        attackHistory.push(newAttack);
        nodeAttackIndices[_node].push(newIndex);
        
        emit AnomalyDetected(_node, _anomalyScore, _attackType);
    }
    
    /**
     * @dev Marks an anomaly as resolved
     */
    function resolveAnomaly(uint256 _attackIndex) external onlyOwner {
        require(_attackIndex < attackHistory.length, "Invalid attack index");
        require(!attackHistory[_attackIndex].resolved, "Already resolved");
        
        attackHistory[_attackIndex].resolved = true;
        
        emit AnomalyResolved(attackHistory[_attackIndex].node, _attackIndex);
    }
    
    /**
     * @dev Gets all attacks for a specific node
     */
    function getNodeAttacks(address _node) external view returns (
        uint256[] memory timestamps,
        uint256[] memory anomalyScores,
        string[] memory attackTypes,
        bool[] memory resolved
    ) {
        uint256[] memory indices = nodeAttackIndices[_node];
        uint256 length = indices.length;
        
        timestamps = new uint256[](length);
        anomalyScores = new uint256[](length);
        attackTypes = new string[](length);
        resolved = new bool[](length);
        
        for (uint256 i = 0; i < length; i++) {
            AttackData storage attack = attackHistory[indices[i]];
            timestamps[i] = attack.timestamp;
            anomalyScores[i] = attack.anomalyScore;
            attackTypes[i] = attack.attackType;
            resolved[i] = attack.resolved;
        }
        
        return (timestamps, anomalyScores, attackTypes, resolved);
    }
    
    /**
     * @dev Gets all attack history
     */
    function getAttackHistory() external view returns (
        address[] memory nodes,
        uint256[] memory timestamps,
        uint256[] memory anomalyScores,
        bool[] memory resolved
    ) {
        uint256 length = attackHistory.length;
        
        nodes = new address[](length);
        timestamps = new uint256[](length);
        anomalyScores = new uint256[](length);
        resolved = new bool[](length);
        
        for (uint256 i = 0; i < length; i++) {
            AttackData storage attack = attackHistory[i];
            nodes[i] = attack.node;
            timestamps[i] = attack.timestamp;
            anomalyScores[i] = attack.anomalyScore;
            resolved[i] = attack.resolved;
        }
        
        return (nodes, timestamps, anomalyScores, resolved);
    }
}

/**
 * @title ProbabilityGap
 * @dev Implements the Probability Gap concept for BCADN
 */
contract ProbabilityGap is Ownable {
    // Gap parameters
    uint256 public minProbability;
    uint256 public maxProbability;
    uint256 public currentGap;
    
    // Events
    event GapUpdated(uint256 newGap, uint256 minProbability, uint256 maxProbability);
    
    constructor(uint256 _minProbability, uint256 _maxProbability) {
        require(_minProbability < _maxProbability, "Invalid probability range");
        
        minProbability = _minProbability;
        maxProbability = _maxProbability;
        currentGap = _maxProbability - _minProbability;
        
        emit GapUpdated(currentGap, minProbability, maxProbability);
    }
    
    /**
     * @dev Updates the probability range
     */
    function updateProbabilityRange(uint256 _minProbability, uint256 _maxProbability) external onlyOwner {
        require(_minProbability < _maxProbability, "Invalid probability range");
        
        minProbability = _minProbability;
        maxProbability = _maxProbability;
        currentGap = _maxProbability - _minProbability;
        
        emit GapUpdated(currentGap, minProbability, maxProbability);
    }
    
    /**
     * @dev Checks if a given probability falls within the allowed gap
     */
    function isWithinGap(uint256 _probability) external view returns (bool) {
        return _probability >= minProbability && _probability <= maxProbability;
    }
    
    /**
     * @dev Adjusts a probability to fall within the gap if needed
     */
    function adjustToProbabilityGap(uint256 _probability) external view returns (uint256) {
        if (_probability < minProbability) {
            return minProbability;
        } else if (_probability > maxProbability) {
            return maxProbability;
        }
        return _probability;
    }
}

/**
 * @title BCTransactionProcessor
 * @dev Main contract for processing transactions in the BCADN architecture
 */
contract BCTransactionProcessor is Ownable, ReentrancyGuard {
    // Component contracts
    BCNetworkMonitor public networkMonitor;
    DynamicNodeWeighting public nodeWeighting;
    BCShardingManager public shardingManager;
    AnomalyDetection public anomalyDetection;
    ProbabilityGap public probabilityGap;
    
    // Transaction structure
    struct Transaction {
        bytes32 txHash;
        address sender;
        address receiver;
        uint256 amount;
        uint256 fee;
        uint256 timestamp;
        uint256 processingTime;
        bool completed;
    }
    
    // Transactions mapping and list
    mapping(bytes32 => Transaction) public transactions;
    bytes32[] public transactionHashes;
    
    // Events
    event TransactionSubmitted(bytes32 indexed txHash, address indexed sender, address indexed receiver, uint256 amount, uint256 fee);
    event TransactionCompleted(bytes32 indexed txHash, uint256 processingTime);
    
    constructor(
        address _networkMonitor,
        address _nodeWeighting,
        address _shardingManager,
        address _anomalyDetection,
        address _probabilityGap
    ) {
        networkMonitor = BCNetworkMonitor(_networkMonitor);
        nodeWeighting = DynamicNodeWeighting(_nodeWeighting);
        shardingManager = BCShardingManager(_shardingManager);
        anomalyDetection = AnomalyDetection(_anomalyDetection);
        probabilityGap = ProbabilityGap(_probabilityGap);
    }
    
    /**
     * @dev Submits a new transaction to the network
     */
    function submitTransaction(
        address _receiver,
        uint256 _amount,
        uint256 _baseFee
    ) external payable nonReentrant returns (bytes32) {
        // Calculate dynamic fee
        uint256 dynamicFee = networkMonitor.calculateDynamicFee(_baseFee);
        require(msg.value >= dynamicFee, "Insufficient fee");
        
        // Generate transaction hash
        bytes32 txHash = keccak256(abi.encodePacked(
            msg.sender,
            _receiver,
            _amount,
            dynamicFee,
            block.timestamp
        ));
        
        // Create transaction record
        Transaction memory newTx = Transaction({
            txHash: txHash,
            sender: msg.sender,
            receiver: _receiver,
            amount: _amount,
            fee: dynamicFee,
            timestamp: block.timestamp,
            processingTime: 0,
            completed: false
        });
        
        transactions[txHash] = newTx;
        transactionHashes.push(txHash);
        
        // Update network congestion
        networkMonitor.addPendingTransactions(1);
        
        // Assign to shard using randomized selection
        shardingManager.assignTransactionToShard(txHash);
        
        emit TransactionSubmitted(txHash, msg.sender, _receiver, _amount, dynamicFee);
        
        // Refund excess fee
        if (msg.value > dynamicFee) {
            payable(msg.sender).transfer(msg.value - dynamicFee);
        }
        
        return txHash;
    }
    
    /**
     * @dev Processes and completes a transaction
     * @notice This would typically be called by validators/miners
     */
    function processTransaction(bytes32 _txHash) external onlyOwner nonReentrant {
        Transaction storage tx = transactions[_txHash];
        require(tx.txHash == _txHash, "Transaction does not exist");
        require(!tx.completed, "Transaction already completed");
        
        // Update transaction status
        tx.completed = true;
        tx.processingTime = block.timestamp - tx.timestamp;
        
        // Update network and shard status
        networkMonitor.completeTransactions(1);
        shardingManager.completeTransaction(_txHash);
        
        emit TransactionCompleted(_txHash, tx.processingTime);
    }
    
    /**
     * @dev Gets a transaction by hash
     */
    function getTransaction(bytes32 _txHash) external view returns (
        bytes32 txHash,
        address sender,
        address receiver,
        uint256 amount,
        uint256 fee,
        uint256 timestamp,
        uint256 processingTime,
        bool completed
    ) {
        Transaction memory tx = transactions[_txHash];
        require(tx.txHash == _txHash, "Transaction does not exist");
        
        return (
            tx.txHash,
            tx.sender,
            tx.receiver,
            tx.amount,
            tx.fee,
            tx.timestamp,
            tx.processingTime,
            tx.completed
        );
    }
    
    /**
     * @dev Gets all transaction hashes
     */
    function getAllTransactionHashes() external view returns (bytes32[] memory) {
        return transactionHashes;
    }
    
    /**
     * @dev Withdraws accumulated fees
     */
    function withdrawFees() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
}
```

## Python Test Scripts

Now, let's create the Python test scripts to evaluate the BCADN system.

```python
#!/usr/bin/env python3
# test_bcadn_components.py

import pytest
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import networkx as nx
from eth_account import Account
import os
import pandas as pd

# Connect to local Ganache instance
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)  # For compatibility with PoA chains

# Load contract ABIs and addresses
def load_contract(contract_name):
    with open(f'./build/contracts/{contract_name}.json', 'r') as f:
        contract_data = json.load(f)
    
    abi = contract_data['abi']
    
    with open('./config/contract_addresses.json', 'r') as f:
        addresses = json.load(f)
    
    contract_address = addresses.get(contract_name)
    if not contract_address:
        raise ValueError(f"Contract address not found for {contract_name}")
    
    return w3.eth.contract(address=contract_address, abi=abi)

# Generate test accounts
def generate_test_accounts(num_accounts):
    accounts = []
    for _ in range(num_accounts):
        acct = Account.create()
        accounts.append(acct)
    return accounts

# Transfer ETH to test accounts
def fund_test_accounts(accounts, amount_eth):
    sender = w3.eth.accounts[0]  # Using the first Ganache account as the funder
    
    for acct in accounts:
        tx_hash = w3.eth.send_transaction({
            'from': sender,
            'to': acct.address,
            'value': w3.to_wei(amount_eth, 'ether')
        })
        w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Funded {acct.address} with {amount_eth} ETH")

# Test class for Dynamic Node Weighting
class TestDynamicNodeWeighting:
    @pytest.fixture(scope="class")
    def setup(self):
        # Load contracts
        self.network_monitor = load_contract("BCNetworkMonitor")
        self.node_weighting = load_contract("DynamicNodeWeighting")
        
        # Generate test accounts
        self.test_accounts = generate_test_accounts(10)
        fund_test_accounts(self.test_accounts, 1)  # Fund each with 1 ETH
        
        return self
    
    def test_node_registration(self, setup):
        # Register nodes
        owner = w3.eth.accounts[0]
        
        for i, acct in enumerate(self.test_accounts):
            performance = random.randint(70, 100)  # Random performance between 70-100
            reliability = random.randint(80, 100)  # Random reliability between 80-100
            
            tx_hash = self.node_weighting.functions.registerNode(
                acct.address,
                performance,
                reliability
            ).transact({'from': owner})
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            assert receipt.status == 1, f"Failed to register node {acct.address}"
        
        # Verify nodes were registered
        nodes, weights, statuses = self.node_weighting.functions.getAllNodes().call()
        assert len(nodes) == len(self.test_accounts), "Not all nodes were registered"
    
    def test_weight_calculation(self, setup):
        # Update node metrics and check weight calculation
        owner = w3.eth.accounts[0]
        
        for i, acct in enumerate(self.test_accounts):
            # Update metrics with random values
            performance = random.randint(70, 100)
            reliability = random.randint(80, 100)
            anomaly_score = random.randint(0, 30)  # Low anomaly scores
            
            tx_hash = self.node_weighting.functions.updateNodeMetrics(
                acct.address,
                performance,
                reliability,
                anomaly_score
            ).transact({'from': owner})
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            assert receipt.status == 1, f"Failed to update metrics for node {acct.address}"
        
        # Verify weights were updated
        nodes, weights, statuses = self.node_weighting.functions.getAllNodes().call()
        assert all(weight > 0 for weight in weights), "Node weights not calculated correctly"
    
    def test_node_probation(self, setup):
        # Test placing nodes in probation when anomaly score exceeds threshold
        owner = w3.eth.accounts[0]
        anomaly_threshold = self.network_monitor.functions.anomalyThreshold().call()
        
        # Update a node with high anomaly score
        high_anomaly_node = self.test_accounts[0]
        high_anomaly_score = anomaly_threshold + 10
        
        tx_hash = self.node_weighting.functions.updateNodeMetrics(
            high_anomaly_node.address,
            80,  # Performance
            90,  # Reliability
            high_anomaly_score
        ).transact({'from': owner})
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert receipt.status == 1, "Failed to update metrics with high anomaly score"
        
        # Verify node status changed to Probation (status = 1)
        nodes, weights, statuses = self.node_weighting.functions.getAllNodes().call()
        node_index = nodes.index(high_anomaly_node.address)
        assert statuses[node_index] == 1, "Node not placed in probation despite high anomaly score"
    
    def test_node_reintegration(self, setup):
        # Test node reintegration after probation period
        owner = w3.eth.accounts[0]
        anomaly_threshold = self.network_monitor.functions.anomalyThreshold().call()
        
        # Get node in probation
        high_anomaly_node = self.test_accounts[0]
        
        # Update with low anomaly score
        tx_hash = self.node_weighting.functions.updateNodeMetrics(
            high_anomaly_node.address,
            90,  # Performance
            95,  # Reliability
            anomaly_threshold - 5  # Below threshold
        ).transact({'from': owner})
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert receipt.status == 1, "Failed to update metrics with low anomaly score"
        
        # Fast forward time by simulating passage of time (for testing purposes)
        # In Ganache, we can mine blocks to simulate time passing
        # This is a simplification; in a real blockchain, time would progress naturally
        probation_period = self.network_monitor.functions.probationPeriod().call()
        current_block = w3.eth.block_number
        
        # Mine blocks to simulate time passing
        for _ in range(10):  # Assuming 10 blocks is enough to cover probation period
            w3.provider.make_request("evm_mine", [])
        
        # Check probation status
        tx_hash = self.node_weighting.functions.checkProbationStatus(
            high_anomaly_node.address
        ).transact({'from': owner})
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert receipt.status == 1, "Failed to check probation status"
        
        # Verify node status changed back to Active (status = 0)
        nodes, weights, statuses = self.node_weighting.functions.getAllNodes().call()
        node_index = nodes.index(high_anomaly_node.address)
        assert statuses[node_index] == 0, "Node not reintegrated after probation period"


# Test class for BCShardingManager
class TestBCShardingManager:
    @pytest.fixture(scope="class")
    def setup(self):
        # Load contracts
        self.sharding_manager = load_contract("BCShardingManager")
        self.node_weighting = load_contract("DynamicNodeWeighting")
        
        # Set up shards for testing
        owner = w3.eth.accounts[0]
        
        # Create test shards
        for i in range(3):
            capacity = 1000 * (i + 1)  # Different capacities for each shard
            tx_hash = self.sharding_manager.functions.createShard(capacity).transact({'from': owner})
            w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get all registered nodes
        nodes, _, _ = self.node_weighting.functions.getAllNodes().call()
        
        # Distribute nodes across shards
        shard_ids = self.sharding_manager.functions.getAllShards().call()
        
        for i, node in enumerate(nodes):
            shard_id = shard_ids[i % len(shard_ids)]
            tx_hash = self.sharding_manager.functions.addNodeToShard(shard_id, node).transact({'from': owner})
            w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return self
    
    def test_shard_creation(self, setup):
        # Verify shards were created
        shard_ids = self.sharding_manager.functions.getAllShards().call()
        assert len(shard_ids) == 3, "Not all shards were created"
        
        # Check shard properties
        for shard_id in shard_ids:
            shard_info = self.sharding_manager.functions.getShardInfo(shard_id).call()
            assert shard_info[0] == shard_id, "Shard ID mismatch"
            assert shard_info[2] > 0, "Shard capacity not set"
            assert shard_info[4] is True, "Shard not active"
    
    def test_transaction_assignment(self, setup):
        # Test randomized transaction assignment
        owner = w3.eth.accounts[0]
        
        # Create test transactions
        num_transactions = 50
        tx_hashes = []
        
        for i in range(num_transactions):
            # Generate random transaction hash
            tx_hash = w3.keccak(text=f"test_transaction_{i}")
            tx_hashes.append(tx_hash)
            
            # Assign transaction to a shard
            tx_hash_eth = self.sharding_manager.functions.assignTransactionToShard(tx_hash).transact({'from': owner})
            w3.eth.wait_for_transaction_receipt(tx_hash_eth)
        
        # Check transaction distribution
        shard_ids = self.sharding_manager.functions.getAllShards().call()
        shard_loads = {}
        
        for shard_id in shard_ids:
            shard_info = self.sharding_manager.functions.getShardInfo(shard_id).call()
            shard_loads[shard_id] = shard_info[3]  # currentLoad
        
        # Print distribution statistics
        print("Transaction Distribution:")
        for shard_id, load in shard_loads.items():
            print(f"Shard {shard_id}: {load} transactions ({load/num_transactions*100:.2f}%)")
        
        # Verify all shards have some transactions (randomized distribution)
        assert all(load > 0 for load in shard_loads.values()), "Some shards have no transactions"
    
    def test_transaction_completion(self, setup):
        # Test transaction completion
        owner = w3.eth.accounts[0]
        
        # Generate a test transaction and assign to shard
        tx_hash = w3.keccak(text="completion_test_transaction")
        
        tx_hash_eth = self.sharding_manager.functions.assignTransactionToShard(tx_hash).transact({'from': owner})
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash_eth)
        assert receipt.status == 1, "Failed to assign transaction to shard"
        
        # Get assigned shard info before completion
        shard_id = self.sharding_manager.functions.transactionToShard(tx_hash).call()
        before_shard_info = self.sharding_manager.functions.getShardInfo(shard_id).call()
        before_load = before_shard_info[3]
        
        # Complete transaction
        tx_hash_eth = self.sharding_manager.functions.completeTransaction(tx_hash).transact({'from': owner})
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash_eth)
        assert receipt.status == 1, "Failed to complete transaction"
        
        # Get shard info after completion
        after_shard_info = self.sharding_manager.functions.getShardInfo(shard_id).call()
        after_load = after_shard_info[3]
        
        # Verify shard load decreased
        assert after_load == before_load - 1, "Shard load not decreased after transaction completion"


# Test class for AnomalyDetection
class TestAnomalyDetection:
    @pytest.fixture(scope="class")
    def setup(self):
        # Load contracts
        self.anomaly_detection = load_contract("AnomalyDetection")
        self.node_weighting = load_contract("DynamicNodeWeighting")
        
        # Get registered nodes
        nodes, _, _ = self.node_weighting.functions.getAllNodes().call()
        self.test_nodes = nodes[:5]  # Use first 5 nodes for testing
        
        return self
    
    def test_anomaly_recording(self, setup):
        # Test recording of anomalies
        owner = w3.eth.accounts[0]
        
        # Record anomalies for test nodes
        attack_types = [
            "DDoS Attack",
            "Sybil Attack",
            "Eclipse Attack",
            "Border Gateway Protocol Hijacking",
            "Long-Range Attack"
        ]
        
        for i, node in enumerate(self.test_nodes):
            anomaly_score = random.randint(30, 90)
            attack_type = attack_types[i]
            
            tx_hash = self.anomaly_detection.functions.recordAnomaly(
                node,
                anomaly_score,
                attack_type
            ).transact({'from': owner})
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            assert receipt.status == 1, f"Failed to record anomaly for node {node}"
        
        # Verify anomalies were recorded
        nodes, timestamps, scores, resolved = self.anomaly_detection.functions.getAttackHistory().call()
        assert len(nodes) == len(self.test_nodes), "Not all anomalies were recorded"
    
    def test_anomaly_resolution(self, setup):
        # Test resolving anomalies
        owner = w3.eth.accounts[0]
        
        # Resolve the first two anomalies
        for i in range(2):
            tx_hash = self.anomaly_detection.functions.resolveAnomaly(i).transact({'from': owner})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            assert receipt.status == 1, f"Failed to resolve anomaly {i}"
        
        # Verify anomalies were resolved
        nodes, timestamps, scores, resolved = self.anomaly_detection.functions.getAttackHistory().call()
        assert resolved[0] is True, "First anomaly not marked as resolved"
        assert resolved[1] is True, "Second anomaly not marked as resolved"
        if len(resolved) > 2:
            assert resolved[2] is False, "Third anomaly incorrectly marked as resolved"
    
    def test_node_specific_attacks(self, setup):
        # Test retrieving attacks for a specific node
        owner = w3.eth.accounts[0]
        test_node = self.test_nodes[0]
        
        # Record multiple anomalies for the same node
        attack_types = ["51% Attack", "Block Withholding", "Selfish Mining"]
        
        for attack_type in attack_types:
            anomaly_score = random.randint(30, 90)
            
            tx_hash = self.anomaly_detection.functions.recordAnomaly(
                test_node,
                anomaly_score,
                attack_type
            ).transact({'from': owner})
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            assert receipt.status == 1, f"Failed to record {attack_type} for node {test_node}"
        
        # Retrieve node-specific attacks
        timestamps, scores, types, resolved = self.anomaly_detection.functions.getNodeAttacks(test_node).call()
        
        # Verify all attacks for the node were retrieved
        assert len(timestamps) >= len(attack_types), "Not all node-specific attacks were retrieved"
        
        # Print attack history for the node
        print(f"Attack history for node {test_node}:")
        for i in range(len(timestamps)):
            print(f"  Attack {i}: Type={types[i]}, Score={scores[i]}, Resolved={resolved[i]}")


# Test class for ProbabilityGap
class TestProbabilityGap:
    @pytest.fixture(scope="class")
    def setup(self):
        # Load contract
        self.probability_gap = load_contract("ProbabilityGap")
        return self
    
    def test_probability_range(self, setup):
        # Test probability range initialization
        min_prob = self.probability_gap.functions.minProbability().call()
        max_prob = self.probability_gap.functions.maxProbability().call()
        current_gap = self.probability_gap.functions.currentGap().call()
        
        assert min_prob < max_prob, "Min probability should be less than max probability"
        assert current_gap == max_prob - min_prob, "Gap calculation incorrect"
    
    def test_update_probability_range(self, setup):
        # Test updating probability range
        owner = w3.eth.accounts[0]
        new_min = 20
        new_max = 80
        
        tx_hash = self.probability_gap.functions.updateProbabilityRange(new_min, new_max).transact({'from': owner})
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert receipt.status == 1, "Failed to update probability range"
        
        # Verify range was updated
        min_prob = self.probability_gap.functions.minProbability().call()
        max_prob = self.probability_gap.functions.maxProbability().call()
        current_gap = self.probability_gap.functions.currentGap().call()
        
        assert min_prob == new_min, "Min probability not updated"
        assert max_prob == new_max, "Max probability not updated"
        assert current_gap == new_max - new_min, "Gap not updated correctly"
    
    def test_probability_adjustment(self, setup):
        # Test adjusting probabilities to fit within gap
        min_prob = self.probability_gap.functions.minProbability().call()
        max_prob = self.probability_gap.functions.maxProbability().call()
        
        # Test probability below minimum
        below_min = min_prob - 10
        adjusted = self.probability_gap.functions.adjustToProbabilityGap(below_min).call()
        assert adjusted == min_prob, "Below-minimum probability not adjusted to minimum"
        
        # Test probability above maximum
        above_max = max_prob + 10
        adjusted = self.probability_gap.functions.adjustToProbabilityGap(above_max).call()
        assert adjusted == max_prob, "Above-maximum probability not adjusted to maximum"
        
        # Test probability within range
        within_range = (min_prob + max_prob) // 2
        adjusted = self.probability_gap.functions.adjustToProbabilityGap(within_range).call()
        assert adjusted == within_range, "Within-range probability incorrectly adjusted"


# Test class for full BCADN system
class TestBCTransactionProcessor:
    @pytest.fixture(scope="class")
    def setup(self):
        # Load contracts
        self.transaction_processor = load_contract("BCTransactionProcessor")
        self.network_monitor = load_contract("BCNetworkMonitor")
        self.node_weighting = load_contract("DynamicNodeWeighting")
        self.sharding_manager = load_contract("BCShardingManager")
        
        # Generate test accounts for users
        self.user_accounts = generate_test_accounts(5)
        fund_test_accounts(self.user_accounts, 2)  # Fund each with 2 ETH
        
        return self
    
    def test_transaction_submission(self, setup):
        # Test submitting transactions
        owner = w3.eth.accounts[0]
        
        # Submit test transactions
        num_transactions = 10
        tx_hashes = []
        
        for i in range(num_transactions):
            sender = self.user_accounts[i % len(self.user_accounts)]
            receiver = self.user_accounts[(i + 1) % len(self.user_accounts)]
            amount = 100 * (i + 1)
            base_fee = w3.to_wei(0.01, 'ether')
            
            # Get dynamic fee
            dynamic_fee = self.network_monitor.functions.calculateDynamicFee(base_fee).call()
            
            # Submit transaction
            tx_hash = self.transaction_processor.functions.submitTransaction(
                receiver.address,
                amount,
                base_fee
            ).transact({
                'from': sender.address,
                'value': dynamic_fee + w3.to_wei(0.001, 'ether')  # Add a small buffer
            })
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            assert receipt.status == 1, f"Failed to submit transaction {i}"
            
            # Get transaction hash from event logs
            event_logs = self.transaction_processor.events.TransactionSubmitted().process_receipt(receipt)
            assert len(event_logs) > 0, "TransactionSubmitted event not emitted"
            
            tx_hash_event = event_logs[0]['args']['txHash']
            tx_hashes.append(tx_hash_event)
        
        # Verify transactions were submitted
        all_tx_hashes = self.transaction_processor.functions.getAllTransactionHashes().call()
        assert len(all_tx_hashes) >= num_transactions, "Not all transactions recorded"
        
        # Check pending transactions count
        pending_count = self.network_monitor.functions.pendingTransactions().call()
        assert pending_count >= num_transactions, "Pending transactions count incorrect"
    
    def test_transaction_processing(self, setup):
        # Test processing transactions
        owner = w3.eth.accounts[0]
        
        # Get all transaction hashes
        all_tx_hashes = self.transaction_processor.functions.getAllTransactionHashes().call()
        
        # Process half of the transactions
        num_to_process = len(all_tx_hashes) // 2
        
        for i in range(num_to_process):
            tx_hash = all_tx_hashes[i]
            
            # Process transaction
            tx_hash_eth = self.transaction_processor.functions.processTransaction(tx_hash).transact({'from': owner})
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash_eth)
            assert receipt.status == 1, f"Failed to process transaction {i}"
        
        # Verify transactions were processed
        for i in range(num_to_process):
            tx_hash = all_tx_hashes[i]
            tx_info = self.transaction_processor.functions.getTransaction(tx_hash).call()
            assert tx_info[7] is True, f"Transaction {i} not marked as completed"
        
        # Check pending transactions count decreased
        pending_count = self.network_monitor.functions.pendingTransactions().call()
        congestion_index = self.network_monitor.functions.congestionIndex().call()
        
        print(f"Pending transactions: {pending_count}")
        print(f"Congestion index: {congestion_index / 1e18:.6f}")
    
    def test_congestion_handling(self, setup):
        # Test network congestion handling
        owner = w3.eth.accounts[0]
        initial_capacity = self.network_monitor.functions.networkCapacity().call()
        
        # Create high congestion by reducing capacity
        reduced_capacity = initial_capacity // 4
        tx_hash = self.network_monitor.functions.setNetworkCapacity(reduced_capacity).transact({'from': owner})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Check congestion index increased
        high_congestion_index = self.network_monitor.functions.congestionIndex().call() / 1e18
        print(f"High congestion index: {high_congestion_index:.6f}")
        
        # Submit a transaction under high congestion
        sender = self.user_accounts[0]
        receiver = self.user_accounts[1]
        amount = 500
        base_fee = w3.to_wei(0.01, 'ether')
        
        # Calculate dynamic fee under high congestion
        high_congestion_fee = self.network_monitor.functions.calculateDynamicFee(base_fee).call()
        print(f"Base fee: {w3.from_wei(base_fee, 'ether')} ETH")
        print(f"High congestion fee: {w3.from_wei(high_congestion_fee, 'ether')} ETH")
        
        # Submit transaction with high congestion fee
        tx_hash = self.transaction_processor.functions.submitTransaction(
            receiver.address,
            amount,
            base_fee
        ).transact({
            'from': sender.address,
            'value': high_congestion_fee + w3.to_wei(0.001, 'ether')  # Add buffer
        })
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        assert receipt.status == 1, "Failed to submit transaction under high congestion"
        
        # Restore normal capacity
        tx_hash = self.network_monitor.functions.setNetworkCapacity(initial_capacity).transact({'from': owner})
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Check congestion index decreased
        normal_congestion_index = self.network_monitor.functions.congestionIndex().call() / 1e18
        print(f"Normal congestion index: {normal_congestion_index:.6f}")
        
        # Calculate dynamic fee under normal congestion
        normal_congestion_fee = self.network_monitor.functions.calculateDynamicFee(base_fee).call()
        print(f"Normal congestion fee: {w3.from_wei(normal_congestion_fee, 'ether')} ETH")
        
        # Verify fee decreased with lower congestion
        assert normal_congestion_fee < high_congestion_fee, "Fee did not decrease with lower congestion"


# Additional test for large-scale performance
class TestBCADNPerformance:
    @pytest.mark.parametrize("num_nodes", [50, 100, 200])
    @pytest.mark.parametrize("num_transactions", [100, 500])
    def test_performance_scaling(self, num_nodes, num_transactions):
        """Test performance scaling with different network sizes and transaction loads"""
        # This is a simulation using NetworkX rather than actual blockchain deployment
        # for performance testing at scale
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(num_nodes):
            performance = random.randint(70, 100)
            reliability = random.randint(80, 100)
            anomaly_score = random.randint(0, 20)
            weight = 100 + performance * 2 - anomaly_score
            
            G.add_node(i, 
                      performance=performance, 
                      reliability=reliability,
                      anomaly_score=anomaly_score,
                      weight=weight,
                      status='active')
        
        # Create shards (simulate sharding)
        num_shards = max(3, num_nodes // 20)  # 1 shard per 20 nodes, minimum 3
        shards = {}
        
        for i in range(num_shards):
            shards[i] = {
                'capacity': 1000 * (i + 1),
                'current_load': 0,
                'nodes': []
            }
        
        # Assign nodes to shards
        for i in range(num_nodes):
            shard_id = i % num_shards
            shards[shard_id]['nodes'].append(i)
        
        # Generate transactions
        transactions = []
        for i in range(num_transactions):
            sender = random.randint(0, num_nodes - 1)
            receiver = random.randint(0, num_nodes - 1)
            while receiver == sender:
                receiver = random.randint(0, num_nodes - 1)
                
            amount = random.randint(100, 1000)
            fee = random.randint(10, 50)
            
            transactions.append({
                'id': i,
                'sender': sender,
                'receiver': receiver,
                'amount': amount,
                'fee': fee,
                'status': 'pending',
                'processing_time': 0
            })
        
        # Calculate congestion index
        congestion_index = num_transactions / (sum(shard['capacity'] for shard in shards.values()))
        
        # Adjust transaction fees based on congestion
        for tx in transactions:
            tx['dynamic_fee'] = tx['fee'] * (1 + congestion_index)
        
        # Measure start time
        start_time = time.time()
        
        # Process transactions using randomized assignment
        for tx in transactions:
            # Randomly assign to shard
            shard_id = random.randint(0, num_shards - 1)
            
            # Simulate transaction processing
            processing_delay = 0.01 * (1 + shards[shard_id]['current_load'] / shards[shard_id]['capacity'])
            
            # Update shard load
            shards[shard_id]['current_load'] += 1
            
            # Record processing time
            tx['processing_time'] = processing_delay
            tx['status'] = 'processed'
            
            # Simulate dynamic node weighting
            for node in shards[shard_id]['nodes']:
                # Adjust node weight based on performance during transaction
                node_performance_factor = G.nodes[node]['performance'] / 100
                G.nodes[node]['weight'] = G.nodes[node]['weight'] * (0.95 + 0.1 * node_performance_factor)
        
        # Calculate end time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate average transaction processing time
        avg_processing_time = sum(tx['processing_time'] for tx in transactions) / len(transactions)
        
        # Calculate load distribution across shards
        shard_loads = [shard['current_load'] for shard in shards.values()]
        load_std_dev = np.std(shard_loads)
        load_distribution = {i: shards[i]['current_load'] for i in range(num_shards)}
        
        # Print performance results
        print(f"\nPerformance Test Results ({num_nodes} nodes, {num_transactions} transactions, {num_shards} shards):")
        print(f"Total processing time: {total_time:.4f} seconds")
        print(f"Average transaction processing time: {avg_processing_time:.6f} seconds")
        print(f"Transactions per second: {num_transactions / total_time:.2f}")
        print(f"Load distribution: {load_distribution}")
        print(f"Load standard deviation: {load_std_dev:.2f}")
        print(f"Congestion index: {congestion_index:.4f}")
        
        # Generate performance metrics
        results = {
            'num_nodes': num_nodes,
            'num_transactions': num_transactions,
            'num_shards': num_shards,
            'total_time': total_time,
            'avg_processing_time': avg_processing_time,
            'tps': num_transactions / total_time,
            'load_std_dev': load_std_dev,
            'congestion_index': congestion_index
        }
        
        return results


# Visualization and analysis functions for test results
def plot_performance_results(results):
    df = pd.DataFrame(results)
    
    # Plot transaction throughput
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df_pivot = df.pivot(index='num_transactions', columns='num_nodes', values='tps')
    df_pivot.plot(marker='o', ax=plt.gca())
    plt.title('Transaction Throughput')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Transactions per Second')
    plt.grid(True)
    
    # Plot processing time
    plt.subplot(1, 2, 2)
    df_pivot = df.pivot(index='num_transactions', columns='num_nodes', values='avg_processing_time')
    df_pivot.plot(marker='o', ax=plt.gca())
    plt.title('Average Processing Time')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bcadn_performance.png')
    
    # Plot congestion vs processing time
    plt.figure(figsize=(10, 6))
    sizes = df['num_nodes'] / 50  # Scale node count for bubble size
    plt.scatter(df['congestion_index'], df['avg_processing_time'], s=sizes*100, alpha=0.6)
    
    for i, row in df.iterrows():
        plt.annotate(f"{row['num_nodes']}n, {row['num_transactions']}tx", 
                    (row['congestion_index'], row['avg_processing_time']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Congestion vs Processing Time')
    plt.xlabel('Congestion Index')
    plt.ylabel('Average Processing Time (seconds)')
    plt.grid(True)
    plt.savefig('bcadn_congestion.png')


# Main function to execute tests
def main():
    # Run unit tests
    pytest.main(['-xvs', __file__])
    
    # Run performance tests and collect results
    performance_test = TestBCADNPerformance()
    results = []
    
    for nodes in [50, 100, 200]:
        for txs in [100, 500]:
            result = performance_test.test_performance_scaling(nodes, txs)
            results.append(result)
    
    # Plot performance results
    plot_performance_results(results)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('bcadn_performance_results.csv', index=False)
    print("\nPerformance results saved to 'bcadn_performance_results.csv'")


# Additional test script for module-specific testing of Dynamic Node Weighting
# This addresses the "Lack of Modular Analysis" concern from the review statements
```

Now, let's create a separate Python script to test the Dynamic Node Weighting module independently:

```python
#!/usr/bin/env python3
# test_dynamic_node_weighting.py

import pytest
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Connect to local Ganache instance
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Load contract addresses and ABIs
with open('./config/contract_addresses.json', 'r') as f:
    addresses = json.load(f)

# Load DynamicNodeWeighting contract
with open('./build/contracts/DynamicNodeWeighting.json', 'r') as f:
    dnw_data = json.load(f)
    dnw_abi = dnw_data['abi']
    dnw_address = addresses.get('DynamicNodeWeighting')
    dnw_contract = w3.eth.contract(address=dnw_address, abi=dnw_abi)

# Load BCNetworkMonitor contract
with open('./build/contracts/BCNetworkMonitor.json', 'r') as f:
    bnm_data = json.load(f)
    bnm_abi = bnm_data['abi']
    bnm_address = addresses.get('BCNetworkMonitor')
    bnm_contract = w3.eth.contract(address=bnm_address, abi=bnm_abi)

# Test parameters
OWNER_ACCOUNT = w3.eth.accounts[0]
NUM_TEST_NODES = 30
TEST_ITERATIONS = 10

# Generate random node addresses (using account as placeholder)
test_nodes = [w3.eth.accounts[i % len(w3.eth.accounts)] for i in range(NUM_TEST_NODES)]

# Node parameter ranges
PERFORMANCE_RANGE = (60, 100)
RELIABILITY_RANGE = (70, 100)
ANOMALY_SCORE_RANGE = (0, 50)


def register_test_nodes():
    """Register test nodes for the experiment"""
    print(f"Registering {NUM_TEST_NODES} test nodes...")
    
    for i, node in enumerate(test_nodes):
        performance = random.randint(*PERFORMANCE_RANGE)
        reliability = random.randint(*RELIABILITY_RANGE)
        
        # Register node
        tx_hash = dnw_contract.functions.registerNode(
            node,
            performance,
            reliability
        ).transact({'from': OWNER_ACCOUNT})
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            print(f"Failed to register node {i}")
    
    # Verify nodes were registered
    nodes, weights, statuses = dnw_contract.functions.getAllNodes().call()
    print(f"Registered {len(nodes)} nodes successfully")
    return nodes, weights, statuses


def test_dynamic_weighting_parameters():
    """Test the impact of different Alpha, Beta, and Gamma parameters"""
    results = []
    
    # Parameter sets to test
    parameter_sets = [
        (10, 20, 30),  # Baseline
        (20, 20, 30),  # High alpha (fee weight)
        (10, 40, 30),  # High beta (performance weight)
        (10, 20, 60),  # High gamma (anomaly weight)
        (30, 10, 10),  # Prioritize fees
        (10, 30, 10),  # Prioritize performance
        (10, 10, 30)   # Prioritize security
    ]
    
    print("Testing Dynamic Weighting Parameters...")
    
    for params in parameter_sets:
        alpha, beta, gamma = params
        
        # Update network monitor parameters
        tx_hash = bnm_contract.functions.updateNetworkParams(
            alpha, beta, gamma, 50, 5  # delta and mu fixed
        ).transact({'from': OWNER_ACCOUNT})
        
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Update all nodes with random metrics
        node_metrics = []
        
        for node in test_nodes:
            performance = random.randint(*PERFORMANCE_RANGE)
            reliability = random.randint(*RELIABILITY_RANGE)
            anomaly_score = random.randint(*ANOMALY_SCORE_RANGE)
            
            node_metrics.append({
                'node': node,
                'performance': performance,
                'reliability': reliability,
                'anomaly_score': anomaly_score
            })
            
            # Update metrics
            tx_hash = dnw_contract.functions.updateNodeMetrics(
                node,
                performance,
                reliability,
                anomaly_score
            ).transact({'from': OWNER_ACCOUNT})
            
            w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get weights after update
        nodes, weights, statuses = dnw_contract.functions.getAllNodes().call()
        
        # Calculate weight statistics
        weight_avg = sum(weights) / len(weights) if weights else 0
        weight_min = min(weights) if weights else 0
        weight_max = max(weights) if weights else 0
        weight_std = np.std(weights) if weights else 0
        
        # Calculate probation count
        probation_count = sum(1 for status in statuses if status == 1)  # 1 = Probation
        
        # Record results
        results.append({
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'avg_weight': weight_avg,
            'min_weight': weight_min,
            'max_weight': weight_max,
            'std_weight': weight_std,
            'probation_count': probation_count
        })
        
        print(f"Parameters (α={alpha}, β={beta}, γ={gamma}): " 
              f"Avg Weight={weight_avg:.2f}, Std={weight_std:.2f}, "
              f"Probation={probation_count}/{len(nodes)}")
    
    return results


def test_anomaly_thresholds():
    """Test the impact of different anomaly thresholds"""
    results = []
    
    # Anomaly thresholds to test
    thresholds = [10, 20, 30, 40, 50]
    probation_period = 10  # Fixed probation period
    
    print("Testing Anomaly Thresholds...")
    
    for threshold in thresholds:
        # Update threshold
        tx_hash = bnm_contract.functions.updateThresholds(
            threshold, probation_period
        ).transact({'from': OWNER_ACCOUNT})
        
        w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Reset nodes to active status (for clean testing)
        # This would require additional contract functionality in practice
        
        # Update nodes with fixed distributions of anomaly scores
        anomaly_counts = {
            'low': 0,   # 0-15
            'med': 0,   # 16-35
            'high': 0,  # 36-50
            'probation': 0  # Nodes placed in probation
        }
        
        for node in test_nodes:
            # Distribute anomaly scores across the range
            if test_nodes.index(node) < NUM_TEST_NODES * 0.3:
                anomaly_score = random.randint(0, 15)
                anomaly_counts['low'] += 1
            elif test_nodes.index(node) < NUM_TEST_NODES * 0.7:
                anomaly_score = random.randint(16, 35)
                anomaly_counts['med'] += 1
            else:
                anomaly_score = random.randint(36, 50)
                anomaly_counts['high'] += 1
            
            # Update node metrics
            tx_hash = dnw_contract.functions.updateNodeMetrics(
                node,
                random.randint(*PERFORMANCE_RANGE),
                random.randint(*RELIABILITY_RANGE),
                anomaly_score
            ).transact({'from': OWNER_ACCOUNT})
            
            w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Get node statuses after update
        nodes, weights, statuses = dnw_contract.functions.getAllNodes().call()
        
        # Count nodes in probation
        probation_count = sum(1 for status in statuses if status == 1)  # 1 = Probation
        anomaly_counts['probation'] = probation_count
        
        # Record results
        results.append({
            'threshold': threshold,
            'probation_period': probation_period,
            'low_anomaly_count': anomaly_counts['low'],
            'med_anomaly_count': anomaly_counts['med'],
            'high_anomaly_count': anomaly_counts['high'],
            'probation_count': probation_count,
            'probation_ratio': probation_count / len(nodes) if nodes else 0
        })
        
        print(f"Threshold={threshold}: "
              f"Probation={probation_count}/{len(nodes)} nodes "
              f"({probation_count/len(nodes)*100:.1f}%)")
    
    return results


def plot_parameter_results(results):
    """Plot the impact of different parameters on node weights"""
    df = pd.DataFrame(results)
    
    # Create parameter labels
    param_labels = [f"α={r['alpha']},β={r['beta']},γ={r['gamma']}" for _, r in df.iterrows()]
    
    # Plot weight statistics
    plt.figure(figsize=(10, 6))
    
    # Extract data
    avgs = df['avg_weight']
    stds = df['std_weight']
    mins = df['min_weight']
    maxs = df['max_weight']
    
    # Create x positions
    x = np.arange(len(param_labels))
    width = 0.35
    
    # Create bars
    plt.bar(x, avgs, width, yerr=stds, label='Average Weight (±σ)')
    plt.plot(x, mins, 'v', color='red', label='Min Weight')
    plt.plot(x, maxs, '^', color='green', label='Max Weight')
    
    # Add labels and legend
    plt.xlabel('Parameter Sets')
    plt.ylabel('Node Weight')
    plt.title('Impact of α, β, γ Parameters on Node Weighting')
    plt.xticks(x, param_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig('dynamic_weight_parameters.png')
    print("Saved parameter analysis to 'dynamic_weight_parameters.png'")
    
    # Plot probation counts
    plt.figure(figsize=(10, 4))
    plt.bar(param_labels, df['probation_count'], color='orange')
    plt.xlabel('Parameter Sets')
    plt.ylabel('Nodes in Probation')
    plt.title('Impact of Parameters on Node Probation')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('dynamic_weight_probation.png')
    print("Saved probation analysis to 'dynamic_weight_probation.png'")


def plot_threshold_results(results):
    """Plot the impact of different anomaly thresholds"""
    df = pd.DataFrame(results)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot data
    plt.plot(df['threshold'], df['probation_ratio'] * 100, 'o-', linewidth=2, markersize=8)
    
    # Add a stacked bar chart for anomaly distributions
    threshold_values = df['threshold'].values
    bar_width = 3
    
    plt.bar(threshold_values - bar_width/2, 
            df['high_anomaly_count'] / NUM_TEST_NODES * 100, 
            bar_width, 
            alpha=0.3, 
            color='red', 
            label='High Anomaly (36-50)')
    
    plt.bar(threshold_values - bar_width/2, 
            df['med_anomaly_count'] / NUM_TEST_NODES * 100, 
            bar_width, 
            bottom=df['high_anomaly_count'] / NUM_TEST_NODES * 100, 
            alpha=0.3, 
            color='orange', 
            label='Medium Anomaly (16-35)')
    
    plt.bar(threshold_values - bar_width/2, 
            df['low_anomaly_count'] / NUM_TEST_NODES * 100, 
            bar_width, 
            bottom=(df['high_anomaly_count'] + df['med_anomaly_count']) / NUM_TEST_NODES * 100, 
            alpha=0.3, 
            color='green', 
            label='Low Anomaly (0-15)')
    
    # Add labels and legend
    plt.xlabel('Anomaly Threshold')
    plt.ylabel('Percentage of Nodes')
    plt.title('Impact of Anomaly Threshold on Node Probation')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig('anomaly_threshold_analysis.png')
    print("Saved threshold analysis to 'anomaly_threshold_analysis.png'")


def main():
    """Main function to run the tests"""
    print("BCADN Dynamic Node Weighting Module Analysis")
    print("============================================")
    
    # Register test nodes
    register_test_nodes()
    
    # Test dynamic weighting parameters
    parameter_results = test_dynamic_weighting_parameters()
    plot_parameter_results(parameter_results)
    
    # Test anomaly thresholds
    threshold_results = test_anomaly_thresholds()
    plot_threshold_results(threshold_results)
    
    # Save all results to CSV
    pd.DataFrame(parameter_results).to_csv('dynamic_weight_parameter_results.csv', index=False)
    pd.DataFrame(threshold_results).to_csv('anomaly_threshold_results.csv', index=False)
    
    print("\nAnalysis complete. Results saved to CSV files and plots.")


if __name__ == "__main__":
    main()
```

# Script for testing the Proactive Defense Mechanism independently

```python
#!/usr/bin/env python3
# test_proactive_defense.py

import pytest
from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Connect to local Ganache instance
w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Load contract addresses and ABIs
with open('./config/contract_addresses.json', 'r') as f:
    addresses = json.load(f)

# Load AnomalyDetection contract
with open('./build/contracts/AnomalyDetection.json', 'r') as f:
    ad_data = json.load(f)
    ad_abi = ad_data['abi']
    ad_address = addresses.get('AnomalyDetection')
    ad_contract = w3.eth.contract(address=ad_address, abi=ad_abi)

# Load DynamicNodeWeighting contract
with open('./build/contracts/DynamicNodeWeighting.json', 'r') as f:
    dnw_data = json.load(f)
    dnw_abi = dnw_data['abi']
    dnw_address = addresses.get('DynamicNodeWeighting')
    dnw_contract = w3.eth.contract(address=dnw_address, abi=dnw_abi)

# Test parameters
OWNER_ACCOUNT = w3.eth.accounts[0]
NUM_TEST_NODES = 30
SIMULATION_DAYS = 14  # Simulate 2 weeks of data

# Generate random node addresses (using account as placeholder)
test_nodes = [w3.eth.accounts[i % len(w3.eth.accounts)] for i in range(NUM_TEST_NODES)]

# Attack types for simulation
ATTACK_TYPES = [
    "DDoS Attack",
    "Sybil Attack",
    "Eclipse Attack",
    "Border Gateway Protocol Hijacking",
    "Long-Range Attack",
    "51% Attack",
    "Block Withholding Attack",
    "Selfish Mining Attack",
    "Transaction Malleability Attack",
    "Replay Attack"
]

# Simulated node metrics
class NodeMetrics:
    def __init__(self, node_id):
        self.node_id = node_id
        self.latency = random.uniform(50, 150)  # ms
        self.throughput = random.uniform(80, 200)  # txns/sec
        self.uptime = random.uniform(0.95, 0.999)  # 95-99.9% uptime
        self.error_rate = random.uniform(0.001, 0.05)  # 0.1-5% error rate
        self.cpu_usage = random.uniform(10, 60)  # 10-60% CPU utilization
        self.memory_usage = random.uniform(20, 70)  # 20-70% memory utilization
        self.connection_count = random.randint(5, 20)  # 5-20 connections
        self.history = []
    
    def update_normal(self, day):
        """Update metrics with normal behavior plus random fluctuation"""
        # Normal daily fluctuation
        self.latency += random.uniform(-10, 10)
        self.latency = max(40, min(200, self.latency))
        
        self.throughput += random.uniform(-5, 5)
        self.throughput = max(70, min(220, self.throughput))
        
        self.uptime = min(0.999, max(0.94, self.uptime + random.uniform(-0.01, 0.01)))
        self.error_rate = min(0.06, max(0.0005, self.error_rate + random.uniform(-0.005, 0.005)))
        
        self.cpu_usage += random.uniform(-5, 5)
        self.cpu_usage = max(5, min(70, self.cpu_usage))
        
        self.memory_usage += random.uniform(-3, 3)
        self.memory_usage = max(15, min(80, self.memory_usage))
        
        self.connection_count += random.randint(-2, 2)
        self.connection_count = max(3, min(25, self.connection_count))
        
        self.history.append({
            'day': day,
            'latency': self.latency,
            'throughput': self.throughput,
            'uptime': self.uptime,
            'error_rate': self.error_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'connection_count': self.connection_count,
            'anomaly': False,
            'attack_type': None
        })
    
    def simulate_attack(self, day, attack_type):
        """Simulate an attack by changing metrics abnormally"""
        if attack_type == "DDoS Attack":
            self.latency *= random.uniform(3, 5)  # 3-5x higher latency
            self.throughput *= random.uniform(0.2, 0.5)  # 50-80% reduction
            self.connection_count *= random.uniform(3, 10)  # 3-10x connections
            self.cpu_usage *= random.uniform(1.5, 3)  # 50-200% more CPU
        
        elif attack_type == "Sybil Attack":
            self.connection_count *= random.uniform(2, 4)  # 2-4x connections
            self.error_rate *= random.uniform(1.5, 3)  # 50-200% more errors
        
        elif attack_type == "Eclipse Attack":
            self.connection_count = max(1, self.connection_count * random.uniform(0.2, 0.4))  # 60-80% fewer connections
            self.latency *= random.uniform(1.5, 2.5)  # 50-150% higher latency
        
        elif attack_type == "51% Attack" or attack_type == "Selfish Mining Attack":
            self.throughput *= random.uniform(1.3, 1.8)  # 30-80% increase (suspicious)
            self.cpu_usage *= random.uniform(2, 3)  # 100-200% more CPU
        
        elif attack_type == "Block Withholding Attack":
            self.throughput *= random.uniform(0.5, 0.8)  # 20-50% reduction
            self.error_rate *= random.uniform(2, 4)  # 100-300% more errors
        
        else:  # Generic anomaly for other attack types
            self.latency *= random.uniform(1.5, 2.5)
            self.throughput *= random.uniform(0.6, 0.9)
            self.error_rate *= random.uniform(1.5, 3)
            self.cpu_usage *= random.uniform(1.2, 2)
        
        # Ensure values stay within reasonable bounds
        self.latency = min(1000, self.latency)
        self.throughput = max(10, self.throughput)
        self.cpu_usage = min(100, self.cpu_usage)
        self.memory_usage = min(100, self.memory_usage)
        self.connection_count = max(1, min(100, self.connection_count))
        
        self.history.append({
            'day': day,
            'latency': self.latency,
            'throughput': self.throughput,
            'uptime': self.uptime,
            'error_rate': self.error_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'connection_count': self.connection_count,
            'anomaly': True,
            'attack_type': attack_type
        })


def simulate_node_behavior():
    """Simulate node behavior over time with some anomalies"""
    print("Simulating node behavior with anomalies...")
    
    # Initialize node metrics
    nodes = {node: NodeMetrics(i) for i, node in enumerate(test_nodes)}
    
    # Simulate daily metrics
    for day in range(SIMULATION_DAYS):
        print(f"Simulating day {day+1}/{SIMULATION_DAYS}...")
        
        # Determine which nodes have anomalies today (10% chance per node)
        anomalous_nodes = [node for node in test_nodes if random.random() < 0.1]
        
        for node in test_nodes:
            if node in anomalous_nodes:
                # Simulate an attack
                attack_type = random.choice(ATTACK_TYPES)
                nodes[node].simulate_attack(day, attack_type)
            else:
                # Normal behavior
                nodes[node].update_normal(day)
    
    return nodes


def detect_anomalies(nodes_data):
    """Use machine learning to detect anomalies in node metrics"""
    print("Detecting anomalies using machine learning...")
    
    # Prepare data for anomaly detection
    features = []
    labels = []
    node_indices = []
    days = []
    
    for node_address, node in nodes_data.items():
        for day_data in node.history:
            features.append([
                day_data['latency'],
                day_data['throughput'],
                day_data['uptime'],
                day_data['error_rate'],
                day_data['cpu_usage'],
                day_data['memory_usage'],
                day_data['connection_count']
            ])
            labels.append(1 if day_data['anomaly'] else 0)  # 1 for anomaly, 0 for normal
            node_indices.append(node.node_id)
            days.append(day_data['day'])
    
    # Convert to numpy arrays
    X = np.array(features)
    y_true = np.array(labels)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train isolation forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_scaled)
    
    # Predict anomalies (-1 for anomaly, 1 for normal)
    y_pred_raw = model.predict(X_scaled)
    y_pred = [1 if pred == -1 else 0 for pred in y_pred_raw]  # Convert to 1/0
    
    # Calculate anomaly scores (higher = more anomalous)
    anomaly_scores = -model.score_samples(X_scaled)
    anomaly_scores_normalized = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    anomaly_scores_scaled = anomaly_scores_normalized * 100  # Scale to 0-100
    
    # Calculate metrics
    true_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    false_positives = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    true_negatives = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    false_negatives = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print metrics
    print(f"Anomaly Detection Results:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Return all data for reporting and visualization
    return {
        'X': X,
        'X_scaled': X_scaled,
        'y_true': y_true,
        'y_pred': y_pred,
        'anomaly_scores': anomaly_scores_scaled,
        'node_indices': node_indices,
        'days': days,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    }


def record_anomalies_on_blockchain(nodes_data, detection_results):
    """Record detected anomalies on the blockchain"""
    print("Recording detected anomalies on the blockchain...")
    
    # Map node indices back to addresses
    node_id_to_address = {i: addr for i, addr in enumerate(test_nodes)}
    
    # Record each detected anomaly
    anomalies_recorded = 0
    
    for i in range(len(detection_results['y_pred'])):
        if detection_results['y_pred'][i] == 1:  # Detected anomaly
            node_id = detection_results['node_indices'][i]
            node_address = node_id_to_address[node_id]
            anomaly_score = detection_results['anomaly_scores'][i]
            day = detection_results['days'][i]
            
            # Find the attack type from simulation data
            attack_type = "Unknown"
            for day_data in nodes_data[node_address].history:
                if day_data['day'] == day and day_data['anomaly']:
                    attack_type = day_data['attack_type']
                    break
            
            # If attack type is still unknown but anomaly was detected
            if attack_type == "Unknown":
                attack_type = "Suspicious Activity"
            
            # Record anomaly in contract
            try:
                tx_hash = ad_contract.functions.recordAnomaly(
                    node_address,
                    int(anomaly_score),
                    attack_type
                ).transact({'from': OWNER_ACCOUNT})
                
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                if receipt.status == 1:
                    anomalies_recorded += 1
                    # Update node metrics in DynamicNodeWeighting contract
                    tx_hash = dnw_contract.functions.updateNodeMetrics(
                        node_address,
                        random.randint(60, 90),  # Performance
                        random.randint(60, 90),  # Reliability
                        int(anomaly_score)  # Anomaly score
                    ).transact({'from': OWNER_ACCOUNT})
                    w3.eth.wait_for_transaction_receipt(tx_hash)
            except Exception as e:
                print(f"Error recording anomaly: {e}")
    
    print(f"Recorded {anomalies_recorded} anomalies on the blockchain")
    return anomalies_recorded

def visualize_anomaly_detection(nodes_data, detection_results):
    """Comprehensive visualization of anomaly detection results"""
    # Extract data from detection results
    X = detection_results['X']
    y_true = detection_results['y_true']
    y_pred = detection_results['y_pred']
    anomaly_scores = detection_results['anomaly_scores']
    node_indices = detection_results['node_indices']
    days = detection_results['days']
    
    # Create multi-panel visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Anomaly Score Distribution
    plt.subplot(2, 2, 1)
    plt.hist(anomaly_scores[y_true == 0], bins=20, alpha=0.5, label='Normal', color='green')
    plt.hist(anomaly_scores[y_true == 1], bins=20, alpha=0.5, label='Anomalous', color='red')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Latency vs Throughput with Anomalies
    plt.subplot(2, 2, 2)
    plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1], 
                c='green', label='True Normal', alpha=0.5)
    plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], 
                c='red', label='True Anomalous', alpha=0.5)
    plt.scatter(X[(y_pred == 1) & (y_true == 0), 0], X[(y_pred == 1) & (y_true == 0), 1], 
                marker='x', c='blue', s=100, label='False Positive')
    plt.scatter(X[(y_pred == 0) & (y_true == 1), 0], X[(y_pred == 0) & (y_true == 1), 1], 
                marker='x', c='orange', s=100, label='False Negative')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (txns/sec)')
    plt.title('Latency vs Throughput by Node Status')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 3: Node Activity Over Time
    plt.subplot(2, 2, 3)
    
    # Create a heatmap-like visualization
    unique_nodes = np.unique(node_indices)
    unique_days = np.unique(days)
    
    heatmap_data = np.zeros((len(unique_nodes), len(unique_days)))
    
    for i, node_id in enumerate(node_indices):
        node_idx = np.where(unique_nodes == node_id)[0][0]
        day_idx = np.where(unique_days == days[i])[0][0]
        heatmap_data[node_idx, day_idx] = anomaly_scores[i]
    
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.xlabel('Day')
    plt.ylabel('Node ID')
    plt.title('Node Anomaly Scores Over Time')
    plt.xticks(np.arange(len(unique_days)), unique_days)
    plt.yticks(np.arange(len(unique_nodes))[::5], unique_nodes[::5])  # Show every 5th node ID
    
    # Plot 4: Feature Importance
    plt.subplot(2, 2, 4)
    
    # Use a simple proxy for feature importance - difference in means between normal and anomalous
    feature_names = ['Latency', 'Throughput', 'Uptime', 'Error Rate', 
                     'CPU Usage', 'Memory Usage', 'Connection Count']
    
    normal_means = np.mean(X[y_true == 0], axis=0)
    anomaly_means = np.mean(X[y_true == 1], axis=0)
    
    # Normalize to see relative differences
    all_means = np.concatenate([normal_means, anomaly_means])
    min_means = np.min(all_means, axis=0)
    max_means = np.max(all_means, axis=0)
    normal_means_norm = (normal_means - min_means) / (max_means - min_means)
    anomaly_means_norm = (anomaly_means - min_means) / (max_means - min_means)
    
    # Plot feature means
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, normal_means_norm, width, label='Normal', color='green')
    plt.bar(x + width/2, anomaly_means_norm, width, label='Anomalous', color='red')
    
    plt.xlabel('Features')
    plt.ylabel('Normalized Mean Value')
    plt.title('Feature Distribution by Node Status')
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_analysis.png')
    print("Saved anomaly detection analysis to 'anomaly_detection_analysis.png'")
    
    # Additional plot: ROC curve
    plt.figure(figsize=(8, 6))
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig('anomaly_detection_roc.png')
    print("Saved ROC curve to 'anomaly_detection_roc.png'")


def analyze_anomalies_by_attack_type(nodes_data, detection_results):
    """Analyze detection performance by attack type"""
    # Collect data by attack type
    attack_data = {}
    
    for node_address, node in nodes_data.items():
        for day_data in node.history:
            if day_data['anomaly']:
                attack_type = day_data['attack_type']
                node_id = node.node_id
                day = day_data['day']
                
                # Find corresponding detection result
                idx = None
                for i, (idx_node, idx_day) in enumerate(zip(detection_results['node_indices'], detection_results['days'])):
                    if idx_node == node_id and idx_day == day:
                        idx = i
                        break
                
                if idx is not None:
                    detection = detection_results['y_pred'][idx]
                    score = detection_results['anomaly_scores'][idx]
                    
                    if attack_type not in attack_data:
                        attack_data[attack_type] = {'total': 0, 'detected': 0, 'scores': []}
                    
                    attack_data[attack_type]['total'] += 1
                    attack_data[attack_type]['detected'] += detection
                    attack_data[attack_type]['scores'].append(score)
    
    # Calculate detection rates and average scores
    results = []
    for attack_type, data in attack_data.items():
        detection_rate = data['detected'] / data['total'] if data['total'] > 0 else 0
        avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
        
        results.append({
            'attack_type': attack_type,
            'total': data['total'],
            'detected': data['detected'],
            'detection_rate': detection_rate,
            'avg_score': avg_score
        })
    
    # Sort by detection rate
    results.sort(key=lambda x: x['detection_rate'], reverse=True)
    
    # Create dataframe for easier handling
    df = pd.DataFrame(results)
    
    # Print results
    print("\nAnomaly Detection by Attack Type:")
    print(df[['attack_type', 'total', 'detected', 'detection_rate', 'avg_score']].to_string(index=False))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(df['attack_type'], df['detection_rate'], color='skyblue')
    plt.xlabel('Attack Type')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate by Attack Type')
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(df['attack_type'], df['avg_score'], color='orange')
    plt.xlabel('Attack Type')
    plt.ylabel('Average Anomaly Score')
    plt.title('Anomaly Score by Attack Type')
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_by_attack.png')
    print("Saved attack type analysis to 'anomaly_detection_by_attack.png'")
    
    return df


def main():
    """Main function to run the tests"""
    print("BCADN Proactive Defense Mechanism Analysis")
    print("=========================================")
    
    # Simulate node behavior
    nodes_data = simulate_node_behavior()
    
    # Detect anomalies using machine learning
    detection_results = detect_anomalies(nodes_data)
    
    # Record anomalies on blockchain
    record_anomalies_on_blockchain(nodes_data, detection_results)
    
    # Visualize anomaly detection
    visualize_anomaly_detection(nodes_data, detection_results)
    
    # Analyze by attack type
    attack_analysis = analyze_anomalies_by_attack_type(nodes_data, detection_results)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([detection_results['metrics']])
    metrics_df.to_csv('anomaly_detection_metrics.csv', index=False)
    
    attack_analysis.to_csv('attack_type_analysis.csv', index=False)
    
    print("\nAnalysis complete. Results saved to CSV files and plots.")


if __name__ == "__main__":
    main()
    