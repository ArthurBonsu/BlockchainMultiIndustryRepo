import matplotlib.pyplot as plt
import numpy as np

# Data from the main deployment metrics table
contracts = ['NIDRegistry', 'NIASRegistry', 'ABATLTranslation', 'SequencePathRouter', 'ClusteringContract']
gas_used = [906381, 1048115, 957070, 2099678, 2010304]
gas_prices = [2.5570, 2.5518, 2.5474, 2.5431, 2.5411]
blocks = [31, 32, 33, 34, 35]
costs = [0.002318, 0.002675, 0.002438, 0.005340, 0.005108]

# Create visualization for the first table (contract metrics)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot gas used as bars
x = np.arange(len(contracts))
width = 0.4
ax1.bar(x, gas_used, width, color='skyblue', label='Gas Used')
ax1.set_ylabel('Gas Used', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(contracts, rotation=45, ha='right')
ax1.tick_params(axis='y')

# Create second y-axis for cost
ax2 = ax1.twinx()
ax2.plot(x, costs, 'ro-', linewidth=2, markersize=8, label='Cost (ETH)')
ax2.set_ylabel('Cost (ETH)', fontsize=12)
ax2.tick_params(axis='y')

# Create third y-axis for gas price
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.plot(x, gas_prices, 'go-', linewidth=2, markersize=8, label='Gas Price (gwei)')
ax3.set_ylabel('Gas Price (gwei)', fontsize=12)
ax3.tick_params(axis='y')

# Title and legend
plt.title('Smart Contract Deployment Metrics', fontsize=16, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

plt.tight_layout()
plt.savefig('deployment_metrics_chart.png', dpi=300)
plt.close()

# Create visualization for the second table (summary statistics)
# Data from the summary table
metrics = ['Total Gas Used', 'Average Gas Price', 'Total Deployment Cost', 'Blocks Span', 'Number of Contracts']
values = [7021548, 2.5481, 0.017878, 4, 5]  # Raw values for plotting
normalized_values = [v/max(values) for v in values]  # Normalize for display purposes

# Create a horizontal bar chart for summary statistics
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
bars = ax.barh(metrics, normalized_values, color=colors)

# Add actual values as text labels
for i, (bar, value) in enumerate(zip(bars, values)):
    if i == 0:  # Total Gas Used
        label = f"{value:,}"
    elif i == 1:  # Average Gas Price
        label = f"{value:.4f} gwei"
    elif i == 2:  # Total Deployment Cost
        label = f"{value:.6f} ETH"
    elif i == 3:  # Blocks Span
        label = f"31-35"
    else:  # Number of Contracts
        label = str(value)
    
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
            label, va='center', fontweight='bold')

# Customize appearance
ax.set_xlabel('Normalized Value')
ax.set_title('Smart Contract Deployment Summary', fontsize=16, fontweight='bold')
ax.set_xlim(0, 1.3)  # Adjust this to ensure text labels fit
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('deployment_summary_chart.png', dpi=300)
plt.close()

# Additional visualization: Gas usage per contract (bar chart)
plt.figure(figsize=(10, 6))
plt.bar(contracts, gas_used, color='skyblue')
plt.title('Gas Used per Contract', fontsize=14, fontweight='bold')
plt.ylabel('Gas Used', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(gas_used):
    plt.text(i, v + 50000, f"{v:,}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('gas_usage_chart.png', dpi=300)
plt.close()

# Additional visualization: Deployment cost per contract (bar chart)
plt.figure(figsize=(10, 6))
plt.bar(contracts, costs, color='lightgreen')
plt.title('Deployment Cost per Contract', fontsize=14, fontweight='bold')
plt.ylabel('Cost (ETH)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(costs):
    plt.text(i, v + 0.0002, f"{v:.6f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('deployment_cost_chart.png', dpi=300)
plt.close()