import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Contract deployment data
contracts = ['ABATLTranslation', 'NIASRegistry', 'NIDRegistry', 'ClusteringContract', 'SequencePathRouter']
gas_used = [2469024, 2235243, 1930371, 506267, 3355414]
execution_cost = [2448425, 2216485, 1914013, 501123, 3327835]
block_numbers = [8144570, 8171066, 8171690, 8218853, 8226684]
deploy_times = [2.8, 2.5, 2.1, 0.6, 3.7]  # estimated based on gas used

# Performance metrics
success_rates = [98.5, 97.2, 99.0, 96.5, 97.6]
response_times = [18, 22, 15, 25, 18]
throughputs = [76, 68, 82, 125, 42]
path_determination = [0, 32, 0, 0, 43]  # N/A for some contracts
verification_times = [62, 48, 42, 36, 78]
resource_utilization = [68, 72, 65, 45, 82]
scalability_factor = [7.2, 6.8, 7.5, 8.6, 6.2]

# Contract size/complexity (relative values)
sizes = [2.1, 1.9, 1.6, 0.4, 2.8]
complexity = [4, 3, 3, 2, 5]  # on scale 1-5

# Colors for consistent visualization
color_list = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f']

# 1. Gas Usage Comparison
plt.figure(figsize=(14, 8))

x = np.arange(len(contracts))
width = 0.35

ax = plt.subplot(111)
bars1 = ax.bar(x - width/2, gas_used, width, label='Gas Used', color=color_list)
bars2 = ax.bar(x + width/2, execution_cost, width, label='Execution Cost')

plt.ylabel('Gas', fontsize=14)
plt.title('Contract Deployment Gas Metrics', fontsize=18)
plt.xticks(x, contracts, rotation=45, ha='right')
plt.legend()

# Add values on top of bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50000,
            f'{int(height):,}',
            ha='center', va='bottom', rotation=0, fontsize=10)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50000,
            f'{int(height):,}',
            ha='center', va='bottom', rotation=0, fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('gas_usage_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance Metrics Comparison
plt.figure(figsize=(14, 10))

# Plot success rates
ax1 = plt.subplot(211)
bars = ax1.bar(contracts, success_rates, color=color_list)
ax1.set_ylabel('Success Rate (%)', fontsize=14)
ax1.set_title('Success Rate by Contract', fontsize=16)
ax1.set_ylim(95, 100)  # Starting from 95% to better show differences
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height}%',
            ha='center', va='bottom', fontsize=10)

# Plot response times
ax2 = plt.subplot(212)
bars = ax2.bar(contracts, response_times, color=color_list)
ax2.set_ylabel('Response Time (ms)', fontsize=14)
ax2.set_title('Response Time by Contract', fontsize=16)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height} ms',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Resource Utilization and Throughput
plt.figure(figsize=(14, 10))

# Plot resource utilization
ax1 = plt.subplot(211)
bars = ax1.bar(contracts, resource_utilization, color=color_list)
ax1.set_ylabel('Resource Utilization (%)', fontsize=14)
ax1.set_title('Resource Utilization by Contract', fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height}%',
            ha='center', va='bottom', fontsize=10)

# Plot throughput
ax2 = plt.subplot(212)
bars = ax2.bar(contracts, throughputs, color=color_list)
ax2.set_ylabel('Throughput (tx/s)', fontsize=14)
ax2.set_title('Throughput by Contract', fontsize=16)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height} tx/s',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('resource_throughput.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Size vs. Complexity Chart
plt.figure(figsize=(10, 8))

# Create a scatter plot with size proportional to gas used
plt.scatter(sizes, complexity, s=np.array(gas_used)/10000, alpha=0.7, c=color_list)

# Annotate each point with contract name
for i, contract in enumerate(contracts):
    plt.annotate(contract, 
               (sizes[i], complexity[i]),
               textcoords="offset points", 
               xytext=(0,10), 
               ha='center',
               fontsize=12)

plt.xlabel('Relative Size', fontsize=14)
plt.ylabel('Complexity Score (1-5)', fontsize=14)
plt.title('Contract Size vs. Complexity', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('size_complexity_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Verification Time and Scalability
plt.figure(figsize=(14, 10))

# Plot verification time
ax1 = plt.subplot(211)
bars = ax1.bar(contracts, verification_times, color=color_list)
ax1.set_ylabel('Verification Time (ms)', fontsize=14)
ax1.set_title('Verification Time by Contract', fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height} ms',
            ha='center', va='bottom', fontsize=10)

# Plot scalability factor
ax2 = plt.subplot(212)
bars = ax2.bar(contracts, scalability_factor, color=color_list)
ax2.set_ylabel('Scalability Factor', fontsize=14)
ax2.set_title('Scalability Factor by Contract', fontsize=16)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('verification_scalability.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Comprehensive Dashboard
plt.figure(figsize=(16, 12))
plt.suptitle('Blockchain Contract Performance Dashboard', fontsize=22)

# Create grid
gs = GridSpec(3, 2, figure=plt.gcf())

# Gas usage
ax1 = plt.subplot(gs[0, 0])
ax1.bar(contracts, gas_used, color=color_list)
ax1.set_title('Gas Used', fontsize=14)
ax1.set_xticklabels(contracts, rotation=45, ha='right')
ax1.set_ylabel('Gas', fontsize=12)
ax1.ticklabel_format(style='plain', axis='y')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Success rate & response time
ax2 = plt.subplot(gs[0, 1])
color = 'tab:blue'
ax2.set_title('Success Rate & Response Time', fontsize=14)
ax2.set_ylabel('Success Rate (%)', fontsize=12, color=color)
ax2.plot(contracts, success_rates, 'o-', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(95, 100)
ax2.set_xticklabels(contracts, rotation=45, ha='right')

ax2b = ax2.twinx()
color = 'tab:red'
ax2b.set_ylabel('Response Time (ms)', fontsize=12, color=color)
ax2b.plot(contracts, response_times, 's-', color=color)
ax2b.tick_params(axis='y', labelcolor=color)
ax2.grid(True, linestyle='--', alpha=0.7)

# Throughput
ax3 = plt.subplot(gs[1, 0])
ax3.bar(contracts, throughputs, color=color_list)
ax3.set_title('Throughput', fontsize=14)
ax3.set_xticklabels(contracts, rotation=45, ha='right')
ax3.set_ylabel('Throughput (tx/s)', fontsize=12)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# Verification time
ax4 = plt.subplot(gs[1, 1])
ax4.bar(contracts, verification_times, color=color_list)
ax4.set_title('Verification Time', fontsize=14)
ax4.set_xticklabels(contracts, rotation=45, ha='right')
ax4.set_ylabel('Time (ms)', fontsize=12)
ax4.grid(axis='y', linestyle='--', alpha=0.7)

# Resource utilization & scalability
ax5 = plt.subplot(gs[2, 0])
ax5.bar(contracts, resource_utilization, color=color_list)
ax5.set_title('Resource Utilization', fontsize=14)
ax5.set_xticklabels(contracts, rotation=45, ha='right')
ax5.set_ylabel('Utilization (%)', fontsize=12)
ax5.grid(axis='y', linestyle='--', alpha=0.7)

# Size vs. complexity
ax6 = plt.subplot(gs[2, 1])
ax6.scatter(sizes, complexity, s=np.array(gas_used)/10000, c=color_list)
for i, contract in enumerate(contracts):
    ax6.annotate(contract.split('Contract')[0], 
               (sizes[i], complexity[i]),
               textcoords="offset points", 
               xytext=(0,10), 
               ha='center',
               fontsize=10)
ax6.set_title('Size vs. Complexity', fontsize=14)
ax6.set_xlabel('Relative Size', fontsize=12)
ax6.set_ylabel('Complexity Score', fontsize=12)
ax6.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations have been generated successfully!")