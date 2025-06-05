import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Node Performance Data
node_data = {
    'name': ['node1', 'node2', 'node3', 'node4', 'node5', 'node6', 'node7', 'node8', 'node9', 'node10', 
             'node11', 'node12', 'node13', 'node14', 'node15', 'node16', 'node17', 'node18', 'node19', 'node20',
             'node21', 'node22', 'node23', 'node24', 'node25'],
    'successRate': [99, 99, 99, 99, 96, 97, 97, 96, 95, 96, 98, 98, 96, 99, 96, 97, 99, 97, 97, 98, 97, 98, 99, 95, 98],
    'packetCount': [1074, 2539, 1814, 2129, 1606, 1027, 1242, 2461, 1385, 1176, 1583, 2568, 2456, 2374, 2376, 
                   2577, 2769, 2162, 1802, 1573, 2967, 1803, 1179, 2250, 1693]
}

node_df = pd.DataFrame(node_data)

# Service Distribution Data
service_data = {
    'name': ['VoIP', 'Streaming', 'Standard'],
    'value': [6, 7, 5],
    'percentage': [33.3, 38.9, 27.8]
}

service_df = pd.DataFrame(service_data)

# Success Rate Distribution
success_rate_counts = node_df.groupby('successRate').size().reset_index(name='count')
success_rate_counts['percentage'] = success_rate_counts['count'] / success_rate_counts['count'].sum() * 100

# 1. Node Packet Count Bar Chart
def plot_node_packet_count():
    plt.figure(figsize=(15, 10))
    
    # Sort by packet count for better visualization
    sorted_df = node_df.sort_values('packetCount', ascending=False)
    
    ax = sns.barplot(x='name', y='packetCount', data=sorted_df)
    plt.title('Node Packet Count', fontsize=18)
    plt.xlabel('Node ID', fontsize=14)
    plt.ylabel('Packet Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom',
                   fontsize=10, rotation=0)
    
    plt.savefig('node_packet_count.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Node Success Rate Line Chart
def plot_node_success_rate():
    plt.figure(figsize=(15, 8))
    
    # Sort alphabetically by node name
    sorted_df = node_df.sort_values('name')
    
    ax = sns.lineplot(x='name', y='successRate', data=sorted_df, marker='o', markersize=10, linewidth=2)
    plt.title('Node Success Rate', fontsize=18)
    plt.xlabel('Node ID', fontsize=14)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(90, 100)  # Starting from 90% to better show the differences
    plt.grid(True)
    
    # Add values above points
    for i, row in enumerate(sorted_df.itertuples()):
        ax.annotate(f'{row.successRate}%', 
                   (i, row.successRate), 
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center', va='bottom',
                   fontsize=10)
    
    plt.tight_layout()
    plt.savefig('node_success_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Service Distribution Pie Chart
def plot_service_distribution():
    plt.figure(figsize=(10, 10))
    
    # Create a pie chart
    plt.pie(service_df['value'], labels=service_df['name'], autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=(0.05, 0.05, 0.05),
            colors=sns.color_palette("deep"))
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Paths by Service Class', fontsize=18)
    plt.tight_layout()
    
    plt.savefig('service_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Service Distribution Bar Chart
def plot_service_bar():
    plt.figure(figsize=(10, 7))
    
    ax = sns.barplot(x='name', y='value', data=service_df, palette="deep")
    plt.title('Number of Paths by Service Class', fontsize=18)
    plt.xlabel('Service Class', fontsize=14)
    plt.ylabel('Number of Paths', fontsize=14)
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height())} ({service_df["percentage"][i]}%)', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom',
                   fontsize=12, rotation=0,
                   xytext=(0, 5),
                   textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('service_distribution_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Success Rate Distribution Pie Chart
def plot_success_rate_distribution():
    plt.figure(figsize=(10, 10))
    
    labels = [f'{int(rate)}% Success Rate' for rate in success_rate_counts['successRate']]
    
    # Create a pie chart
    plt.pie(success_rate_counts['count'], labels=labels, autopct='%1.1f%%', 
            startangle=90, shadow=True,
            colors=sns.color_palette("deep", len(success_rate_counts)))
    
    plt.axis('equal')
    plt.title('Distribution of Nodes by Success Rate', fontsize=18)
    plt.tight_layout()
    
    plt.savefig('success_rate_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Top 10 Nodes By Packet Count
def plot_top_nodes():
    plt.figure(figsize=(12, 8))
    
    top_nodes = node_df.sort_values('packetCount', ascending=False).head(10)
    
    ax = sns.barplot(x='name', y='packetCount', data=top_nodes, palette="deep")
    plt.title('Top 10 Nodes by Packet Count', fontsize=18)
    plt.xlabel('Node ID', fontsize=14)
    plt.ylabel('Packet Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom',
                   fontsize=12, rotation=0,
                   xytext=(0, 5),
                   textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('top_nodes_packet_count.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Network Summary Dashboard
def create_network_dashboard():
    plt.figure(figsize=(18, 12))
    plt.suptitle('SequencePathRouter Network Analysis Dashboard', fontsize=24, y=0.98)
    
    # Create a grid for the subplots
    gs = GridSpec(2, 3, figure=plt.gcf())
    
    # Pie chart of service distribution
    ax1 = plt.subplot(gs[0, 0])
    plt.pie(service_df['value'], labels=service_df['name'], autopct='%1.1f%%', 
            startangle=90, explode=(0.05, 0.05, 0.05),
            colors=sns.color_palette("deep"))
    plt.title('Service Class Distribution', fontsize=16)
    plt.axis('equal')
    
    # Success rate distribution
    ax2 = plt.subplot(gs[0, 1:])
    sorted_df = node_df.sort_values('successRate')
    sns.barplot(x='successRate', y='name', data=sorted_df, palette="deep", ax=ax2)
    plt.title('Node Success Rates', fontsize=16)
    plt.xlabel('Success Rate (%)', fontsize=12)
    plt.ylabel('Node ID', fontsize=12)
    
    # Packet count bar chart
    ax3 = plt.subplot(gs[1, :])
    top_nodes = node_df.sort_values('packetCount', ascending=False).head(10)
    sns.barplot(x='name', y='packetCount', data=top_nodes, palette="deep", ax=ax3)
    plt.title('Top 10 Nodes by Packet Count', fontsize=16)
    plt.xlabel('Node ID', fontsize=12)
    plt.ylabel('Packet Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('network_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Node Performance Matrix
def create_node_performance_matrix():
    # Create performance categories
    node_df['reliability'] = pd.cut(
        node_df['successRate'],
        bins=[94, 96, 98, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    node_df['traffic'] = pd.cut(
        node_df['packetCount'],
        bins=[0, 1200, 2000, 3000],
        labels=['Low', 'Medium', 'High']
    )
    
    # Performance matrix as a heatmap
    plt.figure(figsize=(12, 8))
    
    # Count nodes in each category
    performance_matrix = pd.crosstab(node_df['reliability'], node_df['traffic'])
    
    ax = sns.heatmap(performance_matrix, annot=True, fmt='d', cmap='YlGnBu', linewidths=1, cbar=False)
    plt.title('Node Performance Matrix', fontsize=18)
    plt.xlabel('Traffic Level', fontsize=14)
    plt.ylabel('Reliability Level', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('node_performance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# 9. LaTeX Table as Image
def create_latex_table_image():
    # Create a figure for the table
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    
    # Sort for top 10 nodes
    top_nodes = node_df.sort_values('packetCount', ascending=False).head(10).reset_index(drop=True)
    top_nodes.index += 1  # Start index from 1 instead of 0
    
    # Create the table
    table_data = [
        [i, node, f"{rate}%", count] 
        for i, (node, rate, count) in enumerate(
            zip(top_nodes['name'], top_nodes['successRate'], top_nodes['packetCount']), 1
        )
    ]
    
    columns = ['Rank', 'Node ID', 'Success Rate', 'Packet Count']
    
    plt.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*4,
        colWidths=[0.1, 0.3, 0.25, 0.25]
    )
    
    plt.title('Top 10 Nodes by Packet Count', fontsize=20, pad=20)
    plt.tight_layout()
    
    plt.savefig('top_nodes_table.png', dpi=300, bbox_inches='tight')
    plt.close()

# 10. Service Class Performance Comparison
def plot_service_performance():
    # Path data with service class
    path_data = [
        ("voip_path_1", "VoIP"), ("streaming_path_1", "Streaming"), 
        ("standard_path_1", "Standard"), ("voip_path_2", "VoIP"),
        ("voip_path_3", "VoIP"), ("streaming_path_2", "Streaming"),
        ("streaming_path_3", "Streaming"), ("standard_path_2", "Standard"),
        ("standard_path_3", "Standard"), ("voip_path_4", "VoIP"),
        ("streaming_path_4", "Streaming"), ("standard_path_4", "Standard"),
        ("voip_path_5", "VoIP"), ("streaming_path_5", "Streaming"),
        ("high_priority_path_1", "VoIP"), ("secure_path_1", "Standard"),
        ("low_latency_path_1", "Streaming"), ("high_bandwidth_path_1", "Streaming")
    ]
    
    # Create simulated performance metrics for paths
    np.random.seed(42)  # For reproducibility
    
    # Latency by service class (VoIP should have lowest latency)
    latency_data = {
        "VoIP": np.random.normal(15, 3, len([p for p, s in path_data if s == "VoIP"])),
        "Streaming": np.random.normal(25, 5, len([p for p, s in path_data if s == "Streaming"])),
        "Standard": np.random.normal(35, 8, len([p for p, s in path_data if s == "Standard"]))
    }
    
    # Packet loss by service class
    packet_loss_data = {
        "VoIP": np.random.normal(0.2, 0.1, len([p for p, s in path_data if s == "VoIP"])),
        "Streaming": np.random.normal(0.5, 0.2, len([p for p, s in path_data if s == "Streaming"])),
        "Standard": np.random.normal(1.0, 0.4, len([p for p, s in path_data if s == "Standard"]))
    }
    
    # Bandwidth by service class
    bandwidth_data = {
        "VoIP": np.random.normal(2, 0.5, len([p for p, s in path_data if s == "VoIP"])),
        "Streaming": np.random.normal(8, 1.5, len([p for p, s in path_data if s == "Streaming"])),
        "Standard": np.random.normal(5, 1.0, len([p for p, s in path_data if s == "Standard"]))
    }
    
    # Create the data frames
    latency_df = pd.DataFrame({
        'Service Class': np.repeat(list(latency_data.keys()), 
                                  [len(v) for v in latency_data.values()]),
        'Latency (ms)': np.concatenate(list(latency_data.values()))
    })
    
    packet_loss_df = pd.DataFrame({
        'Service Class': np.repeat(list(packet_loss_data.keys()), 
                                  [len(v) for v in packet_loss_data.values()]),
        'Packet Loss (%)': np.concatenate(list(packet_loss_data.values()))
    })
    
    bandwidth_df = pd.DataFrame({
        'Service Class': np.repeat(list(bandwidth_data.keys()), 
                                  [len(v) for v in bandwidth_data.values()]),
        'Bandwidth (Mbps)': np.concatenate(list(bandwidth_data.values()))
    })
    
    # Plot the comparisons
    plt.figure(figsize=(18, 6))
    plt.suptitle('Performance Metrics by Service Class', fontsize=20, y=1.05)
    
    # Latency
    plt.subplot(131)
    sns.boxplot(x='Service Class', y='Latency (ms)', data=latency_df)
    plt.title('Latency', fontsize=16)
    plt.grid(axis='y')
    
    # Packet Loss
    plt.subplot(132)
    sns.boxplot(x='Service Class', y='Packet Loss (%)', data=packet_loss_df)
    plt.title('Packet Loss', fontsize=16)
    plt.grid(axis='y')
    
    # Bandwidth
    plt.subplot(133)
    sns.boxplot(x='Service Class', y='Bandwidth (Mbps)', data=bandwidth_df)
    plt.title('Bandwidth', fontsize=16)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('service_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all plots
plot_node_packet_count()
plot_node_success_rate()
plot_service_distribution()
plot_service_bar()
plot_success_rate_distribution()
plot_top_nodes()
create_network_dashboard()
create_node_performance_matrix()
create_latex_table_image()
plot_service_performance()

print("All visualization images have been created successfully!")