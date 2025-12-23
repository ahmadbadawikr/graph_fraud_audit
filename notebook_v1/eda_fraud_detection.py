#!/usr/bin/env python3
"""
Fraud Detection - Exploratory Data Analysis (EDA)
==================================================
Comprehensive analysis of graph data for fraud detection.
Paste this code into Jupyter notebook cells.
"""

# ============================================================================
# CELL 1: IMPORTS AND DATA LOADING
# ============================================================================

import os
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuration
ROOT_DIR = "/Volumes/Backup Plus/Zaman/graph"
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "processed_fast")
MAP_DIR = os.path.join(ROOT_DIR, "map_id")

plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 8)

# Load HeteroData
print("Loading HeteroData...")
data = torch.load(os.path.join(OUTPUT_DIR, "heterodata.pt"))

# Load fraud labels
print("Loading fraud labels...")
fraud_df = pd.read_csv(os.path.join(DATA_DIR, "labeled_fraud.csv"), low_memory=False)

# Load pekerja mapping
with open(os.path.join(MAP_DIR, "map_pekerja.pkl"), 'rb') as f:
    pekerja_map = pickle.load(f)
pekerja_map = {str(k): v for k, v in pekerja_map.items()}

print("✅ Data loaded!")


# ============================================================================
# CELL 2: GRAPH STRUCTURE ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("1. GRAPH STRUCTURE ANALYSIS")
print("=" * 70)

# Node counts
print("\n📊 NODE COUNTS:")
node_counts = {}
for nt in data.node_types:
    count = data[nt].num_nodes
    node_counts[nt] = count
    print(f"  {nt:15s}: {count:>15,}")

print(f"\n  {'TOTAL':15s}: {sum(node_counts.values()):>15,}")

# Edge counts
print("\n📊 EDGE COUNTS:")
edge_counts = {}
total_edges = 0
for et in data.edge_types:
    count = data[et].edge_index.shape[1]
    edge_counts[et[1]] = count
    total_edges += count
    print(f"  {et[1]:20s}: {count:>15,}")

print(f"\n  {'TOTAL':20s}: {total_edges:>15,}")

# Graph density
print("\n📊 GRAPH METRICS:")
total_nodes = sum(node_counts.values())
max_possible_edges = total_nodes * (total_nodes - 1)
density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
print(f"  Total nodes: {total_nodes:,}")
print(f"  Total edges: {total_edges:,}")
print(f"  Graph density: {density:.10f}")

# Visualize node/edge distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Node counts bar chart
ax = axes[0]
names = list(node_counts.keys())
values = [node_counts[n] for n in names]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
bars = ax.barh(names, values, color=colors)
ax.set_xlabel('Count')
ax.set_title('Node Counts by Type')
ax.set_xscale('log')
for bar, val in zip(bars, values):
    ax.text(val * 1.1, bar.get_y() + bar.get_height()/2, f'{val:,}', 
            va='center', fontsize=9)

# Edge counts bar chart
ax = axes[1]
names = list(edge_counts.keys())
values = [edge_counts[n] for n in names]
colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))
bars = ax.barh(names, values, color=colors)
ax.set_xlabel('Count')
ax.set_title('Edge Counts by Type')
ax.set_xscale('log')
for bar, val in zip(bars, values):
    ax.text(val * 1.1, bar.get_y() + bar.get_height()/2, f'{val:,}', 
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_graph_structure.png'), dpi=150)
plt.show()


# ============================================================================
# CELL 3: FRAUD LABEL ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("2. FRAUD LABEL ANALYSIS")
print("=" * 70)

# Basic stats from CSV
print("\n📊 FRAUD CSV STATISTICS:")
print(f"  Total rows: {len(fraud_df):,}")
print(f"  Unique fraud PNs: {fraud_df['PN'].nunique():,}")
print(f"  Columns: {fraud_df.columns.tolist()}")

# Match fraud labels to graph
fraud_pns = set(fraud_df['PN'].dropna().astype(str).unique())
num_pekerja = data['pekerja'].num_nodes

fraud_labels = torch.zeros(num_pekerja, dtype=torch.long)
matched_frauds = 0
for pn in fraud_pns:
    if pn in pekerja_map:
        node_id = pekerja_map[pn]
        if node_id < num_pekerja:
            fraud_labels[node_id] = 1
            matched_frauds += 1

print(f"\n📊 LABEL DISTRIBUTION (PEKERJA):")
n_fraud = fraud_labels.sum().item()
n_nonfraud = num_pekerja - n_fraud
print(f"  Non-Fraud: {n_nonfraud:,} ({100*n_nonfraud/num_pekerja:.2f}%)")
print(f"  Fraud:     {n_fraud:,} ({100*n_fraud/num_pekerja:.2f}%)")
print(f"  Imbalance ratio: {n_nonfraud/n_fraud:.1f}:1")
print(f"  Matched from CSV: {matched_frauds}/{len(fraud_pns)}")

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
ax = axes[0]
sizes = [n_nonfraud, n_fraud]
labels = [f'Non-Fraud\n{n_nonfraud:,}', f'Fraud\n{n_fraud:,}']
colors = ['#2ecc71', '#e74c3c']
explode = (0, 0.1)
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.set_title('Fraud vs Non-Fraud Distribution')

# Bar chart
ax = axes[1]
bars = ax.bar(['Non-Fraud', 'Fraud'], [n_nonfraud, n_fraud], color=colors)
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
for bar, val in zip(bars, [n_nonfraud, n_fraud]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,}',
            ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_fraud_distribution.png'), dpi=150)
plt.show()

# Temporal analysis
if 'Awal Fraud Terjadi' in fraud_df.columns:
    print("\n📊 TEMPORAL ANALYSIS:")
    fraud_df['fraud_start'] = pd.to_datetime(fraud_df['Awal Fraud Terjadi'], errors='coerce')
    fraud_df['fraud_year'] = fraud_df['fraud_start'].dt.year
    year_counts = fraud_df['fraud_year'].value_counts().sort_index()
    print(year_counts.to_string())
    
    plt.figure(figsize=(10, 4))
    year_counts.plot(kind='bar', color=plt.cm.Reds(np.linspace(0.3, 0.9, len(year_counts))))
    plt.xlabel('Year')
    plt.ylabel('Fraud Cases')
    plt.title('Fraud Cases Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_fraud_temporal.png'), dpi=150)
    plt.show()


# ============================================================================
# CELL 4: DEGREE DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("3. DEGREE DISTRIBUTION ANALYSIS")
print("=" * 70)

from torch_geometric.utils import degree

def compute_degrees(data, node_type):
    """Compute in-degree and out-degree for a node type."""
    num_nodes = data[node_type].num_nodes
    in_degree = torch.zeros(num_nodes)
    out_degree = torch.zeros(num_nodes)
    
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        
        if dst_type == node_type and ei.shape[1] > 0:
            deg = degree(ei[1], num_nodes=num_nodes)
            in_degree += deg
        
        if src_type == node_type and ei.shape[1] > 0:
            deg = degree(ei[0], num_nodes=num_nodes)
            out_degree += deg
    
    return in_degree, out_degree

# Compute degrees for pekerja
print("\n📊 PEKERJA DEGREE STATS:")
in_deg, out_deg = compute_degrees(data, 'pekerja')
print(f"  In-degree:  mean={in_deg.mean():.2f}, std={in_deg.std():.2f}, max={in_deg.max():.0f}")
print(f"  Out-degree: mean={out_deg.mean():.2f}, std={out_deg.std():.2f}, max={out_deg.max():.0f}")

# Compare fraud vs non-fraud
fraud_mask = fraud_labels == 1
nonfraud_mask = fraud_labels == 0

print("\n📊 FRAUD vs NON-FRAUD DEGREE COMPARISON:")
print(f"  Non-Fraud in-degree:  mean={in_deg[nonfraud_mask].mean():.2f}, std={in_deg[nonfraud_mask].std():.2f}")
print(f"  Fraud in-degree:      mean={in_deg[fraud_mask].mean():.2f}, std={in_deg[fraud_mask].std():.2f}")
print(f"  Non-Fraud out-degree: mean={out_deg[nonfraud_mask].mean():.2f}, std={out_deg[nonfraud_mask].std():.2f}")
print(f"  Fraud out-degree:     mean={out_deg[fraud_mask].mean():.2f}, std={out_deg[fraud_mask].std():.2f}")

# Visualize degree distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# In-degree histogram
ax = axes[0, 0]
ax.hist(in_deg[nonfraud_mask].numpy(), bins=50, alpha=0.7, label='Non-Fraud', color='green')
ax.hist(in_deg[fraud_mask].numpy(), bins=50, alpha=0.7, label='Fraud', color='red')
ax.set_xlabel('In-Degree')
ax.set_ylabel('Count')
ax.set_title('In-Degree Distribution (Pekerja)')
ax.legend()
ax.set_yscale('log')

# Out-degree histogram
ax = axes[0, 1]
ax.hist(out_deg[nonfraud_mask].numpy(), bins=50, alpha=0.7, label='Non-Fraud', color='green')
ax.hist(out_deg[fraud_mask].numpy(), bins=50, alpha=0.7, label='Fraud', color='red')
ax.set_xlabel('Out-Degree')
ax.set_ylabel('Count')
ax.set_title('Out-Degree Distribution (Pekerja)')
ax.legend()
ax.set_yscale('log')

# In-degree boxplot
ax = axes[1, 0]
in_deg_data = [in_deg[nonfraud_mask].numpy(), in_deg[fraud_mask].numpy()]
bp = ax.boxplot(in_deg_data, labels=['Non-Fraud', 'Fraud'], patch_artist=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('red')
ax.set_ylabel('In-Degree')
ax.set_title('In-Degree by Class')

# Out-degree boxplot
ax = axes[1, 1]
out_deg_data = [out_deg[nonfraud_mask].numpy(), out_deg[fraud_mask].numpy()]
bp = ax.boxplot(out_deg_data, labels=['Non-Fraud', 'Fraud'], patch_artist=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('red')
ax.set_ylabel('Out-Degree')
ax.set_title('Out-Degree by Class')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_degree_distribution.png'), dpi=150)
plt.show()


# ============================================================================
# CELL 5: CONNECTIVITY PATTERN ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("4. CONNECTIVITY PATTERN ANALYSIS")
print("=" * 70)

# Get edge indices
is_pekerja_edge = data[('nasabah', 'is_pekerja', 'pekerja')].edge_index
has_simpanan_edge = data[('nasabah', 'has_simpanan', 'simpanan')].edge_index
has_pinjaman_edge = data[('nasabah', 'has_pinjaman', 'pinjaman')].edge_index
debit_edge = data[('simpanan', 'debit', 'transaksi')].edge_index
credit_edge = data[('transaksi', 'credit', 'simpanan')].edge_index

# Nasabah per pekerja
print("\n📊 NASABAH PER PEKERJA:")
nasabah_per_pekerja = torch.zeros(num_pekerja)
for i in range(is_pekerja_edge.shape[1]):
    nasabah_per_pekerja[is_pekerja_edge[1, i]] += 1

print(f"  Overall: mean={nasabah_per_pekerja.mean():.2f}, max={nasabah_per_pekerja.max():.0f}")
print(f"  Non-Fraud: mean={nasabah_per_pekerja[nonfraud_mask].mean():.2f}")
print(f"  Fraud:     mean={nasabah_per_pekerja[fraud_mask].mean():.2f}")

# Aggregate transactions through graph
print("\n📊 COMPUTING TRANSACTION PATTERNS (may take a minute)...")

# Transaction count per simpanan
num_simpanan = data['simpanan'].num_nodes
tx_per_simpanan = torch.zeros(num_simpanan)
tx_per_simpanan.scatter_add_(0, debit_edge[0], torch.ones(debit_edge.shape[1]))
tx_per_simpanan.scatter_add_(0, credit_edge[1], torch.ones(credit_edge.shape[1]))

# Simpanan per nasabah
num_nasabah = data['nasabah'].num_nodes
simpanan_per_nasabah = torch.zeros(num_nasabah)
simpanan_per_nasabah.scatter_add_(0, has_simpanan_edge[0], torch.ones(has_simpanan_edge.shape[1]))

# Transaction count per nasabah
tx_per_nasabah = torch.zeros(num_nasabah)
for i in range(has_simpanan_edge.shape[1]):
    nasabah_id = has_simpanan_edge[0, i].item()
    simpanan_id = has_simpanan_edge[1, i].item()
    tx_per_nasabah[nasabah_id] += tx_per_simpanan[simpanan_id]

# Aggregate to pekerja
simpanan_per_pekerja = torch.zeros(num_pekerja)
tx_per_pekerja = torch.zeros(num_pekerja)

for i in range(is_pekerja_edge.shape[1]):
    nasabah_id = is_pekerja_edge[0, i].item()
    pekerja_id = is_pekerja_edge[1, i].item()
    simpanan_per_pekerja[pekerja_id] += simpanan_per_nasabah[nasabah_id]
    tx_per_pekerja[pekerja_id] += tx_per_nasabah[nasabah_id]

print("\n📊 SIMPANAN PER PEKERJA:")
print(f"  Overall: mean={simpanan_per_pekerja.mean():.2f}, max={simpanan_per_pekerja.max():.0f}")
print(f"  Non-Fraud: mean={simpanan_per_pekerja[nonfraud_mask].mean():.2f}")
print(f"  Fraud:     mean={simpanan_per_pekerja[fraud_mask].mean():.2f}")

print("\n📊 TRANSACTIONS PER PEKERJA:")
print(f"  Overall: mean={tx_per_pekerja.mean():.2f}, max={tx_per_pekerja.max():.0f}")
print(f"  Non-Fraud: mean={tx_per_pekerja[nonfraud_mask].mean():.2f}")
print(f"  Fraud:     mean={tx_per_pekerja[fraud_mask].mean():.2f}")

# Visualize connectivity patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Nasabah per pekerja
ax = axes[0, 0]
ax.hist(nasabah_per_pekerja[nonfraud_mask].numpy(), bins=30, alpha=0.7, label='Non-Fraud', color='green')
ax.hist(nasabah_per_pekerja[fraud_mask].numpy(), bins=30, alpha=0.7, label='Fraud', color='red')
ax.set_xlabel('Nasabah Count')
ax.set_ylabel('Pekerja Count')
ax.set_title('Nasabah per Pekerja')
ax.legend()

# Simpanan per pekerja
ax = axes[0, 1]
ax.hist(simpanan_per_pekerja[nonfraud_mask].numpy(), bins=30, alpha=0.7, label='Non-Fraud', color='green')
ax.hist(simpanan_per_pekerja[fraud_mask].numpy(), bins=30, alpha=0.7, label='Fraud', color='red')
ax.set_xlabel('Simpanan Count')
ax.set_ylabel('Pekerja Count')
ax.set_title('Simpanan per Pekerja')
ax.legend()

# Transactions per pekerja
ax = axes[1, 0]
ax.hist(np.log1p(tx_per_pekerja[nonfraud_mask].numpy()), bins=30, alpha=0.7, label='Non-Fraud', color='green')
ax.hist(np.log1p(tx_per_pekerja[fraud_mask].numpy()), bins=30, alpha=0.7, label='Fraud', color='red')
ax.set_xlabel('Log(1 + Transaction Count)')
ax.set_ylabel('Pekerja Count')
ax.set_title('Transactions per Pekerja (Log Scale)')
ax.legend()

# Boxplot comparison
ax = axes[1, 1]
features_to_plot = [
    ('Nasabah', nasabah_per_pekerja),
    ('Simpanan', simpanan_per_pekerja),
    ('Log(Tx)', torch.log1p(tx_per_pekerja))
]

x = np.arange(len(features_to_plot))
width = 0.35

fraud_means = [f[1][fraud_mask].mean().item() for f in features_to_plot]
nonfraud_means = [f[1][nonfraud_mask].mean().item() for f in features_to_plot]

bars1 = ax.bar(x - width/2, nonfraud_means, width, label='Non-Fraud', color='green')
bars2 = ax.bar(x + width/2, fraud_means, width, label='Fraud', color='red')
ax.set_xticks(x)
ax.set_xticklabels([f[0] for f in features_to_plot])
ax.set_ylabel('Mean Value')
ax.set_title('Mean Connectivity Metrics by Class')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_connectivity_patterns.png'), dpi=150)
plt.show()


# ============================================================================
# CELL 6: FEATURE CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("5. FEATURE CORRELATION WITH FRAUD")
print("=" * 70)

# Build feature matrix
features_dict = {
    'nasabah_count': nasabah_per_pekerja,
    'simpanan_count': simpanan_per_pekerja,
    'tx_count': tx_per_pekerja,
    'in_degree': in_deg,
    'out_degree': out_deg,
    'log_nasabah': torch.log1p(nasabah_per_pekerja),
    'log_simpanan': torch.log1p(simpanan_per_pekerja),
    'log_tx': torch.log1p(tx_per_pekerja),
    'fraud_label': fraud_labels.float(),
}

# Create DataFrame
feature_df = pd.DataFrame({k: v.numpy() for k, v in features_dict.items()})

# Correlation with fraud
print("\n📊 CORRELATION WITH FRAUD LABEL:")
correlations = feature_df.corr()['fraud_label'].drop('fraud_label').sort_values(ascending=False)
print(correlations.to_string())

# Visualize correlations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation bar chart
ax = axes[0]
colors = ['green' if c > 0 else 'red' for c in correlations.values]
correlations.plot(kind='barh', color=colors, ax=ax)
ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
ax.set_xlabel('Correlation with Fraud')
ax.set_title('Feature Correlation with Fraud Label')

# Full correlation heatmap
ax = axes[1]
corr_matrix = feature_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, ax=ax, annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eda_feature_correlation.png'), dpi=150)
plt.show()


# ============================================================================
# CELL 7: EDA SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("📊 EDA SUMMARY")
print("=" * 70)

print("""
KEY FINDINGS:
=============

1. GRAPH STRUCTURE:
   - 5 node types, 5 edge types
   - Very sparse graph (typical for real-world data)
   - Transaksi is the largest node type (185M nodes)

2. CLASS IMBALANCE:
   - ~8% fraud rate (highly imbalanced)
   - Need to use class weights or sampling strategies

3. DEGREE PATTERNS:
   - Fraud employees may have different connection patterns
   - Check the mean comparisons above for insights

4. MOST DISCRIMINATIVE FEATURES:
   - See correlation analysis above
   - Log-transformed features may be more useful

5. RECOMMENDATIONS FOR MODELING:
   - Use aggregated features (nasabah/simpanan/tx counts)
   - Apply log transformations for skewed distributions
   - Use class weighting due to imbalance
   - Consider ratio features (tx per nasabah, etc.)
""")

print("\n✅ EDA Complete! Visualizations saved to:", OUTPUT_DIR)
