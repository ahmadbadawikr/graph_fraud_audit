"""
Graph EDA & Data Visualization
==============================
Comprehensive analysis to diagnose data quality and identify
limitations that may be preventing better model performance.

Generates:
1. Degree Distribution Analysis (Fraud vs Non-Fraud)
2. Neighborhood Connectivity Patterns
3. Feature Separability Visualization (PCA, t-SNE)
4. Class Imbalance Diagnostics
5. Graph Structure Statistics
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Must set before torch import
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from torch_geometric.utils import degree
import torch_geometric.transforms as T

sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_PLOTS = "/Users/kasyfur/graph_fraud_audit/notebook_v1/eda_plots"
os.makedirs(OUTPUT_PLOTS, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_graph_data():
    print("\n[Data] Loading HeteroData...")
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    
    print("[Data] Loading labels...")
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    load_fraud_labels(data, pekerja_map, verbose=False)
    
    print("[Data] Computing features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
    # Get masks
    n_fraud = data['pekerja'].y.sum().item()
    n_total = data['pekerja'].num_nodes
    print(f"✅ Data loaded: {n_total} pekerja, {n_fraud} fraud ({100*n_fraud/n_total:.2f}%)")
    
    return data

# ============================================================================
# 1. DEGREE DISTRIBUTION ANALYSIS
# ============================================================================
def analyze_degree_distribution(data):
    print("\n[1/5] Analyzing Degree Distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get labels
    y = data['pekerja'].y.numpy()
    fraud_mask = y == 1
    non_fraud_mask = y == 0
    
    # Compute degrees for each edge type involving pekerja
    pekerja_degrees = defaultdict(lambda: np.zeros(data['pekerja'].num_nodes))
    
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        
        if src_type == 'pekerja':
            deg = degree(edge_index[0], num_nodes=data['pekerja'].num_nodes).numpy()
            pekerja_degrees[f'out_{rel}_{dst_type}'] = deg
        if dst_type == 'pekerja':
            deg = degree(edge_index[1], num_nodes=data['pekerja'].num_nodes).numpy()
            pekerja_degrees[f'in_{rel}_{src_type}'] = deg
    
    # Total degree
    total_out = np.zeros(data['pekerja'].num_nodes)
    total_in = np.zeros(data['pekerja'].num_nodes)
    for k, v in pekerja_degrees.items():
        if k.startswith('out_'):
            total_out += v
        elif k.startswith('in_'):
            total_in += v
    total_degree = total_out + total_in
    
    # Plot 1: Total Degree Distribution
    ax = axes[0, 0]
    ax.hist(total_degree[non_fraud_mask], bins=50, alpha=0.6, label='Non-Fraud', density=True)
    ax.hist(total_degree[fraud_mask], bins=50, alpha=0.6, label='Fraud', density=True)
    ax.set_xlabel('Total Degree')
    ax.set_ylabel('Density')
    ax.set_title('Degree Distribution: Fraud vs Non-Fraud')
    ax.legend()
    ax.set_xlim(0, np.percentile(total_degree, 99))
    
    # Plot 2: Log Degree Distribution (Better for scale)
    ax = axes[0, 1]
    log_degree = np.log1p(total_degree)
    ax.hist(log_degree[non_fraud_mask], bins=50, alpha=0.6, label='Non-Fraud', density=True)
    ax.hist(log_degree[fraud_mask], bins=50, alpha=0.6, label='Fraud', density=True)
    ax.set_xlabel('Log(1 + Total Degree)')
    ax.set_ylabel('Density')
    ax.set_title('Log Degree Distribution: Fraud vs Non-Fraud')
    ax.legend()
    
    # Plot 3: Box Plot Comparison
    ax = axes[1, 0]
    df = pd.DataFrame({
        'Degree': total_degree,
        'Label': ['Fraud' if f else 'Non-Fraud' for f in fraud_mask]
    })
    sns.boxplot(data=df, x='Label', y='Degree', ax=ax)
    ax.set_title('Degree Box Plot by Label')
    ax.set_ylim(0, np.percentile(total_degree, 95))
    
    # Plot 4: In vs Out Degree Scatter
    ax = axes[1, 1]
    ax.scatter(total_out[non_fraud_mask], total_in[non_fraud_mask], 
               alpha=0.3, s=10, label='Non-Fraud', c='blue')
    ax.scatter(total_out[fraud_mask], total_in[fraud_mask], 
               alpha=0.8, s=30, label='Fraud', c='red', marker='x')
    ax.set_xlabel('Out-Degree')
    ax.set_ylabel('In-Degree')
    ax.set_title('In-Degree vs Out-Degree (Fraud highlighted)')
    ax.legend()
    ax.set_xlim(0, np.percentile(total_out, 99))
    ax.set_ylim(0, np.percentile(total_in, 99))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/1_degree_distribution.png", dpi=150)
    plt.close()
    print(f"   Saved: {OUTPUT_PLOTS}/1_degree_distribution.png")
    
    # Stats
    print(f"   Fraud Mean Degree: {total_degree[fraud_mask].mean():.2f}")
    print(f"   Non-Fraud Mean Degree: {total_degree[non_fraud_mask].mean():.2f}")
    
    return total_degree, fraud_mask

# ============================================================================
# 2. NEIGHBORHOOD ANALYSIS
# ============================================================================
def analyze_neighborhoods(data, total_degree, fraud_mask):
    print("\n[2/5] Analyzing Neighborhood Patterns...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get nasabah connections for each pekerja
    if ('pekerja', 'handles', 'nasabah') in data.edge_types:
        edge_index = data[('pekerja', 'handles', 'nasabah')].edge_index
        nasabah_per_pekerja = degree(edge_index[0], num_nodes=data['pekerja'].num_nodes).numpy()
    else:
        nasabah_per_pekerja = np.zeros(data['pekerja'].num_nodes)
    
    # Plot 1: Nasabah Count Distribution
    ax = axes[0]
    ax.hist(nasabah_per_pekerja[~fraud_mask], bins=50, alpha=0.6, label='Non-Fraud', density=True)
    ax.hist(nasabah_per_pekerja[fraud_mask], bins=50, alpha=0.6, label='Fraud', density=True)
    ax.set_xlabel('Number of Nasabah Handled')
    ax.set_ylabel('Density')
    ax.set_title('Nasabah per Pekerja: Fraud vs Non-Fraud')
    ax.legend()
    ax.set_xlim(0, np.percentile(nasabah_per_pekerja, 99))
    
    # Plot 2: Degree vs Fraud Probability
    ax = axes[1]
    # Bin degrees and compute fraud rate per bin
    bins = np.percentile(total_degree, np.linspace(0, 100, 21))
    bins = np.unique(bins)
    bin_indices = np.digitize(total_degree, bins)
    
    fraud_rates = []
    bin_centers = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 10:
            fraud_rate = fraud_mask[mask].mean()
            fraud_rates.append(fraud_rate)
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    
    ax.plot(bin_centers, fraud_rates, 'o-', linewidth=2, markersize=8)
    ax.axhline(fraud_mask.mean(), color='red', linestyle='--', label=f'Overall Rate ({fraud_mask.mean():.2%})')
    ax.set_xlabel('Degree Bin Center')
    ax.set_ylabel('Fraud Rate')
    ax.set_title('Fraud Rate by Degree Bin')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/2_neighborhood_analysis.png", dpi=150)
    plt.close()
    print(f"   Saved: {OUTPUT_PLOTS}/2_neighborhood_analysis.png")

# ============================================================================
# 3. FEATURE SEPARABILITY (PCA/t-SNE)
# ============================================================================
def analyze_feature_separability(data, fraud_mask):
    print("\n[3/5] Analyzing Feature Separability...")
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    X = data['pekerja'].x.numpy()
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    ax = axes[0]
    ax.scatter(X_pca[~fraud_mask, 0], X_pca[~fraud_mask, 1], 
               alpha=0.3, s=10, label='Non-Fraud', c='blue')
    ax.scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1], 
               alpha=0.8, s=40, label='Fraud', c='red', marker='x')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA: Feature Space (Fraud highlighted)')
    ax.legend()
    
    # t-SNE (sample for speed)
    print("   Running t-SNE (this may take a moment)...")
    n_samples = min(2000, len(X_scaled))
    sample_idx = np.random.choice(len(X_scaled), n_samples, replace=False)
    X_sample = X_scaled[sample_idx]
    y_sample = fraud_mask[sample_idx]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    ax = axes[1]
    ax.scatter(X_tsne[~y_sample, 0], X_tsne[~y_sample, 1], 
               alpha=0.3, s=10, label='Non-Fraud', c='blue')
    ax.scatter(X_tsne[y_sample, 0], X_tsne[y_sample, 1], 
               alpha=0.8, s=40, label='Fraud', c='red', marker='x')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE: Feature Space (n={n_samples})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/3_feature_separability.png", dpi=150)
    plt.close()
    print(f"   Saved: {OUTPUT_PLOTS}/3_feature_separability.png")
    
    # Explained variance
    pca_full = PCA().fit(X_scaled)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = np.searchsorted(cumsum, 0.90) + 1
    print(f"   PCA: {n_90} components explain 90% variance")

# ============================================================================
# 4. CLASS IMBALANCE DIAGNOSTICS
# ============================================================================
def analyze_class_imbalance(data, fraud_mask):
    print("\n[4/5] Analyzing Class Imbalance...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Class Distribution
    ax = axes[0]
    counts = [np.sum(~fraud_mask), np.sum(fraud_mask)]
    labels = ['Non-Fraud', 'Fraud']
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(labels, counts, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                f'{count}\n({100*count/sum(counts):.1f}%)', ha='center', fontsize=10)
    
    # Plot 2: Train/Val/Test Split
    ax = axes[1]
    train_mask = data['pekerja'].train_mask.numpy()
    val_mask = data['pekerja'].val_mask.numpy()
    test_mask = data['pekerja'].test_mask.numpy()
    
    splits = ['Train', 'Val', 'Test']
    fraud_counts = [
        fraud_mask[train_mask].sum(),
        fraud_mask[val_mask].sum(),
        fraud_mask[test_mask].sum()
    ]
    non_fraud_counts = [
        (~fraud_mask[train_mask]).sum(),
        (~fraud_mask[val_mask]).sum(),
        (~fraud_mask[test_mask]).sum()
    ]
    
    x = np.arange(len(splits))
    width = 0.35
    ax.bar(x - width/2, non_fraud_counts, width, label='Non-Fraud', color='#3498db')
    ax.bar(x + width/2, fraud_counts, width, label='Fraud', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution by Split')
    ax.legend()
    
    # Add fraud rate labels
    for i, (nf, f) in enumerate(zip(non_fraud_counts, fraud_counts)):
        rate = 100 * f / (nf + f)
        ax.text(i, max(nf, f) + 50, f'{rate:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/4_class_imbalance.png", dpi=150)
    plt.close()
    print(f"   Saved: {OUTPUT_PLOTS}/4_class_imbalance.png")
    
    # Stats
    print(f"   Train Fraud Rate: {100*fraud_counts[0]/(fraud_counts[0]+non_fraud_counts[0]):.2f}%")
    print(f"   Val Fraud Rate: {100*fraud_counts[1]/(fraud_counts[1]+non_fraud_counts[1]):.2f}%")
    print(f"   Test Fraud Rate: {100*fraud_counts[2]/(fraud_counts[2]+non_fraud_counts[2]):.2f}%")

# ============================================================================
# 5. GRAPH STRUCTURE STATISTICS
# ============================================================================
def analyze_graph_structure(data):
    print("\n[5/5] Analyzing Graph Structure...")
    
    stats = []
    for node_type in data.node_types:
        stats.append({
            'Type': node_type,
            'Nodes': data[node_type].num_nodes,
            'Features': data[node_type].x.shape[1] if hasattr(data[node_type], 'x') and data[node_type].x is not None else 0
        })
    
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        n_edges = data[edge_type].edge_index.shape[1]
        stats.append({
            'Type': f'{src}-[{rel}]->{dst}',
            'Nodes': n_edges,
            'Features': 0
        })
    
    df = pd.DataFrame(stats)
    
    # Create summary figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    node_df = df[~df['Type'].str.contains('->')].copy()
    edge_df = df[df['Type'].str.contains('->')].copy()
    
    # Node counts
    x = np.arange(len(node_df))
    bars = ax.bar(x, node_df['Nodes'], color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(node_df['Type'], rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Graph Structure: Node Type Counts')
    ax.set_yscale('log')
    
    for bar, count in zip(bars, node_df['Nodes']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{count:,}', ha='center', fontsize=9, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PLOTS}/5_graph_structure.png", dpi=150)
    plt.close()
    print(f"   Saved: {OUTPUT_PLOTS}/5_graph_structure.png")
    
    # Print stats
    print("\n   Node Type Statistics:")
    print(node_df.to_string(index=False))
    print("\n   Edge Type Statistics:")
    print(edge_df[['Type', 'Nodes']].rename(columns={'Nodes': 'Edges'}).to_string(index=False))

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("GRAPH EDA & DATA VISUALIZATION")
    print("="*60)
    
    data = load_graph_data()
    
    total_degree, fraud_mask = analyze_degree_distribution(data)
    analyze_neighborhoods(data, total_degree, fraud_mask)
    analyze_feature_separability(data, fraud_mask)
    analyze_class_imbalance(data, fraud_mask)
    analyze_graph_structure(data)
    
    print("\n" + "="*60)
    print("EDA COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {OUTPUT_PLOTS}/")
    print("\nKey Questions Answered:")
    print("  1. Do fraud nodes have distinct degree patterns?")
    print("  2. Are features separable in reduced dimensions?")
    print("  3. Is class imbalance consistent across splits?")
    print("  4. What is the graph scale and connectivity?")

if __name__ == "__main__":
    main()
