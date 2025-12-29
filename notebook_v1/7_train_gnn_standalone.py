"""
Multi-GNN Training Script
=========================
Trains and compares multiple GNN architectures (SAGE, GAT, Transformer)
for large-scale fraud detection using Neighbor Sampling.
"""

import os
import sys

# MUST be set BEFORE importing torch!
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import gc
import pickle
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GATConv, TransformerConv, to_hetero
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Current dir
sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 10
HIDDEN_DIM = 32
HEADS = 1

NUM_NEIGHBORS = [10, 5] 

# ============================================================================
# DATA LOADING
# ============================================================================
DATA_CACHE = None

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data_and_setup_loaders(force_reload=False):
    global DATA_CACHE
    
    if DATA_CACHE is not None and not force_reload:
        print("✅ Data found in memory cache. Skipping load.")
        data = DATA_CACHE
    else:
        start = time.time()
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        print("[1/6] Loading HeteroData from disk...")
        data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing {data_path}")
        
        data = torch.load(data_path)
        print(f"       ✓ Loaded in {time.time()-start:.1f}s")
        
        # Load map for labels
        print("[2/6] Loading pekerja map...")
        with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
            pekerja_map = pickle.load(f)
        pekerja_map = {str(k): v for k, v in pekerja_map.items()}
        print(f"       ✓ {len(pekerja_map):,} pekerja mapped")
        
        # Load labels
        print("[3/6] Loading fraud labels...")
        load_fraud_labels(data, pekerja_map, verbose=False)
        n_fraud = data['pekerja'].y.sum().item()
        print(f"       ✓ {n_fraud} fraud labels loaded")
        
        # Features
        print("[4/6] Computing pekerja features...")
        t0 = time.time()
        data['pekerja'].x = compute_enhanced_features(data, verbose=False)
        print(f"       ✓ {data['pekerja'].x.shape[1]} features in {time.time()-t0:.1f}s")
        
        # Init other features efficiently
        print("[5/6] Initializing other node features...")
        for node_type in ['nasabah', 'simpanan', 'pinjaman']:
            if node_type in data.node_types:
                data[node_type].x = torch.ones((data[node_type].num_nodes, 1))
                print(f"       ✓ {node_type}: {data[node_type].num_nodes:,} nodes")

        # Transaksi special handling
        if 'transaksi' in data.node_types:
            data['transaksi'].x = torch.ones((data['transaksi'].num_nodes, 1))
            print(f"       ✓ transaksi: {data['transaksi'].num_nodes:,} nodes")
             
        # Add connectivity fix
        print("[6/6] Adding reverse edges (ToUndirected)...")
        import torch_geometric.transforms as T
        data = T.ToUndirected()(data)
        print(f"       ✓ Graph now has {len(data.edge_types)} edge types")
             
        # Save to cache
        DATA_CACHE = data
        print(f"\n✅ Data loading complete in {time.time()-start:.1f}s")

    # Create Loaders
    print(f"\nSetting up NeighborLoader (Depth: {len(NUM_NEIGHBORS)} hops)...")    
    kwargs = {
        'data': data,
        'num_neighbors': NUM_NEIGHBORS,
        'batch_size': BATCH_SIZE,
        'num_workers': 0, # mac multiprocessing safety
        'shuffle': True
    }
    
    train_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].train_mask), **kwargs)
    
    kwargs['shuffle'] = False
    val_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].val_mask), **kwargs)
    test_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].test_mask), **kwargs)
    
    return data, train_loader, val_loader, test_loader

# ============================================================================
# MODELS
# ============================================================================

class BaseGNN(nn.Module):
    """Base class for homogeneous GNNs (converted to hetero later)"""
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.hidden = hidden_channels
        self.out = out_channels

class GraphSAGE(BaseGNN):
    """2-layer GraphSAGE (memory efficient)"""
    def __init__(self, hidden_channels, out_channels):
        super().__init__(hidden_channels, out_channels)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

class GAT(BaseGNN):
    """2-layer GAT (lightweight for memory)"""
    def __init__(self, hidden_channels, out_channels):
        super().__init__(hidden_channels, out_channels)
        self.conv1 = GATConv((-1, -1), hidden_channels, heads=HEADS, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, heads=1, add_self_loops=False)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

class GraphTransformer(BaseGNN):
    """2-layer Transformer (lightweight for memory)"""
    def __init__(self, hidden_channels, out_channels):
        super().__init__(hidden_channels, out_channels)
        self.conv1 = TransformerConv((-1, -1), hidden_channels, heads=HEADS)
        self.conv2 = TransformerConv((-1, -1), hidden_channels, heads=1)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

# ============================================================================
# TRAINER
# ============================================================================

def train_model(model_name, data, loaders):
    print(f"\n" + "="*60)
    print(f"TRAINING MODEL: {model_name}")
    print("="*60)
    
    train_loader, val_loader, test_loader = loaders
    
    print(f"[1/4] Initializing {model_name} model...")
    
    # Init Model
    if model_name == "SAGE":
        base_model = GraphSAGE(HIDDEN_DIM, 1)
    elif model_name == "GAT":
        base_model = GAT(HIDDEN_DIM, 1)
    elif model_name == "Transformer":
        base_model = GraphTransformer(HIDDEN_DIM, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Convert to Hetero
    print(f"[2/4] Converting to heterogeneous graph model...")
    model = to_hetero(base_model, data.metadata(), aggr='mean').to(DEVICE)
    
    # Initialize lazy
    print(f"[3/4] Running dummy forward pass...")
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():
        model(batch.x_dict, batch.edge_index_dict)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Loss
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (train_y == 0).sum() / max((train_y == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    n_params = sum(p.numel() for p in model.parameters())
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    
    print(f"[4/4] Setup complete!")
    print(f"       Model params: {n_params:,}")
    print(f"       Train batches: {n_train_batches}, Val batches: {n_val_batches}")
    print(f"       Pos weight: {pos_weight:.2f}")
    print("-" * 60)
    
    # Loop
    best_auc = 0
    history = []
    train_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_samples = 0
        
        # TQDM for batches
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False)
        
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(batch.x_dict, batch.edge_index_dict)
            
            # Slice for target nodes
            bs = batch['pekerja'].batch_size
            out_pekerja = out['pekerja'][:bs].squeeze(-1)
            y = batch['pekerja'].y[:bs].float()
            
            loss = criterion(out_pekerja, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bs
            total_samples += bs
            
            # Update pbar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / total_samples
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                bs = batch['pekerja'].batch_size
                out = model(batch.x_dict, batch.edge_index_dict)
                val_preds.extend(torch.sigmoid(out['pekerja'][:bs].squeeze(-1)).cpu().numpy())
                val_labels.extend(batch['pekerja'].y[:bs].cpu().numpy())
                
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.5
        _, val_f1 = find_optimal_threshold(np.array(val_labels), np.array(val_preds))
        
        print(f"   Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"best_gnn_{model_name}.pt")
    
    # Test
    print(f"\nEvaluating {model_name} on Test Set...")
    model.load_state_dict(torch.load(f"best_gnn_{model_name}.pt"))
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            batch = batch.to(DEVICE)
            bs = batch['pekerja'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)
            test_preds.extend(torch.sigmoid(out['pekerja'][:bs].squeeze(-1)).cpu().numpy())
            test_labels.extend(batch['pekerja'].y[:bs].cpu().numpy())
            
    test_auc = roc_auc_score(test_labels, test_preds)
    opt_thresh, test_f1 = find_optimal_threshold(np.array(test_labels), np.array(test_preds))
    
    # Detailed Evaluation
    test_binary = (np.array(test_preds) > opt_thresh).astype(int)
    
    print("-" * 60)
    print(f"✅ {model_name} Final Results (thresh={opt_thresh:.3f}):")
    print(f"   AUC: {test_auc:.4f}")
    print(f"   F1 : {test_f1:.4f}")
    print("-" * 20)
    print("Classification Report:")
    print(classification_report(test_labels, test_binary, target_names=['Non-Fraud', 'Fraud']))
    print("-" * 20)
    print("Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_binary)
    print(cm)
    
    return {'auc': test_auc, 'f1': test_f1}

# ============================================================================
# MAIN
# ============================================================================
def run():
    print("🚀 Starting Multi-GNN Training Pipeline")
    print("   (Memory-optimized: 2-layer models, reduced hidden dim)")
    
    try:
        data, train_l, val_l, test_l = load_data_and_setup_loaders()
        loaders = (train_l, val_l, test_l)
        
        results = {}
        
        def cleanup():
            """Free memory between models"""
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # 1. GraphSAGE (Baseline)
        results['SAGE'] = train_model('SAGE', data, loaders)
        cleanup()
        
        # 2. GAT (Attention) - may use CPU fallback
        print("\n⚠️  GAT uses CPU fallback for some ops (slower)")
        results['GAT'] = train_model('GAT', data, loaders)
        cleanup()
        
        # 3. Transformer (Modern) - may use CPU fallback
        print("\n⚠️  Transformer uses CPU fallback for some ops (slower)")
        results['Transformer'] = train_model('Transformer', data, loaders)
        cleanup()
        
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        for name, res in results.items():
            print(f"{name:12s}: AUC={res['auc']:.4f}, F1={res['f1']:.4f}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
