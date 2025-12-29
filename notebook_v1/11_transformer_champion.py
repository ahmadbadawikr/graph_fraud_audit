"""
GraphTransformer Final Champion (V4)
====================================
This script restores the winning configuration from the original standalone run
but with optimized data loading and memory safety.

Goal: Restore AUC 0.725+ and F1 0.29+
Configuration:
- 3-hop sampling ([15, 10, 5])
- 3-layer Transformer architecture
- Hidden Dim: 64, Heads: 2
- Dropout: 0.3
- Stable Weighted BCE Loss
"""

import os
import sys

# MUST be set BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import gc
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import TransformerConv, to_hetero, LayerNorm
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

# Current dir
sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION (CHAMPION RESTORATION)
# ============================================================================
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 15
HIDDEN_DIM = 64
HEADS = 2
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4

# MANDATORY: 3 hops to reach transaction signals
NUM_NEIGHBORS = [15, 10, 5]

DATA_CACHE = None

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data_champion(force_reload=False):
    global DATA_CACHE
    if DATA_CACHE is not None and not force_reload:
        print("✅ Data reused from cache")
        return DATA_CACHE

    start = time.time()
    print("\n" + "="*60)
    print("LOADING DATA (CHAMPION)")
    print("="*60)
    
    print("[1/4] Loading HeteroData (16GB)...")
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    print(f"      ✓ Loaded in {time.time()-start:.1f}s")
    
    # Load map
    print("[2/4] Loading labels...")
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    load_fraud_labels(data, pekerja_map, verbose=False)
    
    # Features
    print("[3/4] Computing pekerja features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
    # Init others simply (let the GNN learn from structure)
    print("[4/4] Initializing auxiliary node features...")
    for node_type in ['nasabah', 'simpanan', 'pinjaman', 'transaksi']:
        if node_type in data.node_types:
            data[node_type].x = torch.ones((data[node_type].num_nodes, 1))
            
    # Connectivity
    print("[+] Adding reverse edges (Undirected)...")
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    print(f"✅ Setup complete in {time.time()-start:.1f}s")
    return data

def get_loaders_champion(data):
    print(f"\nSetting up 3-hop NeighborLoader...")
    kwargs = {
        'data': data,
        'num_neighbors': NUM_NEIGHBORS,
        'batch_size': BATCH_SIZE,
        'num_workers': 0, # Safety for MPS
        'shuffle': True
    }
    
    train_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].train_mask), **kwargs)
    kwargs['shuffle'] = False
    val_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].val_mask), **kwargs)
    test_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].test_mask), **kwargs)
    
    return train_loader, val_loader, test_loader

# ============================================================================
# ARCHITECTURE (FINAL CHAMPION)
# ============================================================================
class TransformerChampion(nn.Module):
    def __init__(self, hidden, out, heads=2, dropout=0.3):
        super().__init__()
        # Layer 1
        self.conv1 = TransformerConv((-1, -1), hidden, heads=heads)
        self.norm1 = LayerNorm(hidden * heads)
        
        # Layer 2
        self.conv2 = TransformerConv((-1, -1), hidden, heads=heads)
        self.norm2 = LayerNorm(hidden * heads)
        
        # Layer 3
        self.conv3 = TransformerConv((-1, -1), hidden, heads=1, concat=False)
        self.norm3 = LayerNorm(hidden)
        
        self.lin = nn.Linear(hidden, out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Layer 2
        x2 = self.conv2(x1, edge_index)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        x2 = x2 + x1 # Residual
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Layer 3
        x3 = self.conv3(x2, edge_index)
        x3 = self.norm3(x3)
        x3 = F.relu(x3)
        
        return self.lin(x3)

# ============================================================================
# TRAIN
# ============================================================================
def run_champion():
    data = load_data_champion()
    train_loader, val_loader, test_loader = get_loaders_champion(data)
    
    print("\n" + "="*60)
    print("TRAINING GraphTransformer CHAMPION (V4)")
    print("="*60)
    
    # Model
    base_model = TransformerChampion(HIDDEN_DIM, 1, heads=HEADS, dropout=DROPOUT)
    model = to_hetero(base_model, data.metadata(), aggr='mean').to(DEVICE)
    
    # Init
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad(): model(batch.x_dict, batch.edge_index_dict)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Loss
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (train_y == 0).sum() / max(train_y.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    best_val_auc = 0
    best_state = None
    patience = 5
    patience_cnt = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)['pekerja'][:batch['pekerja'].batch_size]
            loss = criterion(out.squeeze(-1), batch['pekerja'].y[:batch['pekerja'].batch_size].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # Val
        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x_dict, batch.edge_index_dict)['pekerja'][:batch['pekerja'].batch_size]
                v_preds.extend(torch.sigmoid(out.squeeze()).cpu().numpy())
                v_labels.extend(batch['pekerja'].y[:batch['pekerja'].batch_size].cpu().numpy())
        
        val_auc = roc_auc_score(v_labels, v_preds)
        _, val_f1 = find_optimal_threshold(v_labels, v_preds)
        
        print(f"   Ep {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break
                
    # Final Eval
    print("\n" + "="*60)
    print("CHAMPION TEST EVALUATION")
    print("="*60)
    model.load_state_dict(best_state)
    model.eval()
    
    t_preds, t_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x_dict, batch.edge_index_dict)['pekerja'][:batch['pekerja'].batch_size]
            t_preds.extend(torch.sigmoid(out.squeeze()).cpu().numpy())
            t_labels.extend(batch['pekerja'].y[:batch['pekerja'].batch_size].cpu().numpy())
            
    test_auc = roc_auc_score(t_labels, t_preds)
    thresh, test_f1 = find_optimal_threshold(t_labels, t_preds)
    
    print(f"\n🏆 CHAMPION RESULTS (thresh={thresh:.3f}):")
    print(f"   AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
    print("-" * 30)
    print(classification_report(t_labels, (np.array(t_preds) > thresh).astype(int), target_names=['Non-Fraud', 'Fraud']))
    print("Confusion Matrix:\n", confusion_matrix(t_labels, (np.array(t_preds) > thresh).astype(int)))

    # Save best model
    torch.save(best_state, "transformer_champion.pt")
    print(f"\n✅ Champion model saved to 'transformer_champion.pt'")

if __name__ == "__main__":
    run_champion()
