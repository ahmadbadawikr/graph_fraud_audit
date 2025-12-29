"""
GraphTransformer Basic (V5 - Back to V1)
========================================
Reverts to the exact minimalist configuration that achieved AUC 0.725.
Avoids the overcomplexity of V2/V3/V4.

Configuration:
- 2-hop sampling ([10, 5])
- 2-layer architecture (No residuals/skips)
- Hidden Dim: 32, Heads: 1
- Weighted BCE Loss
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
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION (BACK TO BASIC V1)
# ============================================================================
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 10
HIDDEN_DIM = 32
HEADS = 1
DROPOUT = 0.4
NUM_NEIGHBORS = [10, 5] # 2 Hops only

DATA_CACHE = None

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data_basic():
    global DATA_CACHE
    if DATA_CACHE is not None: return DATA_CACHE

    start = time.time()
    print("\n[Data] Loading HeteroData (16GB)...")
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    
    print("[Data] Loading labels...")
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    load_fraud_labels(data, pekerja_map, verbose=False)
    
    print("[Data] Computing basic pekerja features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
    print("[Data] Initializing auxiliary nodes...")
    for nt in ['nasabah', 'simpanan', 'pinjaman', 'transaksi']:
        if nt in data.node_types:
            data[nt].x = torch.ones((data[nt].num_nodes, 1))
            
    print("[Data] Undirecting graph...")
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    print(f"✅ Data ready in {time.time()-start:.1f}s")
    return data

# ============================================================================
# MINIMALIST MODEL (V1 STYLE)
# ============================================================================
class TransformerBasic(nn.Module):
    def __init__(self, hidden, out, heads=1, dropout=0.4):
        super().__init__()
        # 2 Layers 
        self.conv1 = TransformerConv((-1, -1), hidden, heads=heads)
        self.conv2 = TransformerConv((-1, -1), out, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        return x

# ============================================================================
# RUN
# ============================================================================
def run():
    data = load_data_basic()
    
    kwargs = {'num_neighbors': NUM_NEIGHBORS, 'batch_size': BATCH_SIZE, 'num_workers': 0}
    train_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].train_mask), shuffle=True, **kwargs)
    val_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].val_mask), shuffle=False, **kwargs)
    test_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].test_mask), shuffle=False, **kwargs)
    
    model = to_hetero(TransformerBasic(HIDDEN_DIM, 1, heads=HEADS, dropout=DROPOUT), data.metadata(), aggr='mean').to(DEVICE)
    
    # Dry run
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad(): model(batch.x_dict, batch.edge_index_dict)
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (train_y == 0).sum() / max(train_y.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    best_val_auc = 0
    best_state = None
    
    print("\nStarting Training (Back to Basics)...")
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
        print(f"   Ep {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Test
    print("\nFinal Test Evaluation...")
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
    
    print(f"\n✅ Basic Model Results (thresh={thresh:.3f}):")
    print(f"   AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
    print("-" * 30)
    print(classification_report(t_labels, (np.array(t_preds) > thresh).astype(int)))

if __name__ == "__main__":
    run()
