"""
Heterogeneous Graph Transformer (HGT)
=====================================
Pivot Strategy:
Instead of forcing a homogeneous Transformer (via to_hetero) onto the graph,
we use HGT which is designed *natively* for heterogeneous graphs.

It learns separate parameters for each edge type and node type relation,
which should capture the specific semantics (e.g., 'transfer' vs 'withdrawal') better.

Configuration:
- 2 Layers of HGTConv
- 4 Attention Heads (to capture different semantic subspaces)
- Minimalist Feature Set (ones for auxiliary nodes)
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
from torch_geometric.nn import HGTConv, Linear
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

DEVICE = torch.device('cpu')
print(f"Device: {DEVICE} (Forced for HGT stability)")

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 15
HIDDEN_DIM = 64
HEADS = 4         # HGT benefits from multiple heads
DROPOUT = 0.3
NUM_NEIGHBORS = [10, 5] # 2 Hops (Stable Zone)

DATA_CACHE = None

# ============================================================================
# DATA LOADING (CLEAN - V5 Style)
# ============================================================================
def load_data_clean():
    global DATA_CACHE
    if DATA_CACHE is not None: return DATA_CACHE

    start = time.time()
    print("\n[Data] Loading HeteroData...")
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    
    print("[Data] Loading labels...")
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    load_fraud_labels(data, pekerja_map, verbose=False)
    
    print("[Data] Computing basic features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
    # Simple 'Ones' for auxiliary nodes (Let HGT learn structure)
    print("[Data] Setting auxiliary features to ones...")
    for nt in ['nasabah', 'simpanan', 'pinjaman', 'transaksi']:
        if nt in data.node_types:
            data[nt].x = torch.ones((data[nt].num_nodes, 1))
            
    print("[Data] Undirecting graph...")
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    print(f"✅ Data ready in {time.time()-start:.1f}s")
    return data

# ============================================================================
# HGT MODEL
# ============================================================================
class HGT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_dim = data[node_type].x.shape[1]
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = DROPOUT

    def forward(self, x_dict, edge_index_dict):
        # 1. Project all inputs to same hidden dim
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu()

        # 2. HGT Layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply ReLU and Dropout to all node types
            for node_type in x_dict:
                x_dict[node_type] = x_dict[node_type].relu()
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)

        # 3. Output for target node 'pekerja'
        return self.lin(x_dict['pekerja'])

# ============================================================================
# RUN
# ============================================================================
def run():
    data = load_data_clean()
    
    kwargs = {'num_neighbors': NUM_NEIGHBORS, 'batch_size': BATCH_SIZE, 'num_workers': 0}
    train_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].train_mask), shuffle=True, **kwargs)
    val_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].val_mask), shuffle=False, **kwargs)
    test_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].test_mask), shuffle=False, **kwargs)
    
    print("\n[Init] Building HGT Model...")
    model = HGT(data, HIDDEN_DIM, 1, HEADS, num_layers=2).to(DEVICE)
    
    # Init lazy modules
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad(): model(batch.x_dict, batch.edge_index_dict)
    print(f"[Model] Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (train_y == 0).sum() / max(train_y.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    best_val_auc = 0
    best_state = None
    
    print("\nStarting Training (HGT)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict).squeeze(-1)
            # Match batch size of target
            target_size = batch['pekerja'].batch_size
            out = out[:target_size]
            y = batch['pekerja'].y[:target_size].float()
            
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x_dict, batch.edge_index_dict).squeeze(-1)
                target_size = batch['pekerja'].batch_size
                out = out[:target_size]
                
                v_preds.extend(torch.sigmoid(out).cpu().numpy())
                v_labels.extend(batch['pekerja'].y[:target_size].cpu().numpy())
        
        val_auc = roc_auc_score(v_labels, v_preds) if len(set(v_labels)) > 1 else 0.5
        print(f"   Ep {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Final Test
    print("\n" + "="*60)
    print("HGT EVALUATION")
    print("="*60)
    model.load_state_dict(best_state)
    model.eval()
    t_preds, t_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x_dict, batch.edge_index_dict).squeeze(-1)
            target_size = batch['pekerja'].batch_size
            out = out[:target_size]
            
            t_preds.extend(torch.sigmoid(out).cpu().numpy())
            t_labels.extend(batch['pekerja'].y[:target_size].cpu().numpy())
            
    test_auc = roc_auc_score(t_labels, t_preds)
    thresh, test_f1 = find_optimal_threshold(t_labels, t_preds)
    
    print(f"\n🏆 HGT Results (thresh={thresh:.3f}):")
    print(f"   AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
    print("-" * 30)
    print(classification_report(t_labels, (np.array(t_preds) > thresh).astype(int), target_names=['Non-Fraud', 'Fraud']))
    
    # Save
    torch.save(best_state, "hgt_model.pt")

if __name__ == "__main__":
    run()
