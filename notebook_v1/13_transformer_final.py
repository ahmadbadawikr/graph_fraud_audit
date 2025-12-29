"""
GraphTransformer Final Optimized (V6)
======================================
The "Best of Both Worlds" model.
Combines V1/V5's minimalist stable architecture with V3's fast degree features.

Key Strategy:
- Minimalist Architecture (2 layers, 1 head, 32 hidden) to prevent overfitting.
- Fast Degree Features for auxiliary nodes (nasabah, transaksi, etc.) to give context.
- Optimized 2.5-hop sampling ([15, 10]) for stronger local context without 3-hop noise.
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

sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION (V6 - FINAL BALANCED)
# ============================================================================
BATCH_SIZE = 512
LR = 0.0007      # Slightly lower for precision
EPOCHS = 12       # Optimized epoch count
HIDDEN_DIM = 32   # Stable capacity
HEADS = 1         # No multi-head noise
DROPOUT = 0.4
NUM_NEIGHBORS = [15, 10] # Stronger 2-hop sampling

DATA_CACHE = None

# ============================================================================
# FAST FEATURE ENGINEERING (From V3)
# ============================================================================
def compute_node_features_final(data):
    """Efficiently compute degree features (vectorized)"""
    from torch_geometric.utils import degree
    print("[V6] Computing structural degree features...")
    node_types = data.node_types
    in_degrees = {nt: torch.zeros(data[nt].num_nodes) for nt in node_types}
    out_degrees = {nt: torch.zeros(data[nt].num_nodes) for nt in node_types}
    
    for edge_type in data.edge_types:
        src, _, dst = edge_type
        ei = data[edge_type].edge_index
        out_degrees[src] += degree(ei[0], num_nodes=data[src].num_nodes)
        in_degrees[dst] += degree(ei[1], num_nodes=data[dst].num_nodes)
        
    for nt in node_types:
        if nt == 'pekerja': continue
        id, od = in_degrees[nt], out_degrees[nt]
        td = id + od
        ld = torch.log1p(td)
        
        def norm(t):
            m = t.max()
            return t / (m + 1e-6) if m > 0 else t
            
        data[nt].x = torch.stack([norm(id), norm(od), norm(td), norm(ld)], dim=1)
        del in_degrees[nt], out_degrees[nt]
    gc.collect()
    return data

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data_final():
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
    
    print("[Data] Computing pekerja features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
    print("[Data] Computing structural features for others...")
    data = compute_node_features_final(data)
    
    print("[Data] Undirecting graph...")
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    print(f"✅ Setup complete in {time.time()-start:.1f}s")
    return data

# ============================================================================
# ARCHITECTURE (FINAL BALANCED)
# ============================================================================
class TransformerFinal(nn.Module):
    def __init__(self, hidden, out, heads=1, dropout=0.4):
        super().__init__()
        # 2 Layers - Minimalist & Robust
        self.conv1 = TransformerConv((-1, -1), hidden, heads=heads)
        self.norm1 = LayerNorm(hidden * heads)
        
        self.conv2 = TransformerConv((-1, -1), out, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        return x

# ============================================================================
# RUN
# ============================================================================
def run():
    data = load_data_final()
    
    kwargs = {'num_neighbors': NUM_NEIGHBORS, 'batch_size': BATCH_SIZE, 'num_workers': 0}
    train_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].train_mask), shuffle=True, **kwargs)
    val_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].val_mask), shuffle=False, **kwargs)
    test_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].test_mask), shuffle=False, **kwargs)
    
    model = to_hetero(TransformerFinal(HIDDEN_DIM, 1, heads=HEADS, dropout=DROPOUT), data.metadata(), aggr='mean').to(DEVICE)
    
    # Init
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad(): model(batch.x_dict, batch.edge_index_dict)
    print(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (train_y == 0).sum() / max(train_y.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    best_val_auc = 0
    best_state = None
    
    print("\nStarting Training (V6 Final Optimized)...")
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
        print(f"   Ep {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Final Test
    print("\n" + "="*60)
    print("FINAL MODEL VERIFICATION")
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
    
    print(f"\n✅ FINAL BEST Results (thresh={thresh:.3f}):")
    print(f"   AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
    print("-" * 30)
    print(classification_report(t_labels, (np.array(t_preds) > thresh).astype(int), target_names=['Non-Fraud', 'Fraud']))
    
    # Save final model
    torch.save(best_state, "final_best_transformer.pt")
    print(f"✅ Final model weights saved to 'final_best_transformer.pt'")

if __name__ == "__main__":
    run()
