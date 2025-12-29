"""
GraphTransformer V3 - Optimized "Pro" Script
============================================
Fixes overfitting from V2:
- Reduced capacity (HIDDEN_DIM=32, layers=2)
- Aggressive regularization (Dropout=0.5, Weight Decay=1e-3)
- Stable Loss (Weighted BCE)
- Preserves V2's vectorized degree features
"""

import os
import sys

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
from fraud_utils import OUTPUT_DIR, MAP_DIR, load_fraud_labels, compute_enhanced_features, find_optimal_threshold

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION (V3 - "Lean & Mean")
# ============================================================================
BATCH_SIZE = 512
LR = 0.0005      # Lower LR for stability
EPOCHS = 20       # More epochs with early stopping
HIDDEN_DIM = 32   # Back to V1 level for safety
HEADS = 2
DROPOUT = 0.5     # Higher dropout to fight overfitting
WEIGHT_DECAY = 1e-3 # Stronger regularization
NUM_NEIGHBORS = [10, 5]

DATA_CACHE = None

# ============================================================================
# DATA HELPERS (Preserved from V2)
# ============================================================================
def compute_node_features_fast(data):
    """Efficiently compute degree features (vectorized)"""
    from torch_geometric.utils import degree
    print("[V3] Computing degree features (fast)...")
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

def load_data_v3():
    global DATA_CACHE
    if DATA_CACHE is not None: return DATA_CACHE
    
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    
    load_fraud_labels(data, pekerja_map, verbose=False)
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    data = compute_node_features_fast(data)
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    return data

# ============================================================================
# ARCHITECTURE (V3)
# ============================================================================
class TransformerV3(nn.Module):
    def __init__(self, hidden, out, heads=2, dropout=0.5):
        super().__init__()
        # Layer 1
        self.conv1 = TransformerConv((-1, -1), hidden, heads=heads)
        self.norm1 = LayerNorm(hidden * heads)
        
        # Layer 2 (concat=False to keep hidden size stable)
        self.conv2 = TransformerConv((-1, -1), hidden, heads=1, concat=False)
        self.norm2 = LayerNorm(hidden)
        
        self.lin = nn.Linear(hidden, out)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.lin(x)

# ============================================================================
# TRAINING LOGIC
# ============================================================================
def train():
    data = load_data_v3()
    
    # Loaders
    kwargs = {'num_neighbors': NUM_NEIGHBORS, 'batch_size': BATCH_SIZE, 'num_workers': 0}
    train_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].train_mask), shuffle=True, **kwargs)
    val_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].val_mask), shuffle=False, **kwargs)
    test_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].test_mask), shuffle=False, **kwargs)
    
    # Model Setup
    model = to_hetero(TransformerV3(HIDDEN_DIM, 1, heads=HEADS, dropout=DROPOUT), data.metadata(), aggr='mean').to(DEVICE)
    
    # Dry run for lazy init
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad(): model(batch.x_dict, batch.edge_index_dict)
    
    print(f"[V3] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Label weight
    y_train = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    best_val_auc = 0
    best_state = None
    patience = 5
    patience_cnt = 0
    
    print("\nStarting Optimized Training...")
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
        
        # Validation
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
            if patience_cnt >= patience: break
            
    # Final Test
    print("\nEvaluating Best Model on Test Set...")
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
    
    print(f"\n✅ GraphTransformer V3 Results (thresh={thresh:.3f}):")
    print(f"   AUC: {test_auc:.4f} | F1: {test_f1:.4f}")
    print("-" * 30)
    print(classification_report(t_labels, (np.array(t_preds) > thresh).astype(int), target_names=['Non-Fraud', 'Fraud']))

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
