"""
GraphTransformer V2 - Enhanced Training Script
===============================================
Improvements over V1:
- Better node features (degree-based)
- Layer normalization + skip connections
- Focal Loss for class imbalance
- Cosine annealing LR schedule
- Deeper architecture (3 layers)
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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import OUTPUT_DIR, MAP_DIR, load_fraud_labels, compute_enhanced_features, find_optimal_threshold

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION (V2)
# ============================================================================
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 15
HIDDEN_DIM = 64  # Increased
HEADS = 2        # Multi-head
DROPOUT = 0.4    # Increased
NUM_NEIGHBORS = [10, 5]

DATA_CACHE = None

# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ============================================================================
# IMPROVED NODE FEATURES
# ============================================================================
def compute_node_features(data):
    """Compute degree-based features for all node types efficiently"""
    print("[Features] Computing degree features for all nodes (vectorized)...")
    from torch_geometric.utils import degree
    
    # 1. Initialize degree accumulators
    node_types = data.node_types
    in_degrees = {nt: torch.zeros(data[nt].num_nodes) for nt in node_types}
    out_degrees = {nt: torch.zeros(data[nt].num_nodes) for nt in node_types}
    
    # 2. Single pass over all edge types to accumulate degrees
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        
        # Out-degree for source nodes
        out_degrees[src_type] += degree(edge_index[0], num_nodes=data[src_type].num_nodes)
        # In-degree for destination nodes
        in_degrees[dst_type] += degree(edge_index[1], num_nodes=data[dst_type].num_nodes)
        
    # 3. Process each node type (except pekerja which has its own features)
    for node_type in node_types:
        if node_type == 'pekerja':
            continue
            
        in_deg = in_degrees[node_type]
        out_deg = out_degrees[node_type]
        total_deg = in_deg + out_deg
        log_deg = torch.log1p(total_deg)
        
        # Normalized features to keep values in stable range [0, 1]
        def normalize(t):
            m = t.max()
            return t / (m + 1e-6) if m > 0 else t
            
        features = torch.stack([
            normalize(in_deg),
            normalize(out_deg),
            normalize(total_deg),
            normalize(log_deg)
        ], dim=1)
        
        data[node_type].x = features
        print(f"   ✓ {node_type}: {data[node_type].num_nodes:,} nodes, {features.shape[1]} degree features")
        
        # Cleanup degree tensors for this node type to save memory
        del in_degrees[node_type]
        del out_degrees[node_type]
        gc.collect()
    
    return data

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(force_reload=False):
    global DATA_CACHE
    if DATA_CACHE is not None and not force_reload:
        print("✅ Using cached data")
        return DATA_CACHE

    start = time.time()
    print("\n" + "="*60)
    print("LOADING DATA (V2)")
    print("="*60)
    
    print("[1/5] Loading HeteroData...")
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    print(f"      Loaded in {time.time()-start:.1f}s")
    
    print("[2/5] Loading pekerja map...")
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    
    print("[3/5] Loading fraud labels...")
    load_fraud_labels(data, pekerja_map, verbose=False)
    n_fraud = data['pekerja'].y.sum().item()
    print(f"      {n_fraud} fraud labels")
    
    print("[4/5] Computing pekerja features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    print(f"      {data['pekerja'].x.shape[1]} features")
    
    print("[5/5] Computing degree features for other nodes...")
    data = compute_node_features(data)
    
    print("[+] Adding reverse edges...")
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    print(f"\n✅ Data loading complete in {time.time()-start:.1f}s")
    return data

def get_loaders(data):
    kwargs = {
        'data': data,
        'num_neighbors': NUM_NEIGHBORS,
        'batch_size': BATCH_SIZE,
        'num_workers': 0,
        'shuffle': True
    }
    
    train_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].train_mask), **kwargs)
    kwargs['shuffle'] = False
    val_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].val_mask), **kwargs)
    test_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].test_mask), **kwargs)
    
    return train_loader, val_loader, test_loader

# ============================================================================
# MODEL V2
# ============================================================================
class TransformerV2(nn.Module):
    """Enhanced GraphTransformer with LayerNorm and Skip Connections"""
    def __init__(self, hidden_channels, out_channels, heads=2, dropout=0.4):
        super().__init__()
        
        # Layer 1
        self.conv1 = TransformerConv((-1, -1), hidden_channels, heads=heads)
        self.norm1 = LayerNorm(hidden_channels * heads)
        
        # Layer 2
        self.conv2 = TransformerConv((-1, -1), hidden_channels, heads=heads)
        self.norm2 = LayerNorm(hidden_channels * heads)
        
        # Layer 3 (output layer, single head)
        self.conv3 = TransformerConv((-1, -1), hidden_channels, heads=1)
        self.norm3 = LayerNorm(hidden_channels)
        
        # Classifier
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x1 = self.conv1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # Layer 2 with skip connection
        x2 = self.conv2(x1, edge_index)
        x2 = self.norm2(x2)
        x2 = F.relu(x2)
        x2 = x2 + x1  # Skip connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # Layer 3
        x3 = self.conv3(x2, edge_index)
        x3 = self.norm3(x3)
        x3 = F.relu(x3)
        
        return self.lin(x3)

# ============================================================================
# TRAINING
# ============================================================================
def train():
    print("\n" + "="*60)
    print("TRAINING GraphTransformer V2")
    print("="*60)
    
    data = load_data()
    train_loader, val_loader, test_loader = get_loaders(data)
    
    print("\n[Init] Building model...")
    base_model = TransformerV2(HIDDEN_DIM, 1, heads=HEADS, dropout=DROPOUT)
    model = to_hetero(base_model, data.metadata(), aggr='mean').to(DEVICE)
    
    # Lazy init
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():
        model(batch.x_dict, batch.edge_index_dict)
    del batch
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Params: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    print(f"      Loss: Focal Loss (α=0.25, γ=2.0)")
    print(f"      LR Schedule: Cosine Annealing ({LR} → 1e-5)")
    print("-" * 60)
    
    best_val_auc = 0
    best_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False)
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(batch.x_dict, batch.edge_index_dict)
            bs = batch['pekerja'].batch_size
            out_pekerja = out['pekerja'][:bs].squeeze(-1)
            y = batch['pekerja'].y[:bs].float()
            
            loss = criterion(out_pekerja, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bs
            total_samples += bs
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / total_samples
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                bs = batch['pekerja'].batch_size
                out = model(batch.x_dict, batch.edge_index_dict)
                val_preds.extend(torch.sigmoid(out['pekerja'][:bs].squeeze(-1)).cpu().numpy())
                val_labels.extend(batch['pekerja'].y[:bs].cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
        _, val_f1 = find_optimal_threshold(val_labels, val_preds)
        
        lr_now = scheduler.get_last_lr()[0]
        print(f"   Ep {epoch:2d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | F1: {val_f1:.4f} | LR: {lr_now:.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break
    
    # Restore best
    model.load_state_dict(best_state)
    
    # Test evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION (Test Set)")
    print("="*60)
    
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            batch = batch.to(DEVICE)
            bs = batch['pekerja'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)
            test_preds.extend(torch.sigmoid(out['pekerja'][:bs].squeeze(-1)).cpu().numpy())
            test_labels.extend(batch['pekerja'].y[:bs].cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    test_auc = roc_auc_score(test_labels, test_preds)
    thresh, test_f1 = find_optimal_threshold(test_labels, test_preds)
    test_binary = (test_preds > thresh).astype(int)
    
    print(f"\n✅ GraphTransformer V2 Results (thresh={thresh:.3f}):")
    print(f"   AUC: {test_auc:.4f}")
    print(f"   F1 : {test_f1:.4f}")
    print("-" * 20)
    print(classification_report(test_labels, test_binary, target_names=['Non-Fraud', 'Fraud']))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_binary))
    
    # Cleanup
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return {'auc': test_auc, 'f1': test_f1}

if __name__ == "__main__":
    train()
