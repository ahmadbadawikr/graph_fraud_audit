"""
Two-Stage Production Pipeline: GAT → HGT
=========================================
Recommended production approach for fraud detection:
- Stage 1 (GAT): High-recall screening (75% recall) - catch most fraudsters
- Stage 2 (HGT): Precision refinement - filter false positives from Stage 1

Expected improvement:
- Single GAT: 75% recall, 15% precision (F1: 0.25)
- Single HGT: 34% recall, 25% precision (F1: 0.30)
- Two-Stage:  ~70% recall, ~30% precision (F1: ~0.40)
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
from torch_geometric.nn import GATConv, HGTConv, Linear, to_hetero
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

# Force CPU for HGT stability
DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = 512
LR = 0.001
GAT_EPOCHS = 10
HGT_EPOCHS = 15
HIDDEN_DIM = 32
HGT_HIDDEN_DIM = 64
HEADS = 1
HGT_HEADS = 4
DROPOUT = 0.3
NUM_NEIGHBORS = [10, 5]

# Thresholds for two-stage
GAT_THRESHOLD = 0.3   # Low threshold for high recall
HGT_THRESHOLD = 0.5   # Higher threshold for precision

# Model save paths
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
DATA_CACHE = None

def load_data():
    global DATA_CACHE
    if DATA_CACHE is not None: 
        return DATA_CACHE

    start = time.time()
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    load_fraud_labels(data, pekerja_map, verbose=False)
    
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
    for nt in ['nasabah', 'simpanan', 'pinjaman', 'transaksi']:
        if nt in data.node_types:
            data[nt].x = torch.ones((data[nt].num_nodes, 1))
    
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    print(f"✅ Data ready in {time.time()-start:.1f}s")
    return data

# ============================================================================
# GAT MODEL (High Recall)
# ============================================================================
class GATHomogeneous(nn.Module):
    """2-layer GAT for high-recall fraud screening"""
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads=HEADS, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, heads=HEADS, add_self_loops=False)
        self.lin = nn.Linear(hidden_channels * HEADS, 1)
        self.dropout = DROPOUT

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)

# ============================================================================
# HGT MODEL (High Precision)
# ============================================================================
class HGT(nn.Module):
    """2-layer HGT for precision refinement"""
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            in_dim = data[node_type].x.shape[1]
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(), num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        self.dropout = DROPOUT

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = x_dict[node_type].relu()
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)

        return self.lin(x_dict['pekerja'])

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_gat(data, train_loader, val_loader, epochs=GAT_EPOCHS):
    """Train GAT model"""
    print("\n" + "="*60)
    print("STAGE 1: Training GAT Model (High Recall)")
    print("="*60)
    
    # Build homogeneous model
    homo_model = GATHomogeneous(HIDDEN_DIM)
    model = to_hetero(homo_model, data.metadata(), aggr='sum').to(DEVICE)
    
    # Initialize
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():
        model(batch.x_dict, batch.edge_index_dict)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[GAT] Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = torch.tensor([(train_y == 0).sum() / max(1, (train_y == 1).sum())]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_auc = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)['pekerja'][:batch['pekerja'].batch_size]
            y = batch['pekerja'].y[:batch['pekerja'].batch_size].float().unsqueeze(1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x_dict, batch.edge_index_dict)['pekerja'][:batch['pekerja'].batch_size]
                prob = torch.sigmoid(out).cpu().numpy()
                y = batch['pekerja'].y[:batch['pekerja'].batch_size].cpu().numpy()
                preds.extend(prob.flatten())
                labels.extend(y.flatten())
        
        auc = roc_auc_score(labels, preds)
        if auc > best_auc:
            best_auc = auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    torch.save(best_model_state, f"{MODELS_DIR}/gat_best.pt")
    print(f"✅ GAT saved to {MODELS_DIR}/gat_best.pt (Best AUC: {best_auc:.4f})")
    
    return model

def train_hgt(data, train_loader, val_loader, epochs=HGT_EPOCHS):
    """Train HGT model"""
    print("\n" + "="*60)
    print("STAGE 2: Training HGT Model (High Precision)")
    print("="*60)
    
    model = HGT(data, HGT_HIDDEN_DIM, 1, HGT_HEADS, num_layers=2).to(DEVICE)
    
    # Initialize
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():
        model(batch.x_dict, batch.edge_index_dict)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[HGT] Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = torch.tensor([(train_y == 0).sum() / max(1, (train_y == 1).sum())]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_auc = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)[:batch['pekerja'].batch_size]
            y = batch['pekerja'].y[:batch['pekerja'].batch_size].float().unsqueeze(1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch.x_dict, batch.edge_index_dict)[:batch['pekerja'].batch_size]
                prob = torch.sigmoid(out).cpu().numpy()
                y = batch['pekerja'].y[:batch['pekerja'].batch_size].cpu().numpy()
                preds.extend(prob.flatten())
                labels.extend(y.flatten())
        
        auc = roc_auc_score(labels, preds)
        if auc > best_auc:
            best_auc = auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f"Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    torch.save(best_model_state, f"{MODELS_DIR}/hgt_best.pt")
    print(f"✅ HGT saved to {MODELS_DIR}/hgt_best.pt (Best AUC: {best_auc:.4f})")
    
    return model

# ============================================================================
# TWO-STAGE INFERENCE
# ============================================================================
def two_stage_inference(gat_model, hgt_model, test_loader, data):
    """Two-stage inference: GAT screening → HGT refinement"""
    print("\n" + "="*60)
    print("TWO-STAGE INFERENCE")
    print("="*60)
    print(f"GAT Threshold (screening): {GAT_THRESHOLD}")
    print(f"HGT Threshold (refinement): {HGT_THRESHOLD}")
    
    gat_model.eval()
    hgt_model.eval()
    
    gat_probs_all = []
    hgt_probs_all = []
    labels_all = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch = batch.to(DEVICE)
            batch_size = batch['pekerja'].batch_size
            
            # Stage 1: GAT
            gat_out = gat_model(batch.x_dict, batch.edge_index_dict)['pekerja'][:batch_size]
            gat_probs = torch.sigmoid(gat_out).cpu().numpy().flatten()
            
            # Stage 2: HGT
            hgt_out = hgt_model(batch.x_dict, batch.edge_index_dict)[:batch_size]
            hgt_probs = torch.sigmoid(hgt_out).cpu().numpy().flatten()
            
            labels = batch['pekerja'].y[:batch_size].cpu().numpy().flatten()
            
            gat_probs_all.extend(gat_probs)
            hgt_probs_all.extend(hgt_probs)
            labels_all.extend(labels)
    
    gat_probs_all = np.array(gat_probs_all)
    hgt_probs_all = np.array(hgt_probs_all)
    labels_all = np.array(labels_all)
    
    # Two-stage logic
    # Stage 1: GAT flags candidates
    gat_candidates = gat_probs_all > GAT_THRESHOLD
    
    # Stage 2: HGT refines (only among GAT candidates)
    two_stage_preds = gat_candidates & (hgt_probs_all > HGT_THRESHOLD)
    
    # Also compute individual model predictions for comparison
    gat_preds = gat_probs_all > 0.5
    hgt_preds = hgt_probs_all > 0.5
    
    return {
        'gat_probs': gat_probs_all,
        'hgt_probs': hgt_probs_all,
        'labels': labels_all,
        'gat_preds': gat_preds,
        'hgt_preds': hgt_preds,
        'two_stage_preds': two_stage_preds,
        'gat_candidates': gat_candidates,
    }

def evaluate_results(results):
    """Evaluate and compare all approaches"""
    labels = results['labels']
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    approaches = [
        ("GAT Only (threshold=0.5)", results['gat_preds']),
        ("HGT Only (threshold=0.5)", results['hgt_preds']),
        ("Two-Stage (GAT→HGT)", results['two_stage_preds']),
    ]
    
    print(f"\n{'Approach':<30} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 70)
    
    for name, preds in approaches:
        if "GAT" in name and "Two" not in name:
            auc = roc_auc_score(labels, results['gat_probs'])
        elif "HGT" in name and "Two" not in name:
            auc = roc_auc_score(labels, results['hgt_probs'])
        else:
            # For two-stage, use GAT probs for AUC (screening)
            auc = roc_auc_score(labels, results['gat_probs'])
        
        f1 = f1_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds)
        
        print(f"{name:<30} {auc:>8.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.2%}")
    
    # Detailed confusion matrix for two-stage
    print("\n" + "="*60)
    print("TWO-STAGE CONFUSION MATRIX")
    print("="*60)
    
    cm = confusion_matrix(labels, results['two_stage_preds'])
    print(f"\n                  Predicted")
    print(f"               Non-Fraud  Fraud")
    print(f"Actual")
    print(f"Non-Fraud      {cm[0,0]:>7}  {cm[0,1]:>6}")
    print(f"Fraud          {cm[1,0]:>7}  {cm[1,1]:>6}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n✅ True Positives (Fraudsters Caught): {tp}")
    print(f"❌ False Negatives (Fraudsters Missed): {fn}")
    print(f"⚠️  False Positives (Innocent Flagged): {fp}")
    
    # Stage breakdown
    print("\n" + "="*60)
    print("STAGE BREAKDOWN")
    print("="*60)
    print(f"Stage 1 (GAT): {results['gat_candidates'].sum()} candidates flagged")
    print(f"Stage 2 (HGT): {results['two_stage_preds'].sum()} confirmed as fraud")
    print(f"Reduction: {(1 - results['two_stage_preds'].sum()/max(1,results['gat_candidates'].sum()))*100:.1f}% false positives filtered")
    
    return cm

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("TWO-STAGE FRAUD DETECTION PIPELINE")
    print("GAT (High Recall) → HGT (High Precision)")
    print("="*60)
    
    # Load data
    data = load_data()
    
    # Create loaders
    kwargs = {'num_neighbors': NUM_NEIGHBORS, 'batch_size': BATCH_SIZE, 'num_workers': 0}
    train_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].train_mask), shuffle=True, **kwargs)
    val_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].val_mask), shuffle=False, **kwargs)
    test_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].test_mask), shuffle=False, **kwargs)
    
    # Train both models
    gat_model = train_gat(data, train_loader, val_loader)
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    hgt_model = train_hgt(data, train_loader, val_loader)
    
    # Two-stage inference
    results = two_stage_inference(gat_model, hgt_model, test_loader, data)
    
    # Evaluate
    cm = evaluate_results(results)
    
    print("\n" + "="*60)
    print("✅ TWO-STAGE PIPELINE COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {MODELS_DIR}/")
    print("  - gat_best.pt")
    print("  - hgt_best.pt")
    
    return results, cm

if __name__ == "__main__":
    results, cm = main()
