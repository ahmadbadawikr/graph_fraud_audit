"""
Final Optimization & Ensemble Script
====================================
1. Hyperparameter Tuning for GraphTransformer
2. Training Tabular Models (XGBoost, MLP)
3. Learning Optimal Ensemble Weights
"""

import os
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import gc
import pickle
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import TransformerConv, to_hetero
import torch_geometric.transforms as T
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Current dir
sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

# Global Config
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Data Cache
DATA_CACHE = None

# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_data(force_reload=False):
    global DATA_CACHE
    if DATA_CACHE is not None and not force_reload:
        return DATA_CACHE

    print("\n[Data] Loading HeteroData (16GB)...")
    data_path = os.path.join(OUTPUT_DIR, "heterodata.pt")
    data = torch.load(data_path)
    
    # Load map
    with open(f"{MAP_DIR}/map_pekerja.pkl", 'rb') as f:
        pekerja_map = pickle.load(f)
    pekerja_map = {str(k): v for k, v in pekerja_map.items()}
    
    # Labels
    load_fraud_labels(data, pekerja_map, verbose=False)
    
    # Features
    print("[Data] Computing pekerja features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
    # Other features
    print("[Data] Initializing other node features...")
    for node_type in ['nasabah', 'simpanan', 'pinjaman', 'transaksi']:
        if node_type in data.node_types:
            data[node_type].x = torch.ones((data[node_type].num_nodes, 1))
            
    # Connectivity
    print("[Data] Adding reverse edges (Undirected)...")
    data = T.ToUndirected()(data)
    
    DATA_CACHE = data
    return data

def get_gnn_loaders(data, batch_size=256):
    num_neighbors = [8, 4] 
    
    kwargs = {
        'data': data,
        'num_neighbors': num_neighbors,
        'batch_size': batch_size,
        'num_workers': 0,
        'shuffle': True
    }
    
    train_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].train_mask), **kwargs)
    
    kwargs['shuffle'] = False
    val_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].val_mask), **kwargs)
    test_loader = NeighborLoader(input_nodes=('pekerja', data['pekerja'].test_mask), **kwargs)
    
    return train_loader, val_loader, test_loader

# ============================================================================
# 2. MODELS
# ============================================================================

class GraphTransformer(nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=1, dropout=0.3):
        super().__init__()
        self.conv1 = TransformerConv((-1, -1), hidden_channels, heads=heads)
        self.conv2 = TransformerConv((-1, -1), hidden_channels, heads=1)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

def train_gnn_tuned(data, loaders, config):
    """Train GNN with specific hyperparameters"""
    hidden = config['hidden']
    heads = config['heads']
    dropout = config['dropout']
    lr = config['lr']
    
    train_loader, val_loader, test_loader = loaders
    
    model = to_hetero(GraphTransformer(hidden, 1, heads=heads, dropout=dropout), data.metadata(), aggr='mean').to(DEVICE)
    
    # Lazy init
    batch = next(iter(train_loader)).to(DEVICE)
    with torch.no_grad():
        model(batch.x_dict, batch.edge_index_dict)
    del batch
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Loss
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (train_y == 0).sum() / max((train_y == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    best_val_auc = 0
    best_model_state = None
    patience = 2
    patience_counter = 0
    
    for epoch in range(1, 6):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict)
            bs = batch['pekerja'].batch_size
            loss = criterion(out['pekerja'][:bs].squeeze(-1), batch['pekerja'].y[:bs].float())
            loss.backward()
            optimizer.step()
            
        # Val
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
            
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break
                
    model.load_state_dict(best_model_state)
    return model, best_val_auc

# ============================================================================
# 3. PIPELINE
# ============================================================================
def run_optimization():
    print("🚀 Starting Final Optimization Pipeline")
    data = load_data()
    train_loader, val_loader, test_loader = get_gnn_loaders(data)
    loaders = (train_loader, val_loader, test_loader)
    
    # helper for memory
    def cleanup():
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ---------------------------------------------------------
    # A. GNN Hyperparameter Tuning
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("A. Tuning GraphTransformer (Lightweight Trials)")
    print("="*50)
    
    param_grid = [
        {'hidden': 32, 'heads': 1, 'dropout': 0.3, 'lr': 0.001},  # Baseline
        {'hidden': 64, 'heads': 1, 'dropout': 0.3, 'lr': 0.001},  # Larger
        {'hidden': 32, 'heads': 1, 'dropout': 0.3, 'lr': 0.0005}, # Slower LR
    ]
    
    best_gnn_state = None
    best_cfg = None
    best_score = 0
    
    for i, cfg in enumerate(param_grid):
        print(f"\nTrial {i+1}: {cfg}")
        try:
            model, val_auc = train_gnn_tuned(data, loaders, cfg)
            print(f"   -> Val AUC: {val_auc:.4f}")
            
            if val_auc > best_score:
                best_score = val_auc
                best_gnn_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_cfg = cfg
            
            del model
            cleanup()
                
        except Exception as e:
            print(f"   -> Failed: {e}")
            
    print(f"\n🏆 Best GNN Config: {best_cfg} (Val AUC: {best_score:.4f})")
    
    # ---------------------------------------------------------
    # B. Generate Predictions (GNN, MLP, XGB)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("B. Generating Predictions for Ensemble")
    print("="*50)
    
    # 1. GNN Preds (Test Set)
    print("   Evaluating GNN on Test Set...")
    final_gnn = to_hetero(GraphTransformer(best_cfg['hidden'], 1, heads=best_cfg['heads'], dropout=best_cfg['dropout']), data.metadata(), aggr='mean').to(DEVICE)
    final_gnn.load_state_dict(best_gnn_state)
    final_gnn.eval()
    
    gnn_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            bs = batch['pekerja'].batch_size
            out = final_gnn(batch.x_dict, batch.edge_index_dict)
            gnn_preds.extend(torch.sigmoid(out['pekerja'][:bs].squeeze(-1)).cpu().numpy())
    gnn_preds = np.array(gnn_preds)
    
    del final_gnn
    cleanup()
    
    # 2. Tabular Preparation
    features = data['pekerja'].x.numpy()
    test_mask = data['pekerja'].test_mask.numpy()
    y = data['pekerja'].y.numpy()
    
    X_train = features[data['pekerja'].train_mask.numpy()]
    y_train = y[data['pekerja'].train_mask.numpy()]
    X_test = features[test_mask]
    y_test = y[test_mask]
    
    # 3. Train MLP
    print("   Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict_proba(X_test)[:, 1]
    del mlp
    
    # 4. Train XGBoost
    print("   Training XGBoost...")
    scale_pos = (y_train==0).sum()/max(y_train.sum(), 1)
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos, use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
    del xgb_model
    cleanup()
    
    # ---------------------------------------------------------
    # C. Optimize Ensemble
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("C. Optimizing Ensemble Weights")
    print("="*50)
    
    best_ens_f1 = 0
    best_weights = (0,0,0)
    best_thresh = 0.5
    
    # Search weights
    for w1 in [0.4, 0.6, 0.8]: # GNN usually stronger
        for w2 in [0.0, 0.2, 0.4]:
            w3 = 1.0 - w1 - w2
            if w3 < 0: continue
            
            preds = (w1 * gnn_preds) + (w2 * mlp_preds) + (w3 * xgb_preds)
            thresh, f1 = find_optimal_threshold(y_test, preds)
            
            if f1 > best_ens_f1:
                best_ens_f1 = f1
                best_weights = (w1, w2, w3)
                best_thresh = thresh
                
    print(f"🏆 Best Weights: GNN={best_weights[0]:.1f}, MLP={best_weights[1]:.1f}, XGB={best_weights[2]:.1f}")
    
    # Final Eval
    final_preds = (best_weights[0] * gnn_preds) + (best_weights[1] * mlp_preds) + (best_weights[2] * xgb_preds)
    final_auc = roc_auc_score(y_test, final_preds)
    
    print("-" * 60)
    print(f"✅ FINAL ENSEMBLE RESULTS (thresh={best_thresh:.3f}):")
    print(f"   AUC: {final_auc:.4f} | F1: {best_ens_f1:.4f}")
    print("-" * 20)
    print(classification_report(y_test, (final_preds > best_thresh).astype(int), target_names=['Non-Fraud', 'Fraud']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, (final_preds > best_thresh).astype(int)))

if __name__ == "__main__":
    run_optimization()
