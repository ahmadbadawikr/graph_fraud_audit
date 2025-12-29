"""
Hybrid GNN + XGBoost Pipeline (V7 - Stable)
===========================================
Strategy: Use the Graph Neural Network as a "Feature Extractor" and XGBoost as the "Classifier".

Refactored to use a flattened model structure (no nested encoders) to play nicely with to_hetero,
returning (logits, embeddings) directly.

1. Train a basic GraphTransformer to learn structural embeddings.
2. Extract the 32-dimensional node embeddings for all 'pekerja' nodes.
3. Combine these Embeddings + Original Enhanced Features.
4. Train XGBoost on this enriched dataset.
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
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, classification_report

sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')
from fraud_utils import *

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# CONFIG
# ============================================================================
HIDDEN_DIM = 32
HEADS = 1
DROPOUT = 0.4
NUM_NEIGHBORS = [10, 5]
BATCH_SIZE = 512

DATA_CACHE = None

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data_hybrid():
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
    
    print("[Data] Computing basic features...")
    data['pekerja'].x = compute_enhanced_features(data, verbose=False)
    
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
# FLATTENED MODEL (Better for to_hetero)
# ============================================================================
class HybridGNN(nn.Module):
    def __init__(self, hidden, heads=1, dropout=0.4):
        super().__init__()
        self.conv1 = TransformerConv((-1, -1), hidden, heads=heads)
        self.conv2 = TransformerConv((-1, -1), hidden, heads=1, concat=False) # Output embedding dim
        self.lin = nn.Linear(hidden, 1) # Probing head
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 (Embedding)
        x = self.conv2(x, edge_index)
        emb = x # Save embedding
        
        # Output Head
        out = self.lin(x)
        
        return out, emb

# ============================================================================
# EXTRACTION & HYBRID TRAINING
# ============================================================================
def run_hybrid():
    data = load_data_hybrid()
    
    print("\n[Phase 1] Training GNN Proxy...")
    
    # Init flattened model
    base_model = HybridGNN(HIDDEN_DIM, heads=HEADS, dropout=DROPOUT)
    model = to_hetero(base_model, data.metadata(), aggr='mean').to(DEVICE)
    
    kwargs = {'num_neighbors': NUM_NEIGHBORS, 'batch_size': BATCH_SIZE, 'num_workers': 0}
    train_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].train_mask), shuffle=True, **kwargs)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_y = data['pekerja'].y[data['pekerja'].train_mask]
    pos_weight = (train_y == 0).sum() / max(train_y.sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    
    # Train 5 Epochs
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/5", leave=False)
        for batch in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # Returns tuple of dicts
            out_dict, _ = model(batch.x_dict, batch.edge_index_dict)
            
            out = out_dict['pekerja'][:batch['pekerja'].batch_size]
            loss = criterion(out.squeeze(-1), batch['pekerja'].y[:batch['pekerja'].batch_size].float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Ep {epoch}: Loss {total_loss/len(train_loader):.4f}")
        
    print("✅ GNN Trained. Extracting Embeddings...")
    
    # Helper to extract
    def extract_from_loader(loader):
        embs, feats, ys = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting"):
                batch = batch.to(DEVICE)
                
                # Forward pass
                _, emb_dict = model(batch.x_dict, batch.edge_index_dict)
                
                bs = batch['pekerja'].batch_size
                embs.append(emb_dict['pekerja'][:bs].cpu().numpy())
                feats.append(batch['pekerja'].x[:bs].cpu().numpy())
                ys.append(batch['pekerja'].y[:bs].cpu().numpy())
                
        return np.concatenate(embs), np.concatenate(feats), np.concatenate(ys)

    print("   Extracting partitions...")
    # Exact same kwargs as training to match logic
    kwargs_extract = kwargs.copy()
    kwargs_extract['shuffle'] = False
    
    train_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].train_mask), **kwargs_extract)
    test_loader = NeighborLoader(data, input_nodes=('pekerja', data['pekerja'].test_mask), **kwargs_extract)
    
    X_emb_tr, X_feat_tr, y_tr = extract_from_loader(train_loader)
    X_emb_te, X_feat_te, y_te = extract_from_loader(test_loader)
    
    # 3. Combine
    X_train = np.hstack([X_emb_tr, X_feat_tr])
    X_test = np.hstack([X_emb_te, X_feat_te])
    
    print(f"   Train Matrix: {X_train.shape}")
    print(f"   Test Matrix:  {X_test.shape}")
    
    # 4. XGBoost Training
    print("\n[Phase 2] Training XGBoost on Hybrid Features...")
    
    clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_tr)/sum(y_tr)),
        n_jobs=-1,
        eval_metric='auc',
        random_state=42
    )
    
    clf.fit(X_train, y_tr)
    
    # Evaluate
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_te, preds)
    
    print("\n" + "="*60)
    print("HYBRID MODEL RESULTS (V7 Stable)")
    print("="*60)
    
    thresh, f1 = find_optimal_threshold(y_te, preds)
    print(f"✅ XGBoost + Graph Embeddings Results (thresh={thresh:.3f}):")
    print(f"   AUC: {auc:.4f} | F1: {f1:.4f}")
    print("-" * 30)
    print(classification_report(y_te, (preds > thresh).astype(int), target_names=['Non-Fraud', 'Fraud']))
    
    # Compare with pure features
    print("\n(Baseline Comparison: pure features only)")
    clf_base = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=(len(y_tr)/sum(y_tr)), random_state=42)
    clf_base.fit(X_feat_tr, y_tr)
    preds_base = clf_base.predict_proba(X_feat_te)[:, 1]
    auc_base = roc_auc_score(y_te, preds_base)
    print(f"   Pure Tabular AUC: {auc_base:.4f} (Lift: {auc - auc_base:+.4f})")
    
    # Save model
    clf.save_model("hybrid_xgboost.json")
    torch.save(model.state_dict(), "hybrid_gnn_extractor.pt")
    print("✅ Models saved.")

if __name__ == "__main__":
    run_hybrid()
