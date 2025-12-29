"""Quick script to run only Transformer model for 20 epochs"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
sys.path.insert(0, '/Users/kasyfur/graph_fraud_audit/notebook_v1')

# Import training function from main script
from fraud_utils import *
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import TransformerConv, to_hetero
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pickle
import time
import gc

print(f"Device: {DEVICE}")

# Config
BATCH_SIZE = 512
LR = 0.001
EPOCHS = 20
HIDDEN_DIM = 32
HEADS = 1
NUM_NEIGHBORS = [10, 5]

# Model
class GraphTransformer(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TransformerConv((-1, -1), hidden_channels, heads=HEADS, dropout=0.4)
        self.conv2 = TransformerConv((-1, -1), hidden_channels, heads=HEADS, dropout=0.4)
        self.lin = nn.Linear(hidden_channels * HEADS, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

# Load data
print("Loading data...")
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

import torch_geometric.transforms as T
data = T.ToUndirected()(data)

print("✅ Data loaded")

# Create loaders
train_loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE, 
                               input_nodes=('pekerja', data['pekerja'].train_mask), shuffle=True)
val_loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE,
                             input_nodes=('pekerja', data['pekerja'].val_mask), shuffle=False)
test_loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE,
                              input_nodes=('pekerja', data['pekerja'].test_mask), shuffle=False)

# Train
print("\n" + "="*60)
print("TRAINING TRANSFORMER (20 epochs)")
print("="*60)

base_model = GraphTransformer(HIDDEN_DIM, 1)
model = to_hetero(base_model, data.metadata(), aggr='mean').to(DEVICE)

batch = next(iter(train_loader)).to(DEVICE)
with torch.no_grad():
    model(batch.x_dict, batch.edge_index_dict)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
train_y = data['pekerja'].y[data['pekerja'].train_mask]
pos_weight = (train_y == 0).sum() / max((train_y == 1).sum(), 1)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

best_auc = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_samples = 0, 0
    pbar = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS}", leave=False)
    
    for batch in pbar:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        bs = batch['pekerja'].batch_size
        out_p = out['pekerja'][:bs].squeeze(-1)
        y = batch['pekerja'].y[:bs].float()
        loss = criterion(out_p, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bs
        total_samples += bs
        
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
    
    val_auc = roc_auc_score(val_labels, val_preds)
    _, val_f1 = find_optimal_threshold(np.array(val_labels), np.array(val_preds))
    print(f"Ep {epoch:2d} | Loss: {total_loss/total_samples:.4f} | Val AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "best_gnn_Transformer.pt")
        print(f"       ✅ New best! Saved.")

# Test
print("\n" + "="*60)
print("EVALUATING BEST TRANSFORMER ON TEST SET")
print("="*60)
model.load_state_dict(torch.load("best_gnn_Transformer.pt"))
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        bs = batch['pekerja'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        test_preds.extend(torch.sigmoid(out['pekerja'][:bs].squeeze(-1)).cpu().numpy())
        test_labels.extend(batch['pekerja'].y[:bs].cpu().numpy())

test_auc = roc_auc_score(test_labels, test_preds)
thresh, test_f1 = find_optimal_threshold(np.array(test_labels), np.array(test_preds))
test_preds_binary = (np.array(test_preds) >= thresh).astype(int)

print(f"✅ Transformer Results (thresh={thresh:.3f}):")
print(f"   AUC: {test_auc:.4f}")
print(f"   F1 : {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_labels, test_preds_binary, target_names=['Non-Fraud', 'Fraud']))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds_binary))
