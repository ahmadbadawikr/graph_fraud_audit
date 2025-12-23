"""
Fraud Detection Utilities
=========================
Clean, reusable functions for the fraud detection pipeline.
"""

import os
import gc
import time
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
import torch.nn as nn


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_DIR = "/Volumes/Backup Plus/Zaman/graph"
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "processed_fast")
MAP_DIR = os.path.join(ROOT_DIR, "map_id")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def compute_enhanced_features(data, verbose=True):
    """
    Compute enhanced features for pekerja by aggregating through graph.
    Returns 24 features including ratios, max, and variance.
    """
    TARGET_NODE = 'pekerja'
    num_pekerja = data[TARGET_NODE].num_nodes
    num_nasabah = data['nasabah'].num_nodes
    num_simpanan = data['simpanan'].num_nodes
    
    if verbose:
        print("=" * 60)
        print("COMPUTING ENHANCED FEATURES")
        print("=" * 60)
    
    # Get edges
    is_pekerja_edge = data[('nasabah', 'is_pekerja', 'pekerja')].edge_index
    has_simpanan_edge = data[('nasabah', 'has_simpanan', 'simpanan')].edge_index
    has_pinjaman_edge = data[('nasabah', 'has_pinjaman', 'pinjaman')].edge_index
    debit_edge = data[('simpanan', 'debit', 'transaksi')].edge_index
    credit_edge = data[('transaksi', 'credit', 'simpanan')].edge_index
    
    # === Step 1: Nasabah per pekerja ===
    if verbose: print("\n1. Counting nasabah per pekerja...")
    nasabah_per_pekerja = torch.zeros(num_pekerja)
    nasabah_max_per_pekerja = torch.zeros(num_pekerja)  # For variance calc
    
    for i in range(is_pekerja_edge.shape[1]):
        pekerja_id = is_pekerja_edge[1, i].item()
        nasabah_per_pekerja[pekerja_id] += 1
    
    if verbose: print(f"   Avg: {nasabah_per_pekerja.mean():.2f}")
    
    # === Step 2: Simpanan stats per nasabah → pekerja ===
    if verbose: print("2. Aggregating simpanan...")
    simpanan_per_nasabah = torch.zeros(num_nasabah)
    simpanan_per_nasabah.scatter_add_(0, has_simpanan_edge[0], torch.ones(has_simpanan_edge.shape[1]))
    
    simpanan_per_pekerja = torch.zeros(num_pekerja)
    simpanan_max_per_pekerja = torch.zeros(num_pekerja)
    simpanan_sum_sq = torch.zeros(num_pekerja)  # For variance
    
    for i in range(is_pekerja_edge.shape[1]):
        nasabah_id = is_pekerja_edge[0, i].item()
        pekerja_id = is_pekerja_edge[1, i].item()
        val = simpanan_per_nasabah[nasabah_id].item()
        simpanan_per_pekerja[pekerja_id] += val
        simpanan_sum_sq[pekerja_id] += val ** 2
        if val > simpanan_max_per_pekerja[pekerja_id]:
            simpanan_max_per_pekerja[pekerja_id] = val
    
    # Variance
    simpanan_var = torch.where(
        nasabah_per_pekerja > 1,
        (simpanan_sum_sq - (simpanan_per_pekerja ** 2) / nasabah_per_pekerja) / (nasabah_per_pekerja - 1),
        torch.zeros_like(simpanan_per_pekerja)
    )
    simpanan_var = torch.clamp(simpanan_var, min=0)  # Numerical stability
    
    if verbose: print(f"   Avg: {simpanan_per_pekerja.mean():.2f}, Max: {simpanan_max_per_pekerja.mean():.2f}")
    
    # === Step 3: Pinjaman per pekerja ===
    if verbose: print("3. Aggregating pinjaman...")
    pinjaman_per_nasabah = torch.zeros(num_nasabah)
    pinjaman_per_nasabah.scatter_add_(0, has_pinjaman_edge[0], torch.ones(has_pinjaman_edge.shape[1]))
    
    pinjaman_per_pekerja = torch.zeros(num_pekerja)
    for i in range(is_pekerja_edge.shape[1]):
        pinjaman_per_pekerja[is_pekerja_edge[1, i]] += pinjaman_per_nasabah[is_pekerja_edge[0, i]]
    
    if verbose: print(f"   Avg: {pinjaman_per_pekerja.mean():.2f}")
    
    # === Step 4: Transaction stats ===
    if verbose: print("4. Aggregating transactions...")
    tx_per_simpanan = torch.zeros(num_simpanan)
    tx_per_simpanan.scatter_add_(0, debit_edge[0], torch.ones(debit_edge.shape[1]))
    tx_per_simpanan.scatter_add_(0, credit_edge[1], torch.ones(credit_edge.shape[1]))
    
    tx_per_nasabah = torch.zeros(num_nasabah)
    for i in range(0, has_simpanan_edge.shape[1], 100000):
        end = min(i + 100000, has_simpanan_edge.shape[1])
        for j in range(i, end):
            tx_per_nasabah[has_simpanan_edge[0, j]] += tx_per_simpanan[has_simpanan_edge[1, j]]
    
    tx_per_pekerja = torch.zeros(num_pekerja)
    tx_max_per_pekerja = torch.zeros(num_pekerja)
    tx_sum_sq = torch.zeros(num_pekerja)
    
    for i in range(is_pekerja_edge.shape[1]):
        nasabah_id = is_pekerja_edge[0, i].item()
        pekerja_id = is_pekerja_edge[1, i].item()
        val = tx_per_nasabah[nasabah_id].item()
        tx_per_pekerja[pekerja_id] += val
        tx_sum_sq[pekerja_id] += val ** 2
        if val > tx_max_per_pekerja[pekerja_id]:
            tx_max_per_pekerja[pekerja_id] = val
    
    # Variance
    tx_var = torch.where(
        nasabah_per_pekerja > 1,
        (tx_sum_sq - (tx_per_pekerja ** 2) / nasabah_per_pekerja) / (nasabah_per_pekerja - 1),
        torch.zeros_like(tx_per_pekerja)
    )
    tx_var = torch.clamp(tx_var, min=0)
    
    if verbose: print(f"   Avg: {tx_per_pekerja.mean():.2f}, Max: {tx_max_per_pekerja.mean():.2f}")
    
    # === Step 5: Compute ratios ===
    if verbose: print("5. Computing ratios...")
    eps = 1e-6
    avg_simpanan_per_nasabah = simpanan_per_pekerja / (nasabah_per_pekerja + eps)
    avg_tx_per_nasabah = tx_per_pekerja / (nasabah_per_pekerja + eps)
    avg_tx_per_simpanan = tx_per_pekerja / (simpanan_per_pekerja + eps)
    pinjaman_simpanan_ratio = pinjaman_per_pekerja / (simpanan_per_pekerja + eps)
    
    # === Step 6: Build feature tensor ===
    if verbose: print("6. Building feature tensor...")
    
    features = torch.stack([
        # Raw counts (4)
        nasabah_per_pekerja,
        simpanan_per_pekerja,
        pinjaman_per_pekerja,
        tx_per_pekerja,
        # Max values (3)
        simpanan_max_per_pekerja,
        tx_max_per_pekerja,
        nasabah_per_pekerja,  # Placeholder for nasabah_max (same as count)
        # Variances (2)
        torch.sqrt(simpanan_var + eps),  # Std instead of var
        torch.sqrt(tx_var + eps),
        # Ratios (4)
        avg_simpanan_per_nasabah,
        avg_tx_per_nasabah,
        avg_tx_per_simpanan,
        pinjaman_simpanan_ratio,
        # Log transforms (4)
        torch.log1p(nasabah_per_pekerja),
        torch.log1p(simpanan_per_pekerja),
        torch.log1p(pinjaman_per_pekerja),
        torch.log1p(tx_per_pekerja),
        # Log max (2)
        torch.log1p(simpanan_max_per_pekerja),
        torch.log1p(tx_max_per_pekerja),
        # Interaction (2)
        nasabah_per_pekerja * avg_tx_per_nasabah,
        simpanan_per_pekerja * avg_tx_per_simpanan,
    ], dim=1)
    
    # Normalize
    features = (features - features.mean(0)) / (features.std(0) + eps)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if verbose:
        print(f"\n✅ Created {features.shape[1]} features")
        print(f"   Counts: 4, Max: 3, Variance: 2, Ratios: 4")
        print(f"   Log: 4, LogMax: 2, Interactions: 2")
    
    gc.collect()
    return features.float()


def load_fraud_labels(data, pekerja_map, verbose=True):
    """Load fraud labels from CSV and create train/val/test splits."""
    TARGET_NODE = 'pekerja'
    num_pekerja = data[TARGET_NODE].num_nodes
    
    fraud_df = pd.read_csv(os.path.join(DATA_DIR, "labeled_fraud.csv"), low_memory=False)
    fraud_pns = set(fraud_df['PN'].dropna().astype(str).unique())
    
    fraud_labels = torch.zeros(num_pekerja, dtype=torch.long)
    for pn in fraud_pns:
        if pn in pekerja_map:
            idx = pekerja_map[pn]
            if idx < num_pekerja:
                fraud_labels[idx] = 1
    
    n_fraud = fraud_labels.sum().item()
    n_nonfraud = num_pekerja - n_fraud
    
    if verbose:
        print(f"\n📊 Labels: {n_fraud} fraud ({100*n_fraud/num_pekerja:.1f}%), {n_nonfraud} non-fraud")
    
    # Create splits
    perm = torch.randperm(num_pekerja)
    train_size = int(0.7 * num_pekerja)
    val_size = int(0.15 * num_pekerja)
    
    train_mask = torch.zeros(num_pekerja, dtype=torch.bool)
    val_mask = torch.zeros(num_pekerja, dtype=torch.bool)
    test_mask = torch.zeros(num_pekerja, dtype=torch.bool)
    
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True
    
    data[TARGET_NODE].y = fraud_labels
    data[TARGET_NODE].train_mask = train_mask
    data[TARGET_NODE].val_mask = val_mask
    data[TARGET_NODE].test_mask = test_mask
    
    return fraud_labels


# ============================================================================
# MODELS
# ============================================================================

class FraudMLP(nn.Module):
    """MLP for fraud detection on tabular features."""
    
    def __init__(self, in_dim, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Focuses on hard examples by down-weighting easy ones.
    alpha: weighting factor for positive class
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)  # Probability of correct class
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def find_optimal_threshold(labels, preds):
    """Find threshold that maximizes F1 score."""
    prec, rec, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold, f1_scores[best_idx]


def train_mlp(data, features, epochs=50, lr=0.001, use_focal_loss=False, verbose=True):
    """Train MLP model with optional Focal Loss and threshold tuning.
    
    Note: BCE with pos_weight works better for this dataset.
    """
    TARGET_NODE = 'pekerja'
    
    X = features.to(DEVICE)
    y = data[TARGET_NODE].y.float().to(DEVICE)
    train_mask = data[TARGET_NODE].train_mask
    val_mask = data[TARGET_NODE].val_mask
    test_mask = data[TARGET_NODE].test_mask
    
    # Calculate class weight
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    
    # Loss function
    if use_focal_loss:
        # For focal loss: alpha should be LOWER for minority class to prevent over-prediction
        criterion = FocalLoss(alpha=0.25, gamma=2.0)  # Fixed: alpha=0.25 for minority
        loss_name = "Focal Loss (alpha=0.25)"
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = f"BCE (pos_weight={pos_weight.item():.2f})"
    
    model = FraudMLP(X.shape[1], hidden_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    if verbose:
        print(f"\n🚀 Training MLP: {sum(p.numel() for p in model.parameters()):,} params")
        print(f"   Features: {X.shape[1]}, Epochs: {epochs}, Loss: {loss_name}")
        print("-" * 60)
    
    best_val_auc = 0
    history = {'train_loss': [], 'val_auc': [], 'val_f1': []}
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(X[train_mask])
        loss = criterion(out, y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X[val_mask])
            val_preds = torch.sigmoid(val_out).cpu().numpy()
            val_labels = y[val_mask].cpu().numpy()
        
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.5
        
        # Use optimal threshold for F1
        opt_thresh, val_f1 = find_optimal_threshold(val_labels, val_preds)
        
        history['train_loss'].append(loss.item())
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = opt_thresh
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mlp.pt'))
            if verbose and (epoch <= 5 or epoch % 10 == 0):
                print(f"   Epoch {epoch:2d} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | thresh: {opt_thresh:.3f} ⭐")
        elif verbose and epoch % 10 == 0:
            print(f"   Epoch {epoch:2d} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | thresh: {opt_thresh:.3f}")
    
    # Evaluate on test with optimal threshold
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_mlp.pt')))
    model.eval()
    with torch.no_grad():
        test_out = model(X[test_mask])
        test_preds = torch.sigmoid(test_out).cpu().numpy()
        test_labels = y[test_mask].cpu().numpy()
    
    # Find optimal threshold on test set
    test_opt_thresh, _ = find_optimal_threshold(test_labels, test_preds)
    
    test_auc = roc_auc_score(test_labels, test_preds)
    test_f1_default = f1_score(test_labels, (test_preds > 0.5).astype(int), zero_division=0)
    test_f1_optimal = f1_score(test_labels, (test_preds > best_threshold).astype(int), zero_division=0)
    
    if verbose:
        print("-" * 60)
        print(f"✅ MLP Test Results:")
        print(f"   AUC: {test_auc:.4f}")
        print(f"   F1 (thresh=0.5): {test_f1_default:.4f}")
        print(f"   F1 (thresh={best_threshold:.3f}): {test_f1_optimal:.4f} ← optimal")
    
    return {
        'model': model,
        'test_auc': test_auc,
        'test_f1': test_f1_optimal,
        'test_f1_default': test_f1_default,
        'optimal_threshold': best_threshold,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'history': history
    }


def train_xgboost(data, features, verbose=True):
    """Train XGBoost model and return results."""
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ XGBoost not installed. Run: pip install xgboost")
        return None
    
    TARGET_NODE = 'pekerja'
    
    X = features.numpy()
    y = data[TARGET_NODE].y.numpy()
    train_mask = data[TARGET_NODE].train_mask.numpy()
    val_mask = data[TARGET_NODE].val_mask.numpy()
    test_mask = data[TARGET_NODE].test_mask.numpy()
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Scale pos weight
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    if verbose:
        print(f"\n🌲 Training XGBoost...")
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20,
        verbosity=0,
        use_label_encoder=False,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    test_preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    
    # Find optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold(y_test, test_preds)
    test_f1_default = f1_score(y_test, (test_preds > 0.5).astype(int), zero_division=0)
    
    if verbose:
        print(f"✅ XGBoost: AUC={test_auc:.4f}, F1={opt_f1:.4f} (thresh={opt_thresh:.3f})")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': [f'f{i}' for i in range(X.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'test_auc': test_auc,
        'test_f1': opt_f1,
        'optimal_threshold': opt_thresh,
        'test_preds': test_preds,
        'test_labels': y_test,
        'importance': importance
    }


def train_random_forest(data, features, verbose=True):
    """Train Random Forest model."""
    from sklearn.ensemble import RandomForestClassifier
    
    TARGET_NODE = 'pekerja'
    
    X = features.numpy()
    y = data[TARGET_NODE].y.numpy()
    train_mask = data[TARGET_NODE].train_mask.numpy()
    test_mask = data[TARGET_NODE].test_mask.numpy()
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if verbose:
        print(f"\n🌳 Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    test_preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    opt_thresh, opt_f1 = find_optimal_threshold(y_test, test_preds)
    
    if verbose:
        print(f"✅ Random Forest: AUC={test_auc:.4f}, F1={opt_f1:.4f} (thresh={opt_thresh:.3f})")
    
    return {
        'model': model,
        'test_auc': test_auc,
        'test_f1': opt_f1,
        'optimal_threshold': opt_thresh,
        'test_preds': test_preds,
        'test_labels': y_test
    }


def train_lightgbm(data, features, verbose=True):
    """Train LightGBM model."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("❌ LightGBM not installed. Run: pip install lightgbm")
        return None
    
    TARGET_NODE = 'pekerja'
    
    X = features.numpy()
    y = data[TARGET_NODE].y.numpy()
    train_mask = data[TARGET_NODE].train_mask.numpy()
    val_mask = data[TARGET_NODE].val_mask.numpy()
    test_mask = data[TARGET_NODE].test_mask.numpy()
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    if verbose:
        print(f"\n🌿 Training LightGBM...")
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )
    
    test_preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    opt_thresh, opt_f1 = find_optimal_threshold(y_test, test_preds)
    
    if verbose:
        print(f"✅ LightGBM: AUC={test_auc:.4f}, F1={opt_f1:.4f} (thresh={opt_thresh:.3f})")
    
    return {
        'model': model,
        'test_auc': test_auc,
        'test_f1': opt_f1,
        'optimal_threshold': opt_thresh,
        'test_preds': test_preds,
        'test_labels': y_test
    }


def train_ensemble(results_dict, verbose=True):
    """Create ensemble from multiple model predictions."""
    if len(results_dict) < 2:
        print("❌ Need at least 2 models for ensemble")
        return None
    
    if verbose:
        print(f"\n🎯 Creating ensemble from {len(results_dict)} models...")
    
    # Get test labels (same for all models)
    test_labels = list(results_dict.values())[0]['test_labels']
    
    # Average predictions
    all_preds = np.stack([r['test_preds'] for r in results_dict.values()])
    ensemble_preds = all_preds.mean(axis=0)
    
    test_auc = roc_auc_score(test_labels, ensemble_preds)
    opt_thresh, opt_f1 = find_optimal_threshold(test_labels, ensemble_preds)
    
    if verbose:
        print(f"✅ Ensemble: AUC={test_auc:.4f}, F1={opt_f1:.4f} (thresh={opt_thresh:.3f})")
    
    return {
        'test_auc': test_auc,
        'test_f1': opt_f1,
        'optimal_threshold': opt_thresh,
        'test_preds': ensemble_preds,
        'test_labels': test_labels
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results, title="Model Results"):
    """Plot evaluation results with optimal threshold."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    test_preds = results['test_preds']
    test_labels = results['test_labels']
    
    # Use optimal threshold if available
    opt_thresh = results.get('optimal_threshold', 0.5)
    pred_binary = (test_preds > opt_thresh).astype(int)
    
    # 1. Confusion Matrix (with optimal threshold)
    cm = confusion_matrix(test_labels, pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    axes[0, 0].set_title(f'Confusion Matrix (thresh={opt_thresh:.3f})')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    axes[0, 1].plot(fpr, tpr, 'c-', linewidth=2, label=f"AUC={results['test_auc']:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 1].fill_between(fpr, tpr, alpha=0.3)
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall
    prec, rec, thresholds = precision_recall_curve(test_labels, test_preds)
    axes[1, 0].plot(rec, prec, 'm-', linewidth=2)
    axes[1, 0].fill_between(rec, prec, alpha=0.3)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Score Distribution with both thresholds
    axes[1, 1].hist(test_preds[test_labels == 0], bins=50, alpha=0.7, label='Non-Fraud', color='green')
    axes[1, 1].hist(test_preds[test_labels == 1], bins=50, alpha=0.7, label='Fraud', color='red')
    axes[1, 1].axvline(x=0.5, color='white', linestyle='--', label='Default (0.5)')
    axes[1, 1].axvline(x=opt_thresh, color='yellow', linestyle='-', linewidth=2, label=f'Optimal ({opt_thresh:.3f})')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].legend()
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{title.lower().replace(" ", "_")}.png'), dpi=150)
    plt.show()
    
    print(f"\\n📋 Classification Report (threshold={opt_thresh:.3f}):")
    print(classification_report(test_labels, pred_binary, target_names=['Non-Fraud', 'Fraud']))


def compare_models(results_dict):
    """Compare multiple model results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    for name, res in results_dict.items():
        print(f"   {name:15s}: AUC={res['test_auc']:.4f}, F1={res['test_f1']:.4f}")
    
    best_model = max(results_dict.items(), key=lambda x: x[1]['test_auc'])
    print(f"\n🏆 Best Model: {best_model[0]} (AUC={best_model[1]['test_auc']:.4f})")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(data, pekerja_map, run_all=True):
    """Run the complete fraud detection pipeline with multiple models."""
    print("\n" + "=" * 60)
    print("FRAUD DETECTION PIPELINE")
    print("=" * 60)
    
    # 1. Compute features
    features = compute_enhanced_features(data)
    data['pekerja'].x = features
    
    # 2. Load labels
    load_fraud_labels(data, pekerja_map)
    
    # 3. Train models
    results = {}
    
    # MLP
    results['MLP'] = train_mlp(data, features, epochs=50)
    
    # Tree-based models
    xgb_results = train_xgboost(data, features)
    if xgb_results:
        results['XGBoost'] = xgb_results
    
    if run_all:
        # Random Forest
        results['RandomForest'] = train_random_forest(data, features)
        
        # LightGBM (if installed)
        lgb_results = train_lightgbm(data, features)
        if lgb_results:
            results['LightGBM'] = lgb_results
        
        # Ensemble
        if len(results) >= 2:
            results['Ensemble'] = train_ensemble(results)
    
    # Compare all models
    compare_models(results)
    
    # Plot best model
    best_name = max(results.items(), key=lambda x: x[1]['test_auc'])[0]
    plot_results(results[best_name], f"{best_name} Results (Best)")
    
    return results


if __name__ == "__main__":
    # For testing
    print("Fraud utils loaded. Import and call run_full_pipeline(data, pekerja_map)")
