# Graph Fraud Audit

A sophisticated machine learning system for detecting fraudulent actors within financial institutions using **Graph Neural Networks (GNNs)**. This project transforms traditional tabular financial audit data into a **Knowledge Graph** to detect complex fraud patterns—such as circular money flows, employee collusion, and loan layering—that are invisible to standard SQL-based audits.

---

## Key Features

*   **Heterogeneous Graph Modeling**: Financial data is modeled as a graph with 5 node types (Customer, Employee, Savings, Loan, Transaction) and multiple relationship types.
*   **Heterogeneous Graph Transformer (HGT)**: A native GNN architecture that learns distinct attention weights for each relationship type, providing superior semantic understanding.
*   **Hybrid Ensemble**: Combines GNN structural embeddings with XGBoost and MLP for robust fraud classification.
*   **Large-Scale Data Engineering**: Handles graphs with millions of nodes using LMDB, CSR indexing, and Neighbor Sampling.
*   **Apple Silicon Optimized**: Tuned for M1/M2/M3 chips using the MPS backend.

---

## Documentation

| Document | Description |
|:---------|:------------|
| [PROJECT_PAPER.md](PROJECT_PAPER.md) | Comprehensive technical report covering methodology, experiments, and results. |
| [notebook_v1/README.md](notebook_v1/README.md) | Detailed documentation for all notebooks and training scripts. |
| [BUSINESS_DOCUMENTATION.md](BUSINESS_DOCUMENTATION.md) | Business logic and fraud scenarios targeted by the system. |
| [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) | Performance tuning for large-graph training on consumer hardware. |

---

## Directory Structure

```
.
├── README.md                       # This file
├── PROJECT_PAPER.md                # Full technical report
├── BUSINESS_DOCUMENTATION.md       # Business logic and fraud scenarios
├── OPTIMIZATION_GUIDE.md           # Performance tuning guide
├── install_dependencies.sh         # Dependency installation script
│
├── notebook/                       # Original Jupyter notebooks (deprecated)
│
└── notebook_v1/                    # Active development directory
    ├── README.md                   # ⭐ Detailed script documentation
    │
    ├── 0_*.ipynb                   # Data preprocessing notebooks
    ├── 1_lmdb_to_pt.ipynb          # LMDB to PyTorch conversion
    ├── 2_heterodata_builder.ipynb  # HeteroData assembly
    ├── 3_adjacency.ipynb           # CSR adjacency construction
    ├── 4_NeighborLoader.ipynb      # Neighbor sampling validation
    ├── 5_GraphNN.ipynb             # Initial GNN training
    │
    ├── 7_train_gnn_standalone.py   # Multi-GNN comparison (SAGE, GAT, Transformer)
    ├── 8_final_ensemble_optimization.py  # Ensemble weight optimization
    ├── 9-13_transformer_*.py       # GraphTransformer iterations (V2-V6)
    ├── 14_train_hgt.py             # ⭐ Heterogeneous Graph Transformer
    ├── 15_hybrid_gnn_xgboost.py    # ⭐ Hybrid GNN + XGBoost pipeline
    ├── 16_graph_eda.py             # Exploratory Data Analysis
    │
    ├── fraud_utils.py              # Shared utility functions
    ├── eda_fraud_detection.py      # EDA script for Jupyter
    ├── requirements.txt            # Python dependencies
    │
    └── *.pt / *.json               # Saved model checkpoints
```

---

## Graph Schema

The financial data is modeled as a **Heterogeneous Graph**:

**Nodes (Entities):**
| Node Type | Description |
|:----------|:------------|
| `nasabah` | Customer (demographic root) |
| `pekerja` | Employee (**Target for classification**) |
| `simpanan` | Savings Account |
| `pinjaman` | Loan Account |
| `transaksi` | Transaction (modeled as nodes for multi-party flows) |

**Edges (Relationships):**
| Edge Type | Semantics |
|:----------|:----------|
| `nasabah → simpanan` | Ownership |
| `nasabah → pinjaman` | Ownership |
| `nasabah ↔ pekerja` | Identity resolution (is the customer also an employee?) |
| `simpanan → transaksi` | Debit (money out) |
| `transaksi → simpanan` | Credit (money in) |

---

## Quickstart

### 1. Install Dependencies

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

Or manually:
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torch-geometric pandas numpy scipy tqdm xgboost scikit-learn matplotlib seaborn
```

### 2. Prepare Data

Run the data pipeline notebooks in `notebook_v1/` (steps 1-4) to generate `heterodata.pt`.

### 3. Train a Model

```bash
cd notebook_v1

# Train the Heterogeneous Graph Transformer (recommended)
python 14_train_hgt.py

# Or run the Hybrid GNN + XGBoost pipeline
python 15_hybrid_gnn_xgboost.py

# Or run the full ensemble optimization
python 8_final_ensemble_optimization.py
```

### 4. Run EDA

```bash
python 16_graph_eda.py
# Plots saved to notebook_v1/eda_plots/
```

---

## Model Performance

| Model | AUC | F1 | Notes |
|:------|:----|:---|:------|
| GraphSAGE (baseline) | ~0.70 | ~0.25 | Homogeneous baseline |
| GraphTransformer (to_hetero) | ~0.72 | ~0.28 | Wrapped homogeneous model |
| **HGT (Native)** | **~0.75** | **~0.30** | **Recommended** |
| Hybrid GNN + XGBoost | ~0.76 | ~0.31 | Best for production |

---

## Notes

*   **Hardcoded Paths**: Scripts contain user-specific paths (e.g., `/Users/kasyfur/...`). Update these in `fraud_utils.py` before running.
*   **Large Data Handling**: The pipeline is specifically engineered for graphs that don't fit in RAM, using memory-mapped files and streaming.
*   **Apple Silicon**: For M1/M2/M3, see `OPTIMIZATION_GUIDE.md` for MPS-specific tuning.

---

## License

This project is for research and educational purposes.
