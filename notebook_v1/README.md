# Graph Fraud Audit - Notebook V1

This directory contains the complete pipeline for training Graph Neural Networks to detect fraud in financial transaction data. The scripts are numbered sequentially to indicate the order of execution.

---

## Table of Contents

1.  [Data Preprocessing Notebooks (0_*)](#0-data-preprocessing-notebooks)
2.  [Core Data Pipeline (1-5)](#1-5-core-data-pipeline)
3.  [Standalone Training Scripts (6-16)](#6-16-standalone-training-scripts)
4.  [Utility Modules](#utility-modules)
5.  [Model Checkpoints](#model-checkpoints)

---

## 0. Data Preprocessing Notebooks

These notebooks handle the initial conversion and mapping of raw data into a format suitable for PyTorch Geometric. They are exploratory and iterative in nature.

### `0_check_schema.ipynb`
*   **Purpose**: Inspect the schema of the source LMDB databases. Validates that expected tables (edges, nodes) exist and have correct column names.
*   **Output**: Console logs of database schemas.

### `0_node_mapping.ipynb` / `0_node_mapping_fast.ipynb`
*   **Purpose**: Map string identifiers (e.g., Account Numbers `ACC-123`, Employee PNs `PN-456`) to contiguous integer IDs. This is mandatory for PyTorch tensors.
*   **Output**: Pickle files (`map_pekerja.pkl`, `map_nasabah.pkl`, etc.) saved to the `map_id/` directory.
*   **Note**: `_fast.ipynb` is an optimized version using more efficient data structures.

### `0_node_mapping_transaksi.ipynb`
*   **Purpose**: Specifically handles the `Transaksi` (Transaction) node mapping, which is the largest entity in the graph and requires special memory handling.
*   **Output**: `map_transaksi.pkl`.

### `0_edge_indexing.ipynb`
*   **Purpose**: Converts raw edge lists (source ID, destination ID) into the sparse tensor format (`edge_index`) expected by PyG.
*   **Output**: `.pt` files containing edge tensors.

### `0_split_edges.ipynb`
*   **Purpose**: Splits large edge files into smaller, manageable chunks for processing on memory-constrained systems.
*   **Output**: Chunked edge files.

---

## 1-5. Core Data Pipeline

These notebooks form the canonical data preparation workflow. Execute them in order.

### `1_lmdb_to_pt.ipynb`
*   **Purpose**: Read raw edge lists and node attributes from LMDB and serialize them into PyTorch (`.pt`) chunks.
*   **Input**: LMDB database files.
*   **Output**: Chunked `.pt` files in the `processed_fast/` directory.
*   **Key Technique**: Streams data to avoid loading TBs of data into RAM.

### `2_heterodata_builder.ipynb`
*   **Purpose**: Assemble all `.pt` chunks into a unified `HeteroData` object, the core data structure in PyG for heterogeneous graphs.
*   **Input**: `.pt` chunks, node ID maps.
*   **Output**: `heterodata.pt` (a single file representing the full graph).

### `3_adjacency.ipynb`
*   **Purpose**: Build Compressed Sparse Row (CSR) adjacency matrices from the `HeteroData` object.
*   **Output**: `indptr.npy`, `indices.npy` files for each edge type.
*   **Key Technique**: CSR allows for memory-mapped (`mmap`) access, enabling training on graphs larger than RAM.

### `4_NeighborLoader.ipynb`
*   **Purpose**: Validate that the `NeighborLoader` can correctly sample subgraphs from the constructed graph.
*   **Output**: Validation logs (no persistent output).
*   **Key Concept**: `NeighborLoader` is the mini-batching strategy for GNNs. It samples a fixed number of neighbors at each hop, preventing memory explosion.

### `5_GraphNN.ipynb`
*   **Purpose**: The initial, interactive GNN training notebook. Defines the first model architectures and training loops.
*   **Output**: Model checkpoints (`.pt` files).
*   **Note**: This was later refactored into standalone `.py` scripts for reproducibility.

---

## 6-16. Standalone Training Scripts

These are production-quality Python scripts designed to be run from the command line. Each script represents an experiment or model iteration.

### `6_parquet_pipeline_standalone.ipynb`
*   **Purpose**: An end-to-end, self-contained notebook that loads data from Parquet files (an alternative to LMDB), builds the graph, and trains a model. Useful for quick iteration without the full LMDB setup.
*   **Key Feature**: Contains a complete data-to-model pipeline in a single file.

---

### `7_train_gnn_standalone.py`
*   **Purpose**: Train and compare multiple homogeneous GNN architectures (GraphSAGE, GAT, Transformer) converted to heterogeneous via `to_hetero`.
*   **Configuration**:
    *   Batch Size: 512
    *   Hidden Dim: 32
    *   Hops: 2 (`[10, 5]`)
*   **Output**: `best_gnn_SAGE.pt`, `best_gnn_GAT.pt`, `best_gnn_Transformer.pt`.
*   **Usage**: `python 7_train_gnn_standalone.py`

---

### `8_final_ensemble_optimization.py`
*   **Purpose**: Orchestrate the final model pipeline:
    1.  Hyperparameter tuning for GraphTransformer.
    2.  Train XGBoost and MLP on tabular features.
    3.  Grid search for optimal ensemble weights.
*   **Configuration**:
    *   GNN: GraphTransformer with varying hidden dims and heads.
    *   Tabular: XGBoost, MLP.
*   **Output**: Console report of optimal ensemble weights and final AUC/F1.
*   **Usage**: `python 8_final_ensemble_optimization.py`

---

### `9_transformer_v2.py`
*   **Purpose**: Second iteration of the GraphTransformer. Introduces:
    *   LayerNorm and Skip Connections.
    *   Focal Loss for class imbalance.
    *   Cosine Annealing LR schedule.
    *   3 Layers.
*   **Configuration**:
    *   Hidden Dim: 64
    *   Heads: 2
    *   Dropout: 0.4
*   **Result**: Experienced **overfitting** due to excessive depth.
*   **Usage**: `python 9_transformer_v2.py`

---

### `10_transformer_v3.py`
*   **Purpose**: Corrective iteration. Fixes V2's overfitting by:
    *   Reducing capacity (Hidden Dim 32, 2 Layers).
    *   Aggressive regularization (Dropout 0.5, Weight Decay 1e-3).
    *   Reverting to stable Weighted BCE Loss.
*   **Result**: "Lean & Mean" stable model.
*   **Usage**: `python 10_transformer_v3.py`

---

### `11_transformer_champion.py`
*   **Purpose**: The "Champion" model attempt. Restores a high-capacity configuration with 3-hop sampling (`[15, 10, 5]`) to reach deeper into the transaction graph.
*   **Configuration**:
    *   Hidden Dim: 64
    *   Heads: 2
    *   Layers: 3 (with residual connections).
*   **Goal**: Achieve target AUC 0.725+ and F1 0.29+.
*   **Output**: `transformer_champion.pt`.
*   **Usage**: `python 11_transformer_champion.py`

---

### `12_graph_transformer_basic.py`
*   **Purpose**: A deliberate "Back to Basics" model. Reverts to the exact minimalist V1 configuration to establish a stable benchmark.
*   **Configuration**:
    *   Hidden Dim: 32, Heads: 1, Layers: 2.
    *   No LayerNorm, no residuals.
*   **Usage**: `python 12_graph_transformer_basic.py`

---

### `13_transformer_final.py`
*   **Purpose**: The "Best of Both Worlds" model. Combines V1's stability with V3's fast degree features.
*   **Configuration**:
    *   Hidden Dim: 32, Heads: 1.
    *   Fast degree features for auxiliary nodes.
    *   Optimized 2-hop sampling (`[15, 10]`).
*   **Output**: `final_best_transformer.pt`.
*   **Usage**: `python 13_transformer_final.py`

---

### `14_train_hgt.py`
*   **Purpose**: **Key Architectural Pivot**. Replaces the `to_hetero` wrapper with a native **Heterogeneous Graph Transformer (HGT)**. HGT learns distinct parameters for each edge type and node type, which is more semantically appropriate for this dataset.
*   **Configuration**:
    *   Hidden Dim: 64
    *   Heads: 4 (for multi-head attention on different relation types).
    *   Layers: 2.
*   **Output**: `hgt_model.pt`.
*   **Usage**: `python 14_train_hgt.py`

---

### `15_hybrid_gnn_xgboost.py`
*   **Purpose**: **Hybrid Feature Extraction Strategy**. Uses the GNN as a feature extractor rather than a classifier.
    1.  Train a GNN on the classification task.
    2.  Extract the learned 32-dim embeddings for all pekerja nodes.
    3.  Concatenate embeddings with original tabular features.
    4.  Train an XGBoost classifier on this enriched dataset.
*   **Output**: `hybrid_gnn_extractor.pt`, `hybrid_xgboost.json`.
*   **Result**: Demonstrates significant "Lift" from combining graph structure with tabular signal.
*   **Usage**: `python 15_hybrid_gnn_xgboost.py`

---

### `16_graph_eda.py`
*   **Purpose**: Comprehensive Exploratory Data Analysis (EDA) on the graph data.
*   **Generates**:
    1.  Degree Distribution Analysis (Fraud vs Non-Fraud).
    2.  Neighborhood Connectivity Patterns.
    3.  Feature Separability Visualization (PCA, t-SNE).
    4.  Class Imbalance Diagnostics.
    5.  Graph Structure Statistics.
*   **Output**: PNG plots saved to `eda_plots/` directory.
*   **Usage**: `python 16_graph_eda.py`

---

## Utility Modules

### `fraud_utils.py`
*   **Purpose**: A shared utility library containing reusable functions and classes for the entire pipeline.
*   **Contents**:
    *   **Configuration**: `ROOT_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `MAP_DIR`, `DEVICE`.
    *   **Feature Engineering**: `compute_enhanced_features()` - Computes 20+ aggregated features for pekerja nodes by traversing the graph.
    *   **Label Loading**: `load_fraud_labels()` - Loads fraud labels from CSV and creates train/val/test splits.
    *   **Models**: `FraudMLP`, `FocalLoss`.
    *   **Training Helpers**: `train_mlp()`, `train_xgboost()`, `train_random_forest()`, `train_lightgbm()`, `train_ensemble()`.
    *   **Evaluation**: `find_optimal_threshold()`, `plot_results()`, `compare_models()`.
*   **Usage**: `from fraud_utils import *`

### `eda_fraud_detection.py`
*   **Purpose**: An alternative, more detailed EDA script focused on fraud-specific analysis.
*   **Contents**: Similar to `16_graph_eda.py` but structured for Jupyter notebook cell-by-cell execution.

---

## Model Checkpoints

The following `.pt` and `.json` files are saved model weights and configurations:

| File                        | Description                                   |
| :-------------------------- | :-------------------------------------------- |
| `best_gnn_SAGE.pt`          | Best GraphSAGE model weights.                 |
| `best_gnn_GAT.pt`           | Best GAT model weights.                       |
| `best_gnn_Transformer.pt`   | Best GraphTransformer (to_hetero) weights.    |
| `final_best_transformer.pt` | Final optimized transformer (V6) weights.     |
| `transformer_champion.pt`   | Champion 3-layer transformer weights.         |
| `hgt_model.pt`              | Native HGT model weights.                     |
| `hybrid_gnn_extractor.pt`   | GNN feature extractor for hybrid pipeline.    |
| `hybrid_xgboost.json`       | XGBoost model trained on hybrid features.     |

---

## Quickstart

1.  **Prepare Data**: Run notebooks `1_lmdb_to_pt.ipynb` through `4_NeighborLoader.ipynb` to generate `heterodata.pt`.
2.  **Train a Model**: Run `python 14_train_hgt.py` to train the HGT model.
3.  **Run EDA**: Run `python 16_graph_eda.py` to generate analysis plots.

---

## Requirements

See `requirements.txt` for Python dependencies. Key packages include:
*   `torch >= 2.3.0`
*   `torch-geometric >= 2.5.0`
*   `xgboost`
*   `scikit-learn`
*   `pandas`, `numpy`, `matplotlib`, `seaborn`
