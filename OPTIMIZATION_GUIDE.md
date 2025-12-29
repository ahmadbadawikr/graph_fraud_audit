# Optimization Guide for Large Graph Training

The current training pipeline faces performance bottlenecks due to the sheer size of the graph ("Big Data") and the depth of the neighbor sampling. Below are the specific issues identified and the recommended solutions implemented in `train_optimized.py`.

## 1. Issue: Excessive GNN Depth
**Current State:** The notebook attempts to sample **5 layers** of neighbors (`[8, 8, 8, 8, 8]`).
**Impact:** GNN sampling grows exponentially.
- Layer 1: 8 neighbors
- ...
- Layer 5: $8^5 = 32,768$ neighbors *per source node*.
For a batch size of 1024, this potentially touches millions of nodes per step, leading to massive I/O overhead and slow epoch times.

**Solution:** Reduce Graph depth to **2 or 3 layers**.
- Standard industry practice (e.g., GraphSAGE, PinSAGE) typically uses 2-3 layers.
- New configuration: `num_neighbors=[20, 10]` or `[15, 10, 5]`. This captures the immediate neighborhood (most relevant for fraud) without exploding computation.

## 2. Issue: Inefficient Data Loading (RAM Bottleneck)
**Current State:**
The `csr_to_edge_index` function uses `np.vstack` and `torch.from_numpy`, which forces a copy of the memory-mapped arrays into RAM.
```python
edge_index = torch.from_numpy(np.vstack([row, col])).long()  # Forces full RAM load
```
If the graph edges exceed available RAM, the OS will start **swapping** to disk (Virtual Memory), causing training speed to plummet.

**Solution:**
1.  **Pin Memory:** Use `pin_memory=True` in the DataLoader to speed up transfer to GPU/MPS.
2.  **Num Workers:** Increase `num_workers` (e.g., 4 or 8) to perform sampling in parallel processes.
3.  **Refactor Loading**: While full `mmap` optimization requires a custom GraphStore, reducing the GNN depth (Point 1) significantly reduces the memory bandwidth pressure.

## 3. Apple Silicon (M1/M2/M3) Specific Optimizations
**Status:** You identified you are using an **M1 Pro** chip. This requires specific tuning different from NVIDIA CUDA setups.

1.  **Device: `mps`**:
    - The optimized script uses `device = torch.device('mps')`.
    - This leverages the Neural Engine and GPU cores on the M1 Pro for typical GNN matrix multiplications (`SAGEConv`).

2.  **DataLoader Strategy: `num_workers=0`**:
    - **Why?** On macOS, Python multiprocessing (`spawn` method) has high overhead and duplicates memory for large graphs.
    - **Optimization:** We set `num_workers=0`. This runs the dataloader in the main process but utilizes `pyg-lib`'s **C++ multi-threading** for sampling. This is typically faster and far more memory-efficient on Unified Memory architectures than spawning multiple Python workers.

3.  **Memory Management**:
    - **Pin Memory:** `pin_memory=True` is enabled. It allocates page-locked memory, accelerating the data transfer from CPU RAM to MPS (GPU) memory.
    - **Batch Size:** The M1 Pro has high memory bandwidth. We increased the batch size to `2048` to saturate the GPU.

4.  **Recommended Usage**:
    - Run the script with `python train_optimized.py`.
    - Monitor GPU usage via `sudo powermetrics --samplers gpu_power -i 1000`.

## 4. Suggested Architecture Improvements (Implemented in Script)
- **Model**: `GRAPH_SAGE` or `GAT` (Graph Attention Network) with 2 layers.
- **Hidden Channels**: 64 or 128 (start small).
- **Batch Size**: Increase to `2048` or `4096` if VRAM allows, since fewer neighbors are sampled.

## Summary of Changes in `train_optimized.py`
1.  **Shallow Sampling**: Samples only 2 hops `[15, 10]`.
2.  **Parallel Loading**: Sets `num_workers=4`.
3.  **MPS Acceleration**: auto-detects Mac GPU.
4.  **Clean Loop**: A standard PyTorch training loop with tqdm logging.
