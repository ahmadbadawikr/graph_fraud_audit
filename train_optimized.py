import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def load_csr(path):
    """Load extracted CSR components (mmap friendly)."""
    filename = path.split("/")[-1]
    # We use mmap_mode='r' to avoid loading everything if not needed immediately,
    # though csr_to_edge_index currently materializes it.
    indptr = np.load(f"{path}/{filename}_indptr.npy", mmap_mode="r")
    indices = np.load(f"{path}/{filename}_indices.npy", mmap_mode="r")
    return indptr, indices

def csr_to_edge_index(indptr, indices):
    """Convert CSR to edge_index."""
    row = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    col = indices
    # Optimization: Use int32 if graph size allows to save RAM
    edge_index = torch.from_numpy(np.vstack([row, col])).long()
    return edge_index

# ==========================================
# 2. DATA LOADING & GRAPH CONSTRUCTION
# ==========================================
print("Setting up paths...")
base_path = "/Users/ymnzaman/Documents/Project/Graph"  # UPDATED TO USER'S PATH from NB
# Note: You might need to change the above path to your actual path: /Users/kasyfur/graph_fraud_audit/
# But based on NBs, it seems the user has data in Documents.
# I will use the path commonly seen in the notebooks, BUT add a check.

# Adjust this to where your data actually lives
ADJ_PATH = "adjacency"  
if not os.path.exists(ADJ_PATH):
    # Fallback to the path seen in notebooks if local folder not found
    ADJ_PATH = "/Users/ymnzaman/Documents/Project/Graph/adjacency"

print(f"Loading adjacency matrices from {ADJ_PATH}...")

try:
    adj_pekerja = load_csr(f"{ADJ_PATH}/nasabah__is_pekerja__pekerja")
    adj_memiliki_simp = load_csr(f"{ADJ_PATH}/nasabah__memiliki_simp__simpanan")
    adj_memiliki_pinj = load_csr(f"{ADJ_PATH}/nasabah__memiliki_pinj__pinjaman")
    adj_simp_debit = load_csr(f"{ADJ_PATH}/simpanan__out__transaksi")
    adj_simp_credit = load_csr(f"{ADJ_PATH}/transaksi__in__simpanan")
    adj_pinj_debit = load_csr(f"{ADJ_PATH}/pinjaman__out__transaksi")
    adj_pinj_credit = load_csr(f"{ADJ_PATH}/transaksi__in__pinjaman")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure you have run the adjacency generation notebook first and paths are correct.")
    exit(1)

print("Constructing HeteroData object...")
data = HeteroData()

# Node counts (hardcoded from notebook logic, ideally loaded dynamically)
node_num_dict = {
    "pekerja": 6_250,
    "nasabah": 12_270_075,
    "simpanan": 15_636_712,
    "pinjaman": 1_524_589,
    "transaksi": 12_516_002,
}

for ntype, nnum in node_num_dict.items():
    data[ntype].num_nodes = nnum
    # Optimization: Add simple features if none exist (e.g., constant or random embeddings)
    # Here we simulate random features for demonstration
    # In production, load your real features!
    feat_dim = 16
    data[ntype].x = torch.randn(nnum, feat_dim) # WARNING: This uses RAM. 

# Load edges
data[("nasabah", "is_pekerja", "pekerja")].edge_index = csr_to_edge_index(*adj_pekerja)
data[("nasabah", "memiliki_simp", "simpanan")].edge_index = csr_to_edge_index(*adj_memiliki_simp)
data[("nasabah", "memiliki_pinj", "pinjaman")].edge_index = csr_to_edge_index(*adj_memiliki_pinj)
data[("simpanan", "out", "transaksi")].edge_index = csr_to_edge_index(*adj_simp_debit)
data[("transaksi", "in", "simpanan")].edge_index = csr_to_edge_index(*adj_simp_credit)
data[("pinjaman", "out", "transaksi")].edge_index = csr_to_edge_index(*adj_pinj_debit)
data[("transaksi", "in", "pinjaman")].edge_index = csr_to_edge_index(*adj_pinj_credit)

print("converting to undirected for sampling...")
data = ToUndirected()(data)

# ==========================================
# 3. OPTIMIZED DATALOADER
# ==========================================
# Reduce depth from 5 to 2
num_neighbors = {key: [15, 10] for key in data.edge_types}

# ==========================================
# 3. OPTIMIZED DATALOADER (M1 PRO TUNED)
# ==========================================
# Reduce depth from 5 to 2 for performance
num_neighbors = {key: [15, 10] for key in data.edge_types}

# M1 Optimization Note:
# On Apple Silicon + PyG, setting num_workers=0 is often FASTER and more memory efficient
# than using multiprocessing, because pyg-lib (if installed) uses multi-threaded C++ sampling
# which avoids the expensive data pickling/sharing overhead of Python worker processes.
# We also enable pin_memory=True to speed up transfer to the MPS (GPU) device.

if hasattr(data, 'share_memory_'):
    data.share_memory_()

train_loader = NeighborLoader(
    data,
    # Sample 'pekerja' nodes. (Replace with actual training mask if available!)
    input_nodes=("pekerja", None), 
    num_neighbors=num_neighbors,
    batch_size=2048,   # Larger batch size for M1 Unified Memory
    shuffle=True,
    num_workers=0,     # M1 Pro Recommendation: 0 (Main process with C++ threads) 
    pin_memory=True,   # Crucial for MPS performance
    persistent_workers=False 
)

print(f"Loader ready. Batch size: 2048, Workers: 0 (Optimized for M1), Layers: 2")

# ==========================================
# 4. MODEL DEFINITION
# ==========================================
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# M1 Pro Device Selection
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device} (Apple Silicon Acceleration)")

model = HeteroGNN(hidden_channels=64, out_channels=2)
model = to_hetero(model, data.metadata(), aggr='sum').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("Starting training...")
model.train()
for epoch in range(1, 6): # 5 Epochs
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x_dict, batch.edge_index_dict)
        
        # NOTE: You need valid labels. 
        # Using a dummy loss here because labels logic was missing in the notebook snippets.
        # Assuming we classify 'pekerja':
        # target = batch['pekerja'].y
        # For demo, generating fake targets
        batch_size = batch['pekerja'].batch_size
        out_pekerja = out['pekerja'][:batch_size]
        target = torch.randint(0, 2, (batch_size,), device=device)
        
        loss = F.cross_entropy(out_pekerja, target)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss)
        pbar.set_postfix(loss=loss.item())
        
    print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(train_loader):.4f}")

print("Training script finished.")
