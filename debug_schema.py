import pyarrow.dataset as ds
import os

ROOT_DIR = "/Volumes/Backup Plus/Zaman/graph/data"

node_dirs = {
    "node_nasabah": os.path.join(ROOT_DIR, "node_nasabah"),
    "node_pekerja": os.path.join(ROOT_DIR, "node_pekerja"),
    "node_pinjaman": os.path.join(ROOT_DIR, "node_pinjaman"),
    "node_simpanan": os.path.join(ROOT_DIR, "node_simpanan"),
    "node_transaksi": os.path.join(ROOT_DIR, "node_transaksi"),
}

print("=== Checking Parquet Schemas ===")

for name, path in node_dirs.items():
    if not os.path.exists(path):
        print(f"\n[MISSING] {name} at {path}")
        continue
        
    try:
        # Load dataset (lazy, just reads metadata)
        dataset = ds.dataset(path, format="parquet", partitioning="hive")
        print(f"\n[{name}]")
        print(f"Path: {path}")
        print("Columns:", dataset.schema.names)
        
        # Check first row to see actual values
        scanner = dataset.scanner(batch_size=1)
        first_batch = next(scanner.to_batches())
        df = first_batch.to_pandas()
        print("First row sample:")
        print(df.iloc[0].to_dict())
        
    except Exception as e:
        print(f"Error reading {name}: {e}")
