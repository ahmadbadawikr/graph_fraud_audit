import os
import pandas as pd
import glob

ROOT_DIR = "/Volumes/Backup Plus/Zaman/graph/data"

node_dirs = {
    "node_nasabah": (os.path.join(ROOT_DIR, "node_nasabah"), "cif"),
    "node_pekerja": (os.path.join(ROOT_DIR, "node_pekerja"), "pn"),
    "node_pinjaman": (os.path.join(ROOT_DIR, "node_pinjaman"), "acctno"),
    "node_simpanan": (os.path.join(ROOT_DIR, "node_simpanan"), "acctno"),
    "node_transaksi": (os.path.join(ROOT_DIR, "node_transaksi"), "id_trx"),
}

print("=== Quick Check Parquet Values ===")

for name, (path, id_col) in node_dirs.items():
    print(f"\n[{name}]")
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    
    # Find first parquet file
    files = glob.glob(os.path.join(path, "*.parquet"))
    # If recursive/partitioned, try deeper
    if not files:
        files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        
    if not files:
        print("No .parquet files found.")
        continue
        
    target_file = files[0]
    print(f"Reading file: {os.path.basename(target_file)}")
    
    try:
        # Read just a few rows and only columns of interest
        df = pd.read_parquet(target_file, columns=[id_col])
        print(f"Total rows in sample file: {len(df)}")
        print(f"Data Type of '{id_col}': {df[id_col].dtype}")
        print("First 5 values:")
        print(df[id_col].head().tolist())
        
        # Check for nulls
        nulls = df[id_col].isnull().sum()
        print(f"Nulls in sample: {nulls}")
        
    except Exception as e:
        print(f"Error reading: {e}")
