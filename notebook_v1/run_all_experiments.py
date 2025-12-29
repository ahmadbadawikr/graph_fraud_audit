"""
Master Experiment Runner
========================
Runs all training scripts sequentially and captures results to JSON.

Usage:
    python run_all_experiments.py

Output:
    experiment_results.json - All metrics from each experiment
"""

import subprocess
import json
import sys
import os
from datetime import datetime

# Configuration - Starting from script 7
SCRIPTS = [
    ("7_GNN_Standalone", "7_train_gnn_standalone.py"),
    ("8_Ensemble", "8_final_ensemble_optimization.py"),
    ("9_TransformerV2", "9_transformer_v2.py"),
    ("10_TransformerV3", "10_transformer_v3.py"),
    ("11_Champion", "11_transformer_champion.py"),
    ("12_Basic", "12_graph_transformer_basic.py"),
    ("13_Final", "13_transformer_final.py"),
    ("14_HGT", "14_train_hgt.py"),
    ("15_Hybrid", "15_hybrid_gnn_xgboost.py"),
]

OUTPUT_FILE = "/Users/kasyfur/graph_fraud_audit/notebook_v1/experiment_results.json"
LOG_DIR = "/Users/kasyfur/graph_fraud_audit/notebook_v1/training_logs"

def run_experiment(name, script_path):
    """Run a single experiment and capture output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {name} ({script_path})")
    print(f"{'='*60}\n")
    
    log_file = os.path.join(LOG_DIR, f"{name}.log")
    
    try:
        result = subprocess.run(
            ["python", script_path],
            cwd="/Users/kasyfur/graph_fraud_audit/notebook_v1",
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )
        
        output = result.stdout + result.stderr
        
        # Save log
        with open(log_file, 'w') as f:
            f.write(output)
        
        print(output)
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "log_file": log_file,
            "output": output[-5000:] if len(output) > 5000 else output  # Last 5000 chars
        }
        
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "log_file": log_file}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    print("=" * 60)
    print("MASTER EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Log directory: {LOG_DIR}")
    
    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    results = {
        "run_date": datetime.now().isoformat(),
        "experiments": {}
    }
    
    for name, script in SCRIPTS:
        result = run_experiment(name, script)
        results["experiments"][name] = {
            "script": script,
            **result
        }
        
        # Save intermediate results
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Logs saved to: {LOG_DIR}/")
    
    # Print summary
    print("\nSUMMARY:")
    for name, data in results["experiments"].items():
        status = data.get("status", "unknown")
        print(f"  {name}: {status}")

if __name__ == "__main__":
    main()
