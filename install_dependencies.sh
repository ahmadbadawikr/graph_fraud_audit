#!/bin/bash
set -e

# 1. Install Core PyTorch first
# Using exact versions that work together
echo "Installing Core PyTorch..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# 2. Install PyG optional dependencies
# Unpinning strict versions to allow finding available wheels in the repo
echo "Installing PyG dependencies..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
  -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# 3. Install remaining requirements
echo "Installing remaining requirements..."
pip install torch-geometric==2.5.0 numpy>=1.25 tqdm>=4.66 scipy>=1.11 pandas>=2.1
