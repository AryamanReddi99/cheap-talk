#!/bin/bash

# Simple run script for DDPG on SMAX environments
# Activate the conda environment and run DDPG

echo "Running DDPG on SMAX environment..."
echo "Map: 3s_vs_5z"
echo "Algorithm: Deep Deterministic Policy Gradient (DDPG)"
echo "=========================================="

# Run the DDPG algorithm
python ddpg.py

echo "DDPG training completed!"
