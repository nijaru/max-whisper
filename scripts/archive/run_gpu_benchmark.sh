#!/bin/bash
# Run GPU benchmark with proper CUDA library setup

echo "=== MAX-Whisper GPU Benchmark Script ==="
echo "Setting up environment..."

# Set up paths
export PATH="$HOME/.pixi/bin:$PATH"

# Find CUDA libraries in pixi environment
CUDA_LIB_PATH="/home/nick/github/modular-hackathon/.pixi/envs/benchmark/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_LIB_PATH/cublas/lib:$CUDA_LIB_PATH/cudnn/lib:$CUDA_LIB_PATH/nvtx/lib:$LD_LIBRARY_PATH"

echo "CUDA library path configured: $LD_LIBRARY_PATH"

# Run comprehensive benchmark
echo -e "\n=== Running comprehensive GPU benchmark ==="
pixi run -e benchmark python benchmarks/run_benchmark.py

# Run MAX Whisper simple test
echo -e "\n=== Running MAX Whisper GPU test ==="
pixi run -e default python src/model/max_whisper_simple.py

echo -e "\n=== Benchmark complete ===\n"