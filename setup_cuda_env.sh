#!/bin/bash
# Setup CUDA environment for MAX Graph GPU execution

echo "Setting up CUDA environment for MAX-Whisper..."

# Find CUDA libraries in pixi environment
PIXI_ENV="/home/nick/github/modular-hackathon/.pixi/envs/default/lib/python3.11/site-packages/nvidia"

# Set library paths
export LD_LIBRARY_PATH="$PIXI_ENV/cublas/lib:$PIXI_ENV/cudnn/lib:$PIXI_ENV/cuda_runtime/lib:$PIXI_ENV/cuda_nvtx/lib:$LD_LIBRARY_PATH"

echo "CUDA library paths set:"
echo "  CUBLAS: $PIXI_ENV/cublas/lib"
echo "  CUDNN: $PIXI_ENV/cudnn/lib"

# Verify libraries exist
if [ -f "$PIXI_ENV/cublas/lib/libcublas.so.12" ]; then
    echo "✅ Found libcublas.so.12"
else
    echo "❌ libcublas.so.12 not found!"
fi

# Export for child processes
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Environment ready. You can now run:"
echo "  pixi run -e default python src/model/max_whisper_simple.py"
echo "  pixi run -e default python test_max_gpu_simple.py"