# Fedora/RTX 4090 Setup Guide

Complete guide for transitioning MAX-Whisper development from macOS to Fedora with RTX 4090 GPU acceleration.

## Overview

**Current State**: Phase 1 complete on macOS with 8.5x speedup (RTF = 0.0063)  
**Target**: GPU acceleration for 50-100x total speedup using MAX Graph + CUDA  
**Hardware**: Fedora desktop with RTX 4090 (24GB VRAM)

## Pre-Setup Checklist

### Hardware Verification
```bash
# Verify RTX 4090 detection
lspci | grep -i nvidia
nvidia-smi

# Check VRAM availability (should show 24GB)
nvidia-smi --query-gpu=memory.total --format=csv
```

### System Requirements
- **OS**: Fedora 38+ (or compatible RHEL-based)
- **GPU**: RTX 4090 (24GB VRAM) 
- **RAM**: 32GB+ recommended for large model loading
- **Storage**: 50GB+ free space for models and cache
- **CUDA**: 12.0+ compatible drivers

## Step 1: Repository Setup (15 minutes)

### Clone and Initialize
```bash
# Clone the repository
git clone https://github.com/your-username/modular-hackathon.git
cd modular-hackathon

# Verify macOS work is present
ls -la src/
ls -la benchmark_results.json  # Should show recent macOS results
```

### Transfer Audio Cache (Optional)
```bash
# If transferring from macOS, copy YouTube audio cache
# This avoids re-downloading test audio
scp -r user@macos-machine:~/github/modular-hackathon/audio_cache ./
```

## Step 2: Environment Setup (45 minutes)

### Install Pixi Package Manager
```bash
# Install pixi (cross-platform package manager)
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc  # Reload shell

# Verify installation
pixi --version
```

### Install Dependencies
```bash
# Install all dependencies (this may take 15-20 minutes)
pixi install

# Verify environments are created
pixi env list
# Should show: default, benchmark

# Test basic functionality
pixi run hello           # Test Mojo compilation
pixi run graph-test      # Test MAX Graph availability
```

### GPU Environment Verification
```bash
# Test NVIDIA GPU availability in pixi environment
pixi run -e benchmark python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Should output:
# CUDA available: True
# GPU count: 1  
# GPU name: NVIDIA GeForce RTX 4090
```

## Step 3: Baseline Validation (15 minutes)

### Test Existing Components
```bash
# Test Mojo kernel (should work on CPU)
pixi run mojo src/audio/working_kernel.mojo

# Test benchmark suite with real audio
pixi run -e benchmark python src/benchmarks/whisper_comparison.py

# Test demo interface
pixi run -e benchmark python src/demo/end_to_end_demo.py --quick
```

### Verify Performance Baseline
Expected results from macOS (for comparison):
- **Mojo preprocessing**: ~2.7ms (RTF = 0.00009)
- **OpenAI Whisper**: ~315ms (RTF = 0.0105) 
- **Faster-Whisper**: ~428ms (RTF = 0.0143)

## Step 4: GPU Development Setup (30 minutes)

### MAX Graph GPU Configuration
```bash
# Test MAX Graph with GPU support
pixi run python -c "
from max import engine
from max.graph import DeviceRef
print('GPU device available:', DeviceRef.GPU().is_available())
"
```

### CUDA Development Tools
```bash
# Install additional CUDA development tools if needed
sudo dnf groupinstall "Development Tools"
sudo dnf install cuda-toolkit-12-* 

# Verify nvcc compiler
nvcc --version
```

### Mojo GPU Development
```bash
# Test Mojo GPU compilation capabilities
pixi run python -c "
import subprocess
result = subprocess.run(['mojo', '--help'], capture_output=True, text=True)
print('GPU flags available:' if '--gpu' in result.stdout else 'GPU compilation check needed')
"
```

## Step 5: Development Priorities

### Phase 2 Implementation Order

1. **Complete MAX Graph Implementation** (4-6 hours)
   - File: `src/model/max_whisper.py`
   - Priority: Full Whisper encoder/decoder on GPU
   - Target: Real transcription output

2. **GPU Mojo Kernels** (2-4 hours)
   - File: `src/audio/gpu_kernels.mojo` (new)
   - Priority: Mel-spectrogram on CUDA
   - Target: 10-50x preprocessing speedup

3. **Performance Optimization** (2-3 hours)
   - Memory layout optimization
   - Batch processing
   - Precision tuning (fp16/int8)

4. **Integration Testing** (1-2 hours)
   - End-to-end GPU pipeline
   - Accuracy validation
   - Performance benchmarking

## Step 6: Development Workflow

### GPU Resource Management
```bash
# Monitor GPU usage during development
watch -n 1 nvidia-smi

# Profile memory usage
pixi run python -c "
import torch
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
print(f'Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f}GB')
"
```

### Code Development Cycle
```bash
# 1. Edit code
# 2. Test component
pixi run mojo src/audio/gpu_kernels.mojo

# 3. Integration test  
pixi run -e benchmark python src/benchmarks/whisper_comparison.py

# 4. Commit progress
git add . && git commit -m "GPU implementation: [component]"
```

## Step 7: Performance Targets

### GPU Performance Goals
- **Preprocessing**: <0.1ms (100x speedup from current 2.7ms)
- **Inference**: <5ms total (50x speedup from current 315ms)
- **RTF Target**: <0.001 (1000x real-time)
- **Memory**: <8GB GPU utilization

### Benchmarking Commands
```bash
# Run comprehensive benchmark suite
pixi run -e benchmark python src/benchmarks/whisper_comparison.py

# Run performance demo
pixi run -e benchmark python src/demo/end_to_end_demo.py

# Generate results for hackathon submission
pixi run -e benchmark python -c "
import json
with open('gpu_results.json', 'w') as f:
    json.dump({'phase': 'gpu_implementation', 'timestamp': __import__('time').time()}, f)
"
```

## Step 8: Lambda.ai Deployment Preparation

### Container Setup (Future)
```bash
# Prepare for lambda.ai deployment
# (This will be Phase 3 - demo deployment)
docker build -t max-whisper-demo .
docker run --gpus all max-whisper-demo
```

## Troubleshooting

### Common Issues

**MAX Graph GPU not available:**
```bash
# Check CUDA environment
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Reinstall MAX with GPU support
pixi clean
pixi install
```

**NVIDIA driver issues:**
```bash
# Update NVIDIA drivers
sudo dnf update nvidia-driver*
sudo reboot
```

**Memory errors:**
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Monitor memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1
```

### Development Tips

1. **Start Small**: Test individual components before integration
2. **Profile Early**: Use nvidia-smi and torch profiler frequently  
3. **Save Progress**: Commit working code at each milestone
4. **Monitor Resources**: Watch GPU memory and temperature
5. **Backup Strategy**: Keep CPU versions working as fallback

## Success Criteria

### Phase 2 Complete When:
- [ ] MAX Graph Whisper working on GPU
- [ ] Mojo GPU kernels functional  
- [ ] End-to-end pipeline <10ms total
- [ ] Real transcription accuracy maintained
- [ ] Comprehensive benchmarks show >20x speedup

### Ready for Demo When:
- [ ] Consistent <5ms inference times
- [ ] Professional demo interface
- [ ] Side-by-side comparison working
- [ ] Results documented for judges

---

**Next Steps**: Follow this guide sequentially, testing each component before proceeding. The goal is to achieve 50-100x total speedup while maintaining transcription accuracy for the hackathon demo.