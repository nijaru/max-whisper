# MAX-Whisper Scripts

Essential utility scripts for setup, deployment, and weight management.

## 🎯 Essential Scripts (Use These)

### Environment Setup
- **[setup_cuda_env.sh](setup_cuda_env.sh)** - ⭐ **ESSENTIAL** - Configure CUDA environment variables

### Deployment
- **[deploy_lambda_ai.sh](deploy_lambda_ai.sh)** - Automated cloud deployment for maximum performance

### Data Preparation
- **[extract_whisper_weights.py](extract_whisper_weights.py)** - Extract trained weights from OpenAI Whisper models

## Usage

### Environment Setup
```bash
# Configure CUDA environment (run once per session)
source scripts/setup_cuda_env.sh
```

### Cloud Deployment
```bash
# Deploy complete system to Lambda AI
./scripts/deploy_lambda_ai.sh
```

### Weight Extraction
```bash
# Extract trained weights from OpenAI model
pixi run -e benchmark python scripts/extract_whisper_weights.py
```

## 📁 Development Scripts

### Archive
- **[archive/](archive/)** - Development and experimental scripts
  - `benchmark_max_only.py` - MAX-Whisper specific benchmarks
  - `run_full_benchmark.py` - Complete performance testing
  - `run_gpu_benchmark.sh` - GPU performance testing

## Script Details

### setup_cuda_env.sh
Sets up essential CUDA environment variables for GPU acceleration:
- Configures CUDA library paths
- Sets LD_LIBRARY_PATH for cuBLAS
- Enables MAX Graph GPU execution

### deploy_lambda_ai.sh
Automated deployment script for cloud performance testing:
- Transfers project to cloud server
- Sets up environment automatically
- Runs comprehensive benchmarks

### extract_whisper_weights.py
Extracts trained weights from OpenAI Whisper models:
- Converts PyTorch tensors to NumPy format
- Extracts 47 weight tensors from Whisper-tiny
- Saves to `whisper_weights/whisper_tiny_weights.npz`

## 🚀 For New Users

**Essential workflow:**
1. Run `source scripts/setup_cuda_env.sh` 
2. Test with `pixi run -e default python tests/test_everything.py`
3. For cloud deployment: `./scripts/deploy_lambda_ai.sh`

## 🏆 For Hackathon Judges

**Key scripts for evaluation:**
- `setup_cuda_env.sh` - Required for running the system
- `extract_whisper_weights.py` - Demonstrates weight portability from PyTorch
- `deploy_lambda_ai.sh` - Shows cloud deployment capability