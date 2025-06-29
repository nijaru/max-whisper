# Scripts Directory

Utility scripts for MAX-Whisper development and deployment.

## Setup Scripts
- `setup_cuda_env.sh` - Configure CUDA environment variables
- `deploy_lambda_ai.sh` - Automated cloud deployment

## Benchmark Scripts  
- `run_full_benchmark.py` - Complete performance testing
- `benchmark_max_only.py` - MAX-Whisper specific benchmarks
- `run_gpu_benchmark.sh` - GPU performance testing

## Utility Scripts
- `extract_whisper_weights.py` - Extract PyTorch weights to NumPy format

## Usage

### Environment Setup
```bash
# Setup CUDA (run once per session)
source scripts/setup_cuda_env.sh

# Deploy to cloud
./scripts/deploy_lambda_ai.sh
```

### Benchmarking
```bash
# Full benchmark suite
python scripts/run_full_benchmark.py

# MAX-Whisper only
python scripts/benchmark_max_only.py

# GPU performance
./scripts/run_gpu_benchmark.sh
```

### Weight Extraction
```bash
# Extract from OpenAI model
python scripts/extract_whisper_weights.py
```