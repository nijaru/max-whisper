# MAX-Whisper Setup Guide

Complete setup instructions for MAX-Whisper speech recognition system.

## Prerequisites

- Linux system (tested on Fedora 42)
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- Python 3.11+

## Quick Setup

### 1. Install Pixi Package Manager
```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
```

### 2. Clone and Setup Project
```bash
git clone [repository-url]
cd modular-hackathon
pixi install
```

### 3. Setup CUDA Environment
```bash
source setup_cuda_env.sh
```

### 4. Verify Installation
```bash
# Test all components (should show 4/4 passing)
pixi run -e default python test_everything.py

# Test baseline comparison
pixi run -e benchmark python test_baselines_only.py
```

## Environment Details

### Pixi Environments
- `default` - MAX Graph models with CUDA
- `benchmark` - Baseline models (OpenAI, Faster-Whisper)

### Key Dependencies
- `max` - MAX Graph framework
- `nvidia-cublas-cu12` - CUDA libraries
- `openai-whisper` - Baseline comparison
- `faster-whisper` - Performance baseline
- `tiktoken` - Real tokenizer

## Troubleshooting

### CUDA Issues
If you see `ABORT: Failed to load CUDA cuBLAS library`:
```bash
# Check CUDA installation
nvidia-smi
which nvcc

# Verify pixi environment
pixi info
```

### Performance Issues
For optimal performance:
- Use GPU environment: `pixi run -e default`
- Ensure CUDA drivers are current
- Monitor GPU utilization: `nvidia-smi`

## Running Demos

### Component Tests
```bash
# Individual model tests
pixi run -e default python src/model/max_whisper_real_simple.py      # Simple encoder
pixi run -e default python src/model/max_whisper_step2.py             # Multi-head attention  
pixi run -e default python src/model/max_whisper_decoder.py           # Encoder-decoder
pixi run -e default python src/model/max_whisper_complete.py          # Complete model
```

### Comparison Benchmarks
```bash
# Head-to-head comparison (all models)
pixi run -e benchmark python benchmarks/real_audio_comparison.py

# Baseline validation only
pixi run -e benchmark python test_baselines_only.py
```

### Production Components
```bash
# Trained weights demo
pixi run -e benchmark python demo_trained_weights_simple.py

# Real tokenizer integration
pixi run -e benchmark python integrate_real_tokenizer.py
```

## Performance Expectations

### Current Results (RTX 4090)
- MAX-Whisper Complete: 3.6x real-time speedup
- OpenAI Whisper-tiny: 69.7x speedup (CPU)
- Faster-Whisper-tiny: 74.3x speedup (CPU)

### With Trained Weights (Target)
- MAX-Whisper + trained weights: 50-100x speedup expected

## File Structure

### Models
- `src/model/` - MAX Graph implementations
- `whisper_weights/` - Extracted PyTorch weights

### Audio
- `audio_samples/` - Test audio files
- `src/audio/` - Preprocessing utilities

### Benchmarks
- `benchmarks/` - Performance comparison scripts
- `test_*.py` - Component validation tests

### Documentation
- `docs/` - Detailed technical documentation
- `README.md` - Project overview
- `CLAUDE.md` - AI agent instructions