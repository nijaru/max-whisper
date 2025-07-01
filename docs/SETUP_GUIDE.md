# Setup Guide

Complete setup guide for the max-whisper project with enhanced infrastructure.

## Prerequisites

- Linux x86_64 (tested on Fedora, Ubuntu)
- Python 3.9-3.11  
- CUDA 12+ (for GPU acceleration)
- Git

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/nijaru/max-whisper
cd max-whisper
```

### 2. Install Dependencies
```bash
make install    # Installs pixi and sets up environments
```

This will:
- Install pixi package manager
- Set up `default` environment (MAX Graph only)
- Set up `benchmark` environment (full dependencies)

### 3. Verify Installation
```bash
pixi run graph-test                    # Test MAX Graph
pixi run -e benchmark demo             # Test all implementations
```

## Usage

### Basic Commands
```bash
# Run comprehensive comparison
pixi run -e benchmark demo

# Test individual implementations
pixi run -e benchmark test-cpu         # CPU baseline
pixi run -e benchmark test-gpu         # GPU accelerated  
pixi run -e benchmark test-max         # MAX Graph hybrid

# Run tests
pixi run test                          # Full test suite
pixi run test-max                      # MAX Graph tests only
```

### Enhanced Benchmarking
```bash
# Structured benchmarking with error handling
pixi run -e benchmark benchmark

# JSON output for analysis/parsing
pixi run -e benchmark benchmark-json

# Save results to file
pixi run -e benchmark benchmark-save

# Legacy benchmark (for compatibility)
pixi run -e benchmark benchmark-legacy
```

### Development Commands
```bash
# Environment verification
pixi run graph-test                    # MAX Graph availability
make gpu-check                         # CUDA setup

# Individual model testing
pixi run -e benchmark test-cpu --model-size tiny
pixi run -e benchmark test-gpu --model-size small
pixi run -e benchmark test-max --model-size base
```

## Environments

### Default Environment
- MAX Graph and Mojo only
- Minimal dependencies
- Use: `pixi run <command>`

### Benchmark Environment  
- Full dependencies (PyTorch, OpenAI Whisper, etc.)
- GPU support with CUDA
- Use: `pixi run -e benchmark <command>`

## Project Structure

```
max-whisper/           # Main implementations
├── whisper_cpu.py     # CPU baseline
├── whisper_gpu.py     # GPU accelerated
├── whisper_max.py     # MAX Graph hybrid
├── audio/             # Audio processing (includes Mojo kernels)
└── utils/             # Utilities (includes logging)

benchmarks/            # Performance testing
├── benchmark_runner.py    # Enhanced runner
└── benchmark_all.py       # Legacy runner

tests/                 # Test suite
├── test_implementations.py
├── test_audio_processing.py
└── test_max_graph.py

examples/              # Demo scripts
docs/                  # Documentation
└── agent/            # Project tracking
```

## Expected Results

### CPU Baseline
- **Status**: ✅ Working  
- **Performance**: ~10.6s for 161s audio
- **Quality**: Perfect transcription

### GPU Accelerated
- **Status**: ✅ Working
- **Performance**: ~1.9s for 161s audio (5.7x speedup)  
- **Quality**: Perfect transcription

### MAX Graph Hybrid
- **Status**: ⚠️ Technical integration complete
- **Performance**: ~123ms encoder execution  
- **Quality**: Repetitive output (semantic tuning needed)

## Troubleshooting

### Common Issues

**"pixi not found"**
```bash
# Install pixi manually
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

**"MAX Graph not available"**
```bash
# Verify environment
pixi run graph-test
```

**"CUDA errors"**
```bash
# Check GPU setup
make gpu-check
nvidia-smi
```

### Getting Help

1. Check the logs for detailed error information
2. Use `pixi run -e benchmark benchmark-json` for structured debugging output
3. Run `pixi run test` to validate your setup
4. See `docs/agent/` for development guidance

## Next Steps

1. **Test the setup**: `pixi run -e benchmark demo`
2. **Run benchmarks**: `pixi run -e benchmark benchmark-json`  
3. **Explore the code**: Start with `max-whisper/whisper_cpu.py`
4. **Review documentation**: `docs/IMPLEMENTATION_GUIDE.md`

The project demonstrates successful MAX Graph architectural integration with ongoing work on semantic quality optimization.