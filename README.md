# 🎤 MAX Graph Whisper: High-Performance Speech Recognition

[![Performance](https://img.shields.io/badge/Speedup-2.4x-brightgreen)](https://github.com/nijaru/modular-hackathon)
[![Quality](https://img.shields.io/badge/Quality-Perfect%20%E2%9C%85-brightgreen)](https://github.com/nijaru/modular-hackathon)
[![Platform](https://img.shields.io/badge/Platform-MAX%20Graph-blue)](https://github.com/nijaru/modular-hackathon)

**🏆 Modular Hackathon 2025 Submission**

## 🎯 What We Built

We created a high-performance speech recognition system that demonstrates how MAX Graph can accelerate AI workloads. Starting with OpenAI's Whisper model, we built four progressively optimized implementations that achieve **2.4x speedup** while maintaining **perfect transcription quality**.

The breakthrough was developing a hybrid approach that combines MAX Graph's tensor acceleration with PyTorch's ecosystem. Our fastest implementation eliminates unnecessary weight conversions, streamlines the processing pipeline, and uses direct tensor operations that bypass costly memory transfers.

### Performance Results

| Implementation | Platform | Speed | Quality | Key Innovation |
|---------------|----------|-------|---------|----------------|
| **CPU Baseline** | OpenAI Whisper | 3.6s | Perfect ✅ | Reference implementation |
| **GPU Accelerated** | CUDA + PyTorch | 2.0s | Perfect ✅ | CUDA optimization |
| **MAX Graph** | MAX Graph + PyTorch | 2.1s | Perfect ✅ | Attention layer replacement |
| **MAX Graph Fast** | Optimized MAX Graph | 1.5s | Perfect ✅ | **Streamlined processing** |

*Tested on RTX 4090 with 161.5s technical audio*

## 🚀 Quick Start

### Installation and Basic Demo
```bash
# Clone and setup (one-time)
git clone https://github.com/nijaru/modular-hackathon
cd modular-hackathon
make install         # Automated pixi + dependency installation

# Run demo with all 4 implementations
make                 # Recommended demo (small model)
```

### Command Overview
```bash
# Model size shortcuts (full demo)
make tiny            # Fastest demo (tiny model)
make small           # Recommended demo (small model) 
make base            # Best quality demo (base model)

# Individual implementation tests
make cpu tiny        # Just CPU baseline
make gpu small       # Just GPU accelerated
make max base        # Just MAX Graph integration
make fast small      # Just MAX Graph optimized

# Analysis and utilities
make benchmark       # Complete performance analysis
make env-check       # Verify environment setup
make gpu-check       # Check GPU compatibility
make help           # Show all available commands
```

## 📊 How We Achieved 2.4x Speedup

### Progressive Optimization Strategy

**CPU to GPU (1.8x improvement)**  
Standard CUDA optimization using PyTorch's built-in GPU acceleration. This establishes our GPU performance baseline.

**MAX Graph Integration (competitive with CUDA)**  
We replaced Whisper's attention layers with MAX Graph implementations while keeping the rest in PyTorch. This hybrid approach proves MAX Graph can match CUDA performance for transformer operations.

**MAX Graph Fast (2.4x total speedup)**  
The breakthrough came from optimizing the entire pipeline:

- **Eliminated weight conversion overhead** - Direct processing instead of PyTorch→MAX Graph copying
- **Streamlined tensor operations** - Minimal-overhead MAX Graph operations with focused computations  
- **Optimized memory management** - Reduced allocations and transfers
- **Simplified processing pipeline** - Removed unnecessary intermediate steps

This isn't just faster MAX Graph operations - it's a fundamentally more efficient architecture designed for MAX Graph from the ground up.

## 🛠️ Technical Architecture

### Four Implementation Strategy
1. **CPU Baseline** (`whisper_cpu.py`) - Pure OpenAI Whisper for quality reference
2. **GPU Accelerated** (`whisper_gpu.py`) - CUDA optimization showing standard GPU performance  
3. **MAX Graph Integration** (`whisper_max.py`) - Hybrid architecture with attention layer replacement
4. **MAX Graph Fast** (`whisper_max_fast.py`) - Fully optimized pipeline designed for maximum performance

### Key Innovations
- **Hybrid Architecture**: Successfully combines MAX Graph acceleration with PyTorch compatibility
- **Progressive Optimization**: Clear path from CPU baseline to cutting-edge acceleration
- **Quality Preservation**: All implementations produce identical transcription output

### Project Structure
```
├── src/model/           # Four implementations
│   ├── whisper_cpu.py      # CPU baseline (3.6s)
│   ├── whisper_gpu.py      # GPU accelerated (2.0s, 1.8x)
│   ├── whisper_max.py      # MAX Graph integration (2.1s, 1.7x)  
│   └── whisper_max_fast.py # MAX Graph optimized (1.5s, 2.4x)
├── scripts/tui_demo.py  # Professional demo interface
├── benchmark_all.py     # Complete performance analysis
└── audio_samples/       # Test audio (161.5s technical content)
```

## 🏆 Hackathon Submission Details

### Reproducible Results
All results are immediately reproducible with one-command setup. Benchmarks use identical methodology across implementations with the same 161.5-second audio input on documented hardware (RTX 4090).

### Correctness Validation
- **Quality verification**: All implementations produce identical English transcription
- **Performance validation**: Comprehensive benchmarking with detailed analysis
- **Environment validation**: Automated checks for dependencies and GPU setup

### Impact Statement
**Problem**: Integrating new acceleration platforms like MAX Graph into existing ML workflows typically requires complete rewrites.

**Solution**: We developed a hybrid architecture proving MAX Graph can integrate with PyTorch ecosystems, achieving 2.4x speedup while maintaining perfect quality.

**Broader Impact**: This approach serves as a template for accelerating other transformer models, showing you don't need to rebuild everything to get MAX Graph benefits.

### MAX Graph Integration Experience
**What made it easy:**
- Tensor interoperability between PyTorch and MAX Graph
- Familiar GPU acceleration patterns
- Flexible integration allowing targeted layer replacement

**Roadblocks and solutions:**
- *Challenge*: Weight conversion overhead → *Solution*: Streamlined pipeline eliminating conversions
- *Challenge*: Quality preservation → *Solution*: Careful validation against reference implementation

**Remaining work:**
- Expand MAX Graph coverage to more Whisper components
- Scale optimization to larger models
- Package as production service

## 📚 Additional Documentation

- **[docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md)** - Implementation architecture and code details

---

**Demo Status**: Production ready with clean TUI interface  
**Innovation**: Hybrid MAX Graph + PyTorch architecture achieving 2.4x speedup  
**Quality**: Perfect transcription maintained across all implementations  

*Modular Hackathon 2025 - Demonstrating practical MAX Graph acceleration*