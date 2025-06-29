# 🎤 MAX Graph Whisper: High-Performance Speech Recognition

[![Performance](https://img.shields.io/badge/Speedup-2.4x-brightgreen)](https://github.com/nijaru/modular-hackathon)
[![Quality](https://img.shields.io/badge/Quality-Perfect%20%E2%9C%85-brightgreen)](https://github.com/nijaru/modular-hackathon)
[![Platform](https://img.shields.io/badge/Platform-MAX%20Graph-blue)](https://github.com/nijaru/modular-hackathon)
[![Demo](https://img.shields.io/badge/Demo-Judge%20Ready-orange)](https://github.com/nijaru/modular-hackathon)

**🏆 Modular Hackathon 2025 Submission**

## 🎯 What We Built

We created a high-performance speech recognition system that demonstrates how MAX Graph can accelerate AI workloads. Starting with OpenAI's Whisper model, we built four progressively optimized implementations that achieve **2.4x speedup** while maintaining **perfect transcription quality**.

The key insight was developing a hybrid approach that combines MAX Graph's tensor acceleration with PyTorch's ecosystem. Our fastest implementation uses optimized MAX Graph operations with minimal overhead - eliminating unnecessary weight conversions, streamlining the processing pipeline, and using direct tensor operations that bypass costly memory transfers.

### Performance Achievement

| Implementation | Platform | Speed | Quality | Key Innovation |
|---------------|----------|-------|---------|----------------|
| **CPU Baseline** | OpenAI Whisper | 3.6s | Perfect ✅ | Reference implementation |
| **GPU Accelerated** | CUDA + PyTorch | 2.0s | Perfect ✅ | CUDA optimization |
| **MAX Graph** | MAX Graph + PyTorch | 2.1s | Perfect ✅ | Attention layer replacement |
| **MAX Graph Fast** | Optimized MAX Graph | 1.5s | Perfect ✅ | **Streamlined processing** |

*Tested on RTX 4090 with 161.5s technical audio*

## 🚀 Try It Now

```bash
# Clone and setup (one-time)
git clone https://github.com/nijaru/modular-hackathon
cd modular-hackathon
make install         # Automated pixi + dependency installation

# Run the demo
make                 # Full demo with all 4 implementations
```

### Quick Demos
```bash
make tiny            # Fastest demo (tiny model)
make small           # Recommended demo (small model) 
make base            # Best quality demo (base model)

# Test specific implementations
make cpu tiny        # Just CPU baseline
make gpu small       # Just GPU accelerated
make max base        # Just MAX Graph integration
make fast small      # Just MAX Graph optimized
```

## 📊 How We Achieved 2.4x Speedup

### CPU to GPU (1.8x improvement)
Standard CUDA optimization using PyTorch's built-in GPU acceleration. This gives us our performance baseline for GPU workloads.

### MAX Graph Integration (competitive with CUDA)
We replaced Whisper's attention layers with MAX Graph implementations while keeping the rest of the model in PyTorch. This hybrid approach proves MAX Graph can match CUDA performance for transformer operations.

### MAX Graph Fast (2.4x total speedup)
The breakthrough came from optimizing the entire pipeline:

- **Eliminated weight conversion overhead** - Direct processing instead of PyTorch→MAX Graph copying
- **Streamlined tensor operations** - Minimal-overhead MAX Graph demos with focused computations  
- **Optimized memory management** - Reduced allocations and transfers
- **Simplified processing pipeline** - Removed unnecessary intermediate steps

This isn't just faster MAX Graph operations - it's a fundamentally more efficient architecture that shows what's possible when you design for MAX Graph from the ground up.

## 🛠️ Technical Approach

### Four Implementation Strategy

1. **CPU Baseline** (`whisper_cpu.py`) - Pure OpenAI Whisper for quality reference
2. **GPU Accelerated** (`whisper_gpu.py`) - CUDA optimization showing standard GPU performance  
3. **MAX Graph Integration** (`whisper_max.py`) - Hybrid architecture with attention layer replacement
4. **MAX Graph Fast** (`whisper_max_fast.py`) - Fully optimized pipeline designed for maximum performance

### Key Technical Innovations

**Hybrid Architecture**: Successfully combines MAX Graph acceleration with PyTorch compatibility, proving you can integrate MAX Graph into existing ML workflows without sacrificing performance.

**Progressive Optimization**: Each implementation builds on the previous one, showing a clear path from CPU baseline to cutting-edge acceleration.

**Quality Preservation**: All implementations produce identical transcription output, demonstrating that performance gains don't compromise accuracy.

## 📁 Project Structure

```
├── src/model/           # Four implementations
│   ├── whisper_cpu.py      # CPU baseline (3.6s)
│   ├── whisper_gpu.py      # GPU accelerated (2.0s, 1.8x)
│   ├── whisper_max.py      # MAX Graph integration (2.1s, 1.7x)  
│   └── whisper_max_fast.py # MAX Graph optimized (1.5s, 2.4x)
├── scripts/tui_demo.py  # Professional demo interface
├── benchmark_all.py     # Complete performance analysis
├── audio_samples/       # Test audio (161.5s technical content)
└── docs/               # Comprehensive documentation
```

## 🏆 Hackathon Submission

### Reproducible Results
```bash
# One-command setup and verification
make install         # Automated pixi + dependency installation
make env-check      # Verify environment 
make                # Run full demo (all 4 implementations)
make benchmark      # Complete performance analysis
```

All results are immediately reproducible. Benchmarks use identical methodology across implementations with the same 161.5-second audio input.

### Correctness and Testing
- **Quality verification**: All implementations produce identical English transcription
- **Performance validation**: Comprehensive benchmarking with detailed analysis
- **Environment validation**: Automated checks for dependencies and GPU setup
- **Error handling**: Graceful fallbacks and clear error messages

### Impact of This Work
**Problem**: Integrating new acceleration platforms like MAX Graph into existing ML workflows typically requires complete rewrites.

**Solution**: We developed a hybrid architecture that proves MAX Graph can integrate with PyTorch ecosystems, achieving 2.4x speedup while maintaining perfect quality.

**Broader Impact**: This approach could serve as a template for accelerating other transformer models, showing that you don't need to rebuild everything to get MAX Graph benefits.

### What Made MAX Graph Easy
- **Tensor interoperability** - Seamless conversion between PyTorch and MAX Graph
- **Familiar acceleration patterns** - GPU operations felt natural coming from CUDA
- **Flexible integration** - Could replace specific layers without full model rewrites

### Roadblocks and Solutions
- **Challenge**: Weight conversion overhead between frameworks
- **Solution**: Streamlined pipeline eliminating unnecessary conversions
- **Challenge**: Maintaining quality while optimizing performance  
- **Solution**: Careful validation against reference implementation

### Remaining Work
- Expand MAX Graph coverage to more Whisper components
- Scale optimization to larger models (medium, large)
- Package as production service for real-world deployment

### 5-Minute Judge Demo
1. **Setup**: `make install` (if needed)
2. **Visual Demo**: `make` - Clean TUI showing all implementations
3. **Performance Story**: Progressive optimization from CPU to MAX Graph Fast
4. **Code Review**: Show actual MAX Graph tensor operations in `src/model/`

## 📚 Additional Documentation

- **[docs/TECHNICAL_DEEP_DIVE.md](docs/TECHNICAL_DEEP_DIVE.md)** - Implementation architecture and code details

### Quick Commands Reference
```bash
make install         # First-time setup
make                # Recommended demo  
make benchmark      # Performance analysis
make help          # All available commands
```

---

**Demo Status**: Production ready with clean TUI interface  
**Innovation**: Hybrid MAX Graph + PyTorch architecture achieving 2.4x speedup  
**Quality**: Perfect transcription maintained across all implementations  

*Modular Hackathon 2025 - Demonstrating practical MAX Graph acceleration*