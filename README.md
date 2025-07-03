# max-whisper: Speech Recognition with MAX Graph

A technical exploration of integrating MAX Graph operations into OpenAI's Whisper speech recognition model. This project demonstrates real MAX Graph compilation, execution, and cross-framework integration with detailed analysis of implementation approaches.

## Project Status

| Implementation | Status | Performance | Quality | Technical Details |
|---------------|--------|-------------|---------|------------------|
| **CPU Baseline** | ✅ Working | ~10.6s | Perfect transcription | OpenAI Whisper reference |
| **GPU Accelerated** | ✅ Working | ~1.9s (5.7x faster) | Perfect transcription | CUDA acceleration |
| **MAX Graph** | ✅ Working Hybrid | ~1.0s (3.4x faster) | Partial transcription (41.2%) | 838 chars meaningful content |

**Current Status:** **HYBRID IMPLEMENTATION WORKING** - MAX Graph encoder (99.99% similarity) + PyTorch decoder produces 838 characters (41.2% of baseline) with 17x encoder speedup. Limitation: decoder confidence loss causes consistent stopping regardless of parameter tuning. Next: Full MAX Graph decoder implementation.

## Quick Start

```bash
git clone https://github.com/nijaru/max-whisper
cd max-whisper
make install                           # Setup pixi environment
pixi run -e benchmark demo             # Compare all implementations (enhanced UI)
pixi run -e benchmark test-cpu         # Test CPU baseline only
pixi run -e benchmark test-gpu         # Test GPU version only
pixi run -e benchmark test-max         # Test MAX Graph version only
```

### New Enhanced Commands
```bash
pixi run -e benchmark benchmark        # Structured benchmark with error handling
pixi run -e benchmark benchmark-json   # JSON output for parsing/analysis
pixi run -e benchmark debug-encoder    # Debug encoder feature extraction
pixi run -e benchmark compare-simple   # Compare transcription outputs
pixi run test                          # Run comprehensive test suite
```

## MAX Graph Implementation Details

### ✅ **Technical Integration Achievements**

**Environment & Compilation:**
- MAX Graph imports, device setup (GPU/CPU), and graph compilation function correctly
- Environment dependencies resolve without issues

**Weight Extraction:**
- Successfully extracts all 67 pretrained weights from Whisper tiny model (including critical ln_post)
- Maintains correct tensor shapes and data integrity  
- Integrates cleanly with MAX Graph tensor format

**MAX Graph Operations:**
```python
# These operations execute successfully:
ops.matmul(a, b)           # Matrix multiplication 
ops.transpose(x, 0, 1)     # Tensor transpose
ops.layer_norm(x, ...)     # Layer normalization
ops.gelu(x)                # GELU activation
ops.slice_tensor(x, [...]) # Tensor slicing
```

**Cross-Framework Pipeline:**
- MAX Graph encoder → PyTorch decoder integration works without errors
- Proper tensor conversions and device management
- Encoder executes in ~123ms on GPU

### 🔧 **Recent Major Progress**

**Major Breakthrough - Feature Distribution Fix:**
- **Root Cause Identified**: Mel preprocessing differences (librosa vs whisper.log_mel_spectrogram)
- **Impact**: Achieved 99.99% cosine similarity between MAX Graph and OpenAI encoder features
- **Output Quality**: Meaningful 838-character transcription (41.2% of 2035-char baseline)
- **Performance**: 47ms encoder execution (23x faster than CPU encoder alone)

**Current Architecture:**
```
Audio → MAX Graph Encoder (47ms, 99.99% similarity) → PyTorch Decoder → Text
```

### ⚠️ **Current Limitation**

**Decoder Confidence Loss:**
- Hybrid implementation produces consistent 838 characters regardless of parameter extremes
- Despite 99.99% cosine similarity, subtle feature distribution differences cause early stopping
- Tested: patience=1000.0, beam_size=50, feature scaling - identical results
- Root cause: Decoder trained on exact OpenAI feature distributions

**Next Steps:**
Implement full MAX Graph decoder to bypass PyTorch decoder limitations (2-3 weeks estimated).

## Key Technical Insights

**What This Project Demonstrates:**
- MAX Graph can successfully integrate with complex AI architectures (4-layer transformer with attention)
- Cross-framework integration (MAX Graph → PyTorch) is technically feasible
- Weight extraction and integration from pretrained models functions correctly
- Systematic debugging can identify and fix critical numerical issues
- Encoder performance is excellent (~0.2s, encoder portion only)

**Key Learning:**
This project demonstrates the importance of **complete architectural fidelity** when implementing neural network acceleration. Missing even a single layer normalization can cause major semantic quality degradation, while systematic debugging and feature analysis can identify and resolve such issues.

**Development Approach:**
Successfully implementing AI acceleration requires both technical integration (✅ encoder achieved) and complete sequence generation (🔧 decoder integration incomplete).

## Implementation Details

**Architecture:**
```
Audio → Mel Spectrogram → MAX Graph Encoder → PyTorch Decoder → Text
                           ↓ (99.99% similarity ✅)  ↓ (41.2% coverage ⚠️)
                    47ms encoder execution      838 chars meaningful
```

**File Structure:**
```
max-whisper/
├── whisper_cpu.py      # ✅ CPU baseline (perfect transcription)
├── whisper_gpu.py      # ✅ GPU accelerated (perfect transcription)
└── whisper_max.py      # 🔧 MAX Graph integration (encoder works, decoder incomplete)
```

**Performance Results:**
- CPU Baseline: ~3.4s (perfect quality, 2035 chars transcription)
- GPU Accelerated: ~1.9s (perfect quality, 2035 chars transcription)  
- MAX Graph Hybrid: ~1.0s (meaningful quality, 838 chars transcription, 41.2% coverage)

**Technical Foundation:**
This project demonstrates successful MAX Graph integration with complex AI architectures. The hybrid implementation achieves 99.99% encoder feature similarity and meaningful transcription with significant performance gains. Current limitation is decoder confidence loss requiring full MAX Graph decoder implementation.

## Development & Testing

**Environment Setup:**
```bash
make install                    # Install pixi + dependencies  
pixi run graph-test            # Verify MAX Graph environment
make gpu-check                 # Check CUDA compatibility (legacy)
```

**Testing & Benchmarking:**
```bash
pixi run -e benchmark demo               # Interactive comparison (enhanced UI)
pixi run -e benchmark benchmark          # Structured benchmarking
pixi run -e benchmark benchmark-json     # JSON output for analysis
pixi run test                           # Run comprehensive test suite
pixi run -e benchmark test-cpu          # Test CPU implementation
pixi run -e benchmark test-gpu          # Test GPU implementation  
pixi run -e benchmark test-max          # Test MAX Graph implementation
```

**New Capabilities:**
- **Structured logging**: JSON output with performance metrics
- **Error handling**: Robust error recovery and detailed reporting  
- **Enhanced testing**: Comprehensive unit and integration tests
- **Performance tracking**: Detailed timing and memory usage analysis

**Documentation:**
- `/docs/` - Detailed technical documentation
- `COMPLETE_RESULTS.md` - Generated performance reports
- `CLAUDE.md` - Development context and instructions

## Repository Structure

```
├── max-whisper/
│   ├── whisper_cpu.py      # ✅ Reference implementation
│   ├── whisper_gpu.py      # ✅ CUDA acceleration
│   └── whisper_max.py      # ✅ MAX Graph encoder integration
├── benchmarks/             # Performance testing
├── examples/               # Demo scripts
├── docs/                   # Technical documentation
└── audio_samples/          # Test audio files
```

A technical exploration of AI acceleration challenges and cross-framework integration using MAX Graph.

*Originally developed during the Modular Hack Weekend June 2025*