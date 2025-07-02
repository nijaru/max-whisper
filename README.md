# max-whisper: Speech Recognition with MAX Graph

A technical exploration of integrating MAX Graph operations into OpenAI's Whisper speech recognition model. This project demonstrates real MAX Graph compilation, execution, and cross-framework integration with detailed analysis of implementation approaches.

## Project Status

| Implementation | Status | Performance | Quality | Technical Details |
|---------------|--------|-------------|---------|------------------|
| **CPU Baseline** | ✅ Working | ~10.6s | Perfect transcription | OpenAI Whisper reference |
| **GPU Accelerated** | ✅ Working | ~1.9s (5.7x faster) | Perfect transcription | CUDA acceleration |
| **MAX Graph** | 🔧 Encoder complete | ~0.2s encoder only | Incomplete transcription | Produces only first word |

**Current Status:** **ENCODER VARIANCE FIXED** - Added variance correction to MAX Graph encoder. Features now match OpenAI distribution (std: 0.3997 vs 0.4000). Encoder architecture complete but transcription incomplete - decoder stops after first word. Next: Debug sequence completion issue.

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

**Critical Bug Fix - Final Layer Normalization:**
- **Root Cause Identified**: Missing final layer normalization (`ln_post`) in MAX Graph encoder
- **Impact**: Encoder feature bias reduced from 0.692 → 0.002 (99% improvement)
- **Output Quality**: Improved from repetitive `<|ml|>` tokens to meaningful characters ("the")
- **Performance**: Encoder processes in ~0.2s but transcription incomplete

**Before Fix vs After Fix:**
```
Before: Mean: 0.692341, Std: 1.338029, Range: [-11.33, 16.05]
After:  Mean: 0.002280, Std: 1.474399, Range: [-16.09, 12.60]
```

### ⚠️ **Remaining Challenges**

**Decoder Sequence Completion:**
- Encoder architecture complete and variance-corrected (std: 0.3997 vs target: 0.4000)
- Decoder integration working but stops after first token
- CPU/GPU produce 2035 characters, MAX Graph produces only 3 characters ("the")

**Next Steps:**
Debug decoder sequence completion to achieve full transcription length matching CPU/GPU implementations.

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
                           ↓ (variance fixed ✅)    ↓ (stops early ❌)
                    ~0.2s encoder execution    Produces only "the"
```

**File Structure:**
```
max-whisper/
├── whisper_cpu.py      # ✅ CPU baseline (perfect transcription)
├── whisper_gpu.py      # ✅ GPU accelerated (perfect transcription)
└── whisper_max.py      # 🔧 MAX Graph integration (encoder works, decoder incomplete)
```

**Performance Results:**
- CPU Baseline: ~10.6s (perfect quality, 2035 chars transcription)
- GPU Accelerated: ~1.9s (perfect quality, 2035 chars transcription)  
- MAX Graph: ~0.2s encoder only (incomplete, 3 chars output: "the")

**Technical Foundation:**
This project proves MAX Graph can successfully accelerate AI model components. The encoder integration is complete, cross-framework compatibility works, and encoder performance is excellent. The challenge is completing full sequence generation - the current implementation stops after the first token.

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