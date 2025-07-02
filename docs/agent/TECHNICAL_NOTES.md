# Technical Notes

## Architecture Overview

### Pipeline Structure
```
Audio → Mel Spectrogram → MAX Graph Encoder → PyTorch Decoder → Text
                           ↓ (bias fixed ✅)    ↓ (integration works)
                    Complete with ln_post   Scale optimization needed
```

**MAJOR UPDATES**: 
- ✅ Fixed critical missing final layer normalization (ln_post) - bias reduced 0.692 → 0.002 (99% improvement)
- ✅ Replaced inefficient convolution approximation with proper Conv2D operations - significant performance and quality gains

### Implementation Files
- `max-whisper/whisper_cpu.py` - Reference OpenAI Whisper implementation
- `max-whisper/whisper_gpu.py` - CUDA-accelerated version  
- `max-whisper/whisper_max.py` - MAX Graph encoder hybrid

## MAX Graph Integration Details

### Device Setup Pattern
```python
from max.driver import accelerator_count, Accelerator, CPU, DeviceRef, InferenceSession

if accelerator_count() > 0:
    driver_device = Accelerator()
    device = DeviceRef.GPU()
else:
    driver_device = CPU()
    device = DeviceRef.CPU()

session = InferenceSession(devices=[driver_device])
```

### Working Operations
```python
from max.graph import ops

# Verified working operations:
ops.matmul(a, b)           # Matrix multiplication
ops.transpose(x, 0, 1)     # Tensor transpose
ops.layer_norm(x, w, b)    # Layer normalization  
ops.gelu(x)                # GELU activation
ops.slice_tensor(x, [...]) # Tensor slicing
```

### Weight Extraction
Successfully extracts 65 weights from Whisper tiny model:
- Convolutional layers: conv1_weight (384,80,3), conv2_weight (384,384,3)
- Transformer layers: 4 layers × 16 weights each (attention + MLP)
- Position embeddings and layer norms

## Investigation Findings

### What Works
1. **Compilation**: Graph compiles without errors using proper input types
2. **Execution**: Encoder processes in ~123ms on GPU with no exceptions
3. **Integration**: Tensor conversions between MAX Graph and PyTorch work correctly
4. **Performance**: Encoder execution is significantly faster than CPU baseline

### Current Status  
**Decoder Integration Fixed**: Corrected DecodingOptions with proper beam search parameters (beam_size=5, temperature=0.0, sample_len=448).

**Critical Issue Identified**: Conv2D-based Conv1D implementation produces semantically corrupted features despite matching statistics. Cosine similarity: -0.038 indicates structural rather than scale issues.

**Technical Finding**: Native Conv1DV1 unavailable in current MAX Graph version, forcing Conv2D→Conv1D conversion that corrupts feature relationships.

### Root Cause Analysis
1. **Convolution Issue**: Conv2D→Conv1D weight format or tensor layout corrupts semantic content
2. **Statistics vs Semantics**: Features have correct variance (std: 1.708 vs 1.448) but wrong relationships  
3. **Decoder Integration**: Fixed - proper beam search and sequence generation parameters
4. **Performance**: Encoder execution excellent (~0.28s, 13x speedup)

## Key Reference Materials
- `external/modular/examples/pytorch_custom_ops/whisper.py` - Modular's attention example
- `external/modular/max/graph/ops/` - MAX Graph operation implementations
- OpenAI Whisper encoder architecture for comparison

## Development Environment
- **Setup**: `make install` (pixi environment)
- **Testing**: `pixi run -e benchmark demo` (enhanced comparison UI)
- **Benchmarking**: `pixi run -e benchmark benchmark-json` (structured output)
- **Unit Tests**: `pixi run test` (comprehensive test suite)
- **Environment**: Use `pixi run -e benchmark` prefix for all commands
- **GPU Requirements**: CUDA-compatible hardware for full functionality

## Performance Baselines
- CPU: ~3.6s total execution (2035 chars)
- GPU: ~1.0s total execution (2035 chars, 3.6x speedup)
- MAX Graph: ~0.28s encoder execution (13x speedup), semantic corruption issue

## Debugging Tools
- **`benchmarks/encoder_feature_debug.py`**: Systematic encoder feature comparison
- **Feature Statistics**: Mean, std, cosine similarity, L2 norm analysis  
- **Cross-framework Validation**: MAX Graph vs OpenAI Whisper feature extraction
- **Tensor Analysis**: First-value comparison, difference patterns