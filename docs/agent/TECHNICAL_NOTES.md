# Technical Notes

## Architecture Overview

### Pipeline Structure
```
Audio → Mel Spectrogram → MAX Graph Encoder → PyTorch Decoder → Text
```

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

### Current Issue
**Repetitive Token Generation**: Decoder consistently produces repetitive tokens (e.g., `<|ml|>`) regardless of input audio content.

### Hypotheses
1. **Numerical Precision**: Slight differences in floating-point operations may accumulate
2. **Operation Fidelity**: MAX Graph operations might not perfectly match OpenAI implementation
3. **Weight Loading**: Potential issues in weight format conversion or loading
4. **Feature Scale**: Encoder features may have different numerical ranges

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
- CPU: ~10.6s total execution
- GPU: ~1.9s total execution  
- MAX Graph: ~123ms encoder only (decoder produces incorrect output)

## Debugging Tools
- Feature extraction and comparison utilities
- Tensor statistics analysis
- Device compatibility checking
- Cross-framework validation methods