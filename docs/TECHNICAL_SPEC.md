# MAX-Whisper Technical Specification

## Architecture Overview

MAX-Whisper is a GPU-accelerated implementation of OpenAI's Whisper model using Modular's MAX Graph API. The project demonstrates how to build transformer models with MAX Graph for extreme performance on NVIDIA GPUs.

## Current Implementation

### Completed Components

#### 1. Simplified Encoder (`max_whisper_simple.py`)
- Single linear transformation layer (not full transformer)
- Fixed input size: 80 mel bins × 1500 time steps
- GPU execution via MAX Graph
- Achieves 72,290x real-time on RTX 4090

**Note**: This is NOT a fair comparison as it's encoder-only with simplified architecture.

#### 2. Infrastructure
- Benchmarking suite
- CUDA library configuration
- Audio preprocessing pipeline
- Weight loading utilities (not yet integrated)

### Architecture Details

```python
# Current simplified encoder
Input (1, 80, 1500) → Linear(80, 384) → ReLU → Output (1, 384, 1500)

# Target architecture (Whisper-tiny)
Input → Conv1d → Conv1d → Positional → [Transformer×4] → Output
                                ↓
                        Attention + FFN + LayerNorm
```

## Planned Full Implementation

### Phase 1: Real Encoder
```python
class WhisperEncoder:
    - conv1: Conv1d(80, 384, kernel_size=3)
    - conv2: Conv1d(384, 384, kernel_size=3, stride=2)
    - positional_embedding: (750, 384)
    - blocks: [TransformerBlock × 4]
    - ln_post: LayerNorm(384)
```

### Phase 2: Decoder
```python
class WhisperDecoder:
    - token_embedding: Embedding(51865, 384)
    - positional_embedding: (448, 384)
    - blocks: [TransformerDecoderBlock × 4]
    - ln: LayerNorm(384)
    - output: Linear(384, 51865)
```

### Phase 3: Complete Pipeline
```
Audio → Mel-Spectrogram → Encoder → Cross-Attention → Decoder → Tokens → Text
   ↓           ↓              ↓            ↓              ↓         ↓        ↓
16kHz    (80,1500)      (384,750)    Encoder KV     (384,448)  Softmax  Decode
```

## Performance Analysis

### Current Results (Encoder Only)
| Component | Time | Notes |
|-----------|------|-------|
| Preprocessing | ~2ms | CPU-based mel-spectrogram |
| Encoder | 0.41ms | Simplified linear layer |
| Decoder | N/A | Not implemented |
| Total | ~2.5ms | Not comparable to full model |

### Expected Full Model Performance
Based on architecture complexity:
- Encoder: ~5-10ms (with 4 transformer layers)
- Decoder: ~10-20ms (with cross-attention)
- Total: ~20-30ms for 30s audio
- Expected speedup: 30-50x (realistic for full model)

## MAX Graph Implementation Details

### Key Operations Used
```python
# Tensor operations
ops.matmul()      # Matrix multiplication
ops.permute()     # Tensor dimension reordering
ops.reshape()     # Tensor reshaping
ops.constant()    # Weight loading

# Neural network layers
ops.layer_norm()  # Layer normalization
ops.softmax()     # Attention softmax
ops.gelu()        # GELU activation
ops.gather()      # Embedding lookup

# Elementwise operations
elementwise.add() # Residual connections
elementwise.mul() # Scaling
```

### Device Management
```python
# GPU execution
device = DeviceRef.GPU()
session = engine.InferenceSession(devices=[Accelerator(id=0)])

# Tensor transfer
tensor = Tensor.from_numpy(data).to(gpu_device)
```

### Graph Building Pattern
```python
with Graph("model_name", input_types=(input_type,)) as graph:
    x = graph.inputs[0]
    # Build operations
    graph.output(result)
```

## Comparison with Baselines

### OpenAI Whisper (PyTorch)
- Dynamic graph execution
- Automatic differentiation
- ~51ms for 30s audio on GPU

### Faster-Whisper (CTranslate2)
- Optimized C++ backend
- INT8 quantization
- Custom CUDA kernels

### MAX-Whisper (Our Goal)
- Compiled static graph
- Graph-level optimizations
- Zero-copy operations
- Target: 30-50x speedup on full model

## Technical Challenges

1. **Graph Construction**: MAX Graph requires explicit graph building vs PyTorch's dynamic graphs
2. **Weight Loading**: Converting PyTorch weights to MAX Graph constants
3. **Attention Mechanism**: Implementing efficient multi-head attention
4. **Autoregressive Decoding**: Managing state between decoder steps

## Hackathon Deliverables

### Minimum Viable Product
1. 4-layer encoder (simplified from 12)
2. 4-layer decoder with cross-attention  
3. Loaded embedding weights
4. Basic transcription capability
5. Fair performance comparison

### Stretch Goals
1. Full 12-layer architecture
2. Beam search decoding
3. Batch processing
4. INT8 quantization
5. Streaming support