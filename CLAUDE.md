# CLAUDE.md - AI Agent Instructions

## Current Status & Priority

**Project**: MAX-Whisper GPU-accelerated speech recognition  
**Status**: Basic encoder working (72,290x speedup) - need full model for fair comparison  
**Priority**: Build complete Whisper with decoder for actual transcription in next 24 hours

### Completed ✅
- MAX Graph encoder on GPU (but oversimplified)
- Benchmarking infrastructure
- CUDA setup on RTX 4090

### Critical TODOs 🔴
1. Real transformer architecture (not single linear layer)
2. Decoder with cross-attention for text generation
3. Load actual Whisper weights (at minimum embeddings)
4. Tokenizer for text output
5. Fair benchmark comparison

## Implementation Plan (~24 hours)

### Phase 1: Real Encoder (6 hours)
File: `src/model/max_whisper_real.py`
```python
# Build proper encoder with:
- Conv1d input layers (like Whisper)
- 4 transformer layers (simplified from 12)
- Multi-head attention (6 heads, 384 dim)
- Layer normalization
- Positional encoding
- Load weights from whisper-tiny encoder
```

### Phase 2: Decoder (6 hours)  
Add to same file:
```python
# Build decoder with:
- Token embeddings (51865 vocab)
- 4 transformer layers
- Masked self-attention
- Cross-attention to encoder
- Output projection to vocabulary
- Load embedding weights at minimum
```

### Phase 3: End-to-End (4 hours)
```python
# Connect everything:
- Tokenizer from OpenAI
- Greedy decoding (no beam search)
- Audio → Mel → Encoder → Decoder → Tokens → Text
```

### Phase 4: Benchmarking (2 hours)
- Fair comparison: full model vs full model
- Measure actual transcription time
- Document what we built honestly

## Technical Specifications

### Model Architecture (Whisper-tiny simplified)
```python
# Encoder
n_mels = 80
n_audio_ctx = 1500
n_audio_state = 384
n_audio_head = 6
n_audio_layer = 4  # Reduced from 12

# Decoder  
n_text_ctx = 448
n_text_state = 384
n_text_head = 6
n_text_layer = 4   # Reduced from 12
n_vocab = 51865
```

### MAX Graph Key APIs
```python
# Essential operations
ops.matmul()      # Matrix multiplication
ops.layer_norm()  # Layer normalization  
ops.softmax()     # Attention weights
ops.gelu()        # Activation function
ops.gather()      # Embedding lookup
ops.permute()     # Tensor reshaping
ops.constant()    # Load weights
```

### Weight Loading Priority
1. Token embeddings (most important)
2. Output projection
3. Attention weights
4. Layer norm parameters

## Commands

```bash
# CUDA setup
source setup_cuda_env.sh

# Development
pixi run -e default python src/model/max_whisper_real.py

# Benchmarking
pixi run -e benchmark python benchmarks/fair_comparison.py
```

## Critical Success Factors

1. **Working transcription** - Must output actual text
2. **Real weights** - At least embeddings from OpenAI
3. **Fair comparison** - Full model to full model
4. **Honest documentation** - State what we built

## Avoid These Mistakes
- Don't compare encoder-only to full model
- Don't use random weights in final demo
- Don't claim 72,000x on transcription (that's encoder only)
- Don't implement beam search (greedy is fine)

## Repository Structure
```
src/model/
├── max_whisper_real.py    # BUILD THIS - Full model
├── max_whisper_simple.py  # Current encoder-only
└── whisper_weights.py     # UPDATE - Load real weights

benchmarks/
└── fair_comparison.py     # CREATE - Honest benchmarks
```

## If Time Runs Out

Minimum viable submission:
1. Encoder + decoder that produces *some* text
2. Loaded embedding weights (even if other weights are random)
3. One successful transcription example
4. Documentation explaining what we built

Remember: **A working transcription with lower speedup is better than encoder-only with 72,000x**