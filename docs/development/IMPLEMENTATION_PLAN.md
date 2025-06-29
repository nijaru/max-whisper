# Implementation Plan - Final 24 Hours

## Goal
Build a working Whisper model that produces actual transcriptions using MAX Graph.

## Current State
- ✅ Basic encoder working on GPU
- ❌ No decoder
- ❌ No real weights
- ❌ No text output

## Implementation Steps

### Step 1: Real Encoder (6 hours)
**File**: `src/model/max_whisper_real.py`

```python
class RealWhisperEncoder:
    def __init__(self):
        # Build these layers:
        self.conv1 = Conv1d(80, 384, kernel_size=3)
        self.conv2 = Conv1d(384, 384, kernel_size=3, stride=2)
        self.blocks = [TransformerBlock() for _ in range(4)]
        
    def build_graph(self):
        # MAX Graph implementation
        # Load weights from whisper-tiny
```

Key tasks:
- [ ] Implement multi-head attention
- [ ] Add layer normalization  
- [ ] Load conv weights
- [ ] Test encoder output

### Step 2: Decoder (6 hours)
**Add to same file**

```python
class WhisperDecoder:
    def __init__(self):
        # Build these layers:
        self.token_embedding = Embedding(51865, 384)
        self.blocks = [DecoderBlock() for _ in range(4)]
        self.ln_final = LayerNorm(384)
        self.lm_head = Linear(384, 51865)
```

Key tasks:
- [ ] Implement masked attention
- [ ] Add cross-attention
- [ ] Load embedding weights (critical!)
- [ ] Test token generation

### Step 3: Integration (4 hours)

```python
def transcribe(audio):
    # 1. Preprocess
    mel = compute_mel_spectrogram(audio)
    
    # 2. Encode
    encoder_output = encoder.encode(mel)
    
    # 3. Decode
    tokens = [SOT_TOKEN]
    for _ in range(max_length):
        logits = decoder.decode(tokens, encoder_output)
        next_token = argmax(logits[-1])
        tokens.append(next_token)
        if next_token == EOT_TOKEN:
            break
    
    # 4. Detokenize
    text = tokenizer.decode(tokens)
    return text
```

### Step 4: Testing & Benchmarking (2 hours)

Create `benchmarks/fair_comparison.py`:
- Test on real audio files
- Compare full transcription time
- Document what we built

## Code Templates

### Multi-Head Attention
```python
def build_attention(q, k, v, n_heads):
    # Split heads
    q = reshape(q, (batch, n_heads, dim//n_heads, seq))
    k = reshape(k, (batch, n_heads, dim//n_heads, seq))
    v = reshape(v, (batch, n_heads, dim//n_heads, seq))
    
    # Attention
    scores = matmul(q, transpose(k)) / sqrt(dim//n_heads)
    weights = softmax(scores)
    out = matmul(weights, v)
    
    # Merge heads
    return reshape(out, (batch, dim, seq))
```

### Weight Loading
```python
# Priority order:
1. embeddings = np.load("whisper_tiny_embeddings.npy")
2. lm_head = np.load("whisper_tiny_lm_head.npy")  
3. attention_weights = np.load("whisper_tiny_attn.npy")

# Convert to MAX Graph
embedding_const = ops.constant(embeddings, device=device)
```

## Success Criteria

### Minimum (Must Have)
- [ ] Produces some text output
- [ ] Uses real embedding weights
- [ ] Runs on GPU

### Target (Should Have)
- [ ] Coherent transcriptions
- [ ] 20x+ speedup
- [ ] 4-layer architecture

### Stretch (Nice to Have)
- [ ] Accurate transcriptions
- [ ] All weights loaded
- [ ] Batch processing

## If Behind Schedule

**Hour 18 checkpoint**: If encoder not done
- Skip to decoder with dummy encoder output

**Hour 20 checkpoint**: If decoder not done  
- Use only 1-2 layers
- Focus on embeddings

**Hour 22 checkpoint**: If integration not done
- Hard-code a demo transcription
- Document what's built

## Remember
**A model that outputs "hello world" is better than one that outputs nothing.**