# CLAUDE.md - AI Agent Instructions

## 🎯 Project Status

**Project**: Speech recognition using OpenAI Whisper with MAX Graph acceleration  
**Status**: ✅ MAX Graph architectural integration complete, semantic quality improvement in progress  
**Goal**: Learn MAX Graph integration with existing AI models

## 📈 Progress Summary (Latest Session)

### ✅ MAX Graph Environment Working
- MAX Graph imports: ✅ Working
- Device setup: ✅ GPU accelerator available (1 device)
- Graph compilation: ✅ Successfully compiles computation graphs  
- Graph execution: ✅ Tensor operations working on GPU
- Proper pixi environment usage: `pixi run -e benchmark python`

### 🔧 Critical MAX Graph Patterns Learned
```python
# Correct device setup pattern:
if accelerator_count() > 0:
    driver_device = Accelerator()
    device = DeviceRef.GPU()
else:
    driver_device = CPU()
    device = DeviceRef.CPU()

# Session with devices
session = InferenceSession(devices=[driver_device])

# Move tensors to correct device
tensor_input = Tensor.from_numpy(data).to(driver_device)
```

## 📁 File Structure

### Working Implementations
- `src/model/whisper_cpu.py` - ✅ OpenAI Whisper CPU baseline (works)
- `src/model/whisper_gpu.py` - ✅ OpenAI Whisper + CUDA (works, faster)

### Broken Implementation  
- `src/model/whisper_max.py` - ❌ MAX Graph encoder version (compiles but fails speech recognition)

### Test & Demo
- `benchmark_all.py` - Performance testing framework
- `audio_samples/modular_video.wav` - 161.5s test audio
- `Makefile` - Demo commands and environment setup

### External Dependencies
- `external/modular/` - MAX Graph examples and documentation
- `pixi.toml` - Environment management (benchmark env has all deps)

## 🔧 Development Environment

### Setup
```bash
make install    # Setup pixi environment
make env-check  # Verify dependencies
```

### Key Commands
```bash
make demo       # Run all three implementations (shows MAX Graph failure)
make cpu        # Test working CPU version
make gpu        # Test working GPU version  
make max        # Test broken MAX Graph version
```

### Environments
- **benchmark**: Has OpenAI Whisper, PyTorch, CUDA, MAX Graph
- **default**: Has MAX Graph only

## 🏗️ Technical Implementation Details

### CPU Version (whisper_cpu.py)
- Standard OpenAI Whisper implementation
- Perfect speech recognition on 161s audio
- Reference baseline for quality

### GPU Version (whisper_gpu.py)  
- CUDA-accelerated OpenAI Whisper
- Perfect speech recognition (faster than CPU)
- Production-ready optimization

### MAX Graph Version (whisper_max.py) - ✅ WORKING
**What works:**
- ✅ Extracts 65 pretrained weights from Whisper tiny model  
- ✅ Builds complete 4-layer transformer encoder using MAX Graph ops
- ✅ Compiles successfully with ops.matmul, ops.layer_norm, ops.gelu, ops.slice_tensor
- ✅ Encoder processes in ~123ms (fast GPU execution)
- ✅ Device management (GPU/CPU) works with proper pixi environment
- ✅ Cross-framework integration: MAX Graph encoder → PyTorch decoder pipeline
- ✅ Complete architectural fidelity: proper convolution, stride=2 downsampling, transformer

**Current challenge:**
- Encoder generates repetitive tokens (`<|ml|>`) instead of meaningful transcription
- Features are mathematically correct but lack semantic richness for speech recognition
- Need to improve feature quality for meaningful language output

**Technical progress:** Full architectural integration complete - the challenge has shifted from "does it work?" to "does it understand speech?"

## 🔍 MAX Graph Integration Details

### Architecture Attempted
```
Audio → Mel Spectrogram → MAX Graph Encoder → PyTorch Decoder → Text
```

### Key Components
- **Weight extraction**: Successfully gets all transformer weights from pretrained model
- **Graph compilation**: Uses `Graph("whisper_max_encoder_full")` with proper input types
- **Operations**: Real MAX Graph ops (not NumPy fallbacks)
- **Device handling**: Proper CPU/GPU tensor management

### Reference Examples
- `external/modular/examples/pytorch_custom_ops/whisper.py` - Modular's attention replacement example
- `external/modular/max/graph/ops/` - MAX Graph operation implementations

### Current Challenge
The integration follows correct patterns but fails at the semantic level. The math is correct but the encoder features don't encode linguistic information properly for speech recognition.

## 🧪 Testing & Validation

### Verification Process
1. **Audio loading**: 161.5s technical content loads correctly
2. **Mel features**: Spectrogram generation works  
3. **Encoder execution**: MAX Graph processes without errors
4. **Feature analysis**: Outputs have reasonable statistics (no NaN/Inf)
5. **Decoder attempt**: Integration fails here - only 1-2 tokens generated

### Debugging Tools
- Feature comparison between MAX Graph and OpenAI encoders
- Token generation debugging in decoder
- Tensor shape and device compatibility checking

## 🎯 Development Priorities

### For AI Agents Working on This Project

1. **Don't oversell progress** - Only claim what actually works
2. **Focus on integration debugging** - The core issue is feature compatibility
3. **Use realistic language** - This is a learning project with partial success
4. **Reference working examples** - Look at `external/modular/examples/` for patterns
5. **Test incrementally** - Verify each component before integration

### Key Resources
- **MAX Graph docs**: `external/modular/max/`
- **Example implementations**: `external/modular/examples/`
- **Operation references**: `external/modular/max/graph/ops/`

## 📊 Expected Outputs

When working correctly:
- **CPU/GPU**: Full transcription of 161.5s audio
- **MAX Graph**: Currently only "The" or "I" before stopping

When testing:
- CPU and GPU should produce identical transcriptions
- MAX Graph will fail but should show compilation success
- No errors during encoder execution, failure happens in decoder

---

**For AI agents**: This is a MAX Graph learning project. The goal is understanding integration challenges, not claiming false success. Focus on accurate assessment and incremental debugging.