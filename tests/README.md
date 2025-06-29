# MAX-Whisper Tests

Essential test suite for validating MAX-Whisper components and performance.

## 🎯 Primary Tests (Run These)

### Component Validation
- **[test_everything.py](test_everything.py)** - ⭐ **PRIMARY TEST** - All 4 MAX-Whisper components (4/4 passing)
- **[test_baselines_only.py](test_baselines_only.py)** - Baseline model validation for comparison

## Usage

### Quick Validation
```bash
# Setup environment
source scripts/setup_cuda_env.sh
export PATH="$HOME/.pixi/bin:$PATH"

# Run primary test suite (should show 4/4 passing)
pixi run -e default python tests/test_everything.py

# Validate baseline models for comparison  
pixi run -e benchmark python tests/test_baselines_only.py
```

### Expected Results
```
============================================================
COMPREHENSIVE MAX-WHISPER TESTING - ALL PASS
============================================================
Simple Encoder       ✅ PASS (0.25ms inference)
Multi-Head Attention ✅ PASS (0.41ms inference)  
Encoder-Decoder      ✅ PASS (Complete pipeline)
Complete Model       ✅ PASS (3.6x real-time speedup)

Total: 4/4 tests passed
🎉 ALL TESTS PASSING!
```

## 📁 Development Tests

### Archive
- **[archive/](archive/)** - Development and experimental tests
  - `test_all_implementations.py` - Cross-implementation compatibility
  - `test_max_gpu.py` - GPU-specific functionality tests
  - `test_max_gpu_simple.py` - Simple GPU validation

## ✅ Current Status

**All primary tests passing** with GPU acceleration on RTX 4090:
- ✅ Simple encoder: 0.25ms inference
- ✅ Multi-head attention: 0.41ms inference (6 heads, 384 dim)
- ✅ Encoder-decoder: Complete pipeline with cross-attention
- ✅ Complete model: 3.6x real-time speedup on end-to-end transcription

## Test Coverage

### Component Testing
- Simple encoder implementation
- Multi-head attention mechanism
- Encoder-decoder architecture with cross-attention
- Complete end-to-end model

### Performance Testing
- GPU acceleration validation
- Memory management
- CUDA library integration
- Real-time speedup measurement

### Integration Testing
- Baseline model comparison
- Production component validation
- End-to-end pipeline testing