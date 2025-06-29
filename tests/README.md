# Tests Directory

Comprehensive testing suite for MAX-Whisper components.

## Test Scripts

### Component Tests
- `test_everything.py` - **PRIMARY**: All 4 MAX-Whisper components (4/4 passing)
- `test_max_gpu.py` - GPU-specific functionality tests
- `test_max_gpu_simple.py` - Simple GPU validation

### Baseline Tests
- `test_baselines_only.py` - OpenAI Whisper & Faster-Whisper validation
- `test_all_implementations.py` - Cross-implementation compatibility

## Usage

### Primary Test Suite
```bash
# Test all MAX-Whisper components (MAIN TEST)
pixi run -e default python tests/test_everything.py

# Expected output:
# ============================================================
# COMPREHENSIVE MAX-WHISPER TESTING - ALL PASS
# ============================================================
# Simple Encoder       ✅ PASS (0.25ms inference)
# Multi-Head Attention ✅ PASS (0.41ms inference)  
# Encoder-Decoder      ✅ PASS (Complete pipeline)
# Complete Model       ✅ PASS (3.6x real-time speedup)
# Total: 4/4 tests passed
```

### Baseline Validation
```bash
# Validate baseline models for comparison
pixi run -e benchmark python tests/test_baselines_only.py
```

### GPU-Specific Tests
```bash
# Test GPU functionality
pixi run -e default python tests/test_max_gpu.py

# Simple GPU validation
pixi run -e default python tests/test_max_gpu_simple.py
```

## Test Results Status

### ✅ Current Status: ALL PASSING
- **MAX-Whisper Components**: 4/4 tests passing
- **GPU Acceleration**: Fully working with CUDA
- **Baseline Models**: Validated and working
- **Performance**: 3.6x real-time speedup achieved

### Test Coverage
- Simple encoder (0.25ms inference)
- Multi-head attention (0.41ms, 6 heads)
- Encoder-decoder with cross-attention
- Complete end-to-end model (3.6x speedup)
- GPU memory management
- CUDA library integration