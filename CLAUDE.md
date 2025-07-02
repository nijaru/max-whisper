# CLAUDE.md - AI Agent Instructions

## Project Overview
Speech recognition using OpenAI Whisper with MAX Graph acceleration. Three implementations demonstrate different approaches: CPU baseline, GPU accelerated, and MAX Graph hybrid.

*Originally developed during the Modular Hack Weekend June 2025*

## Implementation Status
| Implementation | File | Status | Performance | Quality |
|---------------|------|--------|-------------|---------|
| CPU Baseline | `max-whisper/whisper_cpu.py` | ✅ Working | ~3.6s | Perfect (2035 chars) |
| GPU Accelerated | `max-whisper/whisper_gpu.py` | ✅ Working | ~1.0s | Perfect (2035 chars) |
| MAX Graph Hybrid | `max-whisper/whisper_max.py` | 🔧 Conv issue | ~0.28s encoder | Semantic corruption |

## Quick Commands
```bash
make install    # Setup pixi environment
make demo       # Compare all implementations
make cpu        # Test CPU baseline  
make gpu        # Test GPU version
make max        # Test MAX Graph version
```

## Project Structure
```
max-whisper/           # Main implementations (was src/model/)
├── audio/            # Audio processing utilities
├── utils/            # Helper utilities  
├── whisper_cpu.py    # CPU baseline
├── whisper_gpu.py    # GPU accelerated
└── whisper_max.py    # MAX Graph hybrid

benchmarks/           # Performance testing (was src/benchmarks/ + benchmark_all.py)
examples/             # Demo scripts (was src/demo/)
test/                 # Test files
docs/                 # Documentation
├── agent/           # Project tracking for AI agents
│   ├── PROJECT_STATUS.md      # Current state and blockers
│   ├── DEVELOPMENT_PLAN.md    # Goals and roadmap
│   ├── TECHNICAL_NOTES.md     # Architecture and findings
│   └── PROGRESS_LOG.md        # Session tracking
└── IMPLEMENTATION_GUIDE.md    # Complete technical guide
```

## MAX Graph Patterns
```python
# Device setup
if accelerator_count() > 0:
    driver_device = Accelerator()
    device = DeviceRef.GPU()
else:
    driver_device = CPU()
    device = DeviceRef.CPU()

session = InferenceSession(devices=[driver_device])

# Working operations
ops.matmul(a, b)
ops.layer_norm(x, w, b)  
ops.gelu(x)
ops.transpose(x, 0, 1)
ops.slice_tensor(x, [...])
```

## For AI Agents
- **Project Assessment**: See `docs/agent/CURRENT_STATE_ASSESSMENT.md` (RECOMMENDED START)
- **Current Status**: See `docs/agent/PROJECT_STATUS.md`
- **Development Plan**: See `docs/agent/DEVELOPMENT_PLAN.md`  
- **Technical Details**: See `docs/agent/TECHNICAL_NOTES.md`
- **Progress Tracking**: See `docs/agent/PROGRESS_LOG.md`
- **Infrastructure**: See `docs/agent/IMPROVEMENT_PLAN.md`
- **Mojo Strategy**: See `docs/agent/MOJO_CONVERSION_PLAN.md`
- **Complete Guide**: See `docs/IMPLEMENTATION_GUIDE.md`
- **Setup Instructions**: See `docs/SETUP_GUIDE.md`

## Current Focus
**ENCODER ISSUE IDENTIFIED** - Conv2D-based Conv1D implementation produces semantically corrupted features (cosine similarity: -0.038) despite matching statistics (std: 1.708 vs 1.448). Decoder integration fixed with proper beam search parameters. Next: Fix convolution semantic corruption.

## Key Achievements
- **Architectural Success**: Complete cross-framework integration (MAX Graph encoder → PyTorch decoder)
- **Infrastructure Complete**: Production-quality testing, benchmarking, and logging
- **Performance Excellent**: ~0.28s encoder execution (13x speedup over CPU)
- **Decoder Integration Fixed**: Proper beam search parameters (beam_size=5, temperature=0.0)
- **Debugging Infrastructure**: Comprehensive feature analysis tools (`encoder_feature_debug.py`)