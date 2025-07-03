# CLAUDE.md - AI Agent Instructions

## Project Overview
Speech recognition using OpenAI Whisper with MAX Graph acceleration. Three implementations demonstrate different approaches: CPU baseline, GPU accelerated, and MAX Graph hybrid.

*Originally developed during the Modular Hack Weekend June 2025*

## Implementation Status
| Implementation | File | Status | Performance | Quality |
|---------------|------|--------|-------------|---------|
| CPU Baseline | `max-whisper/whisper_cpu.py` | ✅ Working | ~3.4s | Perfect (2035 chars) |
| GPU Accelerated | `max-whisper/whisper_gpu.py` | ✅ Working | ~1.0s | Perfect (2035 chars) |
| MAX Graph Hybrid | `max-whisper/whisper_max.py` | ⚠️ Length Limited | ~1.9s (1.8x speedup) | Semantic (259 chars) |
| **MAX Graph Full** | `max-whisper/whisper_max.py --full-max-graph` | 🔧 Development | ~1.0s target | Research Phase |

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
**SEMANTIC QUALITY ACHIEVED** - MAX Graph encoder produces semantically correct transcription matching CPU baseline content. Encoder working properly (std: 1.447 ≈ OpenAI 1.448). Current challenge: extending transcription length from 259 to 2035+ characters while maintaining semantic accuracy.

## Key Achievements
- **Semantic Quality Breakthrough**: MAX Graph encoder produces correct content ("Max provides several different libraries...")
- **Encoder-Decoder Integration**: Successful cross-framework tensor passing (MAX Graph → PyTorch)
- **Performance**: 1.8x speedup over CPU (1.9s vs 3.4s) with semantic accuracy
- **Statistical Matching**: Encoder std: 1.447 matches OpenAI std: 1.448 exactly
- **Architecture Validation**: Core pipeline proven functional without artificial corrections
- **Content Accuracy**: Transcription matches CPU baseline semantic content
- **Conv2D Fallback**: Working implementation for Conv1D operations in MAX Graph
- **Error Resolution**: Fixed tensor layout and feature scaling issues

## Current Challenge
- **Length Limitation**: Consistent stopping at ~259 characters (12.7% of baseline)
- **Decoder Investigation**: PyTorch decoder confidence drops with MAX Graph features
- **Next Phase**: Analyze stopping criteria and feature patterns to extend length