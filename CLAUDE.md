# CLAUDE.md - AI Agent Instructions

## Project Overview
Speech recognition using OpenAI Whisper with MAX Graph acceleration. Three implementations demonstrate different approaches: CPU baseline, GPU accelerated, and MAX Graph hybrid.

*Originally developed during the Modular Hack Weekend June 2025*

## Implementation Status
| Implementation | File | Status | Performance | Quality |
|---------------|------|--------|-------------|---------|
| CPU Baseline | `max-whisper/whisper_cpu.py` | ✅ Working | ~3.4s | Perfect (2035 chars) |
| GPU Accelerated | `max-whisper/whisper_gpu.py` | ✅ Working | ~1.0s | Perfect (2035 chars) |
| MAX Graph Hybrid | `max-whisper/whisper_max.py` | ✅ Meaningful | ~1.9s (1.8x speedup) | Semantic (422 chars) |
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
**FULL MAX GRAPH DECODER IMPLEMENTATION** - Complete end-to-end MAX Graph pipeline development. Moving beyond 422-character hybrid limitation with pure MAX Graph autoregressive text generation for unlimited semantic content.

## Key Achievements
### Hybrid Implementation (Completed ✅)
- **Meaningful Text Generation**: 422 characters of technically accurate content about MAX Graph libraries and hardware
- **Feature Post-Processing**: Conservative 30% normalization preserves semantics while improving decoder compatibility
- **Advanced Repetition Cleaning**: Smart pattern detection (2-15 word phrases) with adaptive thresholds
- **Performance**: 1.8x speedup over CPU (1.9s vs 3.49s) maintained throughout optimization
- **Cross-Framework Integration**: Stable MAX Graph encoder → PyTorch decoder pipeline
- **Optimization Plateau Identified**: 422 chars represents hybrid approach limitation

### Full MAX Graph Decoder (In Progress 🚀)
- **Complete Architecture**: 4-layer transformer decoder with 100 weight tensors extracted
- **Autoregressive Generation**: Token-by-token semantic text generation framework designed
- **Cross-Attention Implementation**: Encoder-decoder attention mechanism in pure MAX Graph
- **Vocabulary Integration**: Full Whisper vocabulary (51,865 tokens) with tokenization
- **API Resolution**: Final compatibility fixes needed for complete deployment

## Current Achievement
- **Hybrid Phase Complete**: 422-character meaningful text generation with 1.8x speedup achieved
- **Architecture Breakthrough**: Full MAX Graph decoder designed to bypass hybrid limitations
- **Next Target**: Complete end-to-end MAX Graph pipeline for unlimited semantic text generation