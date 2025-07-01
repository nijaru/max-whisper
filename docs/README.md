# Documentation

Technical documentation for the max-whisper project.

## Documentation Structure

### Core Documentation
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Complete technical guide covering all three implementations

### Agent Documentation
- **[agent/PROJECT_STATUS.md](agent/PROJECT_STATUS.md)** - Current state, blockers, and immediate priorities
- **[agent/DEVELOPMENT_PLAN.md](agent/DEVELOPMENT_PLAN.md)** - Goals, roadmap, and next steps
- **[agent/TECHNICAL_NOTES.md](agent/TECHNICAL_NOTES.md)** - Architecture details and investigation findings
- **[agent/PROGRESS_LOG.md](agent/PROGRESS_LOG.md)** - Session-by-session progress tracking
- **[agent/IMPROVEMENT_PLAN.md](agent/IMPROVEMENT_PLAN.md)** - Infrastructure and tooling improvements
- **[agent/MOJO_CONVERSION_PLAN.md](agent/MOJO_CONVERSION_PLAN.md)** - Strategic Mojo conversion analysis
- **[agent/CURRENT_STATE_ASSESSMENT.md](agent/CURRENT_STATE_ASSESSMENT.md)** - Project goals vs reality check

### Setup & Usage
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete installation and usage guide

## Project Overview

### Three Implementations  
1. **CPU Baseline** (`max-whisper/whisper_cpu.py`) - ✅ Working, perfect quality (~10.6s)
2. **GPU Accelerated** (`max-whisper/whisper_gpu.py`) - ✅ Working, perfect quality (~1.9s)
3. **MAX Graph Hybrid** (`max-whisper/whisper_max.py`) - ⚠️ Technical integration complete, output needs improvement

### Current Status
MAX Graph implementation demonstrates successful architectural integration - all components compile and execute without errors. The encoder processes audio in ~123ms on GPU. However, semantic quality needs improvement as current output produces repetitive tokens rather than meaningful transcription.

## Quick Commands
```bash
make install                           # Setup pixi environment  
pixi run -e benchmark demo             # Compare all implementations (enhanced)
pixi run -e benchmark test-cpu         # Test CPU baseline
pixi run -e benchmark test-gpu         # Test GPU version
pixi run -e benchmark test-max         # Test MAX Graph version
pixi run -e benchmark benchmark-json   # JSON output for analysis
pixi run test                          # Run comprehensive tests
```

## Key Achievement
Complete cross-framework integration (MAX Graph encoder → PyTorch decoder) with proper device management and fast execution, demonstrating both technical success and the challenges of semantic preservation in AI acceleration.