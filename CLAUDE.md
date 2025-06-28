# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MAX-Whisper**: High-performance speech transcription system using Mojo kernels and MAX Graph model architectures. The goal is to achieve 50-100x performance improvement over OpenAI Whisper while maintaining transcription accuracy.

**Current Status**: Phase 2 GPU development on Fedora/RTX 4090 - Environment setup complete, implementing MAX Graph Whisper on GPU.

### Technical Stack
- **Mojo**: High-performance audio preprocessing and GPU kernel development
- **MAX Graph**: Optimized Whisper model implementation (encoder/decoder)
- **Development Hardware**: macOS (structure/setup) + Linux/RTX 4090 (GPU optimization)
- **Target Hardware**: NVIDIA GPUs (optimized for 4GB+ memory)

## Repository Structure

```
src/
├── audio/          # Mojo audio preprocessing (mel-spectrogram, normalization)
├── model/          # MAX Graph Whisper implementation (encoder/decoder)
├── benchmarks/     # Performance testing and comparison suite
└── demo/           # Demo application (CLI or web interface)
models/             # Pre-trained Whisper weights and model files
tests/              # Test audio files and validation scripts
docs/               # Project documentation and hackathon context
external/
└── modular/        # Modular repository submodule
    ├── mojo/       # Mojo stdlib and documentation
    ├── max/        # MAX Graph documentation and APIs
    └── examples/   # Official Modular examples and tutorials
```

## Performance Requirements

### Target Metrics (Updated for Phase 2)
- **Speed**: 50-100x faster than OpenAI Whisper baseline (RTF < 0.001)
- **Memory**: <8GB GPU utilization on RTX 4090 (24GB available)
- **Accuracy**: Maintain transcription quality equivalent to OpenAI Whisper
- **Hardware**: Optimized for RTX 4090 class GPUs (20GB+ memory)

### Benchmark Targets
1. OpenAI Whisper (baseline)
2. Faster-Whisper (current best practice)
3. MAX-Whisper (this implementation)

## Development Approach

### Implementation Strategy
- Start with basic audio preprocessing pipeline in Mojo
- Implement Whisper encoder/decoder using MAX Graph
- Build incremental benchmarking throughout development
- Focus on working implementation before optimization
- Use existing weight conversion tools to avoid complexity

### Hackathon Timeline
**60-hour sprint**: Friday 6PM → Sunday 6PM  
**✅ Phase 1**: setup → foundation (macOS, complete)  
**🔥 Phase 2**: GPU development → optimization (Fedora/RTX 4090, in progress)  
**Phase 3**: demo → submission  
**Detailed schedule**: See `docs/execution_plan.md` for timeline and `docs/phase2_progress.md` for current status

### Risk Mitigation
- Start with encoder-only version if full Whisper proves too complex
- Build incrementally with testing at each stage
- Have fallback to batch processing if real-time demo fails
- Document trade-offs if accuracy targets can't be met
- Decision points at each phase with fallback strategies

## Development Resources

### Local References
- **Mojo stdlib**: `external/modular/mojo/` - Standard library and core functionality
- **Mojo documentation**: `external/modular/mojo/docs/` - Language and API documentation
- **MAX Graph APIs**: `external/modular/max/` - MAX Graph implementation and docs
- **Official examples**: `external/modular/examples/` - Modular tutorials and sample code

### Online Documentation
- **Mojo GPU Puzzles**: Hands-on challenges for GPU programming skills
- **Optimize custom ops for GPUs**: Focused tutorial for GPU optimization
- **Mojo GPU documentation**: Starting point for Mojo on GPUs
- **Get started with MAX graphs**: Quick guide to building MAX Graphs in Python
- **MAX graph Python API reference**: Complete API documentation

### Additional Resources
- **Tutorial on writing custom PyTorch kernels**: Hardware-agnostic custom ops
- **Forum post on Mojo + PyTorch support**: Community examples for PyTorch integration

## Documentation Guide

### Strategic Planning (Read First)
- **`docs/competitive_strategy.md`** - Differentiation strategy, competitive analysis, winning approach
- **`docs/execution_plan.md`** - Hour-by-hour hackathon timeline with decision points
- **`docs/benchmarking_plan.md`** - Performance measurement methodology and targets

### Technical Implementation
- **`docs/max_whisper_spec.md`** - Detailed technical specification and architecture
- **`docs/hackathon_context.md`** - Hackathon rules, submission requirements, resources

### Demo & Presentation
- **`docs/demo_strategy.md`** - Live demonstration plan and wow factors
- **`docs/presentation_strategy.md`** - Forum post strategy and judge positioning

### Task Management
- **`docs/tasks.json`** - Detailed 60-hour development timeline with phases, priorities, and fallback strategies

### Quick Reference
- **Need current status?** → `docs/phase2_progress.md` (Phase 2 GPU development status)
- **Need development timeline?** → `docs/execution_plan.md` (overview timeline)
- **Need performance targets?** → `docs/benchmarking_plan.md`
- **Need technical details?** → `docs/max_whisper_spec.md`
- **Need demo ideas?** → `docs/demo_strategy.md`
- **Need presentation help?** → `docs/presentation_strategy.md`
- **Need GPU setup?** → `docs/fedora_setup_guide.md`