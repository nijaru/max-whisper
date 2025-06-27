# MAX-Whisper

High-performance speech transcription using Mojo kernels and MAX Graph model architectures.

## Quick Start

This project implements an optimized Whisper model targeting 2-3x performance improvement over baseline implementations while maintaining transcription accuracy.

### Architecture
- **Audio Processing**: Mojo kernels for mel-spectrogram computation and preprocessing
- **Model Implementation**: MAX Graph for optimized Whisper encoder/decoder
- **Target Performance**: 2-3x speed improvement, 40-50% memory reduction

### Documentation
- `docs/hackathon_context.md` - Hackathon details and submission requirements
- `docs/max_whisper_spec.md` - Detailed project specification and implementation plan
- `CLAUDE.md` - Development guidance for Claude Code

### Development
See `CLAUDE.md` for development setup and technical architecture details.

### Setup
```bash
# Clone with submodules
git clone --recurse-submodules <this-repo>

# Or initialize submodules after cloning
git submodule update --init --recursive
```

The Modular repository is available at `external/modular/` with Mojo stdlib, MAX documentation, and official examples.