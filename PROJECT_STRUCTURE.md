# MAX-Whisper Project Structure

Clean, organized structure for easy navigation and development.

## Directory Organization

```
modular-hackathon/
├── README.md                     # Project overview and quick start
├── CLAUDE.md                     # AI agent instructions
├── pixi.toml                     # Environment configuration
│
├── src/                          # Source code
│   ├── model/                    # MAX-Whisper implementations
│   │   ├── max_whisper_complete.py          # ⭐ Main end-to-end model
│   │   ├── max_whisper_real_simple.py       # Simple encoder (0.25ms)
│   │   ├── max_whisper_step2.py             # Multi-head attention (0.41ms)
│   │   ├── max_whisper_decoder.py           # Encoder-decoder
│   │   └── max_whisper_with_trained_weights.py # Weight integration
│   ├── audio/                    # Audio processing utilities
│   │   ├── preprocessing.py      # Audio preprocessing
│   │   └── *.mojo               # Mojo GPU kernels
│   └── benchmarks/              # Benchmark utilities
│
├── tests/                        # Test suite (4/4 passing)
│   ├── README.md                # Test documentation
│   ├── test_everything.py       # ⭐ PRIMARY: All component tests
│   ├── test_baselines_only.py   # Baseline validation
│   └── test_*.py               # Additional tests
│
├── demos/                        # Interactive demonstrations
│   ├── README.md                # Demo documentation
│   ├── demo_trained_weights_simple.py      # Weight loading demo
│   ├── integrate_real_tokenizer.py         # Tokenizer demo
│   └── enhanced_comparison.py              # Model comparison
│
├── scripts/                      # Utility scripts
│   ├── README.md                # Script documentation
│   ├── setup_cuda_env.sh        # CUDA environment setup
│   ├── deploy_lambda_ai.sh      # Cloud deployment
│   └── extract_whisper_weights.py          # Weight extraction
│
├── benchmarks/                   # Performance testing
│   ├── real_audio_comparison.py # ⭐ Head-to-head comparison
│   ├── fair_comparison.py       # Synthetic benchmarks
│   └── gpu_comparison.py        # GPU-specific tests
│
├── docs/                         # Documentation
│   ├── README.md                # Documentation index
│   ├── SETUP_GUIDE.md          # Installation guide
│   ├── API_REFERENCE.md        # Technical reference
│   ├── CURRENT_STATUS.md       # Live technical status
│   ├── STATUS_SUMMARY.md       # Executive summary
│   ├── FINAL_24_HOUR_PLAN.md   # Completion roadmap
│   └── [historical docs...]    # Archive documentation
│
├── audio_samples/               # Test audio files
│   └── modular_video.wav       # Primary test audio (161.5s)
│
├── whisper_weights/             # Trained model weights
│   └── whisper_tiny_weights.npz # ⭐ 47 extracted tensors
│
└── archive/                     # Archived files
    ├── start_real_implementation.py
    └── hello.mojo
```

## Key Entry Points

### For New Users
1. **[README.md](README.md)** - Project overview and quick start
2. **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Complete setup instructions
3. **[tests/test_everything.py](tests/test_everything.py)** - Verify everything works

### For Development
1. **[src/model/max_whisper_complete.py](src/model/max_whisper_complete.py)** - Main model
2. **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - Technical reference
3. **[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)** - Current technical state

### For Demonstrations
1. **[demos/](demos/)** - Interactive demonstrations
2. **[benchmarks/real_audio_comparison.py](benchmarks/real_audio_comparison.py)** - Performance comparison
3. **[tests/test_everything.py](tests/test_everything.py)** - Component validation

### For AI Agents
1. **[CLAUDE.md](CLAUDE.md)** - Complete instructions and status
2. **[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)** - Technical details
3. **[docs/FINAL_24_HOUR_PLAN.md](docs/FINAL_24_HOUR_PLAN.md)** - Immediate priorities

## File Naming Conventions

### Models
- `max_whisper_*.py` - MAX Graph implementations
- `*_complete.py` - Full end-to-end models
- `*_simple.py` - Basic/minimal implementations

### Tests
- `test_*.py` - Test scripts
- `test_everything.py` - Primary comprehensive test

### Demos
- `demo_*.py` - Interactive demonstrations
- `integrate_*.py` - Integration examples

### Scripts
- `setup_*.sh` - Environment setup
- `deploy_*.sh` - Deployment automation
- `extract_*.py` - Data extraction utilities

## Dependencies and Environments

### Pixi Environments
- **default** - MAX Graph models with CUDA
- **benchmark** - Baseline models and comparisons

### Key Files
- `pixi.toml` - Environment configuration
- `pixi.lock` - Locked dependencies
- `scripts/setup_cuda_env.sh` - CUDA environment variables

## Quick Commands

### Setup
```bash
source scripts/setup_cuda_env.sh
export PATH="$HOME/.pixi/bin:$PATH"
```

### Primary Tests
```bash
pixi run -e default python tests/test_everything.py    # All components
pixi run -e benchmark python tests/test_baselines_only.py  # Baselines
```

### Primary Demos
```bash
pixi run -e benchmark python demos/demo_trained_weights_simple.py
pixi run -e benchmark python demos/integrate_real_tokenizer.py
```

### Primary Benchmarks
```bash
pixi run -e benchmark python benchmarks/real_audio_comparison.py
```

## Organization Benefits

### Clear Separation
- **Source code** (`src/`) - Implementation
- **Tests** (`tests/`) - Validation
- **Demos** (`demos/`) - Demonstrations
- **Scripts** (`scripts/`) - Utilities
- **Documentation** (`docs/`) - Guides and references

### Easy Discovery
- README files in each directory explain contents
- Consistent naming conventions
- Clear entry points for different use cases

### Maintainability
- Related files grouped together
- Archive directory for old files
- Documentation reflects current structure