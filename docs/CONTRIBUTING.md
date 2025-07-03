# Contributing to MAX Graph Whisper

Thank you for your interest in contributing to MAX Graph Whisper!

## Development Setup

```bash
git clone https://github.com/your-org/max-whisper
cd max-whisper
make install
make verify
```

## Testing Your Changes

```bash
make test           # Run test suite
make demo          # Test interactive demo
make benchmark     # Run performance tests
```

## Code Organization

- `max-whisper/` - Core implementations (CPU, GPU, MAX Graph)
- `benchmarks/` - Main benchmark tools
- `benchmarks/research/` - Research and debugging tools  
- `benchmarks/archive/` - Legacy tools
- `test/` - Test suite
- `docs/` - Documentation

## Development Workflow

1. Make your changes
2. Run tests: `make test`
3. Test functionality: `make demo` 
4. Commit with clear messages
5. Submit pull request

## Debugging Tools

```bash
make debug-encoder    # Debug encoder features
make debug-features   # Compare feature distributions
make results         # View benchmark history
```

## Questions?

Open an issue or check the documentation in `docs/`.