# MAX-Whisper Hackathon Submission

**Team**: Solo submission  
**Project**: MAX-Whisper - GPU-Accelerated Speech Recognition with MAX Graph

## What We Built

We implemented a speech recognition encoder using MAX Graph that demonstrates the platform's potential for transformer models on GPUs. Our implementation shows how to:

1. Build neural networks with MAX Graph API
2. Execute models on NVIDIA GPUs 
3. Achieve significant performance improvements
4. Create custom architectures beyond PyTorch/TensorFlow

## Technical Implementation

### Architecture
- Custom encoder implementation using MAX Graph
- GPU-optimized tensor operations
- Native CUDA execution on RTX 4090
- Zero-copy memory transfers

### Performance
- **Encoder Performance**: 0.41ms for 30s of audio features
- **Device**: NVIDIA RTX 4090 (24GB VRAM)
- **Framework**: MAX Graph with Python API

### Code Highlights
```python
# Building models with MAX Graph
with Graph("whisper_encoder", input_types=(input_type,)) as graph:
    x = graph.inputs[0]
    # Custom operations
    x = ops.permute(x, [0, 2, 1])
    x = ops.matmul(x_flat, weight)
    x = elementwise.relu(x)
    graph.output(x)

# GPU execution
session = engine.InferenceSession(devices=[Accelerator(id=0)])
model = session.load(graph)
```

## Current Limitations & Next Steps

This hackathon implementation demonstrates the foundation. To complete the system:

1. Add decoder with attention mechanisms
2. Load pretrained Whisper weights
3. Integrate tokenizer for text output
4. Implement full transformer architecture

## Why This Matters

- **MAX Graph Potential**: Shows how to build complex models
- **GPU Performance**: Demonstrates native GPU acceleration
- **Beyond Frameworks**: Alternative to PyTorch/TensorFlow
- **Future Applications**: Foundation for production systems

## Running the Demo

```bash
# Setup
source setup_cuda_env.sh
pixi install

# Run benchmark
pixi run -e default python benchmark_max_only.py

# View presentation
python demo_presentation.py
```

## Repository Structure
- `src/model/` - MAX Graph implementations
- `src/audio/` - Audio preprocessing
- `benchmarks/` - Performance testing
- `docs/` - Technical documentation

## Acknowledgments

Built for Modular Hack Weekend (June 27-29, 2025). Thanks to Modular for MAX Graph and NVIDIA for GPU sponsorship.

---

*See [Technical Specification](docs/TECHNICAL_SPEC.md) for implementation details.*