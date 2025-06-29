# max-whisper: Speech Recognition with MAX Graph

**Modular Hackathon 2025 Submission**

An exploration of accelerating OpenAI's Whisper speech recognition using MAX Graph. This project demonstrates the technical challenges and learnings from integrating MAX Graph operations into existing AI models.

## What I Built

| Component | CPU Version | GPU Version | MAX Graph Version |
|-----------|-------------|-------------|-------------------|
| **Speech Recognition** | ✅ Full transcription | ✅ Full transcription | 🔄 Partial output |
| **Text Output** | ✅ Complete audio analysis | ✅ Complete audio analysis | ⚠️ Single words only |
| **Integration** | ✅ Native implementation | ✅ CUDA acceleration | ✅ Compiles and runs |

**Current progress:** Successfully built working CPU and GPU baselines, plus a MAX Graph encoder that compiles and processes audio but produces limited transcription output. The technical foundation is solid - the challenge lies in the mathematical precision required for complex AI model integration.

## Quick Start

```bash
git clone https://github.com/nijaru/max-whisper
cd max-whisper
make install    # Setup environment
make demo      # Compare all three implementations
```

## What I Built

Three implementations:
- `whisper_cpu.py` - baseline OpenAI Whisper 
- `whisper_gpu.py` - CUDA accelerated version
- `whisper_max.py` - MAX Graph encoder version

## MAX Graph Implementation Details

### Technical Achievements
- **Complete MAX Graph encoder** - 4-layer transformer with real operations
- **Full weight integration** - All 65 pretrained weights from Whisper tiny
- **Successful compilation** - Complex computation graphs compile and execute
- **End-to-end pipeline** - Audio → MAX Graph encoder → PyTorch decoder

### Current Limitations
- **Transcription quality** - Produces single words instead of full text
- **Feature compatibility** - Encoder output doesn't fully match PyTorch decoder expectations
- **Mathematical precision** - Subtle differences compound across transformer layers

## The Challenge

Integrating MAX Graph operations into complex AI models reveals fascinating technical challenges. The MAX Graph encoder successfully processes audio and produces mathematically valid features, but the decoder generates limited output. This highlights the precision required in AI model implementations - small mathematical differences can significantly impact semantic understanding.

Key insights:
- Individual operations work correctly, but composition is challenging
- Cross-framework integration requires careful attention to numerical precision
- Transformer architectures are sensitive to implementation details

## What I Learned

This project provided valuable insights into AI acceleration and cross-framework integration:

**Technical Discoveries:**
- MAX Graph operations integrate well with existing codebases
- Weight extraction and tensor conversion work smoothly
- Complex computation graphs compile successfully

**Implementation Challenges:**
- Mathematical precision is critical in transformer models
- Cross-framework compatibility requires careful engineering
- Debugging AI pipelines requires systematic component validation

**Future Potential:**
The foundation demonstrates that MAX Graph acceleration of speech models is technically feasible. With additional refinement of the mathematical precision and decoder integration, this approach could deliver significant performance improvements.

## Try It Yourself

```bash
# Setup
make install        # Install dependencies
make env-check      # Verify environment

# Explore the implementations
make demo          # Compare all three approaches
make max           # Try the MAX Graph version

# Compare working versions
make cpu           # Working CPU baseline
make gpu           # Working GPU version
```

## Project Structure

```
├── src/model/           
│   ├── whisper_cpu.py      # ✅ CPU baseline - works
│   ├── whisper_gpu.py      # ✅ GPU accelerated - works  
│   └── whisper_max.py      # 🔄 MAX Graph encoder - partial
├── benchmark_all.py     # Performance testing
└── audio_samples/       # Test audio
```

Built during the Modular Hackathon 2025 weekend.