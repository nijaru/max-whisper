# max-whisper: Speech Recognition with MAX Graph

**Modular Hackathon 2025 Submission**

I tried to accelerate OpenAI's Whisper speech recognition using MAX Graph. It doesn't work - the MAX Graph version fails to do proper speech recognition.

## What Actually Works vs What Doesn't

| Component | CPU Version | GPU Version | MAX Graph Version |
|-----------|-------------|-------------|-------------------|
| **Speech Recognition** | ✅ Works | ✅ Works | ❌ **BROKEN** |
| **Text Output** | ✅ Full transcript | ✅ Full transcript | ❌ Single words only |

**Reality check:** Only CPU and GPU versions actually do speech recognition. MAX Graph version fails - it compiles and runs but only outputs single words before stopping.

## Quick Start

```bash
git clone https://github.com/nijaru/max-whisper
cd max-whisper
make install    # Setup environment
make demo      # See current state - CPU/GPU work, MAX Graph doesn't
```

## What I Built

Three implementations:
- `whisper_cpu.py` - baseline OpenAI Whisper 
- `whisper_gpu.py` - CUDA accelerated version
- `whisper_max.py` - MAX Graph encoder version

## MAX Graph Implementation Details

### What I Built (But Doesn't Work for Speech Recognition)
- **MAX Graph encoder** - Compiles and runs without errors
- **Weight extraction** - Got 65 tensors from Whisper tiny model
- **Basic integration** - MAX Graph connects to PyTorch decoder

### What's Broken
- **Speech recognition** - Only produces 1-2 words then stops
- **No working transcription** - Can't transcribe audio properly
- **Integration issues** - MAX Graph encoder outputs don't work with decoder

## The Problem

When I connect the MAX Graph encoder to Whisper's decoder, it only generates single words like "The" or "I" before hitting the end-of-text token. The encoder produces valid tensors with reasonable statistics, but they don't contain the semantic information the decoder needs.

This suggests either:
- Subtle math differences that compound across transformer layers
- Missing implementation details in the encoder  
- Incompatibility between MAX Graph tensors and PyTorch decoder expectations

## What I Learned

MAX Graph tensor operations compile and run, but building working AI pipelines is much harder than I expected. Even though the encoder produces valid tensors, something about the integration with PyTorch models breaks the speech recognition.

Getting the math exactly right in complex AI models is difficult. Small differences can break everything even if individual components seem to work.

## Try It Yourself

```bash
# Setup
make install        # Install dependencies
make env-check      # Verify environment

# See the problem
make demo          # Run all three - you'll see MAX Graph fails
make max           # Just run broken MAX Graph version

# Compare working versions
make cpu           # Working CPU baseline
make gpu           # Working GPU version
```

## Project Structure

```
├── src/model/           
│   ├── whisper_cpu.py      # ✅ CPU baseline - works
│   ├── whisper_gpu.py      # ✅ GPU accelerated - works  
│   └── whisper_max.py      # ❌ MAX Graph encoder - broken
├── benchmark_all.py     # Performance testing
└── audio_samples/       # Test audio
```

Built during the Modular Hackathon 2025 weekend.