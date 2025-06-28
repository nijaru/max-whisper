# Benchmarking Guide

## Quick Start

```bash
# Setup CUDA
source setup_cuda_env.sh

# Run current encoder benchmark
pixi run -e default python benchmark_max_only.py

# Test all implementations
pixi run -e benchmark python test_all_implementations.py
```

## Current Benchmarks

### Encoder-Only (Current)
- Measures simplified encoder performance
- Not comparable to full transcription
- Shows MAX Graph GPU potential

### Full Model (TODO)
- Compare complete transcription pipelines
- Measure audio → text time
- Fair comparison across implementations

## Metrics

### Speed Metrics
- **Inference Time**: Time to process audio
- **RTF** (Real-Time Factor): Processing time / audio duration
- **Speedup**: 1 / RTF

### Quality Metrics (Future)
- **WER** (Word Error Rate)
- **CER** (Character Error Rate)

## Fair Comparison Requirements

1. **Same Task**: Audio → Text (not just encoding)
2. **Same Model Size**: Use tiny variants
3. **Same Hardware**: All on same GPU/CPU
4. **Same Input**: Identical audio samples
5. **Warm-up Runs**: Exclude initialization

## Expected Results

### Current (Unfair)
- Encoder-only: 0.41ms (72,290x)
- Not comparable to full models

### Realistic (Full Model)
- MAX-Whisper: ~20-30ms (30-50x)
- OpenAI Whisper: ~50ms (20x)
- Faster-Whisper: ~40ms (25x)

## Running Fair Benchmarks

```python
# benchmarks/fair_comparison.py (TODO)
def benchmark_transcription(audio_path):
    # Measure full pipeline
    start = time.time()
    text = model.transcribe(audio_path)
    end = time.time()
    
    # Calculate metrics
    duration = get_audio_duration(audio_path)
    rtf = (end - start) / duration
    
    return {
        "time": end - start,
        "rtf": rtf,
        "text": text
    }
```