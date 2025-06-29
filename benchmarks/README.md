# Benchmarks

## How to Run

```bash
cd benchmarks
pixi run -e default python benchmark.py
```

Results saved to `benchmarks/results.md`

## What It Tests

- OpenAI Whisper (baseline)
- MAX-Whisper (our implementation)  
- Shows actual outputs for comparison

## Files

- `benchmark.py` - Single benchmark script
- `results.md` - Latest benchmark results