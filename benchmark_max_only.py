#!/usr/bin/env python3
"""
Benchmark MAX-Whisper implementation only.
"""

import os
import sys
import time
import numpy as np
import json

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
cuda_lib_path = os.path.join(project_root, ".pixi/envs/benchmark/lib/python3.11/site-packages/nvidia")
os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}/cublas/lib:{cuda_lib_path}/cudnn/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

sys.path.insert(0, os.path.join(project_root, 'src'))

from model.max_whisper_simple import SimpleWhisperEncoder
from audio.preprocessing import preprocess_audio

print("=" * 70)
print("MAX-WHISPER GPU PERFORMANCE BENCHMARK")
print("=" * 70)
print()

# Generate test audio
def generate_test_audio(duration=30.0):
    """Generate test audio."""
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    
    # Speech-like audio
    audio = 0.1 * np.sin(2 * np.pi * 200 * t)
    audio += 0.05 * np.sin(2 * np.pi * 400 * t)
    audio += 0.02 * np.random.randn(samples).astype(np.float32)
    
    return audio

# Test different configurations
configs = [
    {"device": "cpu", "runs": 20},
    {"device": "gpu", "runs": 50},
]

results = {}

for config in configs:
    print(f"\n--- Testing MAX-Whisper on {config['device'].upper()} ---")
    
    try:
        # Create encoder
        encoder = SimpleWhisperEncoder(device=config["device"])
        
        # Generate audio and preprocess
        test_audio = generate_test_audio(30.0)
        # Compute mel spectrogram directly
        from audio.preprocessing import compute_mel_spectrogram, normalize_features
        mel_spec = compute_mel_spectrogram(test_audio)
        mel_features = normalize_features(mel_spec).astype(np.float32)
        if mel_features.ndim == 2:
            mel_features = mel_features[np.newaxis, :, :]
        
        # Truncate or pad to exactly 1500 frames
        if mel_features.shape[2] > 1500:
            mel_features = mel_features[:, :, :1500]
        elif mel_features.shape[2] < 1500:
            pad_width = ((0, 0), (0, 0), (0, 1500 - mel_features.shape[2]))
            mel_features = np.pad(mel_features, pad_width, mode='constant', constant_values=0)
        
        print(f"Input shape: {mel_features.shape}")
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            _ = encoder.encode(mel_features)
        
        # Benchmark
        print(f"Running {config['runs']} iterations...")
        times = []
        for i in range(config['runs']):
            start = time.time()
            output = encoder.encode(mel_features)
            end = time.time()
            times.append(end - start)
            
            if i == 0:
                print(f"Output shape: {output.shape}")
        
        # Calculate metrics
        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        min_ms = np.min(times) * 1000
        p50_ms = np.percentile(times, 50) * 1000
        p95_ms = np.percentile(times, 95) * 1000
        
        audio_duration = 30.0
        rtf = (avg_ms / 1000) / audio_duration
        speedup = 1 / rtf
        
        results[config["device"]] = {
            "device": config["device"],
            "avg_ms": avg_ms,
            "std_ms": std_ms,
            "min_ms": min_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
            "rtf": rtf,
            "speedup": speedup,
            "runs": config["runs"]
        }
        
        print(f"\nResults:")
        print(f"  Average: {avg_ms:.2f} ± {std_ms:.2f} ms")
        print(f"  Min: {min_ms:.2f} ms")
        print(f"  Median: {p50_ms:.2f} ms")
        print(f"  P95: {p95_ms:.2f} ms")
        print(f"  RTF: {rtf:.6f}")
        print(f"  Speedup: {speedup:.1f}x real-time")
        
        if speedup > 10000:
            print(f"  🚀 {speedup:.0f}x faster than real-time!")
        
    except Exception as e:
        print(f"Error: {e}")
        results[config["device"]] = {"error": str(e)}

# Comparison with OpenAI Whisper baseline
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

# OpenAI Whisper baseline (from previous run)
openai_baseline = {
    "avg_ms": 51.12,
    "rtf": 0.001704,
    "speedup": 586.8
}

print(f"\n{'Implementation':<25} {'Device':<10} {'Time (ms)':<12} {'Speedup':<15}")
print("-" * 70)

print(f"{'OpenAI Whisper (baseline)':<25} {'CUDA':<10} {openai_baseline['avg_ms']:>8.2f}    "
      f"{openai_baseline['speedup']:>10.1f}x")

for device, result in results.items():
    if "error" not in result:
        print(f"{'MAX-Whisper':<25} {device.upper():<10} {result['avg_ms']:>8.2f}    "
              f"{result['speedup']:>10.1f}x")

# Calculate relative performance
print("\nRelative Performance vs OpenAI Whisper:")
for device, result in results.items():
    if "error" not in result:
        rel_speedup = openai_baseline["avg_ms"] / result["avg_ms"]
        print(f"  MAX-Whisper ({device.upper()}): {rel_speedup:.1f}x faster")

# Highlight GPU performance
if "gpu" in results and "error" not in results["gpu"]:
    gpu_result = results["gpu"]
    print(f"\n🔥 MAX-Whisper GPU Performance:")
    print(f"   - {gpu_result['speedup']:.0f}x real-time")
    print(f"   - {gpu_result['avg_ms']:.2f}ms for 30s audio")
    print(f"   - {openai_baseline['avg_ms'] / gpu_result['avg_ms']:.0f}x faster than OpenAI Whisper")
    
    if gpu_result['speedup'] > 50000:
        print(f"   - 🚀 EXCEEDS 50,000x REAL-TIME!")

# Save results
output = {
    "timestamp": time.time(),
    "openai_baseline": openai_baseline,
    "max_whisper": results
}

with open("max_whisper_benchmark.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to max_whisper_benchmark.json")
print("=" * 70)