#!/usr/bin/env python3
"""
Simple MAX-Whisper GPU performance test.
"""

import os
import sys
import time
import numpy as np

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
cuda_lib_path = os.path.join(project_root, ".pixi/envs/benchmark/lib/python3.11/site-packages/nvidia")
os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}/cublas/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

sys.path.insert(0, os.path.join(project_root, 'src'))
from model.max_whisper_simple import SimpleWhisperEncoder


def main():
    print("=== MAX-Whisper GPU Performance Test ===\n")
    
    # Create encoder
    print("Initializing MAX-Whisper on GPU...")
    encoder = SimpleWhisperEncoder(device="gpu")
    
    # Create test input (30s audio at 50fps = 1500 frames)
    mel_features = np.random.randn(1, 80, 1500).astype(np.float32)
    print(f"Input shape: {mel_features.shape} (30s audio)")
    
    # Warmup
    print("\nWarming up GPU...")
    for _ in range(10):
        _ = encoder.encode(mel_features)
    
    # Benchmark
    print("Running benchmark...")
    times = []
    for i in range(50):
        start = time.time()
        output = encoder.encode(mel_features)
        end = time.time()
        times.append(end - start)
        
        if i == 0:
            print(f"Output shape: {output.shape}")
    
    # Results
    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    min_ms = np.min(times) * 1000
    p50_ms = np.percentile(times, 50) * 1000
    p95_ms = np.percentile(times, 95) * 1000
    
    audio_duration = 30.0  # seconds
    rtf = (avg_ms / 1000) / audio_duration
    speedup = 1 / rtf
    
    print("\n" + "="*50)
    print("RESULTS - MAX-Whisper GPU Performance")
    print("="*50)
    print(f"Average time: {avg_ms:.2f} ± {std_ms:.2f} ms")
    print(f"Min time: {min_ms:.2f} ms")
    print(f"Median (P50): {p50_ms:.2f} ms")
    print(f"P95: {p95_ms:.2f} ms")
    print(f"\nAudio duration: {audio_duration}s")
    print(f"Real-time factor: {rtf:.6f}")
    print(f"Speedup: {speedup:.1f}x real-time")
    
    # Compare to target
    target_speedup = 50
    if speedup > target_speedup:
        print(f"\n✅ SUCCESS: Exceeds {target_speedup}x target by {speedup/target_speedup:.1f}x!")
        print(f"   Achieved {speedup:.0f}x speedup vs real-time")
    else:
        print(f"\n⚠️ Below target: Need {target_speedup/speedup:.1f}x improvement")
    
    # Estimated performance for different durations
    print("\nEstimated performance for other durations:")
    for duration in [5, 10, 60, 120]:
        est_time = avg_ms * (duration / 30.0)
        est_rtf = est_time / 1000 / duration
        print(f"  {duration:>3}s audio: ~{est_time:>6.1f}ms (RTF: {est_rtf:.6f})")


if __name__ == "__main__":
    main()