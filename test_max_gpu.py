#!/usr/bin/env python3
"""
Test MAX-Whisper GPU performance specifically.
"""

import os
import sys
import time
import numpy as np

# Set up CUDA library path
project_root = os.path.dirname(os.path.abspath(__file__))
cuda_lib_path = os.path.join(project_root, ".pixi/envs/benchmark/lib/python3.11/site-packages/nvidia")
os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}/cublas/lib:{cuda_lib_path}/cudnn/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add src to path
sys.path.insert(0, os.path.join(project_root, 'src'))

from model.max_whisper_simple import SimpleWhisperEncoder


def test_performance():
    print("=== MAX-Whisper GPU Performance Test ===\n")
    
    # Test parameters
    test_durations = [5, 30, 60]  # seconds
    n_mels = 80
    fps = 50  # mel-spectrogram frames per second
    
    results = []
    
    for duration in test_durations:
        print(f"\n--- Testing {duration}s audio ---")
        
        # Create encoder
        encoder = SimpleWhisperEncoder(device="gpu")
        
        # Create test mel-spectrogram
        # MAX Graph expects fixed 1500 frames (30s at 50fps)
        n_frames = 1500
        mel_features = np.random.randn(1, n_mels, n_frames).astype(np.float32)
        print(f"Input shape: {mel_features.shape}")
        
        # Warmup
        print("Warming up...")
        for _ in range(5):
            _ = encoder.encode(mel_features)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        for i in range(20):
            start = time.time()
            output = encoder.encode(mel_features)
            end = time.time()
            times.append(end - start)
            
            if i == 0:
                print(f"Output shape: {output.shape}")
        
        # Calculate metrics
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000
        
        # Always 30s audio due to fixed input shape
        actual_duration = 30.0
        rtf = (avg_time / 1000) / actual_duration
        speedup = 1 / rtf
        
        results.append({
            "duration": duration,
            "avg_ms": avg_time,
            "std_ms": std_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "rtf": rtf,
            "speedup": speedup
        })
        
        print(f"\nResults:")
        print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  RTF: {rtf:.6f}")
        print(f"  Speedup: {speedup:.1f}x real-time")
        
        # Check performance target
        target_speedup = 50
        if speedup > target_speedup:
            print(f"  ✅ Exceeds {target_speedup}x target by {speedup/target_speedup:.1f}x!")
        else:
            print(f"  ⚠️ Need {target_speedup/speedup:.1f}x improvement to reach target")
    
    # Summary table
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY - MAX-Whisper GPU")
    print("="*70)
    print(f"{'Duration':<10} {'Avg (ms)':<12} {'RTF':<12} {'Speedup':<15} {'Status'}")
    print("-"*70)
    
    for r in results:
        status = "✅" if r["speedup"] > 50 else "⚠️"
        print(f"{r['duration']:>6}s    {r['avg_ms']:>8.2f}    {r['rtf']:>8.6f}    {r['speedup']:>10.1f}x    {status}")
    
    print("-"*70)
    print("Target: 50x real-time (RTF < 0.02)")
    print("="*70)
    
    # Performance scaling
    if len(results) > 1:
        print("\nPerformance Scaling:")
        base_rtf = results[0]["rtf"]
        for r in results:
            scaling = base_rtf / r["rtf"]
            print(f"  {r['duration']}s: {scaling:.2f}x relative to {results[0]['duration']}s")


if __name__ == "__main__":
    test_performance()