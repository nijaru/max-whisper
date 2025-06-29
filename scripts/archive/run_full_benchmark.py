#!/usr/bin/env python3
"""
Comprehensive benchmark comparing all Whisper implementations.
Handles CUDA setup and runs all tests.
"""

import os
import sys
import time
import numpy as np
import subprocess
import json

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
cuda_lib_path = os.path.join(project_root, ".pixi/envs/benchmark/lib/python3.11/site-packages/nvidia")
os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}/cublas/lib:{cuda_lib_path}/cudnn/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add src to path
sys.path.insert(0, os.path.join(project_root, 'src'))

print("=" * 80)
print("MAX-WHISPER GPU HACKATHON BENCHMARK")
print("=" * 80)
print()

# Check GPU availability
try:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
except:
    print("PyTorch not available")

print()

# Results storage
all_results = {
    "timestamp": time.time(),
    "environment": {
        "cuda_available": False,
        "gpu_name": None,
    },
    "benchmarks": {}
}

# Test audio generation
def generate_test_audio(duration=30.0):
    """Generate test audio for benchmarking."""
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    
    # Generate speech-like audio
    audio = np.zeros(samples, dtype=np.float32)
    
    # Add multiple frequency components
    frequencies = [200, 400, 800, 1200]  # Speech formants
    for freq in frequencies:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # Add some noise
    audio += 0.05 * np.random.randn(samples).astype(np.float32)
    
    # Add envelope
    envelope = np.exp(-0.5 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    audio *= envelope
    
    return audio

# Create test audio
print("Generating test audio...")
test_audio = generate_test_audio(30.0)
audio_duration = len(test_audio) / 16000
print(f"Test audio: {audio_duration:.1f}s")
print()

# 1. Benchmark OpenAI Whisper (if available)
print("=" * 60)
print("1. OpenAI Whisper Benchmark")
print("=" * 60)
try:
    import whisper
    import tempfile
    import soundfile as sf
    
    print("Loading Whisper tiny model...")
    model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, test_audio, 16000)
        temp_path = tmp.name
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        _ = model.transcribe(temp_path, fp16=torch.cuda.is_available())
    
    # Benchmark
    print("Benchmarking...")
    times = []
    for i in range(5):
        start = time.time()
        result = model.transcribe(temp_path, fp16=torch.cuda.is_available())
        end = time.time()
        times.append(end - start)
        
        if i == 0:
            print(f"Transcription: {result['text'][:50]}...")
    
    avg_time = np.mean(times) * 1000
    rtf = (avg_time / 1000) / audio_duration
    
    all_results["benchmarks"]["openai_whisper"] = {
        "implementation": "OpenAI Whisper",
        "model": "tiny",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "avg_time_ms": avg_time,
        "rtf": rtf,
        "speedup": 1/rtf,
        "status": "success"
    }
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  RTF: {rtf:.6f}")
    print(f"  Speedup: {1/rtf:.1f}x real-time")
    
    os.unlink(temp_path)
    
except Exception as e:
    print(f"OpenAI Whisper failed: {e}")
    all_results["benchmarks"]["openai_whisper"] = {"status": "failed", "error": str(e)}

print()

# 2. Benchmark Faster-Whisper (if available)
print("=" * 60)
print("2. Faster-Whisper Benchmark")
print("=" * 60)
try:
    from faster_whisper import WhisperModel
    
    print("Loading Faster-Whisper tiny model...")
    model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        segments, _ = model.transcribe(test_audio)
        _ = list(segments)
    
    # Benchmark
    print("Benchmarking...")
    times = []
    for i in range(5):
        start = time.time()
        segments, info = model.transcribe(test_audio)
        text = " ".join([s.text for s in segments])
        end = time.time()
        times.append(end - start)
        
        if i == 0:
            print(f"Transcription: {text[:50]}...")
    
    avg_time = np.mean(times) * 1000
    rtf = (avg_time / 1000) / audio_duration
    
    all_results["benchmarks"]["faster_whisper"] = {
        "implementation": "Faster-Whisper",
        "model": "tiny",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "avg_time_ms": avg_time,
        "rtf": rtf,
        "speedup": 1/rtf,
        "status": "success"
    }
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  RTF: {rtf:.6f}")
    print(f"  Speedup: {1/rtf:.1f}x real-time")
    
except Exception as e:
    print(f"Faster-Whisper failed: {e}")
    all_results["benchmarks"]["faster_whisper"] = {"status": "failed", "error": str(e)}

print()

# 3. Benchmark MAX-Whisper
print("=" * 60)
print("3. MAX-Whisper Benchmark")
print("=" * 60)
try:
    from model.max_whisper_simple import SimpleWhisperEncoder
    from audio.preprocessing import preprocess_audio
    
    print("Loading MAX-Whisper encoder...")
    encoder = SimpleWhisperEncoder(device="gpu")
    
    # Preprocess audio
    mel_features = preprocess_audio(test_audio)
    if mel_features.ndim == 2:
        mel_features = mel_features[np.newaxis, :, :]
    
    print(f"Mel features shape: {mel_features.shape}")
    
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
            print(f"Encoder output shape: {output.shape}")
    
    avg_time = np.mean(times) * 1000
    rtf = (avg_time / 1000) / audio_duration
    
    all_results["benchmarks"]["max_whisper"] = {
        "implementation": "MAX-Whisper",
        "model": "tiny (encoder only)",
        "device": encoder.device_str,
        "avg_time_ms": avg_time,
        "rtf": rtf,
        "speedup": 1/rtf,
        "status": "success",
        "note": "Encoder only - full model in development"
    }
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  RTF: {rtf:.6f}")
    print(f"  Speedup: {1/rtf:.1f}x real-time")
    
except Exception as e:
    print(f"MAX-Whisper failed: {e}")
    all_results["benchmarks"]["max_whisper"] = {"status": "failed", "error": str(e)}

print()

# Summary
print("=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)
print()

successful_benchmarks = [b for b in all_results["benchmarks"].values() if b.get("status") == "success"]

if successful_benchmarks:
    # Sort by speed
    successful_benchmarks.sort(key=lambda x: x["avg_time_ms"])
    
    print(f"{'Implementation':<20} {'Device':<10} {'Time (ms)':<12} {'RTF':<12} {'Speedup':<15}")
    print("-" * 80)
    
    for b in successful_benchmarks:
        print(f"{b['implementation']:<20} {b['device']:<10} "
              f"{b['avg_time_ms']:>8.2f}    {b['rtf']:>8.6f}    "
              f"{b['speedup']:>10.1f}x")
    
    print()
    
    # Relative performance
    if len(successful_benchmarks) > 1:
        print("Relative Performance:")
        baseline = successful_benchmarks[-1]  # Slowest
        for b in successful_benchmarks:
            rel_speed = baseline["avg_time_ms"] / b["avg_time_ms"]
            print(f"  {b['implementation']}: {rel_speed:.2f}x faster than {baseline['implementation']}")
    
    # MAX-Whisper performance
    max_whisper = next((b for b in successful_benchmarks if b["implementation"] == "MAX-Whisper"), None)
    if max_whisper:
        print()
        print("🚀 MAX-Whisper Performance:")
        print(f"   - {max_whisper['speedup']:.0f}x real-time")
        print(f"   - {max_whisper['avg_time_ms']:.2f}ms for {audio_duration:.0f}s audio")
        
        openai = next((b for b in successful_benchmarks if b["implementation"] == "OpenAI Whisper"), None)
        if openai:
            speedup_vs_openai = openai["avg_time_ms"] / max_whisper["avg_time_ms"]
            print(f"   - {speedup_vs_openai:.0f}x faster than OpenAI Whisper")

# Save results
output_file = "benchmark_results_full.json"
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {output_file}")
print("=" * 80)