#!/usr/bin/env python3
"""
Simple GPU comparison test for MAX-Whisper hackathon.
Tests all three implementations on the same audio.
"""

import os
import sys
import time
import numpy as np
import torch

# Set up CUDA library path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cuda_lib_path = os.path.join(project_root, ".pixi/envs/benchmark/lib/python3.11/site-packages/nvidia")
os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}/cublas/lib:{cuda_lib_path}/cudnn/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Add src to path
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import our modules
from benchmarks.whisper_comparison import WhisperBenchmark
from model.max_whisper_simple import SimpleWhisperEncoder
from audio.preprocessing import preprocess_audio


def main():
    print("=== GPU Performance Comparison ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create benchmark instance with synthetic audio
    benchmark = WhisperBenchmark(use_real_audio=False)
    
    # Get test audio (30 seconds)
    test_audio = benchmark.test_audio
    audio_duration = len(test_audio) / 16000  # 16kHz sample rate
    print(f"\nTest audio duration: {audio_duration:.1f}s")
    
    results = []
    
    # 1. Test OpenAI Whisper (if available)
    try:
        import whisper
        print("\n--- OpenAI Whisper (tiny) ---")
        model = whisper.load_model("tiny", device="cuda")
        
        # Create temp file for audio
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, test_audio, 16000)
            temp_audio_path = tmp.name
        
        # Warmup
        for _ in range(2):
            _ = model.transcribe(temp_audio_path, fp16=True)
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            result = model.transcribe(temp_audio_path, fp16=True)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000
        rtf = (avg_time / 1000) / audio_duration
        
        results.append({
            "name": "OpenAI Whisper",
            "device": "GPU (CUDA)",
            "avg_ms": avg_time,
            "rtf": rtf,
            "speedup": 1/rtf
        })
        print(f"Average: {avg_time:.2f}ms, RTF: {rtf:.6f}, Speedup: {1/rtf:.1f}x")
        print(f"Transcription: {result['text'][:50]}...")
        
        os.unlink(temp_audio_path)
        
    except Exception as e:
        print(f"OpenAI Whisper error: {e}")
    
    # 2. Test Faster-Whisper (if available)
    try:
        from faster_whisper import WhisperModel
        print("\n--- Faster-Whisper (tiny) ---")
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        
        # Warmup
        for _ in range(2):
            segments, _ = model.transcribe(test_audio)
            _ = list(segments)
        
        # Benchmark
        times = []
        for _ in range(5):
            start = time.time()
            segments, info = model.transcribe(test_audio)
            text = " ".join([s.text for s in segments])
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000
        rtf = (avg_time / 1000) / audio_duration
        
        results.append({
            "name": "Faster-Whisper",
            "device": "GPU (CUDA)",
            "avg_ms": avg_time,
            "rtf": rtf,
            "speedup": 1/rtf
        })
        print(f"Average: {avg_time:.2f}ms, RTF: {rtf:.6f}, Speedup: {1/rtf:.1f}x")
        print(f"Transcription: {text[:50]}...")
        
    except Exception as e:
        print(f"Faster-Whisper error: {e}")
    
    # 3. Test MAX-Whisper
    try:
        print("\n--- MAX-Whisper (encoder only) ---")
        encoder = SimpleWhisperEncoder(device="gpu")
        
        # Preprocess audio
        mel_features = preprocess_audio(test_audio)
        if mel_features.ndim == 2:
            mel_features = mel_features[np.newaxis, :, :]
        
        # Warmup
        for _ in range(3):
            _ = encoder.encode(mel_features)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            encoded = encoder.encode(mel_features)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000
        rtf = (avg_time / 1000) / audio_duration
        
        results.append({
            "name": "MAX-Whisper",
            "device": "GPU (MAX Graph)",
            "avg_ms": avg_time,
            "rtf": rtf,
            "speedup": 1/rtf
        })
        print(f"Average: {avg_time:.2f}ms, RTF: {rtf:.6f}, Speedup: {1/rtf:.1f}x")
        print(f"Output shape: {encoded.shape}")
        
    except Exception as e:
        print(f"MAX-Whisper error: {e}")
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY - GPU Performance Comparison")
        print("="*70)
        print(f"{'Model':<20} {'Avg Time (ms)':<15} {'RTF':<12} {'Speedup':<12}")
        print("-"*70)
        
        for r in results:
            print(f"{r['name']:<20} {r['avg_ms']:>10.2f}     {r['rtf']:>8.6f}  {r['speedup']:>8.1f}x")
        
        if len(results) >= 2:
            print("\nRelative Performance:")
            baseline = results[0]["avg_ms"]
            for r in results:
                speedup = baseline / r["avg_ms"]
                print(f"  {r['name']}: {speedup:.2f}x vs {results[0]['name']}")


if __name__ == "__main__":
    main()