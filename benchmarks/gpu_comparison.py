"""
Comprehensive GPU benchmark comparing OpenAI Whisper, Faster-Whisper, and MAX-Whisper.
Tests all three implementations on the same audio data with GPU acceleration.
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import benchmark utilities
from benchmarks.test_data import get_test_audio

# Set up CUDA library path for MAX
cuda_lib_path = str(project_root / ".pixi/envs/benchmark/lib/python3.11/site-packages/nvidia")
os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}/cublas/lib:{cuda_lib_path}/cudnn/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"


def benchmark_openai_whisper_gpu(audio_path: str, model_size: str = "tiny"):
    """Benchmark OpenAI Whisper on GPU."""
    import whisper
    
    print(f"\n--- OpenAI Whisper ({model_size}) on GPU ---")
    
    # Load model on GPU
    model = whisper.load_model(model_size, device="cuda")
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        _ = model.transcribe(audio_path, fp16=True)
    
    # Benchmark
    print("Benchmarking...")
    times = []
    transcriptions = []
    
    for i in range(5):
        start = time.time()
        result = model.transcribe(audio_path, fp16=True)
        end = time.time()
        
        times.append(end - start)
        transcriptions.append(result["text"])
        
        if i == 0:
            print(f"Transcription: {result['text'][:100]}...")
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    # Get audio duration
    audio_duration = whisper.load_audio(audio_path).shape[0] / 16000
    rtf = (avg_time / 1000) / audio_duration
    
    return {
        "implementation": "OpenAI Whisper",
        "device": "GPU (CUDA)",
        "model_size": model_size,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "speedup": 1/rtf,
        "transcription": transcriptions[0]
    }


def benchmark_faster_whisper_gpu(audio_path: str, model_size: str = "tiny"):
    """Benchmark Faster-Whisper on GPU."""
    from faster_whisper import WhisperModel
    
    print(f"\n--- Faster-Whisper ({model_size}) on GPU ---")
    
    # Load model on GPU
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        segments, _ = model.transcribe(audio_path)
        _ = list(segments)
    
    # Benchmark
    print("Benchmarking...")
    times = []
    transcriptions = []
    
    for i in range(5):
        start = time.time()
        segments, info = model.transcribe(audio_path)
        text = " ".join([s.text for s in segments])
        end = time.time()
        
        times.append(end - start)
        transcriptions.append(text)
        
        if i == 0:
            print(f"Transcription: {text[:100]}...")
            print(f"Detected language: {info.language}")
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    # Get audio duration
    import wave
    with wave.open(audio_path, 'rb') as wav:
        audio_duration = wav.getnframes() / wav.getframerate()
    
    rtf = (avg_time / 1000) / audio_duration
    
    return {
        "implementation": "Faster-Whisper",
        "device": "GPU (CUDA)",
        "model_size": model_size,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "speedup": 1/rtf,
        "transcription": transcriptions[0]
    }


def benchmark_max_whisper_gpu(audio_path: str):
    """Benchmark MAX-Whisper on GPU."""
    # Import our implementations
    from src.model.max_whisper_simple import SimpleWhisperEncoder
    from src.audio.mojo_audio_kernel import preprocess_audio_from_file
    
    print(f"\n--- MAX-Whisper (simplified) on GPU ---")
    
    # Create encoder on GPU
    encoder = SimpleWhisperEncoder(device="gpu")
    
    # Load and preprocess audio
    print("Loading audio...")
    mel_features = preprocess_audio_from_file(audio_path)
    
    # Ensure correct shape
    if mel_features.ndim == 2:
        mel_features = mel_features[np.newaxis, :, :]
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = encoder.encode(mel_features)
    
    # Benchmark
    print("Benchmarking...")
    times = []
    
    for i in range(10):
        start = time.time()
        encoded = encoder.encode(mel_features)
        end = time.time()
        
        times.append(end - start)
        
        if i == 0:
            print(f"Encoded shape: {encoded.shape}")
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    # Get audio duration
    import wave
    with wave.open(audio_path, 'rb') as wav:
        audio_duration = wav.getnframes() / wav.getframerate()
    
    rtf = (avg_time / 1000) / audio_duration
    
    return {
        "implementation": "MAX-Whisper",
        "device": "GPU (MAX Graph)",
        "model_size": "tiny (encoder only)",
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
        "audio_duration": audio_duration,
        "rtf": rtf,
        "speedup": 1/rtf,
        "transcription": "[Encoder only - no decoder yet]"
    }


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("GPU PERFORMANCE COMPARISON")
    print("="*100)
    
    # Header
    print(f"{'Implementation':<20} {'Device':<15} {'Avg Time (ms)':<15} {'RTF':<12} {'Speedup':<12} {'Status'}")
    print("-"*100)
    
    # Results
    for r in results:
        status = "✅" if r["rtf"] < 0.05 else "⚠️"
        print(f"{r['implementation']:<20} {r['device']:<15} "
              f"{r['avg_time_ms']:>8.2f} ± {r['std_time_ms']:<5.2f} "
              f"{r['rtf']:>8.6f}  {r['speedup']:>8.1f}x  {status}")
    
    print("-"*100)
    
    # Calculate relative performance
    if len(results) >= 2:
        baseline = results[0]["avg_time_ms"]  # OpenAI Whisper
        print("\nRelative Performance:")
        for r in results:
            speedup = baseline / r["avg_time_ms"]
            print(f"  {r['implementation']}: {speedup:.2f}x faster than OpenAI Whisper")
    
    print("\nTarget: RTF < 0.05 (20x real-time)")
    print("="*100)


def main():
    """Run comprehensive GPU benchmark."""
    print("=== MAX-Whisper GPU Benchmark Comparison ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Get test audio
    print("\nPreparing test audio...")
    test_audio = get_test_audio("speech_standard")  # 30 second clip
    audio_path = test_audio["path"]
    print(f"Test audio: {test_audio['description']} ({test_audio['duration']}s)")
    
    results = []
    
    # Benchmark OpenAI Whisper
    try:
        result = benchmark_openai_whisper_gpu(audio_path)
        results.append(result)
    except Exception as e:
        print(f"OpenAI Whisper GPU error: {e}")
    
    # Benchmark Faster-Whisper
    try:
        result = benchmark_faster_whisper_gpu(audio_path)
        results.append(result)
    except Exception as e:
        print(f"Faster-Whisper GPU error: {e}")
    
    # Benchmark MAX-Whisper
    try:
        result = benchmark_max_whisper_gpu(audio_path)
        results.append(result)
    except Exception as e:
        print(f"MAX-Whisper GPU error: {e}")
    
    # Print comparison
    if results:
        print_comparison_table(results)
        
        # Save results
        import json
        output_file = "gpu_benchmark_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "results": results,
                "environment": {
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
                }
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo successful benchmarks to compare.")


if __name__ == "__main__":
    main()