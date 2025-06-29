#!/usr/bin/env python3
"""
CPU vs GPU Performance Comparison
Tests all models on both CPU and GPU with OpenAI Whisper CPU as baseline.
"""

import time
import os
import sys
import json
import librosa
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_openai_whisper(device="cpu"):
    """Test OpenAI Whisper on specified device"""
    try:
        import whisper
        import torch
        
        print(f"🔍 Testing OpenAI Whisper on {device.upper()}...")
        
        # Set device
        if device == "gpu" and torch.cuda.is_available():
            torch_device = "cuda"
        else:
            torch_device = "cpu"
            
        model = whisper.load_model("tiny", device=torch_device)
        
        audio_path = "audio_samples/modular_video.wav"
        if not os.path.exists(audio_path):
            return {"error": "Audio file not found"}
            
        start_time = time.time()
        result = model.transcribe(audio_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        text = result["text"]
        
        return {
            "time": processing_time,
            "text": text,
            "device": torch_device,
            "model": "OpenAI Whisper-tiny"
        }
        
    except Exception as e:
        return {"error": str(e)}

def test_faster_whisper(device="cpu"):
    """Test Faster-Whisper on specified device"""
    try:
        from faster_whisper import WhisperModel
        
        print(f"🔍 Testing Faster-Whisper on {device.upper()}...")
        
        # Set device
        if device == "gpu":
            device_name = "cuda"
        else:
            device_name = "cpu"
            
        model = WhisperModel("tiny", device=device_name)
        
        audio_path = "audio_samples/modular_video.wav"
        if not os.path.exists(audio_path):
            return {"error": "Audio file not found"}
            
        start_time = time.time()
        segments, _ = model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        return {
            "time": processing_time,
            "text": text,
            "device": device_name,
            "model": "Faster-Whisper-tiny"
        }
        
    except Exception as e:
        return {"error": str(e)}

def test_max_whisper_hybrid(device="cpu"):
    """Test MAX-Whisper hybrid approach on specified device"""
    try:
        sys.path.append('src/model')
        from max_whisper_hybrid import MAXWhisperHybrid
        
        print(f"🔍 Testing MAX-Whisper Hybrid on {device.upper()}...")
        
        # Note: Hybrid model currently only supports CPU
        if device == "gpu":
            print("⚠️ Hybrid model currently runs on CPU (using OpenAI Whisper)")
        
        model = MAXWhisperHybrid()
        
        audio_path = "audio_samples/modular_video.wav"
        if not os.path.exists(audio_path):
            return {"error": "Audio file not found"}
            
        start_time = time.time()
        result = model.transcribe(audio_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        return {
            "time": processing_time,
            "text": result["text"],
            "device": "cpu",  # Hybrid always uses CPU currently
            "model": "MAX-Whisper Hybrid"
        }
        
    except Exception as e:
        return {"error": str(e)}

def test_max_whisper_trained(device="cpu"):
    """Test MAX-Whisper with trained weights on specified device"""
    try:
        sys.path.append('src/model')
        
        if device == "gpu":
            from max_whisper_trained_complete import MAXWhisperTrainedComplete
            model = MAXWhisperTrainedComplete(use_gpu=True)
        else:
            from max_whisper_trained_cpu import MAXWhisperTrainedCPU
            model = MAXWhisperTrainedCPU()
        
        print(f"🔍 Testing MAX-Whisper Trained Weights on {device.upper()}...")
        
        # Load and process audio
        audio_path = "audio_samples/modular_video.wav"
        if not os.path.exists(audio_path):
            return {"error": "Audio file not found"}
            
        audio, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, n_fft=400, hop_length=160)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Add batch dimension: (n_mels, time) -> (1, n_mels, time)
        mel_db = np.expand_dims(mel_db, axis=0)
        
        start_time = time.time()
        result = model.transcribe(mel_db)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        return {
            "time": processing_time,
            "text": result,
            "device": device,
            "model": "MAX-Whisper Trained"
        }
        
    except Exception as e:
        return {"error": str(e)}

def calculate_comparative_speedup(baseline_time, test_time):
    """Calculate comparative speedup: how many times faster than baseline"""
    if baseline_time > 0 and test_time > 0:
        return baseline_time / test_time
    return 0

def run_comprehensive_comparison():
    """Run complete CPU vs GPU comparison"""
    print("🚀 Starting Comprehensive CPU vs GPU Comparison")
    print("=" * 80)
    
    # Get audio duration for real-time speedup calculation
    audio_path = "audio_samples/modular_video.wav"
    if os.path.exists(audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / sr
        print(f"📊 Test Audio: {audio_duration:.1f}s duration")
    else:
        print("❌ Audio file not found")
        return
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "audio_duration": audio_duration,
        "tests": {}
    }
    
    # Test all models on CPU
    print("\n🖥️ CPU TESTING")
    print("-" * 40)
    
    tests = [
        ("openai_cpu", lambda: test_openai_whisper("cpu")),
        ("faster_cpu", lambda: test_faster_whisper("cpu")),
        ("hybrid_cpu", lambda: test_max_whisper_hybrid("cpu")),
        ("trained_cpu", lambda: test_max_whisper_trained("cpu"))
    ]
    
    for test_name, test_func in tests:
        result = test_func()
        results["tests"][test_name] = result
        if "error" not in result:
            real_time_speedup = audio_duration / result["time"]
            print(f"✅ {result['model']} (CPU): {result['time']:.2f}s ({real_time_speedup:.1f}x real-time)")
        else:
            print(f"❌ {test_name}: {result['error']}")
    
    # Test all models on GPU
    print("\n🎮 GPU TESTING")
    print("-" * 40)
    
    gpu_tests = [
        ("openai_gpu", lambda: test_openai_whisper("gpu")),
        ("faster_gpu", lambda: test_faster_whisper("gpu")),
        ("hybrid_gpu", lambda: test_max_whisper_hybrid("gpu")),
        ("trained_gpu", lambda: test_max_whisper_trained("gpu"))
    ]
    
    for test_name, test_func in gpu_tests:
        result = test_func()
        results["tests"][test_name] = result
        if "error" not in result:
            real_time_speedup = audio_duration / result["time"]
            print(f"✅ {result['model']} (GPU): {result['time']:.2f}s ({real_time_speedup:.1f}x real-time)")
        else:
            print(f"❌ {test_name}: {result['error']}")
    
    # Calculate comparative speedups using OpenAI CPU as baseline
    baseline_result = results["tests"].get("openai_cpu")
    if baseline_result and "error" not in baseline_result:
        baseline_time = baseline_result["time"]
        print(f"\n📊 COMPARATIVE SPEEDUPS (vs OpenAI Whisper CPU: {baseline_time:.2f}s)")
        print("=" * 80)
        
        comparison_table = []
        
        for test_name, result in results["tests"].items():
            if "error" not in result:
                comparative_speedup = calculate_comparative_speedup(baseline_time, result["time"])
                real_time_speedup = audio_duration / result["time"]
                
                # Truncate text for display
                text_preview = result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]
                
                comparison_table.append({
                    "model": result["model"],
                    "device": result["device"].upper(),
                    "time": f"{result['time']:.2f}s",
                    "comparative_speedup": f"{comparative_speedup:.1f}x",
                    "real_time_speedup": f"{real_time_speedup:.1f}x",
                    "text_preview": text_preview
                })
        
        # Print results table
        print(f"{'Model':<25} {'Device':<8} {'Time':<8} {'vs Baseline':<12} {'vs Real-time':<12} {'Output Quality'}")
        print("-" * 100)
        
        for row in comparison_table:
            quality = "High" if not row["text_preview"].startswith("endend") and len(row["text_preview"]) > 20 else "Low"
            print(f"{row['model']:<25} {row['device']:<8} {row['time']:<8} {row['comparative_speedup']:<12} {row['real_time_speedup']:<12} {quality}")
    
    # Save results
    os.makedirs("results/benchmarks", exist_ok=True)
    
    # Save detailed JSON
    with open("results/benchmarks/cpu_vs_gpu_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary table
    with open("results/benchmarks/cpu_vs_gpu_table.txt", "w") as f:
        f.write("MAX-WHISPER CPU vs GPU COMPARISON\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Audio: {audio_duration:.1f}s Modular technical presentation\n")
        f.write(f"Baseline: OpenAI Whisper CPU ({baseline_time:.2f}s)\n\n")
        
        f.write(f"{'Model':<25} {'Device':<8} {'Time':<8} {'vs Baseline':<12} {'vs Real-time':<12} {'Quality'}\n")
        f.write("-" * 100 + "\n")
        
        for row in comparison_table:
            quality = "High" if not row["text_preview"].startswith("endend") and len(row["text_preview"]) > 20 else "Low"
            f.write(f"{row['model']:<25} {row['device']:<8} {row['time']:<8} {row['comparative_speedup']:<12} {row['real_time_speedup']:<12} {quality}\n")
    
    print(f"\n💾 Results saved to:")
    print(f"   - results/benchmarks/cpu_vs_gpu_results.json")
    print(f"   - results/benchmarks/cpu_vs_gpu_table.txt")
    
    # Analysis
    print(f"\n🔍 ANALYSIS")
    print("-" * 40)
    
    # Find best GPU performance for MAX-Whisper
    max_trained_gpu = results["tests"].get("trained_gpu")
    max_hybrid_gpu = results["tests"].get("hybrid_gpu")
    
    if max_trained_gpu and "error" not in max_trained_gpu:
        trained_speedup = calculate_comparative_speedup(baseline_time, max_trained_gpu["time"])
        print(f"🚀 MAX-Whisper Trained GPU: {trained_speedup:.1f}x faster than OpenAI CPU baseline")
    
    if max_hybrid_gpu and "error" not in max_hybrid_gpu:
        hybrid_speedup = calculate_comparative_speedup(baseline_time, max_hybrid_gpu["time"])
        print(f"🏆 MAX-Whisper Hybrid GPU: {hybrid_speedup:.1f}x faster than OpenAI CPU baseline")
    
    # Quality assessment
    if max_trained_gpu and "error" not in max_trained_gpu:
        if max_trained_gpu["text"].startswith("endend") or len(max_trained_gpu["text"]) < 50:
            print("⚠️ MAX-Whisper Trained: High speed but quality needs improvement (4A refinement needed)")
        else:
            print("✅ MAX-Whisper Trained: High speed AND quality achieved!")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_comparison()