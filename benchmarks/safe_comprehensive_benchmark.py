#!/usr/bin/env python3
"""
Safe Comprehensive Benchmark - handles errors gracefully
"""

import json
import time
import os
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_test_audio():
    """Load test audio"""
    try:
        import librosa
        audio_file = "../audio_samples/modular_video.wav"
        
        if os.path.exists(audio_file):
            audio, sr = librosa.load(audio_file, sr=16000)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            print(f"✅ Audio loaded: {len(audio)/sr:.1f}s")
            return audio, mel_db
        else:
            print("⚠️ Real audio not found")
            return None, None
    except Exception as e:
        print(f"⚠️ Audio loading error: {e}")
        return None, None

def test_model_safe(test_name, test_func, *args):
    """Safely test a model with error handling"""
    try:
        print(f"\n🔍 Testing {test_name}...")
        start_time = time.time()
        result = test_func(*args)
        end_time = time.time()
        
        if isinstance(result, dict) and "error" in result:
            return result
        elif isinstance(result, str):
            return {
                "model": test_name,
                "time": end_time - start_time,
                "text": result,
                "status": "✅ Working"
            }
        else:
            return {
                "model": test_name,
                "time": end_time - start_time,
                "text": str(result),
                "status": "✅ Working"
            }
    except Exception as e:
        print(f"  ❌ {test_name} error: {e}")
        return {
            "model": test_name,
            "error": str(e),
            "status": "❌ Error"
        }

def test_openai_cpu():
    """Test OpenAI Whisper CPU"""
    import whisper
    model = whisper.load_model("tiny", device="cpu")
    audio, _ = load_test_audio()
    result = model.transcribe(audio)
    return result["text"].strip()

def test_openai_gpu():
    """Test OpenAI Whisper GPU"""
    import whisper
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("tiny", device=device)
    audio, _ = load_test_audio()
    result = model.transcribe(audio)
    return result["text"].strip()

def test_faster_whisper_cpu():
    """Test Faster-Whisper CPU (safe)"""
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cpu", compute_type="float32")
    audio, _ = load_test_audio()
    segments, info = model.transcribe(audio, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def test_max_full():
    """Test MAX-Whisper Full"""
    os.chdir("..")
    from model.max_whisper_real import MAXWhisperReal
    _, mel_audio = load_test_audio()
    model = MAXWhisperReal(use_gpu=True)
    result = model.transcribe(mel_audio)
    os.chdir("benchmarks")
    return result

def test_max_fixed():
    """Test MAX-Whisper Fixed (Optimized)"""
    os.chdir("..")
    from model.max_whisper_fixed import MAXWhisperFixed
    model = MAXWhisperFixed(use_gpu=True)
    result = model.transcribe()
    os.chdir("benchmarks")
    return result

def run_safe_benchmark():
    """Run safe comprehensive benchmark"""
    print("🚀 Safe Comprehensive MAX-Whisper Benchmark")
    print("=" * 60)
    
    # Test each model safely
    tests = [
        ("OpenAI Whisper CPU", test_openai_cpu),
        ("OpenAI Whisper GPU", test_openai_gpu), 
        ("Faster-Whisper CPU", test_faster_whisper_cpu),
        ("MAX-Whisper Optimized", test_max_fixed),
        ("MAX-Whisper Experimental", test_max_full)
    ]
    
    results = []
    baseline_time = None
    
    for test_name, test_func in tests:
        result = test_model_safe(test_name, test_func)
        results.append(result)
        
        if "time" in result:
            print(f"  ✅ {test_name}: {result['time']:.3f}s")
            if test_name == "OpenAI Whisper CPU":
                baseline_time = result['time']
        else:
            print(f"  ❌ {test_name}: Failed")
    
    # Generate results
    print(f"\n📊 COMPREHENSIVE RESULTS")
    print("=" * 60)
    
    # Create results table
    print(f"{'Model':<20} {'Time':<10} {'Speedup':<10} {'Quality':<15} {'Status'}")
    print("-" * 70)
    
    for result in results:
        if "error" in result:
            print(f"{result['model']:<20} {'ERROR':<10} {'-':<10} {'❌ Failed':<15} {result['status']}")
        else:
            time_str = f"{result['time']:.3f}s"
            if baseline_time and result['time']:
                speedup = baseline_time / result['time']
                speedup_str = f"{speedup:.1f}x"
            else:
                speedup_str = "N/A"
            
            # Assess quality
            text = result.get('text', '')
            if 'max' in text.lower() and 'library' in text.lower():
                quality = "✅ Good"
            elif 'audio' in text.lower() and 'content' in text.lower():
                quality = "🔧 Generic"
            else:
                quality = "❓ Unknown"
            
            print(f"{result['model']:<20} {time_str:<10} {speedup_str:<10} {quality:<15} {result['status']}")
    
    print(f"\n📝 TRANSCRIPTION OUTPUTS")
    print("=" * 60)
    
    for result in results:
        print(f"\n**{result['model']}**:")
        if "error" in result:
            print(f"   Error: {result['error']}")
        else:
            text = result.get('text', 'No output')
            print(f"   {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Write to file
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("comprehensive_results.md", "w") as f:
        f.write("# Comprehensive Benchmark Results\n\n")
        f.write(f"**Date**: {timestamp}\n")
        f.write(f"**Test Audio**: 161.5s Modular presentation\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Model | Time | Speedup | Quality | Status |\n")
        f.write("|-------|------|---------|---------|--------|\n")
        
        for result in results:
            if "error" in result:
                f.write(f"| {result['model']} | ERROR | - | ❌ Failed | {result['status']} |\n")
            else:
                time_str = f"{result['time']:.3f}s"
                speedup_str = f"{baseline_time/result['time']:.1f}x" if baseline_time and result['time'] else "N/A"
                text = result.get('text', '')
                quality = "✅ Good" if 'max' in text.lower() and 'library' in text.lower() else "🔧 Generic" if 'audio' in text.lower() else "❓ Unknown"
                f.write(f"| {result['model']} | {time_str} | {speedup_str} | {quality} | {result['status']} |\n")
        
        f.write("\n## Transcription Outputs\n\n")
        for result in results:
            f.write(f"**{result['model']}**:\n")
            if "error" in result:
                f.write(f"> Error: {result['error']}\n\n")
            else:
                text = result.get('text', 'No output')
                f.write(f"> {text}\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("- **Working Models**: Models that produce actual transcription\n")
        f.write("- **MAX-Whisper Status**: Currently generates generic audio descriptions instead of real transcription\n")
        f.write("- **Performance**: Speed measurements vs OpenAI CPU baseline\n")
    
    print(f"\n✅ Results saved to: comprehensive_results.md")

if __name__ == "__main__":
    run_safe_benchmark()