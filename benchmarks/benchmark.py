#!/usr/bin/env python3
"""
MAX-Whisper Benchmark
Single script to test all implementations and show results

Usage:
    python benchmark.py

Results saved to: benchmarks/results.md
"""

import json
import time
import os
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports (from benchmarks/ directory)
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_test_audio():
    """Load test audio if available"""
    try:
        import librosa
        audio_file = "../audio_samples/modular_video.wav"
        
        if os.path.exists(audio_file):
            audio, sr = librosa.load(audio_file, sr=16000)
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            print(f"✅ Using real audio: {len(audio)/sr:.1f}s")
            return audio, mel_db
        else:
            print("⚠️ Real audio not found, using synthetic")
            return None, np.random.randn(80, 3000).astype(np.float32)
    except ImportError:
        print("⚠️ Audio processing not available, using synthetic")
        return None, np.random.randn(80, 3000).astype(np.float32)

def test_openai_whisper(audio):
    """Test OpenAI Whisper (baseline)"""
    if audio is None:
        return {"error": "No audio available", "model": "OpenAI Whisper"}
    
    try:
        import whisper
        print("🔍 Testing OpenAI Whisper...")
        
        model = whisper.load_model("tiny")
        start_time = time.time()
        result = model.transcribe(audio)
        end_time = time.time()
        
        return {
            "model": "OpenAI Whisper",
            "time": end_time - start_time,
            "text": result["text"],
            "status": "✅ Working"
        }
    except ImportError:
        return {"error": "OpenAI Whisper not installed", "model": "OpenAI Whisper"}
    except Exception as e:
        return {"error": str(e), "model": "OpenAI Whisper"}

def test_max_whisper(mel_audio):
    """Test MAX-Whisper implementation"""
    try:
        # Change to parent directory to find weights
        os.chdir("..")
        
        from model.max_whisper_real import MAXWhisperReal
        print("🔍 Testing MAX-Whisper...")
        
        model = MAXWhisperReal(use_gpu=True)
        if not model.available:
            return {"error": "MAX Graph not available", "model": "MAX-Whisper"}
        
        start_time = time.time()
        result = model.transcribe(mel_audio)
        end_time = time.time()
        
        # Change back to benchmarks directory
        os.chdir("benchmarks")
        
        return {
            "model": "MAX-Whisper",
            "time": end_time - start_time,
            "text": result,
            "status": "✅ Working" if "content" in result or "audio" in result else "🔧 In Progress"
        }
    except ImportError:
        return {"error": "MAX Graph not available", "model": "MAX-Whisper"}
    except Exception as e:
        return {"error": str(e), "model": "MAX-Whisper"}

def assess_quality(text):
    """Simple quality assessment"""
    if not text or len(text) < 10:
        return "❌ No output"
    
    if "the max graph provide" in text.lower():
        return "❌ Hardcoded demo"
    
    if "tokens:" in text.lower() or "processing" in text.lower():
        return "🔧 Technical output"
    
    if len(text) > 50 and any(word in text.lower() for word in ["the", "and", "of"]):
        return "✅ Real transcription"
    
    return "❓ Unknown"

def create_results_table(results):
    """Create simple results table"""
    baseline_time = None
    
    # Find baseline time
    for result in results:
        if result["model"] == "OpenAI Whisper" and "error" not in result:
            baseline_time = result["time"]
            break
    
    table = "# MAX-Whisper Benchmark Results\n\n"
    table += f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n"
    table += f"**Hardware**: RTX 4090 + CUDA 12.9\n\n"
    
    table += "| Model | Time | vs Baseline | Quality | Status |\n"
    table += "|-------|------|-------------|---------|--------|\n"
    
    for result in results:
        if "error" not in result:
            speedup = "1.0x" if result["model"] == "OpenAI Whisper" else f"{baseline_time / result['time']:.0f}x" if baseline_time else "N/A"
            quality = assess_quality(result.get("text", ""))
            table += f"| {result['model']} | {result['time']:.3f}s | {speedup} | {quality} | {result['status']} |\n"
        else:
            table += f"| {result['model']} | ERROR | - | - | ❌ {result['error'][:30]}... |\n"
    
    table += "\n## Outputs\n\n"
    
    for result in results:
        if "error" not in result and "text" in result:
            table += f"**{result['model']}**:\n"
            table += f"> {result['text'][:100]}...\n\n"
    
    return table

def main():
    """Run benchmark"""
    print("🚀 MAX-Whisper Benchmark")
    print("=" * 40)
    
    # Load test data
    audio, mel_audio = load_test_audio()
    
    # Run tests
    results = []
    results.append(test_openai_whisper(audio))
    results.append(test_max_whisper(mel_audio))
    
    # Create results
    table = create_results_table(results)
    
    # Save results to benchmarks directory
    results_file = "results.md"
    with open(results_file, 'w') as f:
        f.write(table)
    
    # Print summary
    print(f"\n📊 Results:")
    for result in results:
        if "error" not in result:
            print(f"  {result['model']}: {result['time']:.3f}s - {assess_quality(result.get('text', ''))}")
        else:
            print(f"  {result['model']}: ERROR - {result['error']}")
    
    print(f"\n✅ Full results saved to: benchmarks/{results_file}")

if __name__ == "__main__":
    main()