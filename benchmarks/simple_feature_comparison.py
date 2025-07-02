#!/usr/bin/env python3
"""
Simple feature comparison tool.
Directly runs the implementations and compares their encoder outputs.
"""

import json
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add max-whisper to path
project_root = Path(__file__).parent.parent
max_whisper_path = project_root / "max-whisper"
sys.path.insert(0, str(max_whisper_path))

def run_max_extraction(audio_path="audio_samples/modular_video.wav"):
    """Run MAX Graph and extract features."""
    print(f"🔍 Running MAX Graph feature extraction...")
    
    try:
        # Import and run MAX implementation
        import whisper_max
        result = whisper_max.demo_max(audio_file=audio_path)
        
        # Parse features from the output
        # This is a hack - we'll improve it by modifying the MAX implementation
        print(f"    ✅ MAX Graph completed")
        return {"status": "completed", "result": result}
        
    except Exception as e:
        print(f"    ❌ MAX Graph failed: {e}")
        return {"status": "failed", "error": str(e)}

def run_cpu_extraction(audio_path="audio_samples/modular_video.wav"):
    """Run CPU implementation and extract features."""
    print(f"🔍 Running CPU feature extraction...")
    
    try:
        # Import and run CPU implementation
        import whisper_cpu
        result = whisper_cpu.demo_cpu(audio_file=audio_path)
        
        print(f"    ✅ CPU completed")
        return {"status": "completed", "result": result}
        
    except Exception as e:
        print(f"    ❌ CPU failed: {e}")
        return {"status": "failed", "error": str(e)}

def run_gpu_extraction(audio_path="audio_samples/modular_video.wav"):
    """Run GPU implementation and extract features."""
    print(f"🔍 Running GPU feature extraction...")
    
    try:
        # Import and run GPU implementation
        import whisper_gpu
        result = whisper_gpu.demo_gpu(audio_file=audio_path)
        
        print(f"    ✅ GPU completed")
        return {"status": "completed", "result": result}
        
    except Exception as e:
        print(f"    ❌ GPU failed: {e}")
        return {"status": "failed", "error": str(e)}

def compare_outputs(cpu_result, gpu_result, max_result):
    """Compare the transcription outputs."""
    print(f"\n🔍 Comparing transcription outputs...")
    
    def get_text(result):
        if result["status"] == "completed":
            return result["result"] or "NO_OUTPUT"
        return f"FAILED: {result.get('error', 'unknown')}"
    
    cpu_text = get_text(cpu_result)
    gpu_text = get_text(gpu_result) 
    max_text = get_text(max_result)
    
    # Basic comparison
    cpu_len = len(cpu_text)
    gpu_len = len(gpu_text)
    max_len = len(max_text)
    
    print(f"    📊 Text lengths - CPU: {cpu_len}, GPU: {gpu_len}, MAX: {max_len}")
    
    # Check if CPU and GPU match (they should be nearly identical)
    cpu_gpu_match = cpu_text == gpu_text
    print(f"    🔍 CPU vs GPU match: {cpu_gpu_match}")
    
    # Check if MAX is different (we expect it to be)
    cpu_max_match = cpu_text == max_text
    gpu_max_match = gpu_text == max_text
    print(f"    🔍 CPU vs MAX match: {cpu_max_match}")
    print(f"    🔍 GPU vs MAX match: {gpu_max_match}")
    
    # Show first 100 characters of each
    print(f"\n    📄 CPU (first 100 chars): {cpu_text[:100]}...")
    print(f"    📄 GPU (first 100 chars): {gpu_text[:100]}...")
    print(f"    📄 MAX (first 100 chars): {max_text[:100]}...")
    
    return {
        "cpu_length": cpu_len,
        "gpu_length": gpu_len,
        "max_length": max_len,
        "cpu_gpu_match": cpu_gpu_match,
        "cpu_max_match": cpu_max_match,
        "gpu_max_match": gpu_max_match,
        "cpu_text": cpu_text,
        "gpu_text": gpu_text,
        "max_text": max_text
    }

def main():
    parser = argparse.ArgumentParser(description="Simple feature comparison")
    parser.add_argument("--audio", default="audio_samples/modular_video.wav", help="Audio file")
    parser.add_argument("--output", default="simple_comparison.json", help="Output file")
    parser.add_argument("--implementations", nargs="+", default=["cpu", "gpu", "max"], 
                       choices=["cpu", "gpu", "max"], help="Implementations to test")
    
    args = parser.parse_args()
    
    print(f"🎯 Simple feature comparison")
    print(f"    Audio: {args.audio}")
    print(f"    Implementations: {args.implementations}")
    
    results = {}
    
    # Run each implementation
    if "cpu" in args.implementations:
        results['cpu'] = run_cpu_extraction(args.audio)
    
    if "gpu" in args.implementations:
        results['gpu'] = run_gpu_extraction(args.audio)
    
    if "max" in args.implementations:
        results['max'] = run_max_extraction(args.audio)
    
    # Compare results
    if len(args.implementations) > 1:
        cpu_result = results.get('cpu', {"status": "skipped"})
        gpu_result = results.get('gpu', {"status": "skipped"})
        max_result = results.get('max', {"status": "skipped"})
        
        comparison = compare_outputs(cpu_result, gpu_result, max_result)
        results['comparison'] = comparison
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # Print key findings
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\n🔍 Key Findings:")
        print(f"    CPU vs GPU identical: {comp['cpu_gpu_match']}")
        print(f"    MAX produces different output: {not comp['cpu_max_match']}")
        
        if not comp['cpu_max_match']:
            print(f"    ⚠️  MAX output issue confirmed - semantic quality problem")
        else:
            print(f"    ✅ MAX output matches - semantic quality looks good")

if __name__ == "__main__":
    main()