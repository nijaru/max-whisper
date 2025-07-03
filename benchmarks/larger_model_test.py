#!/usr/bin/env python3
"""
Larger Model Test for MAX Graph Whisper
Test if larger Whisper models are more robust to feature distribution differences
Quick validation before moving to full MAX Graph decoder implementation
"""

import numpy as np
import torch
import os
import sys
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from max_whisper.whisper_max import WhisperMAX
    MAX_WHISPER_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'max-whisper'))
        from whisper_max import WhisperMAX
        MAX_WHISPER_AVAILABLE = True
    except ImportError:
        MAX_WHISPER_AVAILABLE = False

try:
    import whisper
    from whisper.decoding import DecodingOptions
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def test_model_size(model_size, audio_file="audio_samples/modular_video.wav"):
    """Test MAX Graph encoder with specific model size"""
    
    print(f"\n🎯 TESTING MAX GRAPH WITH WHISPER {model_size.upper()}")
    print("=" * 60)
    
    if not MAX_WHISPER_AVAILABLE or not WHISPER_AVAILABLE:
        return {"error": "Required libraries not available"}
    
    try:
        # Initialize MAX Graph Whisper with specified model size
        print(f"🚀 Initializing MAX Graph Whisper {model_size}...")
        max_whisper = WhisperMAX(model_size=model_size, use_gpu=True)
        
        if not max_whisper.available:
            return {"error": f"MAX Graph Whisper {model_size} not available"}
        
        # Load audio
        import librosa
        audio, sr = librosa.load(audio_file, sr=16000)
        
        print(f"🎵 Audio: {len(audio)/sr:.1f}s")
        
        # Get baseline for comparison
        baseline_length = 2035  # Known from tiny model baseline
        print(f"📊 Target: {baseline_length} characters (100%)")
        
        # Test transcription
        print(f"\n🔢 Running MAX Graph {model_size} transcription...")
        start_time = time.time()
        
        result = max_whisper.transcribe(audio)
        
        total_time = time.time() - start_time
        
        # Extract results
        text = result['text'].strip() if 'text' in result else "No text"
        length = len(text)
        percentage = (length / baseline_length) * 100
        
        # Detailed analysis
        result_data = {
            "model_size": model_size,
            "timestamp": datetime.now().isoformat(),
            "audio_duration": len(audio)/sr,
            "total_time": total_time,
            "text_length": length,
            "baseline_length": baseline_length,
            "percentage": percentage,
            "text": text,
            "success": percentage > 80,  # 80% threshold for "success"
            "improvement_over_tiny": None,  # Will be calculated later
            "encoder_time": getattr(result, 'encoder_time', None),
            "decoder_time": getattr(result, 'decoder_time', None)
        }
        
        # Display results
        status = "✅" if result_data["success"] else "⚠️" if percentage > 50 else "❌"
        print(f"\n{status} RESULTS:")
        print(f"   Model: Whisper {model_size}")
        print(f"   Length: {length} characters ({percentage:.1f}% of baseline)")
        print(f"   Time: {total_time:.2f}s")
        print(f"   Text preview: '{text[:120]}...'")
        
        return result_data
        
    except Exception as e:
        error_data = {
            "model_size": model_size,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(f"❌ {model_size.upper()} test failed: {e}")
        return error_data


def compare_with_tiny_baseline():
    """Compare results with known tiny model baseline"""
    
    # Known tiny model result
    tiny_baseline = {
        "model_size": "tiny",
        "text_length": 838,
        "percentage": 41.2,
        "total_time": 1.9,
        "note": "Baseline from previous testing"
    }
    
    return tiny_baseline


def save_results(results, filename="larger_model_test_results.json"):
    """Save test results to JSON file"""
    
    results_file = os.path.join("benchmarks", filename)
    
    # Load existing results if they exist
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        except:
            existing_results = []
    else:
        existing_results = []
    
    # Add new results
    existing_results.extend(results)
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"📁 Results saved to {results_file}")


def main():
    """Main function to test multiple model sizes"""
    
    if not os.path.exists("audio_samples/modular_video.wav"):
        print("❌ Audio file not found: audio_samples/modular_video.wav")
        return
    
    print("🎯 LARGER MODEL ROBUSTNESS TEST")
    print("=" * 60)
    print("Testing if larger Whisper models are more robust to MAX Graph feature differences")
    
    # Get tiny baseline for comparison
    tiny_baseline = compare_with_tiny_baseline()
    print(f"\n📊 TINY MODEL BASELINE:")
    print(f"   Length: {tiny_baseline['text_length']} chars ({tiny_baseline['percentage']}%)")
    print(f"   Time: {tiny_baseline['total_time']}s")
    
    # Test model sizes in order of preference
    test_models = ["small"]  # Start with small as requested
    # test_models = ["small", "base"]  # Uncomment to test base as well
    
    results = []
    
    for model_size in test_models:
        result = test_model_size(model_size)
        results.append(result)
        
        # Calculate improvement over tiny
        if "error" not in result:
            improvement = result["percentage"] - tiny_baseline["percentage"]
            result["improvement_over_tiny"] = improvement
            
            print(f"\n📈 COMPARISON WITH TINY:")
            print(f"   {model_size.upper()}: {result['percentage']:.1f}% vs TINY: {tiny_baseline['percentage']:.1f}%")
            print(f"   Improvement: {improvement:+.1f} percentage points")
            
            if improvement > 20:
                print(f"   ✅ SIGNIFICANT IMPROVEMENT! {model_size.upper()} shows {improvement:+.1f}% improvement")
            elif improvement > 5:
                print(f"   ⚠️ Moderate improvement: {improvement:+.1f}%")
            else:
                print(f"   ❌ Limited improvement: {improvement:+.1f}%")
    
    # Save all results
    all_results = [tiny_baseline] + results
    save_results(all_results)
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATION:")
    
    best_result = max([r for r in results if "error" not in r], 
                     key=lambda x: x.get("percentage", 0), default=None)
    
    if best_result and best_result["success"]:
        print(f"✅ SUCCESS: {best_result['model_size'].upper()} achieves {best_result['percentage']:.1f}% baseline length")
        print(f"   Recommendation: Use {best_result['model_size']} model for production")
        print(f"   This solves the decoder stopping issue!")
    elif best_result and best_result["percentage"] > tiny_baseline["percentage"] + 10:
        print(f"⚠️ IMPROVEMENT: {best_result['model_size'].upper()} achieves {best_result['percentage']:.1f}% ({best_result['improvement_over_tiny']:+.1f}%)")
        print(f"   Consider testing 'base' model for further improvement")
    else:
        print(f"❌ LIMITED SUCCESS: Larger models don't solve the decoder stopping issue")
        print(f"   Recommendation: Proceed with full MAX Graph decoder implementation (Phase 4)")
        print(f"   Expected timeline: 2-3 weeks for complete solution")


if __name__ == "__main__":
    main()