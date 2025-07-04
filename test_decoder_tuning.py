#!/usr/bin/env python3
"""
Test different decoder parameters to reduce repetition
Focus on temperature, top_p sampling, and repetition penalties
"""

import whisper
import numpy as np
import time
from pathlib import Path

def analyze_repetition_metrics(text):
    """Analyze repetition patterns in text"""
    words = text.split()
    if len(words) < 4:
        return {"repetition_score": 0, "unique_ratio": 1.0, "pattern_length": 0}
    
    # Find repeated n-grams
    repetitions = []
    for n in range(2, min(8, len(words)//2)):  # Check 2-7 word phrases
        for i in range(len(words) - 2*n + 1):
            ngram = " ".join(words[i:i+n])
            rest_text = " ".join(words[i+n:])
            if ngram in rest_text:
                repetitions.append((ngram, n, i))
    
    # Calculate metrics
    repetition_score = len(repetitions)
    unique_words = len(set(words))
    unique_ratio = unique_words / len(words) if words else 0
    
    longest_pattern = max((len(r[0]) for r in repetitions), default=0)
    
    return {
        "repetition_score": repetition_score,
        "unique_ratio": unique_ratio,
        "pattern_length": longest_pattern,
        "total_length": len(text),
        "word_count": len(words)
    }

def test_decode_parameters():
    """Test different decoding parameters to reduce repetition"""
    model = whisper.load_model("tiny")
    audio_file = "audio_samples/modular_video.wav"
    
    # Load and preprocess audio once
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    test_configs = [
        {
            "name": "Baseline (Greedy)",
            "params": {"temperature": 0.0, "beam_size": None, "best_of": None},
            "description": "Default greedy decoding"
        },
        {
            "name": "Low Temperature",
            "params": {"temperature": 0.2, "beam_size": None, "best_of": None},
            "description": "Slight randomness"
        },
        {
            "name": "Medium Temperature", 
            "params": {"temperature": 0.5, "beam_size": None, "best_of": None},
            "description": "Moderate randomness"
        },
        {
            "name": "High Temperature",
            "params": {"temperature": 0.8, "beam_size": None, "best_of": None},
            "description": "High randomness"
        },
        {
            "name": "Beam Search",
            "params": {"temperature": 0.0, "beam_size": 5, "best_of": None},
            "description": "Beam search with 5 beams"
        },
        {
            "name": "Best of 5",
            "params": {"temperature": 0.0, "beam_size": None, "best_of": 5},
            "description": "Best of 5 samples"
        },
        {
            "name": "Temp + Beam",
            "params": {"temperature": 0.3, "beam_size": 3, "best_of": None},
            "description": "Temperature with beam search"
        }
    ]
    
    results = []
    
    print("🚀 Testing Decoder Parameter Tuning")
    print("=" * 60)
    
    for config in test_configs:
        print(f"\n🔧 Testing: {config['name']}")
        print(f"   📝 {config['description']}")
        print(f"   ⚙️ Params: {config['params']}")
        
        try:
            start_time = time.time()
            
            # Transcribe with specific parameters
            result = whisper.decode(model, mel, **config['params'])
            
            elapsed = time.time() - start_time
            text = result.text.strip()
            
            # Analyze repetition
            metrics = analyze_repetition_metrics(text)
            
            result_data = {
                "config": config,
                "text": text,
                "metrics": metrics,
                "time": elapsed
            }
            results.append(result_data)
            
            print(f"   ⏱️ Time: {elapsed:.2f}s")
            print(f"   📏 Length: {metrics['total_length']} chars, {metrics['word_count']} words")
            print(f"   🔄 Repetition score: {metrics['repetition_score']}")
            print(f"   📊 Unique ratio: {metrics['unique_ratio']:.3f}")
            print(f"   📝 Text preview: {text[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            result_data = {
                "config": config,
                "text": "",
                "metrics": {"error": str(e)},
                "time": 0
            }
            results.append(result_data)
    
    # Summary
    print(f"\n📊 Summary of Results")
    print("=" * 60)
    print(f"{'Config':<20} {'Length':<8} {'Unique':<8} {'Repetition':<12} {'Time':<8}")
    print("-" * 60)
    
    for result in results:
        if "error" not in result["metrics"]:
            config_name = result["config"]["name"][:19]
            length = result["metrics"]["total_length"]
            unique = result["metrics"]["unique_ratio"]
            repetition = result["metrics"]["repetition_score"]
            time_taken = result["time"]
            
            print(f"{config_name:<20} {length:<8} {unique:<8.3f} {repetition:<12} {time_taken:<8.2f}")
    
    # Find best configuration
    valid_results = [r for r in results if "error" not in r["metrics"]]
    if valid_results:
        # Best = lowest repetition score with reasonable length
        best_result = min(valid_results, 
                         key=lambda x: (x["metrics"]["repetition_score"], 
                                       -x["metrics"]["total_length"]))
        
        print(f"\n🏆 Best Configuration: {best_result['config']['name']}")
        print(f"   📊 Metrics: {best_result['metrics']}")
        print(f"   📝 Full text: {best_result['text']}")
        
        return best_result
    
    return None

def test_repetition_penalties():
    """Test custom repetition penalty implementations"""
    print("\n🛠️ Testing Custom Repetition Penalties")
    print("=" * 60)
    
    # This would require modifying the Whisper decoder
    # For now, let's analyze what we would need to implement
    
    penalties = [
        {
            "name": "N-gram Penalty",
            "description": "Penalize repeated 2-4 word phrases",
            "implementation": "Track n-grams and reduce logits for repeats"
        },
        {
            "name": "Attention Diversity",
            "description": "Encourage attention to different encoder positions", 
            "implementation": "Modify attention weights to avoid same positions"
        },
        {
            "name": "Length Reward",
            "description": "Reward longer, more diverse outputs",
            "implementation": "Boost probabilities for continuing generation"
        }
    ]
    
    for penalty in penalties:
        print(f"   • {penalty['name']}: {penalty['description']}")
        print(f"     Implementation: {penalty['implementation']}")
    
    print(f"\n   ⚠️ Note: These require custom decoder modifications")
    return penalties

if __name__ == "__main__":
    print("🔍 Decoder Parameter Tuning for Repetition Reduction")
    print("=" * 60)
    
    # Test standard parameters
    best_config = test_decode_parameters()
    
    # Test custom penalties (analysis only)
    penalty_options = test_repetition_penalties()
    
    print(f"\n✅ Analysis complete. Best configuration identified.")
    if best_config:
        print(f"   🎯 Recommended: {best_config['config']['name']}")
        print(f"   📊 Parameters: {best_config['config']['params']}")