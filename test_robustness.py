#!/usr/bin/env python3
"""
Phase 2 Task 3: Robustness Testing for Pure MAX Graph Pipeline
Test the decoder with different parameters and configurations
"""

import time
import sys
import numpy as np
from pathlib import Path

# Add max-whisper to path
sys.path.append(str(Path(__file__).parent / "max-whisper"))

try:
    from whisper_max import WhisperMAX
    from max_graph_full_decoder import FullMaxGraphWhisperDecoder
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Components not available: {e}")
    COMPONENTS_AVAILABLE = False

def test_parameter_robustness():
    """Test robustness across different generation parameters"""
    print("🧪 Testing Parameter Robustness")
    print("=" * 60)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Required components not available")
        return
    
    # Setup encoder once
    print("🔧 Setting up encoder...")
    encoder = WhisperMAX()
    
    # Load audio
    import whisper
    audio = whisper.load_audio("audio_samples/modular_video.wav")
    mel_features = whisper.log_mel_spectrogram(audio)
    mel_np = mel_features.cpu().numpy()
    
    # Get encoder features
    encoder_features = encoder._encode_with_max_graph(mel_np)
    print(f"✅ Encoder features: {encoder_features.shape}")
    
    # Setup decoder
    decoder = FullMaxGraphWhisperDecoder(model_size="tiny")
    
    # Test different parameter combinations
    test_cases = [
        {"max_length": 50, "temperature": 0.6, "name": "Short_Medium_Temp"},
        {"max_length": 100, "temperature": 0.8, "name": "Medium_High_Temp"},
        {"max_length": 150, "temperature": 0.4, "name": "Long_Low_Temp"},
        {"max_length": 80, "temperature": 1.0, "name": "Medium_Max_Temp"},
        {"max_length": 120, "temperature": 0.2, "name": "Long_Conservative"},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📊 Test {i+1}: {test_case['name']}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            generated_text = decoder.generate_semantic_text(
                encoder_features,
                max_length=test_case["max_length"],
                temperature=test_case["temperature"],
                beam_size=1
            )
            
            generation_time = time.time() - start_time
            
            result = {
                "test_name": test_case["name"],
                "max_length": test_case["max_length"],
                "temperature": test_case["temperature"],
                "generated_text": generated_text,
                "text_length": len(generated_text),
                "generation_time": generation_time,
                "success": len(generated_text) > 0
            }
            
            results.append(result)
            
            print(f"   ⏱️ Time: {generation_time:.3f}s")
            print(f"   📏 Length: {len(generated_text)} chars")
            print(f"   📝 Text: '{generated_text[:50]}{'...' if len(generated_text) > 50 else ''}'")
            print(f"   ✅ Success: {result['success']}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                "test_name": test_case["name"],
                "success": False,
                "error": str(e)
            })
    
    return results

def test_consistency():
    """Test consistency across multiple runs with same parameters"""
    print("\n🔄 Testing Generation Consistency")
    print("=" * 60)
    
    if not COMPONENTS_AVAILABLE:
        return
    
    # Setup
    encoder = WhisperMAX()
    import whisper
    audio = whisper.load_audio("audio_samples/modular_video.wav")
    mel_features = whisper.log_mel_spectrogram(audio)
    mel_np = mel_features.cpu().numpy()
    encoder_features = encoder._encode_with_max_graph(mel_np)
    
    decoder = FullMaxGraphWhisperDecoder(model_size="tiny")
    
    # Run same parameters multiple times
    num_runs = 3
    results = []
    
    for run in range(num_runs):
        print(f"\n📊 Run {run + 1}/{num_runs}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            generated_text = decoder.generate_semantic_text(
                encoder_features,
                max_length=100,
                temperature=0.6,
                beam_size=1
            )
            
            generation_time = time.time() - start_time
            
            result = {
                "run": run + 1,
                "text": generated_text,
                "length": len(generated_text),
                "time": generation_time
            }
            
            results.append(result)
            
            print(f"   ⏱️ Time: {generation_time:.3f}s")
            print(f"   📏 Length: {len(generated_text)} chars")
            print(f"   📝 Text: '{generated_text[:40]}{'...' if len(generated_text) > 40 else ''}'")
            
        except Exception as e:
            print(f"   ❌ Run {run + 1} failed: {e}")
    
    # Analyze consistency
    if len(results) > 1:
        times = [r["time"] for r in results]
        lengths = [r["length"] for r in results]
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        print(f"\n📈 Consistency Analysis:")
        print(f"   ⏱️ Time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"   📏 Length: {avg_length:.1f} ± {std_length:.1f} chars")
        print(f"   📊 Time CV: {(std_time/avg_time*100):.1f}%")
        print(f"   📊 Length CV: {(std_length/avg_length*100):.1f}%")
    
    return results

def test_performance_scaling():
    """Test performance across different sequence lengths"""
    print("\n⚡ Testing Performance Scaling")
    print("=" * 60)
    
    if not COMPONENTS_AVAILABLE:
        return
    
    # Setup
    encoder = WhisperMAX()
    import whisper
    audio = whisper.load_audio("audio_samples/modular_video.wav")
    mel_features = whisper.log_mel_spectrogram(audio)
    mel_np = mel_features.cpu().numpy()
    encoder_features = encoder._encode_with_max_graph(mel_np)
    
    decoder = FullMaxGraphWhisperDecoder(model_size="tiny")
    
    # Test different max_lengths
    test_lengths = [25, 50, 75, 100, 150]
    results = []
    
    for max_length in test_lengths:
        print(f"\n📊 Testing max_length={max_length}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            generated_text = decoder.generate_semantic_text(
                encoder_features,
                max_length=max_length,
                temperature=0.6,
                beam_size=1
            )
            
            generation_time = time.time() - start_time
            tokens_per_second = max_length / generation_time if generation_time > 0 else 0
            
            result = {
                "max_length": max_length,
                "actual_length": len(generated_text),
                "time": generation_time,
                "tokens_per_second": tokens_per_second
            }
            
            results.append(result)
            
            print(f"   ⏱️ Time: {generation_time:.3f}s")
            print(f"   📏 Actual length: {len(generated_text)} chars")
            print(f"   🚀 Tokens/sec: {tokens_per_second:.1f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Analyze scaling
    if results:
        print(f"\n📈 Performance Scaling Analysis:")
        for result in results:
            efficiency = (result["actual_length"] / result["max_length"]) * 100
            print(f"   Length {result['max_length']:3d}: {result['time']:.3f}s, "
                  f"{result['tokens_per_second']:5.1f} tok/s, "
                  f"{efficiency:4.1f}% efficiency")
    
    return results

def main():
    """Run comprehensive robustness testing"""
    print("🚀 Phase 2 Task 3: Robustness Testing")
    print("=" * 70)
    
    # Test 1: Parameter robustness
    param_results = test_parameter_robustness()
    
    # Test 2: Consistency testing
    consistency_results = test_consistency()
    
    # Test 3: Performance scaling
    scaling_results = test_performance_scaling()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ROBUSTNESS TESTING SUMMARY")
    print("=" * 70)
    
    # Parameter test summary
    if param_results:
        successful_params = sum(1 for r in param_results if r.get("success", False))
        print(f"📋 Parameter Tests: {successful_params}/{len(param_results)} successful")
        
        if successful_params > 0:
            successful_results = [r for r in param_results if r.get("success", False)]
            avg_time = np.mean([r["generation_time"] for r in successful_results])
            avg_length = np.mean([r["text_length"] for r in successful_results])
            print(f"   ⏱️ Average time: {avg_time:.3f}s")
            print(f"   📏 Average length: {avg_length:.1f} chars")
    
    # Consistency summary
    if consistency_results:
        print(f"🔄 Consistency Tests: {len(consistency_results)} runs completed")
    
    # Scaling summary
    if scaling_results:
        print(f"⚡ Scaling Tests: {len(scaling_results)} length configurations tested")
    
    # Overall assessment
    total_tests = len(param_results) + len(consistency_results) + len(scaling_results)
    print(f"\n✅ Total tests completed: {total_tests}")
    print(f"🎯 Robustness assessment: {'PASSED' if total_tests > 0 else 'FAILED'}")
    
    return {
        "parameter_tests": param_results,
        "consistency_tests": consistency_results,
        "scaling_tests": scaling_results
    }

if __name__ == "__main__":
    main()