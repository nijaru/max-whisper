#!/usr/bin/env python3
"""
Direct Transcription Fix for MAX Graph Whisper
Address the fundamental issue: decoder stops at 259 chars regardless of parameters
Use direct feature post-processing to match decoder expectations
"""

import numpy as np
import torch
import os
import sys
import time

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


def apply_decoder_confidence_boost(features):
    """Apply confidence boost to features to prevent early stopping"""
    
    print("🔧 Applying decoder confidence boost...")
    
    # Method 1: Attention weight amplification
    # Boost the most confident features to help decoder continue
    feature_magnitudes = np.linalg.norm(features, axis=-1, keepdims=True)
    top_percentile = np.percentile(feature_magnitudes, 85)  # Top 15% of features
    boost_mask = feature_magnitudes >= top_percentile
    
    boosted_features = features.copy()
    boosted_features[boost_mask] *= 1.2  # 20% boost to confident features
    
    print(f"   ✅ Boosted {np.sum(boost_mask)} / {features.size} features")
    
    return boosted_features


def apply_feature_smoothing(features):
    """Apply temporal smoothing to reduce decoder confusion"""
    
    print("🔧 Applying temporal smoothing...")
    
    # Simple moving average across sequence dimension
    kernel_size = 3
    smoothed = features.copy()
    
    for i in range(kernel_size // 2, features.shape[1] - kernel_size // 2):
        smoothed[:, i, :] = np.mean(features[:, i-1:i+2, :], axis=1)
    
    print(f"   ✅ Applied {kernel_size}-point smoothing")
    
    return smoothed


def apply_variance_normalization(features, target_std=0.65):
    """Normalize feature variance to match expected decoder input"""
    
    print(f"🔧 Applying variance normalization (target std: {target_std})...")
    
    current_std = np.std(features)
    scaling_factor = target_std / current_std
    
    # Preserve mean while adjusting variance
    mean = np.mean(features)
    normalized = (features - mean) * scaling_factor + mean
    
    new_std = np.std(normalized)
    print(f"   ✅ Std: {current_std:.4f} → {new_std:.4f}")
    
    return normalized


def test_direct_feature_fixes(audio_file="audio_samples/modular_video.wav"):
    """Test direct feature modifications to achieve full transcription"""
    
    if not MAX_WHISPER_AVAILABLE or not WHISPER_AVAILABLE:
        print("❌ Required libraries not available")
        return
        
    print("🎯 DIRECT FEATURE FIXES FOR FULL TRANSCRIPTION")
    print("=" * 70)
    print("Goal: Fix decoder early stopping through feature post-processing")
    
    # Initialize MAX Graph Whisper
    max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
    if not max_whisper.available:
        print("❌ MAX Graph Whisper not available")
        return
    
    # Load audio
    import librosa
    audio, sr = librosa.load(audio_file, sr=16000)
    mel_db = whisper.log_mel_spectrogram(audio).numpy()
    
    print(f"🎵 Audio: {len(audio)/sr:.1f}s")
    
    # Get baseline length
    baseline_length = 2035  # Known length
    print(f"📊 Target: {baseline_length} characters")
    
    # Extract original MAX Graph features
    print("\n🔢 Extracting MAX Graph features...")
    original_features = max_whisper._encode_with_max_graph(mel_db)
    
    print(f"✅ Original features: {original_features.shape}")
    print(f"   Mean: {np.mean(original_features):.6f}, Std: {np.std(original_features):.6f}")
    
    # Test different feature modification approaches
    modification_methods = [
        {
            "name": "Original",
            "function": lambda x: x,
            "description": "No modification (baseline)"
        },
        {
            "name": "Confidence-Boost",
            "function": apply_decoder_confidence_boost,
            "description": "Amplify top 15% features by 20%"
        },
        {
            "name": "Temporal-Smoothing", 
            "function": apply_feature_smoothing,
            "description": "3-point moving average smoothing"
        },
        {
            "name": "Variance-Norm",
            "function": lambda x: apply_variance_normalization(x, 0.65),
            "description": "Normalize std to 0.65"
        },
        {
            "name": "Conservative-Norm",
            "function": lambda x: apply_variance_normalization(x, 0.85),
            "description": "Normalize std to 0.85 (conservative)"
        },
        {
            "name": "Combined-Fix",
            "function": lambda x: apply_variance_normalization(
                apply_decoder_confidence_boost(x), 0.75
            ),
            "description": "Confidence boost + variance normalization"
        }
    ]
    
    results = []
    
    for method in modification_methods:
        print(f"\n🧪 Testing {method['name']}:")
        print(f"   {method['description']}")
        
        try:
            # Apply modification
            if method['name'] == "Original":
                modified_features = original_features
            else:
                modified_features = method['function'](original_features.copy())
            
            # Test with decoder using optimal parameters
            result = test_decoder_with_features(
                modified_features, max_whisper.whisper_model, method['name'], baseline_length
            )
            
            results.append({
                'method': method['name'],
                'description': method['description'],
                'length': result['length'],
                'text': result['text'],
                'percentage': (result['length'] / baseline_length) * 100,
                'decode_time': result['decode_time']
            })
            
        except Exception as e:
            print(f"   ❌ {method['name']} failed: {e}")
            continue
    
    # Results analysis
    print(f"\n🏆 FEATURE MODIFICATION RESULTS:")
    print(f"Target: {baseline_length} characters (100%)")
    
    results.sort(key=lambda x: x['length'], reverse=True)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['percentage'] > 80 else "⚠️" if result['percentage'] > 50 else "❌"
        print(f"{i}. {status} {result['method']:18}: {result['length']:4d} chars ({result['percentage']:5.1f}%) - {result['description']}")
    
    # Best result analysis
    if results:
        best = results[0]
        original = next(r for r in results if r['method'] == 'Original')
        
        print(f"\n🔍 BEST RESULT ANALYSIS:")
        print(f"Method: {best['method']}")
        print(f"Improvement: {best['length']} vs {original['length']} chars ({best['percentage']:.1f}% vs {original['percentage']:.1f}%)")
        print(f"Description: {best['description']}")
        print(f"Time: {best['decode_time']:.2f}s")
        print(f"\nText preview:")
        print(f"   '{best['text'][:200]}...'")
        
        if best['percentage'] > 80:
            print(f"\n✅ SUCCESS! {best['method']} achieved {best['percentage']:.1f}% of baseline length")
            print("This method can be integrated into whisper_max.py")
            return best
        elif best['percentage'] > original['percentage'] * 1.5:
            print(f"\n⚠️ Significant improvement: {best['percentage']:.1f}% vs {original['percentage']:.1f}%")
            print("Consider combining this with other techniques")
        else:
            print(f"\n❌ Limited improvement over original ({best['percentage']:.1f}% vs {original['percentage']:.1f}%)")
            print("May need architectural changes beyond feature post-processing")
    
    return results


def test_decoder_with_features(features, model, method_name, baseline_length):
    """Test decoder with modified features using optimal parameters"""
    
    try:
        # Convert to tensor
        features_tensor = torch.from_numpy(features.copy()).float()
        device = next(model.parameters()).device
        features_tensor = features_tensor.to(device)
        
        # Use parameters optimized for MAX Graph features
        options = DecodingOptions(
            task="transcribe",
            language="en",
            temperature=0.0,
            sample_len=3000,      # Very generous length
            beam_size=10,         # Optimal from previous tuning
            patience=50.0,        # Very high patience
            without_timestamps=True,
            suppress_blank=True,
            suppress_tokens="-1"
        )
        
        start_time = time.time()
        result = model.decode(features_tensor, options)
        decode_time = time.time() - start_time
        
        # Extract text
        if isinstance(result, list) and len(result) > 0:
            text = result[0].text.strip()
        elif hasattr(result, 'text'):
            text = result.text.strip()
        else:
            text = "Decoder error"
        
        length = len(text)
        percentage = (length / baseline_length) * 100
        
        print(f"   📝 {method_name}: {length} chars ({percentage:.1f}%) in {decode_time:.2f}s")
        print(f"   📄 Preview: '{text[:80]}...'")
        
        return {
            'text': text,
            'length': length,
            'decode_time': decode_time
        }
        
    except Exception as e:
        print(f"   ❌ Decoder failed: {e}")
        return {
            'text': f"Error: {e}",
            'length': 0,
            'decode_time': 0
        }


def main():
    """Main function"""
    if not os.path.exists("audio_samples/modular_video.wav"):
        print("❌ Audio file not found: audio_samples/modular_video.wav")
        return
        
    results = test_direct_feature_fixes()
    
    if results:
        best = results[0]
        
        if best['percentage'] > 80:
            print(f"\n🎯 INTEGRATION READY:")
            print(f"The {best['method']} method achieves {best['percentage']:.1f}% baseline length")
            print("Can be integrated into _encode_with_max_graph() method")
        elif best['percentage'] > 50:
            print(f"\n🔬 PROMISING DIRECTION:")
            print(f"Best method ({best['method']}) achieves {best['percentage']:.1f}%")
            print("Consider refining this approach or combining with other methods")
        else:
            print(f"\n🤔 LIMITED SUCCESS:")
            print(f"Best improvement: {best['percentage']:.1f}%")
            print("May need fundamental decoder architecture investigation")
    else:
        print("\n❌ No successful methods found")


if __name__ == "__main__":
    main()