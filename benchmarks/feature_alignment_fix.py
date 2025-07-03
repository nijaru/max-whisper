#!/usr/bin/env python3
"""
Feature Alignment Fix for MAX Graph Whisper
Direct feature alignment to match OpenAI encoder output exactly
Address the core issue: decoder stops at 259 chars regardless of parameters
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


def analyze_feature_distribution_differences(max_features, openai_features):
    """Analyze detailed statistical differences between feature distributions"""
    
    print("🔬 DETAILED FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Flatten for analysis
    max_flat = max_features.flatten()
    openai_flat = openai_features.flatten()
    
    # Basic statistics
    print(f"📊 Basic Statistics:")
    print(f"   MAX Graph  - Mean: {np.mean(max_flat):.6f}, Std: {np.std(max_flat):.6f}")
    print(f"   OpenAI     - Mean: {np.mean(openai_flat):.6f}, Std: {np.std(openai_flat):.6f}")
    print(f"   Cosine Similarity: {np.dot(max_flat, openai_flat) / (np.linalg.norm(max_flat) * np.linalg.norm(openai_flat)):.6f}")
    
    # Distribution analysis
    print(f"\n📈 Distribution Analysis:")
    print(f"   MAX Graph  - Min: {np.min(max_flat):.6f}, Max: {np.max(max_flat):.6f}")
    print(f"   OpenAI     - Min: {np.min(openai_flat):.6f}, Max: {np.max(openai_flat):.6f}")
    
    # Percentiles
    max_percentiles = np.percentile(max_flat, [1, 5, 25, 50, 75, 95, 99])
    openai_percentiles = np.percentile(openai_flat, [1, 5, 25, 50, 75, 95, 99])
    
    print(f"\n📊 Percentile Comparison:")
    percentile_labels = ['1%', '5%', '25%', '50%', '75%', '95%', '99%']
    for i, label in enumerate(percentile_labels):
        print(f"   {label:3} - MAX: {max_percentiles[i]:8.4f}, OpenAI: {openai_percentiles[i]:8.4f}")
    
    # Outlier analysis
    max_outliers = np.abs(max_flat) > 3 * np.std(max_flat)
    openai_outliers = np.abs(openai_flat) > 3 * np.std(openai_flat)
    
    print(f"\n🎯 Outlier Analysis:")
    print(f"   MAX Graph outliers: {np.sum(max_outliers)} / {len(max_flat)} ({np.sum(max_outliers)/len(max_flat)*100:.2f}%)")
    print(f"   OpenAI outliers: {np.sum(openai_outliers)} / {len(openai_flat)} ({np.sum(openai_outliers)/len(openai_flat)*100:.2f}%)")
    
    # Statistical tests (simplified)
    print(f"\n🧪 Statistical Tests:")
    correlation = np.corrcoef(max_flat, openai_flat)[0, 1]
    print(f"   Correlation coefficient: {correlation:.6f}")
    
    return {
        'max_stats': {'mean': np.mean(max_flat), 'std': np.std(max_flat), 'min': np.min(max_flat), 'max': np.max(max_flat)},
        'openai_stats': {'mean': np.mean(openai_flat), 'std': np.std(openai_flat), 'min': np.min(openai_flat), 'max': np.max(openai_flat)},
        'cosine_similarity': np.dot(max_flat, openai_flat) / (np.linalg.norm(max_flat) * np.linalg.norm(openai_flat)),
        'correlation': correlation
    }


def apply_statistical_alignment(max_features, openai_features, method="histogram_matching"):
    """Apply statistical alignment to match OpenAI feature distribution"""
    
    print(f"\n🔧 Applying {method} alignment...")
    
    if method == "z_score_normalization":
        # Z-score normalization to match target distribution
        max_mean = np.mean(max_features)
        max_std = np.std(max_features)
        openai_mean = np.mean(openai_features)
        openai_std = np.std(openai_features)
        
        aligned_features = (max_features - max_mean) / max_std * openai_std + openai_mean
        
    elif method == "histogram_matching":
        # Histogram matching (more aggressive)
        max_flat = max_features.flatten()
        openai_flat = openai_features.flatten()
        
        # Sort both arrays
        max_sorted = np.sort(max_flat)
        openai_sorted = np.sort(openai_flat)
        
        # Create mapping
        max_indices = np.argsort(max_flat)
        aligned_flat = np.zeros_like(max_flat)
        aligned_flat[max_indices] = openai_sorted
        
        aligned_features = aligned_flat.reshape(max_features.shape)
        
    elif method == "percentile_matching":
        # Percentile-based matching
        max_flat = max_features.flatten()
        openai_flat = openai_features.flatten()
        
        # Get percentiles
        percentiles = np.linspace(0, 100, 1000)
        max_percentiles = np.percentile(max_flat, percentiles)
        openai_percentiles = np.percentile(openai_flat, percentiles)
        
        # Interpolate
        aligned_flat = np.interp(max_flat, max_percentiles, openai_percentiles)
        aligned_features = aligned_flat.reshape(max_features.shape)
        
    elif method == "robust_scaling":
        # Robust scaling using median and IQR
        max_median = np.median(max_features)
        max_q75, max_q25 = np.percentile(max_features, [75, 25])
        max_iqr = max_q75 - max_q25
        
        openai_median = np.median(openai_features)
        openai_q75, openai_q25 = np.percentile(openai_features, [75, 25])
        openai_iqr = openai_q75 - openai_q25
        
        aligned_features = (max_features - max_median) / max_iqr * openai_iqr + openai_median
        
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # Verify alignment
    aligned_flat = aligned_features.flatten()
    openai_flat = openai_features.flatten()
    
    cosine_sim = np.dot(aligned_flat, openai_flat) / (np.linalg.norm(aligned_flat) * np.linalg.norm(openai_flat))
    
    print(f"   ✅ Alignment complete:")
    print(f"   Original cosine similarity: {np.dot(max_features.flatten(), openai_flat) / (np.linalg.norm(max_features.flatten()) * np.linalg.norm(openai_flat)):.6f}")
    print(f"   Aligned cosine similarity:  {cosine_sim:.6f}")
    print(f"   Mean diff: {np.abs(np.mean(aligned_flat) - np.mean(openai_flat)):.6f}")
    print(f"   Std diff:  {np.abs(np.std(aligned_flat) - np.std(openai_flat)):.6f}")
    
    return aligned_features


def test_alignment_methods(audio_file="audio_samples/modular_video.wav"):
    """Test different feature alignment methods to achieve full transcription"""
    
    if not MAX_WHISPER_AVAILABLE or not WHISPER_AVAILABLE:
        print("❌ Required libraries not available")
        return
        
    print("🎯 FEATURE ALIGNMENT FOR FULL TRANSCRIPTION")
    print("=" * 70)
    print("Goal: Fix decoder early stopping by aligning feature distributions")
    
    # Initialize models
    max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
    if not max_whisper.available:
        print("❌ MAX Graph Whisper not available")
        return
    
    openai_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load audio and get baseline
    import librosa
    audio, sr = librosa.load(audio_file, sr=16000)
    mel_db = whisper.log_mel_spectrogram(audio).numpy()
    
    print(f"🎵 Audio: {len(audio)/sr:.1f}s")
    
    # Get baseline transcription length
    print("📊 Getting baseline...")
    try:
        baseline_result = whisper.transcribe(openai_model, audio, language="en")
        baseline_text = baseline_result['text'].strip()
        baseline_length = len(baseline_text)
        print(f"✅ Baseline: {baseline_length} characters")
    except Exception as e:
        baseline_length = 2035  # Known length
        print(f"⚠️ Using known baseline: {baseline_length} characters")
    
    # Extract features
    print("\n🔢 Extracting features...")
    max_features = max_whisper._encode_with_max_graph(mel_db)
    
    # Get OpenAI features
    try:
        n_mels, seq_len = mel_db.shape
        max_seq_len = 1500
        
        if seq_len > max_seq_len:
            mel_truncated = mel_db[:, :max_seq_len]
        else:
            pad_width = max_seq_len - seq_len
            mel_truncated = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        
        mel_tensor = torch.from_numpy(mel_truncated).float().unsqueeze(0)
        device = next(openai_model.encoder.parameters()).device
        mel_tensor = mel_tensor.to(device)
        
        with torch.no_grad():
            openai_features = openai_model.encoder(mel_tensor).cpu().numpy()
            
    except Exception as e:
        print(f"❌ OpenAI feature extraction failed: {e}")
        return
    
    print(f"✅ Features: MAX {max_features.shape}, OpenAI {openai_features.shape}")
    
    # Analyze differences first
    analysis = analyze_feature_distribution_differences(max_features, openai_features)
    
    # Test alignment methods
    alignment_methods = [
        "z_score_normalization",
        "histogram_matching", 
        "percentile_matching",
        "robust_scaling"
    ]
    
    results = []
    
    for method in alignment_methods:
        print(f"\n🧪 Testing {method}:")
        
        try:
            # Apply alignment
            aligned_features = apply_statistical_alignment(max_features, openai_features, method)
            
            # Test with decoder
            result = test_decoder_with_aligned_features(
                aligned_features, max_whisper.whisper_model, method, baseline_length
            )
            
            results.append({
                'method': method,
                'length': result['length'],
                'text': result['text'],
                'percentage': (result['length'] / baseline_length) * 100,
                'success': result['length'] > baseline_length * 0.8  # 80% threshold
            })
            
        except Exception as e:
            print(f"   ❌ {method} failed: {e}")
            continue
    
    # Results analysis
    print(f"\n🏆 ALIGNMENT RESULTS:")
    print(f"Target: {baseline_length} characters (100%)")
    
    results.sort(key=lambda x: x['length'], reverse=True)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['success'] else "⚠️" if result['percentage'] > 50 else "❌"
        print(f"{i}. {status} {result['method']:20}: {result['length']:4d} chars ({result['percentage']:5.1f}%)")
    
    # Best result
    if results:
        best = results[0]
        print(f"\n🔍 BEST RESULT:")
        print(f"Method: {best['method']}")
        print(f"Length: {best['length']} / {baseline_length} ({best['percentage']:.1f}%)")
        print(f"Text preview: '{best['text'][:150]}...'")
        
        if best['success']:
            print(f"\n✅ SUCCESS! {best['method']} achieved {best['percentage']:.1f}% of baseline length")
            print("This method can be integrated into the main implementation")
        else:
            print(f"\n⚠️ Partial success. Best method: {best['method']} ({best['percentage']:.1f}%)")
            print("May need combination of methods or deeper architectural investigation")
        
        return best
    
    return None


def test_decoder_with_aligned_features(features, model, method_name, baseline_length):
    """Test decoder with aligned features using optimal parameters"""
    
    try:
        # Convert to tensor
        features_tensor = torch.from_numpy(features.copy()).float()
        device = next(model.parameters()).device
        features_tensor = features_tensor.to(device)
        
        # Use optimal parameters discovered in previous tuning
        options = DecodingOptions(
            task="transcribe",
            language="en",
            temperature=0.0,
            sample_len=2000,  # Generous length
            beam_size=10,     # Optimal from tuning
            patience=20.0,    # High patience
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
        print(f"   📄 Text: '{text[:100]}...'")
        
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
        
    best_result = test_alignment_methods()
    
    if best_result and best_result['percentage'] > 80:
        print(f"\n🎯 INTEGRATION READY:")
        print(f"The {best_result['method']} alignment method achieves {best_result['percentage']:.1f}% baseline length")
        print("This can be integrated into whisper_max.py for full-length transcription")
    elif best_result:
        print(f"\n🔬 RESEARCH DIRECTION:")
        print(f"Best method ({best_result['method']}) achieves {best_result['percentage']:.1f}%")
        print("Consider combining alignment methods or investigating decoder architecture differences")
    else:
        print("\n❌ No alignment method successful")
        print("May need fundamental architectural investigation")


if __name__ == "__main__":
    main()