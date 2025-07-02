#!/usr/bin/env python3
"""
Direct encoder feature extraction and debugging tool.
Focus on the numerical differences in encoder outputs.
"""

import json
import numpy as np
import torch
import argparse
import sys
from pathlib import Path

# Add max-whisper to path
project_root = Path(__file__).parent.parent
max_whisper_path = project_root / "max-whisper"
sys.path.insert(0, str(max_whisper_path))

def extract_max_encoder_features(audio_path="audio_samples/modular_video.wav"):
    """Extract encoder features directly from MAX Graph implementation."""
    print(f"🔍 Extracting MAX Graph encoder features...")
    
    try:
        import whisper_max
        
        # Create instance
        whisper = whisper_max.WhisperMAX()
        if not whisper.available:
            print(f"    ❌ MAX Graph not available")
            return None
        
        # Load audio and extract mel features using the same method as the implementation
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"    ✅ Audio loaded: {len(audio)/sr:.1f}s")
        
        # Extract mel features using librosa (same as implementation)
        mel_features = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        mel_db = librosa.power_to_db(mel_features, ref=np.max)
        print(f"    ✅ Mel features shape: {mel_db.shape}")
        
        # Run MAX Graph encoder
        max_features = whisper._encode_with_max_graph(mel_db)
        print(f"    ✅ MAX Graph features shape: {max_features.shape}")
        
        # Calculate statistics
        stats = {
            'shape': max_features.shape,
            'mean': float(np.mean(max_features)),
            'std': float(np.std(max_features)),
            'min': float(np.min(max_features)),
            'max': float(np.max(max_features)),
            'first_10_values': max_features.flatten()[:10].tolist()
        }
        
        print(f"    📊 MAX features - mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
        print(f"    📊 MAX features - range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        return {'features': max_features, 'stats': stats}
        
    except Exception as e:
        print(f"    ❌ MAX extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_openai_encoder_features(audio_path="audio_samples/modular_video.wav"):
    """Extract encoder features from OpenAI Whisper directly."""
    print(f"🔍 Extracting OpenAI encoder features...")
    
    try:
        import whisper
        
        # Load model
        model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Load audio
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"    ✅ Audio loaded: {len(audio)/sr:.1f}s")
        
        # Extract mel features using Whisper's method
        mel_features = whisper.log_mel_spectrogram(audio)
        print(f"    ✅ Mel features shape: {mel_features.shape}")
        
        # Run encoder
        with torch.no_grad():
            # Ensure proper shape and device
            if len(mel_features.shape) == 2:
                mel_features = mel_features.unsqueeze(0)  # Add batch dimension
            
            device = next(model.encoder.parameters()).device
            mel_features = mel_features.to(device)
            
            encoder_output = model.encoder(mel_features)
            features = encoder_output.cpu().numpy()
        
        print(f"    ✅ OpenAI features shape: {features.shape}")
        
        # Calculate statistics  
        stats = {
            'shape': features.shape,
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'first_10_values': features.flatten()[:10].tolist()
        }
        
        print(f"    📊 OpenAI features - mean: {stats['mean']:.6f}, std: {stats['std']:.6f}")
        print(f"    📊 OpenAI features - range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        return {'features': features, 'stats': stats}
        
    except Exception as e:
        print(f"    ❌ OpenAI extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_encoder_features(max_result, openai_result):
    """Compare encoder features numerically."""
    print(f"\n🔍 Comparing encoder features...")
    
    if not max_result or not openai_result:
        print(f"    ❌ Cannot compare - extraction failed")
        return None
    
    max_features = max_result['features']
    openai_features = openai_result['features']
    
    # Ensure same shape
    print(f"    📊 Shapes - MAX: {max_features.shape}, OpenAI: {openai_features.shape}")
    
    if max_features.shape != openai_features.shape:
        print(f"    ⚠️  Shape mismatch - cannot do direct comparison")
        return {
            'shape_match': False,
            'max_stats': max_result['stats'],
            'openai_stats': openai_result['stats']
        }
    
    # Calculate differences
    diff = np.abs(max_features - openai_features)
    l2_norm = np.linalg.norm(diff)
    
    # Cosine similarity
    max_flat = max_features.flatten()
    openai_flat = openai_features.flatten()
    cosine_sim = np.dot(max_flat, openai_flat) / (
        np.linalg.norm(max_flat) * np.linalg.norm(openai_flat)
    )
    
    comparison = {
        'shape_match': True,
        'l2_norm': float(l2_norm),
        'cosine_similarity': float(cosine_sim),
        'mean_abs_diff': float(np.mean(diff)),
        'max_abs_diff': float(np.max(diff)),
        'max_stats': max_result['stats'],
        'openai_stats': openai_result['stats']
    }
    
    print(f"    🔍 L2 norm difference: {l2_norm:.6f}")
    print(f"    🔍 Cosine similarity: {cosine_sim:.6f}")
    print(f"    🔍 Mean absolute difference: {np.mean(diff):.6f}")
    print(f"    🔍 Max absolute difference: {np.max(diff):.6f}")
    
    # Analyze the difference pattern
    mean_diff = np.mean(max_features) - np.mean(openai_features)
    std_ratio = np.std(max_features) / np.std(openai_features)
    
    print(f"    🔍 Mean shift: {mean_diff:.6f}")
    print(f"    🔍 Std ratio: {std_ratio:.6f}")
    
    comparison['mean_shift'] = float(mean_diff)
    comparison['std_ratio'] = float(std_ratio)
    
    return comparison

def analyze_first_values(max_result, openai_result):
    """Analyze the first few values to understand the pattern."""
    print(f"\n🔍 Analyzing first 10 values...")
    
    if not max_result or not openai_result:
        return None
    
    max_vals = max_result['stats']['first_10_values']
    openai_vals = openai_result['stats']['first_10_values']
    
    print(f"    📊 MAX first 10:    {[f'{v:.6f}' for v in max_vals]}")
    print(f"    📊 OpenAI first 10: {[f'{v:.6f}' for v in openai_vals]}")
    
    # Calculate element-wise differences
    diffs = [m - o for m, o in zip(max_vals, openai_vals)]
    print(f"    📊 Differences:     {[f'{d:.6f}' for d in diffs]}")
    
    return {
        'max_first_10': max_vals,
        'openai_first_10': openai_vals,
        'differences': diffs
    }

def main():
    parser = argparse.ArgumentParser(description="Debug encoder features")
    parser.add_argument("--audio", default="audio_samples/modular_video.wav", help="Audio file")
    parser.add_argument("--output", default="encoder_debug.json", help="Output file")
    
    args = parser.parse_args()
    
    print(f"🎯 Encoder Feature Debugging")
    print(f"    Audio: {args.audio}")
    
    results = {}
    
    # Extract features
    max_result = extract_max_encoder_features(args.audio)
    openai_result = extract_openai_encoder_features(args.audio)
    
    results['max'] = max_result
    results['openai'] = openai_result
    
    # Compare features
    if max_result and openai_result:
        comparison = compare_encoder_features(max_result, openai_result)
        first_values = analyze_first_values(max_result, openai_result)
        
        results['comparison'] = comparison
        results['first_values_analysis'] = first_values
    
    # Remove large arrays before saving
    if results.get('max') and 'features' in results['max']:
        del results['max']['features']
    if results.get('openai') and 'features' in results['openai']:
        del results['openai']['features']
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")
    
    # Print key findings
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\n🔍 Key Findings:")
        print(f"    Shape match: {comp.get('shape_match', False)}")
        print(f"    Cosine similarity: {comp.get('cosine_similarity', 'N/A'):.6f}")
        print(f"    Mean shift: {comp.get('mean_shift', 'N/A'):.6f}")
        print(f"    Std ratio: {comp.get('std_ratio', 'N/A'):.6f}")
        
        # Diagnosis
        cosine = comp.get('cosine_similarity', 0)
        mean_shift = abs(comp.get('mean_shift', 0))
        std_ratio = comp.get('std_ratio', 1)
        
        print(f"\n🔍 Diagnosis:")
        if cosine < 0.9:
            print(f"    ⚠️  Low cosine similarity ({cosine:.3f}) - structural differences")
        if mean_shift > 0.1:
            print(f"    ⚠️  Large mean shift ({mean_shift:.3f}) - bias/offset issue")  
        if std_ratio > 2 or std_ratio < 0.5:
            print(f"    ⚠️  Scale mismatch (ratio {std_ratio:.3f}) - normalization issue")

if __name__ == "__main__":
    main()