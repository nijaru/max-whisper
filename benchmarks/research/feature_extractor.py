#!/usr/bin/env python3
"""
Feature extraction tool for analyzing encoder outputs across implementations.
"""

import json
import numpy as np
import torch
import argparse
from pathlib import Path
import sys
import os

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "max-whisper"))

# Import from max-whisper directory (note the dash, not underscore)
import whisper_cpu
import whisper_gpu  
import whisper_max

def extract_cpu_features(audio_path, model_size="tiny"):
    """Extract encoder features from CPU implementation."""
    print(f"🔍 Extracting features from CPU implementation...")
    
    try:
        whisper = whisper_cpu.WhisperCPU(model_size)
        # Load audio and extract mel features
        mel_features = whisper_cpu._load_and_preprocess_audio(audio_path)
        
        # Get the actual encoder from the model
        model = whisper_cpu._load_model(model_size)
        encoder = model.encoder
        
        # Extract encoder features
        with torch.no_grad():
            # Convert mel features to tensor if needed
            if isinstance(mel_features, np.ndarray):
                mel_tensor = torch.from_numpy(mel_features).float()
            else:
                mel_tensor = mel_features
            
            # Ensure proper shape (batch_size, n_mels, n_frames)
            if len(mel_tensor.shape) == 2:
                mel_tensor = mel_tensor.unsqueeze(0)
            
            # Run through encoder
            encoder_output = encoder(mel_tensor)
            features = encoder_output.cpu().numpy()
            
        print(f"    ✅ CPU features shape: {features.shape}")
        print(f"    📊 CPU features - mean: {features.mean():.6f}, std: {features.std():.6f}")
        
        return {
            'features': features,
            'shape': features.shape,
            'mean': float(features.mean()),
            'std': float(features.std()),
            'min': float(features.min()),
            'max': float(features.max())
        }
        
    except Exception as e:
        print(f"    ❌ CPU extraction failed: {e}")
        return None

def extract_gpu_features(audio_path, model_size="tiny"):
    """Extract encoder features from GPU implementation."""
    print(f"🔍 Extracting features from GPU implementation...")
    
    try:
        whisper_gpu = WhisperGPU()
        # Load audio and extract mel features
        mel_features = whisper_gpu._load_and_preprocess_audio(audio_path)
        
        # Get the actual encoder from the model
        model = whisper_gpu._load_model(model_size)
        encoder = model.encoder.cuda()
        
        # Extract encoder features
        with torch.no_grad():
            # Convert mel features to tensor if needed
            if isinstance(mel_features, np.ndarray):
                mel_tensor = torch.from_numpy(mel_features).float().cuda()
            else:
                mel_tensor = mel_features.cuda()
            
            # Ensure proper shape (batch_size, n_mels, n_frames)
            if len(mel_tensor.shape) == 2:
                mel_tensor = mel_tensor.unsqueeze(0)
            
            # Run through encoder
            encoder_output = encoder(mel_tensor)
            features = encoder_output.cpu().numpy()
            
        print(f"    ✅ GPU features shape: {features.shape}")
        print(f"    📊 GPU features - mean: {features.mean():.6f}, std: {features.std():.6f}")
        
        return {
            'features': features,
            'shape': features.shape,
            'mean': float(features.mean()),
            'std': float(features.std()),
            'min': float(features.min()),
            'max': float(features.max())
        }
        
    except Exception as e:
        print(f"    ❌ GPU extraction failed: {e}")
        return None

def extract_max_features(audio_path, model_size="tiny"):
    """Extract encoder features from MAX Graph implementation."""
    print(f"🔍 Extracting features from MAX Graph implementation...")
    
    try:
        whisper_max = WhisperMAX()
        
        # Load and preprocess audio using MAX implementation
        mel_features = whisper_max._load_and_preprocess_audio(audio_path)
        
        # Get encoder features directly from MAX Graph encoder
        max_encoder_features = whisper_max._run_max_encoder(mel_features)
        
        print(f"    ✅ MAX features shape: {max_encoder_features.shape}")
        print(f"    📊 MAX features - mean: {max_encoder_features.mean():.6f}, std: {max_encoder_features.std():.6f}")
        
        return {
            'features': max_encoder_features,
            'shape': max_encoder_features.shape,
            'mean': float(max_encoder_features.mean()),
            'std': float(max_encoder_features.std()),
            'min': float(max_encoder_features.min()),
            'max': float(max_encoder_features.max())
        }
        
    except Exception as e:
        print(f"    ❌ MAX extraction failed: {e}")
        return None

def compare_features(cpu_features, gpu_features, max_features):
    """Compare features between implementations."""
    print(f"\n🔍 Comparing features between implementations...")
    
    if not all([cpu_features, gpu_features, max_features]):
        print("    ❌ Cannot compare - some extractions failed")
        return None
    
    # Extract numpy arrays
    cpu_arr = cpu_features['features']
    gpu_arr = gpu_features['features'] 
    max_arr = max_features['features']
    
    # Ensure same shape for comparison
    print(f"    📊 Shapes - CPU: {cpu_arr.shape}, GPU: {gpu_arr.shape}, MAX: {max_arr.shape}")
    
    # Compare CPU vs GPU (should be nearly identical)
    cpu_gpu_diff = np.abs(cpu_arr - gpu_arr)
    cpu_gpu_l2 = np.linalg.norm(cpu_gpu_diff)
    cpu_gpu_cosine = np.dot(cpu_arr.flatten(), gpu_arr.flatten()) / (
        np.linalg.norm(cpu_arr.flatten()) * np.linalg.norm(gpu_arr.flatten())
    )
    
    # Compare CPU vs MAX (this is where we expect differences)
    cpu_max_diff = np.abs(cpu_arr - max_arr)
    cpu_max_l2 = np.linalg.norm(cpu_max_diff)
    cpu_max_cosine = np.dot(cpu_arr.flatten(), max_arr.flatten()) / (
        np.linalg.norm(cpu_arr.flatten()) * np.linalg.norm(max_arr.flatten())
    )
    
    # Compare GPU vs MAX
    gpu_max_diff = np.abs(gpu_arr - max_arr)
    gpu_max_l2 = np.linalg.norm(gpu_max_diff)
    gpu_max_cosine = np.dot(gpu_arr.flatten(), max_arr.flatten()) / (
        np.linalg.norm(gpu_arr.flatten()) * np.linalg.norm(max_arr.flatten())
    )
    
    comparison = {
        'cpu_gpu': {
            'l2_norm': float(cpu_gpu_l2),
            'cosine_similarity': float(cpu_gpu_cosine),
            'mean_abs_diff': float(cpu_gpu_diff.mean()),
            'max_abs_diff': float(cpu_gpu_diff.max())
        },
        'cpu_max': {
            'l2_norm': float(cpu_max_l2),
            'cosine_similarity': float(cpu_max_cosine),
            'mean_abs_diff': float(cpu_max_diff.mean()),
            'max_abs_diff': float(cpu_max_diff.max())
        },
        'gpu_max': {
            'l2_norm': float(gpu_max_l2),
            'cosine_similarity': float(gpu_max_cosine),
            'mean_abs_diff': float(gpu_max_diff.mean()),
            'max_abs_diff': float(gpu_max_diff.max())
        }
    }
    
    print(f"    🔍 CPU vs GPU - L2: {cpu_gpu_l2:.6f}, Cosine: {cpu_gpu_cosine:.6f}")
    print(f"    🔍 CPU vs MAX - L2: {cpu_max_l2:.6f}, Cosine: {cpu_max_cosine:.6f}")
    print(f"    🔍 GPU vs MAX - L2: {gpu_max_l2:.6f}, Cosine: {gpu_max_cosine:.6f}")
    
    return comparison

def analyze_layer_by_layer(audio_path, model_size="tiny"):
    """Analyze differences layer by layer (future enhancement)."""
    # TODO: Implement layer-by-layer analysis
    # This would require modifying the implementations to output intermediate results
    print(f"🔍 Layer-by-layer analysis not yet implemented")
    return None

def main():
    parser = argparse.ArgumentParser(description="Extract and compare encoder features")
    parser.add_argument("--audio", default="audio_samples/modular_video.wav", help="Audio file path")
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "small", "base"], help="Model size")
    parser.add_argument("--output", default="feature_comparison.json", help="Output JSON file")
    parser.add_argument("--implementations", nargs="+", default=["cpu", "gpu", "max"], 
                       choices=["cpu", "gpu", "max"], help="Implementations to compare")
    
    args = parser.parse_args()
    
    print(f"🎯 Feature extraction and comparison")
    print(f"    Audio: {args.audio}")
    print(f"    Model: {args.model_size}")
    print(f"    Implementations: {args.implementations}")
    
    results = {}
    
    # Extract features from each implementation
    if "cpu" in args.implementations:
        results['cpu'] = extract_cpu_features(args.audio, args.model_size)
    
    if "gpu" in args.implementations:
        results['gpu'] = extract_gpu_features(args.audio, args.model_size)
    
    if "max" in args.implementations:
        results['max'] = extract_max_features(args.audio, args.model_size)
    
    # Compare features
    if len(args.implementations) > 1:
        cpu_features = results.get('cpu')
        gpu_features = results.get('gpu')
        max_features = results.get('max')
        
        comparison = compare_features(cpu_features, gpu_features, max_features)
        if comparison:
            results['comparison'] = comparison
    
    # Save results
    # Convert numpy arrays to lists for JSON serialization
    for impl in results:
        if impl == 'comparison':
            continue
        if results[impl] and 'features' in results[impl]:
            # Save feature statistics but not the full array (too large)
            del results[impl]['features']
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()