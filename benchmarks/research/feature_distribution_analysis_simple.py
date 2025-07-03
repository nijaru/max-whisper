#!/usr/bin/env python3
"""
Simple Feature Distribution Analysis for MAX Graph Whisper
Compare statistical differences between MAX Graph and OpenAI encoder features
"""

import numpy as np
import os
import sys

# Add parent directory to path to import our MAX Graph implementation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from max_whisper.whisper_max import WhisperMAX
    MAX_WHISPER_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'max-whisper'))
        from whisper_max import WhisperMAX
        MAX_WHISPER_AVAILABLE = True
    except ImportError:
        MAX_WHISPER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def analyze_feature_differences(audio_file="audio_samples/modular_video.wav"):
    """Simple analysis of MAX Graph vs OpenAI feature differences"""
    
    if not MAX_WHISPER_AVAILABLE or not WHISPER_AVAILABLE:
        print("❌ Required libraries not available")
        return
        
    print("🔬 Simple Feature Distribution Analysis")
    print("=" * 50)
    
    # Initialize MAX Graph Whisper
    max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
    if not max_whisper.available:
        print("❌ MAX Graph Whisper not available")
        return
    
    # Load OpenAI model
    openai_model = whisper.load_model("tiny", device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    
    # Load audio
    import librosa
    audio, sr = librosa.load(audio_file, sr=16000)
    mel_db = whisper.log_mel_spectrogram(audio).numpy()
    
    print(f"🎵 Audio loaded: {audio.shape}, Mel: {mel_db.shape}")
    
    # Get MAX Graph features
    print("🔢 Extracting MAX Graph encoder features...")
    max_features = max_whisper._encode_with_max_graph(mel_db)
    
    # Get OpenAI features  
    print("🔢 Extracting OpenAI encoder features...")
    try:
        n_mels, seq_len = mel_db.shape
        max_seq_len = 1500
        
        if seq_len > max_seq_len:
            mel_db_truncated = mel_db[:, :max_seq_len]
        else:
            pad_width = max_seq_len - seq_len
            mel_db_truncated = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        
        import torch
        mel_tensor = torch.from_numpy(mel_db_truncated).float().unsqueeze(0)
        device = next(openai_model.encoder.parameters()).device
        mel_tensor = mel_tensor.to(device)
        
        with torch.no_grad():
            openai_features = openai_model.encoder(mel_tensor).cpu().numpy()
            
    except Exception as e:
        print(f"❌ OpenAI feature extraction failed: {e}")
        return
    
    # Analyze differences
    print("\n📊 FEATURE ANALYSIS:")
    print(f"MAX Graph shape: {max_features.shape}")
    print(f"OpenAI shape: {openai_features.shape}")
    
    # Flatten for analysis
    max_flat = max_features.flatten()
    openai_flat = openai_features.flatten()
    
    # Basic statistics
    print(f"\n📈 STATISTICAL COMPARISON:")
    print(f"MAX Graph - mean: {np.mean(max_flat):.6f}, std: {np.std(max_flat):.6f}")
    print(f"OpenAI    - mean: {np.mean(openai_flat):.6f}, std: {np.std(openai_flat):.6f}")
    print(f"MAX Graph - range: [{np.min(max_flat):.3f}, {np.max(max_flat):.3f}]")
    print(f"OpenAI    - range: [{np.min(openai_flat):.3f}, {np.max(openai_flat):.3f}]")
    
    # Key differences
    mean_diff = np.mean(max_flat) - np.mean(openai_flat)
    std_ratio = np.std(max_flat) / np.std(openai_flat)
    cosine_sim = np.dot(max_flat, openai_flat) / (np.linalg.norm(max_flat) * np.linalg.norm(openai_flat))
    
    print(f"\n🔍 KEY DIFFERENCES:")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Std deviation ratio: {std_ratio:.3f}")
    print(f"Cosine similarity: {cosine_sim:.6f}")
    
    # Analysis
    print(f"\n💡 ANALYSIS:")
    
    if std_ratio > 2.0:
        print(f"⚠️  CRITICAL: MAX Graph std is {std_ratio:.2f}x higher than OpenAI")
        print("   This high variance difference may cause decoder issues")
    elif std_ratio > 1.5:
        print(f"⚠️  WARNING: MAX Graph std is {std_ratio:.2f}x higher than OpenAI")
        print("   Consider feature normalization")
    else:
        print(f"✅ Standard deviation ratio looks reasonable ({std_ratio:.2f})")
        
    if abs(mean_diff) > 0.1:
        print(f"⚠️  Mean difference is significant ({mean_diff:.6f})")
        print("   Consider mean centering")
    else:
        print(f"✅ Mean difference is small ({mean_diff:.6f})")
        
    if cosine_sim < 0.7:
        print(f"⚠️  Low cosine similarity ({cosine_sim:.6f})")
        print("   Features have structural differences")
    elif cosine_sim > 0.99:
        print(f"✅ Excellent cosine similarity ({cosine_sim:.6f})")
        print("   Features are very well aligned")
    else:
        print(f"✅ Good cosine similarity ({cosine_sim:.6f})")
        
    # Recommendations based on current results (259 chars)
    print(f"\n🎯 RECOMMENDATIONS FOR CURRENT PERFORMANCE:")
    print(f"Current transcription: 259 chars (vs 2035 baseline)")
    print(f"Current approach: No scaling + repetition detection ✅")
    
    if std_ratio > 3.0:
        print("❌ High variance suggests aggressive scaling might help")
    else:
        print("✅ Current no-scaling approach is working well")
        print("✅ Focus on decoder parameter tuning rather than feature scaling")
        

def main():
    """Main function"""
    if not os.path.exists("audio_samples/modular_video.wav"):
        print("❌ Audio file not found: audio_samples/modular_video.wav")
        return
        
    analyze_feature_differences()
    print("\n✅ Simple feature analysis complete!")


if __name__ == "__main__":
    main()