#!/usr/bin/env python3
"""
Simple Extreme Tuning - Focus on basic parameters to achieve full transcription
Test extreme values of patience, beam_size, sample_len, temperature only
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


def test_extreme_basic_parameters():
    """Test extreme basic parameters only"""
    
    if not MAX_WHISPER_AVAILABLE or not WHISPER_AVAILABLE:
        print("❌ Required libraries not available")
        return
        
    print("🎯 Simple Extreme Parameter Tuning - Target: 2035 chars")
    print("=" * 60)
    
    # Initialize
    max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
    if not max_whisper.available:
        print("❌ MAX Graph Whisper not available")
        return
    
    # Load audio
    import librosa
    audio, sr = librosa.load("audio_samples/modular_video.wav", sr=16000)
    mel_db = whisper.log_mel_spectrogram(audio).numpy()
    
    # Get baseline
    print("📊 Getting baseline length...")
    baseline_length = 2035  # Known baseline
    
    # Test extreme configurations
    configs = [
        {"name": "Extreme-Patience", "patience": 200.0, "beam_size": 5, "sample_len": 5000, "temperature": 0.0},
        {"name": "Max-Length", "patience": 50.0, "beam_size": 5, "sample_len": 8000, "temperature": 0.0},
        {"name": "High-Beam", "patience": 50.0, "beam_size": 25, "sample_len": 5000, "temperature": 0.0},
        {"name": "Temperature-Boost", "patience": 50.0, "beam_size": 10, "sample_len": 5000, "temperature": 0.5},
        {"name": "Ultra-Max", "patience": 1000.0, "beam_size": 50, "sample_len": 10000, "temperature": 0.0},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']}:")
        print(f"   patience={config['patience']}, beam={config['beam_size']}, sample_len={config['sample_len']}, temp={config['temperature']}")
        
        try:
            start_time = time.time()
            
            # Get MAX Graph features
            max_features = max_whisper._encode_with_max_graph(mel_db)
            
            # Convert to tensor
            features_tensor = torch.from_numpy(max_features.copy()).float()
            device = next(max_whisper.whisper_model.parameters()).device
            features_tensor = features_tensor.to(device)
            
            # Create options with only basic parameters
            options = DecodingOptions(
                task="transcribe",
                language="en",
                temperature=config['temperature'],
                sample_len=config['sample_len'],
                beam_size=config['beam_size'],
                patience=config['patience'],
                without_timestamps=True,
                suppress_blank=True,
                suppress_tokens="-1"
            )
            
            # Decode
            result = max_whisper.whisper_model.decode(features_tensor, options)
            
            # Extract text
            if isinstance(result, list) and len(result) > 0:
                text = result[0].text.strip()
            elif hasattr(result, 'text'):
                text = result.text.strip()
            else:
                text = "Decoder error"
            
            # Clean repetition
            original_length = len(text)
            text = max_whisper._clean_repetitive_text(text)
            cleaned_length = len(text)
            
            decode_time = time.time() - start_time
            percentage = (cleaned_length / baseline_length) * 100
            
            result_data = {
                "config": config,
                "original_length": original_length,
                "cleaned_length": cleaned_length,
                "text": text,
                "decode_time": decode_time,
                "percentage": percentage
            }
            
            results.append(result_data)
            
            print(f"   ✅ {original_length} → {cleaned_length} chars ({percentage:.1f}% of baseline)")
            print(f"   ⏱️ Time: {decode_time:.2f}s")
            print(f"   📝 Text: '{text[:120]}...'")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Results summary
    if results:
        results.sort(key=lambda x: x['cleaned_length'], reverse=True)
        
        print(f"\n🏆 RESULTS SUMMARY:")
        print(f"Target: {baseline_length} characters")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['config']['name']:15}: {result['cleaned_length']:4d} chars ({result['percentage']:5.1f}%)")
        
        # Best result analysis
        best = results[0]
        print(f"\n🔍 BEST RESULT:")
        print(f"Configuration: {best['config']['name']}")
        print(f"Length: {best['cleaned_length']} / {baseline_length} ({best['percentage']:.1f}%)")
        print(f"Parameters: patience={best['config']['patience']}, beam_size={best['config']['beam_size']}")
        print(f"sample_len={best['config']['sample_len']}, temperature={best['config']['temperature']}")
        print(f"Time: {best['decode_time']:.2f}s")
        print(f"\nFull text:")
        print(f"'{best['text']}'")
        
        if best['percentage'] > 80:
            print(f"\n✅ SUCCESS! Achieved {best['percentage']:.1f}% of baseline length")
        elif best['percentage'] > 50:
            print(f"\n⚠️ Moderate success ({best['percentage']:.1f}%). Try even more extreme parameters or feature scaling.")
        else:
            print(f"\n❌ Limited success ({best['percentage']:.1f}%). Feature distribution issue likely.")
        
        return best
    
    return None


def main():
    """Main function"""
    if not os.path.exists("audio_samples/modular_video.wav"):
        print("❌ Audio file not found: audio_samples/modular_video.wav")
        return
        
    best = test_extreme_basic_parameters()
    
    if best and best['percentage'] > 80:
        print("\n🎯 NEXT STEPS:")
        print("Update your main implementation with these parameters for full transcription!")
    elif best:
        print(f"\n🎯 NEXT STEPS:")
        print(f"Best achieved: {best['percentage']:.1f}% of baseline")
        print("Consider feature scaling or distribution adjustment to reach 100%")
    else:
        print("\n❌ No successful configurations found")


if __name__ == "__main__":
    main()