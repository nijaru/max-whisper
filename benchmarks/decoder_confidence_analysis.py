#!/usr/bin/env python3
"""
Decoder Confidence Analysis for MAX Graph Whisper
Analyze why MAX Graph features cause early stopping compared to OpenAI features
"""

import numpy as np
import torch
import os
import sys

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


def analyze_decoder_confidence(audio_file="audio_samples/modular_video.wav"):
    """Analyze decoder confidence differences between MAX Graph and OpenAI features"""
    
    if not MAX_WHISPER_AVAILABLE or not WHISPER_AVAILABLE:
        print("❌ Required libraries not available")
        return
        
    print("🔬 Decoder Confidence Analysis")
    print("=" * 60)
    
    # Initialize models
    max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
    if not max_whisper.available:
        print("❌ MAX Graph Whisper not available")
        return
    
    openai_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load audio and get features
    import librosa
    audio, sr = librosa.load(audio_file, sr=16000)
    mel_db = whisper.log_mel_spectrogram(audio).numpy()
    
    print(f"🎵 Audio loaded: {len(audio)/sr:.1f}s, Mel: {mel_db.shape}")
    
    # Get features from both models
    print("\n🔢 Extracting encoder features...")
    max_features = max_whisper._encode_with_max_graph(mel_db)
    
    # Get OpenAI features properly
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
    
    print(f"✅ Features extracted: MAX {max_features.shape}, OpenAI {openai_features.shape}")
    
    # Analyze decoder behavior with different confidence thresholds
    print("\n🎯 DECODER CONFIDENCE ANALYSIS:")
    
    decoder_configs = [
        {"name": "Conservative", "patience": 5.0, "beam_size": 1, "sample_len": 2000},
        {"name": "Balanced", "patience": 10.0, "beam_size": 5, "sample_len": 2000},
        {"name": "Aggressive", "patience": 20.0, "beam_size": 10, "sample_len": 2000},
        {"name": "Ultra-patient", "patience": 50.0, "beam_size": 5, "sample_len": 3000},
    ]
    
    results = {}
    
    for config in decoder_configs:
        print(f"\n📊 Testing {config['name']} decoder settings:")
        print(f"   patience={config['patience']}, beam_size={config['beam_size']}, sample_len={config['sample_len']}")
        
        # Test with MAX Graph features
        max_result = test_decoder_with_features(
            max_features, openai_model, config, "MAX Graph"
        )
        
        # Test with OpenAI features  
        openai_result = test_decoder_with_features(
            openai_features, openai_model, config, "OpenAI"
        )
        
        results[config['name']] = {
            "max_graph": max_result,
            "openai": openai_result,
            "config": config
        }
        
        print(f"   MAX Graph: {max_result['length']} chars")
        print(f"   OpenAI: {openai_result['length']} chars")
        print(f"   Ratio: {max_result['length']/openai_result['length']*100:.1f}%")
    
    # Analysis
    print(f"\n💡 CONFIDENCE ANALYSIS RESULTS:")
    
    best_config = None
    best_ratio = 0
    
    for name, result in results.items():
        ratio = result['max_graph']['length'] / result['openai']['length']
        print(f"{name:12}: {ratio*100:5.1f}% length ratio")
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_config = name
    
    print(f"\n🏆 Best configuration: {best_config} ({best_ratio*100:.1f}% of OpenAI length)")
    
    # Detailed analysis of best result
    if best_config:
        best = results[best_config]
        print(f"\n📝 Best MAX Graph transcription ({best['max_graph']['length']} chars):")
        print(f"   '{best['max_graph']['text'][:200]}...'")
        print(f"\n📝 Corresponding OpenAI transcription ({best['openai']['length']} chars):")
        print(f"   '{best['openai']['text'][:200]}...'")
        
    return results


def test_decoder_with_features(features, model, config, name):
    """Test decoder with given features and configuration"""
    
    try:
        # Convert to tensor
        features_tensor = torch.from_numpy(features.copy()).float()
        device = next(model.parameters()).device
        features_tensor = features_tensor.to(device)
        
        # Create decoder options
        options = DecodingOptions(
            task="transcribe",
            language="en", 
            temperature=0.0,
            sample_len=config['sample_len'],
            beam_size=config['beam_size'],
            patience=config['patience'],
            without_timestamps=True,
            suppress_blank=True,
            suppress_tokens="-1"
        )
        
        # Decode
        result = model.decode(features_tensor, options)
        
        # Extract text
        if isinstance(result, list) and len(result) > 0:
            text = result[0].text.strip()
            # Get additional metrics
            avg_logprob = getattr(result[0], 'avg_logprob', None)
            compression_ratio = getattr(result[0], 'compression_ratio', None)
        elif hasattr(result, 'text'):
            text = result.text.strip()
            avg_logprob = getattr(result, 'avg_logprob', None)
            compression_ratio = getattr(result, 'compression_ratio', None)
        else:
            text = "Decoder error"
            avg_logprob = None
            compression_ratio = None
        
        return {
            "text": text,
            "length": len(text),
            "avg_logprob": avg_logprob,
            "compression_ratio": compression_ratio
        }
        
    except Exception as e:
        print(f"   ❌ {name} decoding failed: {e}")
        return {
            "text": f"Error: {e}",
            "length": 0,
            "avg_logprob": None,
            "compression_ratio": None
        }


def main():
    """Main function"""
    if not os.path.exists("audio_samples/modular_video.wav"):
        print("❌ Audio file not found: audio_samples/modular_video.wav")
        return
        
    results = analyze_decoder_confidence()
    
    if results:
        print("\n✅ Decoder confidence analysis complete!")
        print("\n🎯 RECOMMENDATION:")
        
        # Find the configuration that gets closest to OpenAI length
        best_config = None
        best_ratio = 0
        
        for name, result in results.items():
            ratio = result['max_graph']['length'] / result['openai']['length']
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = name
        
        if best_config and best_ratio > 0.8:
            print(f"✅ Use {best_config} configuration for {best_ratio*100:.1f}% transcription length")
            config = results[best_config]['config']
            print(f"   Recommended: patience={config['patience']}, beam_size={config['beam_size']}, sample_len={config['sample_len']}")
        elif best_config:
            print(f"⚠️  Best found: {best_config} gives {best_ratio*100:.1f}% length - may need feature adjustment")
        else:
            print("❌ No configuration achieved good results - feature distribution issue likely")
    else:
        print("❌ Analysis failed")


if __name__ == "__main__":
    main()