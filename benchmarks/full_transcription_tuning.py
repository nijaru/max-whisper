#!/usr/bin/env python3
"""
Full Transcription Tuning for MAX Graph Whisper
Focus on achieving 2035 character baseline transcription length instead of 259 chars
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


def test_extreme_decoder_parameters(audio_file="audio_samples/modular_video.wav"):
    """Test extreme decoder parameters to force full-length transcription"""
    
    if not MAX_WHISPER_AVAILABLE or not WHISPER_AVAILABLE:
        print("❌ Required libraries not available")
        return
        
    print("🎯 Full Transcription Tuning - Goal: 2035 chars like baseline")
    print("=" * 70)
    
    # Initialize MAX Graph Whisper
    max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
    if not max_whisper.available:
        print("❌ MAX Graph Whisper not available")
        return
    
    # Load baseline for comparison
    print("📊 Getting baseline transcription length...")
    try:
        import librosa
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Get baseline with OpenAI
        openai_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
        baseline_result = whisper.transcribe(openai_model, audio, language="en")
        baseline_text = baseline_result['text'].strip()
        baseline_length = len(baseline_text)
        
        print(f"✅ Baseline: {baseline_length} characters")
        print(f"📝 Baseline text: '{baseline_text[:150]}...'")
        
    except Exception as e:
        print(f"❌ Baseline failed: {e}")
        baseline_length = 2035  # Known baseline length
        print(f"📊 Using known baseline length: {baseline_length} characters")
    
    # Test configurations designed to force full transcription
    print(f"\n🧪 Testing extreme decoder configurations to reach {baseline_length} chars:")
    
    test_configs = [
        {
            "name": "Ultra-Conservative",
            "patience": 2.0,
            "beam_size": 1,
            "sample_len": 4000,
            "temperature": 0.0,
            "compression_ratio_threshold": 1.0,  # Very low threshold
            "logprob_threshold": -2.0,  # Very low threshold
        },
        {
            "name": "Force-Continue",
            "patience": 100.0,  # Extreme patience
            "beam_size": 1,
            "sample_len": 4000,
            "temperature": 0.0,
            "compression_ratio_threshold": 1.0,
            "logprob_threshold": -5.0,  # Very permissive
        },
        {
            "name": "High-Beam-Patient",
            "patience": 50.0,
            "beam_size": 15,  # High beam search
            "sample_len": 4000,
            "temperature": 0.0,
            "compression_ratio_threshold": 1.5,
            "logprob_threshold": -3.0,
        },
        {
            "name": "Temperature-Diverse",
            "patience": 30.0,
            "beam_size": 10,
            "sample_len": 4000,
            "temperature": 0.2,  # Add randomness
            "compression_ratio_threshold": 1.0,
            "logprob_threshold": -3.0,
        },
        {
            "name": "Max-Length-Force",
            "patience": 1000.0,  # Ridiculous patience
            "beam_size": 20,
            "sample_len": 6000,  # Very long
            "temperature": 0.0,
            "compression_ratio_threshold": 0.5,  # Very permissive
            "logprob_threshold": -10.0,  # Extremely permissive
        }
    ]
    
    results = []
    best_length = 0
    best_config = None
    
    for config in test_configs:
        print(f"\n📋 Testing {config['name']}:")
        print(f"   patience={config['patience']}, beam={config['beam_size']}, sample_len={config['sample_len']}")
        print(f"   temperature={config['temperature']}, compression_threshold={config['compression_ratio_threshold']}")
        
        try:
            start_time = time.time()
            
            # Get mel features 
            mel_db = whisper.log_mel_spectrogram(audio).numpy()
            
            # Get MAX Graph features
            max_features = max_whisper._encode_with_max_graph(mel_db)
            
            # Convert to tensor
            features_tensor = torch.from_numpy(max_features.copy()).float()
            device = next(max_whisper.whisper_model.parameters()).device
            features_tensor = features_tensor.to(device)
            
            # Create extreme decoder options
            options = DecodingOptions(
                task="transcribe",
                language="en",
                temperature=config['temperature'],
                sample_len=config['sample_len'],
                beam_size=config['beam_size'],
                patience=config['patience'],
                compression_ratio_threshold=config['compression_ratio_threshold'],
                logprob_threshold=config['logprob_threshold'],
                without_timestamps=True,
                suppress_blank=True,
                suppress_tokens="-1"
            )
            
            # Decode
            result = max_whisper.whisper_model.decode(features_tensor, options)
            
            # Extract transcription
            if isinstance(result, list) and len(result) > 0:
                transcription = result[0].text.strip()
                avg_logprob = getattr(result[0], 'avg_logprob', None)
                compression_ratio = getattr(result[0], 'compression_ratio', None)
            elif hasattr(result, 'text'):
                transcription = result.text.strip()
                avg_logprob = getattr(result, 'avg_logprob', None)
                compression_ratio = getattr(result, 'compression_ratio', None)
            else:
                transcription = "Decoder error"
                avg_logprob = None
                compression_ratio = None
            
            # Clean repetition
            original_length = len(transcription)
            transcription = max_whisper._clean_repetitive_text(transcription)
            cleaned_length = len(transcription)
            
            decode_time = time.time() - start_time
            
            result_data = {
                "config": config,
                "original_length": original_length,
                "cleaned_length": cleaned_length,
                "text": transcription,
                "avg_logprob": avg_logprob,
                "compression_ratio": compression_ratio,
                "decode_time": decode_time,
                "baseline_percentage": (cleaned_length / baseline_length) * 100
            }
            
            results.append(result_data)
            
            print(f"   ✅ Result: {original_length} → {cleaned_length} chars ({result_data['baseline_percentage']:.1f}% of baseline)")
            print(f"   📊 Metrics: avg_logprob={avg_logprob:.3f}, compression_ratio={compression_ratio:.2f}")
            print(f"   ⏱️ Time: {decode_time:.2f}s")
            print(f"   📝 Text: '{transcription[:100]}...'")
            
            if cleaned_length > best_length:
                best_length = cleaned_length
                best_config = config['name']
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Analysis
    print(f"\n🏆 RESULTS SUMMARY:")
    print(f"Baseline target: {baseline_length} characters")
    print(f"Current best: {best_length} characters ({(best_length/baseline_length)*100:.1f}% of baseline)")
    print(f"Best configuration: {best_config}")
    
    # Sort results by length
    results.sort(key=lambda x: x['cleaned_length'], reverse=True)
    
    print(f"\n📊 ALL RESULTS (sorted by length):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['config']['name']:15}: {result['cleaned_length']:4d} chars ({result['baseline_percentage']:5.1f}%)")
    
    # Detailed analysis of best result
    if results:
        best = results[0]
        print(f"\n🔍 BEST RESULT ANALYSIS:")
        print(f"Configuration: {best['config']['name']}")
        print(f"Length: {best['cleaned_length']} / {baseline_length} ({best['baseline_percentage']:.1f}%)")
        print(f"Metrics: avg_logprob={best['avg_logprob']:.3f}, compression_ratio={best['compression_ratio']:.2f}")
        print(f"Time: {best['decode_time']:.2f}s")
        print(f"\nFull transcription:")
        print(f"'{best['text']}'")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if best['baseline_percentage'] > 90:
            print("✅ Excellent! This configuration achieves near-complete transcription")
        elif best['baseline_percentage'] > 70:
            print("✅ Good progress! Try further tuning compression_ratio_threshold and logprob_threshold")
        elif best['baseline_percentage'] > 50:
            print("⚠️ Moderate progress. Consider feature scaling or distribution adjustment")
        else:
            print("❌ Limited progress. Likely need fundamental feature distribution fix")
            
        return best
    
    return None


def main():
    """Main function"""
    if not os.path.exists("audio_samples/modular_video.wav"):
        print("❌ Audio file not found: audio_samples/modular_video.wav")
        return
        
    best_result = test_extreme_decoder_parameters()
    
    if best_result and best_result['baseline_percentage'] > 80:
        print("\n🎯 SUCCESS: Found configuration achieving >80% baseline length!")
        print("Update your main implementation with these parameters:")
        config = best_result['config']
        print(f"""
        options = DecodingOptions(
            task="transcribe",
            language="en",
            temperature={config['temperature']},
            sample_len={config['sample_len']},
            beam_size={config['beam_size']},
            patience={config['patience']},
            compression_ratio_threshold={config['compression_ratio_threshold']},
            logprob_threshold={config['logprob_threshold']},
            without_timestamps=True,
            suppress_blank=True,
            suppress_tokens="-1"
        )""")
    else:
        print("\n⚠️ No configuration achieved >80% baseline length")
        print("Next steps: Feature distribution adjustment needed")


if __name__ == "__main__":
    main()