#!/usr/bin/env python3
"""
Test what's actually working in our implementations.
Shows the real status of each Whisper variant.
"""

import os
import sys
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("TESTING ALL WHISPER IMPLEMENTATIONS")
print("=" * 70)
print()

# Generate test audio
def generate_test_audio(duration=5.0):
    """Generate simple test audio."""
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    return audio

test_audio = generate_test_audio(5.0)
print(f"Test audio: {len(test_audio)/16000:.1f}s\n")

# 1. Test OpenAI Whisper
print("1. OPENAI WHISPER")
print("-" * 70)
try:
    import whisper
    import tempfile
    import soundfile as sf
    
    # Check both CPU and GPU
    for device in ["cpu", "cuda"]:
        try:
            print(f"\nTesting on {device.upper()}:")
            model = whisper.load_model("tiny", device=device)
            
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, test_audio, 16000)
                temp_path = tmp.name
            
            # Test transcription
            start = time.time()
            result = model.transcribe(temp_path, fp16=(device=="cuda"))
            end = time.time()
            
            print(f"✅ Success! Time: {(end-start)*1000:.2f}ms")
            print(f"   Transcription: '{result['text']}'")
            print(f"   Language: {result['language']}")
            
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"❌ Failed on {device}: {e}")
            
except ImportError:
    print("❌ OpenAI Whisper not installed")
except Exception as e:
    print(f"❌ Error: {e}")

# 2. Test Faster-Whisper
print("\n\n2. FASTER-WHISPER")
print("-" * 70)
try:
    from faster_whisper import WhisperModel
    
    # Check both CPU and GPU
    for device in ["cpu", "cuda"]:
        try:
            print(f"\nTesting on {device.upper()}:")
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel("tiny", device=device, compute_type=compute_type)
            
            # Test transcription
            start = time.time()
            segments, info = model.transcribe(test_audio)
            text = " ".join([s.text for s in segments])
            end = time.time()
            
            print(f"✅ Success! Time: {(end-start)*1000:.2f}ms")
            print(f"   Transcription: '{text}'")
            print(f"   Language: {info.language}")
            
        except Exception as e:
            print(f"❌ Failed on {device}: {e}")
            
except ImportError:
    print("❌ Faster-Whisper not installed")
except Exception as e:
    print(f"❌ Error: {e}")

# 3. Test MAX-Whisper
print("\n\n3. MAX-WHISPER (Our Implementation)")
print("-" * 70)

# Set up CUDA paths
cuda_lib_path = "/home/nick/github/modular-hackathon/.pixi/envs/benchmark/lib/python3.11/site-packages/nvidia"
os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}/cublas/lib:{cuda_lib_path}/cudnn/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    from model.max_whisper_simple import SimpleWhisperEncoder
    from audio.preprocessing import compute_mel_spectrogram, normalize_features
    
    # Prepare audio
    mel_spec = compute_mel_spectrogram(test_audio)
    mel_features = normalize_features(mel_spec).astype(np.float32)
    if mel_features.ndim == 2:
        mel_features = mel_features[np.newaxis, :, :]
    
    # Pad to 1500 frames
    if mel_features.shape[2] < 1500:
        pad_width = ((0, 0), (0, 0), (0, 1500 - mel_features.shape[2]))
        mel_features = np.pad(mel_features, pad_width, mode='constant')
    elif mel_features.shape[2] > 1500:
        mel_features = mel_features[:, :, :1500]
    
    # Test both CPU and GPU
    for device in ["cpu", "gpu"]:
        try:
            print(f"\nTesting on {device.upper()}:")
            encoder = SimpleWhisperEncoder(device=device)
            
            # Warmup
            for _ in range(3):
                _ = encoder.encode(mel_features)
            
            # Test encoding
            start = time.time()
            output = encoder.encode(mel_features)
            end = time.time()
            
            print(f"✅ Success! Time: {(end-start)*1000:.2f}ms")
            print(f"   Output shape: {output.shape}")
            print(f"   Note: Encoder only - no transcription")
            
        except Exception as e:
            print(f"❌ Failed on {device}: {e}")
            
except ImportError as e:
    print(f"❌ MAX-Whisper import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

# Summary
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nKey Differences:")
print("1. OpenAI Whisper: Full model with transcription")
print("2. Faster-Whisper: Optimized full model with transcription")
print("3. MAX-Whisper: Custom encoder only (no decoder/transcription yet)")
print("\nThe MAX-Whisper speedup numbers compare encoder-only performance")
print("vs full transcription pipelines - not a fair comparison!")
print("\nFor fair comparison, MAX-Whisper needs:")
print("- Decoder implementation")
print("- Tokenizer")
print("- Actual text output")
print("=" * 70)