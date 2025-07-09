#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "max-whisper"))

from whisper_max import WhisperMAX
import whisper

# Load audio
audio = whisper.load_audio('audio_samples/modular_video.wav')

# Test hybrid approach with proper initialization
encoder = WhisperMAX(model_size='tiny', full_max_graph=False)

# Get encoder features 
mel_features = whisper.log_mel_spectrogram(audio)
mel_np = mel_features.cpu().numpy()
print(f"Mel features shape: {mel_np.shape}")

# Run encoder
encoder_features = encoder._encode_with_max_graph(mel_np)
print(f"Encoder features shape: {encoder_features.shape}")

# Run hybrid decoder
hybrid_result = encoder._decode_with_pytorch(encoder_features)
print(f"Hybrid result: '{hybrid_result}'")