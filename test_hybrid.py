#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "max-whisper"))

from whisper_max import WhisperMAX
import whisper

# Load audio
audio = whisper.load_audio('audio_samples/modular_video.wav')

# Test hybrid approach
encoder = WhisperMAX(model_size='tiny', full_max_graph=False)
result = encoder.transcribe(audio)
print('Hybrid result:', result)