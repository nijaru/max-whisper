#!/usr/bin/env python3
"""
CPU Whisper Implementation
OpenAI Whisper baseline on CPU (no acceleration)
"""

import time
import numpy as np
import os

class WhisperCPU:
    """CPU-only OpenAI Whisper baseline implementation"""
    
    def __init__(self, model_size="tiny"):
        self.available = True
        self.model_size = model_size
        self._setup_whisper()
    
    def _setup_whisper(self):
        """Setup OpenAI Whisper on CPU"""
        try:
            import whisper
            
            # Force CPU device
            self.whisper_model = whisper.load_model(self.model_size, device="cpu")
            self.device = "cpu"
            
            print(f"✅ OpenAI Whisper {self.model_size} baseline on CPU")
            
        except Exception as e:
            print(f"❌ Whisper CPU setup failed: {e}")
            self.available = False
    
    def _load_real_audio(self, audio_file=None):
        """Load real audio file"""
        try:
            import librosa
            if not audio_file:
                audio_file = "audio_samples/modular_video.wav"
            
            if not os.path.exists(audio_file):
                print(f"❌ Audio file not found: {audio_file}")
                return None
            
            # Load audio at 16kHz (Whisper's expected sample rate)
            audio, sr = librosa.load(audio_file, sr=16000)
            print(f"    ✅ Real audio loaded: {len(audio)/sr:.1f}s")
            
            return audio
            
        except Exception as e:
            print(f"❌ Audio loading failed: {e}")
            return None
    
    def transcribe(self, audio_file: str = None) -> str:
        """
        CPU baseline transcription using OpenAI Whisper
        """
        if not self.available:
            return "❌ CPU Whisper not available"
        
        print("🚀 Starting CPU BASELINE transcription...")
        total_start = time.time()
        
        try:
            # Load real audio 
            audio = self._load_real_audio(audio_file)
            
            if audio is None:
                return "❌ Audio loading failed"
            
            # CPU transcription (no optimizations)
            print("  🎯 Running CPU baseline transcription...")
            whisper_start = time.time()
            
            # Use basic parameters for CPU baseline
            result = self.whisper_model.transcribe(
                audio,
                verbose=False,
                temperature=0.0
            )
            
            text = result["text"].strip()
            whisper_time = time.time() - whisper_start
            print(f"    ✅ CPU baseline: {whisper_time*1000:.1f}ms")
            
            # Clean up text
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]
            
            total_time = time.time() - total_start
            print(f"🏆 Total CPU BASELINE: {total_time*1000:.3f}ms")
            
            return text
            
        except Exception as e:
            print(f"❌ CPU baseline failed: {e}")
            return f"Transcription error: {e}"

def demo_cpu(model_size="tiny", audio_file=None):
    """Demo of CPU baseline Whisper implementation"""
    print(f"🚀 CPU Whisper Baseline Demo (model: {model_size})")
    print("=" * 50)
    
    model = WhisperCPU(model_size=model_size)
    
    if not model.available:
        print("❌ Demo cannot run - model not available")
        return
    
    try:
        result = model.transcribe(audio_file=audio_file)
        print(f"\n📝 CPU Baseline Result:")
        print(f"   {result}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    
    print(f"\n🎯 CPU Baseline Features:")
    print(f"   ✅ Pure OpenAI Whisper (no acceleration)")
    print(f"   ✅ CPU-only processing")
    print(f"   ✅ Baseline performance reference")
    print(f"   ✅ Guaranteed compatibility")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CPU Whisper Demo")
    parser.add_argument('--model-size', choices=['tiny', 'small', 'base'], default='tiny',
                       help='Whisper model size (default: tiny)')
    parser.add_argument('--audio-file', default=None,
                       help='Audio file path (default: audio_samples/modular_video.wav)')
    
    args = parser.parse_args()
    demo_cpu(model_size=args.model_size, audio_file=args.audio_file)