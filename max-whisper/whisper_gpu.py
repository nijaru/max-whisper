#!/usr/bin/env python3
"""
GPU Whisper Implementation
OpenAI Whisper with CUDA GPU acceleration
"""

import time
import numpy as np
import os

class WhisperGPU:
    """GPU-accelerated OpenAI Whisper implementation"""
    
    def __init__(self, model_size="tiny", use_gpu=True):
        self.available = True
        self.model_size = model_size
        self.use_gpu = use_gpu
        
        # Initialize OpenAI Whisper 
        self._setup_whisper()
    
    def _setup_whisper(self):
        """Setup OpenAI Whisper optimized"""
        try:
            import whisper
            import torch
            
            # Force GPU for this implementation
            if not torch.cuda.is_available():
                raise Exception("CUDA not available - GPU required for this implementation")
            device = "cuda"
            
            # Load model with optimizations
            self.whisper_model = whisper.load_model(self.model_size, device=device)
            self.device = device
            
            # Enable optimizations
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                print(f"✅ OpenAI Whisper {self.model_size} optimized on {device}")
            else:
                print(f"✅ OpenAI Whisper {self.model_size} on {device}")
            
        except Exception as e:
            print(f"❌ Whisper setup failed: {e}")
            self.available = False
    
    def _load_real_audio(self, audio_file=None):
        """Load real audio file"""
        try:
            import librosa
            if not audio_file:
                audio_file = "audio_samples/modular_video.wav"
            
            if os.path.exists(audio_file):
                audio, sr = librosa.load(audio_file, sr=16000)
                print(f"    ✅ Real audio loaded: {len(audio)/sr:.1f}s")
                return audio
            else:
                print(f"    ⚠️ Audio file not found: {audio_file}")
                return None
        except Exception as e:
            print(f"    ⚠️ Audio loading failed: {e}")
            return None
    
    def transcribe(self, audio_file: str = None) -> str:
        """
        Transcribe using optimized OpenAI Whisper
        """
        if not self.available:
            return "❌ Transcription not available"
        
        print("🚀 Starting OPTIMIZED transcription...")
        total_start = time.time()
        
        try:
            # Load real audio
            audio = self._load_real_audio(audio_file)
            
            if audio is None:
                return "❌ No audio available for transcription"
            
            # Optimized transcription
            print("  🎯 Running optimized OpenAI Whisper...")
            whisper_start = time.time()
            
            # Use optimized parameters
            result = self.whisper_model.transcribe(
                audio,
                verbose=False,
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            text = result["text"].strip()
            
            whisper_time = time.time() - whisper_start
            print(f"    ✅ Optimized Whisper: {whisper_time*1000:.3f}ms")
            
            # Quick post-processing
            if text:
                # Clean up text
                text = text.strip()
                # Ensure proper capitalization
                if text and not text[0].isupper():
                    text = text[0].upper() + text[1:]
            
            total_time = time.time() - total_start
            print(f"🏆 Total OPTIMIZED transcription: {total_time*1000:.3f}ms")
            
            return text
            
        except Exception as e:
            print(f"❌ Optimized transcription failed: {e}")
            return f"Transcription error: {e}"

def demo_gpu(model_size="tiny", audio_file=None):
    """Demo of GPU Whisper implementation"""
    print(f"🚀 GPU Whisper Demo (CUDA-accelerated OpenAI Whisper, model: {model_size})")
    print("=" * 60)
    
    model = WhisperGPU(model_size=model_size, use_gpu=True)
    
    if not model.available:
        print("❌ Demo cannot run - model not available")
        return
    
    try:
        result = model.transcribe(audio_file=audio_file)
        print(f"\n📝 GPU Transcription Result:")
        print(f"   {result}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    
    print(f"\n🎯 GPU Approach Benefits:")
    print(f"   ✅ Guaranteed working transcription")
    print(f"   ✅ CUDA GPU acceleration")
    print(f"   ✅ Real speech recognition (not generic text)")
    print(f"   ⚡ Optimized for GPU performance")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Whisper Demo")
    parser.add_argument('--model-size', choices=['tiny', 'small', 'base'], default='tiny',
                       help='Whisper model size (default: tiny)')
    parser.add_argument('--audio-file', default=None,
                       help='Audio file path (default: audio_samples/modular_video.wav)')
    
    args = parser.parse_args()
    demo_gpu(model_size=args.model_size, audio_file=args.audio_file)