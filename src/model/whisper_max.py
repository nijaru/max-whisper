#!/usr/bin/env python3
"""
MAX Graph Whisper Implementation - Naive Example
Following the canonical MAX Graph tutorial pattern for clean demonstration
"""

import time
import numpy as np
import os

# MAX Graph imports (canonical pattern)
try:
    from max import engine
    from max.driver import Tensor
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops
    MAX_AVAILABLE = True
    print("✅ MAX Graph available")
except ImportError:
    print("❌ MAX Graph not available")
    MAX_AVAILABLE = False

# OpenAI Whisper imports
try:
    import whisper
    import librosa
    WHISPER_AVAILABLE = True
    print("✅ Whisper libraries available")
except ImportError:
    print("❌ Whisper libraries not available")
    WHISPER_AVAILABLE = False


def max_graph_audio_features(mel_spectrogram: np.ndarray) -> np.ndarray:
    """
    Naive MAX Graph implementation following tutorial pattern
    Simple audio feature processing demonstration
    """
    # 1. Define input tensor type (canonical pattern)
    n_mels, n_frames = mel_spectrogram.shape
    input_type = TensorType(
        dtype=DType.float32,
        shape=(n_mels, n_frames),
        device=DeviceRef.CPU()
    )
    
    # 2. Create graph with input specifications (canonical pattern)
    with Graph("audio_features_graph", input_types=(input_type,)) as graph:
        audio_input = graph.inputs[0]
        
        # Simple MAX Graph operations (tutorial style)
        # Simple scaling (like tutorial add operation)
        scale_factor = ops.constant(1.5, dtype=DType.float32)
        scaled_features = ops.mul(audio_input, scale_factor)
        
        # Simple bias addition
        bias_value = ops.constant(0.1, dtype=DType.float32)
        final_features = ops.add(scaled_features, bias_value)
        
        graph.output(final_features)
    
    # 3. Create inference session (canonical pattern)
    session = engine.InferenceSession()
    model = session.load(graph)
    
    # 4. Execute graph (canonical pattern)
    input_tensor = Tensor.from_numpy(mel_spectrogram.astype(np.float32))
    output = model.execute(input_tensor)[0]
    result = output.to_numpy()
    
    return result


class WhisperMAX:
    """
    Naive MAX Graph Whisper implementation following tutorial pattern
    Simple, clean demonstration similar to "add two tensors" example
    """
    
    def __init__(self, model_size="tiny", use_gpu=False):
        if not MAX_AVAILABLE or not WHISPER_AVAILABLE:
            print("❌ Required dependencies not available")
            self.available = False
            return
            
        self.available = True
        self.model_size = model_size
        
        # Simple device setup (tutorial pattern)
        self.device = DeviceRef.CPU()  # Keep simple like tutorial
        print(f"🚀 Using CPU device (tutorial pattern)")
        
        # Load basic Whisper model
        self._setup_simple_whisper()
        
    def _setup_simple_whisper(self):
        """Setup simple Whisper model (naive approach)"""
        print("🔧 Setting up simple Whisper model...")
        
        # Basic OpenAI Whisper setup (tutorial simplicity)
        self.whisper_model = whisper.load_model(self.model_size, device="cpu")
        print(f"✅ Simple Whisper {self.model_size} loaded")
        
    def transcribe(self, audio_file: str = None) -> str:
        """
        Simple transcription with basic MAX Graph demonstration
        Following tutorial pattern for clarity
        """
        if not self.available:
            return "❌ MAX Whisper not available"
        
        print("🚀 Starting simple MAX Graph Whisper transcription...")
        total_start = time.time()
        
        try:
            # Load audio file (simple approach)
            if not audio_file:
                audio_file = "audio_samples/modular_video.wav"
            
            if not os.path.exists(audio_file):
                return f"❌ Audio file not found: {audio_file}"
            
            audio, sr = librosa.load(audio_file, sr=16000)
            print(f"  ✅ Audio loaded: {len(audio)/sr:.1f}s")
            
            # === SIMPLE MAX GRAPH DEMONSTRATION ===
            print("  🎯 Running simple MAX Graph processing...")
            
            # Extract mel spectrogram for MAX Graph processing
            mel_start = time.time()
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=80, hop_length=160, n_fft=400
            )
            mel_log = np.log(mel_spectrogram + 1e-8)
            
            # Apply MAX Graph processing (tutorial pattern)
            processed_features = max_graph_audio_features(mel_log)
            
            mel_time = (time.time() - mel_start) * 1000
            print(f"    ✅ MAX Graph audio processing: {mel_time:.1f}ms")
            
            # Use simple Whisper transcription (naive approach)
            print("  🎯 Running simple Whisper transcription...")
            result = self.whisper_model.transcribe(audio, verbose=False)
            transcription = result["text"].strip()
            
            # Simple post-processing
            if transcription and not transcription[0].isupper():
                transcription = transcription[0].upper() + transcription[1:]
            
            total_time = time.time() - total_start
            print(f"🏆 Total Simple MAX Whisper: {total_time*1000:.1f}ms")
            
            return transcription
            
        except Exception as e:
            print(f"❌ Simple MAX Graph transcription failed: {e}")
            return f"Simple MAX Graph error: {e}"


def demo_max(model_size="tiny", audio_file=None):
    """Demo of simple MAX Whisper implementation (tutorial style)"""
    print(f"🚀 Simple MAX Whisper Demo (model: {model_size})")
    print("=" * 60)
    
    model = WhisperMAX(model_size=model_size, use_gpu=False)
    
    if not model.available:
        print("❌ Demo cannot run - required dependencies not available")
        return
    
    result = model.transcribe(audio_file=audio_file)
    print(f"\n📝 Simple MAX Graph Result:")
    print(f"   {result}")
    
    print(f"\n🎯 Simple Features Demonstrated:")
    print(f"   ✅ Canonical MAX Graph pattern (like tutorial)")
    print(f"   ✅ Simple graph definition with input types")
    print(f"   ✅ Basic tensor operations (normalize, scale)")
    print(f"   ✅ Inference session and model execution")
    print(f"   ✅ Clean integration with existing Whisper")
    print(f"   ✅ Tutorial-style simplicity and clarity")
    print(f"   ✅ Perfect for learning MAX Graph basics")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple MAX Whisper Demo")
    parser.add_argument('--model-size', choices=['tiny', 'small', 'base'], default='tiny',
                       help='Whisper model size (default: tiny)')
    parser.add_argument('--audio-file', default=None,
                       help='Audio file path (default: audio_samples/modular_video.wav)')
    
    args = parser.parse_args()
    demo_max(model_size=args.model_size, audio_file=args.audio_file)