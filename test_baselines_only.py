"""
Test only OpenAI Whisper and Faster-Whisper baselines
"""

import numpy as np
import time
import os

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("librosa not available")
    LIBROSA_AVAILABLE = False

try:
    import torch
    import whisper
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    print("OpenAI Whisper not available")
    OPENAI_WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    print("Faster-Whisper not available")
    FASTER_WHISPER_AVAILABLE = False

def load_audio(file_path, sample_rate=16000):
    """Load audio file"""
    if not LIBROSA_AVAILABLE:
        return None
    
    audio, sr = librosa.load(file_path, sr=sample_rate)
    duration = len(audio) / sample_rate
    print(f"Loaded audio: {duration:.1f}s, {sr}Hz")
    return audio, duration

def benchmark_openai_whisper(audio, duration):
    """Benchmark OpenAI Whisper"""
    if not OPENAI_WHISPER_AVAILABLE:
        return None
    
    print("Testing OpenAI Whisper...")
    
    try:
        model = whisper.load_model("tiny")
        
        # Warm-up
        _ = whisper.transcribe(model, audio[:16000])  # 1 second
        
        # Benchmark
        start_time = time.time()
        result = whisper.transcribe(model, audio)
        end_time = time.time()
        
        inference_time = end_time - start_time
        rtf = inference_time / duration
        speedup = 1 / rtf
        
        return {
            'model': 'OpenAI Whisper-tiny',
            'inference_time': inference_time,
            'rtf': rtf,
            'speedup': speedup,
            'text': result['text'].strip(),
            'status': 'success',
            'device': 'GPU' if torch.cuda.is_available() else 'CPU'
        }
        
    except Exception as e:
        return {
            'model': 'OpenAI Whisper-tiny',
            'status': 'error',
            'error': str(e)
        }

def benchmark_faster_whisper(audio_path, duration):
    """Benchmark Faster-Whisper"""
    if not FASTER_WHISPER_AVAILABLE:
        return None
    
    print("Testing Faster-Whisper...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel("tiny", device=device, compute_type="float16" if device == "cuda" else "float32")
        
        # Warm-up
        list(model.transcribe(audio_path, beam_size=1))
        
        # Benchmark
        start_time = time.time()
        segments, info = model.transcribe(audio_path, beam_size=1)
        text = " ".join([segment.text for segment in segments])
        end_time = time.time()
        
        inference_time = end_time - start_time
        rtf = inference_time / duration
        speedup = 1 / rtf
        
        return {
            'model': 'Faster-Whisper-tiny',
            'inference_time': inference_time,
            'rtf': rtf,
            'speedup': speedup,
            'text': text.strip(),
            'status': 'success',
            'device': device.upper()
        }
        
    except Exception as e:
        return {
            'model': 'Faster-Whisper-tiny',
            'status': 'error',
            'error': str(e)
        }

def main():
    print("="*80)
    print("BASELINE COMPARISON: OpenAI Whisper vs Faster-Whisper")
    print("="*80)
    
    # Load audio
    audio_file = "audio_samples/modular_video.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    audio_data = load_audio(audio_file)
    if not audio_data:
        print("❌ Failed to load audio")
        return
    
    audio, duration = audio_data
    print(f"✅ Loaded {duration:.1f}s of Modular video")
    
    # Run benchmarks
    results = []
    
    # Test OpenAI Whisper
    openai_result = benchmark_openai_whisper(audio, duration)
    if openai_result:
        results.append(openai_result)
    
    # Test Faster-Whisper
    faster_result = benchmark_faster_whisper(audio_file, duration)
    if faster_result:
        results.append(faster_result)
    
    # Display results
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    print(f"Audio: Modular video ({duration:.1f}s)")
    print("-" * 80)
    
    for result in results:
        if result['status'] == 'success':
            print(f"\n📊 {result['model']} ({result['device']})")
            print(f"⏱️  Inference Time: {result['inference_time']:.3f}s")
            print(f"📈 Real-Time Factor: {result['rtf']:.3f}")
            print(f"🚀 Speedup: {result['speedup']:.1f}x")
            print(f"📝 Text Preview: '{result['text'][:150]}...'")
        else:
            print(f"\n❌ {result['model']}: {result['status']}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
    
    # Analysis
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        fastest = max(successful_results, key=lambda x: x['speedup'])
        print(f"\n🏆 Fastest: {fastest['model']} at {fastest['speedup']:.1f}x real-time")
        
        print(f"\n💡 NEXT STEP:")
        print(f"Fix MAX-Whisper CUDA cuBLAS library issue to enable complete comparison")
    
    return len(successful_results) > 0

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉' if success else '💥'} Baseline comparison {'completed' if success else 'failed'}")