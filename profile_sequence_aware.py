#!/usr/bin/env python3
"""
Profile the current sequence-aware implementation to identify bottlenecks
"""

import time
import sys
import numpy as np
sys.path.append('max-whisper')
from whisper_max import WhisperMAX

def profile_sequence_aware():
    """Profile current sequence-aware decoder performance"""
    print("🔧 Profiling current sequence-aware implementation...")
    
    # Create decoder with full MAX Graph mode
    decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
    
    # Load test audio (shorter for faster profiling)
    import librosa
    audio, _ = librosa.load('audio_samples/modular_video.wav', sr=16000, duration=10)  # 10 seconds
    
    print(f"Audio duration: {len(audio) / 16000:.1f}s ({len(audio)} samples)")
    
    # Profile the full transcription (which includes both encoder and decoder)
    print("\n📊 Profiling full transcription pipeline...")
    
    sequence_lengths = [5, 10, 15, 20]
    results = []
    
    for max_len in sequence_lengths:
        print(f"\n🔄 Testing max_length={max_len}...")
        
        start = time.time()
        # For now, use the main transcribe method (we'll analyze timings internally)
        # Write audio to temp file since transcribe expects file path
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            text = decoder.transcribe(tmp.name)
        total_time = time.time() - start
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Generated text: {text[:80]}...")
        print(f"  Tokens/second: {max_len / total_time:.1f}")
        
        results.append((max_len, total_time, text))
    
    # If decoder has sequence-aware methods, profile them
    if hasattr(decoder, 'max_graph_decoder') and decoder.max_graph_decoder:
        print("\n📈 Memory usage analysis...")
        
        # Test sequence preparation if available
        max_decoder = decoder.max_graph_decoder
        if hasattr(max_decoder, '_prepare_sequence_inputs'):
            test_tokens = [50258, 50363, 562, 291, 1133, 498, 383, 314, 770, 921]
            
            start = time.time()
            for _ in range(100):  # 100 iterations to measure overhead
                padded_seq, seq_len, causal_mask = max_decoder._prepare_sequence_inputs(test_tokens)
            prep_time = time.time() - start
            
            print(f"Sequence preparation (100x): {prep_time:.3f}s")
            print(f"Per iteration: {prep_time * 10:.3f}ms")
            print(f"Causal mask size: {causal_mask.shape} = {causal_mask.nbytes} bytes")
            
            # Calculate memory overhead
            max_seq_len = max_decoder.max_seq_len
            d_model = max_decoder.d_model
            causal_mask_memory = max_seq_len * max_seq_len * 4  # float32
            sequence_memory = max_seq_len * 4  # int32
            
            print(f"\n💾 Memory overhead per token:")
            print(f"  Causal mask: {causal_mask_memory:,} bytes ({causal_mask_memory/1024/1024:.1f} MB)")
            print(f"  Sequence buffer: {sequence_memory:,} bytes")
            print(f"  Total overhead: {(causal_mask_memory + sequence_memory)/1024/1024:.1f} MB")
    
    # Analyze results
    print("\n🎯 Performance analysis:")
    for max_len, total_time, text in results:
        tokens_per_sec = max_len / total_time
        print(f"  {max_len:2d} tokens: {total_time:.3f}s ({tokens_per_sec:.1f} tok/s)")
    
    # Check for scaling patterns
    if len(results) >= 2:
        time_per_token = [(total_time / max_len) for max_len, total_time, _ in results]
        print(f"\n📈 Time per token scaling:")
        for i, (max_len, _, _) in enumerate(results):
            print(f"  {max_len:2d} tokens: {time_per_token[i]*1000:.1f}ms/token")
        
        # Check if time scales linearly or quadratically
        if len(results) >= 3:
            ratio_1 = time_per_token[1] / time_per_token[0]
            ratio_2 = time_per_token[2] / time_per_token[1]
            print(f"\n🔍 Scaling analysis:")
            print(f"  Ratio 10/5: {ratio_1:.2f}x")
            print(f"  Ratio 15/10: {ratio_2:.2f}x")
            if ratio_1 > 1.5 or ratio_2 > 1.5:
                print("  ⚠️  Non-linear scaling detected - potential quadratic complexity")

if __name__ == "__main__":
    profile_sequence_aware()