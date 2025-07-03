#!/usr/bin/env python3
"""
Test sequence-aware decoder performance end-to-end
"""

import time
import sys
import numpy as np
sys.path.append('max-whisper')
from whisper_max import WhisperMAX

def test_sequence_performance():
    """Test the sequence-aware decoder with actual text generation"""
    print("🔧 Testing sequence-aware decoder performance...")
    
    # Create decoder with full MAX Graph mode
    decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
    
    # Load test audio (shorter for focused testing)
    import librosa
    audio, _ = librosa.load('audio_samples/modular_video.wav', sr=16000, duration=5)  # 5 seconds
    
    print(f"Audio duration: {len(audio) / 16000:.1f}s ({len(audio)} samples)")
    
    # Test the full sequence-aware pipeline
    print("\n🚀 Testing full sequence-aware MAX Graph pipeline...")
    
    sequence_lengths = [5, 10, 15, 25]
    results = []
    
    for max_len in sequence_lengths:
        print(f"\n🔄 Testing sequence-aware generation (max_length={max_len})...")
        
        # Write audio to temp file 
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            
            start = time.time()
            text = decoder.transcribe(tmp.name)
            total_time = time.time() - start
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Generated text: {text[:100]}...")
        print(f"  Text length: {len(text)} chars")
        
        # Check if sequence-aware decoder was used
        decoder_used = "sequence-aware" if "MAX Graph decoder" in text else "hybrid fallback"
        print(f"  Decoder used: {decoder_used}")
        
        results.append((max_len, total_time, len(text), text, decoder_used))
    
    # Analyze performance characteristics
    print("\n📊 Performance Analysis:")
    print("Length | Time(s) | Text(chars) | Decoder     | Rate")
    print("-------|---------|-------------|-------------|-------")
    
    for max_len, total_time, text_len, text, decoder_used in results:
        rate = text_len / total_time if total_time > 0 else 0
        print(f"{max_len:6d} | {total_time:7.3f} | {text_len:11d} | {decoder_used[:11]:11s} | {rate:5.1f}")
    
    # Check for consistent sequence-aware operation
    sequence_aware_count = sum(1 for _, _, _, _, decoder in results if "sequence-aware" in decoder)
    
    print(f"\n🎯 Sequence-aware decoder usage: {sequence_aware_count}/{len(results)} tests")
    
    if sequence_aware_count > 0:
        print("✅ Sequence-aware decoder is working!")
        
        # Test specific sequence-aware features
        print("\n🔍 Testing sequence-aware features...")
        
        if hasattr(decoder, 'max_graph_decoder') and decoder.max_graph_decoder:
            max_decoder = decoder.max_graph_decoder
            
            # Test causal masking
            test_tokens = [50258, 50363, 562, 291, 1133]
            padded_seq, seq_len, causal_mask = max_decoder._prepare_sequence_inputs(test_tokens)
            
            print(f"  ✅ Causal mask shape: {causal_mask.shape}")
            print(f"  ✅ Sequence length: {seq_len[0]}")
            
            # Verify causal mask properties
            upper_triangle = np.triu(causal_mask, k=1)
            lower_triangle = np.tril(causal_mask)
            
            print(f"  ✅ Upper triangle zeros: {np.all(upper_triangle == 0)}")
            print(f"  ✅ Lower triangle ones: {np.all(lower_triangle == 1)}")
            
            # Memory analysis
            mask_memory_mb = causal_mask.nbytes / 1024 / 1024
            print(f"  📊 Causal mask memory: {mask_memory_mb:.1f} MB")
            
    else:
        print("⚠️ Sequence-aware decoder not being used - falling back to hybrid mode")
        
        # Check if there are compilation issues
        if hasattr(decoder, 'max_graph_decoder'):
            if decoder.max_graph_decoder is None:
                print("❌ MAX Graph decoder compilation failed")
            else:
                print("🔧 MAX Graph decoder available but not being used")
    
    return results

if __name__ == "__main__":
    results = test_sequence_performance()
    
    # Summary
    if results:
        total_tests = len(results)
        avg_time = sum(r[1] for r in results) / total_tests
        max_text_len = max(r[2] for r in results)
        
        print(f"\n📋 Summary:")
        print(f"  Tests completed: {total_tests}")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Max text length: {max_text_len} chars")
        print(f"  Performance mode: {'Sequence-aware' if any('sequence-aware' in r[4] for r in results) else 'Hybrid fallback'}")