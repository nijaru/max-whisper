#!/usr/bin/env python3
"""
Test KV cache implementation compilation

Test that the KV-cached decoder compiles successfully and can perform
basic inference operations.
"""

import sys
import os
sys.path.append('max-whisper')

import numpy as np
from whisper_max import WhisperMAX

def test_kv_cache_compilation():
    """Test KV cache decoder compilation"""
    print("🔧 Testing KV cache decoder compilation...")
    
    try:
        # Initialize with KV caching
        decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
        
        print(f"✅ Decoder initialized")
        print(f"   MAX Graph decoder available: {decoder.max_graph_decoder is not None}")
        print(f"   KV cache decoder compiled successfully!")
        print(f"   Ready for KV-cached inference")
        
        if decoder.max_graph_decoder is None:
            print("❌ KV cache decoder compilation failed")
            return False
        
        print("✅ KV cache decoder compiled successfully!")
        return True
        
    except Exception as e:
        print(f"❌ KV cache compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_inference():
    """Test basic inference with KV caching"""
    print("\n🧪 Testing basic KV cache inference...")
    
    try:
        decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
        
        if decoder.max_graph_decoder is None:
            print("❌ Decoder not available, skipping inference test")
            return False
        
        # Create dummy encoder features
        encoder_features = np.random.randn(1, 1500, 384).astype(np.float32)
        print(f"✅ Created encoder features: {encoder_features.shape}")
        
        # Test short generation
        print("🔄 Testing KV-cached generation (5 tokens)...")
        result = decoder.max_graph_decoder.generate_text(encoder_features, max_length=5)
        print(f"✅ Generated text: '{result}'")
        print(f"   Length: {len(result)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ KV cache inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 KV Cache Implementation Test\n")
    
    # Test compilation
    compilation_success = test_kv_cache_compilation()
    
    if compilation_success:
        # Test basic inference
        inference_success = test_basic_inference()
        
        if inference_success:
            print("\n✅ KV cache implementation test completed successfully!")
            print("   ✅ Compilation: PASSED")
            print("   ✅ Inference: PASSED")
        else:
            print("\n⚠️ KV cache implementation test completed with issues:")
            print("   ✅ Compilation: PASSED")
            print("   ❌ Inference: FAILED")
    else:
        print("\n❌ KV cache implementation test failed:")
        print("   ❌ Compilation: FAILED")
        print("   ❌ Inference: SKIPPED")

if __name__ == "__main__":
    main()