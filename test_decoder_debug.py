#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'max-whisper'))

from whisper_max import WhisperMAX
import numpy as np

print("🔧 Testing MAX Graph decoder generation...")

try:
    decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
    print(f"MAX Graph decoder available: {decoder.max_graph_decoder is not None}")
    
    if decoder.max_graph_decoder:
        # Test with small encoder features
        test_features = np.random.randn(1, 1500, 384).astype(np.float32)
        print(f"Testing with features shape: {test_features.shape}")
        
        result = decoder.max_graph_decoder.generate_text(test_features, max_length=15)
        print(f"Generated text ({len(result)} chars): '{result}'")
        
        # Test multiple times to see consistency
        for i in range(3):
            result2 = decoder.max_graph_decoder.generate_text(test_features, max_length=10)
            print(f"Test {i+1}: '{result2}'")
    else:
        print("❌ MAX Graph decoder not available")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()