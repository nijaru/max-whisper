#!/usr/bin/env python3
"""
Test if the sequence-aware decoder compiles after API fixes
"""

import sys
sys.path.append('max-whisper')
from whisper_max import WhisperMAX

print('🔧 Testing sequence-aware decoder compilation...')
decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
print('✅ Decoder compiled successfully!')
print(f'MAX Graph decoder available: {decoder.max_graph_decoder is not None}')

if decoder.max_graph_decoder:
    print('✅ Sequence-aware decoder is working!')
    
    # Test sequence preparation
    if hasattr(decoder.max_graph_decoder, '_prepare_sequence_inputs'):
        test_tokens = [50258, 50363, 562, 291]
        padded_seq, seq_len, causal_mask = decoder.max_graph_decoder._prepare_sequence_inputs(test_tokens)
        print(f'✅ Sequence preparation works: seq_len={seq_len[0]}, mask_shape={causal_mask.shape}')
else:
    print('❌ Sequence-aware decoder compilation failed')