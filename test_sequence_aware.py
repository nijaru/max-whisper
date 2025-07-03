#!/usr/bin/env python3
"""
Test script for sequence-aware self-attention implementation
Tests the helper methods without requiring MAX Graph
"""

import numpy as np

class MockMaxGraphWhisperDecoder:
    """Mock decoder for testing sequence helper methods"""
    
    def __init__(self):
        self.max_seq_len = 32  # Small for testing
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create a causal mask for autoregressive generation"""
        # Lower triangular matrix: 1 for allowed positions, 0 for masked
        mask = np.tril(np.ones((self.max_seq_len, self.max_seq_len), dtype=np.float32))
        return mask
    
    def _prepare_sequence_inputs(self, tokens: list) -> tuple:
        """Prepare sequence inputs with proper padding and masking"""
        # Pad or truncate tokens to max_seq_len
        seq_len = min(len(tokens), self.max_seq_len)
        
        # Create padded sequence
        padded_sequence = np.zeros((1, self.max_seq_len), dtype=np.int32)
        padded_sequence[0, :seq_len] = tokens[:seq_len]
        
        # Sequence length
        sequence_length = np.array([seq_len], dtype=np.int32)
        
        # Causal mask
        causal_mask = self._create_causal_mask(seq_len)
        
        return padded_sequence, sequence_length, causal_mask

def test_sequence_helpers():
    """Test sequence-aware helper methods"""
    print("🧪 Testing sequence-aware helper methods...")
    
    try:
        # Create mock decoder
        decoder = MockMaxGraphWhisperDecoder()
        print("✅ Mock decoder created")
        
        # Test with various token sequences
        test_cases = [
            [50258, 50363, 562, 291],  # 4 tokens
            [50258, 50363],  # 2 tokens
            [50258, 50363, 562, 291, 1133, 498, 383, 314, 770, 921, 262],  # 11 tokens
            list(range(50)),  # 50 tokens (will be truncated)
        ]
        
        for i, tokens in enumerate(test_cases):
            print(f"\n📝 Test case {i+1}: {len(tokens)} tokens")
            
            padded_seq, seq_len, mask = decoder._prepare_sequence_inputs(tokens)
            
            print(f"   Input tokens: {tokens[:8]}{'...' if len(tokens) > 8 else ''}")
            print(f"   Padded shape: {padded_seq.shape}")
            print(f"   Sequence length: {seq_len[0]}")
            print(f"   Mask shape: {mask.shape}")
            print(f"   First 8 padded tokens: {padded_seq[0, :8]}")
            
            # Verify causal mask properties
            expected_len = min(len(tokens), decoder.max_seq_len)
            assert seq_len[0] == expected_len, f"Sequence length mismatch: {seq_len[0]} != {expected_len}"
            
            # Check causal mask is lower triangular
            upper_triangle = np.triu(mask, k=1)  # Upper triangle excluding diagonal
            assert np.all(upper_triangle == 0), "Causal mask should be lower triangular"
            
            # Check diagonal is 1
            diagonal = np.diag(mask)
            assert np.all(diagonal == 1), "Causal mask diagonal should be 1"
            
            print("   ✅ All assertions passed")
        
        # Test causal mask visualization
        print(f"\n🔍 Causal mask visualization (8x8):")
        test_tokens = [50258, 50363, 562, 291, 1133]
        _, _, mask = decoder._prepare_sequence_inputs(test_tokens)
        
        for i in range(8):
            row_str = " ".join([f"{int(mask[i, j]):1d}" for j in range(8)])
            print(f"   Row {i}: {row_str}")
        
        print("\n✅ All sequence-aware helper tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sequence_helpers()
    if success:
        print("\n🎉 Sequence-aware implementation ready for MAX Graph integration!")
    else:
        print("\n💥 Tests failed - implementation needs fixes")