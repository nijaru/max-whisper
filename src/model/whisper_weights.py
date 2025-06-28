"""
Whisper weight loading utilities for MAX Graph implementation.
Loads pre-trained weights from OpenAI Whisper models.
"""

import numpy as np
import os
from typing import Dict, Any, Optional
import json

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    print("OpenAI Whisper not available - install with: pip install openai-whisper")
    WHISPER_AVAILABLE = False


class WhisperWeightLoader:
    """Load and convert Whisper weights for MAX Graph."""
    
    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self.weights_dir = "models/whisper_weights"
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Model dimensions
        self.dims = {
            "tiny": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
                "n_text_ctx": 448,
                "n_text_state": 384,
                "n_text_head": 6,
                "n_text_layer": 4,
                "n_vocab": 51865,
            },
            "base": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 512,
                "n_audio_head": 8,
                "n_audio_layer": 6,
                "n_text_ctx": 448,
                "n_text_state": 512,
                "n_text_head": 8,
                "n_text_layer": 6,
                "n_vocab": 51865,
            }
        }
        
        self.config = self.dims[model_size]
    
    def load_whisper_model(self) -> Optional[Dict[str, np.ndarray]]:
        """Load OpenAI Whisper model and extract weights."""
        if not WHISPER_AVAILABLE:
            print("Using random weights - OpenAI Whisper not available")
            return self._generate_random_weights()
        
        # Check if weights already extracted
        weights_file = os.path.join(self.weights_dir, f"whisper_{self.model_size}_weights.npz")
        if os.path.exists(weights_file):
            print(f"Loading cached weights from {weights_file}")
            return dict(np.load(weights_file))
        
        print(f"Loading Whisper {self.model_size} model...")
        model = whisper.load_model(self.model_size)
        
        # Extract weights
        weights = {}
        
        # Encoder weights
        print("Extracting encoder weights...")
        encoder_weights = self._extract_encoder_weights(model.encoder)
        weights.update(encoder_weights)
        
        # Decoder weights
        print("Extracting decoder weights...")
        decoder_weights = self._extract_decoder_weights(model.decoder)
        weights.update(decoder_weights)
        
        # Save weights
        print(f"Saving weights to {weights_file}")
        np.savez_compressed(weights_file, **weights)
        
        return weights
    
    def _extract_encoder_weights(self, encoder) -> Dict[str, np.ndarray]:
        """Extract encoder weights."""
        weights = {}
        
        # Conv layers
        weights['encoder.conv1.weight'] = encoder.conv1.weight.cpu().numpy()
        weights['encoder.conv1.bias'] = encoder.conv1.bias.cpu().numpy()
        weights['encoder.conv2.weight'] = encoder.conv2.weight.cpu().numpy()
        weights['encoder.conv2.bias'] = encoder.conv2.bias.cpu().numpy()
        
        # Positional embedding
        weights['encoder.positional_embedding'] = encoder.positional_embedding.cpu().numpy()
        
        # Transformer blocks
        for i, block in enumerate(encoder.blocks):
            prefix = f'encoder.blocks.{i}'
            
            # Multi-head attention
            weights[f'{prefix}.attn.qkv.weight'] = block.attn.qkv.weight.cpu().numpy()
            weights[f'{prefix}.attn.out.weight'] = block.attn.out.weight.cpu().numpy()
            weights[f'{prefix}.attn.out.bias'] = block.attn.out.bias.cpu().numpy()
            
            # Layer norms
            weights[f'{prefix}.attn_ln.weight'] = block.attn_ln.weight.cpu().numpy()
            weights[f'{prefix}.attn_ln.bias'] = block.attn_ln.bias.cpu().numpy()
            weights[f'{prefix}.mlp_ln.weight'] = block.mlp_ln.weight.cpu().numpy()
            weights[f'{prefix}.mlp_ln.bias'] = block.mlp_ln.bias.cpu().numpy()
            
            # MLP
            weights[f'{prefix}.mlp.0.weight'] = block.mlp[0].weight.cpu().numpy()
            weights[f'{prefix}.mlp.0.bias'] = block.mlp[0].bias.cpu().numpy()
            weights[f'{prefix}.mlp.2.weight'] = block.mlp[2].weight.cpu().numpy()
            weights[f'{prefix}.mlp.2.bias'] = block.mlp[2].bias.cpu().numpy()
        
        # Final layer norm
        weights['encoder.ln_post.weight'] = encoder.ln_post.weight.cpu().numpy()
        weights['encoder.ln_post.bias'] = encoder.ln_post.bias.cpu().numpy()
        
        return weights
    
    def _extract_decoder_weights(self, decoder) -> Dict[str, np.ndarray]:
        """Extract decoder weights."""
        weights = {}
        
        # Token embedding
        weights['decoder.token_embedding.weight'] = decoder.token_embedding.weight.cpu().numpy()
        
        # Positional embedding
        weights['decoder.positional_embedding'] = decoder.positional_embedding.cpu().numpy()
        
        # Transformer blocks
        for i, block in enumerate(decoder.blocks):
            prefix = f'decoder.blocks.{i}'
            
            # Self-attention
            weights[f'{prefix}.attn.qkv.weight'] = block.attn.qkv.weight.cpu().numpy()
            weights[f'{prefix}.attn.out.weight'] = block.attn.out.weight.cpu().numpy()
            weights[f'{prefix}.attn.out.bias'] = block.attn.out.bias.cpu().numpy()
            
            # Cross-attention
            weights[f'{prefix}.cross_attn.q.weight'] = block.cross_attn.q.weight.cpu().numpy()
            weights[f'{prefix}.cross_attn.kv.weight'] = block.cross_attn.kv.weight.cpu().numpy()
            weights[f'{prefix}.cross_attn.out.weight'] = block.cross_attn.out.weight.cpu().numpy()
            weights[f'{prefix}.cross_attn.out.bias'] = block.cross_attn.out.bias.cpu().numpy()
            
            # Layer norms
            weights[f'{prefix}.attn_ln.weight'] = block.attn_ln.weight.cpu().numpy()
            weights[f'{prefix}.attn_ln.bias'] = block.attn_ln.bias.cpu().numpy()
            weights[f'{prefix}.cross_attn_ln.weight'] = block.cross_attn_ln.weight.cpu().numpy()
            weights[f'{prefix}.cross_attn_ln.bias'] = block.cross_attn_ln.bias.cpu().numpy()
            weights[f'{prefix}.mlp_ln.weight'] = block.mlp_ln.weight.cpu().numpy()
            weights[f'{prefix}.mlp_ln.bias'] = block.mlp_ln.bias.cpu().numpy()
            
            # MLP
            weights[f'{prefix}.mlp.0.weight'] = block.mlp[0].weight.cpu().numpy()
            weights[f'{prefix}.mlp.0.bias'] = block.mlp[0].bias.cpu().numpy()
            weights[f'{prefix}.mlp.2.weight'] = block.mlp[2].weight.cpu().numpy()
            weights[f'{prefix}.mlp.2.bias'] = block.mlp[2].bias.cpu().numpy()
        
        # Final layer norm
        weights['decoder.ln.weight'] = decoder.ln.weight.cpu().numpy()
        weights['decoder.ln.bias'] = decoder.ln.bias.cpu().numpy()
        
        return weights
    
    def _generate_random_weights(self) -> Dict[str, np.ndarray]:
        """Generate random weights with correct shapes."""
        weights = {}
        cfg = self.config
        
        # Simplified random weights for demo
        # Encoder conv layers
        weights['encoder.conv1.weight'] = np.random.randn(cfg['n_audio_state'], cfg['n_mels'], 3).astype(np.float32) * 0.02
        weights['encoder.conv1.bias'] = np.zeros(cfg['n_audio_state']).astype(np.float32)
        
        # Positional embeddings
        weights['encoder.positional_embedding'] = np.random.randn(cfg['n_audio_ctx'] // 2, cfg['n_audio_state']).astype(np.float32) * 0.02
        weights['decoder.positional_embedding'] = np.random.randn(cfg['n_text_ctx'], cfg['n_text_state']).astype(np.float32) * 0.02
        
        # Token embedding
        weights['decoder.token_embedding.weight'] = np.random.randn(cfg['n_vocab'], cfg['n_text_state']).astype(np.float32) * 0.02
        
        return weights
    
    def get_tokenizer(self):
        """Get Whisper tokenizer."""
        if WHISPER_AVAILABLE:
            # Use the actual Whisper tokenizer
            return whisper.tokenizer.get_tokenizer(multilingual=False)
        else:
            # Simple mock tokenizer for demo
            class MockTokenizer:
                def encode(self, text):
                    # Simple character-level encoding
                    return [ord(c) for c in text[:100]]
                
                def decode(self, tokens):
                    # Simple decoding
                    try:
                        return ''.join([chr(t) for t in tokens if 32 <= t < 127])
                    except:
                        return "[decoded text]"
            
            return MockTokenizer()


# Singleton instance
_weight_loader = None

def get_weight_loader(model_size: str = "tiny") -> WhisperWeightLoader:
    """Get or create weight loader instance."""
    global _weight_loader
    if _weight_loader is None or _weight_loader.model_size != model_size:
        _weight_loader = WhisperWeightLoader(model_size)
    return _weight_loader