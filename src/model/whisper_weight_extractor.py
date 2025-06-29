#!/usr/bin/env python3
"""
Whisper Weight Extractor for MAX Graph
Extracts pretrained weights from OpenAI Whisper models and prepares them for MAX Graph usage
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
import json

try:
    import whisper
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper libraries not available")


class WhisperWeightExtractor:
    """
    Extracts and processes weights from pretrained Whisper models
    Converts PyTorch tensors to numpy arrays suitable for MAX Graph
    """
    
    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self.model_configs = {
            "tiny": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 384,
                "n_audio_head": 6,
                "n_audio_layer": 4,
                "n_vocab": 51865,
                "n_text_ctx": 224,
                "n_text_state": 384,
                "n_text_head": 6,
                "n_text_layer": 4
            },
            "small": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 768,
                "n_audio_head": 12,
                "n_audio_layer": 12,
                "n_vocab": 51865,
                "n_text_ctx": 224,
                "n_text_state": 768,
                "n_text_head": 12,
                "n_text_layer": 12
            },
            "base": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 512,
                "n_audio_head": 8,
                "n_audio_layer": 6,
                "n_vocab": 51865,
                "n_text_ctx": 224,
                "n_text_state": 512,
                "n_text_head": 8,
                "n_text_layer": 6
            }
        }
        
        if model_size not in self.model_configs:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        self.config = self.model_configs[model_size]
        self.weights = {}
        
    def extract_openai_whisper_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract weights from OpenAI Whisper model
        
        Returns:
            Dictionary containing all model weights as numpy arrays
        """
        if not WHISPER_AVAILABLE:
            print("❌ Cannot extract weights - Whisper not available")
            return {}
        
        print(f"🔧 Loading OpenAI Whisper {self.model_size} model...")
        
        try:
            # Load OpenAI Whisper model
            model = whisper.load_model(self.model_size)
            model.eval()
            
            weights = {}
            
            # Extract encoder weights
            print("  📦 Extracting encoder weights...")
            encoder_weights = self._extract_encoder_weights(model.encoder)
            weights.update(encoder_weights)
            
            # Extract decoder weights
            print("  📦 Extracting decoder weights...")
            decoder_weights = self._extract_decoder_weights(model.decoder)
            weights.update(decoder_weights)
            
            # Extract other components
            print("  📦 Extracting additional components...")
            
            # Token embeddings
            if hasattr(model.decoder, 'token_embedding'):
                weights['token_embedding'] = model.decoder.token_embedding.weight.detach().cpu().numpy()
            
            # Positional embeddings
            if hasattr(model.decoder, 'positional_embedding'):
                weights['positional_embedding'] = model.decoder.positional_embedding.detach().cpu().numpy()
            
            print(f"✅ Successfully extracted {len(weights)} weight tensors")
            
            # Store config and weights
            self.weights = weights
            return weights
            
        except Exception as e:
            print(f"❌ Failed to extract OpenAI Whisper weights: {e}")
            return {}
    
    def _extract_encoder_weights(self, encoder) -> Dict[str, np.ndarray]:
        """Extract encoder-specific weights"""
        weights = {}
        
        try:
            # Convolutional layers
            if hasattr(encoder, 'conv1'):
                weights['encoder.conv1.weight'] = encoder.conv1.weight.detach().cpu().numpy()
                if encoder.conv1.bias is not None:
                    weights['encoder.conv1.bias'] = encoder.conv1.bias.detach().cpu().numpy()
            
            if hasattr(encoder, 'conv2'):
                weights['encoder.conv2.weight'] = encoder.conv2.weight.detach().cpu().numpy()
                if encoder.conv2.bias is not None:
                    weights['encoder.conv2.bias'] = encoder.conv2.bias.detach().cpu().numpy()
            
            # Positional embedding
            if hasattr(encoder, 'positional_embedding'):
                weights['encoder.positional_embedding'] = encoder.positional_embedding.detach().cpu().numpy()
            
            # Layer normalization
            if hasattr(encoder, 'ln_post'):
                weights['encoder.ln_post.weight'] = encoder.ln_post.weight.detach().cpu().numpy()
                weights['encoder.ln_post.bias'] = encoder.ln_post.bias.detach().cpu().numpy()
            
            # Transformer blocks
            if hasattr(encoder, 'blocks'):
                for layer_idx, block in enumerate(encoder.blocks):
                    layer_prefix = f'encoder.blocks.{layer_idx}'
                    
                    # Multi-head attention
                    if hasattr(block, 'attn'):
                        attn = block.attn
                        if hasattr(attn, 'query'):
                            weights[f'{layer_prefix}.attn.query.weight'] = attn.query.weight.detach().cpu().numpy()
                            if attn.query.bias is not None:
                                weights[f'{layer_prefix}.attn.query.bias'] = attn.query.bias.detach().cpu().numpy()
                        
                        if hasattr(attn, 'key'):
                            weights[f'{layer_prefix}.attn.key.weight'] = attn.key.weight.detach().cpu().numpy()
                        
                        if hasattr(attn, 'value'):
                            weights[f'{layer_prefix}.attn.value.weight'] = attn.value.weight.detach().cpu().numpy()
                            if attn.value.bias is not None:
                                weights[f'{layer_prefix}.attn.value.bias'] = attn.value.bias.detach().cpu().numpy()
                        
                        if hasattr(attn, 'out'):
                            weights[f'{layer_prefix}.attn.out.weight'] = attn.out.weight.detach().cpu().numpy()
                            if attn.out.bias is not None:
                                weights[f'{layer_prefix}.attn.out.bias'] = attn.out.bias.detach().cpu().numpy()
                    
                    # Layer normalization
                    if hasattr(block, 'attn_ln'):
                        weights[f'{layer_prefix}.attn_ln.weight'] = block.attn_ln.weight.detach().cpu().numpy()
                        weights[f'{layer_prefix}.attn_ln.bias'] = block.attn_ln.bias.detach().cpu().numpy()
                    
                    if hasattr(block, 'mlp_ln'):
                        weights[f'{layer_prefix}.mlp_ln.weight'] = block.mlp_ln.weight.detach().cpu().numpy()
                        weights[f'{layer_prefix}.mlp_ln.bias'] = block.mlp_ln.bias.detach().cpu().numpy()
                    
                    # MLP
                    if hasattr(block, 'mlp'):
                        mlp = block.mlp
                        if hasattr(mlp, 'c_fc'):
                            weights[f'{layer_prefix}.mlp.c_fc.weight'] = mlp.c_fc.weight.detach().cpu().numpy()
                            if mlp.c_fc.bias is not None:
                                weights[f'{layer_prefix}.mlp.c_fc.bias'] = mlp.c_fc.bias.detach().cpu().numpy()
                        
                        if hasattr(mlp, 'c_proj'):
                            weights[f'{layer_prefix}.mlp.c_proj.weight'] = mlp.c_proj.weight.detach().cpu().numpy()
                            if mlp.c_proj.bias is not None:
                                weights[f'{layer_prefix}.mlp.c_proj.bias'] = mlp.c_proj.bias.detach().cpu().numpy()
            
            print(f"    ✅ Extracted {len([k for k in weights.keys() if k.startswith('encoder')])} encoder weights")
            
        except Exception as e:
            print(f"    ❌ Error extracting encoder weights: {e}")
        
        return weights
    
    def _extract_decoder_weights(self, decoder) -> Dict[str, np.ndarray]:
        """Extract decoder-specific weights"""
        weights = {}
        
        try:
            # Token embedding
            if hasattr(decoder, 'token_embedding'):
                weights['decoder.token_embedding.weight'] = decoder.token_embedding.weight.detach().cpu().numpy()
            
            # Positional embedding
            if hasattr(decoder, 'positional_embedding'):
                weights['decoder.positional_embedding'] = decoder.positional_embedding.detach().cpu().numpy()
            
            # Layer normalization
            if hasattr(decoder, 'ln'):
                weights['decoder.ln.weight'] = decoder.ln.weight.detach().cpu().numpy()
                weights['decoder.ln.bias'] = decoder.ln.bias.detach().cpu().numpy()
            
            # Transformer blocks
            if hasattr(decoder, 'blocks'):
                for layer_idx, block in enumerate(decoder.blocks):
                    layer_prefix = f'decoder.blocks.{layer_idx}'
                    
                    # Self-attention
                    if hasattr(block, 'attn'):
                        attn = block.attn
                        if hasattr(attn, 'query'):
                            weights[f'{layer_prefix}.attn.query.weight'] = attn.query.weight.detach().cpu().numpy()
                            if attn.query.bias is not None:
                                weights[f'{layer_prefix}.attn.query.bias'] = attn.query.bias.detach().cpu().numpy()
                        
                        if hasattr(attn, 'key'):
                            weights[f'{layer_prefix}.attn.key.weight'] = attn.key.weight.detach().cpu().numpy()
                        
                        if hasattr(attn, 'value'):
                            weights[f'{layer_prefix}.attn.value.weight'] = attn.value.weight.detach().cpu().numpy()
                            if attn.value.bias is not None:
                                weights[f'{layer_prefix}.attn.value.bias'] = attn.value.bias.detach().cpu().numpy()
                        
                        if hasattr(attn, 'out'):
                            weights[f'{layer_prefix}.attn.out.weight'] = attn.out.weight.detach().cpu().numpy()
                            if attn.out.bias is not None:
                                weights[f'{layer_prefix}.attn.out.bias'] = attn.out.bias.detach().cpu().numpy()
                    
                    # Cross-attention
                    if hasattr(block, 'cross_attn'):
                        cross_attn = block.cross_attn
                        if hasattr(cross_attn, 'query'):
                            weights[f'{layer_prefix}.cross_attn.query.weight'] = cross_attn.query.weight.detach().cpu().numpy()
                            if cross_attn.query.bias is not None:
                                weights[f'{layer_prefix}.cross_attn.query.bias'] = cross_attn.query.bias.detach().cpu().numpy()
                        
                        if hasattr(cross_attn, 'key'):
                            weights[f'{layer_prefix}.cross_attn.key.weight'] = cross_attn.key.weight.detach().cpu().numpy()
                        
                        if hasattr(cross_attn, 'value'):
                            weights[f'{layer_prefix}.cross_attn.value.weight'] = cross_attn.value.weight.detach().cpu().numpy()
                            if cross_attn.value.bias is not None:
                                weights[f'{layer_prefix}.cross_attn.value.bias'] = cross_attn.value.bias.detach().cpu().numpy()
                        
                        if hasattr(cross_attn, 'out'):
                            weights[f'{layer_prefix}.cross_attn.out.weight'] = cross_attn.out.weight.detach().cpu().numpy()
                            if cross_attn.out.bias is not None:
                                weights[f'{layer_prefix}.cross_attn.out.bias'] = cross_attn.out.bias.detach().cpu().numpy()
                    
                    # Layer normalizations
                    if hasattr(block, 'attn_ln'):
                        weights[f'{layer_prefix}.attn_ln.weight'] = block.attn_ln.weight.detach().cpu().numpy()
                        weights[f'{layer_prefix}.attn_ln.bias'] = block.attn_ln.bias.detach().cpu().numpy()
                    
                    if hasattr(block, 'cross_attn_ln'):
                        weights[f'{layer_prefix}.cross_attn_ln.weight'] = block.cross_attn_ln.weight.detach().cpu().numpy()
                        weights[f'{layer_prefix}.cross_attn_ln.bias'] = block.cross_attn_ln.bias.detach().cpu().numpy()
                    
                    if hasattr(block, 'mlp_ln'):
                        weights[f'{layer_prefix}.mlp_ln.weight'] = block.mlp_ln.weight.detach().cpu().numpy()
                        weights[f'{layer_prefix}.mlp_ln.bias'] = block.mlp_ln.bias.detach().cpu().numpy()
                    
                    # MLP
                    if hasattr(block, 'mlp'):
                        mlp = block.mlp
                        if hasattr(mlp, 'c_fc'):
                            weights[f'{layer_prefix}.mlp.c_fc.weight'] = mlp.c_fc.weight.detach().cpu().numpy()
                            if mlp.c_fc.bias is not None:
                                weights[f'{layer_prefix}.mlp.c_fc.bias'] = mlp.c_fc.bias.detach().cpu().numpy()
                        
                        if hasattr(mlp, 'c_proj'):
                            weights[f'{layer_prefix}.mlp.c_proj.weight'] = mlp.c_proj.weight.detach().cpu().numpy()
                            if mlp.c_proj.bias is not None:
                                weights[f'{layer_prefix}.mlp.c_proj.bias'] = mlp.c_proj.bias.detach().cpu().numpy()
            
            print(f"    ✅ Extracted {len([k for k in weights.keys() if k.startswith('decoder')])} decoder weights")
            
        except Exception as e:
            print(f"    ❌ Error extracting decoder weights: {e}")
        
        return weights
    
    def save_weights(self, filepath: str) -> bool:
        """
        Save extracted weights to file
        
        Args:
            filepath: Path to save weights file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.weights:
            print("❌ No weights to save")
            return False
        
        try:
            # Save as compressed numpy archive
            np.savez_compressed(filepath, **self.weights)
            print(f"✅ Saved {len(self.weights)} weights to {filepath}")
            
            # Also save config
            config_path = filepath.replace('.npz', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Saved config to {config_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save weights: {e}")
            return False
    
    def load_weights(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load weights from file
        
        Args:
            filepath: Path to weights file
            
        Returns:
            Dictionary containing weights
        """
        try:
            weights_data = np.load(filepath)
            weights = {key: weights_data[key] for key in weights_data.files}
            
            print(f"✅ Loaded {len(weights)} weights from {filepath}")
            self.weights = weights
            return weights
            
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return {}
    
    def get_layer_weights(self, component: str, layer_idx: int) -> Dict[str, np.ndarray]:
        """
        Get weights for a specific layer
        
        Args:
            component: 'encoder' or 'decoder'
            layer_idx: Layer index
            
        Returns:
            Dictionary containing layer weights
        """
        layer_weights = {}
        layer_prefix = f'{component}.blocks.{layer_idx}'
        
        for weight_name, weight_tensor in self.weights.items():
            if weight_name.startswith(layer_prefix):
                # Remove the layer prefix to get relative weight name
                relative_name = weight_name[len(layer_prefix)+1:]  # +1 for the dot
                layer_weights[relative_name] = weight_tensor
        
        return layer_weights
    
    def get_weight_summary(self) -> Dict[str, Tuple[str, int]]:
        """
        Get summary of extracted weights
        
        Returns:
            Dictionary with weight names, shapes, and parameter counts
        """
        summary = {}
        total_params = 0
        
        for name, tensor in self.weights.items():
            shape_str = 'x'.join(map(str, tensor.shape))
            param_count = np.prod(tensor.shape)
            total_params += param_count
            summary[name] = (shape_str, param_count)
        
        summary['TOTAL_PARAMETERS'] = ('', total_params)
        return summary


# Test the weight extractor
if __name__ == "__main__":
    if not WHISPER_AVAILABLE:
        print("❌ Cannot test - Whisper not available")
        exit(1)
    
    print("🧪 Testing Whisper Weight Extractor...")
    
    # Test weight extraction
    for model_size in ["tiny"]:  # Start with tiny for testing
        print(f"\n📊 Testing {model_size} model...")
        
        try:
            extractor = WhisperWeightExtractor(model_size)
            weights = extractor.extract_openai_whisper_weights()
            
            if weights:
                # Print weight summary
                summary = extractor.get_weight_summary()
                print(f"\n📋 Weight Summary for {model_size}:")
                for name, (shape, count) in summary.items():
                    if name == 'TOTAL_PARAMETERS':
                        print(f"  🔢 Total Parameters: {count:,}")
                    else:
                        print(f"  {name}: {shape} ({count:,} params)")
                
                # Test saving and loading
                weights_file = f"whisper_{model_size}_weights.npz"
                if extractor.save_weights(weights_file):
                    print(f"✅ Successfully saved {model_size} weights")
                    
                    # Test loading
                    new_extractor = WhisperWeightExtractor(model_size)
                    loaded_weights = new_extractor.load_weights(weights_file)
                    
                    if len(loaded_weights) == len(weights):
                        print(f"✅ Successfully loaded {model_size} weights")
                    else:
                        print(f"❌ Weight count mismatch: {len(loaded_weights)} vs {len(weights)}")
                
                # Test layer weight extraction
                encoder_layer_0 = extractor.get_layer_weights('encoder', 0)
                print(f"  📦 Encoder layer 0 has {len(encoder_layer_0)} weight tensors")
                
            else:
                print(f"❌ Failed to extract {model_size} weights")
                
        except Exception as e:
            print(f"❌ Error testing {model_size}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Weight extractor testing complete!")