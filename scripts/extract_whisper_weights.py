"""
Extract weights from OpenAI Whisper-tiny model for MAX-Whisper
"""

import numpy as np
import torch
import whisper
import os

def extract_whisper_weights():
    """Extract key weights from OpenAI Whisper-tiny model"""
    print("Loading OpenAI Whisper-tiny model...")
    
    # Load the model
    model = whisper.load_model("tiny")
    
    print("Model structure:")
    print(f"  Encoder layers: {len(model.encoder.blocks)}")
    print(f"  Decoder layers: {len(model.decoder.blocks)}")
    
    # Extract key weights
    weights = {}
    
    # Token embeddings (critical for text generation)
    print("\nExtracting token embeddings...")
    weights['token_embedding'] = model.decoder.token_embedding.weight.detach().cpu().numpy()
    print(f"  Token embedding shape: {weights['token_embedding'].shape}")  # Should be (51865, 384)
    
    # Positional embeddings
    print("Extracting positional embeddings...")
    weights['positional_embedding'] = model.decoder.positional_embedding.detach().cpu().numpy()
    print(f"  Positional embedding shape: {weights['positional_embedding'].shape}")  # Should be (224, 384)
    
    # Encoder input projection
    print("Extracting encoder input projection...")
    weights['encoder_conv1_weight'] = model.encoder.conv1.weight.detach().cpu().numpy()
    weights['encoder_conv1_bias'] = model.encoder.conv1.bias.detach().cpu().numpy()
    print(f"  Encoder conv1 weight shape: {weights['encoder_conv1_weight'].shape}")
    
    weights['encoder_conv2_weight'] = model.encoder.conv2.weight.detach().cpu().numpy()
    weights['encoder_conv2_bias'] = model.encoder.conv2.bias.detach().cpu().numpy()
    print(f"  Encoder conv2 weight shape: {weights['encoder_conv2_weight'].shape}")
    
    # First encoder layer weights (most critical)
    print("Extracting first encoder layer...")
    enc_layer_0 = model.encoder.blocks[0]
    
    # Attention weights
    weights['enc_0_attn_query_weight'] = enc_layer_0.attn.query.weight.detach().cpu().numpy()
    weights['enc_0_attn_query_bias'] = enc_layer_0.attn.query.bias.detach().cpu().numpy()
    weights['enc_0_attn_key_weight'] = enc_layer_0.attn.key.weight.detach().cpu().numpy()
    weights['enc_0_attn_value_weight'] = enc_layer_0.attn.value.weight.detach().cpu().numpy()
    weights['enc_0_attn_value_bias'] = enc_layer_0.attn.value.bias.detach().cpu().numpy()
    weights['enc_0_attn_out_weight'] = enc_layer_0.attn.out.weight.detach().cpu().numpy()
    weights['enc_0_attn_out_bias'] = enc_layer_0.attn.out.bias.detach().cpu().numpy()
    
    print(f"  Encoder attention query weight shape: {weights['enc_0_attn_query_weight'].shape}")
    
    # Layer norm weights
    weights['enc_0_ln1_weight'] = enc_layer_0.attn_ln.weight.detach().cpu().numpy()
    weights['enc_0_ln1_bias'] = enc_layer_0.attn_ln.bias.detach().cpu().numpy()
    weights['enc_0_ln2_weight'] = enc_layer_0.mlp_ln.weight.detach().cpu().numpy()
    weights['enc_0_ln2_bias'] = enc_layer_0.mlp_ln.bias.detach().cpu().numpy()
    
    # MLP weights
    weights['enc_0_mlp_0_weight'] = enc_layer_0.mlp[0].weight.detach().cpu().numpy()
    weights['enc_0_mlp_0_bias'] = enc_layer_0.mlp[0].bias.detach().cpu().numpy()
    weights['enc_0_mlp_2_weight'] = enc_layer_0.mlp[2].weight.detach().cpu().numpy()
    weights['enc_0_mlp_2_bias'] = enc_layer_0.mlp[2].bias.detach().cpu().numpy()
    
    # First decoder layer weights (most critical)
    print("Extracting first decoder layer...")
    dec_layer_0 = model.decoder.blocks[0]
    
    # Self-attention
    weights['dec_0_self_attn_query_weight'] = dec_layer_0.attn.query.weight.detach().cpu().numpy()
    weights['dec_0_self_attn_query_bias'] = dec_layer_0.attn.query.bias.detach().cpu().numpy()
    weights['dec_0_self_attn_key_weight'] = dec_layer_0.attn.key.weight.detach().cpu().numpy()
    weights['dec_0_self_attn_value_weight'] = dec_layer_0.attn.value.weight.detach().cpu().numpy()
    weights['dec_0_self_attn_value_bias'] = dec_layer_0.attn.value.bias.detach().cpu().numpy()
    weights['dec_0_self_attn_out_weight'] = dec_layer_0.attn.out.weight.detach().cpu().numpy()
    weights['dec_0_self_attn_out_bias'] = dec_layer_0.attn.out.bias.detach().cpu().numpy()
    
    # Cross-attention
    weights['dec_0_cross_attn_query_weight'] = dec_layer_0.cross_attn.query.weight.detach().cpu().numpy()
    weights['dec_0_cross_attn_query_bias'] = dec_layer_0.cross_attn.query.bias.detach().cpu().numpy()
    weights['dec_0_cross_attn_key_weight'] = dec_layer_0.cross_attn.key.weight.detach().cpu().numpy()
    weights['dec_0_cross_attn_value_weight'] = dec_layer_0.cross_attn.value.weight.detach().cpu().numpy()
    weights['dec_0_cross_attn_value_bias'] = dec_layer_0.cross_attn.value.bias.detach().cpu().numpy()
    weights['dec_0_cross_attn_out_weight'] = dec_layer_0.cross_attn.out.weight.detach().cpu().numpy()
    weights['dec_0_cross_attn_out_bias'] = dec_layer_0.cross_attn.out.bias.detach().cpu().numpy()
    
    print(f"  Decoder cross-attention query weight shape: {weights['dec_0_cross_attn_query_weight'].shape}")
    
    # Layer norms
    weights['dec_0_ln1_weight'] = dec_layer_0.attn_ln.weight.detach().cpu().numpy()
    weights['dec_0_ln1_bias'] = dec_layer_0.attn_ln.bias.detach().cpu().numpy()
    weights['dec_0_ln2_weight'] = dec_layer_0.cross_attn_ln.weight.detach().cpu().numpy()
    weights['dec_0_ln2_bias'] = dec_layer_0.cross_attn_ln.bias.detach().cpu().numpy()
    weights['dec_0_ln3_weight'] = dec_layer_0.mlp_ln.weight.detach().cpu().numpy()
    weights['dec_0_ln3_bias'] = dec_layer_0.mlp_ln.bias.detach().cpu().numpy()
    
    # MLP
    weights['dec_0_mlp_0_weight'] = dec_layer_0.mlp[0].weight.detach().cpu().numpy()
    weights['dec_0_mlp_0_bias'] = dec_layer_0.mlp[0].bias.detach().cpu().numpy()
    weights['dec_0_mlp_2_weight'] = dec_layer_0.mlp[2].weight.detach().cpu().numpy()
    weights['dec_0_mlp_2_bias'] = dec_layer_0.mlp[2].bias.detach().cpu().numpy()
    
    # Final layer norm and output projection (critical for text generation)
    print("Extracting final layers...")
    weights['decoder_ln_weight'] = model.decoder.ln.weight.detach().cpu().numpy()
    weights['decoder_ln_bias'] = model.decoder.ln.bias.detach().cpu().numpy()
    
    # The token_embedding is also used as output projection in Whisper (tied weights)
    # So we don't need a separate extraction for that
    
    print(f"\\nExtracted {len(weights)} weight tensors")
    
    # Save weights
    os.makedirs("whisper_weights", exist_ok=True)
    np.savez_compressed("whisper_weights/whisper_tiny_weights.npz", **weights)
    print(f"✅ Saved weights to whisper_weights/whisper_tiny_weights.npz")
    
    # Print summary
    print(f"\\nWeight Summary:")
    print(f"  Token embedding: {weights['token_embedding'].shape} - Maps tokens to features")
    print(f"  Positional embedding: {weights['positional_embedding'].shape} - Position encoding")
    print(f"  Encoder conv1: {weights['encoder_conv1_weight'].shape} - Audio input projection")
    print(f"  Decoder final: {weights['decoder_ln_weight'].shape} - Output layer norm")
    print(f"\\n🎯 Key insight: These weights will enable meaningful text generation!")
    
    return weights

def test_weight_loading():
    """Test that we can load the saved weights"""
    print("\\n" + "="*50)
    print("Testing weight loading...")
    
    if not os.path.exists("whisper_weights/whisper_tiny_weights.npz"):
        print("❌ Weight file not found")
        return False
    
    weights = np.load("whisper_weights/whisper_tiny_weights.npz")
    print(f"✅ Loaded {len(weights.files)} weight tensors")
    
    # Check key weights
    key_weights = ['token_embedding', 'positional_embedding', 'decoder_ln_weight']
    for key in key_weights:
        if key in weights:
            print(f"  ✅ {key}: {weights[key].shape}")
        else:
            print(f"  ❌ Missing: {key}")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("WHISPER WEIGHT EXTRACTION")
    print("="*60)
    
    try:
        weights = extract_whisper_weights()
        test_weight_loading()
        
        print(f"\\n🎉 SUCCESS: Ready to integrate weights into MAX-Whisper!")
        print(f"Next step: Modify MAX-Whisper to use these trained weights")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()