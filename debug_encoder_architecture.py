#!/usr/bin/env python3
"""
Debug script to systematically compare MAX Graph encoder vs OpenAI Whisper encoder
layer by layer to identify architectural differences.
"""

import numpy as np
import torch
import whisper

def analyze_openai_encoder_step_by_step():
    """Run OpenAI encoder step by step to understand the exact flow"""
    
    print("=== LOADING OPENAI WHISPER MODEL ===")
    model = whisper.load_model('tiny')
    encoder = model.encoder
    
    # Create synthetic mel spectrogram for testing
    # Typical mel spectrogram has small values, often in range [-80, 0] dB
    np.random.seed(42)  # For reproducible results
    mel_db = np.random.randn(80, 3000) * 20 - 40  # Range roughly [-100, 20] dB
    
    # Convert to tensor and add batch dimension
    x = torch.from_numpy(mel_db).float().unsqueeze(0)  # [1, 80, 3000]
    
    # Move to same device as model
    device = next(encoder.parameters()).device
    x = x.to(device)
    
    print(f"Input mel spectrogram shape: {x.shape}")
    print(f"Input statistics: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    
    # Step 1: Conv1
    print("\n=== CONV1 LAYER ===")
    x = encoder.conv1(x)  # [1, 384, 3000]
    print(f"After conv1 shape: {x.shape}")
    print(f"After conv1 stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    
    # Apply GELU
    x = torch.nn.functional.gelu(x)
    print(f"After conv1+GELU stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    
    # Step 2: Conv2  
    print("\n=== CONV2 LAYER ===")
    x = encoder.conv2(x)  # [1, 384, 1500] (stride=2 downsampling)
    print(f"After conv2 shape: {x.shape}")
    print(f"After conv2 stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    
    # Apply GELU
    x = torch.nn.functional.gelu(x)
    print(f"After conv2+GELU stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    
    # Step 3: Transpose and add positional embeddings
    print("\n=== POSITIONAL EMBEDDINGS ===")
    x = x.permute(0, 2, 1)  # [1, 1500, 384]
    print(f"After transpose shape: {x.shape}")
    print(f"Positional embedding shape: {encoder.positional_embedding.shape}")
    
    x = x + encoder.positional_embedding
    print(f"After pos_embed stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    
    # Step 4: Run through transformer blocks
    for i, block in enumerate(encoder.blocks):
        print(f"\n=== TRANSFORMER BLOCK {i} ===")
        
        # Pre-attention LayerNorm
        residual = x
        x_norm = block.attn_ln(x)
        print(f"After attn_ln stats: mean={x_norm.mean():.6f}, std={x_norm.std():.6f}")
        
        # Self-attention (returns tuple, take first element)
        attn_output, _ = block.attn(x_norm)
        print(f"After attention stats: mean={attn_output.mean():.6f}, std={attn_output.std():.6f}")
        
        # Residual connection
        x = residual + attn_output
        print(f"After attn residual stats: mean={x.mean():.6f}, std={x.std():.6f}")
        
        # Pre-MLP LayerNorm
        residual = x
        x_norm = block.mlp_ln(x)
        print(f"After mlp_ln stats: mean={x_norm.mean():.6f}, std={x_norm.std():.6f}")
        
        # MLP
        mlp_output = block.mlp(x_norm)
        print(f"After MLP stats: mean={mlp_output.mean():.6f}, std={mlp_output.std():.6f}")
        
        # Residual connection
        x = residual + mlp_output
        print(f"After MLP residual stats: mean={x.mean():.6f}, std={x.std():.6f}")
    
    # Step 5: Final layer norm
    print("\n=== FINAL LAYER NORM (ln_post) ===")
    x = encoder.ln_post(x)
    print(f"After ln_post stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    print(f"Final output shape: {x.shape}")
    
    return x.detach().numpy()

def compare_max_graph_architecture():
    """Compare our MAX Graph architecture against OpenAI step by step"""
    
    print("\n" + "="*60)
    print("COMPARING ARCHITECTURES")
    print("="*60)
    
    # Get OpenAI reference
    openai_output = analyze_openai_encoder_step_by_step()
    
    print(f"\n=== OPENAI FINAL OUTPUT ===")
    print(f"Shape: {openai_output.shape}")
    print(f"Mean: {np.mean(openai_output):.6f}")
    print(f"Std: {np.std(openai_output):.6f}")
    print(f"Min: {np.min(openai_output):.6f}")
    print(f"Max: {np.max(openai_output):.6f}")
    
    # Check specific potential issues with MAX Graph implementation
    print(f"\n=== ARCHITECTURE ANALYSIS ===")
    print("Checking for common implementation issues:")
    
    print("\n1. Conv1D Implementation:")
    print("   - OpenAI uses Conv1d with kernel=3, stride=1, padding=1")
    print("   - MAX Graph uses Conv2d emulation with proper kernel/stride mapping")
    print("   - ✓ Should be equivalent")
    
    print("\n2. Conv1D Stride=2 Downsampling:")
    print("   - OpenAI uses Conv1d with kernel=3, stride=2, padding=1")
    print("   - This reduces sequence length from 3000 → 1500")
    print("   - MAX Graph implementation needs to match exactly")
    
    print("\n3. Attention Mechanism:")
    print("   - OpenAI uses standard multi-head attention")
    print("   - Query has bias=True, Key has bias=False, Value has bias=True, Out has bias=True")
    print("   - Scaling by 1/sqrt(head_dim) = 1/sqrt(64) = 0.125")
    
    print("\n4. Layer Normalization:")
    print("   - OpenAI uses eps=1e-5 for all LayerNorm")
    print("   - Applied before attention and MLP (pre-norm)")
    print("   - Final ln_post applied at the end")
    
    print("\n5. MLP Structure:")
    print("   - Linear(384 → 1536) + GELU + Linear(1536 → 384)")
    print("   - Both linear layers have bias=True")
    
    print("\n6. Residual Connections:")
    print("   - Applied after both attention and MLP")
    print("   - Standard: output = input + sublayer(norm(input))")

if __name__ == "__main__":
    compare_max_graph_architecture()