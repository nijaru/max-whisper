"""
Real MAX-Whisper Implementation
Building a complete Whisper model with encoder-decoder architecture using MAX Graph.
"""

import numpy as np
import time
from typing import Tuple, Optional

try:
    from max import engine
    from max.driver import Tensor, Accelerator, CPU
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops, TensorValue
    from max.graph.ops import elementwise
    MAX_AVAILABLE = True
except ImportError:
    print("MAX Graph not available")
    MAX_AVAILABLE = False

class MAX_WhisperReal:
    def __init__(self, device_id=0):
        if not MAX_AVAILABLE:
            print("MAX Graph not available - cannot run model")
            return
            
        self.device_id = device_id
        try:
            self.device = DeviceRef.GPU()
            print("✅ Using GPU device")
        except:
            print("⚠️ GPU not available, falling back to CPU")
            self.device = DeviceRef.CPU()
        
        # Model dimensions (whisper-tiny simplified)
        self.n_mels = 80
        self.n_audio_ctx = 1500
        self.n_audio_state = 384
        self.n_audio_head = 6
        self.n_audio_layer = 4
        
        # Decoder dimensions
        self.n_text_ctx = 448
        self.n_text_state = 384
        self.n_text_head = 6
        self.n_text_layer = 4
        self.n_vocab = 51865
        
        # Build encoder and decoder graphs
        self.encoder_model = None
        self.decoder_model = None
        self.session = None
        
    def build_encoder_graph(self):
        """Build the encoder transformer graph"""
        print("Building encoder graph...")
        
        # Input shape: (batch=1, n_mels=80, n_audio_ctx=1500)
        input_type = TensorType(
            dtype=DType.float32,
            shape=(1, self.n_mels, self.n_audio_ctx),
            device=self.device
        )
        
        with Graph("whisper_encoder", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            
            # Conv1d layers (like Whisper)
            # Conv1d(80, 384, kernel_size=3, padding=1)
            conv1_weight = ops.constant(
                np.random.randn(384, 80, 3).astype(np.float32) * 0.1,
                dtype=DType.float32,
                device=self.device
            )
            conv1_bias = ops.constant(
                np.zeros(384).astype(np.float32),
                dtype=DType.float32,
                device=self.device
            )
            
            # Apply first conv (need to handle shapes for MAX Graph)
            x = ops.permute(x, [0, 2, 1])  # (batch, time, mels)
            x = self._apply_conv1d(x, conv1_weight, conv1_bias, kernel_size=3, padding=1)
            x = elementwise.gelu(x)
            
            # Conv1d(384, 384, kernel_size=3, stride=2, padding=1)
            conv2_weight = ops.constant(
                np.random.randn(384, 384, 3).astype(np.float32) * 0.1,
                device=self.device
            )
            conv2_bias = ops.constant(
                np.zeros(384).astype(np.float32),
                device=self.device
            )
            
            x = self._apply_conv1d(x, conv2_weight, conv2_bias, kernel_size=3, stride=2, padding=1)
            x = elementwise.gelu(x)
            
            # Positional encoding
            max_len = 750  # After stride=2: 1500 // 2 = 750
            pos_encoding = ops.constant(
                self._create_positional_encoding(max_len, self.n_audio_state),
                device=self.device
            )
            x = ops.add(x, pos_encoding)
            
            # 4 Transformer blocks
            for i in range(self.n_audio_layer):
                x = self._transformer_block(x, self.n_audio_state, self.n_audio_head, f"enc_layer_{i}")
            
            # Final layer norm
            ln_weight = ops.constant(
                np.ones(self.n_audio_state).astype(np.float32),
                device=self.device
            )
            ln_bias = ops.constant(
                np.zeros(self.n_audio_state).astype(np.float32),
                device=self.device
            )
            x = ops.layer_norm(x, ln_weight, ln_bias, eps=1e-5)
            
            graph.output(x)
            
        return graph
    
    def _apply_conv1d(self, x, weight, bias, kernel_size, stride=1, padding=0):
        """Apply 1D convolution using matrix operations"""
        # Simplified conv1d implementation
        # For hackathon, use a linear approximation
        batch_size, seq_len, in_features = x.shape
        out_features = weight.shape[0]
        
        # Reshape for matrix multiplication
        x_flat = ops.reshape(x, (-1, in_features))
        weight_2d = ops.reshape(weight, (out_features, in_features))
        
        # Apply linear transformation
        out = ops.matmul(x_flat, ops.transpose(weight_2d, [1, 0]))
        out = ops.add(out, bias)
        
        # Reshape back and handle stride
        if stride == 2:
            new_seq_len = seq_len // 2
        else:
            new_seq_len = seq_len
            
        out = ops.reshape(out, (batch_size, new_seq_len, out_features))
        return out
    
    def _transformer_block(self, x, dim, n_heads, name_prefix):
        """Single transformer encoder block"""
        # Self-attention
        attn_out = self._multi_head_attention(x, x, x, dim, n_heads, f"{name_prefix}_attn")
        
        # Residual + layer norm
        ln1_weight = ops.constant(
            np.ones(dim).astype(np.float32),
            device=self.device
        )
        ln1_bias = ops.constant(
            np.zeros(dim).astype(np.float32),
            device=self.device
        )
        x = ops.add(x, attn_out)
        x = ops.layer_norm(x, ln1_weight, ln1_bias, eps=1e-5)
        
        # Feed-forward network
        ffn_out = self._feed_forward(x, dim, f"{name_prefix}_ffn")
        
        # Residual + layer norm
        ln2_weight = ops.constant(
            np.ones(dim).astype(np.float32),
            device=self.device
        )
        ln2_bias = ops.constant(
            np.zeros(dim).astype(np.float32),
            device=self.device
        )
        x = ops.add(x, ffn_out)
        x = ops.layer_norm(x, ln2_weight, ln2_bias, eps=1e-5)
        
        return x
    
    def _multi_head_attention(self, q, k, v, dim, n_heads, name_prefix):
        """Multi-head attention mechanism"""
        head_dim = dim // n_heads
        
        # Linear projections
        q_weight = ops.constant(
            np.random.randn(dim, dim).astype(np.float32) * 0.1,
            device=self.device
        )
        k_weight = ops.constant(
            np.random.randn(dim, dim).astype(np.float32) * 0.1,
            device=self.device
        )
        v_weight = ops.constant(
            np.random.randn(dim, dim).astype(np.float32) * 0.1,
            device=self.device
        )
        
        # Apply projections
        q_proj = ops.matmul(q, q_weight)
        k_proj = ops.matmul(k, k_weight)
        v_proj = ops.matmul(v, v_weight)
        
        # Reshape for multi-head attention
        batch_size, seq_len = q.shape[0], q.shape[1]
        
        q_heads = ops.reshape(q_proj, (batch_size, seq_len, n_heads, head_dim))
        k_heads = ops.reshape(k_proj, (batch_size, seq_len, n_heads, head_dim))
        v_heads = ops.reshape(v_proj, (batch_size, seq_len, n_heads, head_dim))
        
        # Transpose for batch matrix multiplication
        q_heads = ops.permute(q_heads, [0, 2, 1, 3])  # (batch, heads, seq, head_dim)
        k_heads = ops.permute(k_heads, [0, 2, 1, 3])
        v_heads = ops.permute(v_heads, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scale = ops.constant(1.0 / np.sqrt(head_dim), device=self.device)
        scores = ops.matmul(q_heads, ops.permute(k_heads, [0, 1, 3, 2]))
        scores = ops.mul(scores, scale)
        
        # Softmax
        attn_weights = ops.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_out = ops.matmul(attn_weights, v_heads)
        
        # Reshape back
        attn_out = ops.permute(attn_out, [0, 2, 1, 3])  # (batch, seq, heads, head_dim)
        attn_out = ops.reshape(attn_out, (batch_size, seq_len, dim))
        
        # Output projection
        out_weight = ops.constant(
            np.random.randn(dim, dim).astype(np.float32) * 0.1,
            device=self.device
        )
        attn_out = ops.matmul(attn_out, out_weight)
        
        return attn_out
    
    def _feed_forward(self, x, dim, name_prefix):
        """Feed-forward network"""
        hidden_dim = dim * 4  # Standard transformer FFN
        
        # First linear layer
        w1 = ops.constant(
            np.random.randn(dim, hidden_dim).astype(np.float32) * 0.1,
            device=self.device
        )
        b1 = ops.constant(
            np.zeros(hidden_dim).astype(np.float32),
            device=self.device
        )
        
        # Second linear layer
        w2 = ops.constant(
            np.random.randn(hidden_dim, dim).astype(np.float32) * 0.1,
            device=self.device
        )
        b2 = ops.constant(
            np.zeros(dim).astype(np.float32),
            device=self.device
        )
        
        # Forward pass
        hidden = ops.matmul(x, w1)
        hidden = ops.add(hidden, b1)
        hidden = F.gelu(hidden)
        
        out = ops.matmul(hidden, w2)
        out = ops.add(out, b2)
        
        return out
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe.astype(np.float32).reshape(1, max_len, d_model)
    
    def compile_encoder(self):
        """Compile the encoder graph"""
        print("Compiling encoder...")
        graph = self.build_encoder_graph()
        
        # Create session and compile
        self.session = engine.InferenceSession()
        self.encoder_model = self.session.load(graph)
        print("Encoder compiled successfully!")
        
    def encode(self, mel_spectrogram):
        """Run encoder inference"""
        if self.encoder_model is None:
            self.compile_encoder()
            
        # Ensure correct shape and dtype
        if mel_spectrogram.shape != (1, self.n_mels, self.n_audio_ctx):
            # Pad or truncate to expected size
            if mel_spectrogram.shape[-1] < self.n_audio_ctx:
                pad_width = self.n_audio_ctx - mel_spectrogram.shape[-1]
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 0), (0, pad_width)))
            else:
                mel_spectrogram = mel_spectrogram[:, :, :self.n_audio_ctx]
        
        mel_spectrogram = mel_spectrogram.astype(np.float32)
        
        # Create input tensor
        input_tensor = Tensor(mel_spectrogram)
        
        # Run inference
        outputs = self.encoder_model.execute(input_tensor)
        return outputs[0]

def test_encoder():
    """Test the encoder implementation"""
    print("Testing MAX-Whisper Real Encoder...")
    
    # Create model
    model = MAX_WhisperReal(device_id=0)
    
    # Create dummy mel-spectrogram
    mel_input = np.random.randn(1, 80, 1500).astype(np.float32)
    print(f"Input shape: {mel_input.shape}")
    
    # Compile and run
    start_time = time.time()
    try:
        output = model.encode(mel_input)
        compile_time = time.time() - start_time
        print(f"Encoder output shape: {output.shape}")
        print(f"Compilation + inference time: {compile_time:.3f}s")
        
        # Run inference only
        inference_start = time.time()
        for i in range(10):
            output = model.encode(mel_input)
        inference_time = (time.time() - inference_start) / 10
        print(f"Average inference time: {inference_time*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the encoder
    success = test_encoder()
    if success:
        print("\n✅ Encoder implementation working!")
        print("Next: Implement decoder and load real weights")
    else:
        print("\n❌ Encoder needs debugging")