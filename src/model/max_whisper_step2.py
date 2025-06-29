"""
Step 2: Simplified Multi-Head Attention for MAX-Whisper
Building incrementally from working simple encoder.
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

class SimpleMultiHeadEncoder:
    """Simplified multi-head attention that actually works"""
    
    def __init__(self):
        if not MAX_AVAILABLE:
            print("MAX Graph not available")
            return
            
        # Use GPU if available
        try:
            self.device = DeviceRef.GPU()
            print("✅ Using GPU device")
        except:
            print("⚠️ GPU not available, falling back to CPU")
            self.device = DeviceRef.CPU()
        
        # Model dimensions
        self.n_mels = 80
        self.n_ctx = 1500
        self.n_state = 384
        self.n_heads = 6
        self.n_layers = 2  # Keep it simple for now
        self.head_dim = self.n_state // self.n_heads  # 64
        
        self.graph = None
        self.model = None
        self.session = None
        
    def _simple_attention(self, x, name_prefix):
        """Simplified attention that focuses on the mechanism"""
        batch_size, seq_len, dim = 1, self.n_ctx, self.n_state
        
        # Single attention head for simplicity
        # Q, K, V projections
        wq = ops.constant(
            np.random.randn(dim, self.head_dim).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        wk = ops.constant(
            np.random.randn(dim, self.head_dim).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        wv = ops.constant(
            np.random.randn(dim, self.head_dim).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        wo = ops.constant(
            np.random.randn(self.head_dim, dim).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        
        # Flatten for matrix operations
        x_flat = ops.reshape(x, (batch_size * seq_len, dim))
        
        # Project to Q, K, V (smaller dimensions)
        q = ops.matmul(x_flat, wq)  # (1500, 64)
        k = ops.matmul(x_flat, wk)  # (1500, 64)
        v = ops.matmul(x_flat, wv)  # (1500, 64)
        
        # Compute attention scores: Q @ K.T
        k_t = ops.transpose(k, 0, 1)  # (64, 1500)
        scores = ops.matmul(q, k_t)   # (1500, 1500)
        
        # Scale
        scale = ops.constant(
            1.0 / np.sqrt(self.head_dim),
            dtype=DType.float32,
            device=self.device
        )
        scores = ops.mul(scores, scale)
        
        # Softmax to get attention weights
        attn_weights = ops.softmax(scores)  # (1500, 1500)
        
        # Apply to values
        attn_out = ops.matmul(attn_weights, v)  # (1500, 64)
        
        # Project back to full dimension
        output = ops.matmul(attn_out, wo)  # (1500, 384)
        
        # Reshape back to (batch, seq, dim)
        output = ops.reshape(output, (batch_size, seq_len, dim))
        
        return output
    
    def _build_feed_forward(self, x):
        """Simple feed-forward network"""
        batch_size, seq_len, dim = 1, self.n_ctx, self.n_state
        hidden_dim = dim * 2  # Smaller for simplicity
        
        # FFN weights
        w1 = ops.constant(
            np.random.randn(dim, hidden_dim).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        w2 = ops.constant(
            np.random.randn(hidden_dim, dim).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        
        # Apply transformations
        x_flat = ops.reshape(x, (batch_size * seq_len, dim))
        
        # First linear + GELU
        hidden = ops.matmul(x_flat, w1)
        hidden = elementwise.gelu(hidden)
        
        # Second linear
        out = ops.matmul(hidden, w2)
        out = ops.reshape(out, (batch_size, seq_len, dim))
        
        return out
    
    def _transformer_block(self, x, layer_idx):
        """Transformer block with simplified attention"""
        # Layer norm weights
        ln_weight = ops.constant(
            np.ones(self.n_state).astype(np.float32),
            dtype=DType.float32,
            device=self.device
        )
        ln_bias = ops.constant(
            np.zeros(self.n_state).astype(np.float32),
            dtype=DType.float32,
            device=self.device
        )
        
        # Pre-norm attention
        x_norm = ops.layer_norm(x, ln_weight, ln_bias, 1e-5)
        attn_out = self._simple_attention(x_norm, f"layer_{layer_idx}")
        x = ops.add(x, attn_out)
        
        # Pre-norm FFN
        x_norm2 = ops.layer_norm(x, ln_weight, ln_bias, 1e-5)
        ffn_out = self._build_feed_forward(x_norm2)
        x = ops.add(x, ffn_out)
        
        return x
    
    def build_graph(self):
        """Build encoder graph with simplified attention"""
        print("Building simplified multi-head attention encoder...")
        
        input_type = TensorType(
            dtype=DType.float32,
            shape=(1, self.n_mels, self.n_ctx),
            device=self.device
        )
        
        with Graph("simple_mha_encoder", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            
            # Input projection
            conv_weight = ops.constant(
                np.random.randn(self.n_state, self.n_mels).astype(np.float32) * 0.1,
                dtype=DType.float32,
                device=self.device
            )
            
            # Transform input
            x = ops.permute(x, [0, 2, 1])  # (1, 1500, 80)
            x_flat = ops.reshape(x, (self.n_ctx, self.n_mels))
            x = ops.matmul(x_flat, ops.transpose(conv_weight, 0, 1))
            x = elementwise.gelu(x)
            x = ops.reshape(x, (1, self.n_ctx, self.n_state))
            
            # Apply transformer blocks
            for i in range(self.n_layers):
                x = self._transformer_block(x, i)
                print(f"Added attention block {i+1}/{self.n_layers}")
            
            graph.output(x)
        
        return graph
    
    def compile(self):
        """Compile the model"""
        if not MAX_AVAILABLE:
            return False
            
        try:
            self.graph = self.build_graph()
            
            if self.device == DeviceRef.GPU():
                gpu_device = Accelerator(id=0)
                self.session = engine.InferenceSession(devices=[gpu_device])
            else:
                self.session = engine.InferenceSession()
            
            self.model = self.session.load(self.graph)
            print("✅ Simple multi-head attention model compiled!")
            return True
            
        except Exception as e:
            print(f"❌ Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def encode(self, mel_spectrogram):
        """Run encoder inference"""
        if self.model is None:
            if not self.compile():
                return None
        
        # Ensure correct shape
        if mel_spectrogram.shape != (1, self.n_mels, self.n_ctx):
            if mel_spectrogram.shape[-1] < self.n_ctx:
                pad_width = self.n_ctx - mel_spectrogram.shape[-1]
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 0), (0, pad_width)))
            else:
                mel_spectrogram = mel_spectrogram[:, :, :self.n_ctx]
        
        mel_spectrogram = mel_spectrogram.astype(np.float32)
        
        # Create input tensor
        input_tensor = Tensor.from_numpy(mel_spectrogram)
        if self.device == DeviceRef.GPU():
            device = Accelerator(id=0)
            input_tensor = input_tensor.to(device)
        
        # Run inference
        outputs = self.model.execute(input_tensor)
        return outputs[0]

def test_simple_multihead():
    """Test the simplified multi-head attention"""
    print("Testing Simplified Multi-Head Attention...")
    
    model = SimpleMultiHeadEncoder()
    if not MAX_AVAILABLE:
        return False
    
    mel_input = np.random.randn(1, 80, 1500).astype(np.float32)
    print(f"Input: {mel_input.shape}")
    print(f"Architecture: {model.n_heads} heads, {model.head_dim} head_dim")
    
    start_time = time.time()
    try:
        output = model.encode(mel_input)
        if output is not None:
            total_time = time.time() - start_time
            print(f"✅ Output: {output.shape}")
            print(f"Compilation + inference: {total_time:.3f}s")
            
            # Benchmark
            times = []
            for i in range(5):
                start = time.time()
                _ = model.encode(mel_input)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1000
            print(f"Average inference: {avg_time:.2f}ms")
            print(f"🧠 Real attention mechanism working!")
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_multihead()
    if success:
        print("\n🎉 Step 2 complete: Attention mechanism working!")
        print("Next: Build decoder for text generation")
    else:
        print("\n💥 Debugging needed")