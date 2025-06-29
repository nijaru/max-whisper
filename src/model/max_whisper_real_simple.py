"""
Simplified Real MAX-Whisper Implementation
Building towards complete model step by step.
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

class SimpleRealEncoder:
    """Step 1: Build a working real encoder with transformer blocks"""
    
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
        self.n_layers = 2  # Start with 2 layers
        
        self.graph = None
        self.model = None
        self.session = None
        
    def _build_simple_attention(self, x, name_prefix):
        """Simple multi-head attention"""
        # For now, just a linear transformation
        # Will expand to real attention later
        weight = ops.constant(
            np.random.randn(self.n_state, self.n_state).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        
        # Reshape for matmul: (batch, seq, dim) -> (batch*seq, dim)
        batch_size, seq_len, dim = 1, 1500, self.n_state
        x_flat = ops.reshape(x, (batch_size * seq_len, dim))
        
        # Apply linear transformation
        out = ops.matmul(x_flat, weight)
        
        # Reshape back
        out = ops.reshape(out, (batch_size, seq_len, dim))
        return out
    
    def _build_feed_forward(self, x):
        """Simple feed forward network"""
        # Linear + GELU + Linear
        w1 = ops.constant(
            np.random.randn(self.n_state, self.n_state * 2).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        w2 = ops.constant(
            np.random.randn(self.n_state * 2, self.n_state).astype(np.float32) * 0.1,
            dtype=DType.float32,
            device=self.device
        )
        
        # Apply transformations
        batch_size, seq_len, dim = 1, 1500, self.n_state
        x_flat = ops.reshape(x, (batch_size * seq_len, dim))
        
        # First linear
        hidden = ops.matmul(x_flat, w1)
        hidden = elementwise.gelu(hidden)
        
        # Second linear
        out = ops.matmul(hidden, w2)
        out = ops.reshape(out, (batch_size, seq_len, dim))
        
        return out
    
    def _transformer_block(self, x, layer_idx):
        """Single transformer block with attention + FFN"""
        # Self-attention
        attn_out = self._build_simple_attention(x, f"layer_{layer_idx}")
        
        # Residual connection
        x = ops.add(x, attn_out)
        
        # Layer norm (simplified)
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
        x = ops.layer_norm(x, ln_weight, ln_bias, 1e-5)
        
        # Feed-forward
        ffn_out = self._build_feed_forward(x)
        
        # Residual connection
        x = ops.add(x, ffn_out)
        
        # Final layer norm
        x = ops.layer_norm(x, ln_weight, ln_bias, 1e-5)
        
        return x
    
    def build_graph(self):
        """Build the encoder graph"""
        print("Building simplified real encoder graph...")
        
        input_type = TensorType(
            dtype=DType.float32,
            shape=(1, self.n_mels, self.n_ctx),
            device=self.device
        )
        
        with Graph("real_encoder", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            
            # Simple conv layers to reduce sequence length
            # Conv1: 80 -> 384 channels
            conv1_weight = ops.constant(
                np.random.randn(self.n_state, self.n_mels).astype(np.float32) * 0.1,
                dtype=DType.float32,
                device=self.device
            )
            
            # Permute to (batch, time, mels) for processing
            x = ops.permute(x, [0, 2, 1])  # (1, 1500, 80)
            
            # Apply linear transformation as simplified conv
            batch_time = self.n_ctx
            x_flat = ops.reshape(x, (batch_time, self.n_mels))
            x = ops.matmul(x_flat, ops.transpose(conv1_weight, 0, 1))
            x = elementwise.gelu(x)
            
            # Reshape for transformer blocks (keep full sequence for now)
            x = ops.reshape(x, (1, batch_time, self.n_state))
            
            # Apply transformer blocks
            for i in range(self.n_layers):
                x = self._transformer_block(x, i)
            
            # Final output: (1, 1500, 384)
            graph.output(x)
        
        return graph
    
    def compile(self):
        """Compile the model"""
        if not MAX_AVAILABLE:
            return False
            
        try:
            self.graph = self.build_graph()
            
            # Create session with GPU
            if self.device == DeviceRef.GPU():
                gpu_device = Accelerator(id=0)
                self.session = engine.InferenceSession(devices=[gpu_device])
            else:
                self.session = engine.InferenceSession()
            
            self.model = self.session.load(self.graph)
            print("✅ Model compiled successfully!")
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
                # Pad
                pad_width = self.n_ctx - mel_spectrogram.shape[-1]
                mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 0), (0, pad_width)))
            else:
                # Truncate
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

def test_simple_real_encoder():
    """Test the simplified real encoder"""
    print("Testing Simplified Real Encoder...")
    
    # Create model
    model = SimpleRealEncoder()
    if not MAX_AVAILABLE:
        print("MAX Graph not available - skipping test")
        return False
    
    # Create test input
    mel_input = np.random.randn(1, 80, 1500).astype(np.float32)
    print(f"Input shape: {mel_input.shape}")
    
    # Test compilation and inference
    start_time = time.time()
    try:
        output = model.encode(mel_input)
        if output is not None:
            total_time = time.time() - start_time
            print(f"✅ Output shape: {output.shape}")
            print(f"Total time: {total_time:.3f}s")
            
            # Run multiple inferences for timing
            inference_times = []
            for i in range(10):
                start = time.time()
                _ = model.encode(mel_input)
                inference_times.append(time.time() - start)
            
            avg_time = np.mean(inference_times) * 1000  # Convert to ms
            print(f"Average inference time: {avg_time:.2f}ms")
            print(f"Expected output shape for transformer: (1, 1500, 384)")
            
            return True
        else:
            print("❌ Encoding failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_real_encoder()
    if success:
        print("\n🎉 Step 1 complete: Simplified real encoder working!")
        print("Next: Add proper multi-head attention and more layers")
    else:
        print("\n💥 Need to debug step 1 first")