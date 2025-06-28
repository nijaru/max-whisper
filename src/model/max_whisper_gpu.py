"""
MAX Graph implementation of Whisper model optimized for GPU.
Full encoder-decoder transformer architecture for speech recognition.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time

try:
    from max import engine
    from max.driver import Tensor
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops, TensorValue
    from max.graph.ops import elementwise
    MAX_AVAILABLE = True
except ImportError:
    print("MAX Graph not available - please ensure MAX is installed")
    MAX_AVAILABLE = False


class WhisperConfig:
    """Configuration for Whisper model sizes."""
    
    CONFIGS = {
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
    
    def __init__(self, model_size: str = "tiny"):
        config = self.CONFIGS[model_size]
        self.model_size = model_size
        self.n_mels = config["n_mels"]
        self.n_audio_ctx = config["n_audio_ctx"]
        self.n_audio_state = config["n_audio_state"]
        self.n_audio_head = config["n_audio_head"]
        self.n_audio_layer = config["n_audio_layer"]
        self.n_text_ctx = config["n_text_ctx"]
        self.n_text_state = config["n_text_state"]
        self.n_text_head = config["n_text_head"]
        self.n_text_layer = config["n_text_layer"]
        self.n_vocab = config["n_vocab"]


class MAXWhisperModel:
    """Full Whisper model implementation using MAX Graph for GPU acceleration."""
    
    def __init__(self, config: WhisperConfig, device: str = "gpu"):
        self.config = config
        self.device = DeviceRef.GPU() if device == "gpu" else DeviceRef.CPU()
        
        # Model components
        self.encoder_graph = None
        self.decoder_graph = None
        self.encoder_session = None
        self.decoder_session = None
        
        # Build the model
        self._build_encoder()
        self._build_decoder()
    
    def _get_device(self):
        """Get the appropriate device reference."""
        if not MAX_AVAILABLE:
            return None
        
        # Check if GPU is available
        try:
            if hasattr(DeviceRef, 'GPU'):
                return DeviceRef.GPU()
        except:
            print("GPU not available, falling back to CPU")
        
        return DeviceRef.CPU()
    
    def _build_encoder(self):
        """Build the Whisper encoder graph."""
        if not MAX_AVAILABLE:
            print("MAX not available - encoder not built")
            return
        
        print(f"Building MAX Graph Whisper encoder ({self.config.model_size})...")
        
        # Define input shape for mel-spectrogram
        input_type = TensorType(
            dtype=DType.float32,
            shape=(1, self.config.n_mels, self.config.n_audio_ctx),
            device=self.device
        )
        
        with Graph("whisper_encoder", input_types=(input_type,)) as graph:
            mel_input = graph.inputs[0]
            
            # Initial convolution layers (similar to Whisper's conv1 and conv2)
            x = self._conv1d_block(mel_input, out_channels=self.config.n_audio_state, kernel_size=3, name="conv1")
            x = elementwise.gelu(x)
            x = self._conv1d_block(x, out_channels=self.config.n_audio_state, kernel_size=3, stride=2, name="conv2")
            x = elementwise.gelu(x)
            
            # Positional encoding
            x = self._add_positional_encoding(x, self.config.n_audio_ctx // 2)
            
            # Transformer encoder blocks
            for i in range(self.config.n_audio_layer):
                x = self._transformer_encoder_block(x, f"encoder_block_{i}")
            
            # Final layer norm
            x = self._layer_norm(x, f"encoder_ln_final")
            
            graph.output(x)
        
        self.encoder_graph = graph
        
        # Create inference session
        self.encoder_session = engine.InferenceSession()
        self.encoder_model = self.encoder_session.load(self.encoder_graph)
        
        print("✅ Encoder graph built successfully")
    
    def _build_decoder(self):
        """Build the Whisper decoder graph."""
        if not MAX_AVAILABLE:
            print("MAX not available - decoder not built")
            return
        
        print(f"Building MAX Graph Whisper decoder ({self.config.model_size})...")
        
        # Define input types
        token_input_type = TensorType(
            dtype=DType.int32,
            shape=(1, self.config.n_text_ctx),
            device=self.device
        )
        
        encoder_output_type = TensorType(
            dtype=DType.float32,
            shape=(1, self.config.n_audio_state, self.config.n_audio_ctx // 2),
            device=self.device
        )
        
        with Graph("whisper_decoder", input_types=(token_input_type, encoder_output_type)) as graph:
            tokens = graph.inputs[0]
            encoder_output = graph.inputs[1]
            
            # Token embedding
            x = self._token_embedding(tokens)
            
            # Positional encoding
            x = self._add_positional_encoding(x, self.config.n_text_ctx)
            
            # Transformer decoder blocks with cross-attention
            for i in range(self.config.n_text_layer):
                x = self._transformer_decoder_block(x, encoder_output, f"decoder_block_{i}")
            
            # Final layer norm
            x = self._layer_norm(x, "decoder_ln_final")
            
            # Project to vocabulary
            logits = self._linear(x, self.config.n_vocab, "lm_head")
            
            graph.output(logits)
        
        self.decoder_graph = graph
        
        # Create inference session
        self.decoder_session = engine.InferenceSession()
        self.decoder_model = self.decoder_session.load(self.decoder_graph)
        
        print("✅ Decoder graph built successfully")
    
    def _conv1d_block(self, x: TensorValue, out_channels: int, kernel_size: int, stride: int = 1, name: str = "conv") -> TensorValue:
        """1D convolution block."""
        # For hackathon demo, using simplified convolution
        # In production, would use proper conv1d with loaded weights
        
        # Get input channels from config
        in_channels = self.config.n_mels if name == "conv1" else self.config.n_audio_state
        
        # Create weight matrix (would be loaded from checkpoint in production)
        weight_shape = (out_channels, in_channels)
        weight = ops.constant(
            np.random.randn(out_channels, in_channels).astype(np.float32) * 0.02,
            dtype=DType.float32,
            device=self.device
        )
        
        # Apply "convolution" via matmul
        # x has shape (batch, channels, time)
        # Transpose to (batch, time, channels) for matmul
        x_transposed = ops.transpose(x, (0, 2, 1))
        
        # Flatten batch and time dimensions
        batch_size = 1  # Fixed for now
        time_dim = self.config.n_audio_ctx if name == "conv1" else self.config.n_audio_ctx // 2
        x_reshaped = ops.reshape(x_transposed, (batch_size * time_dim, in_channels))
        
        # Apply linear transformation
        x_conv = ops.matmul(x_reshaped, ops.transpose(weight, (1, 0)))
        
        # Reshape back
        x_out = ops.reshape(x_conv, (batch_size, time_dim, out_channels))
        x_out = ops.transpose(x_out, (0, 2, 1))  # Back to (batch, channels, time)
        
        # Handle stride by slicing (simplified for demo)
        if stride > 1:
            # Take every 'stride' element in the time dimension
            # This is a simplified stride - proper conv would aggregate
            # Create indices for strided selection
            start = ops.constant(0, dtype=DType.int32, device=self.device)
            stop = ops.constant(time_dim, dtype=DType.int32, device=self.device)
            step = ops.constant(stride, dtype=DType.int32, device=self.device)
            indices = ops.range(start, stop, step, device=self.device, dtype=DType.int32)
            x_out = ops.gather(x_out, indices, axis=2)
        
        return x_out
    
    def _add_positional_encoding(self, x: TensorValue, max_len: int) -> TensorValue:
        """Add sinusoidal positional encoding."""
        batch, n_state, seq_len = 1, x.shape[1], x.shape[2]
        
        # Create position encoding (simplified for demo)
        pos_encoding = self._create_sinusoidal_encoding(seq_len, n_state)
        pos_tensor = ops.constant(pos_encoding, dtype=DType.float32, device=self.device)
        
        # Add to input
        return x + pos_tensor
    
    def _create_sinusoidal_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((1, d_model, seq_len))
        pos_encoding[0, 0::2, :] = np.sin(position * div_term).T
        pos_encoding[0, 1::2, :] = np.cos(position * div_term).T
        
        return pos_encoding.astype(np.float32)
    
    def _transformer_encoder_block(self, x: TensorValue, name: str) -> TensorValue:
        """Transformer encoder block with self-attention."""
        # Self-attention
        attn_output = self._multi_head_attention(x, x, x, self.config.n_audio_head, f"{name}_attn")
        x = x + attn_output
        x = self._layer_norm(x, f"{name}_ln1")
        
        # Feed-forward network
        ff_output = self._feed_forward(x, f"{name}_ff")
        x = x + ff_output
        x = self._layer_norm(x, f"{name}_ln2")
        
        return x
    
    def _transformer_decoder_block(self, x: TensorValue, encoder_output: TensorValue, name: str) -> TensorValue:
        """Transformer decoder block with self-attention and cross-attention."""
        # Self-attention (masked)
        self_attn_output = self._multi_head_attention(x, x, x, self.config.n_text_head, f"{name}_self_attn", masked=True)
        x = x + self_attn_output
        x = self._layer_norm(x, f"{name}_ln1")
        
        # Cross-attention with encoder output
        cross_attn_output = self._multi_head_attention(x, encoder_output, encoder_output, self.config.n_text_head, f"{name}_cross_attn")
        x = x + cross_attn_output
        x = self._layer_norm(x, f"{name}_ln2")
        
        # Feed-forward network
        ff_output = self._feed_forward(x, f"{name}_ff")
        x = x + ff_output
        x = self._layer_norm(x, f"{name}_ln3")
        
        return x
    
    def _multi_head_attention(self, query: TensorValue, key: TensorValue, value: TensorValue, 
                             n_heads: int, name: str, masked: bool = False) -> TensorValue:
        """Multi-head attention mechanism."""
        batch = 1
        n_state = query.shape[1]
        head_dim = n_state // n_heads
        
        # Linear projections for Q, K, V
        q = self._linear(query, n_state, f"{name}_q")
        k = self._linear(key, n_state, f"{name}_k")
        v = self._linear(value, n_state, f"{name}_v")
        
        # Reshape for multi-head attention
        q = ops.reshape(q, (batch, n_heads, head_dim, -1))
        k = ops.reshape(k, (batch, n_heads, head_dim, -1))
        v = ops.reshape(v, (batch, n_heads, head_dim, -1))
        
        # Scaled dot-product attention
        scale = ops.constant(1.0 / np.sqrt(head_dim), dtype=DType.float32, device=self.device)
        scores = ops.matmul(
            ops.transpose(q, (0, 1, 3, 2)),
            k
        ) * scale
        
        # Apply mask if needed (for decoder self-attention)
        if masked:
            # Create causal mask
            seq_len = scores.shape[-1]
            mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1).astype(np.float32)
            mask_tensor = ops.constant(mask, dtype=DType.float32, device=self.device)
            scores = scores + mask_tensor
        
        # Softmax
        attn_weights = elementwise.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = ops.matmul(attn_weights, ops.transpose(v, (0, 1, 3, 2)))
        
        # Reshape back
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch, n_state, -1))
        
        # Output projection
        output = self._linear(attn_output, n_state, f"{name}_out")
        
        return output
    
    def _feed_forward(self, x: TensorValue, name: str) -> TensorValue:
        """Feed-forward network."""
        n_state = x.shape[1]
        n_mlp = n_state * 4  # Standard transformer FFN expansion
        
        # Two linear layers with GELU activation
        x = self._linear(x, n_mlp, f"{name}_fc1")
        x = elementwise.gelu(x)
        x = self._linear(x, n_state, f"{name}_fc2")
        
        return x
    
    def _linear(self, x: TensorValue, out_features: int, name: str) -> TensorValue:
        """Linear transformation."""
        in_features = x.shape[1]
        
        # Weight matrix (would be loaded from checkpoint in production)
        weight = ops.constant(
            np.random.randn(in_features, out_features).astype(np.float32) * 0.02,
            dtype=DType.float32,
            device=self.device
        )
        
        # Bias
        bias = ops.constant(
            np.zeros(out_features).astype(np.float32),
            dtype=DType.float32,
            device=self.device
        )
        
        # Apply linear transformation
        batch, n_state, seq_len = x.shape[0], x.shape[1], x.shape[2]
        x_reshaped = ops.reshape(x, (batch * seq_len, n_state))
        output = ops.matmul(x_reshaped, weight) + bias
        output = ops.reshape(output, (batch, out_features, seq_len))
        
        return output
    
    def _layer_norm(self, x: TensorValue, name: str) -> TensorValue:
        """Layer normalization."""
        n_state = x.shape[1]
        
        # Layer norm parameters (would be loaded from checkpoint)
        gamma = ops.constant(
            np.ones(n_state).astype(np.float32),
            dtype=DType.float32,
            device=self.device
        )
        beta = ops.constant(
            np.zeros(n_state).astype(np.float32),
            dtype=DType.float32,
            device=self.device
        )
        
        # Apply layer norm
        return ops.layer_norm(x, gamma, beta, epsilon=1e-5)
    
    def _token_embedding(self, tokens: TensorValue) -> TensorValue:
        """Token embedding layer."""
        # Create embedding matrix (would be loaded from checkpoint)
        embedding_matrix = ops.constant(
            np.random.randn(self.config.n_vocab, self.config.n_text_state).astype(np.float32) * 0.02,
            dtype=DType.float32,
            device=self.device
        )
        
        # Gather embeddings
        embeddings = ops.gather(embedding_matrix, tokens, axis=0)
        
        # Transpose to match expected shape
        return ops.transpose(embeddings, (0, 2, 1))
    
    def encode(self, mel_features: np.ndarray) -> np.ndarray:
        """Encode mel-spectrogram features."""
        if not MAX_AVAILABLE or self.encoder_model is None:
            raise RuntimeError("Encoder not available")
        
        # Ensure input is correct shape
        if mel_features.ndim == 2:
            mel_features = mel_features[np.newaxis, :, :]
        
        # Run encoder
        outputs = self.encoder_model.execute(mel_features)
        return outputs[0].to_numpy()
    
    def decode(self, tokens: np.ndarray, encoder_output: np.ndarray) -> np.ndarray:
        """Decode tokens with encoder output."""
        if not MAX_AVAILABLE or self.decoder_model is None:
            raise RuntimeError("Decoder not available")
        
        # Run decoder
        outputs = self.decoder_model.execute(tokens, encoder_output)
        return outputs[0].to_numpy()
    
    def transcribe(self, mel_features: np.ndarray) -> str:
        """Full transcription pipeline."""
        # Encode audio
        encoder_output = self.encode(mel_features)
        
        # Initialize decoder with start token
        start_token = 50258  # <|startoftranscript|>
        end_token = 50257    # <|endoftext|>
        
        tokens = [start_token]
        max_length = self.config.n_text_ctx
        
        # Autoregressive decoding
        for _ in range(max_length):
            # Prepare token input
            token_input = np.array([tokens + [0] * (max_length - len(tokens))], dtype=np.int32)
            
            # Decode
            logits = self.decode(token_input, encoder_output)
            
            # Get next token
            next_token = np.argmax(logits[0, :, len(tokens) - 1])
            tokens.append(int(next_token))
            
            # Stop if end token
            if next_token == end_token:
                break
        
        # Convert tokens to text (simplified - would use proper tokenizer)
        text = f"[Transcription of {len(tokens)} tokens]"
        return text


def benchmark_gpu_whisper():
    """Benchmark the GPU-optimized Whisper implementation."""
    print("=== MAX-Whisper GPU Benchmark ===")
    
    # Create model
    config = WhisperConfig("tiny")
    model = MAXWhisperModel(config, device="gpu")
    
    # Test input
    batch_size = 1
    n_mels = 80
    n_ctx = 1500
    
    mel_features = np.random.randn(batch_size, n_mels, n_ctx).astype(np.float32)
    
    print(f"Model: Whisper-{config.model_size}")
    print(f"Device: GPU (RTX 4090)")
    print(f"Input shape: {mel_features.shape}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = model.encode(mel_features)
    
    # Benchmark encoding
    print("\nBenchmarking encoder...")
    num_runs = 10
    encode_times = []
    
    for _ in range(num_runs):
        start = time.time()
        encoder_output = model.encode(mel_features)
        end = time.time()
        encode_times.append(end - start)
    
    avg_encode_time = np.mean(encode_times) * 1000
    print(f"Average encoding time: {avg_encode_time:.2f} ms")
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Calculate RTF
    audio_duration = n_ctx / 50.0  # 50 FPS mel-spectrogram
    rtf = (avg_encode_time / 1000) / audio_duration
    
    print(f"\nPerformance Metrics:")
    print(f"Audio duration: {audio_duration:.1f} s")
    print(f"Real-time factor: {rtf:.6f}")
    print(f"Speedup vs real-time: {1/rtf:.1f}x")
    
    # Compare to targets
    target_rtf = 0.001
    if rtf < target_rtf:
        print(f"✅ Exceeds target RTF of {target_rtf}!")
    else:
        print(f"⚠️ Need {rtf/target_rtf:.1f}x more speedup for target")
    
    return {
        "model": config.model_size,
        "device": "GPU",
        "encode_time_ms": avg_encode_time,
        "rtf": rtf,
        "speedup": 1/rtf
    }


if __name__ == "__main__":
    if MAX_AVAILABLE:
        results = benchmark_gpu_whisper()
        
        print("\n=== Results Summary ===")
        print(f"Model: Whisper-{results['model']} on {results['device']}")
        print(f"Encoding time: {results['encode_time_ms']:.2f} ms")
        print(f"RTF: {results['rtf']:.6f}")
        print(f"Speedup: {results['speedup']:.1f}x real-time")
        
        print("\n=== Next Steps ===")
        print("1. Load pre-trained Whisper weights")
        print("2. Implement proper tokenizer")
        print("3. Optimize memory transfers")
        print("4. Add batch processing")
        print("5. Implement Mojo GPU kernels for preprocessing")
    else:
        print("MAX Graph not available - please install MAX")