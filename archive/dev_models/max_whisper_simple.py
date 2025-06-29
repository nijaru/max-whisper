"""
Simplified MAX Graph Whisper implementation for GPU testing.
Focus on getting basic encoder working first.
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


class SimpleWhisperEncoder:
    """Simplified Whisper encoder for testing MAX Graph GPU acceleration."""
    
    def __init__(self, device: str = "gpu"):
        self.n_mels = 80
        self.n_ctx = 1500
        self.n_state = 384  # tiny model size
        
        # Set device
        self.device_str = device
        if device == "gpu" and MAX_AVAILABLE:
            try:
                self.device = DeviceRef.GPU()
                print("✅ Using GPU device")
            except:
                print("⚠️ GPU not available, falling back to CPU")
                self.device = DeviceRef.CPU()
                self.device_str = "cpu"
        else:
            self.device = DeviceRef.CPU()
            self.device_str = "cpu"
        
        self.graph = None
        self.model = None
        self._build_graph()
    
    def _build_graph(self):
        """Build simplified encoder graph."""
        if not MAX_AVAILABLE:
            return
        
        print(f"Building MAX Graph encoder on {self.device_str.upper()}...")
        
        # Fixed input shape
        input_type = TensorType(
            dtype=DType.float32,
            shape=(1, self.n_mels, self.n_ctx),
            device=self.device
        )
        
        with Graph("simple_whisper_encoder", input_types=(input_type,)) as graph:
            x = graph.inputs[0]
            
            # Simple linear transformation to simulate encoding
            # Create weight matrix
            weight = ops.constant(
                np.random.randn(self.n_mels, self.n_state).astype(np.float32) * 0.02,
                dtype=DType.float32,
                device=self.device
            )
            
            # Permute input to (batch, time, mels)
            x = ops.permute(x, [0, 2, 1])
            
            # Flatten batch and time
            batch_time = 1 * self.n_ctx
            x_flat = ops.reshape(x, (batch_time, self.n_mels))
            
            # Apply transformation
            x_encoded = ops.matmul(x_flat, weight)
            
            # Reshape back to (batch, n_state, time)
            x_out = ops.reshape(x_encoded, (1, self.n_ctx, self.n_state))
            x_out = ops.permute(x_out, [0, 2, 1])
            
            # Apply activation
            x_out = elementwise.relu(x_out)
            
            graph.output(x_out)
        
        self.graph = graph
        
        # Create inference session with appropriate device
        if self.device_str == "gpu":
            # Configure session for GPU
            gpu_device = Accelerator(id=0)
            session = engine.InferenceSession(devices=[gpu_device])
        else:
            cpu_device = CPU()
            session = engine.InferenceSession(devices=[cpu_device])
        
        self.model = session.load(self.graph)
        
        print(f"✅ Graph built successfully on {self.device_str.upper()}")
    
    def encode(self, mel_features: np.ndarray) -> np.ndarray:
        """Encode mel-spectrogram features."""
        if not MAX_AVAILABLE or self.model is None:
            # Dummy output for testing
            return np.random.randn(1, self.n_state, self.n_ctx).astype(np.float32)
        
        # Ensure correct shape
        if mel_features.ndim == 2:
            mel_features = mel_features[np.newaxis, :, :]
        
        # Create tensor on appropriate device
        if self.device_str == "gpu":
            # Create tensor on GPU
            device = Accelerator(id=0)
        else:
            device = CPU()
        
        input_tensor = Tensor.from_numpy(mel_features)
        if self.device_str == "gpu":
            # Transfer to GPU
            input_tensor = input_tensor.to(device)
        
        # Run inference
        outputs = self.model.execute(input_tensor)
        return outputs[0].to_numpy()


def benchmark_simple_encoder():
    """Benchmark the simplified encoder."""
    print("=== MAX-Whisper Simple Encoder Benchmark ===\n")
    
    # Test both CPU and GPU
    for device in ["cpu", "gpu"]:
        print(f"\n--- Testing on {device.upper()} ---")
        
        # Create model
        encoder = SimpleWhisperEncoder(device=device)
        
        # Test input
        mel_features = np.random.randn(1, 80, 1500).astype(np.float32)
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            _ = encoder.encode(mel_features)
        
        # Benchmark
        print("Benchmarking...")
        times = []
        num_runs = 10
        
        for i in range(num_runs):
            start = time.time()
            output = encoder.encode(mel_features)
            end = time.time()
            times.append(end - start)
            
            if i == 0:
                print(f"Output shape: {output.shape}")
        
        # Calculate metrics
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        
        audio_duration = 1500 / 50.0  # 30 seconds
        rtf = (avg_time / 1000) / audio_duration
        
        print(f"\nResults:")
        print(f"Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"Min time: {min_time:.2f} ms")
        print(f"Audio duration: {audio_duration:.1f} s")
        print(f"Real-time factor: {rtf:.6f}")
        print(f"Speedup: {1/rtf:.1f}x real-time")
        
        # Check against target
        target_rtf = 0.001
        if rtf < target_rtf:
            print(f"✅ Exceeds target RTF of {target_rtf}!")
        else:
            print(f"⚠️ Need {rtf/target_rtf:.1f}x speedup for target")


if __name__ == "__main__":
    if MAX_AVAILABLE:
        benchmark_simple_encoder()
    else:
        print("MAX Graph not available - please install MAX")