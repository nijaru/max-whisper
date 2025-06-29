#!/usr/bin/env python3
"""Test MAX Graph GPU capability directly"""

import time
import numpy as np

try:
    from max import engine
    from max.driver import Tensor
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops
    
    print("🔍 Testing MAX Graph GPU capability...")
    
    # Test GPU device creation
    print("Testing GPU device creation...")
    gpu_device = DeviceRef.GPU() 
    print("✅ GPU device created successfully")
    
    # Test simple GPU operation
    print("Testing simple GPU computation...")
    
    # Create a simple graph for matrix multiplication
    graph = Graph(gpu_device)
    
    # Input tensors
    input1_type = TensorType(DType.float32, (100, 100))
    input2_type = TensorType(DType.float32, (100, 100))
    
    # Graph operations
    with graph:
        input1 = ops.input(input1_type)
        input2 = ops.input(input2_type)
        result = ops.matmul(input1, input2)
        output = ops.output(result)
    
    # Compile for GPU
    session = engine.InferenceSession()
    gpu_model = session.load(graph)
    
    # Test data
    a = np.random.randn(100, 100).astype(np.float32)
    b = np.random.randn(100, 100).astype(np.float32)
    
    a_tensor = Tensor.from_numpy(a)
    b_tensor = Tensor.from_numpy(b)
    
    # Benchmark GPU execution
    print("Running GPU benchmark...")
    start_time = time.time()
    
    for _ in range(100):  # Run 100 iterations for stable timing
        results = gpu_model.execute(a_tensor, b_tensor)
    
    end_time = time.time()
    gpu_time = (end_time - start_time) / 100  # Average per operation
    
    print(f"✅ GPU execution successful!")
    print(f"   Average GPU time: {gpu_time*1000:.3f}ms per operation")
    print(f"   GPU seems to be working properly")
    
    # Compare with CPU numpy
    print("Comparing with CPU numpy...")
    start_time = time.time()
    
    for _ in range(100):
        cpu_result = np.matmul(a, b)
    
    end_time = time.time()
    cpu_time = (end_time - start_time) / 100
    
    speedup = cpu_time / gpu_time
    print(f"   CPU time: {cpu_time*1000:.3f}ms per operation")
    print(f"   GPU speedup: {speedup:.1f}x faster than CPU")
    
    if speedup > 1:
        print("🚀 MAX Graph GPU acceleration is working!")
    else:
        print("⚠️ GPU not significantly faster - may need optimization")
        
except Exception as e:
    print(f"❌ MAX Graph GPU test failed: {e}")
    import traceback
    traceback.print_exc()