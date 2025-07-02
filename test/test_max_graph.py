#!/usr/bin/env python3
"""Test MAX Graph availability and basic functionality"""

try:
    from max import engine
    from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops
    print("✅ MAX Graph imports successful")
    print(f"Accelerator count: {accelerator_count()}")
    
    # Test device setup first
    if accelerator_count() > 0:
        driver_device = Accelerator()
        device = DeviceRef.GPU()
    else:
        driver_device = CPU()
        device = DeviceRef.CPU()
    print(f"✅ Device: {device}")
    
    # Test session creation with proper device
    session = engine.InferenceSession(devices=[driver_device])
    print("✅ InferenceSession created")
    
    # Test basic graph creation
    input_type = TensorType(DType.float32, (2, 3), device=device)
    with Graph("test_graph", input_types=[input_type]) as graph:
        x = graph.inputs[0]
        y = ops.add(x, x)  # Simple operation
        graph.output(y)
    
    compiled_graph = session.load(graph)
    print("✅ Graph compilation successful")
    
    # Test execution
    import numpy as np
    test_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    tensor_input = Tensor.from_numpy(test_input).to(driver_device)
    outputs = compiled_graph.execute(tensor_input)
    result = outputs[0].to_numpy()
    print(f"✅ Graph execution successful: {result}")
    
except Exception as e:
    print(f"❌ MAX Graph error: {e}")
    import traceback
    traceback.print_exc()