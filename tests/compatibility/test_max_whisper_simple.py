#!/usr/bin/env python3
"""Test basic MAX-Whisper functionality to isolate uint16 error"""

import sys
import os
sys.path.append('src/model')

print("🔍 Testing MAX-Whisper imports...")

try:
    print("Testing torch import...")
    import torch
    print(f"✅ PyTorch {torch.__version__} imported")
    
    print("Testing numpy import...")
    import numpy as np
    print("✅ NumPy imported")
    
    print("Testing MAX Graph import...")
    from max import engine
    from max.driver import Tensor
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops
    print("✅ MAX Graph imported")
    
    print("Testing MAX-Whisper trained CPU import...")
    from max_whisper_trained_cpu import MAXWhisperTrainedCPU
    print("✅ MAX-Whisper trained CPU imported")
    
    print("Testing model creation...")
    model = MAXWhisperTrainedCPU()
    print("✅ Model created successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()