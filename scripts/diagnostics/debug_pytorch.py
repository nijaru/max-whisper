#!/usr/bin/env python3
"""Debug PyTorch installation and CUDA availability"""

import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path[:3])  # First 3 entries

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch location: {torch.__file__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA compiled: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available - possible reasons:")
        print("  1. PyTorch was compiled without CUDA support")
        print("  2. CUDA runtime libraries not found")
        print("  3. NVIDIA drivers not compatible")
        
except Exception as e:
    print(f"Error importing PyTorch: {e}")

# Check what's in the environment
import os
print(f"\nEnvironment check:")
print(f"PATH entries with 'pixi': {[p for p in os.environ.get('PATH', '').split(':') if 'pixi' in p]}")