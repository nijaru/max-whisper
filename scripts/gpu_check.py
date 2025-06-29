#!/usr/bin/env python3
"""
Quick GPU compatibility check
"""

try:
    import torch
    print("✅ PyTorch available")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠️ CUDA not available - will use CPU")
except ImportError:
    print("❌ PyTorch not available")

try:
    from max import engine
    print("✅ MAX Graph available")
except ImportError:
    print("❌ MAX Graph not available")

try:
    import whisper
    print("✅ OpenAI Whisper available")
except ImportError:
    print("❌ OpenAI Whisper not available")

print("\n🎯 Status: Environment check complete")