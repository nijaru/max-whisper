#!/usr/bin/env python3
"""Test CUDA setup for GPU development"""

def test_pytorch_cuda():
    """Test PyTorch CUDA availability"""
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name()}")
        return torch.cuda.is_available()
    except ImportError:
        print("❌ PyTorch not available")
        return False

def test_max_graph_cuda():
    """Test MAX Graph CUDA availability"""
    try:
        from max import engine
        from max.graph import DeviceRef
        
        print("✅ MAX Graph imported successfully")
        
        # Test GPU device creation
        try:
            gpu_device = DeviceRef.GPU()
            print("✅ MAX Graph GPU device created successfully")
            return True
        except Exception as e:
            print(f"❌ MAX Graph GPU device creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ MAX Graph not available: {e}")
        return False

def test_whisper_models():
    """Test if we can load Whisper models"""
    try:
        import whisper
        print("✅ OpenAI Whisper available")
        
        # Try CPU model
        try:
            model = whisper.load_model("tiny", device="cpu")
            print("✅ OpenAI Whisper CPU model loaded")
        except Exception as e:
            print(f"❌ OpenAI Whisper CPU failed: {e}")
        
        return True
    except ImportError:
        print("❌ OpenAI Whisper not available")
        return False

def main():
    print("🔍 CUDA Environment Diagnostic")
    print("=" * 50)
    
    print("\n🧪 PyTorch CUDA Test")
    print("-" * 30)
    pytorch_cuda = test_pytorch_cuda()
    
    print("\n🚀 MAX Graph CUDA Test")
    print("-" * 30)
    max_cuda = test_max_graph_cuda()
    
    print("\n🎤 Whisper Models Test")
    print("-" * 30)
    whisper_available = test_whisper_models()
    
    print("\n📊 Summary")
    print("-" * 30)
    print(f"PyTorch CUDA: {'✅' if pytorch_cuda else '❌'}")
    print(f"MAX Graph GPU: {'✅' if max_cuda else '❌'}")
    print(f"Whisper Available: {'✅' if whisper_available else '❌'}")
    
    if pytorch_cuda and max_cuda:
        print("\n🎯 GPU development environment ready!")
        print("   You can proceed with GPU benchmarking")
    else:
        print("\n⚠️ GPU environment needs setup")
        if not pytorch_cuda:
            print("   - Install PyTorch with CUDA support")
        if not max_cuda:
            print("   - Fix MAX Graph GPU device access")

if __name__ == "__main__":
    main()