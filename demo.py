#!/usr/bin/env python3
"""
MAX-Whisper Demo Script
Simple demonstration of working speech recognition
"""

import sys
import os

def main():
    print("🚀 MAX-Whisper Demo")
    print("=" * 50)
    
    # Change to project root if needed
    if os.path.basename(os.getcwd()) == 'benchmarks':
        os.chdir('..')
    
    try:
        print("\n🔧 Loading MAX-Whisper...")
        sys.path.append('src')
        from model.max_whisper_fixed import MAXWhisperFixed
        
        # Create model
        model = MAXWhisperFixed(use_gpu=True)
        
        if not model.available:
            print("❌ Model not available")
            return
        
        print("\n🎯 Running transcription...")
        result = model.transcribe()
        
        print(f"\n📝 Transcription Result:")
        print(f"   {result[:100]}{'...' if len(result) > 100 else ''}")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"   Full implementation in: src/model/max_whisper_fixed.py")
        print(f"   Comprehensive benchmark: benchmarks/safe_comprehensive_benchmark.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(f"   Try running: pixi run -e benchmark python demo.py")

if __name__ == "__main__":
    main()