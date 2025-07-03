#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'max-whisper'))

import json

def test_health_check():
    """Test the health check functionality"""
    print("🏥 Testing production health check...")
    
    try:
        from whisper_max import WhisperMAX
        
        # Test hybrid mode health check
        decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=False)
        health = decoder.health_check()
        
        print(f"📊 Health check results:")
        print(json.dumps(health, indent=2))
        
        if health['healthy']:
            print("✅ System healthy")
        else:
            print("⚠️ System has issues")
            for error in health['errors']:
                print(f"   - {error}")
        
        # Test full MAX Graph mode health check
        print(f"\n🔬 Testing full MAX Graph health check...")
        decoder_full = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
        health_full = decoder_full.health_check()
        
        print(f"📊 Full MAX Graph health:")
        print(json.dumps(health_full, indent=2))
        
        return health['healthy'] and health_full['healthy']
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def test_error_recovery():
    """Test error recovery mechanisms"""
    print("🔄 Testing production error recovery...")
    
    try:
        from whisper_max import WhisperMAX
        
        # Create decoder
        decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=False)
        
        # Test with invalid input to trigger error recovery
        print("🎯 Testing error recovery with invalid audio path...")
        result = decoder.transcribe("nonexistent_file.wav")
        
        print(f"📝 Recovery result: {result[:100]}...")
        
        if "FALLBACK" in result or "error" in result.lower():
            print("✅ Error recovery mechanism activated")
            return True
        else:
            print("⚠️ Error recovery not detected")
            return False
            
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Production Error Handling & Recovery Test")
    print("=" * 50)
    
    health_ok = test_health_check()
    recovery_ok = test_error_recovery()
    
    print(f"\n📊 Test Results:")
    print(f"   Health check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Error recovery: {'✅ PASS' if recovery_ok else '❌ FAIL'}")
    
    if health_ok and recovery_ok:
        print(f"\n🎯 Production error handling: ✅ READY")
    else:
        print(f"\n⚠️ Production error handling: NEEDS WORK")