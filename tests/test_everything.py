"""
Comprehensive test of all MAX-Whisper components
"""

import sys
import traceback

def test_component(name, test_func):
    """Test a component and return result"""
    try:
        print(f"\n🔍 Testing {name}...")
        result = test_func()
        if result:
            print(f"✅ {name}: PASS")
            return True
        else:
            print(f"❌ {name}: FAIL")
            return False
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")
        return False

def test_simple_encoder():
    """Test simple encoder"""
    from src.model.max_whisper_real_simple import test_simple_real_encoder
    return test_simple_real_encoder()

def test_attention():
    """Test multi-head attention"""
    from src.model.max_whisper_step2 import test_simple_multihead
    return test_simple_multihead()

def test_encoder_decoder():
    """Test encoder-decoder"""
    from src.model.max_whisper_decoder import test_encoder_decoder
    return test_encoder_decoder()

def test_complete_model():
    """Test complete end-to-end model"""
    from src.model.max_whisper_complete import demo_complete_transcription
    return demo_complete_transcription()

def main():
    """Run all tests"""
    print("="*60)
    print("COMPREHENSIVE MAX-WHISPER TESTING")
    print("="*60)
    
    tests = [
        ("Simple Encoder", test_simple_encoder),
        ("Multi-Head Attention", test_attention),
        ("Encoder-Decoder", test_encoder_decoder),
        ("Complete Model", test_complete_model),
    ]
    
    results = []
    for name, test_func in tests:
        results.append(test_component(name, test_func))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for i, (name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{name:20} {status}")
    
    total_pass = sum(results)
    total_tests = len(results)
    
    print(f"\nTotal: {total_pass}/{total_tests} tests passed")
    
    if total_pass == total_tests:
        print("🎉 ALL TESTS PASSING!")
        return True
    else:
        print("💥 Some tests failed - need debugging")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)