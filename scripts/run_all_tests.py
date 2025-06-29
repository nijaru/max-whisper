#!/usr/bin/env python3
"""
Unified Test Suite - Run all MAX-Whisper tests
"""

import subprocess
import sys
import os

def run_test(name, command, env="default"):
    print(f"🔍 Running {name}...")
    try:
        result = subprocess.run(
            f"export PATH="$HOME/.pixi/bin:$PATH" && pixi run -e {env} {command}",
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✅ {name}: PASS")
            return True
        else:
            print(f"❌ {name}: FAIL")
            print(f"   Error: {result.stderr[:100]}...")
            return False
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")
        return False

def main():
    print("🚀 MAX-Whisper Unified Test Suite")
    print("=" * 60)
    
    tests = [
        ("CUDA Setup", "python tests/compatibility/test_cuda_setup.py", "benchmark"),
        ("GPU Compatibility", "python scripts/diagnostics/diagnose_gpu_compatibility.py", "default"),
        ("MAX Graph Components", "python tests/test_everything.py", "default"),
        ("GPU Implementation", "python src/model/max_whisper_gpu_direct.py", "default"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, command, env in tests:
        if run_test(name, command, env):
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passing!")
    else:
        print("🔧 Some tests need attention")

if __name__ == "__main__":
    main()
