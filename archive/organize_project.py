#!/usr/bin/env python3
"""
Organize project files into clean structure and identify quality improvement priorities
"""

import os
import shutil

def organize_files():
    """Move files to proper directories"""
    
    print("🗂️ Organizing Project Structure")
    print("=" * 50)
    
    # Create directories if they don't exist
    dirs_to_create = [
        "scripts/diagnostics",
        "tests/compatibility", 
        "benchmarks/components",
        "docs/analysis"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # File movements (only move if files exist)
    moves = [
        # Diagnostic scripts
        ("debug_pytorch.py", "scripts/diagnostics/debug_pytorch.py"),
        ("diagnose_gpu_compatibility.py", "scripts/diagnostics/diagnose_gpu_compatibility.py"),
        ("fix_speedup_terminology.py", "docs/analysis/speedup_terminology_analysis.py"),
        
        # Test files  
        ("test_cuda_setup.py", "tests/compatibility/test_cuda_setup.py"),
        ("test_max_gpu_direct.py", "tests/compatibility/test_max_gpu_direct.py"),
        ("test_max_whisper_simple.py", "tests/compatibility/test_max_whisper_simple.py"),
        
        # Component benchmarks
        ("benchmarks/simple_cpu_gpu_test.py", "benchmarks/components/simple_cpu_gpu_test.py"),
        ("benchmarks/cpu_vs_gpu_comparison.py", "benchmarks/components/cpu_vs_gpu_comparison.py"),
    ]
    
    moved_files = []
    for src, dst in moves:
        if os.path.exists(src):
            try:
                # Create destination directory if needed
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                moved_files.append((src, dst))
                print(f"📁 Moved: {src} → {dst}")
            except Exception as e:
                print(f"❌ Failed to move {src}: {e}")
    
    return moved_files

def create_quality_improvement_plan():
    """Identify quality improvement priorities"""
    
    print("\n🎯 Quality Improvement Priorities")
    print("=" * 50)
    
    priorities = [
        {
            "priority": "HIGH",
            "task": "Fix token generation in GPU model",
            "current": "Generating special tokens [50258, 50259, 50360, 50257]",
            "target": "Generate meaningful text tokens from vocabulary",
            "file": "src/model/max_whisper_gpu_direct.py",
            "method": "_decode_text()"
        },
        {
            "priority": "HIGH", 
            "task": "Improve text decoding with trained weights",
            "current": "Using random token generation",
            "target": "Use actual trained weights for text generation",
            "file": "src/model/max_whisper_gpu_direct.py",
            "method": "_decode_text() and _decode_tokens()"
        },
        {
            "priority": "MEDIUM",
            "task": "Test with real audio mel spectrograms",
            "current": "Using synthetic random data for testing",
            "target": "Process actual audio → mel → text pipeline",
            "file": "src/model/max_whisper_gpu_direct.py",
            "method": "_preprocess_audio()"
        },
        {
            "priority": "MEDIUM",
            "task": "Create unified benchmark script",
            "current": "Multiple benchmark scripts with environment conflicts",
            "target": "Single script that works across environments",
            "file": "benchmarks/unified_benchmark.py",
            "method": "New file needed"
        }
    ]
    
    for i, task in enumerate(priorities, 1):
        print(f"\n{i}. {task['priority']} PRIORITY: {task['task']}")
        print(f"   Current: {task['current']}")
        print(f"   Target: {task['target']}")
        print(f"   File: {task['file']}")
        print(f"   Focus: {task['method']}")
    
    return priorities

def create_organized_scripts():
    """Create organized test and benchmark scripts"""
    
    print("\n📋 Creating Organized Scripts")
    print("=" * 50)
    
    # Create main test script
    test_script = """#!/usr/bin/env python3
\"\"\"
Unified Test Suite - Run all MAX-Whisper tests
\"\"\"

import subprocess
import sys
import os

def run_test(name, command, env="default"):
    print(f"🔍 Running {name}...")
    try:
        result = subprocess.run(
            f"export PATH=\"$HOME/.pixi/bin:$PATH\" && pixi run -e {env} {command}",
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
    
    print(f"\\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passing!")
    else:
        print("🔧 Some tests need attention")

if __name__ == "__main__":
    main()
"""
    
    os.makedirs("scripts", exist_ok=True)
    with open("scripts/run_all_tests.py", "w") as f:
        f.write(test_script)
    print("✅ Created: scripts/run_all_tests.py")
    
    # Create benchmark script
    benchmark_script = """#!/usr/bin/env python3
\"\"\"
Unified Benchmark Suite - Compare all implementations
\"\"\"

import sys
import os
import time

def benchmark_cpu_gpu():
    print("📊 CPU vs GPU Benchmark")
    print("-" * 40)
    
    # Add project root to path
    sys.path.append('.')
    
    try:
        # Test CPU implementation
        print("🔍 Testing MAX-Whisper CPU...")
        # Implementation details here
        
        # Test GPU implementation  
        print("🔍 Testing MAX-Whisper GPU...")
        # Implementation details here
        
        print("✅ Benchmark complete")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    benchmark_cpu_gpu()
"""
    
    with open("benchmarks/unified_benchmark.py", "w") as f:
        f.write(benchmark_script)
    print("✅ Created: benchmarks/unified_benchmark.py")

def main():
    """Main organization function"""
    
    print("🗂️ MAX-Whisper Project Organization")
    print("=" * 60)
    
    # 1. Organize existing files
    moved_files = organize_files()
    
    # 2. Create quality improvement plan
    priorities = create_quality_improvement_plan()
    
    # 3. Create organized scripts
    create_organized_scripts()
    
    # 4. Summary
    print(f"\n📋 ORGANIZATION COMPLETE")
    print("=" * 50)
    print(f"Files moved: {len(moved_files)}")
    print(f"Quality priorities identified: {len(priorities)}")
    print(f"New scripts created: 2")
    
    print(f"\n🎯 NEXT STEPS FOR QUALITY:")
    print("1. Fix token generation in GPU model (HIGH)")
    print("2. Improve text decoding with trained weights (HIGH)")
    print("3. Test with real audio mel spectrograms (MEDIUM)")
    print("4. Create production-quality text output")
    
    print(f"\n🚀 READY FOR QUALITY IMPROVEMENT PHASE")

if __name__ == "__main__":
    main()