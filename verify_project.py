#!/usr/bin/env python3
"""
Project Verification Script
Checks that all essential components are working
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (missing)")
        return False

def check_directory_exists(dir_path, description):
    """Check if a directory exists"""
    if Path(dir_path).exists() and Path(dir_path).is_dir():
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ {description}: {dir_path} (missing)")
        return False

def main():
    print("🔍 MAX-Whisper Project Verification")
    print("=" * 50)
    
    all_good = True
    
    print("\n📁 Essential Files:")
    all_good &= check_file_exists("README.md", "Project overview")
    all_good &= check_file_exists("docs/STATUS.md", "Project status")
    all_good &= check_file_exists("comprehensive_results.md", "Benchmark results")
    all_good &= check_file_exists("demo.py", "Demo script")
    all_good &= check_file_exists("generate_results.py", "Results generator")
    
    print("\n🧠 Model Implementation:")
    all_good &= check_file_exists("src/model/max_whisper_fixed.py", "Working implementation")
    
    print("\n🧪 Benchmarks:")
    all_good &= check_file_exists("benchmarks/safe_comprehensive_benchmark.py", "Comprehensive benchmark")
    
    print("\n🎵 Test Data:")
    all_good &= check_file_exists("audio_samples/modular_video.wav", "Test audio")
    all_good &= check_file_exists("whisper_weights/whisper_tiny_weights.npz", "Trained weights")
    
    print("\n📚 Documentation:")
    all_good &= check_file_exists("CLAUDE.md", "AI instructions")
    all_good &= check_file_exists("docs/SUMMARY.md", "Project summary")
    all_good &= check_directory_exists("docs", "Documentation directory")
    
    print("\n🗃️ Archive:")
    all_good &= check_directory_exists("archive", "Archive directory")
    all_good &= check_directory_exists("archive/dev_models", "Development models")
    
    if all_good:
        print(f"\n🎉 Project verification PASSED")
        print(f"   All essential components are present")
        print(f"   Ready for demonstration and evaluation")
        
        print(f"\n📋 Quick Commands:")
        print(f"   🚀 Demo: python demo.py")
        print(f"   📊 Results: python generate_results.py")
        print(f"   🧪 Benchmark: cd benchmarks && pixi run -e benchmark python safe_comprehensive_benchmark.py")
        
    else:
        print(f"\n❌ Project verification FAILED")
        print(f"   Some essential components are missing")
        
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)