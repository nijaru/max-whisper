#!/usr/bin/env python3
"""
Generate Results Script
Runs comprehensive benchmark and displays results
"""

import subprocess
import sys
import os
from pathlib import Path

def run_benchmark():
    """Run the comprehensive benchmark"""
    print("🚀 Running MAX-Whisper Comprehensive Benchmark")
    print("=" * 60)
    
    # Ensure we're in the right directory
    benchmark_dir = Path("benchmarks")
    if not benchmark_dir.exists():
        print("❌ benchmarks/ directory not found")
        return False
    
    # Run the benchmark
    try:
        os.chdir("benchmarks")
        result = subprocess.run([
            "pixi", "run", "-e", "benchmark", "python", "safe_comprehensive_benchmark.py"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Benchmark completed successfully")
            return True
        else:
            print(f"❌ Benchmark failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out")
        return False
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return False
    finally:
        os.chdir("..")

def display_results():
    """Display the results"""
    results_file = Path("comprehensive_results.md")
    
    if not results_file.exists():
        print("❌ Results file not found")
        return
    
    print("\n📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Extract and display the results table
    lines = content.split('\n')
    in_table = False
    
    for line in lines:
        if line.startswith('| Model'):
            in_table = True
            print(line)
        elif in_table and line.startswith('|'):
            print(line)
        elif in_table and not line.startswith('|') and line.strip():
            break
    
    print(f"\n📄 Full results available in: {results_file}")

def main():
    print("🎯 MAX-Whisper Results Generator")
    print("=" * 50)
    
    # Check if results already exist
    if Path("comprehensive_results.md").exists():
        print("\n📊 Existing results found. Displaying...")
        display_results()
        
        # Check for --force flag
        if len(sys.argv) > 1 and sys.argv[1] == '--force':
            print("\n🔄 --force flag detected, generating new results...")
        else:
            print("\n✅ Using existing results")
            print("   💡 Use 'python generate_results.py --force' to generate new results")
            return
    
    # Run new benchmark
    print("\n🚀 Generating new benchmark results...")
    
    if run_benchmark():
        print("\n📊 Displaying results...")
        display_results()
        
        print(f"\n🎉 Results generation complete!")
        print(f"   📄 Detailed results: comprehensive_results.md")
        print(f"   🎯 Quick demo: python demo.py")
        print(f"   📊 Status: STATUS.md")
    else:
        print("❌ Failed to generate results")

if __name__ == "__main__":
    main()