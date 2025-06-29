#!/usr/bin/env python3
"""
Demo Grid Script - Performance Table Generator
Shows a clean performance comparison grid for all 4 implementations
"""

import argparse
import time
import subprocess
import sys
import os

def run_implementation(impl_name, model_size, audio_file):
    """Run a single implementation and extract timing"""
    try:
        # Construct the command
        cmd = [
            "pixi", "run", "-e", "benchmark", "python", 
            f"src/model/{impl_name}.py", 
            "--model-size", model_size
        ]
        
        if audio_file:
            cmd.extend(["--audio-file", audio_file])
        
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode != 0:
            return None, f"Error: {result.stderr[:50]}..."
        
        # Extract timing from output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "Total" in line and ("ms" in line or "s" in line):
                # Extract timing - look for patterns like "1234.5ms" or "1.23s"
                import re
                time_match = re.search(r'(\d+\.?\d*)(ms|s)', line)
                if time_match:
                    time_val = float(time_match.group(1))
                    unit = time_match.group(2)
                    
                    # Convert to seconds
                    if unit == "ms":
                        time_val = time_val / 1000
                    
                    return time_val, None
        
        # Fallback - if no timing found, return a reasonable estimate
        return None, "No timing found"
        
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, f"Error: {str(e)[:50]}..."

def format_time(seconds):
    """Format time nicely"""
    if seconds is None:
        return "Failed"
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

def format_speedup(time_seconds, baseline_seconds):
    """Format speedup calculation"""
    if time_seconds is None or baseline_seconds is None or baseline_seconds == 0:
        return "N/A"
    
    speedup = baseline_seconds / time_seconds
    return f"{speedup:.1f}x"

def print_table_row(impl_name, display_name, time_seconds, baseline_seconds, platform):
    """Print a formatted table row"""
    perf_str = format_time(time_seconds)
    speedup_str = format_speedup(time_seconds, baseline_seconds)
    
    # Pad strings to fit in columns
    impl_col = f"│ {display_name:<23} "
    perf_col = f"│ {perf_str:<12} "
    speedup_col = f"│ {speedup_str:<11} "
    platform_col = f"│ {platform:<14} │"
    
    print(impl_col + perf_col + speedup_col + platform_col)

def main():
    parser = argparse.ArgumentParser(description="Generate performance demo grid")
    parser.add_argument('--model-size', default='tiny', choices=['tiny', 'small', 'base'],
                       help='Model size to test')
    parser.add_argument('--audio-file', 
                       help='Audio file to use (default: audio_samples/modular_video.wav)')
    
    args = parser.parse_args()
    
    # Implementation definitions
    implementations = [
        ("whisper_cpu", "CPU Baseline", "OpenAI Whisper"),
        ("whisper_gpu", "GPU Accelerated", "CUDA + PyTorch"),
        ("whisper_max", "MAX Graph", "MAX Graph Hybrid"),
        ("whisper_max_fast", "MAX Graph Fast", "Ultra-Optimized")
    ]
    
    # Run all implementations
    results = {}
    baseline_time = None
    
    print("Running implementations (this may take a moment)...", file=sys.stderr)
    
    for impl_name, display_name, platform in implementations:
        print(f"  Testing {display_name}...", file=sys.stderr)
        
        time_seconds, error = run_implementation(impl_name, args.model_size, args.audio_file)
        
        results[impl_name] = {
            'display_name': display_name,
            'time': time_seconds,
            'platform': platform,
            'error': error
        }
        
        # Set baseline from CPU implementation
        if impl_name == "whisper_cpu" and time_seconds is not None:
            baseline_time = time_seconds
    
    # Print the table rows
    for impl_name, display_name, platform in implementations:
        result = results[impl_name]
        print_table_row(
            impl_name, 
            result['display_name'],
            result['time'],
            baseline_time,
            result['platform']
        )
    
    # Print any errors to stderr
    errors = [r for r in results.values() if r['error']]
    if errors:
        print("\nWarnings:", file=sys.stderr)
        for result in errors:
            print(f"  {result['display_name']}: {result['error']}", file=sys.stderr)

if __name__ == "__main__":
    main()