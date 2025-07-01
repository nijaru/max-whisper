#!/usr/bin/env python3
"""
Demo script showing the improved testing and benchmarking capabilities
"""

import sys
from pathlib import Path

# Add max-whisper to path
sys.path.append(str(Path(__file__).parent.parent / "max-whisper"))

from utils.logging import setup_logger, BenchmarkLogger, log_execution_time
import time

def demo_logging():
    """Demonstrate the new logging capabilities"""
    print("🔬 Testing Enhanced Logging System")
    print("=" * 50)
    
    # Setup logger with JSON output
    logger = setup_logger("demo", level="INFO", json_output=False)
    benchmark_logger = BenchmarkLogger(logger)
    
    # Demo execution timing
    with log_execution_time("sample_operation", logger, operation_type="demo"):
        time.sleep(0.1)  # Simulate work
        logger.info("Performing sample operation...")
    
    # Demo benchmark logging
    benchmark_logger.log_benchmark_result(
        implementation="demo_impl",
        model_size="tiny",
        audio_file="demo.wav",
        execution_time=0.123,
        result_text="This is a demo transcription result",
        metadata={"demo": True, "version": "1.0"}
    )
    
    print("✅ Logging demo completed - check output above")

def demo_json_logging():
    """Demonstrate JSON logging output"""
    print("\n🔬 Testing JSON Logging Output")
    print("=" * 50)
    
    # Setup logger with JSON output
    logger = setup_logger("demo_json", level="INFO", json_output=True)
    benchmark_logger = BenchmarkLogger(logger)
    
    print("JSON log entries:")
    
    logger.info("Starting JSON logging demo")
    
    benchmark_logger.log_benchmark_result(
        implementation="json_demo",
        model_size="tiny",
        audio_file="test.wav",
        execution_time=0.456,
        result_text="JSON logging test"
    )
    
    print("✅ JSON logging demo completed")

def demo_tasks():
    """Show available pixi tasks"""
    print("\n🔬 Available Pixi Tasks")
    print("=" * 50)
    
    tasks = {
        "Basic Tasks": [
            "pixi run test",
            "pixi run test-max",
            "pixi run graph-test"
        ],
        "Benchmark Tasks": [
            "pixi run -e benchmark demo",
            "pixi run -e benchmark benchmark", 
            "pixi run -e benchmark benchmark-json",
            "pixi run -e benchmark benchmark-save",
            "pixi run -e benchmark test-cpu",
            "pixi run -e benchmark test-gpu",
            "pixi run -e benchmark test-max"
        ]
    }
    
    for category, task_list in tasks.items():
        print(f"\n📋 {category}:")
        for task in task_list:
            print(f"  {task}")
    
    print("\n💡 Try running: pixi run -e benchmark benchmark-json")
    print("   This will give you structured JSON output for parsing!")

def main():
    """Run all demos"""
    print("🚀 MAX-Whisper Improvements Demo")
    print("=" * 60)
    
    demo_logging()
    demo_json_logging() 
    demo_tasks()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("\nKey improvements:")
    print("  🔍 Structured logging with JSON support")
    print("  📊 Enhanced benchmark runner with error handling")
    print("  🧪 Comprehensive test suite")
    print("  ⚙️  Pixi tasks replacing complex Makefile")
    print("  📋 Clear Mojo conversion strategy")
    
    print("\nNext steps:")
    print("  1. Run: pixi run -e benchmark test")
    print("  2. Try: pixi run -e benchmark benchmark-json")
    print("  3. Focus on MAX Graph semantic quality improvements")

if __name__ == "__main__":
    main()