#!/usr/bin/env python3
"""
Enhanced benchmark runner with JSON output, error handling, and structured logging
"""

import time
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

# Add max-whisper to path
sys.path.append(str(Path(__file__).parent.parent / "max-whisper"))

from utils.logging import setup_logger, BenchmarkLogger, log_execution_time

class WhisperBenchmarkRunner:
    """Enhanced benchmark runner with proper error handling and logging"""
    
    def __init__(self, json_output: bool = False, log_level: str = "INFO"):
        self.logger = setup_logger("benchmark", level=log_level, json_output=json_output)
        self.benchmark_logger = BenchmarkLogger(self.logger)
        self.results = []
        
    def run_implementation(self, impl_name: str, model_size: str, audio_file: str) -> Dict[str, Any]:
        """Run a single implementation with proper error handling"""
        
        result = {
            'implementation': impl_name,
            'model_size': model_size,
            'audio_file': audio_file,
            'timestamp': time.time()
        }
        
        try:
            with log_execution_time(f"{impl_name}_benchmark", self.logger, 
                                  implementation=impl_name, model_size=model_size):
                
                if impl_name == "cpu":
                    from whisper_cpu import WhisperCPU
                    model = WhisperCPU(model_size=model_size)
                    
                elif impl_name == "gpu":
                    from whisper_gpu import WhisperGPU  
                    model = WhisperGPU(model_size=model_size, use_gpu=True)
                    
                elif impl_name == "max":
                    from whisper_max import WhisperMAX
                    model = WhisperMAX(model_size=model_size, use_gpu=True)
                    
                else:
                    raise ValueError(f"Unknown implementation: {impl_name}")
                
                if not model.available:
                    result.update({
                        'status': 'unavailable',
                        'error': f"{impl_name} model not available",
                        'execution_time': 0.0
                    })
                    return result
                
                # Execute with timing
                start_time = time.time()
                transcription = model.transcribe(audio_file=audio_file)
                execution_time = time.time() - start_time
                
                result.update({
                    'status': 'success',
                    'execution_time': execution_time,
                    'transcription': transcription,
                    'transcription_length': len(transcription) if transcription else 0
                })
                
                # Log structured result
                self.benchmark_logger.log_benchmark_result(
                    implementation=impl_name,
                    model_size=model_size,
                    audio_file=audio_file,
                    execution_time=execution_time,
                    result_text=transcription
                )
                
        except Exception as e:
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            
            result.update({
                'status': 'error',
                'execution_time': 0.0,
                'error': error_details
            })
            
            self.logger.error(f"Benchmark failed for {impl_name}: {e}", 
                            extra={'extra_fields': error_details})
            
        return result
    
    def run_all_implementations(self, model_size: str = "tiny", 
                              audio_file: str = "audio_samples/modular_video.wav",
                              implementations: List[str] = None) -> List[Dict[str, Any]]:
        """Run benchmarks for all implementations"""
        
        if implementations is None:
            implementations = ["cpu", "gpu", "max"]
            
        self.logger.info(f"Starting benchmark run: {implementations} with {model_size} model")
        
        # Log system info
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if gpu_available else None
            self.benchmark_logger.log_system_info(gpu_available, cuda_version)
        except ImportError:
            self.benchmark_logger.log_system_info(False)
        
        results = []
        for impl in implementations:
            self.logger.info(f"Running {impl} implementation...")
            result = self.run_implementation(impl, model_size, audio_file)
            results.append(result)
            self.results.append(result)
            
        return results
    
    def save_results(self, output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        successful = [r for r in results if r['status'] == 'success']
        
        if successful:
            fastest = min(successful, key=lambda x: x['execution_time'])
            print(f"\n✅ Successful implementations: {len(successful)}/{len(results)}")
            print(f"🏆 Fastest: {fastest['implementation']} - {fastest['execution_time']:.3f}s")
            
            print(f"\n📊 Performance comparison:")
            for result in results:
                status_icon = "✅" if result['status'] == 'success' else "❌"
                if result['status'] == 'success':
                    print(f"  {status_icon} {result['implementation']:8} {result['execution_time']:8.3f}s")
                else:
                    print(f"  {status_icon} {result['implementation']:8} {'FAILED':>8}")
        else:
            print("❌ No implementations completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Whisper benchmark runner")
    parser.add_argument("--model-size", default="tiny", choices=["tiny", "small", "base"], 
                       help="Model size to benchmark")
    parser.add_argument("--audio-file", default="audio_samples/modular_video.wav",
                       help="Audio file to process")
    parser.add_argument("--implementations", nargs="+", default=["cpu", "gpu", "max"],
                       choices=["cpu", "gpu", "max"], help="Implementations to test")
    parser.add_argument("--json-output", action="store_true", 
                       help="Output structured JSON logs")
    parser.add_argument("--save-results", help="Save results to JSON file")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARN", "ERROR"])
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = WhisperBenchmarkRunner(
        json_output=args.json_output,
        log_level=args.log_level
    )
    
    # Run benchmarks
    results = runner.run_all_implementations(
        model_size=args.model_size,
        audio_file=args.audio_file,
        implementations=args.implementations
    )
    
    # Save results if requested
    if args.save_results:
        runner.save_results(args.save_results)
    
    # Print summary (unless pure JSON mode)
    if not args.json_output:
        runner.print_summary(results)

if __name__ == "__main__":
    main()