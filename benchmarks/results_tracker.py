#!/usr/bin/env python3
"""
Centralized Results Tracking for MAX Graph Whisper
Consolidates benchmark results and provides unified reporting
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class ResultsTracker:
    """Centralized results tracking and reporting"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def record_benchmark(self, 
                        implementation: str,
                        audio_file: str,
                        performance: Dict[str, float],
                        quality: Dict[str, Any],
                        metadata: Optional[Dict] = None) -> None:
        """Record a benchmark result"""
        
        result = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "implementation": implementation,
            "audio_file": audio_file,
            "performance": performance,
            "quality": quality,
            "metadata": metadata or {}
        }
        
        # Save individual result
        filename = f"{self.session_id}_{implementation}_{int(datetime.now().timestamp())}.json"
        with open(self.results_dir / filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    def get_latest_results(self, limit: int = 10) -> List[Dict]:
        """Get latest benchmark results"""
        
        result_files = sorted(self.results_dir.glob("*.json"), 
                            key=lambda x: x.stat().st_mtime, reverse=True)
        
        results = []
        for file_path in result_files[:limit]:
            try:
                with open(file_path, 'r') as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        return results
    
    def generate_summary_report(self) -> Dict:
        """Generate summary report of all results"""
        
        results = self.get_latest_results(limit=100)
        
        # Group by implementation
        by_implementation = {}
        for result in results:
            impl = result["implementation"]
            if impl not in by_implementation:
                by_implementation[impl] = []
            by_implementation[impl].append(result)
        
        # Generate summary statistics
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_results": len(results),
            "implementations": {},
            "latest_session": self.session_id
        }
        
        for impl, impl_results in by_implementation.items():
            if not impl_results:
                continue
                
            latest = impl_results[0]  # Most recent
            
            # Calculate averages for performance metrics
            perf_metrics = {}
            if "performance" in latest:
                for metric, value in latest["performance"].items():
                    if isinstance(value, (int, float)):
                        perf_metrics[metric] = value
            
            # Get quality metrics
            quality_metrics = latest.get("quality", {})
            
            summary["implementations"][impl] = {
                "latest_result": latest["timestamp"],
                "total_runs": len(impl_results),
                "performance": perf_metrics,
                "quality": quality_metrics,
                "status": "working" if perf_metrics else "unknown"
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print formatted summary report"""
        
        summary = self.generate_summary_report()
        
        print("🎯 MAX GRAPH WHISPER - RESULTS SUMMARY")
        print("=" * 60)
        print(f"Generated: {summary['generated_at']}")
        print(f"Total Results: {summary['total_results']}")
        print(f"Session: {summary['latest_session']}")
        
        print(f"\n📊 IMPLEMENTATIONS:")
        
        for impl, data in summary["implementations"].items():
            status_icon = "✅" if data["status"] == "working" else "❓"
            print(f"\n{status_icon} {impl.upper()}")
            print(f"   Last run: {data['latest_result']}")
            print(f"   Total runs: {data['total_runs']}")
            
            if data["performance"]:
                print(f"   Performance:")
                for metric, value in data["performance"].items():
                    if metric.endswith("_time"):
                        print(f"     {metric}: {value:.2f}s")
                    else:
                        print(f"     {metric}: {value}")
            
            if data["quality"]:
                print(f"   Quality:")
                for metric, value in data["quality"].items():
                    if isinstance(value, (int, float)):
                        print(f"     {metric}: {value}")
                    else:
                        print(f"     {metric}: {str(value)[:50]}...")


def main():
    """Demo/test the results tracker"""
    
    tracker = ResultsTracker()
    
    # Example usage - record some sample results
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("📝 Recording demo results...")
        
        # Sample MAX Graph result
        tracker.record_benchmark(
            implementation="max_graph",
            audio_file="modular_video.wav",
            performance={
                "total_time": 1.0,
                "encoder_time": 0.047,
                "decoder_time": 0.95
            },
            quality={
                "text_length": 838,
                "baseline_percentage": 41.2,
                "cosine_similarity": 0.999993
            },
            metadata={"model_size": "tiny", "gpu": True}
        )
        
        print("✅ Demo results recorded")
    
    # Always show summary
    tracker.print_summary()


if __name__ == "__main__":
    main()