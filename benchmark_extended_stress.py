#!/usr/bin/env python3
"""
Extended Stress Testing for KV Cache Implementation (Phase 10)

Production-grade stress testing with:
- Extended sequence lengths (50-100+ tokens)
- Memory usage validation under load
- Production reliability testing
- Enterprise deployment readiness
"""

import time
import sys
import numpy as np
import tempfile
import json
import gc
import os
from pathlib import Path
import resource

sys.path.append('max-whisper')
from whisper_max import WhisperMAX

class ExtendedStressTester:
    def __init__(self):
        self.results = {
            'extended_stress': [],
            'memory_usage': [],
            'reliability_tests': [],
            'production_metrics': {}
        }
        
    def setup_test_audio(self, duration=30):
        """Create extended test audio for stress testing"""
        import librosa
        
        # Use longer audio sample for extended testing
        audio, _ = librosa.load('audio_samples/modular_video.wav', sr=16000, duration=duration)
        
        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        # Save as numpy array for now
        np.save(temp_file.name.replace('.wav', '.npy'), audio)
        return temp_file.name.replace('.wav', '.npy'), audio
        
    def measure_memory_usage(self):
        """Get memory usage using resource module"""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            'rss_mb': usage.ru_maxrss / 1024,  # Convert KB to MB on Linux
            'vms_mb': 0,  # Not available via resource module
            'cpu_percent': 0  # Not available via resource module
        }
        
    def stress_test_implementation(self, decoder, audio_file, sequence_lengths):
        """Run extended stress test with detailed monitoring"""
        results = []
        
        print(f"🔬 Extended Stress Testing: {len(sequence_lengths)} sequence lengths")
        print("Length | Time(s) | Time/Token | Tok/s | Memory | Status | Assessment")
        print("-------|---------|------------|-------|--------|--------|----------")
        
        for i, max_len in enumerate(sequence_lengths):
            try:
                # Clear memory before each test
                gc.collect()
                mem_before = self.measure_memory_usage()
                
                # Configure for extended testing
                decoder.max_graph_decoder.max_seq_len = max_len
                
                # Time the transcription
                start_time = time.time()
                text = decoder.transcribe(audio_file)
                total_time = time.time() - start_time
                
                # Measure memory after
                mem_after = self.measure_memory_usage()
                memory_delta = mem_after['rss_mb'] - mem_before['rss_mb']
                
                # Calculate metrics
                time_per_token = (total_time / max_len) * 1000  # ms
                tokens_per_sec = max_len / total_time if total_time > 0 else 0
                
                # Assessment based on sequence length
                if max_len <= 30:
                    assessment = "Baseline"
                elif max_len <= 50:
                    assessment = "Extended"
                elif max_len <= 75:
                    assessment = "Stress"
                elif max_len <= 100:
                    assessment = "Production"
                else:
                    assessment = "Enterprise"
                
                result = {
                    'max_length': max_len,
                    'time': total_time,
                    'time_per_token': time_per_token,
                    'tokens_per_sec': tokens_per_sec,
                    'text_length': len(text),
                    'memory_delta': memory_delta,
                    'memory_before': mem_before,
                    'memory_after': mem_after,
                    'text': text,
                    'success': True,
                    'assessment': assessment
                }
                
                results.append(result)
                
                print(f"{max_len:6d} | {total_time:7.3f} | {time_per_token:8.1f}ms | "
                      f"{tokens_per_sec:5.1f} | {memory_delta:6.1f}MB | ✅ Pass | {assessment}")
                
            except Exception as e:
                print(f"{max_len:6d} | ERROR   | ERROR      | ERROR | ERROR  | ❌ Fail | {str(e)[:20]}")
                result = {
                    'max_length': max_len,
                    'success': False,
                    'error': str(e),
                    'assessment': 'Failed'
                }
                results.append(result)
                
        return results
        
    def run_extended_stress_test(self):
        """Run comprehensive extended stress test"""
        print("🚀 Extended Stress Test - Phase 10 Production Validation")
        print("=" * 70)
        
        # Extended sequence lengths for production validation
        extended_lengths = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150]
        
        try:
            # Setup extended test audio
            audio_file, audio_data = self.setup_test_audio(duration=30)
            
            print(f"📁 Test audio: {len(audio_data)/16000:.1f}s duration")
            print(f"🎯 Extended testing: {min(extended_lengths)}-{max(extended_lengths)} tokens")
            print(f"🏭 Production target: Linear scaling to 150+ tokens")
            
            # Initialize KV-cached decoder
            decoder = WhisperMAX(model_size='tiny', use_gpu=True, full_max_graph=True)
            
            if decoder.max_graph_decoder is None:
                print("❌ KV-cached decoder not available")
                return False
                
            print(f"\n🔬 Running extended KV cache stress test...")
            
            # Run stress test
            self.results['extended_stress'] = self.stress_test_implementation(
                decoder, audio_file, extended_lengths
            )
            
            print(f"\n✅ Extended stress test completed!")
            
            # Analyze results
            self.analyze_extended_results()
            
            # Save results
            self.save_results('extended_stress_results.json')
            
            return True
            
        except Exception as e:
            print(f"❌ Extended stress test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup
            if 'audio_file' in locals():
                try:
                    os.unlink(audio_file)
                except:
                    pass
                    
    def analyze_extended_results(self):
        """Analyze extended stress test results"""
        results = self.results['extended_stress']
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("❌ No successful results to analyze")
            return
            
        print(f"\n📊 Extended Stress Test Analysis:")
        print("=" * 50)
        
        # Performance metrics
        lengths = [r['max_length'] for r in successful_results]
        times = [r['time'] for r in successful_results]
        time_per_token = [r['time_per_token'] for r in successful_results]
        tok_per_sec = [r['tokens_per_sec'] for r in successful_results]
        
        print(f"📈 Performance Summary:")
        print(f"   Sequences tested: {len(successful_results)}/{len(results)}")
        print(f"   Success rate: {len(successful_results)/len(results)*100:.1f}%")
        print(f"   Max sequence length: {max(lengths)} tokens")
        print(f"   Average speed: {np.mean(tok_per_sec):.1f} tokens/sec")
        print(f"   Peak speed: {max(tok_per_sec):.1f} tokens/sec")
        
        # Scaling analysis
        variance = np.var(time_per_token)
        print(f"\n🎯 Scaling Characteristics:")
        print(f"   Time/token variance: {variance:.6f}")
        
        if variance < 0.001:
            scaling_assessment = "✅ EXCELLENT: Perfect linear scaling"
        elif variance < 0.005:
            scaling_assessment = "✅ GOOD: Near-linear scaling"
        elif variance < 0.01:
            scaling_assessment = "⚠️ ACCEPTABLE: Minor scaling issues"
        else:
            scaling_assessment = "❌ POOR: Significant scaling problems"
            
        print(f"   Assessment: {scaling_assessment}")
        
        # Production readiness
        production_results = [r for r in successful_results if r['max_length'] >= 75]
        if production_results:
            prod_avg_speed = np.mean([r['tokens_per_sec'] for r in production_results])
            prod_min_speed = min([r['tokens_per_sec'] for r in production_results])
            
            print(f"\n🏭 Production Readiness (75+ tokens):")
            print(f"   Production sequences: {len(production_results)}")
            print(f"   Average speed: {prod_avg_speed:.1f} tokens/sec")
            print(f"   Minimum speed: {prod_min_speed:.1f} tokens/sec")
            
            if prod_min_speed > 50:
                prod_assessment = "✅ EXCELLENT: Production-ready"
            elif prod_min_speed > 30:
                prod_assessment = "✅ GOOD: Acceptable for production"
            else:
                prod_assessment = "⚠️ WARNING: May need optimization"
                
            print(f"   Assessment: {prod_assessment}")
            
        # Enterprise readiness
        enterprise_results = [r for r in successful_results if r['max_length'] >= 100]
        if enterprise_results:
            ent_avg_speed = np.mean([r['tokens_per_sec'] for r in enterprise_results])
            
            print(f"\n🏢 Enterprise Readiness (100+ tokens):")
            print(f"   Enterprise sequences: {len(enterprise_results)}")
            print(f"   Average speed: {ent_avg_speed:.1f} tokens/sec")
            
            if ent_avg_speed > 40:
                ent_assessment = "✅ EXCELLENT: Enterprise-ready"
            elif ent_avg_speed > 25:
                ent_assessment = "✅ GOOD: Enterprise-capable"
            else:
                ent_assessment = "⚠️ WARNING: Enterprise optimization needed"
                
            print(f"   Assessment: {ent_assessment}")
            
        # Memory efficiency
        memory_deltas = [r['memory_delta'] for r in successful_results if 'memory_delta' in r]
        if memory_deltas:
            avg_memory = np.mean(memory_deltas)
            max_memory = max(memory_deltas)
            
            print(f"\n💾 Memory Efficiency:")
            print(f"   Average memory per test: {avg_memory:.1f} MB")
            print(f"   Peak memory usage: {max_memory:.1f} MB")
            
            if avg_memory < 5:
                mem_assessment = "✅ EXCELLENT: Minimal memory overhead"
            elif avg_memory < 20:
                mem_assessment = "✅ GOOD: Acceptable memory usage"
            else:
                mem_assessment = "⚠️ WARNING: High memory overhead"
                
            print(f"   Assessment: {mem_assessment}")
            
        # Store production metrics
        self.results['production_metrics'] = {
            'success_rate': len(successful_results)/len(results)*100,
            'max_sequence_length': max(lengths),
            'average_speed': np.mean(tok_per_sec),
            'peak_speed': max(tok_per_sec),
            'scaling_variance': variance,
            'production_ready': len(production_results) > 0 and prod_min_speed > 30 if production_results else False,
            'enterprise_ready': len(enterprise_results) > 0 and ent_avg_speed > 25 if enterprise_results else False
        }
        
    def save_results(self, filename):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"💾 Results saved to {filename}")

def main():
    """Run extended stress testing"""
    tester = ExtendedStressTester()
    
    success = tester.run_extended_stress_test()
    
    if success:
        print(f"\n🎯 Extended Stress Testing: ✅ COMPLETED")
        
        metrics = tester.results.get('production_metrics', {})
        if metrics:
            print(f"📊 Production Readiness: {'✅ READY' if metrics.get('production_ready', False) else '⚠️ NEEDS WORK'}")
            print(f"🏢 Enterprise Readiness: {'✅ READY' if metrics.get('enterprise_ready', False) else '⚠️ NEEDS WORK'}")
    else:
        print(f"\n❌ Extended Stress Testing: FAILED")

if __name__ == "__main__":
    main()