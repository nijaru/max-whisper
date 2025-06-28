"""
Whisper implementation comparison for MAX-Whisper hackathon demo.
Compare: OpenAI Whisper vs Faster-Whisper vs MAX-Whisper
"""

import time
import numpy as np
from typing import Dict, Any, Optional
import sys
import os
import subprocess
import librosa

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from audio.preprocessing import load_audio, preprocess_audio


class WhisperBenchmark:
    """Benchmark different Whisper implementations with unified audio pipeline."""
    
    def __init__(self, use_real_audio: bool = False, youtube_url: Optional[str] = None):
        self.results = {}
        self.audio_cache_dir = "audio_cache"
        os.makedirs(self.audio_cache_dir, exist_ok=True)
        
        if use_real_audio and youtube_url:
            print("🎬 Using real audio from YouTube...")
            self.test_audio = self._get_youtube_audio(youtube_url)
            self.audio_source = f"YouTube: {youtube_url}"
        else:
            print("🔊 Using synthetic test audio...")
            self.test_audio = self._generate_test_audio()
            self.audio_source = "Synthetic audio"
    
    def _get_youtube_audio(self, youtube_url: str, max_duration: float = 120.0) -> np.ndarray:
        """Download and load audio from YouTube with caching."""
        # Generate cache filename from URL
        video_id = youtube_url.split('v=')[-1].split('&')[0]
        audio_file = os.path.join(self.audio_cache_dir, f"{video_id}.wav")
        
        if not os.path.exists(audio_file):
            print(f"📥 Downloading audio from YouTube (ID: {video_id})...")
            try:
                # Download audio using yt-dlp
                cmd = [
                    "yt-dlp",
                    "--extract-audio",
                    "--audio-format", "wav",
                    "--audio-quality", "0",  # Best quality
                    "--output", os.path.join(self.audio_cache_dir, f"{video_id}.%(ext)s"),
                    youtube_url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✅ Audio downloaded successfully")
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download audio: {e}")
                print(f"🔄 Falling back to synthetic audio")
                return self._generate_test_audio()
            except FileNotFoundError:
                print(f"❌ yt-dlp not found. Install with: pip install yt-dlp")
                print(f"🔄 Falling back to synthetic audio")
                return self._generate_test_audio()
        else:
            print(f"♻️  Using cached audio file: {audio_file}")
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            
            # Trim to max duration if needed
            if len(audio) > max_duration * sr:
                print(f"✂️  Trimming audio to {max_duration}s")
                audio = audio[:int(max_duration * sr)]
            
            print(f"🎵 Loaded audio: {len(audio)/sr:.1f}s at {sr}Hz")
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Failed to load audio file: {e}")
            print(f"🔄 Falling back to synthetic audio")
            return self._generate_test_audio()
    
    def _generate_test_audio(self, duration: float = 30.0, test_type: str = "speech") -> np.ndarray:
        """Generate realistic test audio for consistent benchmarking."""
        sample_rate = 16000
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        
        if test_type == "speech":
            # Generate more speech-like audio with formants
            # Simulate vowel formants and consonant bursts
            audio = np.zeros(samples, dtype=np.float32)
            
            # Add speech-like segments
            segment_length = sample_rate // 4  # 250ms segments
            num_segments = samples // segment_length
            
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = min(start_idx + segment_length, samples)
                segment_t = t[start_idx:end_idx]
                
                # Simulate different phonemes with varying formants
                if i % 3 == 0:  # Vowel-like (formants at 800, 1200, 2400 Hz)
                    formant1 = 0.3 * np.sin(2 * np.pi * 800 * segment_t)
                    formant2 = 0.2 * np.sin(2 * np.pi * 1200 * segment_t)
                    formant3 = 0.1 * np.sin(2 * np.pi * 2400 * segment_t)
                    audio[start_idx:end_idx] = formant1 + formant2 + formant3
                elif i % 3 == 1:  # Consonant-like (higher frequency, shorter)
                    burst = 0.4 * np.sin(2 * np.pi * 3000 * segment_t) * np.exp(-10 * segment_t)
                    audio[start_idx:end_idx] = burst
                else:  # Mixed (transitional)
                    mixed = 0.2 * np.sin(2 * np.pi * 500 * segment_t) + 0.15 * np.sin(2 * np.pi * 1500 * segment_t)
                    audio[start_idx:end_idx] = mixed
                
                # Add some pause between words
                if i % 8 == 7:  # Pause every ~2 seconds
                    pause_length = min(segment_length // 2, samples - end_idx)
                    audio[end_idx:end_idx + pause_length] = 0.05 * np.random.randn(pause_length).astype(np.float32)
            
            # Add realistic background noise
            noise = 0.02 * np.random.randn(samples).astype(np.float32)
            audio = audio + noise
            
            # Apply speech envelope (amplitude variation)
            envelope = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
            audio = audio * envelope
            
        elif test_type == "music":
            # Generate music-like audio
            audio = (0.5 * np.sin(2 * np.pi * 440 * t) +  # A4
                    0.3 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
                    0.2 * np.sin(2 * np.pi * 659.25 * t))  # E5
        else:
            # Simple sine wave for testing
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Ensure proper dtype and range for Whisper
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        
        return audio
    
    def benchmark_openai_whisper(self) -> Dict[str, Any]:
        """Benchmark OpenAI Whisper implementation."""
        try:
            import whisper
            
            print("  🔥 Loading OpenAI Whisper model...")
            model = whisper.load_model("tiny")
            
            print("  📊 Processing test audio...")
            start_time = time.time()
            
            # Use whisper's transcribe method with our test audio
            result_whisper = model.transcribe(self.test_audio)
            
            inference_time = time.time() - start_time
            print(f"  ⚡ OpenAI Whisper completed: {inference_time*1000:.1f} ms")
            
            return {
                'status': 'success',
                'inference_time': inference_time,
                'model_size': 'tiny',
                'transcription': result_whisper.get('text', ''),
                'language': result_whisper.get('language', 'unknown'),
                'implementation': 'openai-whisper'
            }
            
        except ImportError:
            return {
                'status': 'not_available',
                'message': 'OpenAI Whisper not installed',
                'implementation': 'openai-whisper'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'implementation': 'openai-whisper'
            }
    
    def benchmark_faster_whisper(self) -> Dict[str, Any]:
        """Benchmark Faster-Whisper implementation."""
        try:
            from faster_whisper import WhisperModel
            
            print("  🚀 Loading Faster-Whisper model...")
            # Use tiny model for fast comparison
            model = WhisperModel("tiny", device="cpu", compute_type="float32")
            
            # Convert our test audio to the right format
            print("  📊 Processing test audio...")
            start_time = time.time()
            
            # Faster-Whisper expects audio as numpy array
            segments, info = model.transcribe(self.test_audio, beam_size=1)
            
            # Force execution by consuming segments
            transcription = list(segments)
            
            inference_time = time.time() - start_time
            print(f"  ⚡ Faster-Whisper completed: {inference_time*1000:.1f} ms")
            
            return {
                'status': 'success',
                'inference_time': inference_time,
                'model_size': 'tiny',
                'transcription_length': len(transcription),
                'language': info.language if hasattr(info, 'language') else 'unknown',
                'implementation': 'faster-whisper'
            }
            
        except ImportError:
            return {
                'status': 'not_available',
                'message': 'Faster-Whisper not installed',
                'implementation': 'faster-whisper'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'implementation': 'faster-whisper'
            }
    
    def benchmark_max_whisper(self) -> Dict[str, Any]:
        """Benchmark our MAX-Whisper implementation (development status)."""
        try:
            from model.max_whisper import MaxWhisperEncoder, MAX_AVAILABLE
            
            if not MAX_AVAILABLE:
                return {
                    'status': 'not_available',
                    'message': 'MAX Graph not available on this platform',
                    'implementation': 'max-whisper'
                }
            
            print("  🔥 Loading MAX-Whisper pipeline...")
            start_time = time.time()
            
            # Stage 1: Actual preprocessing
            features = preprocess_audio("dummy")
            preprocessing_time = time.time() - start_time
            
            # Stage 2: MAX Graph encoder (basic implementation)
            encoder_start = time.time()
            encoder = MaxWhisperEncoder()
            
            # Prepare input for MAX Graph
            batch_features = features[np.newaxis, :, :].astype(np.float32)
            target_length = 1500
            if batch_features.shape[2] < target_length:
                padding = target_length - batch_features.shape[2]
                batch_features = np.pad(batch_features, ((0,0), (0,0), (0,padding)), 'constant')
            else:
                batch_features = batch_features[:, :, :target_length]
            
            # Run actual MAX Graph inference (limited on macOS)
            try:
                encoded_features = encoder.encode(batch_features)
                encoder_time = time.time() - encoder_start
                total_time = time.time() - start_time
                
                print(f"  ⚡ MAX-Whisper completed: {total_time*1000:.1f} ms")
                print(f"    📊 Preprocessing: {preprocessing_time*1000:.1f} ms")
                print(f"    🔥 MAX Graph: {encoder_time*1000:.1f} ms")
                
                return {
                    'status': 'partial_implementation',
                    'inference_time': total_time,
                    'preprocessing_time': preprocessing_time,
                    'encoder_time': encoder_time,
                    'model_size': 'basic_encoder',
                    'features_shape': features.shape,
                    'encoded_shape': encoded_features.shape,
                    'implementation': 'max-whisper',
                    'note': 'Basic encoder only - full Whisper model needed for fair comparison'
                }
                
            except Exception as e:
                return {
                    'status': 'encoder_error',
                    'message': f'MAX Graph encoder failed: {str(e)}',
                    'preprocessing_time': preprocessing_time,
                    'implementation': 'max-whisper'
                }
            
        except ImportError:
            return {
                'status': 'not_available',
                'message': 'MAX-Whisper components not available',
                'implementation': 'max-whisper'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'implementation': 'max-whisper'
            }
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison with multiple test scenarios."""
        print("🎯 === COMPREHENSIVE WHISPER BENCHMARK SUITE ===")
        print("Testing multiple scenarios for robust performance evaluation")
        print("=" * 60)
        
        # Test scenarios
        test_scenarios = [
            ("speech_short", "🗣️  Short Speech (5s)", 5.0, "speech"),
            ("speech_standard", "🎤 Standard Speech (30s)", 30.0, "speech"), 
            ("speech_long", "📻 Long Speech (60s)", 60.0, "speech"),
            ("music", "🎵 Music Test (15s)", 15.0, "music")
        ]
        
        all_results = {}
        
        for scenario_id, description, duration, audio_type in test_scenarios:
            print(f"\n{description}")
            print("─" * 50)
            
            # Use real YouTube audio if available, otherwise generate synthetic
            if hasattr(self, 'test_audio') and len(self.test_audio) > duration * 16000:
                # Extract segment from real audio
                start_sample = 0  # Could randomize this
                end_sample = int(duration * 16000)
                test_audio = self.test_audio[start_sample:end_sample]
                print(f"🎬 Using real audio segment ({duration}s from YouTube)")
            else:
                # Fall back to synthetic audio
                test_audio = self._generate_test_audio(duration, audio_type)
                print(f"🔊 Using synthetic {audio_type} audio ({duration}s)")
            
            original_audio = self.test_audio  # Save original
            self.test_audio = test_audio  # Temporarily replace
            
            # Run benchmarks - include MAX if working on macOS
            implementations = {
                'OpenAI Whisper': self.benchmark_openai_whisper,
                'Faster-Whisper': self.benchmark_faster_whisper,
            }
            
            # Add MAX-Whisper only for short test (it's slow on macOS CPU)
            if scenario_id == "speech_short":
                try:
                    from model.max_whisper import MAX_AVAILABLE
                    if MAX_AVAILABLE:
                        implementations['MAX-Whisper'] = self.benchmark_max_whisper
                        print(f"🔥 MAX Graph available - testing on short scenario only")
                except ImportError:
                    pass
            
            scenario_results = {}
            
            for name, benchmark_func in implementations.items():
                print(f"\n🔄 Benchmarking {name}...")
                result = benchmark_func()
                scenario_results[name] = result
                
                if result['status'] == 'success':
                    print(f"  ⚡ Time: {result['inference_time']*1000:.1f} ms")
                    rtf = result['inference_time'] / duration
                    print(f"  📊 RTF: {rtf:.4f}")
                    
                    # Show transcription
                    transcription = result.get('transcription', '')
                    if transcription.strip():
                        preview = transcription[:100] + "..." if len(transcription) > 100 else transcription
                        print(f"  📝 Output: '{preview}'")
                    else:
                        print(f"  📝 Output: (no transcription detected)")
                        
                    # Performance assessment
                    if rtf < 0.05:
                        print(f"  🏆 Performance: EXCELLENT (>20x real-time)")
                    elif rtf < 0.2:
                        print(f"  🚀 Performance: GOOD (>5x real-time)")
                    else:
                        print(f"  ⚠️  Performance: SLOW (<5x real-time)")
                else:
                    error_msg = result.get('message', f"Status: {result.get('status', 'unknown error')}")
                    print(f"  ❌ Failed: {error_msg}")
            
            all_results[scenario_id] = {
                'description': description,
                'duration': duration,
                'audio_type': audio_type,
                'results': scenario_results
            }
            
            # Restore original test audio
            self.test_audio = original_audio
        
        # Print comprehensive summary
        self._print_comprehensive_summary(all_results)
        
        return all_results
    
    def _print_comprehensive_summary(self, all_results: Dict):
        """Print comprehensive benchmark summary."""
        print("\n" + "🏆" * 60)
        print("📊 COMPREHENSIVE BENCHMARK SUMMARY")
        print("🏆" * 60)
        
        # Collect all timing data
        performance_data = {}
        
        for scenario_id, scenario_data in all_results.items():
            description = scenario_data['description']
            duration = scenario_data['duration']
            
            print(f"\n{description}")
            print("─" * 40)
            
            for impl_name, result in scenario_data['results'].items():
                if result['status'] == 'success':
                    time_ms = result['inference_time'] * 1000
                    rtf = result['inference_time'] / duration
                    
                    # Store for overall analysis
                    if impl_name not in performance_data:
                        performance_data[impl_name] = []
                    performance_data[impl_name].append({
                        'scenario': scenario_id,
                        'time_ms': time_ms,
                        'rtf': rtf,
                        'duration': duration
                    })
                    
                    print(f"  {impl_name}: {time_ms:.1f} ms (RTF: {rtf:.4f})")
        
        # Overall performance comparison
        print(f"\n📈 OVERALL PERFORMANCE ANALYSIS")
        print("=" * 40)
        
        for impl_name, data_points in performance_data.items():
            avg_rtf = sum(dp['rtf'] for dp in data_points) / len(data_points)
            avg_time = sum(dp['time_ms'] for dp in data_points) / len(data_points)
            
            print(f"\n🔹 {impl_name}:")
            print(f"   Average RTF: {avg_rtf:.4f}")
            print(f"   Average Time: {avg_time:.1f} ms")
            print(f"   Scenarios tested: {len(data_points)}")
            
            # Performance grade
            if avg_rtf < 0.01:
                grade = "🏆 OUTSTANDING"
            elif avg_rtf < 0.05:
                grade = "🚀 EXCELLENT"
            elif avg_rtf < 0.2:
                grade = "⚡ GOOD"
            else:
                grade = "⚠️ NEEDS OPTIMIZATION"
            
            print(f"   Overall Grade: {grade}")
        
        # Compare implementations
        if len(performance_data) > 1:
            impl_names = list(performance_data.keys())
            impl1, impl2 = impl_names[0], impl_names[1]
            
            avg_rtf1 = sum(dp['rtf'] for dp in performance_data[impl1]) / len(performance_data[impl1])
            avg_rtf2 = sum(dp['rtf'] for dp in performance_data[impl2]) / len(performance_data[impl2])
            
            if avg_rtf1 < avg_rtf2:
                speedup = avg_rtf2 / avg_rtf1
                print(f"\n🚀 {impl1} is {speedup:.1f}x faster than {impl2}")
            else:
                speedup = avg_rtf1 / avg_rtf2  
                print(f"\n🚀 {impl2} is {speedup:.1f}x faster than {impl1}")
        
        print(f"\n✅ Benchmark suite complete - ready for hackathon demo!")
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run standard comparison (legacy method)."""
        return self.run_comprehensive_comparison()
    
    def _print_comparison(self, results: Dict[str, Any]):
        """Print detailed comparison results."""
        print("=== Performance Comparison ===")
        
        # Extract inference times for successful benchmarks
        times = {}
        notes = {}
        for name, result in results.items():
            if result['status'] in ['implemented', 'success', 'partial_implementation']:
                times[name] = result['inference_time']
                if result['status'] == 'partial_implementation':
                    notes[name] = "(partial implementation - not fair comparison)"
        
        if not times:
            print("No successful benchmarks to compare")
            return
        
        # Sort by speed (fastest first)
        sorted_times = sorted(times.items(), key=lambda x: x[1])
        
        print("🏆 Ranking (fastest to slowest):")
        for i, (name, time_val) in enumerate(sorted_times, 1):
            note = notes.get(name, "")
            print(f"  {i}. {name}: {time_val*1000:.1f} ms {note}")
        
        # Calculate speedups relative to slowest
        if len(sorted_times) > 1:
            slowest_time = sorted_times[-1][1]
            print(f"\n⚡ Speedups relative to {sorted_times[-1][0]}:")
            
            for name, time_val in sorted_times[:-1]:
                speedup = slowest_time / time_val
                print(f"  🚀 {name}: {speedup:.1f}x faster")
        
        # Calculate RTF (Real-Time Factor)
        audio_duration = len(self.test_audio) / 16000
        print(f"\n📊 Real-Time Factor (RTF = processing_time / audio_duration):")
        print(f"Audio duration: {audio_duration:.1f}s")
        
        for name, time_val in times.items():
            rtf = time_val / audio_duration
            status = "🏆 EXCELLENT" if rtf < 0.05 else "🚀 GOOD" if rtf < 0.2 else "⚠️ SLOW"
            note = notes.get(name, "")
            print(f"  {name}: RTF = {rtf:.4f} {status} {note}")
        
        # Add honest assessment
        if any("partial implementation" in note for note in notes.values()):
            print(f"\n⚠️  IMPORTANT: Partial implementations shown are not fair comparisons.")
            print(f"   Real speedup claims require complete Whisper model implementations.")
    
    def calculate_target_metrics(self) -> Dict[str, float]:
        """Calculate our target performance metrics."""
        audio_duration = len(self.test_audio) / 16000
        
        # Target metrics from our competitive analysis
        target_rtf = 0.05  # 20x faster than real-time
        target_speedup = 3.0  # 3x faster than baseline
        
        return {
            'audio_duration': audio_duration,
            'target_rtf': target_rtf,
            'target_inference_time': audio_duration * target_rtf,
            'target_speedup': target_speedup
        }


if __name__ == "__main__":
    # Use Modular's YouTube video for real audio
    youtube_url = "https://www.youtube.com/watch?v=DCAMCzRXGQ4"
    
    # First install yt-dlp if running in pixi benchmark environment
    try:
        import yt_dlp
        use_real_audio = True
        print("🎬 yt-dlp available - using real audio from Modular's channel!")
    except ImportError:
        use_real_audio = False
        print("⚠️  yt-dlp not available - using synthetic audio")
    
    benchmark = WhisperBenchmark(use_real_audio=use_real_audio, youtube_url=youtube_url)
    
    # Run the comparison
    results = benchmark.run_comparison()
    
    # Show targets
    targets = benchmark.calculate_target_metrics()
    
    print("\n=== Hackathon Success Targets ===")
    print(f"🎯 Target RTF: {targets['target_rtf']}")
    print(f"🎯 Target inference time: {targets['target_inference_time']*1000:.1f} ms")
    print(f"🎯 Target speedup: {targets['target_speedup']}x")
    
    print(f"\n=== Next Implementation Steps ===")
    print(f"1. 🔧 Implement Mojo mel-spectrogram GPU kernel")
    print(f"2. ⚡ Create MAX Graph Whisper encoder")
    print(f"3. 📊 Optimize memory layout and batching")
    print(f"4. 🎪 Build side-by-side demo interface")
    
    # Save results for later analysis
    import json
    
    # Convert comprehensive results to JSON format
    serializable_results = {}
    for scenario_id, scenario_data in results.items():
        serializable_results[scenario_id] = {
            'description': scenario_data['description'],
            'duration': scenario_data['duration'],
            'audio_type': scenario_data['audio_type'],
            'results': {}
        }
        
        for impl_name, result in scenario_data['results'].items():
            serializable_results[scenario_id]['results'][impl_name] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in result.items()
            }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump({
            'comprehensive_results': serializable_results,
            'targets': targets,
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\n💾 Results saved to benchmark_results.json")