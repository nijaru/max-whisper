#!/usr/bin/env python3
"""
Decoder Parameter Tuning for MAX Graph Whisper
Systematically test different decoding parameters to fix early stopping at 218 characters
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List
import json
import os

# MAX Graph imports
try:
    from max import engine
    from max.driver import CPU, Accelerator, accelerator_count
    from max.graph import DeviceRef
    MAX_AVAILABLE = True
except ImportError:
    MAX_AVAILABLE = False

# Whisper imports
try:
    import whisper
    from whisper.decoding import DecodingOptions
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Add parent directory to path to import our MAX Graph implementation
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from max_whisper.whisper_max import WhisperMAX
    MAX_WHISPER_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'max-whisper'))
        from whisper_max import WhisperMAX
        MAX_WHISPER_AVAILABLE = True
    except ImportError:
        MAX_WHISPER_AVAILABLE = False


class DecoderParameterTuner:
    """Systematic decoder parameter optimization for MAX Graph Whisper"""
    
    def __init__(self, audio_file: str = "audio_samples/modular_video.wav"):
        self.audio_file = audio_file
        self.max_whisper = None
        self.baseline_result = None
        self.results = []
        
        # Initialize MAX Graph Whisper
        if MAX_WHISPER_AVAILABLE and MAX_AVAILABLE and WHISPER_AVAILABLE:
            print("🚀 Initializing MAX Graph Whisper for parameter tuning...")
            self.max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
            if not self.max_whisper.available:
                print("❌ MAX Graph Whisper not available")
                self.max_whisper = None
        else:
            print("❌ Required libraries not available")
    
    def get_baseline_result(self) -> str:
        """Get baseline CPU Whisper result for comparison"""
        if self.baseline_result is not None:
            return self.baseline_result
            
        print("📊 Getting baseline CPU Whisper result...")
        try:
            import librosa
            audio, sr = librosa.load(self.audio_file, sr=16000)
            
            # Load CPU model for comparison
            model = whisper.load_model("tiny", device="cpu")
            result = whisper.transcribe(model, audio, language="en")
            self.baseline_result = result['text'].strip()
            
            print(f"✅ Baseline result length: {len(self.baseline_result)} chars")
            print(f"📝 Baseline: '{self.baseline_result[:100]}...'")
            
            return self.baseline_result
            
        except Exception as e:
            print(f"❌ Failed to get baseline: {e}")
            return "Baseline unavailable"
    
    def test_decoder_parameters(self, 
                               beam_sizes: List[int] = [1, 3, 5, 10],
                               temperatures: List[float] = [0.0, 0.1, 0.3, 0.5],
                               patience_values: List[float] = [1.0, 2.0, 5.0, 10.0],
                               max_lengths: List[int] = [448, 1000, 2000]) -> List[Dict[str, Any]]:
        """Test different combinations of decoder parameters"""
        
        if not self.max_whisper:
            print("❌ MAX Graph Whisper not available for testing")
            return []
        
        print("🧪 Starting systematic decoder parameter testing...")
        print(f"📋 Testing: {len(beam_sizes)} beam sizes × {len(temperatures)} temperatures × {len(patience_values)} patience × {len(max_lengths)} max lengths")
        print(f"📋 Total combinations: {len(beam_sizes) * len(temperatures) * len(patience_values) * len(max_lengths)}")
        
        # Get MAX Graph encoder features once (to avoid recomputation)
        print("🔧 Pre-computing MAX Graph encoder features...")
        max_features = self._get_max_graph_features()
        if max_features is None:
            print("❌ Failed to get MAX Graph features")
            return []
        
        baseline = self.get_baseline_result()
        baseline_length = len(baseline)
        
        test_count = 0
        best_result = {"length": 0, "parameters": {}, "text": ""}
        
        for beam_size in beam_sizes:
            for temperature in temperatures:
                for patience in patience_values:
                    for max_length in max_lengths:
                        test_count += 1
                        print(f"\n🧪 Test {test_count}: beam={beam_size}, temp={temperature}, patience={patience}, max_len={max_length}")
                        
                        # Test these parameters
                        result = self._test_single_parameter_set(
                            max_features, beam_size, temperature, patience, max_length
                        )
                        
                        if result:
                            result_length = len(result['text'])
                            accuracy_pct = (result_length / baseline_length) * 100 if baseline_length > 0 else 0
                            
                            print(f"📊 Result length: {result_length} chars ({accuracy_pct:.1f}% of baseline)")
                            print(f"📝 Text: '{result['text'][:80]}...'")
                            
                            # Track best result
                            if result_length > best_result["length"]:
                                best_result = {
                                    "length": result_length,
                                    "parameters": result["parameters"],
                                    "text": result["text"]
                                }
                                print(f"🏆 New best result: {result_length} chars!")
                            
                            self.results.append(result)
        
        print(f"\n🎯 Testing complete! Best result: {best_result['length']} chars")
        print(f"🎯 Best parameters: {best_result['parameters']}")
        print(f"📝 Best text: '{best_result['text'][:100]}...'")
        
        return self.results
    
    def _get_max_graph_features(self) -> np.ndarray:
        """Get MAX Graph encoder features for testing"""
        try:
            import librosa
            
            # Load audio
            audio, sr = librosa.load(self.audio_file, sr=16000)
            
            # Get mel features using OpenAI method (matching our implementation)
            mel_db = whisper.log_mel_spectrogram(audio).numpy()
            
            # Process through MAX Graph encoder
            max_features = self.max_whisper._encode_with_max_graph(mel_db)
            
            print(f"✅ MAX Graph features shape: {max_features.shape}")
            print(f"📊 Features stats: mean={np.mean(max_features):.4f}, std={np.std(max_features):.4f}")
            
            return max_features
            
        except Exception as e:
            print(f"❌ Failed to get MAX Graph features: {e}")
            return None
    
    def _test_single_parameter_set(self, 
                                   max_features: np.ndarray,
                                   beam_size: int, 
                                   temperature: float, 
                                   patience: float, 
                                   max_length: int) -> Dict[str, Any]:
        """Test a single set of decoder parameters"""
        
        try:
            start_time = time.time()
            
            # Convert MAX Graph features to PyTorch tensor
            features_tensor = torch.from_numpy(max_features.copy()).float()
            device = next(self.max_whisper.whisper_model.parameters()).device
            features_tensor = features_tensor.to(device)
            
            # Create decoding options
            options = DecodingOptions(
                task="transcribe",
                language="en",
                temperature=temperature,
                beam_size=beam_size,
                patience=patience,
                sample_len=max_length,
                without_timestamps=True,
                suppress_blank=True,
                suppress_tokens="-1"
            )
            
            # Decode with these parameters
            result = self.max_whisper.whisper_model.decode(features_tensor, options)
            
            # Extract transcription text
            if isinstance(result, list) and len(result) > 0:
                transcription = result[0].text.strip()
            elif hasattr(result, 'text'):
                transcription = result.text.strip()
            else:
                transcription = "Decoder error"
            
            decode_time = time.time() - start_time
            
            return {
                "parameters": {
                    "beam_size": beam_size,
                    "temperature": temperature,
                    "patience": patience,
                    "max_length": max_length
                },
                "text": transcription,
                "length": len(transcription),
                "decode_time": decode_time
            }
            
        except Exception as e:
            print(f"      ❌ Parameter test failed: {e}")
            return None
    
    def save_results(self, filename: str = "decoder_tuning_results.json"):
        """Save results to JSON file"""
        if not self.results:
            print("❌ No results to save")
            return
        
        # Add baseline for comparison
        baseline = self.get_baseline_result()
        
        output = {
            "baseline": {
                "text": baseline,
                "length": len(baseline)
            },
            "test_results": self.results,
            "summary": {
                "total_tests": len(self.results),
                "best_length": max(r["length"] for r in self.results),
                "baseline_length": len(baseline)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Results saved to {filename}")
    
    def analyze_results(self):
        """Analyze results and provide recommendations"""
        if not self.results:
            print("❌ No results to analyze")
            return
        
        print("\n📊 DECODER PARAMETER ANALYSIS")
        print("=" * 50)
        
        baseline = self.get_baseline_result()
        baseline_length = len(baseline)
        
        # Sort by text length
        sorted_results = sorted(self.results, key=lambda x: x["length"], reverse=True)
        
        print(f"🎯 Baseline length: {baseline_length} characters")
        print(f"🎯 Best MAX Graph length: {sorted_results[0]['length']} characters ({(sorted_results[0]['length']/baseline_length)*100:.1f}% of baseline)")
        print(f"🎯 Current length: 218 characters ({(218/baseline_length)*100:.1f}% of baseline)")
        
        # Top 5 results
        print("\n🏆 TOP 5 RESULTS:")
        for i, result in enumerate(sorted_results[:5], 1):
            params = result["parameters"]
            accuracy = (result["length"] / baseline_length) * 100
            print(f"{i}. Length: {result['length']} ({accuracy:.1f}%) - beam={params['beam_size']}, temp={params['temperature']}, patience={params['patience']}, max_len={params['max_length']}")
        
        # Parameter impact analysis
        print("\n📈 PARAMETER IMPACT ANALYSIS:")
        
        # Beam size impact
        beam_impact = {}
        for result in self.results:
            beam = result["parameters"]["beam_size"]
            if beam not in beam_impact:
                beam_impact[beam] = []
            beam_impact[beam].append(result["length"])
        
        print("Beam size impact:")
        for beam, lengths in beam_impact.items():
            avg_length = np.mean(lengths)
            print(f"  beam_size={beam}: avg={avg_length:.1f} chars")
        
        # Temperature impact
        temp_impact = {}
        for result in self.results:
            temp = result["parameters"]["temperature"]
            if temp not in temp_impact:
                temp_impact[temp] = []
            temp_impact[temp].append(result["length"])
        
        print("Temperature impact:")
        for temp, lengths in temp_impact.items():
            avg_length = np.mean(lengths)
            print(f"  temperature={temp}: avg={avg_length:.1f} chars")
        
        # Patience impact
        patience_impact = {}
        for result in self.results:
            patience = result["parameters"]["patience"]
            if patience not in patience_impact:
                patience_impact[patience] = []
            patience_impact[patience].append(result["length"])
        
        print("Patience impact:")
        for patience, lengths in patience_impact.items():
            avg_length = np.mean(lengths)
            print(f"  patience={patience}: avg={avg_length:.1f} chars")
        
        # Best parameter recommendation
        best = sorted_results[0]
        print(f"\n🎯 RECOMMENDED PARAMETERS:")
        print(f"beam_size: {best['parameters']['beam_size']}")
        print(f"temperature: {best['parameters']['temperature']}")
        print(f"patience: {best['parameters']['patience']}")
        print(f"max_length: {best['parameters']['max_length']}")


def main():
    """Main function to run decoder parameter tuning"""
    print("🧪 MAX Graph Whisper Decoder Parameter Tuning")
    print("=" * 60)
    
    # Check if audio file exists
    audio_file = "audio_samples/modular_video.wav"
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    # Initialize tuner
    tuner = DecoderParameterTuner(audio_file)
    
    if not tuner.max_whisper:
        print("❌ Cannot run tuning - MAX Graph Whisper not available")
        return
    
    # Run tests with focused parameter ranges
    print("🎯 Testing focused parameter ranges for early stopping issue...")
    results = tuner.test_decoder_parameters(
        beam_sizes=[1, 5, 10],              # Test different beam search strategies
        temperatures=[0.0, 0.1, 0.3],       # Test deterministic vs slightly random
        patience_values=[1.0, 5.0, 10.0],   # Key parameter for early stopping
        max_lengths=[448, 1000, 2000]       # Test longer max lengths
    )
    
    # Analyze results
    tuner.analyze_results()
    
    # Save results
    tuner.save_results("decoder_tuning_results.json")
    
    print("\n✅ Decoder parameter tuning complete!")


if __name__ == "__main__":
    main()