#!/usr/bin/env python3
"""
Feature Distribution Analysis for MAX Graph Whisper
Analyze the statistical differences between MAX Graph and OpenAI encoder features
that cause decoder repetition issues
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

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


class FeatureDistributionAnalyzer:
    """Analyze feature distribution differences causing decoder issues"""
    
    def __init__(self, audio_file: str = "audio_samples/modular_video.wav"):
        self.audio_file = audio_file
        self.max_whisper = None
        self.openai_model = None
        
        # Initialize models
        if MAX_WHISPER_AVAILABLE and MAX_AVAILABLE and WHISPER_AVAILABLE:
            print("🚀 Initializing models for feature analysis...")
            self.max_whisper = WhisperMAX(model_size="tiny", use_gpu=True)
            if not self.max_whisper.available:
                print("❌ MAX Graph Whisper not available")
                self.max_whisper = None
            
            # Load OpenAI model for comparison
            self.openai_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            print("❌ Required libraries not available")
    
    def analyze_feature_distributions(self) -> Dict[str, Any]:
        """Comprehensive analysis of feature distribution differences"""
        
        if not self.max_whisper or not self.openai_model:
            print("❌ Models not available for analysis")
            return {}
        
        print("🔬 Starting comprehensive feature distribution analysis...")
        
        # Load and preprocess audio
        import librosa
        audio, sr = librosa.load(self.audio_file, sr=16000)
        mel_db = whisper.log_mel_spectrogram(audio).numpy()
        
        # Get encoder features from both models
        print("🔢 Extracting MAX Graph encoder features...")
        max_features = self.max_whisper._encode_with_max_graph(mel_db)
        
        print("🔢 Extracting OpenAI encoder features...")
        openai_features = self._get_openai_features(mel_db)
        
        if max_features is None or openai_features is None:
            print("❌ Failed to extract features")
            return {}
        
        print(f"✅ Feature shapes: MAX={max_features.shape}, OpenAI={openai_features.shape}")
        
        # Comprehensive statistical analysis
        analysis = {
            "basic_stats": self._analyze_basic_statistics(max_features, openai_features),
            "distribution_analysis": self._analyze_distributions(max_features, openai_features),
            "correlation_analysis": self._analyze_correlations(max_features, openai_features),
            "attention_analysis": self._analyze_attention_patterns(max_features, openai_features),
            "token_probability_analysis": self._analyze_token_probabilities(max_features, openai_features)
        }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _get_openai_features(self, mel_features: np.ndarray) -> Optional[np.ndarray]:
        """Get OpenAI encoder features"""  
        try:
            # Pad/truncate to match MAX Graph processing
            n_mels, seq_len = mel_features.shape
            max_seq_len = 3000
            
            if seq_len > max_seq_len:
                mel_features = mel_features[:, :max_seq_len]
            else:
                pad_width = max_seq_len - seq_len
                mel_features = np.pad(mel_features, ((0, 0), (0, pad_width)), mode='constant')
            
            # Convert to tensor and process
            mel_tensor = torch.from_numpy(mel_features).float().unsqueeze(0)
            device = next(self.openai_model.encoder.parameters()).device
            mel_tensor = mel_tensor.to(device)
            
            with torch.no_grad():
                encoder_features = self.openai_model.encoder(mel_tensor)
                return encoder_features.cpu().numpy()
                
        except Exception as e:
            print(f"❌ Failed to get OpenAI features: {e}")
            return None
    
    def _analyze_basic_statistics(self, max_features: np.ndarray, openai_features: np.ndarray) -> Dict[str, Any]:\n        """Analyze basic statistical differences"""\n        print("  📊 Analyzing basic statistics...")\n        \n        max_flat = max_features.flatten()\n        openai_flat = openai_features.flatten()\n        \n        return {\n            "max_graph": {\n                "mean": float(np.mean(max_flat)),\n                "std": float(np.std(max_flat)),\n                "min": float(np.min(max_flat)),\n                "max": float(np.max(max_flat)),\n                "median": float(np.median(max_flat)),\n                "skewness": float(self._calculate_skewness(max_flat)),\n                "kurtosis": float(self._calculate_kurtosis(max_flat))\n            },\n            "openai": {\n                "mean": float(np.mean(openai_flat)),\n                "std": float(np.std(openai_flat)),\n                "min": float(np.min(openai_flat)),\n                "max": float(np.max(openai_flat)),\n                "median": float(np.median(openai_flat)),\n                "skewness": float(self._calculate_skewness(openai_flat)),\n                "kurtosis": float(self._calculate_kurtosis(openai_flat))\n            },\n            "differences": {\n                "mean_diff": float(np.mean(max_flat) - np.mean(openai_flat)),\n                "std_ratio": float(np.std(max_flat) / np.std(openai_flat)),\n                "range_ratio": float((np.max(max_flat) - np.min(max_flat)) / (np.max(openai_flat) - np.min(openai_flat))),\n                "cosine_similarity": float(np.dot(max_flat, openai_flat) / (np.linalg.norm(max_flat) * np.linalg.norm(openai_flat)))\n            }\n        }\n    \n    def _analyze_distributions(self, max_features: np.ndarray, openai_features: np.ndarray) -> Dict[str, Any]:\n        """Analyze distribution shapes and differences"""\n        print("  📈 Analyzing feature distributions...")\n        \n        max_flat = max_features.flatten()\n        openai_flat = openai_features.flatten()\n        \n        # Sample for distribution analysis (too many points otherwise)\n        sample_size = min(10000, len(max_flat))\n        max_sample = np.random.choice(max_flat, sample_size, replace=False)\n        openai_sample = np.random.choice(openai_flat, sample_size, replace=False)\n        \n        return {\n            "wasserstein_distance": float(wasserstein_distance(max_sample, openai_sample)),\n            "histogram_analysis": self._compare_histograms(max_sample, openai_sample),\n            "outlier_analysis": {\n                "max_outliers": self._count_outliers(max_flat),\n                "openai_outliers": self._count_outliers(openai_flat)\n            }\n        }\n    \n    def _analyze_correlations(self, max_features: np.ndarray, openai_features: np.ndarray) -> Dict[str, Any]:\n        """Analyze feature correlations and relationships"""\n        print("  🔗 Analyzing feature correlations...")\n        \n        # Reshape for correlation analysis\n        batch, seq_len, d_model = max_features.shape\n        max_reshaped = max_features.reshape(-1, d_model)\n        openai_reshaped = openai_features.reshape(-1, d_model)\n        \n        # Sample for efficiency\n        sample_size = min(1000, max_reshaped.shape[0])\n        indices = np.random.choice(max_reshaped.shape[0], sample_size, replace=False)\n        max_sample = max_reshaped[indices]\n        openai_sample = openai_reshaped[indices]\n        \n        return {\n            "feature_correlations": self._calculate_feature_correlations(max_sample, openai_sample),\n            "sequence_correlations": self._calculate_sequence_correlations(max_features, openai_features),\n            "pca_analysis": self._analyze_pca_differences(max_sample, openai_sample)\n        }\n    \n    def _analyze_attention_patterns(self, max_features: np.ndarray, openai_features: np.ndarray) -> Dict[str, Any]:\n        """Analyze attention pattern differences that affect decoder behavior"""\n        print("  👁️ Analyzing attention patterns...")\n        \n        # Analyze feature magnitudes across sequence positions\n        max_magnitudes = np.linalg.norm(max_features, axis=-1).squeeze()\n        openai_magnitudes = np.linalg.norm(openai_features, axis=-1).squeeze()\n        \n        return {\n            "magnitude_patterns": {\n                "max_variance": float(np.var(max_magnitudes)),\n                "openai_variance": float(np.var(openai_magnitudes)),\n                "correlation": float(np.corrcoef(max_magnitudes, openai_magnitudes)[0, 1])\n            },\n            "sequence_energy": {\n                "max_energy_distribution": max_magnitudes.tolist()[:100],  # First 100 for analysis\n                "openai_energy_distribution": openai_magnitudes.tolist()[:100]\n            }\n        }\n    \n    def _analyze_token_probabilities(self, max_features: np.ndarray, openai_features: np.ndarray) -> Dict[str, Any]:\n        """Analyze how feature differences affect token probabilities"""\n        print("  🎯 Analyzing token probability distributions...")\n        \n        try:\n            # Get token probabilities for first few positions\n            max_probs = self._get_token_probabilities(max_features, num_tokens=50)\n            openai_probs = self._get_token_probabilities(openai_features, num_tokens=50)\n            \n            return {\n                "probability_divergence": self._calculate_kl_divergence(max_probs, openai_probs),\n                "top_token_differences": self._analyze_top_token_differences(max_probs, openai_probs),\n                "confidence_analysis": {\n                    "max_confidence": float(np.mean(np.max(max_probs, axis=-1))),\n                    "openai_confidence": float(np.mean(np.max(openai_probs, axis=-1)))\n                }\n            }\n        except Exception as e:\n            print(f"    ⚠️ Token probability analysis failed: {e}")\n            return {"error": str(e)}\n    \n    def _get_token_probabilities(self, features: np.ndarray, num_tokens: int = 50) -> np.ndarray:\n        """Get token probabilities from decoder for given features"""\n        try:\n            # Convert to tensor\n            features_tensor = torch.from_numpy(features.copy()).float()\n            device = next(self.openai_model.parameters()).device\n            features_tensor = features_tensor.to(device)\n            \n            # Get decoder logits for first few positions\n            with torch.no_grad():\n                # Start with BOS token\n                tokens = torch.tensor([[50258]], device=device)  # Whisper BOS token\n                \n                probs_list = []\n                for i in range(min(num_tokens, 20)):  # Limit to prevent infinite loops\n                    # Get next token logits\n                    logits = self.openai_model.decoder(tokens, features_tensor)\n                    probs = torch.softmax(logits[0, -1], dim=-1)\n                    probs_list.append(probs.cpu().numpy())\n                    \n                    # Get next token (greedy)\n                    next_token = torch.argmax(probs).unsqueeze(0).unsqueeze(0)\n                    tokens = torch.cat([tokens, next_token], dim=1)\n                    \n                    # Stop at EOS or if we get repetitive\n                    if next_token.item() == 50257:  # Whisper EOS token\n                        break\n                \n                return np.array(probs_list)\n                \n        except Exception as e:\n            print(f"      ❌ Failed to get token probabilities: {e}")\n            # Return dummy probabilities\n            vocab_size = 51865  # Whisper vocab size\n            return np.random.dirichlet(np.ones(vocab_size), size=(10,))\n    \n    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:\n        """Generate recommendations based on analysis"""\n        print("  💡 Generating recommendations...")\n        \n        recommendations = []\n        \n        # Check standard deviation difference\n        basic_stats = analysis.get("basic_stats", {})\n        differences = basic_stats.get("differences", {})\n        \n        std_ratio = differences.get("std_ratio", 1.0)\n        if std_ratio > 2.0:\n            recommendations.append(\n                f"CRITICAL: MAX Graph std deviation is {std_ratio:.2f}x higher than OpenAI. "\n                "Apply feature scaling: features = features * (openai_std / max_std)"\n            )\n        elif std_ratio > 1.5:\n            recommendations.append(\n                f"WARNING: MAX Graph std deviation is {std_ratio:.2f}x higher. "\n                "Consider feature normalization."\n            )\n        \n        # Check mean difference\n        mean_diff = abs(differences.get("mean_diff", 0.0))\n        if mean_diff > 0.1:\n            recommendations.append(\n                f"Mean difference detected ({mean_diff:.4f}). "\n                "Consider centering features: features = features - mean_diff"\n            )\n        \n        # Check cosine similarity\n        cosine_sim = differences.get("cosine_similarity", 0.0)\n        if cosine_sim < 0.7:\n            recommendations.append(\n                f"Low cosine similarity ({cosine_sim:.4f}). "\n                "Features have significant structural differences. Consider feature alignment."\n            )\n        \n        # Check decoder parameters\n        token_analysis = analysis.get("token_probability_analysis", {})\n        if "confidence_analysis" in token_analysis:\n            max_conf = token_analysis["confidence_analysis"].get("max_confidence", 0.0)\n            openai_conf = token_analysis["confidence_analysis"].get("openai_confidence", 0.0)\n            \n            if max_conf < openai_conf * 0.8:\n                recommendations.append(\n                    "Decoder confidence is lower for MAX Graph features. "\n                    "Increase patience parameter and consider temperature adjustment."\n                )\n        \n        if not recommendations:\n            recommendations.append("Features look reasonably aligned. Issues may be in decoder parameters.")\n        \n        return recommendations\n    \n    # Helper methods\n    def _calculate_skewness(self, data: np.ndarray) -> float:\n        """Calculate skewness of distribution"""\n        from scipy.stats import skew\n        return skew(data)\n    \n    def _calculate_kurtosis(self, data: np.ndarray) -> float:\n        """Calculate kurtosis of distribution"""\n        from scipy.stats import kurtosis\n        return kurtosis(data)\n    \n    def _count_outliers(self, data: np.ndarray, threshold: float = 3.0) -> int:\n        """Count outliers using z-score threshold"""\n        z_scores = np.abs((data - np.mean(data)) / np.std(data))\n        return int(np.sum(z_scores > threshold))\n    \n    def _compare_histograms(self, data1: np.ndarray, data2: np.ndarray, bins: int = 50) -> Dict[str, Any]:\n        """Compare histogram shapes"""\n        hist1, edges = np.histogram(data1, bins=bins, density=True)\n        hist2, _ = np.histogram(data2, bins=edges, density=True)\n        \n        # Calculate histogram difference\n        hist_diff = np.sum(np.abs(hist1 - hist2))\n        \n        return {\n            "histogram_difference": float(hist_diff),\n            "bins": bins\n        }\n    \n    def _calculate_feature_correlations(self, features1: np.ndarray, features2: np.ndarray) -> Dict[str, float]:\n        """Calculate correlations between feature dimensions"""\n        correlations = []\n        for i in range(min(features1.shape[1], 20)):  # Analyze first 20 dimensions\n            corr = np.corrcoef(features1[:, i], features2[:, i])[0, 1]\n            if not np.isnan(corr):\n                correlations.append(corr)\n        \n        return {\n            "mean_correlation": float(np.mean(correlations)),\n            "min_correlation": float(np.min(correlations)),\n            "max_correlation": float(np.max(correlations))\n        }\n    \n    def _calculate_sequence_correlations(self, features1: np.ndarray, features2: np.ndarray) -> Dict[str, float]:\n        """Calculate correlations across sequence positions"""\n        seq_corrs = []\n        for i in range(min(features1.shape[1], 100)):  # First 100 sequence positions\n            f1_seq = features1[0, i, :].flatten()\n            f2_seq = features2[0, i, :].flatten()\n            corr = np.corrcoef(f1_seq, f2_seq)[0, 1]\n            if not np.isnan(corr):\n                seq_corrs.append(corr)\n        \n        return {\n            "mean_sequence_correlation": float(np.mean(seq_corrs)),\n            "sequence_correlation_variance": float(np.var(seq_corrs))\n        }\n    \n    def _analyze_pca_differences(self, features1: np.ndarray, features2: np.ndarray) -> Dict[str, Any]:\n        """Analyze differences in principal components"""\n        try:\n            # Fit PCA on both feature sets\n            pca1 = PCA(n_components=10)\n            pca2 = PCA(n_components=10)\n            \n            pca1.fit(features1)\n            pca2.fit(features2)\n            \n            return {\n                "explained_variance_ratio_diff": {\n                    "max_graph": pca1.explained_variance_ratio_.tolist(),\n                    "openai": pca2.explained_variance_ratio_.tolist()\n                },\n                "total_variance_captured": {\n                    "max_graph": float(np.sum(pca1.explained_variance_ratio_)),\n                    "openai": float(np.sum(pca2.explained_variance_ratio_))\n                }\n            }\n        except Exception as e:\n            return {"error": str(e)}\n    \n    def _calculate_kl_divergence(self, probs1: np.ndarray, probs2: np.ndarray) -> float:\n        """Calculate KL divergence between probability distributions"""\n        try:\n            # Add small epsilon to avoid log(0)\n            eps = 1e-10\n            probs1_safe = np.clip(probs1, eps, 1.0)\n            probs2_safe = np.clip(probs2, eps, 1.0)\n            \n            # Calculate mean KL divergence across all positions\n            kl_divs = []\n            for i in range(min(len(probs1_safe), len(probs2_safe))):\n                kl_div = np.sum(probs1_safe[i] * np.log(probs1_safe[i] / probs2_safe[i]))\n                if not np.isnan(kl_div) and not np.isinf(kl_div):\n                    kl_divs.append(kl_div)\n            \n            return float(np.mean(kl_divs)) if kl_divs else 0.0\n        except Exception as e:\n            print(f"    ⚠️ KL divergence calculation failed: {e}")\n            return 0.0\n    \n    def _analyze_top_token_differences(self, probs1: np.ndarray, probs2: np.ndarray) -> Dict[str, Any]:\n        """Analyze differences in top token predictions"""\n        try:\n            # Get top tokens for each position\n            top_k = 5\n            differences = []\n            \n            for i in range(min(len(probs1), len(probs2))):\n                top1 = np.argsort(probs1[i])[-top_k:]\n                top2 = np.argsort(probs2[i])[-top_k:]\n                \n                # Calculate overlap\n                overlap = len(set(top1) & set(top2)) / top_k\n                differences.append(1.0 - overlap)  # Convert to difference\n            \n            return {\n                "mean_top_token_difference": float(np.mean(differences)),\n                "max_top_token_difference": float(np.max(differences))\n            }\n        except Exception as e:\n            return {"error": str(e)}\n    \n    def save_analysis(self, analysis: Dict[str, Any], filename: str = "feature_distribution_analysis.json"):\n        """Save analysis results to file"""\n        with open(filename, 'w') as f:\n            json.dump(analysis, f, indent=2)\n        print(f"✅ Analysis saved to {filename}")\n    \n    def print_summary(self, analysis: Dict[str, Any]):\n        """Print analysis summary"""\n        print("\\n" + "="*60)\n        print("🔬 FEATURE DISTRIBUTION ANALYSIS SUMMARY")\n        print("="*60)\n        \n        # Basic statistics\n        basic_stats = analysis.get("basic_stats", {})\n        if basic_stats:\n            diffs = basic_stats.get("differences", {})\n            print(f"\\n📊 BASIC STATISTICS:")\n            print(f"  Mean difference: {diffs.get('mean_diff', 0):.6f}")\n            print(f"  Std deviation ratio: {diffs.get('std_ratio', 1):.3f}")\n            print(f"  Cosine similarity: {diffs.get('cosine_similarity', 0):.6f}")\n        \n        # Distribution analysis\n        dist_analysis = analysis.get("distribution_analysis", {})\n        if dist_analysis:\n            print(f"\\n📈 DISTRIBUTION ANALYSIS:")\n            print(f"  Wasserstein distance: {dist_analysis.get('wasserstein_distance', 0):.6f}")\n        \n        # Recommendations\n        recommendations = analysis.get("recommendations", [])\n        if recommendations:\n            print(f"\\n💡 RECOMMENDATIONS:")\n            for i, rec in enumerate(recommendations, 1):\n                print(f"  {i}. {rec}")\n        \n        print("\\n" + "="*60)\n\n\ndef main():\n    """Main function to run feature distribution analysis"""\n    print("🔬 MAX Graph Whisper Feature Distribution Analysis")\n    print("=" * 60)\n    \n    # Check if audio file exists\n    audio_file = "audio_samples/modular_video.wav"\n    if not os.path.exists(audio_file):\n        print(f"❌ Audio file not found: {audio_file}")\n        return\n    \n    # Initialize analyzer\n    analyzer = FeatureDistributionAnalyzer(audio_file)\n    \n    if not analyzer.max_whisper or not analyzer.openai_model:\n        print("❌ Cannot run analysis - models not available")\n        return\n    \n    # Run analysis\n    analysis = analyzer.analyze_feature_distributions()\n    \n    if analysis:\n        # Print summary\n        analyzer.print_summary(analysis)\n        \n        # Save results\n        analyzer.save_analysis(analysis)\n        \n        print("\\n✅ Feature distribution analysis complete!")\n    else:\n        print("❌ Analysis failed")\n\n\nif __name__ == "__main__":\n    main()