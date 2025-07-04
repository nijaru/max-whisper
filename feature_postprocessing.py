#!/usr/bin/env python3
"""
Feature Post-Processing for MAX Graph Whisper
Transforms MAX Graph encoder features to better match OpenAI distribution
while preserving semantic patterns for improved decoder performance
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any

class FeaturePostProcessor:
    """
    Post-process MAX Graph encoder features to improve decoder compatibility
    """
    
    def __init__(self):
        # Target statistics from OpenAI encoder
        self.target_mean = 0.0002
        self.target_std = 0.4000
        
        # Source statistics from MAX Graph encoder  
        self.source_mean = 0.0310
        self.source_std = 1.4475
        
        # Precomputed scaling factors
        self.scale_factor = self.target_std / self.source_std  # ~0.276
        self.shift_factor = self.target_mean - (self.source_mean * self.scale_factor)
    
    def simple_statistical_matching(self, features: np.ndarray) -> np.ndarray:
        """
        Simple linear transformation to match target statistics
        
        Args:
            features: MAX Graph encoder features [batch, seq_len, d_model]
            
        Returns:
            Transformed features with target statistics
        """
        # Apply linear transformation: (features - source_mean) * scale + target_mean
        transformed = (features - self.source_mean) * self.scale_factor + self.target_mean
        
        return transformed.astype(np.float32)
    
    def adaptive_statistical_matching(self, features: np.ndarray) -> np.ndarray:
        """
        Adaptive transformation that preserves feature relationships
        
        Args:
            features: MAX Graph encoder features [batch, seq_len, d_model]
            
        Returns:
            Adaptively transformed features
        """
        batch_size, seq_len, d_model = features.shape
        
        # Compute current statistics
        current_mean = np.mean(features)
        current_std = np.std(features)
        
        # Progressive scaling based on current vs expected
        scale = self.target_std / current_std
        shift = self.target_mean - (current_mean * scale)
        
        # Apply transformation with some preservation of original range
        transformed = features * scale + shift
        
        return transformed.astype(np.float32)
    
    def robust_percentile_matching(self, features: np.ndarray) -> np.ndarray:
        """
        Use percentile-based transformation to handle outliers better
        
        Args:
            features: MAX Graph encoder features [batch, seq_len, d_model]
            
        Returns:
            Percentile-matched features
        """
        # Get percentiles for robust transformation
        p5, p95 = np.percentile(features, [5, 95])
        target_p5, target_p95 = np.percentile(np.random.normal(self.target_mean, self.target_std, features.size), [5, 95])
        
        # Robust scale and shift
        scale = (target_p95 - target_p5) / (p95 - p5) if (p95 - p5) > 1e-8 else 1.0
        shift = target_p5 - p5 * scale
        
        transformed = features * scale + shift
        
        return transformed.astype(np.float32)
    
    def semantic_preserving_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform that preserves semantic relationships between tokens
        
        Args:
            features: MAX Graph encoder features [batch, seq_len, d_model]
            
        Returns:
            Semantically-preserved transformed features
        """
        batch_size, seq_len, d_model = features.shape
        
        # Normalize per-token to preserve relative patterns
        token_means = np.mean(features, axis=2, keepdims=True)  # [batch, seq_len, 1]
        token_stds = np.std(features, axis=2, keepdims=True)    # [batch, seq_len, 1]
        
        # Avoid division by zero
        token_stds = np.maximum(token_stds, 1e-8)
        
        # Normalize each token
        normalized = (features - token_means) / token_stds
        
        # Apply global scaling to match target distribution
        global_scale = self.target_std * 0.8  # Slightly conservative
        global_shift = self.target_mean
        
        # Reconstruct with target statistics
        transformed = normalized * global_scale + global_shift
        
        return transformed.astype(np.float32)
    
    def hybrid_approach(self, features: np.ndarray, strategy_weights: Dict[str, float] = None) -> np.ndarray:
        """
        Combine multiple transformation strategies
        
        Args:
            features: MAX Graph encoder features
            strategy_weights: Weights for different strategies
            
        Returns:
            Hybrid transformed features
        """
        if strategy_weights is None:
            strategy_weights = {
                'simple': 0.3,
                'adaptive': 0.3, 
                'percentile': 0.2,
                'semantic': 0.2
            }
        
        # Apply all transformations
        simple = self.simple_statistical_matching(features)
        adaptive = self.adaptive_statistical_matching(features)
        percentile = self.robust_percentile_matching(features)
        semantic = self.semantic_preserving_transform(features)
        
        # Weighted combination
        transformed = (strategy_weights['simple'] * simple + 
                      strategy_weights['adaptive'] * adaptive +
                      strategy_weights['percentile'] * percentile +
                      strategy_weights['semantic'] * semantic)
        
        return transformed.astype(np.float32)
    
    def analyze_transformation(self, original: np.ndarray, transformed: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the effect of transformation
        
        Args:
            original: Original MAX Graph features
            transformed: Post-processed features
            
        Returns:
            Analysis results
        """
        analysis = {
            'original_stats': {
                'mean': float(np.mean(original)),
                'std': float(np.std(original)),
                'min': float(np.min(original)),
                'max': float(np.max(original))
            },
            'transformed_stats': {
                'mean': float(np.mean(transformed)),
                'std': float(np.std(transformed)), 
                'min': float(np.min(transformed)),
                'max': float(np.max(transformed))
            },
            'target_stats': {
                'mean': self.target_mean,
                'std': self.target_std
            }
        }
        
        # Compute difference metrics
        analysis['mean_error'] = abs(analysis['transformed_stats']['mean'] - self.target_mean)
        analysis['std_error'] = abs(analysis['transformed_stats']['std'] - self.target_std)
        analysis['total_error'] = analysis['mean_error'] + analysis['std_error']
        
        # Semantic preservation metrics (correlation)
        correlation = np.corrcoef(original.flatten(), transformed.flatten())[0, 1]
        analysis['correlation'] = float(correlation)
        
        return analysis

def test_all_strategies():
    """Test all post-processing strategies with synthetic MAX Graph features"""
    
    print("🔧 Testing Feature Post-Processing Strategies")
    print("=" * 60)
    
    # Create synthetic MAX Graph features matching our actual distribution
    np.random.seed(42)
    batch_size, seq_len, d_model = 1, 1500, 384
    
    # Generate features with MAX Graph statistics
    features = np.random.normal(0.0310, 1.4475, (batch_size, seq_len, d_model)).astype(np.float32)
    
    processor = FeaturePostProcessor()
    strategies = [
        ('Simple Statistical', processor.simple_statistical_matching),
        ('Adaptive', processor.adaptive_statistical_matching),
        ('Percentile Robust', processor.robust_percentile_matching),
        ('Semantic Preserving', processor.semantic_preserving_transform),
        ('Hybrid', processor.hybrid_approach)
    ]
    
    results = []
    
    for name, strategy_func in strategies:
        print(f"\n🔍 Testing: {name}")
        
        # Apply transformation
        transformed = strategy_func(features)
        
        # Analyze results
        analysis = processor.analyze_transformation(features, transformed)
        results.append((name, analysis))
        
        print(f"   📊 Original:    mean={analysis['original_stats']['mean']:.4f}, std={analysis['original_stats']['std']:.4f}")
        print(f"   📊 Transformed: mean={analysis['transformed_stats']['mean']:.4f}, std={analysis['transformed_stats']['std']:.4f}")
        print(f"   📊 Target:      mean={analysis['target_stats']['mean']:.4f}, std={analysis['target_stats']['std']:.4f}")
        print(f"   ✅ Mean error:  {analysis['mean_error']:.4f}")
        print(f"   ✅ Std error:   {analysis['std_error']:.4f}")
        print(f"   ✅ Total error: {analysis['total_error']:.4f}")
        print(f"   ✅ Correlation: {analysis['correlation']:.4f}")
    
    # Find best strategy
    best_strategy = min(results, key=lambda x: x[1]['total_error'])
    print(f"\n🏆 Best Strategy: {best_strategy[0]}")
    print(f"   📊 Total error: {best_strategy[1]['total_error']:.4f}")
    print(f"   📊 Correlation: {best_strategy[1]['correlation']:.4f}")
    
    return results, best_strategy

if __name__ == "__main__":
    results, best = test_all_strategies()
    
    print(f"\n✅ Feature post-processing analysis complete")
    print(f"   🎯 Recommended strategy: {best[0]}")
    print(f"   📈 Expected improvement: Better decoder compatibility with maintained semantics")