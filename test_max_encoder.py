#!/usr/bin/env python3

import sys
sys.path.append('max-whisper')
from whisper_max import WhisperMAX
import numpy as np

print('=== TESTING MAX GRAPH ENCODER OUTPUT ===')
model = WhisperMAX(model_size='tiny', use_gpu=False)  # Use CPU for consistent comparison

# Create same test input as our debug script
np.random.seed(42)
mel_db = np.random.randn(80, 3000) * 20 - 40

print(f'Input shape: {mel_db.shape}')
print(f'Input stats: mean={np.mean(mel_db):.6f}, std={np.std(mel_db):.6f}')

# Process through MAX Graph encoder
max_output = model._encode_with_max_graph(mel_db)

print(f'\nMAX Graph output shape: {max_output.shape}')
print(f'MAX Graph output stats: mean={np.mean(max_output):.6f}, std={np.std(max_output):.6f}, min={np.min(max_output):.6f}, max={np.max(max_output):.6f}')

# Compare with expected OpenAI output (from our debug script)
expected_mean = 0.022354
expected_std = 1.723039
expected_min = -8.963874  
expected_max = 17.015629

print(f'\n=== COMPARISON WITH OPENAI REFERENCE ===')
print(f'Mean difference: {np.mean(max_output) - expected_mean:.6f}')
print(f'Std difference: {np.std(max_output) - expected_std:.6f}')
print(f'Min difference: {np.min(max_output) - expected_min:.6f}')
print(f'Max difference: {np.max(max_output) - expected_max:.6f}')

# Calculate similarity percentage
mean_similarity = 1 - abs(np.mean(max_output) - expected_mean) / abs(expected_mean)
std_similarity = 1 - abs(np.std(max_output) - expected_std) / expected_std
print(f'\nSimilarity: Mean {mean_similarity*100:.2f}%, Std {std_similarity*100:.2f}%')