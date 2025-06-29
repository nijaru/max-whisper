#!/usr/bin/env python3
import numpy as np

weights = np.load('whisper_weights/whisper_tiny_weights.npz')
print('Available weights:')
for name in sorted(weights.files)[:20]:
    print(f'  {name}: {weights[name].shape}')

print(f'\nTotal weights: {len(weights.files)}')

# Look for key components
key_weights = ['encoder.conv1.weight', 'encoder.conv2.weight', 'decoder.token_embedding.weight', 'token_embedding']
print('\nKey weights:')
for key in key_weights:
    if key in weights.files:
        print(f'  {key}: {weights[key].shape}')