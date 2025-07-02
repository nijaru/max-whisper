#!/usr/bin/env python3

from max.graph import ops
import inspect

print('=== AVAILABLE MAX GRAPH OPERATIONS ===')
op_members = [name for name, obj in inspect.getmembers(ops) if callable(obj) and not name.startswith('_')]
conv_ops = [op for op in op_members if 'conv' in op.lower()]
print(f'Conv operations: {conv_ops}')

norm_ops = [op for op in op_members if 'norm' in op.lower()]  
print(f'Normalization operations: {norm_ops}')

# Check specific operations
print(f'\nAll available ops: {sorted(op_members)[:20]}...({len(op_members)} total)')

# Check if conv1d exists
if 'conv1d' in conv_ops:
    print('\n✅ Native conv1d available!')
    try:
        print(f'conv1d signature: {inspect.signature(ops.conv1d)}')
    except:
        print('Could not get conv1d signature')
else:
    print('\n❌ No native conv1d found, using conv2d emulation is correct')