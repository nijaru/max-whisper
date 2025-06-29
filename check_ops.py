#!/usr/bin/env python3
from max.graph import ops

print("All MAX Graph operations:")
ops_list = [attr for attr in dir(ops) if not attr.startswith("_")]
for op in sorted(ops_list):
    print(f"  {op}")

print("\nLooking for slicing/indexing operations:")
slice_ops = [attr for attr in ops_list if any(word in attr.lower() for word in ["slice", "split", "index", "gather", "select"])]
for op in slice_ops:
    print(f"  {op}")