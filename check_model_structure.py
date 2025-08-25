#!/usr/bin/env python3
"""
Check CLIP-RT model structure and sizes
"""

import pickle
import os
import sys

def check_model_file(filepath):
    """Check model file structure without torch"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Get file size
    size_gb = os.path.getsize(filepath) / 1e9
    print(f"\nFile: {os.path.basename(filepath)}")
    print(f"Size: {size_gb:.2f} GB")
    
    try:
        # Try to load with pickle to check structure
        with open(filepath, 'rb') as f:
            # Read first few bytes to check if it's a valid pickle/torch file
            header = f.read(10)
            f.seek(0)
            
            # PyTorch files typically start with specific magic numbers
            if header.startswith(b'PK'):
                print("Format: ZIP archive (standard PyTorch format)")
            elif header.startswith(b'\x80\x02'):
                print("Format: Pickle protocol 2")
            else:
                print(f"Format: Unknown (header: {header[:6]})")
                
    except Exception as e:
        print(f"Error reading file: {e}")

# Check the models
models = [
    'clip-rt/clip-rt/cliprt_libero_10.pt',
    'clip-rt/clip-rt/cliprt_libero_goal.pt', 
    'clip-rt/clip-rt/cliprt_libero_object.pt'
]

print("Checking CLIP-RT model files...")
print("=" * 60)

for model_path in models:
    check_model_file(model_path)

print("\n" + "=" * 60)
print("\nAnalysis:")
print("-" * 40)

# Check if files exist and compare sizes
existing = [m for m in models if os.path.exists(m)]
if len(existing) >= 2:
    sizes = {os.path.basename(m): os.path.getsize(m) for m in existing}
    max_size = max(sizes.values())
    
    for name, size in sizes.items():
        ratio = size / max_size
        if ratio < 0.6:
            print(f"⚠️  {name} is {ratio:.1%} of the largest model")
            print(f"   This suggests it might be missing components:")
            print(f"   - Possibly missing vision/text encoders")
            print(f"   - May only contain action decoder")
            print(f"   - Could be a checkpoint vs full model")
        else:
            print(f"✓ {name} appears to be a complete model")

