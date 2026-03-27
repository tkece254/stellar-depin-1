#!/usr/bin/env python3
"""
Simple GPU Test - Consumers can submit this code
Provider will run it on their GPU and return results
"""

import sys

print("🚀 BroccoByte GPU Job Starting...")

# Try to use GPU if available
try:
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Found: {gpu_name}")

        # Do some GPU computation
        print("Computing on GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)

        result = z.sum().item()
        print(f"✅ Computation Complete!")
        print(f"Result: {result:.6f}")

    else:
        print("⚠️ No GPU found, using CPU")
        result = "cpu_fallback"

except ImportError:
    print("⚠️ PyTorch not available")
    result = "basic_compute"

# This output is what gets submitted to blockchain
print(f"\nFINAL_RESULT: {result}")
