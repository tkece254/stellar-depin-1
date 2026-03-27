#!/usr/bin/env python3
"""
Matrix Multiplication Benchmark
Tests GPU performance with large matrix operations
"""

import time

try:
    import torch

    print("🔢 Matrix Multiplication GPU Benchmark")
    print("=" * 50)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("GPU: Not available, using CPU")

    # Create large matrices
    size = 5000
    print(f"Matrix size: {size}x{size}")

    print("\nGenerating random matrices...")
    A = torch.randn(size, size).to(device)
    B = torch.randn(size, size).to(device)

    print("Starting multiplication...")
    start = time.time()

    C = torch.matmul(A, B)

    end = time.time()
    elapsed = end - start

    print(f"✅ Complete in {elapsed:.3f} seconds")
    print(f"Performance: {(size**3 * 2) / elapsed / 1e9:.2f} GFLOPS")

    # Result hash
    result_hash = f"0x{int(C.sum().item()) & 0xffffffff:08x}"
    print(f"\nFINAL_RESULT: {result_hash}")

except ImportError:
    print("Error: PyTorch not installed")
    print("FINAL_RESULT: error_no_pytorch")
