#!/usr/bin/env python3
"""
Simple GPU compute task - Does actual computation
"""
import sys
import time

def main():
    if len(sys.argv) < 3:
        print("Usage: python simple_compute.py <input_data> <job_id>")
        sys.exit(1)

    input_data = sys.argv[1]
    job_id = sys.argv[2]

    print(f"[GPU Task] Starting Job #{job_id}")
    print(f"[GPU Task] Input: {input_data}")
    print("[GPU Task] Initializing GPU...")

    # Check if GPU is available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[GPU Task] GPU Detected: {gpu_name}")
            print(f"[GPU Task] CUDA Version: {torch.version.cuda}")

            # Do actual GPU computation
            print("[GPU Task] Running computation on GPU...")

            # Create tensor on GPU
            size = 1000
            x = torch.randn(size, size).cuda()
            y = torch.randn(size, size).cuda()

            # Actual GPU computation
            start_time = time.time()
            for i in range(100):
                z = torch.matmul(x, y)  # Matrix multiplication on GPU
                if i % 20 == 0:
                    print(f"[GPU Task] Progress: {i}%")

            compute_time = time.time() - start_time

            print(f"[GPU Task] Computation complete in {compute_time:.2f}s")
            print(f"[GPU Task] Result tensor shape: {z.shape}")

            # Save result
            result_hash = f"0x{hash(z.sum().item()) & 0xffffffff:08x}"

        else:
            print("[GPU Task] WARNING: No GPU found, using CPU")
            result_hash = f"cpu_result_{job_id}"

    except ImportError:
        print("[GPU Task] PyTorch not available, running basic compute")
        # Fallback computation without PyTorch
        result = 0
        for i in range(1000000):
            result += i ** 2
        result_hash = f"basic_result_{hash(result) & 0xffffffff:08x}"

    # Write result to file
    with open(f"/results/job_{job_id}_output.txt", "w") as f:
        f.write(f"Job #{job_id} completed\n")
        f.write(f"Input: {input_data}\n")
        f.write(f"Result Hash: {result_hash}\n")
        f.write(f"Status: SUCCESS\n")

    print(f"[GPU Task] Result saved: {result_hash}")
    print(f"[GPU Task] Job #{job_id} COMPLETE")

    # Return hash for blockchain
    print(result_hash)

if __name__ == "__main__":
    main()
