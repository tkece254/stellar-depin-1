# GPU Performance Benchmark
# Returns performance metrics

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

RUN printf 'import torch\n\
import time\n\
print("GPU Benchmark Starting...")\n\
print("="*50)\n\
device = "cuda" if torch.cuda.is_available() else "cpu"\n\
print(f"Device: {device}")\n\
if device == "cuda":\n\
    print(f"GPU: {torch.cuda.get_device_name(0)}")\n\
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")\n\
print("\\nRunning benchmark...")\n\
# Matrix multiplication benchmark\n\
sizes = [1000, 2000, 3000, 4000, 5000]\n\
results = []\n\
for size in sizes:\n\
    x = torch.randn(size, size).to(device)\n\
    y = torch.randn(size, size).to(device)\n\
    start = time.time()\n\
    z = torch.matmul(x, y)\n\
    torch.cuda.synchronize() if device == "cuda" else None\n\
    elapsed = time.time() - start\n\
    gflops = (size**3 * 2) / elapsed / 1e9\n\
    results.append(f"{size}x{size}: {elapsed:.3f}s ({gflops:.1f} GFLOPS)")\n\
    print(f"  {size}x{size}: {elapsed:.3f}s - {gflops:.1f} GFLOPS")\n\
print("\\nBenchmark Complete!")\n\
avg_gflops = sum(float(r.split("(")[1].split()[0]) for r in results) / len(results)\n\
print(f"Average Performance: {avg_gflops:.1f} GFLOPS")\n\
print(f"RESULT:benchmark_{avg_gflops:.1f}_gflops")' > /task.py

CMD ["python3", "/task.py"]
