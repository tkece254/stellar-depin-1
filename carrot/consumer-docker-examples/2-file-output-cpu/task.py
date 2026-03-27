import torch
import json
import sys
import time

# Print Docker execution proof
print("="*60)
print("🐳 DOCKER CONTAINER STARTED")
print("="*60)
print(f"Container hostname: {open('/etc/hostname').read().strip()}")
print(f"PyTorch version: {torch.__version__}")

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n⚡ Using device: {device.upper()}")
print("\n" + "="*60)
print("🚀 STARTING COMPUTATION...")
print("="*60 + "\n")
sys.stdout.flush()

results = []
for i in range(100):
    # Compute matrix multiplication
    x = torch.randn(100, 100).to(device)
    y = torch.matmul(x, x)

    results.append({
        "iteration": i,
        "result": float(y.sum().item()),
        "mean": float(y.mean().item())
    })

    # Show progress every 10 iterations
    if (i + 1) % 10 == 0:
        print(f"✓ Completed {i + 1}/100 iterations on {device.upper()}")
        sys.stdout.flush()

    # Small delay to make progress visible
    time.sleep(0.05)

# Save to mounted volume
print("\n" + "="*60)
print("💾 SAVING RESULTS...")
print("="*60)
with open("/results/output.json", "w") as f:
    json.dump({
        "device": device,
        "total_iterations": 100,
        "results": results
    }, f, indent=2)

print(f"✅ Data saved to /results/output.json ({len(results)} iterations)")
print("\n" + "="*60)
print("🎉 COMPUTATION COMPLETE!")
print("="*60)
print("\nRESULT:computation_complete")
