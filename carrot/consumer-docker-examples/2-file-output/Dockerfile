# Task that saves results to file
# Provider will upload this file to IPFS

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# Task that generates large output and saves to file
RUN printf 'import torch\n\
import json\n\
print("Generating large dataset...")\n\
# Simulate computation\n\
results = []\n\
for i in range(100):\n\
    x = torch.randn(100, 100).cuda()\n\
    y = torch.matmul(x, x)\n\
    results.append({\n\
        "iteration": i,\n\
        "result": float(y.sum().item()),\n\
        "mean": float(y.mean().item())\n\
    })\n\
# Save to file (provider will upload to IPFS)\n\
with open("/results/output.json", "w") as f:\n\
    json.dump({"results": results}, f, indent=2)\n\
print("Data saved to /results/output.json")\n\
print("RESULT:file_generated")' > /task.py

CMD ["python3", "/task.py"]
