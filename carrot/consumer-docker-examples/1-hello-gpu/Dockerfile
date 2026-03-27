# Simplest GPU Test
# Consumer can submit: username/hello-gpu

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# The actual task
RUN printf 'import torch\n\
print("="*50)\n\
print("BroccoByte GPU Task Running...")\n\
print("="*50)\n\
if torch.cuda.is_available():\n\
    gpu_name = torch.cuda.get_device_name(0)\n\
    print(f"GPU Detected: {gpu_name}")\n\
    x = torch.randn(1000, 1000).cuda()\n\
    y = torch.matmul(x, x)\n\
    result = y.sum().item()\n\
    print(f"Computation Complete")\n\
    print(f"RESULT:{result:.6f}")\n\
else:\n\
    print("No GPU found")\n\
    print("RESULT:no_gpu")' > /task.py

CMD ["python3", "/task.py"]
