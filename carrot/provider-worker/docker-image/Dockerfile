# Simple GPU-enabled Docker image for BroccoByte
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (lightweight)
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# Copy task scripts
COPY tasks/ /app/tasks/
RUN chmod +x /app/tasks/*.py

WORKDIR /app

# Default command
CMD ["python3", "--version"]
