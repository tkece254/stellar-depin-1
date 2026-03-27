# Proof of Work - Verifiable Computation
# Consumer can verify the provider actually did the work

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3

# Hash computation task - verifiable by consumer
RUN printf 'import hashlib\n\
import time\n\
print("Proof of Work Task")\n\
print("="*50)\n\
print("Finding hash with prefix 0000...")\n\
target = "0000"\n\
start = time.time()\n\
nonce = 0\n\
max_iterations = 10000000\n\
while nonce < max_iterations:\n\
    data = f"BroccoByte-Job-42-{nonce}".encode()\n\
    hash_result = hashlib.sha256(data).hexdigest()\n\
    if hash_result.startswith(target):\n\
        elapsed = time.time() - start\n\
        print(f"Found!")\n\
        print(f"Nonce: {nonce}")\n\
        print(f"Hash: {hash_result}")\n\
        print(f"Time: {elapsed:.2f}s")\n\
        print(f"Hashrate: {nonce/elapsed:.0f} H/s")\n\
        print(f"RESULT:{nonce}:{hash_result}")\n\
        exit(0)\n\
    if nonce %% 500000 == 0 and nonce > 0:\n\
        print(f"Progress: {nonce:,} hashes...")\n\
    nonce += 1\n\
print(f"Not found in {max_iterations:,} iterations")\n\
print("RESULT:not_found")' > /task.py

CMD ["python3", "/task.py"]
