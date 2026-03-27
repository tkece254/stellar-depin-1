#!/usr/bin/env python3
"""
Hash Computation - Proof of Work Style
Consumer can verify the provider actually did the computation
"""

import hashlib
import time

# Parameters (consumer can customize these)
TARGET_PREFIX = "0000"  # Find hash starting with 0000
MAX_ITERATIONS = 1000000

print("🔐 Hash Computation Task")
print("=" * 50)
print(f"Target: Hash starting with '{TARGET_PREFIX}'")
print(f"Max iterations: {MAX_ITERATIONS:,}")
print()

start_time = time.time()
nonce = 0

while nonce < MAX_ITERATIONS:
    data = f"BroccoByte-{nonce}".encode()
    hash_result = hashlib.sha256(data).hexdigest()

    if hash_result.startswith(TARGET_PREFIX):
        elapsed = time.time() - start_time
        print(f"✅ Found!")
        print(f"Nonce: {nonce}")
        print(f"Hash: {hash_result}")
        print(f"Time: {elapsed:.3f}s")
        print(f"Hash rate: {nonce/elapsed:.0f} H/s")
        print(f"\nFINAL_RESULT: {hash_result}")
        exit(0)

    if nonce % 100000 == 0 and nonce > 0:
        print(f"Progress: {nonce:,} iterations...")

    nonce += 1

# Not found
print(f"⚠️ Target not found in {MAX_ITERATIONS:,} iterations")
print(f"FINAL_RESULT: not_found")
