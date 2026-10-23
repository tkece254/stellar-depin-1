# Shingan

真贋, "genuine or fake". Verified speedups for AI-written GPU kernels.

AI kernel agents are known to fake their speedups: hardcoded outputs, shape-specific shortcuts, work moved outside the timer. Shingan has NVIDIA Nemotron write Triton kernels, checks them with fixed gates against an fp64 reference, and then has a second Nemotron model read each kernel and write executable attacks aimed at the assumptions that kernel makes. A kernel only ships if it survives all of it.

## How it works

| Stage | Model / runtime | What it does |
|---|---|---|
| Builder | Nemotron 3 Ultra on Nebius Token Factory | Writes a Triton kernel matching a PyTorch reference |
| Triage | Nemotron 3 Nano on Token Factory | Turns compile and load errors into a short fix hint |
| G0 static scan | local | Rejects PyTorch fallbacks, forbidden imports, introspection and text aimed at a reviewer before any code runs |
| G1–G7 | GPU worker | Distributions, held-out shapes, NaN/Inf, determinism, precision, freshness, input mutation |
| G9–G10 | GPU worker | CUDA-event vs wall-clock timing integrity, hardware speed floor |
| Breaker | Nemotron 3 Super on Token Factory | Reads the kernel, states its hidden assumptions, writes up to 8 input generators that violate them |
| Verdict | local | GENUINE, FAKE, HONEST-BUT-SLOW, UNVERIFIED or ERROR |

Every candidate runs in a separate worker process with a time limit. Gate seeds rotate on every run. When a kernel fails, the Builder sees only the verdict and the failure family, never the failing inputs, so it cannot learn to game the tests. Every failure is stored with the seed and shape needed to replay it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[gpu,dev]'
export NEBIUS_API_KEY=...
```

Or in a container on a GPU host:

```bash
docker build -t shingan .
docker run --rm --gpus all --entrypoint pytest shingan
docker run --rm --gpus all -e NEBIUS_API_KEY -v "$PWD/runs:/app/runs" shingan build --problem softmax
```

Model IDs are picked from Token Factory's model list by role. Override them with `SHINGAN_BUILDER_MODEL`, `SHINGAN_BREAKER_MODEL` and `SHINGAN_TRIAGE_MODEL`.

## Usage

```bash
shingan problems
shingan models
shingan build --problem softmax --dtype float16
shingan verify path/to/kernel.py --problem softmax --dtype float16
shingan verify path/to/kernel.py --problem softmax --gates-only
shingan verify path/to/kernel.py --problem softmax --attacks runs/.../attacks.json
shingan verify path/to/kernel.py --problem softmax --seed 12345
shingan replay path/to/kernel.py --verdict runs/.../verdict.json --gate G2
```

Each verdict records its seed, so `verify --seed` reproduces the whole run, and `replay` re-runs one failing case and prints the elements furthest outside tolerance.

A kernel file defines `run(...)` with the same arguments as the problem's reference and launches at least one `@triton.jit` kernel. `shingan build` writes each attempt, its verdict, the Breaker's attacks and the full transcript under `runs/`. Known-good kernels for checking a new GPU box are in `examples/kernels/`.

## Benchmarks

Every number is measured under three configurations:

| Config | Checks |
|---|---|
| B0 | KernelBench-style: one shape, 5 random inputs, `allclose` at 1e-2 |
| B1 | Shingan gates G0–G10 |
| B2 | B1 plus the Breaker's targeted attacks |

```bash
shingan bench --manifest benchmarks/cheats/manifest.json --breaker live
shingan bench --manifest benchmarks/cheats/manifest.json --breaker replay --replay runs/bench-<earlier>
shingan bench --problems softmax,layernorm,rmsnorm,gelu --dtype float16
shingan block --runs runs/softmax-float16-<id> runs/rmsnorm-float16-<id> runs/swiglu-float16-<id>
```

`--manifest` reports cheat recall and false-reject rate for each configuration, and the cheats caught only by the Breaker. `--problems` runs the Builder per problem and reports how often it reaches GENUINE, which of its kernels the standard check accepted but Shingan rejected, and verified speedups against eager and `torch.compile`. `block` loads the GENUINE kernels from build runs into a transformer block and reports the op time ranking, end-to-end latency and error against fp64.

A corpus manifest lists kernel files relative to the manifest:

```json
{
  "name": "shingan-cheats",
  "entries": [
    {"kernel": "C3/softmax_fixed_block.py", "problem": "softmax", "dtype": "float16", "cheat": true, "family": "C3", "note": "assumes rows fit one 1024-wide block"},
    {"kernel": "controls/softmax.py", "problem": "softmax", "dtype": "float16", "cheat": false}
  ]
}
```

`bench --breaker live` saves each kernel's attacks, so a later `--breaker replay` reproduces B2 without calling the model.

## Tests

```bash
pytest
```

The CPU test suite runs every gate against honest kernels and seeded cheats for each failure family, including a kernel that passes all fixed gates and is caught only by a targeted attack.

## Prior work

Shingan builds on
[KernelBench-Verified](https://arxiv.org/abs/2607.16241),
[A Contract-Grade Verifier for LLM-Generated GPU Kernels](https://arxiv.org/abs/2608.12700),
[The Correctness Illusion in LLM-Generated GPU Kernels](https://arxiv.org/abs/2606.20128),
[KernelGuard](https://icml.cc/virtual/2026/67922) (the hardware speed floor check) and
[Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops](https://arxiv.org/abs/2606.08960).

## License

Apache-2.0
