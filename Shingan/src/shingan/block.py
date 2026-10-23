from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from shingan import timing
from shingan.contracts import Timing
from shingan.numerics import Expected, normalized_error
from shingan.problems import PROBLEMS

Op = Callable[..., torch.Tensor]
BLOCK_OPS = ("rmsnorm", "softmax", "swiglu", "matmul")
PROFILE_PREFIX = "shingan::"
BLOCK_ERROR_RATIO = 4.0
BLOCK_ERROR_FLOOR = 1.0


@dataclass(frozen=True)
class BlockConfig:
    seq: int = 1024
    hidden: int = 2048
    heads: int = 16
    ffn: int = 5632

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


@dataclass(frozen=True)
class Ops:
    rmsnorm: Op
    softmax: Op
    swiglu: Op
    matmul: Op

    @classmethod
    def reference(cls) -> Ops:
        return cls(**{name: PROBLEMS[name].reference for name in BLOCK_OPS})


class TransformerBlock:
    def __init__(self, config: BlockConfig, weights: dict[str, torch.Tensor], mask: torch.Tensor) -> None:
        self.config = config
        self.weights = weights
        self.mask = mask

    @classmethod
    def random(cls, config: BlockConfig, seed: int) -> TransformerBlock:
        gen = torch.Generator().manual_seed(seed)
        matrices = {
            "wq": (config.hidden, config.hidden),
            "wk": (config.hidden, config.hidden),
            "wv": (config.hidden, config.hidden),
            "wo": (config.hidden, config.hidden),
            "w_gate": (config.hidden, config.ffn),
            "w_up": (config.hidden, config.ffn),
            "w_down": (config.ffn, config.hidden),
        }
        weights = {name: _init(shape, gen) for name, shape in matrices.items()}
        weights["attn_norm"] = torch.ones(config.hidden, dtype=torch.float64)
        weights["mlp_norm"] = torch.ones(config.hidden, dtype=torch.float64)
        mask = torch.full((config.seq, config.seq), float("-inf"), dtype=torch.float64).triu(1)
        return cls(config, weights, mask)

    def to(self, dtype: torch.dtype, device: torch.device) -> TransformerBlock:
        weights = {name: w.to(device=device, dtype=dtype) for name, w in self.weights.items()}
        return TransformerBlock(self.config, weights, self.mask.to(device=device, dtype=dtype))

    def __call__(self, x: torch.Tensor, ops: Ops) -> torch.Tensor:
        w = self.weights
        h = x + self._attention(_traced(ops, "rmsnorm")(x, w["attn_norm"]), ops)
        normed = _traced(ops, "rmsnorm")(h, w["mlp_norm"])
        matmul = _traced(ops, "matmul")
        hidden = _traced(ops, "swiglu")(matmul(normed, w["w_gate"]), matmul(normed, w["w_up"]))
        return h + matmul(hidden, w["w_down"])

    def _attention(self, x: torch.Tensor, ops: Ops) -> torch.Tensor:
        c, w = self.config, self.weights
        matmul = _traced(ops, "matmul")
        q, k, v = (matmul(x, w[name]).view(c.seq, c.heads, c.head_dim).transpose(0, 1) for name in ("wq", "wk", "wv"))
        scores = (q @ k.transpose(1, 2)) / math.sqrt(c.head_dim) + self.mask
        probs = _traced(ops, "softmax")(scores.reshape(-1, c.seq)).view(c.heads, c.seq, c.seq)
        attended = (probs @ v).transpose(0, 1).reshape(c.seq, c.hidden)
        return matmul(attended, w["wo"])


@dataclass(frozen=True)
class BlockReport:
    profile: list[tuple[str, float]]
    original: Timing | None
    patched: Timing | None
    original_error: float
    patched_error: float

    @property
    def speedup(self) -> float | None:
        if self.original is None or self.patched is None:
            return None
        return self.original.median_ms / self.patched.median_ms

    @property
    def correct(self) -> bool:
        return self.patched_error <= BLOCK_ERROR_RATIO * max(self.original_error, BLOCK_ERROR_FLOOR)


def evaluate(master: TransformerBlock, patched_ops: Ops, dtype: torch.dtype, device: torch.device, seed: int) -> BlockReport:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(master.config.seq, master.config.hidden, generator=gen, dtype=torch.float64).to(dtype).to(device)
    block = master.to(dtype, device)
    original_ops = Ops.reference()
    with torch.no_grad():
        exact = master(x.double().cpu(), original_ops)
        expected = Expected.of(exact.to(dtype))
        original_error = normalized_error(block(x, original_ops), expected, dtype)
        patched_error = normalized_error(block(x, patched_ops), expected, dtype)
        ranking = profile_ops(block, x, original_ops)
        original = patched = None
        if device.type == "cuda":
            original = timing.measure(lambda t: block(t, original_ops), [x])
            patched = timing.measure(lambda t: block(t, patched_ops), [x])
    return BlockReport(ranking, original, patched, original_error, patched_error)


def profile_ops(block: TransformerBlock, x: torch.Tensor, ops: Ops) -> list[tuple[str, float]]:
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if x.is_cuda else [])
    with profile(activities=activities) as prof:
        block(x, ops)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
    totals = []
    for event in prof.key_averages():
        if event.key.startswith(PROFILE_PREFIX):
            device_us = event.device_time_total if x.is_cuda else event.cpu_time_total
            totals.append((event.key.removeprefix(PROFILE_PREFIX), device_us / 1e3))
    return sorted(totals, key=lambda item: item[1], reverse=True)


def _traced(ops: Ops, name: str) -> Op:
    op = getattr(ops, name)

    def call(*args: torch.Tensor) -> torch.Tensor:
        with record_function(f"{PROFILE_PREFIX}{name}"):
            return op(*args)

    return call


def _init(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=gen, dtype=torch.float64) / math.sqrt(shape[0])
