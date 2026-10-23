from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence

import torch

from shingan.contracts import Timing

WARMUP = 10
ITERS = 100
INTEGRITY_CALLS = 50
L2_FLUSH_FACTOR = 2


def measure(fn: Callable[..., object], args: Sequence[torch.Tensor]) -> Timing:
    samples = _cuda_samples(fn, args) if args[0].is_cuda else _cpu_samples(fn, args)
    q1, median, q3 = statistics.quantiles(samples, n=4)
    return Timing(median_ms=median, q1_ms=q1, q3_ms=q3)


def event_wall_ratio(fn: Callable[..., object], args: Sequence[torch.Tensor]) -> float:
    device = args[0].device
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    began = time.perf_counter()
    start.record()
    for _ in range(INTEGRITY_CALLS):
        fn(*args)
    end.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - began) * 1e3
    return start.elapsed_time(end) / wall_ms


def _cuda_samples(fn: Callable[..., object], args: Sequence[torch.Tensor]) -> list[float]:
    device = args[0].device
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
    flush = torch.empty(L2_FLUSH_FACTOR * l2_bytes, dtype=torch.uint8, device=device)
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for start, end in zip(starts, ends):
        flush.zero_()
        start.record()
        fn(*args)
        end.record()
    torch.cuda.synchronize(device)
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _cpu_samples(fn: Callable[..., object], args: Sequence[torch.Tensor]) -> list[float]:
    for _ in range(WARMUP):
        fn(*args)
    samples = []
    for _ in range(ITERS):
        began = time.perf_counter()
        fn(*args)
        samples.append((time.perf_counter() - began) * 1e3)
    return samples
