from __future__ import annotations

import torch

from shingan.problems import Problem, Shape

DISTRIBUTIONS = ("normal", "uniform", "positive", "large", "small", "ramp", "sparse")
SPECIAL_VALUES = {"nan": float("nan"), "posinf": float("inf"), "neginf": float("-inf")}
SPECIAL_FRACTION = 0.01
SPARSE_DENSITY = 0.01
LAYOUTS = ("contiguous", "strided")


def make_inputs(
    problem: Problem,
    shape: Shape,
    dtype: torch.dtype,
    dist: str,
    seed: int,
    device: torch.device,
    special: str | None = None,
    layout: str = "contiguous",
) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    tensors = []
    for arg in problem.args:
        dims = arg.resolve(shape)
        if arg.kind == "param":
            values = _param(dims, gen)
        else:
            values = _data(dims, dist, dtype, gen)
            if special is not None:
                values = _inject(values, SPECIAL_VALUES[special], gen)
        tensors.append(_with_layout(values.to(dtype).to(device), layout))
    return tensors


def _with_layout(t: torch.Tensor, layout: str) -> torch.Tensor:
    match layout:
        case "contiguous":
            return t
        case "strided":
            if t.dim() >= 2:
                return t.transpose(-1, -2).contiguous().transpose(-1, -2)
            padded = t.new_empty(2 * t.numel())
            padded[::2] = t
            return padded[::2]
    raise ValueError(f"unknown layout {layout!r}")


def _param(dims: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    return 1.0 + 0.1 * torch.randn(dims, generator=gen, dtype=torch.float64)


def _data(dims: tuple[int, ...], dist: str, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    info = torch.finfo(dtype)
    match dist:
        case "normal":
            return torch.randn(dims, generator=gen, dtype=torch.float64)
        case "uniform":
            return 2.0 * torch.rand(dims, generator=gen, dtype=torch.float64) - 1.0
        case "positive":
            return torch.rand(dims, generator=gen, dtype=torch.float64)
        case "large":
            return info.max**0.25 * torch.randn(dims, generator=gen, dtype=torch.float64)
        case "small":
            return info.tiny**0.25 * torch.randn(dims, generator=gen, dtype=torch.float64)
        case "ramp":
            numel = torch.Size(dims).numel()
            return torch.linspace(-4.0, 4.0, numel, dtype=torch.float64).reshape(dims)
        case "sparse":
            mask = torch.rand(dims, generator=gen, dtype=torch.float64) < SPARSE_DENSITY
            return torch.randn(dims, generator=gen, dtype=torch.float64) * mask
    raise ValueError(f"unknown distribution {dist!r}")


def _inject(values: torch.Tensor, special: float, gen: torch.Generator) -> torch.Tensor:
    mask = torch.rand(values.shape, generator=gen, dtype=torch.float64) < SPECIAL_FRACTION
    mask.view(-1)[0] = True
    return values.masked_fill(mask, special)
