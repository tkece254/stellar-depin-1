from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

Shape = Mapping[str, int]
ArgKind = Literal["data", "param"]

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"unsupported dtype {name!r}; choose from {', '.join(DTYPES)}")
    return DTYPES[name]


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class Arg:
    name: str
    dims: tuple[str, ...]
    kind: ArgKind = "data"

    def resolve(self, shape: Shape) -> tuple[int, ...]:
        return tuple(shape[d] for d in self.dims)


def _memory_bound(shape: Shape) -> float:
    return 0.0


@dataclass(frozen=True)
class Problem:
    name: str
    reference: Callable[..., torch.Tensor]
    args: tuple[Arg, ...]
    out_dims: tuple[str, ...]
    public_shapes: tuple[Shape, ...]
    heldout_shapes: tuple[Shape, ...]
    dtypes: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32)
    flops: Callable[[Shape], float] = _memory_bound
    error_scale: Callable[..., torch.Tensor] | None = None
    deterministic: bool = True

    def output_numel(self, shape: Shape) -> int:
        return math.prod(shape[d] for d in self.out_dims)

    @property
    def dims(self) -> frozenset[str]:
        return frozenset(d for arg in self.args for d in arg.dims) | frozenset(self.out_dims)


def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:], weight, bias, eps=1e-5)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * weight


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def _matmul_flops(shape: Shape) -> float:
    return 2.0 * shape["M"] * shape["N"] * shape["K"]


def _matmul_error_scale(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.abs() @ b.abs()


ROWS = ("M", "N")
ROW_PUBLIC: tuple[Shape, ...] = ({"M": 4096, "N": 4096},)
ROW_HELDOUT: tuple[Shape, ...] = (
    {"M": 1, "N": 4096},
    {"M": 4093, "N": 4000},
    {"M": 3, "N": 7},
    {"M": 512, "N": 16384},
)


def _rowwise(name: str, reference: Callable[..., torch.Tensor], *args: Arg) -> Problem:
    return Problem(name, reference, args, ROWS, ROW_PUBLIC, ROW_HELDOUT)


PROBLEMS: dict[str, Problem] = {
    p.name: p
    for p in (
        _rowwise("add_relu", add_relu, Arg("x", ROWS), Arg("y", ROWS)),
        _rowwise("gelu", gelu, Arg("x", ROWS)),
        _rowwise("softmax", softmax, Arg("x", ROWS)),
        _rowwise("layernorm", layernorm, Arg("x", ROWS), Arg("weight", ("N",), "param"), Arg("bias", ("N",), "param")),
        _rowwise("rmsnorm", rmsnorm, Arg("x", ROWS), Arg("weight", ("N",), "param")),
        _rowwise("swiglu", swiglu, Arg("gate", ROWS), Arg("up", ROWS)),
        Problem(
            "matmul",
            matmul,
            (Arg("a", ("M", "K")), Arg("b", ("K", "N"))),
            ("M", "N"),
            ({"M": 2048, "N": 2048, "K": 2048},),
            (
                {"M": 1, "N": 2048, "K": 2048},
                {"M": 1000, "N": 1030, "K": 999},
                {"M": 17, "N": 5, "K": 3},
            ),
            flops=_matmul_flops,
            error_scale=_matmul_error_scale,
        ),
    )
}


def get_problem(name: str) -> Problem:
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name!r}; choose from {', '.join(PROBLEMS)}")
    return PROBLEMS[name]
