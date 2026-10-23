from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from shingan.problems import Problem

Tolerance = tuple[float, float]

TOLERANCES: dict[torch.dtype, Tolerance] = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}

BIT_VIEWS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

NONFINITE_CHECKS = (("NaN", torch.isnan), ("+Inf", torch.isposinf), ("-Inf", torch.isneginf))


@dataclass(frozen=True)
class Expected:
    value: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def of(cls, value: torch.Tensor) -> Expected:
        exact = value.detach().cpu().double()
        return cls(exact, exact.abs())


def expected_output(problem: Problem, inputs: Sequence[torch.Tensor], dtype: torch.dtype) -> Expected:
    exact = [t.detach().cpu().double() for t in inputs]
    with torch.no_grad():
        value = problem.reference(*exact).to(dtype).double()
        scale = problem.error_scale(*exact) if problem.error_scale else value.abs()
    return Expected(value, scale)


def compare(out: object, expected: Expected, dtype: torch.dtype) -> str | None:
    if not isinstance(out, torch.Tensor):
        return f"returned {type(out).__name__}, expected a tensor"
    if out.dtype != dtype:
        return f"returned dtype {out.dtype}, expected {dtype}"
    if out.shape != expected.value.shape:
        return f"returned shape {tuple(out.shape)}, expected {tuple(expected.value.shape)}"
    got = out.detach().cpu().double()
    for label, check in NONFINITE_CHECKS:
        disagree = check(got) != check(expected.value)
        if disagree.any():
            return f"{int(disagree.sum())} elements disagree with the reference on {label}"
    finite = torch.isfinite(expected.value)
    error = (got - expected.value).abs()[finite]
    bad = error > _bound(expected, dtype)[finite]
    if bad.any():
        return f"{int(bad.sum())} of {bad.numel()} elements outside tolerance, max abs error {float(error.max()):.3e}"
    return None


def normalized_error(out: torch.Tensor, expected: Expected, dtype: torch.dtype) -> float:
    got = out.detach().cpu().double()
    finite = torch.isfinite(expected.value) & torch.isfinite(got)
    if not finite.any():
        return 0.0
    return float(error_ratio(got, expected, dtype)[finite].max())


def error_ratio(out: torch.Tensor, expected: Expected, dtype: torch.dtype) -> torch.Tensor:
    got = out.detach().cpu().double()
    ratio = (got - expected.value).abs() / _bound(expected, dtype)
    agree = (got == expected.value) | (torch.isnan(got) & torch.isnan(expected.value))
    return ratio.masked_fill(agree, 0.0).nan_to_num(nan=math.inf, posinf=math.inf)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(BIT_VIEWS[t.element_size()])


def _bound(expected: Expected, dtype: torch.dtype) -> torch.Tensor:
    atol, rtol = TOLERANCES[dtype]
    return atol + rtol * expected.scale
