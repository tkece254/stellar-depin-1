from __future__ import annotations

import itertools
import math
import traceback
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from shingan import timing
from shingan.contracts import GateResult, Replay, Status, Timing
from shingan.hardware import Peak, peak_for
from shingan.inputs import DISTRIBUTIONS, SPECIAL_VALUES, make_inputs
from shingan.numerics import Expected, bits, compare, expected_output, normalized_error
from shingan.problems import Problem, Shape, dtype_name

DIST_FAMILY = {
    "normal": "C1",
    "uniform": "C1",
    "positive": "C5",
    "large": "C4",
    "small": "C4",
    "ramp": "C2",
    "sparse": "C2",
}
PRECISION_DISTS = ("normal", "positive")
PRECISION_RATIO = 8.0
PRECISION_FLOOR = 0.1
DETERMINISM_RUNS = 3
MIN_EVENT_WALL_RATIO = 0.7
PHYSICS_SLACK = 0.9
MAX_ERROR_CHARS = 300

Kernel = Callable[..., object]


@dataclass(frozen=True)
class GateContext:
    problem: Problem
    kernel: Kernel
    dtype: torch.dtype
    device: torch.device
    seed: int

    def case_seed(self, gate: str, index: int) -> int:
        return zlib.crc32(f"{self.seed}:{gate}:{index}".encode())

    def inputs(
        self, shape: Shape, dist: str, seed: int, special: str | None = None, layout: str = "contiguous"
    ) -> list[torch.Tensor]:
        return make_inputs(self.problem, shape, self.dtype, dist, seed, self.device, special, layout)

    def replay(
        self, shape: Shape, dist: str, seed: int, special: str | None = None, layout: str = "contiguous"
    ) -> Replay:
        return Replay(seed, dict(shape), dtype_name(self.dtype), dist, special, layout)

    def call(self, inputs: Sequence[torch.Tensor]) -> tuple[object, str | None]:
        try:
            out = self.kernel(*inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception as exc:
            return None, "".join(traceback.format_exception_only(exc)).strip()[:MAX_ERROR_CHARS]
        return out, None


def correctness_gates(ctx: GateContext) -> list[GateResult]:
    return [
        distributions(ctx),
        heldout_shapes(ctx),
        special_values(ctx),
        determinism(ctx),
        precision(ctx),
        freshness(ctx),
        input_integrity(ctx),
    ]


def integrity_gates(ctx: GateContext, candidate: Timing) -> list[GateResult]:
    return [timing_integrity(ctx), physics_floor(ctx, candidate)]


def distributions(ctx: GateContext) -> GateResult:
    cases = itertools.product(ctx.problem.public_shapes, DISTRIBUTIONS)
    for index, (shape, dist) in enumerate(cases):
        failure = _correctness_case(ctx, "G1", DIST_FAMILY[dist], shape, dist, ctx.case_seed("G1", index))
        if failure:
            return failure
    return GateResult("G1", Status.PASS)


def heldout_shapes(ctx: GateContext) -> GateResult:
    cases = [(shape, "contiguous") for shape in ctx.problem.heldout_shapes]
    cases += [(shape, "strided") for shape in ctx.problem.public_shapes]
    for index, (shape, layout) in enumerate(cases):
        seed = ctx.case_seed("G2", index)
        failure = _correctness_case(ctx, "G2", "C3", shape, "normal", seed, layout=layout)
        if failure:
            return failure
    return GateResult("G2", Status.PASS)


def special_values(ctx: GateContext) -> GateResult:
    cases = itertools.product(ctx.problem.public_shapes, SPECIAL_VALUES)
    for index, (shape, special) in enumerate(cases):
        seed = ctx.case_seed("G3", index)
        failure = _correctness_case(ctx, "G3", "C4", shape, "normal", seed, special)
        if failure:
            return failure
    return GateResult("G3", Status.PASS)


def determinism(ctx: GateContext) -> GateResult:
    shape = ctx.problem.public_shapes[0]
    seed = ctx.case_seed("G4", 0)
    inputs = ctx.inputs(shape, "normal", seed)
    first, error = ctx.call(_cloned(inputs))
    if error or not isinstance(first, torch.Tensor):
        return GateResult("G4", Status.FAIL, "C6", error or "did not return a tensor", ctx.replay(shape, "normal", seed))
    for _ in range(DETERMINISM_RUNS - 1):
        again, error = ctx.call(_cloned(inputs))
        if error:
            return GateResult("G4", Status.FAIL, "C6", error, ctx.replay(shape, "normal", seed))
        if _differs(first, again, ctx):
            return GateResult("G4", Status.FAIL, "C6", "repeated runs on identical inputs disagree", ctx.replay(shape, "normal", seed))
    return GateResult("G4", Status.PASS)


def precision(ctx: GateContext) -> GateResult:
    cases = itertools.product(ctx.problem.public_shapes, PRECISION_DISTS)
    for index, (shape, dist) in enumerate(cases):
        seed = ctx.case_seed("G5", index)
        inputs = ctx.inputs(shape, dist, seed)
        expected = expected_output(ctx.problem, inputs, ctx.dtype)
        with torch.no_grad():
            native = ctx.problem.reference(*inputs)
        out, error = ctx.call(inputs)
        mismatch = error or compare(out, expected, ctx.dtype)
        if mismatch:
            return GateResult("G5", Status.FAIL, "C5", mismatch, ctx.replay(shape, dist, seed))
        candidate_error = normalized_error(out, expected, ctx.dtype)
        native_error = normalized_error(native, expected, ctx.dtype)
        if candidate_error > max(PRECISION_RATIO * native_error, PRECISION_FLOOR):
            detail = f"error {candidate_error:.3g} tolerance units, PyTorch gets {native_error:.3g}"
            return GateResult("G5", Status.FAIL, "C5", detail, ctx.replay(shape, dist, seed))
    return GateResult("G5", Status.PASS)


def freshness(ctx: GateContext) -> GateResult:
    shape = ctx.problem.public_shapes[0]
    seed_first, seed_second = ctx.case_seed("G6", 0), ctx.case_seed("G6", 1)
    first = ctx.inputs(shape, "normal", seed_first)
    second = ctx.inputs(shape, "normal", seed_second)
    expected = expected_output(ctx.problem, second, ctx.dtype)
    replay = ctx.replay(shape, "normal", seed_second)

    mismatch = _error(ctx, _cloned(first)) or _mismatch(ctx, _cloned(second), expected)
    if mismatch:
        return GateResult("G6", Status.FAIL, "C7", f"new inputs after a call: {mismatch}", replay)

    reused = _cloned(first)
    error = _error(ctx, reused)
    if error is None:
        for target, source in zip(reused, second):
            target.copy_(source)
    mismatch = error or _mismatch(ctx, reused, expected)
    if mismatch:
        return GateResult("G6", Status.FAIL, "C7", f"inputs rewritten in place: {mismatch}", replay)
    return GateResult("G6", Status.PASS)


def input_integrity(ctx: GateContext) -> GateResult:
    shape = ctx.problem.public_shapes[0]
    seed = ctx.case_seed("G7", 0)
    inputs = ctx.inputs(shape, "normal", seed)
    before = [bits(t).clone() for t in inputs]
    _, error = ctx.call(inputs)
    if error:
        return GateResult("G7", Status.FAIL, "C9", error, ctx.replay(shape, "normal", seed))
    for arg, tensor, snapshot in zip(ctx.problem.args, inputs, before):
        if not torch.equal(bits(tensor), snapshot):
            return GateResult("G7", Status.FAIL, "C9", f"input {arg.name!r} was modified", ctx.replay(shape, "normal", seed))
    return GateResult("G7", Status.PASS)


def timing_integrity(ctx: GateContext) -> GateResult:
    if ctx.device.type != "cuda":
        return GateResult("G9", Status.SKIP, detail="needs a CUDA device")
    shape = ctx.problem.public_shapes[0]
    seed = ctx.case_seed("G9", 0)
    ratio = timing.event_wall_ratio(ctx.kernel, ctx.inputs(shape, "normal", seed))
    if ratio < MIN_EVENT_WALL_RATIO:
        detail = f"CUDA events saw {ratio:.0%} of wall-clock time; work ran outside the timed stream"
        return GateResult("G9", Status.FAIL, "C8", detail, ctx.replay(shape, "normal", seed))
    return GateResult("G9", Status.PASS, detail=f"event/wall ratio {ratio:.2f}")


def physics_floor(ctx: GateContext, candidate: Timing) -> GateResult:
    if ctx.device.type != "cuda":
        return GateResult("G10", Status.SKIP, detail="needs a CUDA device")
    device_name = torch.cuda.get_device_name(ctx.device)
    peak = peak_for(device_name)
    if peak is None:
        return GateResult("G10", Status.SKIP, detail=f"no peak figures for {device_name}")
    floor_ms = hardware_floor_ms(ctx.problem, ctx.problem.public_shapes[0], ctx.dtype, peak)
    if candidate.median_ms < PHYSICS_SLACK * floor_ms:
        detail = f"{candidate.median_ms:.4f} ms is below the {floor_ms:.4f} ms hardware floor on {device_name}"
        return GateResult("G10", Status.FAIL, "C8", detail)
    return GateResult("G10", Status.PASS, detail=f"floor {floor_ms:.4f} ms")


def hardware_floor_ms(problem: Problem, shape: Shape, dtype: torch.dtype, peak: Peak) -> float:
    itemsize = torch.finfo(dtype).bits // 8
    elements = sum(math.prod(arg.resolve(shape)) for arg in problem.args) + problem.output_numel(shape)
    return 1e3 * max(itemsize * elements / peak.bytes_per_s, problem.flops(shape) / peak.flops_per_s)


def _correctness_case(
    ctx: GateContext,
    gate: str,
    family: str,
    shape: Shape,
    dist: str,
    seed: int,
    special: str | None = None,
    layout: str = "contiguous",
) -> GateResult | None:
    inputs = ctx.inputs(shape, dist, seed, special, layout)
    expected = expected_output(ctx.problem, inputs, ctx.dtype)
    mismatch = _mismatch(ctx, inputs, expected)
    if mismatch is None:
        return None
    return GateResult(gate, Status.FAIL, family, mismatch, ctx.replay(shape, dist, seed, special, layout))


def _mismatch(ctx: GateContext, inputs: Sequence[torch.Tensor], expected: Expected) -> str | None:
    out, error = ctx.call(inputs)
    return error or compare(out, expected, ctx.dtype)


def _error(ctx: GateContext, inputs: Sequence[torch.Tensor]) -> str | None:
    return ctx.call(inputs)[1]


def _differs(first: torch.Tensor, again: object, ctx: GateContext) -> bool:
    if not isinstance(again, torch.Tensor) or again.shape != first.shape or again.dtype != first.dtype:
        return True
    if ctx.problem.deterministic:
        return not torch.equal(bits(first), bits(again))
    return compare(again, Expected.of(first), ctx.dtype) is not None


def _cloned(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [t.clone() for t in inputs]
