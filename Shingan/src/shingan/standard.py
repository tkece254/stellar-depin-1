from __future__ import annotations

import traceback

import torch

from shingan.contracts import GateResult, Replay, Status
from shingan.gates import Kernel
from shingan.inputs import make_inputs
from shingan.problems import Problem, dtype_name

STANDARD_TRIALS = 5
STANDARD_TOLERANCE = 1e-2
MAX_ERROR_CHARS = 300


def standard_check(problem: Problem, kernel: Kernel, dtype: torch.dtype, device: torch.device, seed: int) -> GateResult:
    shape = problem.public_shapes[0]
    for trial in range(STANDARD_TRIALS):
        trial_seed = seed + trial
        replay = Replay(trial_seed, dict(shape), dtype_name(dtype), "normal")
        inputs = make_inputs(problem, shape, dtype, "normal", trial_seed, device)
        with torch.no_grad():
            expected = problem.reference(*inputs)
        try:
            out = kernel(*inputs)
        except Exception as exc:
            reason = "".join(traceback.format_exception_only(exc)).strip()[:MAX_ERROR_CHARS]
            return GateResult("B0", Status.FAIL, detail=reason, replay=replay)
        if not isinstance(out, torch.Tensor) or out.shape != expected.shape:
            return GateResult("B0", Status.FAIL, detail="output shape mismatch", replay=replay)
        if not torch.allclose(out.to(expected.dtype), expected, atol=STANDARD_TOLERANCE, rtol=STANDARD_TOLERANCE):
            return GateResult("B0", Status.FAIL, detail="allclose failed", replay=replay)
    return GateResult("B0", Status.PASS, detail=f"{STANDARD_TRIALS} random inputs at one shape")
