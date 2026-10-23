from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from shingan.contracts import Outcome, Verdict
from shingan.gates import Kernel
from shingan.pipeline import load_kernel
from shingan.problems import get_problem, parse_dtype


@dataclass(frozen=True)
class VerifiedKernel:
    problem: str
    dtype: torch.dtype
    path: Path
    kernel: Kernel
    speedup_vs_eager: float | None


@dataclass
class KernelTable:
    kernels: dict[tuple[str, torch.dtype], VerifiedKernel] = field(default_factory=dict)

    @classmethod
    def from_runs(cls, run_dirs: Iterable[Path]) -> KernelTable:
        table = cls()
        for run_dir in run_dirs:
            verified = _genuine_attempt(run_dir)
            if verified is not None:
                table.kernels[(verified.problem, verified.dtype)] = verified
        return table

    def op(self, problem: str) -> Callable[..., torch.Tensor]:
        reference = get_problem(problem).reference

        def dispatch(*args: torch.Tensor) -> torch.Tensor:
            verified = self.kernels.get((problem, args[0].dtype))
            if verified is None or not args[0].is_cuda:
                return reference(*args)
            return verified.kernel(*args)

        return dispatch


def _genuine_attempt(run_dir: Path) -> VerifiedKernel | None:
    run = json.loads((run_dir / "run.json").read_text())
    for attempt in reversed(run["attempts"]):
        if attempt["verdict"] is None or attempt["kernel"] is None:
            continue
        verdict = Verdict.from_dict(attempt["verdict"])
        if verdict.outcome is not Outcome.GENUINE:
            continue
        path = Path(attempt["kernel"])
        return VerifiedKernel(run["problem"], parse_dtype(run["dtype"]), path, load_kernel(path), verdict.speedup_vs_eager)
    return None
