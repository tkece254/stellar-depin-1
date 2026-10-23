from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

import torch

from shingan import timing
from shingan.attacks import Attack, run_attacks
from shingan.contracts import GateResult, Outcome, Status, Verdict
from shingan.gates import GateContext, Kernel, correctness_gates, integrity_gates
from shingan.inputs import make_inputs
from shingan.problems import Problem, dtype_name
from shingan.scan import ENTRY_POINT, scan
from shingan.standard import standard_check

MAX_ERROR_CHARS = 2000


def verify(problem: Problem, kernel_path: Path, dtype: torch.dtype, device: torch.device, seed: int) -> Verdict:
    verdict = Verdict(problem.name, dtype_name(dtype), Outcome.ERROR, environment=environment(device), seed=seed)

    findings = scan(kernel_path.read_text())
    if findings:
        detail = "; ".join(f"line {f.line}: {f.message}" for f in findings)
        verdict.gates.append(GateResult("G0", Status.FAIL, "C10", detail))
        verdict.outcome, verdict.family = Outcome.FAKE, "C10"
        return verdict
    verdict.gates.append(GateResult("G0", Status.PASS))

    try:
        kernel = load_kernel(kernel_path)
        smoke_inputs = make_inputs(problem, problem.public_shapes[0], dtype, "normal", seed, device)
        kernel(*smoke_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception as exc:
        verdict.error = "".join(traceback.format_exception_only(exc)).strip()[:MAX_ERROR_CHARS]
        return verdict

    ctx = GateContext(problem, kernel, dtype, device, seed)
    verdict.gates.extend(correctness_gates(ctx))
    if device.type == "cuda":
        verdict.candidate = timing.measure(kernel, smoke_inputs)
        verdict.eager = timing.measure(problem.reference, smoke_inputs)
        verdict.compiled = timing.measure(torch.compile(problem.reference), smoke_inputs)
        verdict.gates.extend(integrity_gates(ctx, verdict.candidate))

    verdict.outcome, verdict.family = decide(verdict)
    return verdict


def attack(
    problem: Problem,
    kernel_path: Path,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    attacks: Sequence[Attack],
) -> list[GateResult]:
    findings = scan(kernel_path.read_text())
    if findings:
        raise ValueError(f"{kernel_path} fails the static scan and cannot be attacked")
    ctx = GateContext(problem, load_kernel(kernel_path), dtype, device, seed)
    return run_attacks(ctx, attacks)


def standard(problem: Problem, kernel_path: Path, dtype: torch.dtype, device: torch.device, seed: int) -> GateResult:
    try:
        kernel = load_kernel(kernel_path)
    except Exception as exc:
        return GateResult("B0", Status.FAIL, detail="".join(traceback.format_exception_only(exc)).strip()[:MAX_ERROR_CHARS])
    return standard_check(problem, kernel, dtype, device, seed)


def apply_attacks(verdict: Verdict, results: Sequence[GateResult]) -> Verdict:
    verdict.gates.extend(results)
    verdict.outcome, verdict.family = decide(verdict)
    return verdict


def decide(verdict: Verdict) -> tuple[Outcome, str | None]:
    failed = next((g for g in verdict.gates if g.status is Status.FAIL), None)
    if failed:
        return Outcome.FAKE, failed.family
    speedup = verdict.speedup_vs_eager
    if speedup is None:
        return Outcome.UNVERIFIED, None
    return (Outcome.GENUINE if speedup >= 1.0 else Outcome.HONEST_BUT_SLOW), None


def load_kernel(path: Path) -> Kernel:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    name = f"shingan_candidate_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    kernel = vars(module).get(ENTRY_POINT)
    if not callable(kernel):
        raise TypeError(f"{path.name} defines no callable {ENTRY_POINT}()")
    return kernel


def environment(device: torch.device) -> dict[str, str]:
    env = {
        "torch": torch.__version__,
        "triton": _package_version("triton"),
        "python": sys.version.split()[0],
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    if torch.version.cuda:
        env["cuda"] = torch.version.cuda
    return env


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "absent"
