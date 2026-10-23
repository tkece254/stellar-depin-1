from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from shingan.attacks import GENERATOR_TIMEOUT_S, MAX_ATTACKS
from shingan.contracts import GateResult, Outcome, Status, Verdict

DEFAULT_TIMEOUT_S = 600
STANDARD_TIMEOUT_S = 300
ATTACK_TIMEOUT_S = 120 + 2 * MAX_ATTACKS * GENERATOR_TIMEOUT_S

WorkerMode = Literal["verify", "standard", "attack"]


def verify_isolated(
    problem: str,
    kernel_path: Path,
    dtype: str,
    verdict_path: Path,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    seed: int | None = None,
) -> Verdict:
    returncode = _run_worker("verify", problem, kernel_path, dtype, verdict_path, timeout_s, seed)
    if returncode is None:
        return Verdict(problem, dtype, Outcome.ERROR, error=f"verification timed out after {timeout_s}s")
    if returncode != 0 or not verdict_path.exists():
        return Verdict(problem, dtype, Outcome.ERROR, error=f"verification worker exited with code {returncode}")
    return Verdict.from_dict(json.loads(verdict_path.read_text()))


def standard_isolated(
    problem: str,
    kernel_path: Path,
    dtype: str,
    result_path: Path,
    timeout_s: int = STANDARD_TIMEOUT_S,
    seed: int | None = None,
) -> GateResult:
    returncode = _run_worker("standard", problem, kernel_path, dtype, result_path, timeout_s, seed)
    if returncode is None:
        return GateResult("B0", Status.FAIL, detail=f"timed out after {timeout_s}s")
    if returncode != 0 or not result_path.exists():
        return GateResult("B0", Status.FAIL, detail=f"worker exited with code {returncode}")
    return GateResult.from_dict(json.loads(result_path.read_text()))


def attack_isolated(
    problem: str,
    kernel_path: Path,
    dtype: str,
    attacks_path: Path,
    results_path: Path,
    timeout_s: int = ATTACK_TIMEOUT_S,
) -> list[GateResult]:
    extra = ["--attacks", str(attacks_path)]
    returncode = _run_worker("attack", problem, kernel_path, dtype, results_path, timeout_s, None, extra)
    if returncode is None:
        return [GateResult("B", Status.FAIL, "NEW", f"attacks timed out after {timeout_s}s")]
    if returncode != 0 or not results_path.exists():
        return [GateResult("B", Status.FAIL, "NEW", f"kernel crashed the worker under attack (exit code {returncode})")]
    return [GateResult.from_dict(item) for item in json.loads(results_path.read_text())]


def _run_worker(
    mode: WorkerMode,
    problem: str,
    kernel_path: Path,
    dtype: str,
    out_path: Path,
    timeout_s: int,
    seed: int | None,
    extra: list[str] | None = None,
) -> int | None:
    command = [
        sys.executable, "-m", "shingan", "worker", str(kernel_path), "--mode", mode,
        "--problem", problem, "--dtype", dtype, "--out", str(out_path),
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    command += extra or []
    out_path.unlink(missing_ok=True)
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    out_path.with_suffix(".log").write_text(proc.stdout + proc.stderr)
    return proc.returncode
