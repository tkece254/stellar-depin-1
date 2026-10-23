from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from shingan.attacks import attack_inputs, attacks_from_json
from shingan.contracts import GateResult, Verdict
from shingan.inputs import make_inputs
from shingan.numerics import Expected, compare, error_ratio, expected_output, normalized_error
from shingan.pipeline import load_kernel
from shingan.problems import Problem, get_problem, parse_dtype
from shingan.scan import scan

SINGLE_CALL_GATES = frozenset({"G1", "G2", "G3", "G5"})
ATTACK_PREFIX = "attack:"
WORST_ELEMENTS = 5


@dataclass(frozen=True)
class Discrepancy:
    index: tuple[int, ...]
    expected: float
    got: float
    tolerance_units: float


@dataclass(frozen=True)
class ReplayResult:
    gate: GateResult
    mismatch: str | None
    error: float | None
    worst: list[Discrepancy]


def replay_gate(
    kernel_path: Path,
    verdict: Verdict,
    gate_name: str,
    device: torch.device,
    attacks_path: Path | None = None,
) -> ReplayResult:
    gate = next((g for g in verdict.gates if g.gate == gate_name), None)
    if gate is None or gate.replay is None:
        raise ValueError(f"{gate_name} has no replay record in this verdict")
    if gate_name not in SINGLE_CALL_GATES and not gate.replay.dist.startswith(ATTACK_PREFIX):
        raise ValueError(f"{gate_name} needs repeated calls; reproduce it with: shingan verify --seed {verdict.seed}")
    if scan(kernel_path.read_text()):
        raise ValueError(f"{kernel_path} fails the static scan")

    problem = get_problem(verdict.problem)
    dtype = parse_dtype(verdict.dtype)
    inputs = _inputs(problem, gate, dtype, device, attacks_path)
    expected = expected_output(problem, inputs, dtype)
    out = load_kernel(kernel_path)(*inputs)
    mismatch = compare(out, expected, dtype)
    if not isinstance(out, torch.Tensor) or out.shape != expected.value.shape:
        return ReplayResult(gate, mismatch, None, [])
    return ReplayResult(gate, mismatch, normalized_error(out, expected, dtype), _worst(out, expected, dtype))


def _inputs(
    problem: Problem,
    gate: GateResult,
    dtype: torch.dtype,
    device: torch.device,
    attacks_path: Path | None,
) -> list[torch.Tensor]:
    replay = gate.replay
    if not replay.dist.startswith(ATTACK_PREFIX):
        return make_inputs(problem, replay.shape, dtype, replay.dist, replay.seed, device, replay.special, replay.layout)
    if attacks_path is None:
        raise ValueError("this failure came from an attack; pass the attacks file")
    attack_id = replay.dist.removeprefix(ATTACK_PREFIX)
    attack = next((a for a in attacks_from_json(attacks_path.read_text()) if a.attack_id == attack_id), None)
    if attack is None:
        raise ValueError(f"attack {attack_id!r} is not in {attacks_path}")
    return attack_inputs(attack, problem, replay.seed, dtype, device)


def _worst(out: torch.Tensor, expected: Expected, dtype: torch.dtype) -> list[Discrepancy]:
    ratio = error_ratio(out, expected, dtype)
    got = out.detach().cpu().double()
    count = min(WORST_ELEMENTS, ratio.numel())
    values, flat = ratio.flatten().topk(count)
    worst = []
    for units, position in zip(values.tolist(), flat.tolist()):
        if units == 0.0:
            break
        index = tuple(int(i) for i in torch.unravel_index(torch.tensor(position), ratio.shape))
        worst.append(Discrepancy(index, float(expected.value[index]), float(got[index]), units))
    return worst
