from __future__ import annotations

import json
import math
import signal
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch

from shingan.contracts import FAMILIES, GateResult, Replay, Status
from shingan.fences import last_block
from shingan.gates import GateContext
from shingan.numerics import compare, expected_output
from shingan.problems import Problem, dtype_name
from shingan.scan import GENERATOR_ENTRY_POINT, scan_generator

MAX_ATTACKS = 8
GENERATOR_TIMEOUT_S = 20
MAX_ATTACK_ELEMENTS = 1 << 26
MAX_DETAIL_CHARS = 300
JSON_LANGUAGES = frozenset({"json", ""})
REQUIRED_FIELDS = ("attack_id", "family", "hypothesis", "shape", "generator")


class AttackFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Attack:
    attack_id: str
    family: str
    hypothesis: str
    shape: dict[str, int]
    generator: str

    @property
    def gate(self) -> str:
        return f"B:{self.attack_id}"


def parse_attacks(text: str, problem: Problem) -> list[Attack]:
    raw = last_block(text, JSON_LANGUAGES) or text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AttackFormatError(f"reply is not valid JSON: {exc}") from exc
    items = data.get("attacks") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise AttackFormatError('expected a non-empty JSON list of attacks, or {"attacks": [...]}')
    attacks = [_attack(item, index, problem) for index, item in enumerate(items[:MAX_ATTACKS])]
    ids = [a.attack_id for a in attacks]
    if len(set(ids)) != len(ids):
        raise AttackFormatError("attack_id values must be unique")
    return attacks


def _attack(item: Any, index: int, problem: Problem) -> Attack:
    if not isinstance(item, dict):
        raise AttackFormatError(f"attack {index} is not a JSON object")
    missing = [f for f in REQUIRED_FIELDS if f not in item]
    if missing:
        raise AttackFormatError(f"attack {index} is missing {', '.join(missing)}")
    if item["family"] not in FAMILIES:
        raise AttackFormatError(f"attack {index} has unknown family {item['family']!r}")
    shape = item["shape"]
    if not isinstance(shape, dict) or set(shape) != problem.dims:
        raise AttackFormatError(f"attack {index} shape must set exactly {', '.join(sorted(problem.dims))}")
    if not all(isinstance(v, int) and v > 0 for v in shape.values()):
        raise AttackFormatError(f"attack {index} shape sizes must be positive integers")
    largest = max(math.prod(arg.resolve(shape)) for arg in problem.args)
    if largest > MAX_ATTACK_ELEMENTS:
        raise AttackFormatError(f"attack {index} tensors exceed {MAX_ATTACK_ELEMENTS} elements")
    return Attack(str(item["attack_id"]), item["family"], str(item["hypothesis"]), shape, str(item["generator"]))


def attacks_to_json(attacks: Sequence[Attack]) -> str:
    return json.dumps([asdict(a) for a in attacks], indent=2)


def attacks_from_json(text: str) -> list[Attack]:
    return [Attack(**item) for item in json.loads(text)]


def run_attacks(ctx: GateContext, attacks: Sequence[Attack]) -> list[GateResult]:
    return [run_attack(ctx, attack, ctx.case_seed("B", index)) for index, attack in enumerate(attacks)]


def run_attack(ctx: GateContext, attack: Attack, seed: int) -> GateResult:
    findings = scan_generator(attack.generator)
    if findings:
        detail = "; ".join(f"line {f.line}: {f.message}" for f in findings)
        return GateResult(attack.gate, Status.SKIP, attack.family, f"generator rejected: {detail}")
    try:
        inputs = attack_inputs(attack, ctx.problem, seed, ctx.dtype, ctx.device)
    except Exception as exc:
        reason = "".join(traceback.format_exception_only(exc)).strip()[:MAX_DETAIL_CHARS]
        return GateResult(attack.gate, Status.SKIP, attack.family, f"generator failed: {reason}")
    expected = expected_output(ctx.problem, inputs, ctx.dtype)
    out, error = ctx.call(inputs)
    mismatch = error or compare(out, expected, ctx.dtype)
    replay = Replay(seed, dict(attack.shape), dtype_name(ctx.dtype), f"attack:{attack.attack_id}")
    if mismatch:
        return GateResult(attack.gate, Status.FAIL, attack.family, f"{attack.hypothesis} -> {mismatch}", replay)
    return GateResult(attack.gate, Status.PASS, attack.family, attack.hypothesis, replay)


def attack_inputs(
    attack: Attack, problem: Problem, seed: int, dtype: torch.dtype, device: torch.device
) -> list[torch.Tensor]:
    tensors = _validate(_generate(attack, seed), problem, attack.shape)
    return [t.to(dtype).to(device) for t in tensors]


def _generate(attack: Attack, seed: int) -> object:
    namespace: dict[str, Any] = {}
    with _time_limit(GENERATOR_TIMEOUT_S):
        exec(compile(attack.generator, f"<attack {attack.attack_id}>", "exec"), namespace)
        return namespace[GENERATOR_ENTRY_POINT](seed)


def _validate(tensors: object, problem: Problem, shape: dict[str, int]) -> list[torch.Tensor]:
    if not isinstance(tensors, (list, tuple)) or len(tensors) != len(problem.args):
        raise ValueError(f"generate() must return {len(problem.args)} tensors")
    validated = []
    for arg, tensor in zip(problem.args, tensors):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{arg.name} is {type(tensor).__name__}, not a tensor")
        if tuple(tensor.shape) != arg.resolve(shape):
            raise ValueError(f"{arg.name} has shape {tuple(tensor.shape)}, expected {arg.resolve(shape)}")
        validated.append(tensor.detach().cpu().to(torch.float64))
    return validated


@contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    def expire(signum: int, frame: object) -> None:
        raise TimeoutError(f"generator ran longer than {seconds}s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
