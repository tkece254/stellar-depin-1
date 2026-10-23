from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from shingan.attacks import Attack, AttackFormatError, attacks_to_json, parse_attacks
from shingan.contracts import GateResult, Outcome, Status, Verdict
from shingan.isolation import attack_isolated
from shingan.pipeline import apply_attacks
from shingan.problems import Problem
from shingan.prompts import breaker_system, breaker_task
from shingan.tokenfactory import Completion, Message, TokenFactory

ATTACKABLE = frozenset({Outcome.GENUINE, Outcome.HONEST_BUT_SLOW, Outcome.UNVERIFIED})
BREAKER_TEMPERATURE = 0.6


@dataclass
class BreakerReply:
    attacks: list[Attack] = field(default_factory=list)
    completions: list[Completion] = field(default_factory=list)
    error: str | None = None


def propose_attacks(client: TokenFactory, model: str, problem: Problem, dtype: str, source: str) -> BreakerReply:
    reply = BreakerReply()
    messages: list[Message] = [
        {"role": "system", "content": breaker_system(problem)},
        {"role": "user", "content": breaker_task(problem, dtype, source)},
    ]
    for _ in range(2):
        completion = client.complete(model, messages, temperature=BREAKER_TEMPERATURE)
        reply.completions.append(completion)
        try:
            reply.attacks = parse_attacks(completion.text, problem)
            reply.error = None
            return reply
        except AttackFormatError as exc:
            reply.error = str(exc)
            messages += [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"That reply was invalid: {exc}. Send the corrected ```json block only."},
            ]
    return reply


def break_kernel(
    client: TokenFactory,
    problem: Problem,
    kernel_path: Path,
    dtype: str,
    verdict: Verdict,
    out_dir: Path,
) -> tuple[Verdict, list[Completion]]:
    if verdict.outcome not in ATTACKABLE:
        return verdict, []
    reply = propose_attacks(client, client.model_for("breaker"), problem, dtype, kernel_path.read_text())
    if not reply.attacks:
        verdict.gates.append(GateResult("B", Status.SKIP, detail=f"breaker produced no usable attacks: {reply.error}"))
        return verdict, reply.completions
    attacks_path = out_dir / "attacks.json"
    attacks_path.write_text(attacks_to_json(reply.attacks))
    results = attack_isolated(problem.name, kernel_path, dtype, attacks_path, out_dir / "attack-results.json")
    return apply_attacks(verdict, results), reply.completions
