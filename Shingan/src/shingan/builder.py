from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shingan.breaker import break_kernel
from shingan.contracts import FAMILIES, Outcome, Status, Verdict
from shingan.fences import last_block
from shingan.isolation import verify_isolated
from shingan.problems import Problem
from shingan.prompts import BUILDER_SYSTEM, TRIAGE_SYSTEM, builder_task, triage_task
from shingan.tokenfactory import Completion, Message, TokenFactory, Usage, tally

CODE_LANGUAGES = frozenset({"python", "py", ""})
FINAL_OUTCOMES = frozenset({Outcome.GENUINE, Outcome.UNVERIFIED})


@dataclass
class Attempt:
    index: int
    kernel_path: Path | None
    verdict: Verdict | None


@dataclass
class BuildRun:
    problem: str
    dtype: str
    attempts: list[Attempt] = field(default_factory=list)
    usage: dict[str, Usage] = field(default_factory=dict)

    def record(self, completion: Completion) -> None:
        tally(self.usage, completion)

    @property
    def final(self) -> Verdict | None:
        return next((a.verdict for a in reversed(self.attempts) if a.verdict), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "problem": self.problem,
            "dtype": self.dtype,
            "attempts": [
                {
                    "index": a.index,
                    "kernel": str(a.kernel_path) if a.kernel_path else None,
                    "verdict": a.verdict.to_dict() if a.verdict else None,
                }
                for a in self.attempts
            ],
            "usage": {model: asdict(totals) for model, totals in self.usage.items()},
        }


def build(
    problem: Problem,
    dtype: str,
    client: TokenFactory,
    run_dir: Path,
    max_attempts: int,
    device_name: str | None,
) -> BuildRun:
    builder_model = client.model_for("builder")
    triage_model = client.model_for("triage")
    run = BuildRun(problem.name, dtype)
    messages: list[Message] = [
        {"role": "system", "content": BUILDER_SYSTEM},
        {"role": "user", "content": builder_task(problem, dtype, device_name)},
    ]
    run_dir.mkdir(parents=True, exist_ok=True)

    for index in range(1, max_attempts + 1):
        completion = client.complete(builder_model, messages)
        run.record(completion)
        messages.append({"role": "assistant", "content": completion.text})

        source = extract_code(completion.text)
        if source is None:
            run.attempts.append(Attempt(index, None, None))
            messages.append({"role": "user", "content": "Your reply had no ```python block. Send the complete file."})
            continue

        attempt_dir = run_dir / f"attempt-{index}"
        attempt_dir.mkdir(exist_ok=True)
        kernel_path = attempt_dir / "kernel.py"
        kernel_path.write_text(source)
        verdict_path = attempt_dir / "verdict.json"
        verdict = verify_isolated(problem.name, kernel_path, dtype, verdict_path)
        verdict, breaker_calls = break_kernel(client, problem, kernel_path, dtype, verdict, attempt_dir)
        for call in breaker_calls:
            run.record(call)
        verdict_path.write_text(json.dumps(verdict.to_dict(), indent=2))
        run.attempts.append(Attempt(index, kernel_path, verdict))
        if verdict.outcome in FINAL_OUTCOMES:
            break

        feedback = feedback_for_builder(verdict)
        if verdict.outcome is Outcome.ERROR and verdict.error:
            hint = client.complete(triage_model, [
                {"role": "system", "content": TRIAGE_SYSTEM},
                {"role": "user", "content": triage_task(verdict.error, source)},
            ], max_tokens=1024)
            run.record(hint)
            feedback += f"\n\nLikely cause and fix:\n{hint.text.strip()}"
        messages.append({"role": "user", "content": feedback})

    (run_dir / "transcript.json").write_text(json.dumps(messages, indent=2))
    (run_dir / "run.json").write_text(json.dumps(run.to_dict(), indent=2))
    return run


def extract_code(text: str) -> str | None:
    return last_block(text, CODE_LANGUAGES)


def feedback_for_builder(verdict: Verdict) -> str:
    if verdict.outcome is Outcome.ERROR:
        return f"The kernel failed before any test ran:\n\n{verdict.error}"
    scan_failure = next((g for g in verdict.gates if g.gate == "G0" and g.status is Status.FAIL), None)
    if scan_failure:
        return f"The source breaks the verifier rules: {scan_failure.detail}"
    if verdict.outcome is Outcome.FAKE:
        return (
            f"Verdict: FAKE, failure family {verdict.family} ({FAMILIES[verdict.family]}). "
            "The failing inputs are hidden. Fix the kernel so it is correct for every input and shape the reference accepts."
        )
    return (
        f"Verdict: correct, but {verdict.speedup_vs_eager:.2f}x the speed of PyTorch eager. "
        "Make it faster without changing its results."
    )
