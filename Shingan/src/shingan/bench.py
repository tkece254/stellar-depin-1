from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from shingan.breaker import ATTACKABLE, break_kernel
from shingan.builder import build
from shingan.contracts import GateResult, Outcome, Status, Verdict
from shingan.isolation import attack_isolated, standard_isolated, verify_isolated
from shingan.pipeline import apply_attacks
from shingan.problems import get_problem
from shingan.tokenfactory import TokenFactory, Usage, tally

BreakerMode = Literal["live", "replay", "off"]
CONFIGS = ("B0", "B1", "B2")
FLAGGED = frozenset({Outcome.FAKE, Outcome.ERROR})


@dataclass(frozen=True)
class Entry:
    id: str
    kernel: Path
    problem: str
    dtype: str
    cheat: bool
    family: str | None = None
    note: str = ""


def load_manifest(path: Path) -> list[Entry]:
    data = json.loads(path.read_text())
    entries = []
    for item in data["entries"]:
        kernel = path.parent / item["kernel"]
        if not kernel.is_file():
            raise FileNotFoundError(f"manifest entry {item['kernel']} does not exist")
        get_problem(item["problem"])
        entries.append(
            Entry(
                id=item["kernel"].removesuffix(".py").replace("/", "-"),
                kernel=kernel,
                problem=item["problem"],
                dtype=item["dtype"],
                cheat=bool(item["cheat"]),
                family=item.get("family"),
                note=item.get("note", ""),
            )
        )
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest kernel paths must be unique")
    return entries


@dataclass
class EntryResult:
    entry: Entry
    b0: GateResult
    b1: Verdict
    b2: Verdict

    def flagged(self, config: str) -> bool:
        match config:
            case "B0":
                return self.b0.status is Status.FAIL
            case "B1":
                return self.b1.outcome in FLAGGED
            case "B2":
                return self.b2.outcome in FLAGGED
        raise ValueError(f"unknown config {config!r}")

    def to_dict(self) -> dict[str, Any]:
        entry = asdict(self.entry)
        entry["kernel"] = str(self.entry.kernel)
        return {
            "entry": entry,
            "flagged": {c: self.flagged(c) for c in CONFIGS},
            "b0": asdict(self.b0),
            "b1": self.b1.to_dict(),
            "b2": self.b2.to_dict(),
        }


@dataclass(frozen=True)
class CorpusSummary:
    cheats: int
    controls: int
    recall: dict[str, float | None]
    false_reject: dict[str, float | None]
    breaker_only: list[str]
    breaker_only_families: list[str]
    geomean_vs_eager: float | None
    geomean_vs_compiled: float | None


def run_corpus(
    entries: Sequence[Entry],
    out_dir: Path,
    breaker: BreakerMode,
    client: TokenFactory | None = None,
    replay_dir: Path | None = None,
) -> tuple[list[EntryResult], dict[str, Usage]]:
    usage: dict[str, Usage] = {}
    results = []
    for entry in entries:
        entry_dir = out_dir / entry.id
        entry_dir.mkdir(parents=True, exist_ok=True)
        b0 = standard_isolated(entry.problem, entry.kernel, entry.dtype, entry_dir / "standard.json")
        b1 = verify_isolated(entry.problem, entry.kernel, entry.dtype, entry_dir / "verdict.json")
        b2 = Verdict.from_dict(b1.to_dict())
        if breaker == "live":
            if client is None:
                raise ValueError("live breaker needs a Token Factory client")
            b2, calls = break_kernel(client, get_problem(entry.problem), entry.kernel, entry.dtype, b2, entry_dir)
            for call in calls:
                tally(usage, call)
        elif breaker == "replay":
            if replay_dir is None:
                raise ValueError("replay breaker needs a replay directory")
            attacks = replay_dir / entry.id / "attacks.json"
            if b2.outcome in ATTACKABLE and attacks.is_file():
                results_path = entry_dir / "attack-results.json"
                b2 = apply_attacks(b2, attack_isolated(entry.problem, entry.kernel, entry.dtype, attacks, results_path))
        result = EntryResult(entry, b0, b1, b2)
        (entry_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
        results.append(result)
    return results, usage


def summarize_corpus(results: Sequence[EntryResult]) -> CorpusSummary:
    cheats = [r for r in results if r.entry.cheat]
    controls = [r for r in results if not r.entry.cheat]
    breaker_only = [r for r in cheats if r.flagged("B2") and not r.flagged("B1")]
    genuine = [r.b2 for r in results if r.b2.outcome is Outcome.GENUINE]
    return CorpusSummary(
        cheats=len(cheats),
        controls=len(controls),
        recall={c: _flag_rate(cheats, c) for c in CONFIGS},
        false_reject={c: _flag_rate(controls, c) for c in CONFIGS},
        breaker_only=[r.entry.id for r in breaker_only],
        breaker_only_families=sorted({r.entry.family or "unlabelled" for r in breaker_only}),
        geomean_vs_eager=geomean(v.speedup_vs_eager for v in genuine),
        geomean_vs_compiled=geomean(v.speedup_vs_compiled for v in genuine),
    )


@dataclass
class ProblemOutcome:
    problem: str
    final: Verdict | None
    attempts: list[tuple[str, GateResult, Verdict]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "final": self.final.to_dict() if self.final else None,
            "attempts": [{"kernel": k, "b0": asdict(b0), "verdict": v.to_dict()} for k, b0, v in self.attempts],
        }


@dataclass(frozen=True)
class CampaignSummary:
    problems: int
    genuine: int
    builder_success: float | None
    fakes_accepted_by_standard_check: list[dict[str, Any]]
    geomean_vs_eager: float | None
    geomean_vs_compiled: float | None
    above_compile: int


def run_campaign(
    problems: Sequence[str],
    dtype: str,
    attempts: int,
    client: TokenFactory,
    out_dir: Path,
    device_name: str | None,
) -> tuple[list[ProblemOutcome], dict[str, Usage]]:
    usage: dict[str, Usage] = {}
    outcomes = []
    for name in problems:
        run = build(get_problem(name), dtype, client, out_dir / name, attempts, device_name)
        for model, totals in run.usage.items():
            usage.setdefault(model, Usage()).add(totals.prompt, totals.completion)
        outcome = ProblemOutcome(name, run.final)
        for attempt in run.attempts:
            if attempt.kernel_path is None or attempt.verdict is None:
                continue
            b0 = standard_isolated(name, attempt.kernel_path, dtype, attempt.kernel_path.with_name("standard.json"))
            outcome.attempts.append((str(attempt.kernel_path), b0, attempt.verdict))
        outcomes.append(outcome)
    return outcomes, usage


def summarize_campaign(outcomes: Sequence[ProblemOutcome]) -> CampaignSummary:
    finals = [o.final for o in outcomes if o.final is not None and o.final.outcome is Outcome.GENUINE]
    gap = [
        {
            "kernel": kernel,
            "family": verdict.family,
            "claimed_vs_eager": verdict.speedup_vs_eager,
        }
        for o in outcomes
        for kernel, b0, verdict in o.attempts
        if b0.status is Status.PASS and verdict.outcome is Outcome.FAKE
    ]
    return CampaignSummary(
        problems=len(outcomes),
        genuine=len(finals),
        builder_success=len(finals) / len(outcomes) if outcomes else None,
        fakes_accepted_by_standard_check=gap,
        geomean_vs_eager=geomean(v.speedup_vs_eager for v in finals),
        geomean_vs_compiled=geomean(v.speedup_vs_compiled for v in finals),
        above_compile=sum(1 for v in finals if (v.speedup_vs_compiled or 0.0) > 1.0),
    )


def geomean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None and v > 0]
    if not present:
        return None
    return math.exp(sum(math.log(v) for v in present) / len(present))


def _flag_rate(results: Sequence[EntryResult], config: str) -> float | None:
    if not results:
        return None
    return sum(r.flagged(config) for r in results) / len(results)
