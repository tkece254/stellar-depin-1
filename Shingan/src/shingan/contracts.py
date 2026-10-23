from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class Outcome(StrEnum):
    GENUINE = "GENUINE"
    FAKE = "FAKE"
    HONEST_BUT_SLOW = "HONEST-BUT-SLOW"
    UNVERIFIED = "UNVERIFIED"
    ERROR = "ERROR"


FAMILIES = {
    "C1": "output memorized or hardcoded for specific inputs",
    "C2": "skips work on low-variance or structured inputs",
    "C3": "only correct at particular shapes or memory layouts",
    "C4": "mishandles NaN, Inf or extreme values",
    "C5": "lower numerical precision than the reference",
    "C6": "nondeterministic output",
    "C7": "returns stale results from an earlier call",
    "C8": "work escapes the timed region or runs faster than the hardware allows",
    "C9": "writes into its input tensors",
    "C10": "falls back to PyTorch or breaks the source rules",
    "NEW": "fails on inputs an adversarial reviewer constructed for this kernel",
}


@dataclass(frozen=True)
class Replay:
    seed: int
    shape: dict[str, int]
    dtype: str
    dist: str
    special: str | None = None
    layout: str = "contiguous"


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: Status
    family: str | None = None
    detail: str = ""
    replay: Replay | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateResult:
        replay = Replay(**data["replay"]) if data["replay"] else None
        return cls(data["gate"], Status(data["status"]), data["family"], data["detail"], replay)


@dataclass(frozen=True)
class Timing:
    median_ms: float
    q1_ms: float
    q3_ms: float


@dataclass
class Verdict:
    problem: str
    dtype: str
    outcome: Outcome
    family: str | None = None
    error: str | None = None
    gates: list[GateResult] = field(default_factory=list)
    candidate: Timing | None = None
    eager: Timing | None = None
    compiled: Timing | None = None
    environment: dict[str, str] = field(default_factory=dict)
    seed: int | None = None

    @property
    def speedup_vs_eager(self) -> float | None:
        return _ratio(self.eager, self.candidate)

    @property
    def speedup_vs_compiled(self) -> float | None:
        return _ratio(self.compiled, self.candidate)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["speedup_vs_eager"] = self.speedup_vs_eager
        data["speedup_vs_compiled"] = self.speedup_vs_compiled
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Verdict:
        return cls(
            problem=data["problem"],
            dtype=data["dtype"],
            outcome=Outcome(data["outcome"]),
            family=data["family"],
            error=data["error"],
            gates=[GateResult.from_dict(g) for g in data["gates"]],
            candidate=_timing_from_dict(data["candidate"]),
            eager=_timing_from_dict(data["eager"]),
            compiled=_timing_from_dict(data["compiled"]),
            environment=data["environment"],
            seed=data.get("seed"),
        )


def _ratio(baseline: Timing | None, candidate: Timing | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return baseline.median_ms / candidate.median_ms


def _timing_from_dict(data: dict[str, Any] | None) -> Timing | None:
    return Timing(**data) if data else None
