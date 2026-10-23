import json

import torch

from shingan.builder import extract_code, feedback_for_builder
from shingan.contracts import GateResult, Outcome, Replay, Status, Timing, Verdict
from shingan.numerics import Expected, compare
from shingan.pipeline import decide


def sample_verdict() -> Verdict:
    return Verdict(
        problem="softmax",
        dtype="float16",
        outcome=Outcome.FAKE,
        family="C3",
        gates=[
            GateResult("G0", Status.PASS),
            GateResult("G2", Status.FAIL, "C3", "12 of 4096 elements outside tolerance", Replay(99, {"M": 3, "N": 7}, "float16", "normal")),
        ],
        candidate=Timing(0.10, 0.09, 0.11),
        eager=Timing(0.20, 0.19, 0.21),
        environment={"device": "cpu"},
    )


def test_verdict_round_trips_through_json():
    verdict = sample_verdict()
    restored = Verdict.from_dict(json.loads(json.dumps(verdict.to_dict())))
    assert restored == verdict
    assert restored.speedup_vs_eager == 2.0


def test_decide():
    verdict = sample_verdict()
    assert decide(verdict) == (Outcome.FAKE, "C3")
    verdict.gates = [GateResult("G0", Status.PASS)]
    assert decide(verdict) == (Outcome.GENUINE, None)
    verdict.candidate = Timing(0.4, 0.4, 0.4)
    assert decide(verdict) == (Outcome.HONEST_BUT_SLOW, None)
    verdict.candidate = None
    assert decide(verdict) == (Outcome.UNVERIFIED, None)


def test_builder_feedback_hides_failing_inputs():
    feedback = feedback_for_builder(sample_verdict())
    assert "C3" in feedback
    assert "12 of 4096" not in feedback
    assert "99" not in feedback


def test_extract_code_takes_last_block():
    text = "plan\n```python\nfirst\n```\nthen\n```python\nsecond\n```\n"
    assert extract_code(text) == "second\n"
    assert extract_code("no code here") is None


def test_compare_requires_matching_nonfinite_positions():
    expected = Expected.of(torch.tensor([1.0, float("nan"), float("inf")], dtype=torch.float64))
    assert compare(torch.tensor([1.0, float("nan"), float("inf")]), expected, torch.float32) is None
    assert "NaN" in compare(torch.tensor([1.0, 0.0, float("inf")]), expected, torch.float32)
    assert "+Inf" in compare(torch.tensor([1.0, float("nan"), 1e30]), expected, torch.float32)
