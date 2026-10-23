import pytest
import torch

from shingan.attacks import Attack, attacks_to_json, run_attack
from shingan.contracts import GateResult, Outcome, Replay, Status, Verdict
from shingan.gates import GateContext, correctness_gates
from shingan.problems import get_problem
from shingan.replay import replay_gate

CPU = torch.device("cpu")

SHAPE_SPECIALISED = """\
import torch


def run(x):
    if x.shape[0] != 4096:
        return torch.zeros_like(x)
    return torch.softmax(x, dim=-1)
"""

TRUNCATES_ROWS = """\
import torch


def run(x):
    out = torch.zeros_like(x)
    out[..., :4096] = torch.softmax(x[..., :4096], dim=-1)
    return out
"""


@pytest.fixture(autouse=True)
def skip_triton_requirement(monkeypatch):
    monkeypatch.setattr("shingan.replay.scan", lambda source: [])


def load(path):
    namespace = {}
    exec(path.read_text(), namespace)
    return namespace["run"]


def verdict_with(gates, seed=77):
    return Verdict("softmax", "float32", Outcome.FAKE, family=gates[-1].family, gates=gates, seed=seed)


def test_heldout_shape_failure_replays_with_worst_elements(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text(SHAPE_SPECIALISED)
    results = correctness_gates(GateContext(get_problem("softmax"), load(kernel), torch.float32, CPU, seed=77))
    failure = next(r for r in results if r.status is Status.FAIL)
    assert failure.gate == "G2"

    result = replay_gate(kernel, verdict_with([failure]), "G2", CPU)
    assert result.mismatch == failure.detail
    assert result.worst and result.worst[0].got == 0.0
    assert result.worst[0].tolerance_units > 1.0


def test_attack_failure_replays_from_the_attacks_file(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text(TRUNCATES_ROWS)
    generator = "import torch\n\ndef generate(seed):\n    g = torch.Generator().manual_seed(seed)\n    return [torch.randn(2, 5000, generator=g, dtype=torch.float64)]\n"
    attack = Attack("long-rows", "C3", "rows longer than 4096 are truncated", {"M": 2, "N": 5000}, generator)
    ctx = GateContext(get_problem("softmax"), load(kernel), torch.float32, CPU, seed=1)
    failure = run_attack(ctx, attack, seed=9)
    assert failure.status is Status.FAIL
    attacks = tmp_path / "attacks.json"
    attacks.write_text(attacks_to_json([attack]))

    result = replay_gate(kernel, verdict_with([failure]), "B:long-rows", CPU, attacks)
    assert result.mismatch is not None
    assert all(d.index[1] >= 4096 for d in result.worst)


def test_multi_call_gates_point_to_a_seeded_rerun(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text(SHAPE_SPECIALISED)
    failure = GateResult("G4", Status.FAIL, "C6", "repeated runs disagree", Replay(1, {"M": 4, "N": 8}, "float32", "normal"))
    with pytest.raises(ValueError, match="--seed 77"):
        replay_gate(kernel, verdict_with([failure]), "G4", CPU)

