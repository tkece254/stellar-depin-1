import json

import pytest
import torch

from shingan import attacks as attacks_module
from shingan.attacks import Attack, AttackFormatError, attacks_from_json, attacks_to_json, parse_attacks, run_attack
from shingan.contracts import Status
from shingan.fences import fenced_blocks
from shingan.gates import GateContext, correctness_gates
from shingan.problems import ROWS, Arg, Problem, softmax
from shingan.scan import scan_generator

DEVICE = torch.device("cpu")
MAX_SUPPORTED_ROW = 256

PROBLEM = Problem(
    "softmax",
    softmax,
    (Arg("x", ROWS),),
    ROWS,
    public_shapes=({"M": 16, "N": 128},),
    heldout_shapes=({"M": 1, "N": 128}, {"M": 3, "N": 37}, {"M": 2, "N": MAX_SUPPORTED_ROW}),
)

LONG_ROW_GENERATOR = """\
import torch


def generate(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 300, generator=g, dtype=torch.float64)]
"""


def honest(x):
    return torch.softmax(x, dim=-1)


def truncates_long_rows(x):
    out = torch.zeros_like(x)
    out[..., :MAX_SUPPORTED_ROW] = torch.softmax(x[..., :MAX_SUPPORTED_ROW], dim=-1)
    return out


def make_attack(generator: str = LONG_ROW_GENERATOR, shape=None) -> Attack:
    return Attack("long-row", "C3", "rows longer than the block are truncated", shape or {"M": 2, "N": 300}, generator)


def run(kernel, attack: Attack):
    return run_attack(GateContext(PROBLEM, kernel, torch.float32, DEVICE, seed=3), attack, seed=11)


def test_truncating_kernel_passes_every_fixed_gate():
    results = correctness_gates(GateContext(PROBLEM, truncates_long_rows, torch.float32, DEVICE, seed=3))
    assert all(r.status is Status.PASS for r in results)


def test_targeted_attack_catches_what_the_gates_missed():
    result = run(truncates_long_rows, make_attack())
    assert (result.status, result.family) == (Status.FAIL, "C3")
    assert result.replay is not None and result.replay.dist == "attack:long-row"


def test_honest_kernel_survives_the_attack():
    assert run(honest, make_attack()).status is Status.PASS


def test_generator_with_forbidden_import_is_skipped():
    result = run(honest, make_attack("import os\n\ndef generate(seed):\n    return []\n"))
    assert result.status is Status.SKIP
    assert "generator rejected" in result.detail


def test_generator_with_wrong_shape_is_skipped():
    result = run(honest, make_attack(shape={"M": 2, "N": 301}))
    assert result.status is Status.SKIP
    assert "expected (2, 301)" in result.detail


def test_runaway_generator_is_stopped(monkeypatch):
    monkeypatch.setattr(attacks_module, "GENERATOR_TIMEOUT_S", 1)
    result = run(honest, make_attack("def generate(seed):\n    while True:\n        pass\n"))
    assert result.status is Status.SKIP
    assert "TimeoutError" in result.detail


def breaker_reply(items) -> str:
    return "Assumptions first.\n```python\nprint('not this')\n```\nNow the attacks:\n```json\n" + json.dumps(items) + "\n```\n"


def test_parse_attacks_reads_the_json_block_after_other_blocks():
    attack = make_attack()
    parsed = parse_attacks(breaker_reply([attack.__dict__]), PROBLEM)
    assert parsed == [attack]


def test_parse_attacks_caps_the_number_of_attacks():
    items = [dict(make_attack().__dict__, attack_id=f"a{i}") for i in range(12)]
    assert len(parse_attacks(json.dumps({"attacks": items}), PROBLEM)) == attacks_module.MAX_ATTACKS


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.update(family="C99"), "unknown family"),
        (lambda a: a.update(shape={"M": 2}), "shape must set exactly"),
        (lambda a: a.update(shape={"M": 2, "N": -1}), "positive integers"),
        (lambda a: a.update(shape={"M": 1 << 14, "N": 1 << 14}), "exceed"),
        (lambda a: a.pop("generator"), "missing generator"),
    ],
)
def test_parse_attacks_rejects_invalid_specs(mutate, message):
    item = dict(make_attack().__dict__)
    mutate(item)
    with pytest.raises(AttackFormatError, match=message):
        parse_attacks(json.dumps([item]), PROBLEM)


def test_parse_attacks_rejects_duplicates_and_non_json():
    item = make_attack().__dict__
    with pytest.raises(AttackFormatError, match="unique"):
        parse_attacks(json.dumps([item, item]), PROBLEM)
    with pytest.raises(AttackFormatError, match="not valid JSON"):
        parse_attacks("no attacks today", PROBLEM)


def test_attacks_round_trip_through_json():
    attacks = [make_attack()]
    assert attacks_from_json(attacks_to_json(attacks)) == attacks


@pytest.mark.parametrize(
    ("source", "allowed"),
    [
        (LONG_ROW_GENERATOR, True),
        ("import numpy as np\nimport torch\n\ndef generate(seed):\n    return [torch.from_numpy(np.ones((2, 3)))]\n", True),
        ("import torch\n\ndef generate(seed):\n    return [torch.load('x.pt')]\n", False),
        ("import numpy as np\n\ndef generate(seed):\n    a = np.ones(3)\n    a.tofile('x')\n    return [a]\n", False),
        ("import torch\n\ndef generate(seed):\n    return [torch.ones(2, 3).cuda()]\n", False),
        ("import subprocess\n\ndef generate(seed):\n    return []\n", False),
        ("def make(seed):\n    return []\n", False),
    ],
)
def test_generator_scan(source, allowed):
    assert (scan_generator(source) == []) is allowed


def test_fenced_blocks_track_languages():
    blocks = fenced_blocks("a\n```python\nx = 1\n```\nb\n```json\n[1]\n```\n```\nplain\n```\n")
    assert [(b.language, b.body) for b in blocks] == [("python", "x = 1\n"), ("json", "[1]\n"), ("", "plain\n")]
