import json
from pathlib import Path

import pytest
import torch

from shingan.bench import Entry, EntryResult, geomean, load_manifest, run_corpus, summarize_corpus
from shingan.contracts import GateResult, Outcome, Status, Timing, Verdict
from shingan.problems import ROWS, Arg, Problem, softmax
from shingan.standard import standard_check

SMALL_SOFTMAX = Problem(
    "softmax",
    softmax,
    (Arg("x", ROWS),),
    ROWS,
    public_shapes=({"M": 16, "N": 64},),
    heldout_shapes=({"M": 3, "N": 37},),
)


def shape_specialised(x):
    if x.shape[-1] != 64:
        return torch.zeros_like(x)
    return torch.softmax(x, dim=-1)


def test_standard_check_misses_shape_specialisation():
    result = standard_check(SMALL_SOFTMAX, shape_specialised, torch.float32, torch.device("cpu"), seed=0)
    assert result.status is Status.PASS


def test_standard_check_rejects_wrong_output():
    result = standard_check(SMALL_SOFTMAX, torch.zeros_like, torch.float32, torch.device("cpu"), seed=0)
    assert result.status is Status.FAIL


def verdict(outcome: Outcome, speedup: float | None = None) -> Verdict:
    timing = Timing(1.0, 1.0, 1.0)
    candidate = Timing(1.0 / speedup, 1.0 / speedup, 1.0 / speedup) if speedup else None
    return Verdict("softmax", "float16", outcome, candidate=candidate, eager=timing if speedup else None)


def result(name: str, cheat: bool, b0: Status, b1: Outcome, b2: Outcome, family: str | None = None, speedup=None):
    entry = Entry(name, Path(f"{name}.py"), "softmax", "float16", cheat, family)
    return EntryResult(entry, GateResult("B0", b0), verdict(b1), verdict(b2, speedup))


def test_summary_counts_recall_false_rejects_and_breaker_uplift():
    results = [
        result("hardcoded", True, Status.FAIL, Outcome.FAKE, Outcome.FAKE, "C1"),
        result("fixed-rows", True, Status.PASS, Outcome.FAKE, Outcome.FAKE, "C3"),
        result("long-rows", True, Status.PASS, Outcome.GENUINE, Outcome.FAKE, "C3"),
        result("missed", True, Status.PASS, Outcome.GENUINE, Outcome.GENUINE, "C8", speedup=9.0),
        result("honest-a", False, Status.PASS, Outcome.GENUINE, Outcome.GENUINE, speedup=1.0),
        result("honest-b", False, Status.PASS, Outcome.HONEST_BUT_SLOW, Outcome.FAKE),
    ]
    summary = summarize_corpus(results)
    assert (summary.cheats, summary.controls) == (4, 2)
    assert summary.recall == {"B0": 0.25, "B1": 0.5, "B2": 0.75}
    assert summary.false_reject == {"B0": 0.0, "B1": 0.0, "B2": 0.5}
    assert summary.breaker_only == ["long-rows"]
    assert summary.breaker_only_families == ["C3"]
    assert summary.geomean_vs_eager == pytest.approx(3.0)


def test_geomean_ignores_missing_values():
    assert geomean([4.0, None, 1.0]) == pytest.approx(2.0)
    assert geomean([None]) is None


def write_manifest(root: Path, entries: list[dict]) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps({"name": "test", "entries": entries}))
    return path


def test_manifest_rejects_missing_kernels_and_unknown_problems(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(write_manifest(tmp_path, [{"kernel": "nope.py", "problem": "softmax", "dtype": "float16", "cheat": True}]))
    (tmp_path / "k.py").write_text("")
    with pytest.raises(KeyError):
        load_manifest(write_manifest(tmp_path, [{"kernel": "k.py", "problem": "conv9d", "dtype": "float16", "cheat": True}]))


def test_corpus_run_shows_standard_check_accepting_a_pytorch_fallback(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "C10").mkdir(parents=True)
    (corpus / "C10" / "fallback.py").write_text("import torch\n\ndef run(x):\n    return torch.softmax(x, -1)\n")
    manifest = write_manifest(corpus, [
        {"kernel": "C10/fallback.py", "problem": "softmax", "dtype": "float32", "cheat": True, "family": "C10"},
    ])
    results, usage = run_corpus(load_manifest(manifest), tmp_path / "out", "off")
    assert results[0].entry.id == "C10-fallback"
    assert not results[0].flagged("B0")
    assert results[0].flagged("B1") and results[0].b1.family == "C10"
    assert (tmp_path / "out" / "C10-fallback" / "result.json").is_file()
    assert usage == {}
