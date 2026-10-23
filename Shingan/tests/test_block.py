import json
from dataclasses import replace

import torch

from shingan.block import BLOCK_OPS, BlockConfig, Ops, TransformerBlock, evaluate
from shingan.contracts import Outcome, Timing, Verdict
from shingan.patch import KernelTable

CPU = torch.device("cpu")
TINY = BlockConfig(seq=32, hidden=64, heads=4, ffn=96)


def test_reference_ops_reproduce_the_block():
    report = evaluate(TransformerBlock.random(TINY, seed=0), Ops.reference(), torch.float32, CPU, seed=1)
    assert report.correct
    assert report.patched_error == report.original_error
    assert {name for name, _ in report.profile} == set(BLOCK_OPS)
    assert report.speedup is None


def test_wrong_op_makes_the_block_incorrect():
    broken = replace(Ops.reference(), softmax=lambda x: torch.full_like(x, 1.0 / x.shape[-1]))
    report = evaluate(TransformerBlock.random(TINY, seed=0), broken, torch.float32, CPU, seed=1)
    assert not report.correct


def write_run(run_dir, attempts):
    run_dir.mkdir(parents=True)
    entries = []
    for index, (outcome, source) in enumerate(attempts, start=1):
        kernel = run_dir / f"attempt-{index}" / "kernel.py"
        kernel.parent.mkdir()
        kernel.write_text(source)
        verdict = Verdict("softmax", "float32", outcome, candidate=Timing(1, 1, 1), eager=Timing(2, 2, 2))
        entries.append({"index": index, "kernel": str(kernel), "verdict": verdict.to_dict()})
    (run_dir / "run.json").write_text(json.dumps({"problem": "softmax", "dtype": "float32", "attempts": entries, "usage": {}}))


def test_table_loads_only_genuine_kernels(tmp_path):
    write_run(tmp_path / "good", [
        (Outcome.FAKE, "def run(x):\n    raise AssertionError\n"),
        (Outcome.GENUINE, "import torch\n\ndef run(x):\n    return torch.softmax(x, -1)\n"),
    ])
    write_run(tmp_path / "bad", [(Outcome.FAKE, "def run(x):\n    return x\n")])
    table = KernelTable.from_runs([tmp_path / "good", tmp_path / "bad"])
    assert list(table.kernels) == [("softmax", torch.float32)]
    verified = table.kernels[("softmax", torch.float32)]
    assert verified.path.parent.name == "attempt-2"
    assert verified.speedup_vs_eager == 2.0


def test_patched_op_falls_back_to_reference_off_gpu(tmp_path):
    write_run(tmp_path / "run", [(Outcome.GENUINE, "def run(x):\n    raise AssertionError('kernel called on CPU')\n")])
    op = KernelTable.from_runs([tmp_path / "run"]).op("softmax")
    x = torch.randn(4, 8)
    assert torch.equal(op(x), torch.softmax(x, -1))
