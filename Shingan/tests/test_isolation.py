import importlib.util

import pytest

from shingan.contracts import Outcome
from shingan.isolation import verify_isolated
from shingan.prompts import TEMPLATE_KERNEL


def test_pytorch_fallback_is_rejected_before_execution(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("import torch\n\ndef run(x):\n    return torch.softmax(x, -1)\n")
    verdict = verify_isolated("softmax", kernel, "float32", tmp_path / "verdict.json")
    assert (verdict.outcome, verdict.family) == (Outcome.FAKE, "C10")
    assert verdict.gates[0].gate == "G0"
    assert (tmp_path / "verdict.json").exists()


@pytest.mark.skipif(importlib.util.find_spec("triton") is not None, reason="needs an environment without Triton")
def test_load_failure_is_reported_as_error(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text(TEMPLATE_KERNEL)
    verdict = verify_isolated("add_relu", kernel, "float32", tmp_path / "verdict.json")
    assert verdict.outcome is Outcome.ERROR
    assert "triton" in verdict.error
