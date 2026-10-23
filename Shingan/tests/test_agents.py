import json
from collections import defaultdict, deque

import httpx
import pytest

from shingan.attacks import parse_attacks
from shingan.breaker import propose_attacks
from shingan.builder import build, feedback_for_builder
from shingan.contracts import GateResult, Outcome, Replay, Status, Verdict
from shingan.problems import get_problem
from shingan.tokenfactory import TokenFactory

BUILDER = "nvidia/nemotron-3-ultra-test"
BREAKER = "nvidia/nemotron-3-super-test"
TRIAGE = "nvidia/nemotron-3-nano-test"
MODELS = [BUILDER, BREAKER, TRIAGE, "other/llama-test"]

FALLBACK_KERNEL = "```python\nimport torch\n\ndef run(x):\n    return torch.softmax(x, -1)\n```"
BROKEN_KERNEL = """```python
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, n, BLOCK: tl.constexpr):
    pass


def run(x):
    return missing_helper(x)
```"""


class FakeTokenFactory:
    def __init__(self, replies: dict[str, list[str]], failures: int = 0) -> None:
        self.replies = {model: deque(texts) for model, texts in replies.items()}
        self.requests: dict[str, list[list[dict]]] = defaultdict(list)
        self.failures = failures

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in MODELS]})
        if self.failures:
            self.failures -= 1
            return httpx.Response(429, json={"error": "slow down"})
        body = json.loads(request.content)
        self.requests[body["model"]].append(body["messages"])
        text = self.replies[body["model"]].popleft()
        usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": usage})

    def client(self) -> TokenFactory:
        return TokenFactory("test-key", "https://tokenfactory.test/v1", transport=httpx.MockTransport(self.handle))


@pytest.fixture(autouse=True)
def no_model_overrides(monkeypatch):
    for role in ("BUILDER", "BREAKER", "TRIAGE"):
        monkeypatch.delenv(f"SHINGAN_{role}_MODEL", raising=False)
    monkeypatch.setattr("shingan.tokenfactory.time.sleep", lambda seconds: None)


def test_models_are_chosen_by_role_and_can_be_overridden(monkeypatch):
    client = FakeTokenFactory({}).client()
    assert [client.model_for(r) for r in ("builder", "breaker", "triage")] == [BUILDER, BREAKER, TRIAGE]
    monkeypatch.setenv("SHINGAN_BREAKER_MODEL", "custom/model")
    assert client.model_for("breaker") == "custom/model"


def test_rate_limited_requests_are_retried():
    fake = FakeTokenFactory({BUILDER: ["ok"]}, failures=2)
    completion = fake.client().complete(BUILDER, [{"role": "user", "content": "hi"}])
    assert (completion.text, completion.prompt_tokens, completion.completion_tokens) == ("ok", 100, 10)


def test_build_loop_repairs_through_rules_errors_and_triage(tmp_path):
    fake = FakeTokenFactory({
        BUILDER: ["I will think first.", FALLBACK_KERNEL, BROKEN_KERNEL],
        TRIAGE: ["missing_helper is undefined; define it or inline the launch."],
    })
    run = build(get_problem("softmax"), "float32", fake.client(), tmp_path, max_attempts=3, device_name=None)

    assert [a.verdict.outcome if a.verdict else None for a in run.attempts] == [None, Outcome.FAKE, Outcome.ERROR]
    assert run.attempts[1].verdict.family == "C10"
    assert len(fake.requests[TRIAGE]) == 1
    assert BREAKER not in fake.requests
    assert run.usage[BUILDER].prompt == 300
    assert run.usage[TRIAGE].completion == 10

    transcript = json.loads((tmp_path / "transcript.json").read_text())
    feedback = [m["content"] for m in transcript if m["role"] == "user"][1:]
    assert "no ```python block" in feedback[0]
    assert "torch.softmax" in feedback[1]
    assert "missing_helper is undefined" in feedback[2]
    assert (tmp_path / "attempt-2" / "verdict.json").is_file()
    assert json.loads((tmp_path / "run.json").read_text())["attempts"][0]["kernel"] is None


def test_builder_feedback_never_reveals_breaker_inputs():
    verdict = Verdict(
        "softmax",
        "float16",
        Outcome.FAKE,
        family="C10",
        gates=[
            GateResult("G0", Status.PASS),
            GateResult("B:long-rows", Status.FAIL, "C10", "rows of 300 -> mismatch", Replay(5, {"M": 2, "N": 300}, "float16", "attack:long-rows")),
        ],
    )
    feedback = feedback_for_builder(verdict)
    assert "FAKE" in feedback and "C10" in feedback
    assert "300" not in feedback
    assert "long-rows" not in feedback


def test_breaker_gets_one_repair_round():
    problem = get_problem("softmax")
    valid = [{
        "attack_id": "long-rows",
        "family": "C3",
        "hypothesis": "BLOCK is capped at 1024",
        "shape": {"M": 2, "N": 3000},
        "generator": "import torch\n\ndef generate(seed):\n    return [torch.zeros(2, 3000, dtype=torch.float64)]\n",
    }]
    fake = FakeTokenFactory({BREAKER: ["not json at all", "```json\n" + json.dumps(valid) + "\n```"]})
    reply = propose_attacks(fake.client(), BREAKER, problem, "float16", "kernel source")
    assert reply.error is None
    assert reply.attacks == parse_attacks(json.dumps(valid), problem)
    assert len(reply.completions) == 2
    assert "invalid" in fake.requests[BREAKER][1][-1]["content"]


def test_breaker_gives_up_after_two_invalid_replies():
    fake = FakeTokenFactory({BREAKER: ["nope", "still nope"]})
    reply = propose_attacks(fake.client(), BREAKER, get_problem("softmax"), "float16", "kernel source")
    assert reply.attacks == []
    assert "not valid JSON" in reply.error
