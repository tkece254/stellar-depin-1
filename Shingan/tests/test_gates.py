import pytest
import torch

from shingan.contracts import Status
from shingan.gates import GateContext, correctness_gates, freshness, hardware_floor_ms
from shingan.hardware import peak_for
from shingan.inputs import make_inputs
from shingan.problems import PROBLEMS, ROWS, Arg, Problem, softmax

DEVICE = torch.device("cpu")

SMALL_SOFTMAX = Problem(
    "softmax",
    softmax,
    (Arg("x", ROWS),),
    ROWS,
    public_shapes=({"M": 32, "N": 64},),
    heldout_shapes=({"M": 1, "N": 64}, {"M": 5, "N": 37}),
)


def gate_results(kernel, dtype=torch.float32):
    ctx = GateContext(SMALL_SOFTMAX, kernel, dtype, DEVICE, seed=1234)
    return correctness_gates(ctx)


def first_failure(results):
    return next((r for r in results if r.status is Status.FAIL), None)


def honest(x):
    return torch.softmax(x, dim=-1)


def shape_specialised(x):
    if x.shape[-1] != 64:
        return torch.zeros_like(x)
    return torch.softmax(x, dim=-1)


def hides_nonfinite(x):
    return torch.softmax(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), dim=-1)


def noisy(x):
    return torch.softmax(x, dim=-1) + 1e-9 * torch.randn_like(x)


def writes_into_input(x):
    x.copy_(torch.softmax(x, dim=-1))
    return x.clone()


_cache: dict[int, torch.Tensor] = {}


def caches_by_pointer(x):
    key = x.data_ptr()
    if key not in _cache:
        _cache[key] = torch.softmax(x, dim=-1)
    return _cache[key]


MEMORISED = torch.softmax(torch.randn(32, 64, generator=torch.Generator().manual_seed(0)), dim=-1)


def memorised(x):
    if x.shape == MEMORISED.shape:
        return MEMORISED.clone()
    return torch.softmax(x, dim=-1)


def skips_structured(x):
    if bool((x[:, 1:] >= x[:, :-1]).all()):
        return torch.full_like(x, 1.0 / x.shape[-1])
    return torch.softmax(x, dim=-1)


def low_precision(x):
    return torch.softmax(x.to(torch.bfloat16), dim=-1).to(x.dtype)


def returns_wrong_dtype(x):
    return torch.softmax(x, dim=-1).to(torch.float64)


def test_honest_kernel_passes_every_gate():
    results = gate_results(honest)
    assert [r.gate for r in results] == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    assert first_failure(results) is None


def test_honest_half_precision_kernel_passes():
    assert first_failure(gate_results(honest, torch.float16)) is None


@pytest.mark.parametrize(
    ("kernel", "gate", "family"),
    [
        (memorised, "G1", "C1"),
        (skips_structured, "G1", "C2"),
        (shape_specialised, "G2", "C3"),
        (hides_nonfinite, "G3", "C4"),
        (noisy, "G4", "C6"),
        (writes_into_input, "G7", "C9"),
        (returns_wrong_dtype, "G1", "C1"),
    ],
)
def test_cheat_is_caught_by_expected_gate(kernel, gate, family):
    failure = first_failure(gate_results(kernel))
    assert failure is not None
    assert (failure.gate, failure.family) == (gate, family)
    assert failure.replay is not None


def test_pointer_cache_is_caught_by_freshness():
    _cache.clear()
    result = freshness(GateContext(SMALL_SOFTMAX, caches_by_pointer, torch.float32, DEVICE, seed=7))
    assert (result.status, result.family) == (Status.FAIL, "C7")


def test_precision_downgrade_is_caught():
    failure = first_failure(gate_results(low_precision, torch.float16))
    assert failure is not None
    assert failure.family in {"C4", "C5"}


def test_seeds_rotate_between_runs():
    a = GateContext(SMALL_SOFTMAX, honest, torch.float32, DEVICE, seed=1)
    b = GateContext(SMALL_SOFTMAX, honest, torch.float32, DEVICE, seed=2)
    assert a.case_seed("G1", 0) != b.case_seed("G1", 0)


SMALL_MATMUL = Problem(
    "matmul",
    PROBLEMS["matmul"].reference,
    PROBLEMS["matmul"].args,
    PROBLEMS["matmul"].out_dims,
    public_shapes=({"M": 64, "N": 64, "K": 1024},),
    heldout_shapes=({"M": 3, "N": 5, "K": 7},),
    error_scale=PROBLEMS["matmul"].error_scale,
)


def fp32_accumulate(a, b):
    return (a.float() @ b.float()).to(a.dtype)


def tf32_inputs(a, b):
    def truncate(t):
        return (t.contiguous().view(torch.int32) & ~0x1FFF).view(torch.float32)

    return truncate(a) @ truncate(b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_honest_matmul_is_not_rejected_for_cancellation(dtype):
    ctx = GateContext(SMALL_MATMUL, fp32_accumulate, dtype, DEVICE, seed=5)
    assert first_failure(correctness_gates(ctx)) is None


def test_tf32_downgrade_is_caught():
    failure = first_failure(correctness_gates(GateContext(SMALL_MATMUL, tf32_inputs, torch.float32, DEVICE, seed=5)))
    assert failure is not None
    assert failure.family in {"C1", "C5"}


def reads_storage_as_row_major(x):
    raw = torch.as_strided(x, x.shape, (x.shape[1], 1))
    return torch.softmax(raw, dim=-1)


def test_kernel_that_ignores_strides_is_caught_on_a_strided_layout():
    failure = first_failure(gate_results(reads_storage_as_row_major))
    assert (failure.gate, failure.family) == ("G2", "C3")
    assert failure.replay.layout == "strided"


def test_strided_inputs_hold_the_same_values():
    problem = PROBLEMS["layernorm"]
    shape = {"M": 6, "N": 10}
    plain = make_inputs(problem, shape, torch.float32, "normal", 3, DEVICE)
    strided = make_inputs(problem, shape, torch.float32, "normal", 3, DEVICE, layout="strided")
    assert all(torch.equal(a, b) for a, b in zip(plain, strided))
    assert not any(t.is_contiguous() for t in strided)


@pytest.mark.parametrize(
    ("name", "shape", "dtype", "expected_us"),
    [
        ("add_relu", {"M": 4096, "N": 4096}, torch.float16, 3 * 4096 * 4096 * 2 / 3.35e12 * 1e6),
        ("matmul", {"M": 2048, "N": 2048, "K": 2048}, torch.float16, 2 * 2048**3 / 989e12 * 1e6),
        ("layernorm", {"M": 4096, "N": 4096}, torch.float32, (2 * 4096 * 4096 + 2 * 4096) * 4 / 3.35e12 * 1e6),
    ],
)
def test_hardware_floor_on_h100(name, shape, dtype, expected_us):
    floor_ms = hardware_floor_ms(PROBLEMS[name], shape, dtype, peak_for("NVIDIA H100 80GB HBM3"))
    assert floor_ms * 1e3 == pytest.approx(expected_us)


def test_peak_lookup_prefers_the_specific_variant():
    assert peak_for("NVIDIA H100 NVL").bytes_per_s == 3.9e12
    assert peak_for("NVIDIA L40S").bytes_per_s == 864e9
    assert peak_for("NVIDIA L4").bytes_per_s == 300e9
    assert peak_for("NVIDIA GeForce RTX 4090") is None
