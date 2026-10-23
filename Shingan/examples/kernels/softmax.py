import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    x = tl.load(in_ptr + row * in_stride + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    shifted = x - tl.max(x, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * out_stride + offsets, numerator / denominator, mask=mask)


def run(x):
    x = x.contiguous()
    rows, cols = x.shape
    out = torch.empty_like(x)
    _softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols, BLOCK=triton.next_power_of_2(cols))
    return out
