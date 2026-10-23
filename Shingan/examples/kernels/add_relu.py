import torch
import triton
import triton.language as tl


@triton.jit
def _add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float32)
    total = x + y
    tl.store(out_ptr + offsets, tl.where(total < 0, 0.0, total), mask=mask)


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    _add_relu_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out
