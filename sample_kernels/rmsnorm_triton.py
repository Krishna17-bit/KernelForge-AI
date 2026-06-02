# RMSNorm Triton kernel skeleton
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_cols:tl.constexpr, eps:tl.constexpr, BLOCK_SIZE:tl.constexpr):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_start = row_id * n_cols
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    variance = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    y = x * rstd * w
    tl.store(y_ptr + row_start + offsets, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    rmsnorm_kernel[(n_rows,)](x, weight, y, n_cols, eps, BLOCK_SIZE=block_size, num_warps=4)
    return y
