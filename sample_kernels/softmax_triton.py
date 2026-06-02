# Row-wise softmax Triton kernel
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols:tl.constexpr, BLOCK_SIZE:tl.constexpr):
    row_id = tl.program_id(0)
    row_start = row_id * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf"))
    row = row - tl.max(row, axis=0)
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator
    tl.store(y_ptr + row_start + offsets, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, y, n_cols, BLOCK_SIZE=block_size, num_warps=4)
    return y
