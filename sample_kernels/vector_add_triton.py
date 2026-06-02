# Triton vector addition kernel
# Run on a CUDA machine with: pip install torch triton
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements:tl.constexpr, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, block_size:int = 1024) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.numel() == y.numel()
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out
