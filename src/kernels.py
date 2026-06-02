from __future__ import annotations

KERNEL_TEMPLATES: dict[str, str] = {
    "vector_add_triton.py": r'''# Triton vector addition kernel
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
''',
    "softmax_triton.py": r'''# Row-wise softmax Triton kernel
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
''',
    "rmsnorm_triton.py": r'''# RMSNorm Triton kernel skeleton
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
''',
    "block_matmul_triton.py": r'''# Educational block matmul Triton kernel skeleton
# For production, compare against torch.matmul and use autotuning for BLOCK_M/N/K.
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
                  stride_am:tl.constexpr, stride_ak:tl.constexpr,
                  stride_bk:tl.constexpr, stride_bn:tl.constexpr,
                  stride_cm:tl.constexpr, stride_cn:tl.constexpr,
                  BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak,
                    mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K), other=0.0)
        b = tl.load(b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                    mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
''',
}


def get_kernel_template(name: str) -> str:
    return KERNEL_TEMPLATES.get(name, next(iter(KERNEL_TEMPLATES.values())))


def list_kernel_templates() -> list[str]:
    return list(KERNEL_TEMPLATES.keys())


def save_templates(base_dir):
    from pathlib import Path
    out = Path(base_dir) / "sample_kernels"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, code in KERNEL_TEMPLATES.items():
        p = out / name
        p.write_text(code, encoding="utf-8")
        paths.append(p)
    return paths
