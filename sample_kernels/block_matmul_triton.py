# Educational block matmul Triton kernel skeleton
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
