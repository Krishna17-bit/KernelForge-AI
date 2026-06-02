from __future__ import annotations

import importlib.util
import math
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable, Any

import numpy as np
import pandas as pd


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class BenchmarkResult:
    operation: str
    backend: str
    device: str
    size_label: str
    dtype: str
    p50_ms: float
    p95_ms: float
    mean_ms: float
    throughput: float
    bandwidth_gbps: float
    correctness_error: float
    status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _time_callable(fn: Callable[[], Any], repeats: int, warmups: int = 3, sync: Callable[[], None] | None = None) -> tuple[list[float], Any]:
    result = None
    for _ in range(max(0, warmups)):
        result = fn()
        if sync:
            sync()
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times, result


def _summarize(times: list[float]) -> tuple[float, float, float]:
    if not times:
        return 0.0, 0.0, 0.0
    sorted_times = sorted(times)
    p50 = statistics.median(sorted_times)
    p95_idx = max(0, min(len(sorted_times) - 1, int(math.ceil(len(sorted_times) * 0.95)) - 1))
    return p50, sorted_times[p95_idx], statistics.mean(sorted_times)


def _dtype_np(dtype: str):
    return np.float32 if dtype in {"fp32", "float32"} else np.float64


def run_numpy_benchmark(operation: str, n: int, matrix_dim: int, dtype: str, repeats: int) -> BenchmarkResult:
    dt = _dtype_np(dtype)
    rng = np.random.default_rng(42)
    size_label = f"n={n:,}" if operation != "matmul" else f"{matrix_dim}x{matrix_dim}"
    bytes_est = 0

    if operation == "vector_add":
        x = rng.normal(size=n).astype(dt)
        y = rng.normal(size=n).astype(dt)
        ref = x + y
        bytes_est = x.nbytes * 3
        fn = lambda: x + y
    elif operation == "reduction_sum":
        x = rng.normal(size=n).astype(dt)
        ref = np.sum(x)
        bytes_est = x.nbytes
        fn = lambda: np.sum(x)
    elif operation == "softmax":
        rows = max(1, n // 1024)
        x = rng.normal(size=(rows, 1024)).astype(dt)
        def _softmax():
            z = x - x.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        ref = _softmax()
        bytes_est = x.nbytes * 3
        fn = _softmax
    elif operation == "gelu":
        x = rng.normal(size=n).astype(dt)
        def _gelu():
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        ref = _gelu()
        bytes_est = x.nbytes * 2
        fn = _gelu
    elif operation == "layer_norm":
        rows = max(1, n // 1024)
        x = rng.normal(size=(rows, 1024)).astype(dt)
        def _layernorm():
            mu = x.mean(axis=1, keepdims=True)
            var = x.var(axis=1, keepdims=True)
            return (x - mu) / np.sqrt(var + 1e-5)
        ref = _layernorm()
        bytes_est = x.nbytes * 4
        fn = _layernorm
    elif operation == "matmul":
        a = rng.normal(size=(matrix_dim, matrix_dim)).astype(dt)
        b = rng.normal(size=(matrix_dim, matrix_dim)).astype(dt)
        ref = a @ b
        bytes_est = a.nbytes + b.nbytes + ref.nbytes
        fn = lambda: a @ b
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    times, out = _time_callable(fn, repeats=repeats)
    p50, p95, mean = _summarize(times)
    err = float(np.max(np.abs(np.asarray(out) - np.asarray(ref)))) if not np.isscalar(out) else float(abs(float(out) - float(ref)))
    throughput = (n / (p50 / 1000.0)) if p50 > 0 and operation != "matmul" else ((2 * matrix_dim**3) / (p50 / 1000.0) if p50 > 0 and operation == "matmul" else 0.0)
    bandwidth = (bytes_est / (p50 / 1000.0) / 1e9) if p50 > 0 else 0.0
    return BenchmarkResult(operation, "NumPy", "CPU", size_label, str(np.dtype(dt)), p50, p95, mean, throughput, bandwidth, err, "ok", "CPU fallback benchmark")


def run_torch_benchmark(operation: str, n: int, matrix_dim: int, dtype: str, repeats: int, device: str = "cpu", use_compile: bool = False) -> BenchmarkResult:
    if not has_module("torch"):
        return BenchmarkResult(operation, "PyTorch", device, "unavailable", dtype, 0, 0, 0, 0, 0, 0, "skipped", "PyTorch is not installed")
    import torch  # type: ignore

    if device == "cuda" and not torch.cuda.is_available():
        return BenchmarkResult(operation, "PyTorch", device, "unavailable", dtype, 0, 0, 0, 0, 0, 0, "skipped", "CUDA is not available")

    torch_dtype = torch.float16 if dtype in {"fp16", "float16"} else torch.float32
    if device == "cpu" and torch_dtype == torch.float16:
        torch_dtype = torch.float32

    size_label = f"n={n:,}" if operation != "matmul" else f"{matrix_dim}x{matrix_dim}"
    gen = torch.Generator(device="cpu").manual_seed(42)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    if operation == "vector_add":
        x = torch.randn(n, dtype=torch_dtype, generator=gen, device=device)
        y = torch.randn(n, dtype=torch_dtype, generator=gen, device=device)
        ref = x + y
        fn = lambda: x + y
        bytes_est = x.numel() * x.element_size() * 3
    elif operation == "reduction_sum":
        x = torch.randn(n, dtype=torch_dtype, generator=gen, device=device)
        ref = x.sum()
        fn = lambda: x.sum()
        bytes_est = x.numel() * x.element_size()
    elif operation == "softmax":
        rows = max(1, n // 1024)
        x = torch.randn((rows, 1024), dtype=torch_dtype, generator=gen, device=device)
        ref = torch.softmax(x, dim=1)
        fn = lambda: torch.softmax(x, dim=1)
        bytes_est = x.numel() * x.element_size() * 3
    elif operation == "gelu":
        x = torch.randn(n, dtype=torch_dtype, generator=gen, device=device)
        ref = torch.nn.functional.gelu(x)
        fn = lambda: torch.nn.functional.gelu(x)
        bytes_est = x.numel() * x.element_size() * 2
    elif operation == "layer_norm":
        rows = max(1, n // 1024)
        x = torch.randn((rows, 1024), dtype=torch_dtype, generator=gen, device=device)
        ref = torch.nn.functional.layer_norm(x, (1024,))
        fn = lambda: torch.nn.functional.layer_norm(x, (1024,))
        bytes_est = x.numel() * x.element_size() * 4
    elif operation == "matmul":
        x = torch.randn((matrix_dim, matrix_dim), dtype=torch_dtype, generator=gen, device=device)
        y = torch.randn((matrix_dim, matrix_dim), dtype=torch_dtype, generator=gen, device=device)
        ref = x @ y
        fn = lambda: x @ y
        bytes_est = (x.numel() + y.numel() + ref.numel()) * x.element_size()
    else:
        raise ValueError(operation)

    backend_name = "PyTorch CUDA" if device == "cuda" else "PyTorch CPU"
    if use_compile and hasattr(torch, "compile"):
        try:
            fn = torch.compile(fn)  # type: ignore[assignment]
            backend_name += " + torch.compile"
        except Exception:
            backend_name += " compile unavailable"

    times, out = _time_callable(fn, repeats=repeats, sync=sync)
    p50, p95, mean = _summarize(times)
    try:
        err = float((out - ref).abs().max().detach().cpu())
    except Exception:
        err = 0.0
    throughput = (n / (p50 / 1000.0)) if p50 > 0 and operation != "matmul" else ((2 * matrix_dim**3) / (p50 / 1000.0) if p50 > 0 and operation == "matmul" else 0.0)
    bandwidth = (bytes_est / (p50 / 1000.0) / 1e9) if p50 > 0 else 0.0
    return BenchmarkResult(operation, backend_name, device.upper(), size_label, str(torch_dtype).replace("torch.", ""), p50, p95, mean, throughput, bandwidth, err, "ok", "Torch benchmark")


def run_triton_vector_add(n: int, dtype: str, repeats: int, block_size: int = 1024) -> BenchmarkResult:
    if not has_module("torch") or not has_module("triton"):
        return BenchmarkResult("vector_add", "Triton", "CUDA", f"n={n:,}", dtype, 0, 0, 0, 0, 0, 0, "skipped", "PyTorch or Triton is not installed")
    import torch  # type: ignore
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    if not torch.cuda.is_available():
        return BenchmarkResult("vector_add", "Triton", "CUDA", f"n={n:,}", dtype, 0, 0, 0, 0, 0, 0, "skipped", "CUDA is not available")

    torch_dtype = torch.float16 if dtype in {"fp16", "float16"} else torch.float32

    @triton.jit
    def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements:tl.constexpr, BLOCK_SIZE:tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    x = torch.randn(n, device="cuda", dtype=torch_dtype)
    y = torch.randn(n, device="cuda", dtype=torch_dtype)
    ref = x + y
    out = torch.empty_like(x)

    def fn():
        grid = (triton.cdiv(n, block_size),)
        _vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
        return out

    times, result = _time_callable(fn, repeats=repeats, sync=torch.cuda.synchronize)
    p50, p95, mean = _summarize(times)
    err = float((result - ref).abs().max().detach().cpu())
    bytes_est = x.numel() * x.element_size() * 3
    throughput = n / (p50 / 1000.0) if p50 > 0 else 0.0
    bandwidth = bytes_est / (p50 / 1000.0) / 1e9 if p50 > 0 else 0.0
    return BenchmarkResult("vector_add", f"Triton BLOCK={block_size}", "CUDA", f"n={n:,}", str(torch_dtype).replace("torch.", ""), p50, p95, mean, throughput, bandwidth, err, "ok", "Custom Triton kernel")


def benchmark_suite(operation: str, n: int, matrix_dim: int, dtype: str, repeats: int, include_torch_cpu: bool, include_torch_cuda: bool, include_triton: bool, include_compile: bool) -> pd.DataFrame:
    rows = [run_numpy_benchmark(operation, n, matrix_dim, dtype if dtype != "fp16" else "fp32", repeats).to_dict()]
    if include_torch_cpu:
        rows.append(run_torch_benchmark(operation, n, matrix_dim, dtype, repeats, device="cpu", use_compile=False).to_dict())
        if include_compile:
            rows.append(run_torch_benchmark(operation, n, matrix_dim, dtype, repeats, device="cpu", use_compile=True).to_dict())
    if include_torch_cuda:
        rows.append(run_torch_benchmark(operation, n, matrix_dim, dtype, repeats, device="cuda", use_compile=False).to_dict())
        if include_compile:
            rows.append(run_torch_benchmark(operation, n, matrix_dim, dtype, repeats, device="cuda", use_compile=True).to_dict())
    if include_triton and operation == "vector_add":
        for bs in [256, 512, 1024, 2048]:
            rows.append(run_triton_vector_add(n, dtype, repeats, block_size=bs).to_dict())
    return pd.DataFrame(rows)


def autotune_vector_add(n: int, dtype: str, repeats: int, block_sizes: list[int]) -> pd.DataFrame:
    rows = []
    for bs in block_sizes:
        rows.append(run_triton_vector_add(n, dtype, repeats, bs).to_dict())
    df = pd.DataFrame(rows)
    if not df.empty and "p50_ms" in df:
        df["rank"] = df["p50_ms"].rank(method="dense")
    return df
