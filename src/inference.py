from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class InferenceBenchRow:
    mode: str
    device: str
    batch_size: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    p50_ms: float
    tokens_or_rows_per_sec: float
    status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values)-1, max(0, int(round((len(values)-1)*q))))
    return float(values[idx])


def _bench(fn, repeats: int, sync=None) -> float:
    for _ in range(3):
        fn()
        if sync:
            sync()
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)
    return _percentile(times, 0.5)


def numpy_mlp_bench(batch_size: int, input_dim: int, hidden_dim: int, output_dim: int, repeats: int) -> InferenceBenchRow:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(batch_size, input_dim)).astype(np.float32)
    w1 = rng.normal(size=(input_dim, hidden_dim)).astype(np.float32) / np.sqrt(input_dim)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = rng.normal(size=(hidden_dim, output_dim)).astype(np.float32) / np.sqrt(hidden_dim)
    b2 = np.zeros((output_dim,), dtype=np.float32)
    def forward():
        h = np.maximum(x @ w1 + b1, 0)
        return h @ w2 + b2
    p50 = _bench(forward, repeats)
    return InferenceBenchRow("NumPy MLP", "CPU", batch_size, input_dim, hidden_dim, output_dim, p50, batch_size/(p50/1000.0) if p50 > 0 else 0, "ok", "CPU baseline")


def torch_mlp_bench(batch_size: int, input_dim: int, hidden_dim: int, output_dim: int, repeats: int, device: str, compile_model: bool = False, dtype: str = "fp32") -> InferenceBenchRow:
    if not has_module("torch"):
        return InferenceBenchRow("PyTorch MLP", device, batch_size, input_dim, hidden_dim, output_dim, 0, 0, "skipped", "PyTorch not installed")
    import torch  # type: ignore
    if device == "cuda" and not torch.cuda.is_available():
        return InferenceBenchRow("PyTorch MLP", device, batch_size, input_dim, hidden_dim, output_dim, 0, 0, "skipped", "CUDA unavailable")
    td = torch.float16 if dtype == "fp16" and device == "cuda" else torch.float32
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).to(device=device, dtype=td).eval()
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            mode = f"PyTorch MLP + compile ({dtype})"
        except Exception:
            mode = f"PyTorch MLP compile failed ({dtype})"
    else:
        mode = f"PyTorch MLP ({dtype})"
    x = torch.randn(batch_size, input_dim, device=device, dtype=td)
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    @torch.no_grad()
    def forward():
        return model(x)
    p50 = _bench(forward, repeats, sync=sync)
    return InferenceBenchRow(mode, device.upper(), batch_size, input_dim, hidden_dim, output_dim, p50, batch_size/(p50/1000.0) if p50 > 0 else 0, "ok", "Toy inference benchmark")


def run_inference_sweep(batch_sizes: list[int], input_dim: int, hidden_dim: int, output_dim: int, repeats: int, include_torch: bool, include_cuda: bool, include_compile: bool, dtype: str) -> pd.DataFrame:
    rows = []
    for bs in batch_sizes:
        rows.append(numpy_mlp_bench(bs, input_dim, hidden_dim, output_dim, repeats).to_dict())
        if include_torch:
            rows.append(torch_mlp_bench(bs, input_dim, hidden_dim, output_dim, repeats, "cpu", False, dtype).to_dict())
            if include_compile:
                rows.append(torch_mlp_bench(bs, input_dim, hidden_dim, output_dim, repeats, "cpu", True, dtype).to_dict())
        if include_cuda:
            rows.append(torch_mlp_bench(bs, input_dim, hidden_dim, output_dim, repeats, "cuda", False, dtype).to_dict())
            if include_compile:
                rows.append(torch_mlp_bench(bs, input_dim, hidden_dim, output_dim, repeats, "cuda", True, dtype).to_dict())
    return pd.DataFrame(rows)
