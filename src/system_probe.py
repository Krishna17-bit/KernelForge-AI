from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any

import psutil


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass
class SystemInfo:
    python_version: str
    platform: str
    cpu_count: int
    ram_gb: float
    torch_available: bool
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_name: str
    gpu_count: int
    total_vram_gb: float
    triton_language_available: bool
    cupy_available: bool
    numba_available: bool
    onnx_available: bool
    tensor_rt_hint: str
    nvidia_smi: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_nvidia_smi() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            timeout=3,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "nvidia-smi unavailable"


def probe_system() -> SystemInfo:
    torch_available = has_module("torch")
    torch_version = "not installed"
    cuda_available = False
    cuda_version = "unavailable"
    gpu_name = "No CUDA GPU detected"
    gpu_count = 0
    total_vram_gb = 0.0

    if torch_available:
        try:
            import torch  # type: ignore

            torch_version = str(torch.__version__)
            cuda_available = bool(torch.cuda.is_available())
            cuda_version = str(getattr(torch.version, "cuda", None) or "unavailable")
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total_vram_gb = float(props.total_memory) / (1024 ** 3)
        except Exception as exc:
            torch_version = f"import failed: {exc}"

    return SystemInfo(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu_count=os.cpu_count() or 1,
        ram_gb=round(psutil.virtual_memory().total / (1024 ** 3), 2),
        torch_available=torch_available,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        total_vram_gb=round(total_vram_gb, 2),
        triton_language_available=has_module("triton"),
        cupy_available=has_module("cupy"),
        numba_available=has_module("numba"),
        onnx_available=has_module("onnx"),
        tensor_rt_hint="Check separately with `trtexec --version` or NVIDIA TensorRT Python wheels.",
        nvidia_smi=_run_nvidia_smi(),
    )


def capability_recommendations(info: SystemInfo) -> list[str]:
    recs: list[str] = []
    if info.cuda_available and info.triton_language_available:
        recs.append("CUDA + Triton language path is available. You can run custom Triton kernel benchmarks.")
    elif info.cuda_available and not info.triton_language_available:
        recs.append("CUDA is available, but Triton language is not installed. Install `triton` to run custom GPU kernels.")
    elif not info.cuda_available:
        recs.append("CUDA GPU is not detected. CPU/NumPy benchmarks and code generation still work.")

    if not info.torch_available:
        recs.append("PyTorch is not installed. Install PyTorch to enable GPU inference benchmarks and torch.compile checks.")
    if info.cupy_available:
        recs.append("CuPy is available. You can extend this project with CUDA-array benchmarks.")
    if info.numba_available:
        recs.append("Numba is available. You can add @cuda.jit kernels as a second GPU path.")
    if info.ram_gb < 8:
        recs.append("RAM is below 8 GB. Use smaller benchmark sizes to avoid system pressure.")
    return recs
