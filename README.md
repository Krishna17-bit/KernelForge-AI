# KernelForge AI — GPU Kernel & Inference Optimization Workbench

**KernelForge AI** is a local GPU and inference optimization workbench for CUDA-aware benchmarking, Triton kernel experimentation, PyTorch baseline comparison, autotuning, model-serving export, and forward-deployed engineering style performance reports.

It is designed to test advanced AI systems capability beyond normal RAG/chatbot apps: GPU systems thinking, performance benchmarking, correctness testing, Triton kernel awareness, inference serving, profiling, and deployment recommendations.

The app runs safely on normal CPU laptops and unlocks GPU/Triton paths automatically when PyTorch, CUDA, and Triton are available.

---

## Why this project exists

Modern AI systems are not only about model accuracy. Production AI teams also care about:

- latency
- throughput
- GPU cost
- p50 / p95 / p99 behavior
- memory bandwidth
- batching strategy
- inference serving
- correctness after optimization
- rollback plans
- monitoring and observability

KernelForge AI gives a practical local workspace to explore those ideas.

---

## Features

### System Check

- Detects Python version
- Detects operating system and CPU count
- Detects RAM
- Detects PyTorch installation
- Detects CUDA availability
- Detects GPU name and VRAM
- Detects Triton language package
- Detects CuPy, Numba, ONNX availability
- Reads `nvidia-smi` when available
- Produces environment recommendations

### Kernel Benchmark Lab

Compare multiple backends:

- NumPy CPU baseline
- PyTorch CPU
- PyTorch CUDA
- `torch.compile` where available
- custom Triton vector-add kernel when available

Supported benchmark operations:

- vector add
- matrix multiplication
- softmax
- GELU
- layer norm
- reduction sum

Metrics include:

- p50 latency
- p95 latency
- mean latency
- throughput estimate
- memory bandwidth estimate
- correctness error
- benchmark status
- backend notes

### Triton Kernel Editor

Includes editable/exportable Triton kernel templates:

- vector add
- softmax
- RMSNorm
- block matrix multiplication skeleton

You can inspect code, edit it, download one kernel, or download all templates as a ZIP.

### Autotune Playground

Runs Triton vector-add block-size sweeps when CUDA + Triton are available.

Autotune variables:

- block size
- dtype
- tensor size
- repeat count

Outputs:

- latency table
- best configuration
- skipped status when GPU/Triton is unavailable

### Inference Benchmark

Benchmarks a toy MLP to study inference latency and throughput tradeoffs.

Modes:

- NumPy CPU
- PyTorch CPU
- PyTorch CUDA
- optional `torch.compile`

Configurable:

- batch sizes
- input dimension
- hidden dimension
- output dimension
- dtype
- repeat count

### Model Serving Export

Generates a NVIDIA Triton Inference Server repository template:

- `config.pbtxt`
- model repository directory layout
- client script
- deployment notes
- Docker command
- `perf_analyzer` commands
- dynamic batching config
- instance group suggestion

### Profiler Report

Generates a markdown report with:

- system summary
- kernel benchmark table
- inference benchmark table
- bottleneck hypothesis
- deployment recommendation
- next optimization steps

### Recommendation

- whether the workload is GPU-worthy
- whether PyTorch, Triton kernels, or serving optimization is the next step
- what benchmark result is currently best
- what production risks remain
- what to test next before deployment

### Export Center

Exports:

- audit JSON
- markdown performance report
- kernel benchmark CSV
- inference benchmark CSV
- Triton serving repository ZIP
- kernel templates ZIP

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Benchmarking | NumPy, optional PyTorch |
| GPU kernels | optional OpenAI Triton language |
| Inference baseline | NumPy / PyTorch / torch.compile |
| Serving export | NVIDIA Triton Inference Server templates |
| Charts | Plotly |
| System probe | psutil, nvidia-smi |
| Optional AI narrative | Gemini API |
| Reports | Markdown, CSV, JSON |

---

## Installation

### Standard local install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

---

## Optional GPU packages

The base app intentionally does not force heavy GPU installs. Install the correct PyTorch build for your CUDA version from the official PyTorch instructions.

Example only:

```bash
pip install torch torchvision torchaudio
```

For Triton kernels:

```bash
pip install triton
```

On some Windows systems, Triton support may depend on environment and package availability. The app still runs in CPU-safe mode.

---

## Optional AI Narrative Mode

Open `.env` and add:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-pro
```
Benchmarks and exports work even without an API key.

---

## Quick Test

1. Run the app.
2. Open **System Check**.
3. Open **Kernel Lab**.
4. Choose `vector_add`.
5. Click **Run kernel benchmark**.
6. Open **Triton Kernel Editor** and download kernel templates.
7. Open **Inference Benchmark** and run the toy MLP sweep.
8. Open **Model Serving Export** and download the serving template ZIP.
9. Open **Profiler Report** and download the performance report.


## Disclaimer

This project is an engineering workbench and educational optimizer. Benchmark results depend heavily on machine, drivers, CUDA version, installed packages, tensor shape, dtype, warmup count, and background load. Always validate with production-like traffic before making deployment or infrastructure decisions.
