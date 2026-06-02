from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_benchmarks(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {"best_backend": "No benchmark yet", "best_p50_ms": None, "notes": []}
    ok = df[df.get("status", "") == "ok"] if "status" in df.columns else df
    if ok.empty or "p50_ms" not in ok.columns:
        return {"best_backend": "No successful benchmark", "best_p50_ms": None, "notes": []}
    best = ok.sort_values("p50_ms", ascending=True).iloc[0].to_dict()
    baseline = ok[ok["backend"].astype(str).str.contains("NumPy", case=False, na=False)]
    speedup = None
    if not baseline.empty and float(best.get("p50_ms", 0) or 0) > 0:
        speedup = float(baseline.iloc[0]["p50_ms"]) / float(best["p50_ms"])
    return {"best_backend": best.get("backend"), "best_p50_ms": best.get("p50_ms"), "best_device": best.get("device"), "speedup_vs_numpy": speedup}


def bottleneck_guess(row: dict[str, Any]) -> str:
    op = str(row.get("operation", ""))
    bw = float(row.get("bandwidth_gbps", 0) or 0)
    if op in {"vector_add", "reduction_sum", "layer_norm"}:
        return "Likely memory-bandwidth sensitive. Optimize contiguous access, block size, vectorization, and reduce extra reads/writes."
    if op in {"matmul"}:
        return "Likely compute/tensor-core sensitive. Optimize tile size, dtype, tensor-core usage, and batching."
    if op in {"softmax"}:
        return "Likely reduction + memory sensitive. Optimize row blocking, masking, numerical stability, and fusion."
    if bw < 5:
        return "CPU or overhead dominated. Increase problem size, reduce Python overhead, or move workload to GPU."
    return "Mixed bottleneck. Use profiler traces to separate launch overhead, memory transfer, and compute time."


def make_markdown_report(system: dict[str, Any], benchmark_df: pd.DataFrame | None, inference_df: pd.DataFrame | None, recommendation: str) -> str:
    bench_summary = summarize_benchmarks(benchmark_df if benchmark_df is not None else pd.DataFrame())
    lines: list[str] = []
    lines.append("# KernelForge AI Performance Report")
    lines.append("")
    lines.append("## System Summary")
    for k, v in system.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Kernel Benchmark Summary")
    lines.append(f"- Best backend: **{bench_summary.get('best_backend')}**")
    lines.append(f"- Best p50 latency: **{bench_summary.get('best_p50_ms')} ms**")
    if bench_summary.get("speedup_vs_numpy") is not None:
        lines.append(f"- Speedup vs NumPy baseline: **{bench_summary.get('speedup_vs_numpy'):.2f}x**")
    lines.append("")
    if benchmark_df is not None and not benchmark_df.empty:
        lines.append("### Benchmark Table")
        lines.append(benchmark_df.to_markdown(index=False))
        lines.append("")
        ok = benchmark_df[benchmark_df.get("status", "") == "ok"] if "status" in benchmark_df.columns else benchmark_df
        if not ok.empty:
            best_row = ok.sort_values("p50_ms", ascending=True).iloc[0].to_dict()
            lines.append("### Bottleneck Hypothesis")
            lines.append(bottleneck_guess(best_row))
            lines.append("")
    lines.append("## Inference Benchmark Summary")
    if inference_df is not None and not inference_df.empty:
        lines.append(inference_df.to_markdown(index=False))
    else:
        lines.append("No inference benchmark was run.")
    lines.append("")
    lines.append("## Client Deployment Recommendation")
    lines.append(recommendation or "Run benchmarks first to generate a deployment recommendation.")
    lines.append("")
    lines.append("## Suggested Next Steps")
    lines.append("- Validate correctness before optimizing latency.")
    lines.append("- Track p50 and p95 latency separately.")
    lines.append("- Compare CPU, PyTorch CUDA, torch.compile, Triton kernels, ONNX Runtime, TensorRT, and Triton Inference Server when available.")
    lines.append("- Add real production traces before making GPU sizing decisions.")
    lines.append("- Keep a rollback path for kernels and optimized serving configs.")
    return "\n".join(lines)


def deployment_recommendation(system: dict[str, Any], benchmark_df: pd.DataFrame | None, inference_df: pd.DataFrame | None) -> str:
    recs = []
    cuda = bool(system.get("cuda_available"))
    triton = bool(system.get("triton_language_available"))
    torch = bool(system.get("torch_available"))
    if not cuda:
        recs.append("This environment is currently CPU-only. Use the tool for code generation, CPU baselines, and deployment planning; run final GPU benchmarks on the target GPU instance.")
    elif cuda and triton:
        recs.append("CUDA and Triton language are available, so custom GPU kernel benchmarking is suitable for bandwidth-sensitive or fused operations.")
    elif cuda and not triton:
        recs.append("CUDA is available, but Triton language is missing. Start with PyTorch CUDA and torch.compile, then install Triton for custom kernels.")
    if torch:
        recs.append("Use PyTorch eager as the correctness baseline, then compare torch.compile and custom kernels only after numeric equivalence is proven.")
    if benchmark_df is not None and not benchmark_df.empty and "status" in benchmark_df:
        ok = benchmark_df[benchmark_df["status"] == "ok"]
        if not ok.empty:
            best = ok.sort_values("p50_ms").iloc[0]
            recs.append(f"Current best kernel/backend is {best['backend']} on {best['device']} with p50 latency around {best['p50_ms']:.4f} ms.")
            if "Triton" in str(best["backend"]):
                recs.append("The custom Triton path is competitive in this run; next step is to expand autotuning and profile memory bandwidth/occupancy.")
            elif "PyTorch CUDA" in str(best["backend"]):
                recs.append("PyTorch CUDA is strong for this operation; custom kernels should focus on fusion or special shapes rather than replacing vendor-optimized primitives blindly.")
            elif "NumPy" in str(best["backend"]):
                recs.append("CPU baseline is currently best or only successful; this may indicate small problem size, launch overhead, or missing GPU runtime.")
    if inference_df is not None and not inference_df.empty:
        ok = inference_df[inference_df.get("status", "") == "ok"] if "status" in inference_df.columns else inference_df
        if not ok.empty:
            best = ok.sort_values("p50_ms").iloc[0]
            recs.append(f"For the toy inference sweep, best mode is {best['mode']} at batch size {int(best['batch_size'])}, with p50 latency near {best['p50_ms']:.4f} ms.")
    recs.append("For production serving, export ONNX/TensorRT candidates and test NVIDIA Triton Inference Server dynamic batching with perf_analyzer on realistic traffic.")
    return "\n\n".join(recs)
