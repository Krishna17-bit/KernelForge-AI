from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai_client import AIReasoningClient
from src.benchmarks import autotune_vector_add, benchmark_suite
from src.inference import run_inference_sweep
from src.kernels import get_kernel_template, list_kernel_templates, save_templates
from src.reports import deployment_recommendation, make_markdown_report
from src.serving_export import make_config_pbtxt, make_deployment_notes, make_serving_export_zip
from src.system_probe import capability_recommendations, probe_system
from src.ui_styles import APP_CSS
from src.utils import ensure_dir, metric_card, now_id, pill, render_dark_table, write_json

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_dir(BASE_DIR / "outputs")
save_templates(BASE_DIR)

st.set_page_config(
    page_title="KernelForge AI",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(APP_CSS, unsafe_allow_html=True)

client = AIReasoningClient.from_env()

if "system_info" not in st.session_state:
    st.session_state.system_info = probe_system()
if "kernel_bench" not in st.session_state:
    st.session_state.kernel_bench = pd.DataFrame()
if "autotune_df" not in st.session_state:
    st.session_state.autotune_df = pd.DataFrame()
if "inference_df" not in st.session_state:
    st.session_state.inference_df = pd.DataFrame()
if "recommendation" not in st.session_state:
    st.session_state.recommendation = ""
if "ai_narrative" not in st.session_state:
    st.session_state.ai_narrative = ""


def downloadable_zip_from_templates() -> bytes:
    mem = BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in list_kernel_templates():
            zf.writestr(name, get_kernel_template(name))
    mem.seek(0)
    return mem.getvalue()


def chart_latency(df: pd.DataFrame, title: str) -> None:
    if df is None or df.empty or "p50_ms" not in df.columns:
        st.info("Run a benchmark to see chart output.")
        return
    ok = df[df.get("status", "") == "ok"] if "status" in df.columns else df
    if ok.empty:
        st.info("No successful benchmark rows to chart.")
        return
    label_col = "backend" if "backend" in ok.columns else "mode"
    fig = px.bar(ok, x=label_col, y="p50_ms", color="device" if "device" in ok.columns else None, title=title)
    fig.update_layout(height=380, paper_bgcolor="#070707", plot_bgcolor="#111111", font_color="#ffffff")
    fig.update_xaxes(gridcolor="#242424")
    fig.update_yaxes(gridcolor="#242424")
    st.plotly_chart(fig, use_container_width=True)


with st.sidebar:
    st.markdown("### KernelForge AI")
    st.markdown(
        "<span class='small-muted'>GPU kernel and inference optimization workbench for CUDA-aware benchmarking, Triton kernels, PyTorch baselines, autotuning, profiling, serving export, and FDE-style deployment recommendations.</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Runtime status**")
    info = st.session_state.system_info
    if info.cuda_available:
        st.success("CUDA runtime detected")
    else:
        st.warning("CPU-safe mode")
    if client.configured:
        st.success("AI narrative mode configured")
    else:
        st.info("Static profiler mode")
    st.caption(client.status_help)
    st.divider()
    if st.button("Refresh system check", use_container_width=True):
        st.session_state.system_info = probe_system()
        st.rerun()
    st.divider()
    st.markdown("**Advanced stack covered**")
    st.markdown(
        "- PyTorch CPU/CUDA baselines\n"
        "- Optional Triton kernels\n"
        "- torch.compile checks\n"
        "- Autotuning playground\n"
        "- Latency p50 and p95\n"
        "- Bandwidth estimates\n"
        "- Correctness checks\n"
        "- Inference sweep\n"
        "- NVIDIA Triton serving export\n"
        "- FDE deployment report"
    )

st.markdown(
    """
    <div class='hero'>
        <div class='hero-title'>KernelForge AI</div>
        <div class='hero-subtitle'>
            GPU kernel and inference optimization workbench for engineers who need to compare PyTorch baselines,
            custom Triton kernels, CPU fallbacks, autotuned block sizes, model-serving configs, and deployment-ready
            performance reports. Runs safely without a GPU, and unlocks CUDA/Triton paths when available.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

info = st.session_state.system_info
m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("CUDA", "Yes" if info.cuda_available else "No", info.gpu_name if info.cuda_available else "CPU fallback ready")
with m2:
    metric_card("PyTorch", "Yes" if info.torch_available else "No", info.torch_version)
with m3:
    metric_card("Triton language", "Yes" if info.triton_language_available else "No", "Custom kernels enabled" if info.triton_language_available else "Kernel templates still export")
with m4:
    metric_card("System RAM", f"{info.ram_gb} GB", f"CPU cores: {info.cpu_count}")

tabs = st.tabs(
    [
        "System Check",
        "Kernel Lab",
        "Triton Kernel Editor",
        "Autotune",
        "Inference Benchmark",
        "Model Serving Export",
        "Profiler Report",
        "Client Recommendation",
        "Export Center",
    ]
)

with tabs[0]:
    st.markdown("### Runtime capability map")
    caps = []
    caps.append(pill("CUDA available" if info.cuda_available else "CUDA unavailable", "ok" if info.cuda_available else "warn"))
    caps.append(pill("PyTorch installed" if info.torch_available else "PyTorch missing", "ok" if info.torch_available else "warn"))
    caps.append(pill("Triton installed" if info.triton_language_available else "Triton missing", "ok" if info.triton_language_available else "warn"))
    caps.append(pill("CuPy installed" if info.cupy_available else "CuPy optional", "ok" if info.cupy_available else ""))
    caps.append(pill("Numba installed" if info.numba_available else "Numba optional", "ok" if info.numba_available else ""))
    st.markdown(" ".join(caps), unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### System details")
        render_dark_table(pd.DataFrame([info.to_dict()]).T.reset_index().rename(columns={"index": "field", 0: "value"}), height=440)
    with c2:
        st.markdown("#### Recommendations")
        for rec in capability_recommendations(info):
            st.markdown(f"<div class='panel'>{rec}</div>", unsafe_allow_html=True)
        st.markdown("#### nvidia-smi")
        st.code(info.nvidia_smi, language="text")

with tabs[1]:
    st.markdown("### Kernel benchmark lab")
    st.caption("Compare CPU NumPy, optional PyTorch CPU/CUDA, torch.compile, and Triton vector-add kernels. Keep sizes moderate on laptops.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        operation = st.selectbox("Operation", ["vector_add", "matmul", "softmax", "gelu", "layer_norm", "reduction_sum"])
    with c2:
        dtype = st.selectbox("Dtype", ["fp32", "fp16"])
    with c3:
        n = st.selectbox("Vector/tensor elements", [65_536, 262_144, 1_048_576, 4_194_304], index=2)
    with c4:
        matrix_dim = st.selectbox("Matmul dim", [128, 256, 512, 1024], index=1)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        repeats = st.slider("Repeats", 5, 100, 20, 5)
    with b2:
        include_torch_cpu = st.checkbox("Include PyTorch CPU", value=info.torch_available)
    with b3:
        include_torch_cuda = st.checkbox("Include PyTorch CUDA", value=info.cuda_available and info.torch_available)
    with b4:
        include_triton = st.checkbox("Include Triton custom kernel", value=info.cuda_available and info.triton_language_available and operation == "vector_add")
    include_compile = st.checkbox("Include torch.compile where available", value=False)

    if st.button("Run kernel benchmark", use_container_width=True):
        with st.spinner("Running kernel benchmarks and correctness checks..."):
            df = benchmark_suite(operation, int(n), int(matrix_dim), dtype, int(repeats), include_torch_cpu, include_torch_cuda, include_triton, include_compile)
            st.session_state.kernel_bench = df
            st.session_state.recommendation = deployment_recommendation(info.to_dict(), df, st.session_state.inference_df)
        st.success("Benchmark complete.")

    chart_latency(st.session_state.kernel_bench, "Kernel p50 latency by backend")
    render_dark_table(st.session_state.kernel_bench, height=360)

with tabs[2]:
    st.markdown("### Triton kernel editor and export")
    st.caption("These kernels are educational but runnable on a CUDA machine with PyTorch + Triton installed. Use them as a starting point for deeper profiling.")
    selected_kernel = st.selectbox("Kernel template", list_kernel_templates())
    code = st.text_area("Kernel code", value=get_kernel_template(selected_kernel), height=520)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download selected kernel", data=code, file_name=selected_kernel, mime="text/x-python", use_container_width=True)
    with d2:
        st.download_button("Download all kernel templates ZIP", data=downloadable_zip_from_templates(), file_name="kernelforge_triton_kernels.zip", mime="application/zip", use_container_width=True)

    st.markdown("#### Optimization notes")
    st.markdown(
        """
        <div class='panel-blue'>
        Start with correctness, then measure p50/p95 latency. For memory-bound operations like vector add, layer norm, and RMSNorm,
        focus on contiguous access, block size, extra memory reads/writes, fusion, and launch overhead. For matmul, compare against
        vendor libraries before claiming custom-kernel speedups.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.markdown("### Autotune playground")
    st.caption("Autotunes Triton vector-add block sizes when CUDA + Triton are available. Otherwise, rows are marked skipped so the report still explains the environment gap.")
    c1, c2, c3 = st.columns(3)
    with c1:
        tune_n = st.selectbox("Elements", [262_144, 1_048_576, 4_194_304, 16_777_216], index=1)
    with c2:
        tune_dtype = st.selectbox("Autotune dtype", ["fp32", "fp16"], key="tune_dtype")
    with c3:
        tune_repeats = st.slider("Autotune repeats", 5, 60, 15, 5)
    block_sizes = st.multiselect("Block sizes", [128, 256, 512, 1024, 2048, 4096], default=[256, 512, 1024, 2048])

    if st.button("Run autotune", use_container_width=True):
        with st.spinner("Testing Triton block sizes..."):
            st.session_state.autotune_df = autotune_vector_add(int(tune_n), tune_dtype, int(tune_repeats), block_sizes)
        st.success("Autotune run complete.")
    chart_latency(st.session_state.autotune_df, "Autotune p50 latency by block size")
    render_dark_table(st.session_state.autotune_df, height=340)

with tabs[4]:
    st.markdown("### Inference benchmark")
    st.caption("Benchmark a toy MLP across batch sizes to reason about latency/throughput tradeoffs. This is a deployment-pattern benchmark, not a model accuracy test.")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        input_dim = st.selectbox("Input dim", [128, 256, 512, 1024, 2048], index=2)
    with p2:
        hidden_dim = st.selectbox("Hidden dim", [256, 512, 1024, 2048, 4096], index=2)
    with p3:
        output_dim = st.selectbox("Output dim", [8, 16, 32, 64, 128], index=2)
    with p4:
        inf_dtype = st.selectbox("Inference dtype", ["fp32", "fp16"], key="inf_dtype")
    batch_sizes = st.multiselect("Batch sizes", [1, 2, 4, 8, 16, 32, 64, 128, 256], default=[1, 8, 32, 128])
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        inf_repeats = st.slider("Inference repeats", 5, 100, 25, 5)
    with q2:
        inf_torch = st.checkbox("PyTorch CPU", value=info.torch_available)
    with q3:
        inf_cuda = st.checkbox("PyTorch CUDA", value=info.cuda_available and info.torch_available)
    with q4:
        inf_compile = st.checkbox("torch.compile", value=False)

    if st.button("Run inference sweep", use_container_width=True):
        with st.spinner("Running inference latency/throughput sweep..."):
            st.session_state.inference_df = run_inference_sweep(batch_sizes, int(input_dim), int(hidden_dim), int(output_dim), int(inf_repeats), inf_torch, inf_cuda, inf_compile, inf_dtype)
            st.session_state.recommendation = deployment_recommendation(info.to_dict(), st.session_state.kernel_bench, st.session_state.inference_df)
        st.success("Inference benchmark complete.")
    chart_latency(st.session_state.inference_df, "Inference p50 latency by mode")
    render_dark_table(st.session_state.inference_df, height=360)

with tabs[5]:
    st.markdown("### Model serving export")
    st.caption("Generate a NVIDIA Triton Inference Server model repository template, config.pbtxt, client script, and perf_analyzer commands.")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        model_name = st.text_input("Model name", value="sample_mlp")
    with s2:
        backend = st.selectbox("Backend", ["onnxruntime", "tensorrt", "python", "pytorch"])
    with s3:
        max_batch = st.selectbox("Max batch size", [1, 4, 8, 16, 32, 64], index=3)
    with s4:
        dynamic_batching = st.checkbox("Dynamic batching", value=True)
    sv1, sv2 = st.columns(2)
    with sv1:
        serve_input_dim = st.number_input("Serving input dim", min_value=1, max_value=16384, value=512)
    with sv2:
        serve_output_dim = st.number_input("Serving output dim", min_value=1, max_value=4096, value=32)

    config_text = make_config_pbtxt(model_name, int(max_batch), int(serve_input_dim), int(serve_output_dim), backend, dynamic_batching)
    notes_text = make_deployment_notes(model_name, int(max_batch))
    st.markdown("#### config.pbtxt")
    st.code(config_text, language="protobuf")
    st.markdown("#### Deployment notes")
    st.code(notes_text, language="markdown")
    st.download_button(
        "Download serving repository ZIP",
        data=make_serving_export_zip(model_name, int(max_batch), int(serve_input_dim), int(serve_output_dim), backend, dynamic_batching),
        file_name=f"{model_name}_triton_serving_template.zip",
        mime="application/zip",
        use_container_width=True,
    )

with tabs[6]:
    st.markdown("### Profiler report")
    rec = deployment_recommendation(info.to_dict(), st.session_state.kernel_bench, st.session_state.inference_df)
    st.session_state.recommendation = rec
    report_md = make_markdown_report(info.to_dict(), st.session_state.kernel_bench, st.session_state.inference_df, rec)
    st.markdown("#### Built-in profiler summary")
    st.markdown(f"<div class='panel'>{rec.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

    if client.configured:
        if st.button("Generate AI narrative from benchmark report", use_container_width=True):
            prompt = (
                "You are a senior ML systems engineer. Summarize this GPU/inference benchmark report for a forward deployed engineering context. "
                "Focus on bottlenecks, production risks, kernel choices, serving strategy, and next experiments.\n\n"
                + report_md[:12000]
            )
            st.session_state.ai_narrative = client.summarize(prompt)
    else:
        st.info("Configure local AI narrative mode in .env for generated senior-engineer summaries. Static report is already available below.")

    if st.session_state.ai_narrative:
        st.markdown("#### AI narrative")
        st.markdown(f"<div class='panel-blue'>{st.session_state.ai_narrative}</div>", unsafe_allow_html=True)

    st.markdown("#### Markdown report preview")
    st.code(report_md, language="markdown")

with tabs[7]:
    st.markdown("### Client deployment recommendation")
    rec = st.session_state.recommendation or deployment_recommendation(info.to_dict(), st.session_state.kernel_bench, st.session_state.inference_df)
    st.markdown(f"<div class='panel-orange'>{rec.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
    st.markdown("#### FDE-style implementation plan")
    steps = [
        "Confirm target workload shape, latency SLA, and throughput target.",
        "Capture real production inputs and batch-size distribution.",
        "Benchmark CPU, PyTorch eager, PyTorch CUDA, torch.compile, ONNX Runtime, TensorRT, and custom Triton kernels where relevant.",
        "Use custom kernels only when the operation is not already optimized by vendor libraries or when fusion removes memory traffic.",
        "Export a serving config with dynamic batching and run perf_analyzer before production rollout.",
        "Add runtime monitoring: p50/p95/p99 latency, GPU memory, queue delay, error rate, and fallback rate.",
        "Keep rollback path to PyTorch eager or previous serving config.",
    ]
    for i, step in enumerate(steps, start=1):
        st.markdown(f"<div class='panel'><b>{i}.</b> {step}</div>", unsafe_allow_html=True)

with tabs[8]:
    st.markdown("### Export center")
    rec = st.session_state.recommendation or deployment_recommendation(info.to_dict(), st.session_state.kernel_bench, st.session_state.inference_df)
    report_md = make_markdown_report(info.to_dict(), st.session_state.kernel_bench, st.session_state.inference_df, rec)
    audit_payload = {
        "system": info.to_dict(),
        "kernel_benchmark": st.session_state.kernel_bench.to_dict(orient="records") if not st.session_state.kernel_bench.empty else [],
        "autotune": st.session_state.autotune_df.to_dict(orient="records") if not st.session_state.autotune_df.empty else [],
        "inference_benchmark": st.session_state.inference_df.to_dict(orient="records") if not st.session_state.inference_df.empty else [],
        "recommendation": rec,
    }
    path = write_json(OUTPUT_DIR / f"kernelforge_audit_{now_id()}.json", audit_payload)
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.download_button("Download audit JSON", data=json.dumps(audit_payload, indent=2, default=str), file_name="kernelforge_audit.json", mime="application/json", use_container_width=True)
    with e2:
        st.download_button("Download report MD", data=report_md, file_name="kernelforge_performance_report.md", mime="text/markdown", use_container_width=True)
    with e3:
        if not st.session_state.kernel_bench.empty:
            st.download_button("Download kernel CSV", data=st.session_state.kernel_bench.to_csv(index=False), file_name="kernelforge_kernel_benchmark.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("No kernel CSV yet", disabled=True, use_container_width=True)
    with e4:
        if not st.session_state.inference_df.empty:
            st.download_button("Download inference CSV", data=st.session_state.inference_df.to_csv(index=False), file_name="kernelforge_inference_benchmark.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("No inference CSV yet", disabled=True, use_container_width=True)
    st.caption(f"Latest local audit saved at: {path}")
