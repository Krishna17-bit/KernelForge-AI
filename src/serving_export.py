from __future__ import annotations

import io
import zipfile
from pathlib import Path


def make_config_pbtxt(model_name: str, max_batch_size: int, input_dim: int, output_dim: int, backend: str = "onnxruntime", dynamic_batching: bool = True) -> str:
    batching = f'''
dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 1000
}}
''' if dynamic_batching else ""
    return f'''name: "{model_name}"
backend: "{backend}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ {input_dim} ]
  }}
]

output [
  {{
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ {output_dim} ]
  }}
]
{batching}
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
'''


def make_client_py(model_name: str, input_dim: int) -> str:
    return f'''# Minimal NVIDIA Triton Inference Server HTTP client example
# pip install tritonclient[http] numpy
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

MODEL_NAME = "{model_name}"
INPUT_NAME = "INPUT__0"
OUTPUT_NAME = "OUTPUT__0"

client = httpclient.InferenceServerClient(url="localhost:8000")
x = np.random.randn(1, {input_dim}).astype(np.float32)
inputs = [httpclient.InferInput(INPUT_NAME, x.shape, np_to_triton_dtype(x.dtype))]
inputs[0].set_data_from_numpy(x)
outputs = [httpclient.InferRequestedOutput(OUTPUT_NAME)]
result = client.infer(MODEL_NAME, inputs=inputs, outputs=outputs)
print(result.as_numpy(OUTPUT_NAME))
'''


def make_deployment_notes(model_name: str, max_batch_size: int) -> str:
    return f'''# Deployment Notes for {model_name}

## Repository layout

```text
model_repository/
└── {model_name}/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

## Run NVIDIA Triton Inference Server

```bash
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 \\
  -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:latest \\
  tritonserver --model-repository=/models
```

## Basic health check

```bash
curl localhost:8000/v2/health/ready
curl localhost:8000/v2/models/{model_name}
```

## Performance Analyzer

```bash
perf_analyzer -m {model_name} -u localhost:8000 --concurrency-range 1:8 --batch-size 1
perf_analyzer -m {model_name} -u localhost:8000 --concurrency-range 1:16 --batch-size {max_batch_size}
```

## Optimization ideas

- Enable dynamic batching for throughput workloads.
- Keep batch size small for low-latency interactive workloads.
- Test multiple instance groups if GPU has unused memory.
- Compare ONNX Runtime, TensorRT, and Python backends.
- Profile p50/p95 latency and throughput, not average latency only.
'''


def make_serving_export_zip(model_name: str, max_batch_size: int, input_dim: int, output_dim: int, backend: str, dynamic_batching: bool) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"model_repository/{model_name}/config.pbtxt", make_config_pbtxt(model_name, max_batch_size, input_dim, output_dim, backend, dynamic_batching))
        zf.writestr(f"model_repository/{model_name}/1/README_PLACE_MODEL_ONNX_HERE.txt", "Place your model.onnx file in this directory.\n")
        zf.writestr("client.py", make_client_py(model_name, input_dim))
        zf.writestr("DEPLOYMENT_NOTES.md", make_deployment_notes(model_name, max_batch_size))
    mem.seek(0)
    return mem.getvalue()
