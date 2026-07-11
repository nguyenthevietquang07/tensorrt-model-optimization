from __future__ import annotations


def describe_export_paths() -> list[dict[str, str]]:
    return [
        {
            "name": "pytorch_baseline",
            "status": "planned_colab",
            "notes": "Run baseline model inference in Colab and save timing report.",
        },
        {
            "name": "onnx_runtime",
            "status": "planned_colab",
            "notes": "Export model to ONNX and benchmark with ONNX Runtime.",
        },
        {
            "name": "tensorrt_fp16_int8",
            "status": "optional_gpu",
            "notes": "Use TensorRT only when compatible GPU/runtime support is available.",
        },
    ]
