from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modelopt.comparison import compare_reports  # noqa: E402
from src.modelopt.real_data import fetch_iris_dataset, split_by_class  # noqa: E402
from src.modelopt.torch_onnx import (  # noqa: E402
    TorchOnnxConfig,
    benchmark_onnxruntime,
    benchmark_pytorch,
    compare_correctness,
    export_onnx_model,
    prepare_iris_tensors,
    train_iris_mlp,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_demo() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    config = TorchOnnxConfig()
    samples = fetch_iris_dataset()
    train, test = split_by_class(samples)
    data = prepare_iris_tensors(train, test)
    model = train_iris_mlp(data, config)
    onnx_path = export_onnx_model(model, data, root / config.onnx_path)
    pytorch_report = benchmark_pytorch(model, data, config)
    onnx_report = benchmark_onnxruntime(onnx_path, data, config)
    correctness_report = compare_correctness(model, onnx_path, data)
    comparison_report = compare_reports(pytorch_report, onnx_report)

    reports = {
        "reports/pytorch_baseline_report.json": pytorch_report,
        "reports/onnxruntime_report.json": onnx_report,
        "reports/onnx_correctness_report.json": correctness_report,
        "reports/onnx_comparison_report.json": comparison_report,
    }
    for relative_path, payload in reports.items():
        write_json(root / relative_path, payload)

    checks = {
        "pytorch_accuracy_recorded": pytorch_report["metadata"].get("accuracy", 0) >= 0.8,
        "onnx_accuracy_recorded": onnx_report["metadata"].get("accuracy", 0) >= 0.8,
        "onnx_export_exists": onnx_path.exists(),
        "correctness_passed": correctness_report["passed"] is True,
        "comparison_is_comparable": comparison_report["comparable_settings"] is True,
        "candidate_backend_is_onnxruntime": comparison_report["candidate_backend"] == "onnxruntime_cpu",
    }
    return {
        "project": "tensorrt-model-optimization",
        "stage": "torch_onnx_demo",
        "onnx_path": str(onnx_path.relative_to(root)),
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": (
            "This demo verifies PyTorch CPU baseline, ONNX export, ONNX Runtime CPU inference, "
            "and correctness comparison. It is not TensorRT GPU speedup evidence."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = run_demo()
    output = root / "reports" / "torch_onnx_demo.json"
    write_json(output, report)
    print(json.dumps({"report_path": str(output.relative_to(root)), "summary": report}, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
