from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load_report(name: str, reports_dir: Path = REPORTS) -> dict[str, Any]:
    path = reports_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing Colab artifact: reports/{name}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(reports_dir: Path = REPORTS) -> dict[str, Any]:
    environment = load_report("colab_gpu_environment.json", reports_dir)
    pytorch = load_report("colab_gpu_pytorch_report.json", reports_dir)
    candidate = load_report("colab_gpu_candidate_report.json", reports_dir)
    correctness = load_report("colab_gpu_correctness_report.json", reports_dir)
    comparison = load_report("colab_gpu_comparison_report.json", reports_dir)

    candidate_provider = candidate.get("metadata", {}).get("provider")
    selected_provider = environment.get("selected_provider")
    provider_consistent = selected_provider != "TensorrtExecutionProvider" or candidate_provider == selected_provider
    checks = {
        "cuda_available": environment.get("cuda_available") is True,
        "provider_is_gpu": candidate.get("backend") in {"onnxruntime_cuda", "onnxruntime_tensorrt"},
        "provider_consistent": provider_consistent,
        "minimum_trials": min(int(pytorch.get("trials", 0)), int(candidate.get("trials", 0))) >= 50,
        "comparable_settings": comparison.get("comparable_settings") is True,
        "correctness_passed": correctness.get("passed") is True,
        "finite_speedup": float(comparison.get("mean_speedup", 0.0)) > 0,
    }
    caveats: list[str] = []
    if selected_provider != "TensorrtExecutionProvider":
        caveats.append(
            "TensorRT provider was not selected; artifacts support GPU/ONNX Runtime evidence, not TensorRT speedup claims."
        )
    if not provider_consistent:
        caveats.append("Selected TensorRT provider does not match the candidate report provider metadata.")
    if float(comparison.get("mean_speedup", 0.0)) > 25:
        caveats.append("Mean speedup is very large; manually inspect warmup, synchronization, and timing methodology.")

    report = {
        "project": "tensorrt-model-optimization",
        "stage": "colab_gpu_artifact_validation",
        "checks": checks,
        "passed": all(checks.values()),
        "selected_provider": selected_provider,
        "gpu_name": environment.get("gpu_name"),
        "mean_speedup": comparison.get("mean_speedup"),
        "p95_speedup": comparison.get("p95_speedup"),
        "caveats": caveats,
        "claim_boundary": (
            "Passing validation means Colab GPU artifacts are structurally usable. "
            "TensorRT-specific claims require selected_provider == TensorrtExecutionProvider."
        ),
    }
    return report


def main() -> None:
    report = validate_artifacts(REPORTS)
    output = REPORTS / "colab_gpu_validation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
