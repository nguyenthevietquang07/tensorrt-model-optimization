import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_colab_gpu_artifacts import validate_artifacts

ROOT = Path(__file__).resolve().parents[1]


def write_report(reports_dir: Path, name: str, payload: dict[str, object]) -> None:
    (reports_dir / name).write_text(json.dumps(payload), encoding="utf-8")


class ColabArtifactValidationTests(unittest.TestCase):
    def test_validation_passes_for_cuda_artifacts_with_caveat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_report(
                reports_dir,
                "colab_gpu_environment.json",
                {
                    "cuda_available": True,
                    "selected_provider": "CUDAExecutionProvider",
                    "gpu_name": "T4",
                },
            )
            write_report(reports_dir, "colab_gpu_pytorch_report.json", {"trials": 100})
            write_report(
                reports_dir,
                "colab_gpu_candidate_report.json",
                {"backend": "onnxruntime_cuda", "trials": 100},
            )
            write_report(reports_dir, "colab_gpu_correctness_report.json", {"passed": True})
            write_report(
                reports_dir,
                "colab_gpu_comparison_report.json",
                {"comparable_settings": True, "mean_speedup": 1.4, "p95_speedup": 1.2},
            )

            report = validate_artifacts(reports_dir)

        self.assertTrue(report["passed"])
        self.assertIn("not TensorRT speedup claims", report["caveats"][0])

    def test_validation_rejects_cpu_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_report(
                reports_dir,
                "colab_gpu_environment.json",
                {
                    "cuda_available": True,
                    "selected_provider": "CPUExecutionProvider",
                    "gpu_name": "T4",
                },
            )
            write_report(reports_dir, "colab_gpu_pytorch_report.json", {"trials": 100})
            write_report(
                reports_dir,
                "colab_gpu_candidate_report.json",
                {"backend": "onnxruntime_cpu", "trials": 100},
            )
            write_report(reports_dir, "colab_gpu_correctness_report.json", {"passed": True})
            write_report(
                reports_dir,
                "colab_gpu_comparison_report.json",
                {"comparable_settings": True, "mean_speedup": 1.0, "p95_speedup": 1.0},
            )

            report = validate_artifacts(reports_dir)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["provider_is_gpu"])

    def test_committed_colab_tensorrt_artifacts_validate(self):
        report = validate_artifacts(ROOT / "reports")

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_provider"], "TensorrtExecutionProvider")
        self.assertEqual(report["caveats"], [])
        self.assertGreater(report["mean_speedup"], 1.0)


if __name__ == "__main__":
    unittest.main()
