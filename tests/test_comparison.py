import unittest

from src.modelopt.comparison import compare_reports


class ComparisonTests(unittest.TestCase):
    def test_compare_reports_requires_comparable_metadata(self):
        baseline = {
            "backend": "pytorch",
            "mean_ms": 10.0,
            "p95_ms": 12.0,
            "metadata": {"hardware": "gpu-a", "trials": 100, "vector_size": 256},
        }
        candidate = {
            "backend": "tensorrt",
            "mean_ms": 5.0,
            "p95_ms": 6.0,
            "metadata": {"hardware": "gpu-a", "trials": 100, "vector_size": 256},
        }

        report = compare_reports(baseline, candidate)

        self.assertTrue(report["comparable_settings"])
        self.assertEqual(report["mean_speedup"], 2.0)
        self.assertIn("Only use speedup", report["claim_boundary"])

    def test_compare_reports_flags_mismatched_hardware(self):
        baseline = {
            "backend": "pytorch",
            "mean_ms": 10.0,
            "p95_ms": 12.0,
            "metadata": {"hardware": "cpu", "trials": 100, "vector_size": 256},
        }
        candidate = {
            "backend": "onnx",
            "mean_ms": 5.0,
            "p95_ms": 6.0,
            "metadata": {"hardware": "gpu", "trials": 100, "vector_size": 256},
        }

        report = compare_reports(baseline, candidate)

        self.assertFalse(report["comparable_settings"])
