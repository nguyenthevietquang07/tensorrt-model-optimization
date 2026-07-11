import unittest

from src.modelopt.benchmark import BenchmarkConfig, run_fallback_benchmark
from src.modelopt.reporting import build_report


class ReportingTests(unittest.TestCase):
    def test_report_has_latency_fields(self):
        report = build_report("test", [1.0, 2.0, 3.0], {"hardware": "ci"})

        self.assertEqual(report["backend"], "test")
        self.assertEqual(report["status"], "benchmark_report_not_speedup_claim")
        self.assertEqual(report["trials"], 3)
        self.assertIn("p95_ms", report)
        self.assertEqual(report["metadata"]["hardware"], "ci")
        self.assertIn("do not claim TensorRT speedup", report["claim_boundary"])

    def test_fallback_benchmark_runs_without_gpu(self):
        report = run_fallback_benchmark(BenchmarkConfig(trials=3, vector_size=16))

        self.assertEqual(report["backend"], "local_cpu_fallback")
        self.assertEqual(report["trials"], 3)
        self.assertGreaterEqual(report["mean_ms"], 0)


if __name__ == "__main__":
    unittest.main()
