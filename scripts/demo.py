from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modelopt.benchmark import BenchmarkConfig, run_fallback_benchmark


def run_demo() -> dict[str, object]:
    report = run_fallback_benchmark(BenchmarkConfig(trials=10, vector_size=256))
    report_path = Path("reports/demo_benchmark.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"report_path": str(report_path), "summary": report}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))
