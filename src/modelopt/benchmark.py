from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from dataclasses import dataclass

from .reporting import build_report


@dataclass(frozen=True)
class BenchmarkConfig:
    trials: int = 50
    vector_size: int = 256
    backend: str = "local_cpu_fallback"


def _toy_inference(vector_size: int) -> float:
    values = [((idx % 17) - 8) / 17 for idx in range(vector_size)]
    weights = [((idx % 13) - 6) / 13 for idx in range(vector_size)]
    return sum(value * weight for value, weight in zip(values, weights))


def run_fallback_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    if config.trials <= 0:
        raise ValueError("trials must be positive")
    durations_ms: list[float] = []
    checksum = 0.0
    for _ in range(config.trials):
        start = time.perf_counter()
        checksum += _toy_inference(config.vector_size)
        durations_ms.append((time.perf_counter() - start) * 1000)
    return build_report(
        backend=config.backend,
        durations_ms=durations_ms,
        metadata={
            "vector_size": config.vector_size,
            "trials": config.trials,
            "checksum": round(checksum, 6),
            "hardware": "local_cpu",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local CPU fallback benchmark.")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--vector-size", type=int, default=256)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    report = run_fallback_benchmark(BenchmarkConfig(trials=args.trials, vector_size=args.vector_size))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
