from __future__ import annotations

import statistics


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * percentile_value) - 1))
    return sorted_values[index]


def build_report(
    backend: str,
    durations_ms: list[float],
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    if not durations_ms:
        raise ValueError("durations_ms cannot be empty")
    return {
        "backend": backend,
        "status": "benchmark_report_not_speedup_claim",
        "trials": len(durations_ms),
        "mean_ms": round(statistics.mean(durations_ms), 4),
        "p50_ms": round(percentile(durations_ms, 0.50), 4),
        "p95_ms": round(percentile(durations_ms, 0.95), 4),
        "max_ms": round(max(durations_ms), 4),
        "metadata": metadata or {},
        "claim_boundary": (
            "Use this report as measurement evidence for the named backend only; "
            "do not claim TensorRT speedup until baseline and optimized reports "
            "are saved from comparable hardware and input settings."
        ),
    }
