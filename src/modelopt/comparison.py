from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_METADATA_KEYS = {"hardware", "trials", "vector_size"}


def load_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_report(report)
    return report


def validate_report(report: dict[str, Any]) -> None:
    for field in ("backend", "mean_ms", "p95_ms", "metadata"):
        if field not in report:
            raise ValueError(f"Missing benchmark report field: {field}")
    missing_metadata = REQUIRED_METADATA_KEYS - set(report["metadata"])
    if missing_metadata:
        raise ValueError(f"Missing metadata fields: {sorted(missing_metadata)}")


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    validate_report(baseline)
    validate_report(candidate)
    comparable = _comparable_metadata(baseline["metadata"], candidate["metadata"])
    mean_speedup = baseline["mean_ms"] / candidate["mean_ms"] if candidate["mean_ms"] else 0.0
    p95_speedup = baseline["p95_ms"] / candidate["p95_ms"] if candidate["p95_ms"] else 0.0

    return {
        "baseline_backend": baseline["backend"],
        "candidate_backend": candidate["backend"],
        "comparable_settings": comparable,
        "mean_speedup": round(mean_speedup, 4),
        "p95_speedup": round(p95_speedup, 4),
        "claim_boundary": (
            "Only use speedup values when comparable_settings is true and both "
            "reports were collected from equivalent hardware, model, batch, and "
            "input settings."
        ),
    }


def _comparable_metadata(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return all(baseline.get(key) == candidate.get(key) for key in REQUIRED_METADATA_KEYS)
