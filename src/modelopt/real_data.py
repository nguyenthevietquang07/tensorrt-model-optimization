from __future__ import annotations

import csv
import io
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .reporting import build_report


IRIS_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


@dataclass(frozen=True)
class IrisSample:
    features: tuple[float, float, float, float]
    label: str


@dataclass(frozen=True)
class CentroidModel:
    centroids: dict[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class IrisBenchmarkConfig:
    trials: int = 200
    output_path: str = "reports/iris_real_data_benchmark.json"


def fetch_iris_dataset(timeout_seconds: float = 20.0) -> list[IrisSample]:
    with urllib.request.urlopen(IRIS_DATA_URL, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    return parse_iris_csv(raw)


def load_iris_dataset(path: str | Path) -> list[IrisSample]:
    return parse_iris_csv(Path(path).read_text(encoding="utf-8"))


def parse_iris_csv(raw_csv: str) -> list[IrisSample]:
    samples: list[IrisSample] = []
    reader = csv.reader(io.StringIO(raw_csv))
    for row in reader:
        if not row:
            continue
        if len(row) != 5:
            raise ValueError(f"Expected 5 Iris columns, got {len(row)}")
        samples.append(
            IrisSample(
                features=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                label=row[4],
            )
        )
    if len(samples) < 30:
        raise ValueError("Iris benchmark requires at least 30 samples")
    return samples


def split_by_class(samples: Iterable[IrisSample], holdout_per_class: int = 10) -> tuple[list[IrisSample], list[IrisSample]]:
    grouped: dict[str, list[IrisSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    train: list[IrisSample] = []
    test: list[IrisSample] = []
    for label, group in sorted(grouped.items()):
        if len(group) <= holdout_per_class:
            raise ValueError(f"Not enough samples for class {label}")
        train.extend(group[:-holdout_per_class])
        test.extend(group[-holdout_per_class:])
    return train, test


def train_centroid_model(samples: list[IrisSample]) -> CentroidModel:
    grouped: dict[str, list[IrisSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    centroids: dict[str, tuple[float, float, float, float]] = {}
    for label, group in grouped.items():
        centroids[label] = tuple(
            sum(sample.features[index] for sample in group) / len(group)
            for index in range(4)
        )  # type: ignore[assignment]
    return CentroidModel(centroids=centroids)


def predict(model: CentroidModel, features: tuple[float, float, float, float]) -> str:
    best_label = ""
    best_distance = float("inf")
    for label, centroid in model.centroids.items():
        distance = sum((value - expected) ** 2 for value, expected in zip(features, centroid))
        if distance < best_distance:
            best_label = label
            best_distance = distance
    return best_label


def score_accuracy(model: CentroidModel, samples: list[IrisSample]) -> float:
    correct = sum(1 for sample in samples if predict(model, sample.features) == sample.label)
    return correct / len(samples)


def run_iris_real_data_benchmark(
    config: IrisBenchmarkConfig,
    samples: list[IrisSample] | None = None,
) -> dict[str, object]:
    fetch_started = time.perf_counter()
    dataset = samples if samples is not None else fetch_iris_dataset()
    fetch_seconds = time.perf_counter() - fetch_started

    train, test = split_by_class(dataset)
    model = train_centroid_model(train)
    accuracy = score_accuracy(model, test)

    durations_ms: list[float] = []
    checksum = 0
    for _ in range(config.trials):
        start = time.perf_counter()
        predictions = [predict(model, sample.features) for sample in test]
        durations_ms.append((time.perf_counter() - start) * 1000)
        checksum += sum(len(label) for label in predictions)

    report = build_report(
        backend="local_cpu_centroid_classifier",
        durations_ms=durations_ms,
        metadata={
            "dataset": "UCI Iris",
            "source_url": IRIS_DATA_URL,
            "sample_count": len(dataset),
            "train_samples": len(train),
            "test_samples": len(test),
            "feature_count": 4,
            "class_count": len(model.centroids),
            "accuracy": round(accuracy, 6),
            "fetch_seconds": round(fetch_seconds, 4),
            "checksum": checksum,
        },
    )
    report["pipeline"] = "iris_real_data_inference_benchmark"
    report["passion_project_note"] = (
        "Built to practice reproducible model-inference benchmarking on real data "
        "before moving the same measurement discipline to ONNX/TensorRT artifacts."
    )
    return report
