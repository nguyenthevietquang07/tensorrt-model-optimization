# Real Data Benchmark

## Purpose

This benchmark replaces toy-only timing with a real tabular inference workload.
It uses the UCI Iris dataset to measure local CPU inference latency and accuracy
before any ONNX/TensorRT acceleration claim is attempted.

## Source

- Provider: UCI Machine Learning Repository
- Dataset: Iris
- URL: `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`

## Command

```bash
python scripts/iris_real_data_benchmark.py --trials 200
```

## Processing Steps

1. Fetch the Iris CSV.
2. Parse 4 numeric features and class labels.
3. Split each class into train and holdout samples.
4. Train a deterministic nearest-centroid classifier.
5. Run repeated local inference trials over the holdout set.
6. Save latency, accuracy, and dataset metadata to `reports/iris_real_data_benchmark.json`.

## Latest Measurements

| Metric | Value |
|---|---:|
| Samples processed | 150 |
| Train/test split | 120 / 30 |
| Feature count | 4 |
| Class count | 3 |
| Accuracy | 0.966667 |
| Mean latency | 0.1000 ms |
| p95 latency | 0.1575 ms |
| Trials | 200 |

## Claim Boundary

This proves reproducible local inference benchmarking with real data. It does
not prove TensorRT speedup until a comparable GPU/ONNX/TensorRT report is saved
from the same workload boundary.
