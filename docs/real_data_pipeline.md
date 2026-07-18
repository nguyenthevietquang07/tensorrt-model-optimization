# Real Data Benchmark

## Purpose

This benchmark replaces toy-only timing with real local inference workloads.
It uses UCI Iris for a small tabular pipeline and scikit-learn handwritten
digits for a larger offline image-like workload before any ONNX/TensorRT
acceleration claim is attempted.

## Source

- Provider: UCI Machine Learning Repository
- Dataset: Iris
- URL: `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`
- Provider: scikit-learn bundled datasets
- Dataset: handwritten digits
- Source: `sklearn.datasets.load_digits`

## Command

```bash
python scripts/iris_real_data_benchmark.py --trials 200
python scripts/digits_real_data_benchmark.py --trials 200
```

## Processing Steps

1. Fetch the Iris CSV.
2. Parse 4 numeric features and class labels.
3. Split each class into train and holdout samples.
4. Train a deterministic nearest-centroid classifier.
5. Run repeated local inference trials over the holdout set.
6. Save latency, accuracy, and dataset metadata to `reports/iris_real_data_benchmark.json`.
7. Load the offline digits dataset, split by class, train a deterministic
   nearest-centroid classifier, and save `reports/digits_real_data_benchmark.json`.

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

## Latest Digits Measurements

| Metric | Value |
|---|---:|
| Samples processed | 1797 |
| Train/test split | 1547 / 250 |
| Feature count | 64 |
| Class count | 10 |
| Accuracy | 0.860000 |
| Mean latency | 18.8749 ms |
| p95 latency | 23.9109 ms |
| Trials | 200 |

## Claim Boundary

This proves reproducible local inference benchmarking with named datasets. It
does not prove TensorRT speedup for those workloads until comparable
GPU/ONNX/TensorRT reports are saved from the same workload boundary.
