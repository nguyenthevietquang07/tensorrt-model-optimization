# TensorRT Model Optimization

ML systems starter repo for benchmarking model-export and inference paths. The
local code is dependency-light and CPU-safe; the Colab notebook path is where
GPU training or TensorRT-specific work should happen.

## Why this exists

The project supports ML engineering and computer-vision resumes by showing
performance thinking: baseline timing, export boundaries, reproducible
benchmarks, and clear fallback behavior when TensorRT is unavailable.

## Features

- Benchmark result schema
- Local CPU fallback benchmark
- Report comparison utility with comparable-setting checks
- Optional ONNX Runtime and TensorRT extension points
- Colab-first training/acceleration notebook plan
- CI tests for benchmark reporting and configuration
- QA notes for avoiding unverifiable speedup claims

## Quickstart

```bash
python -m src.modelopt.benchmark --trials 25
python -m src.modelopt.benchmark --trials 25 --output reports/local_cpu_report.json
python scripts/compare_reports.py --baseline reports/local_cpu_report.json --candidate reports/local_cpu_report.json
python -m unittest discover -s tests
```

## Colab workflow

Use `notebooks/colab_training_plan.ipynb` when training or GPU acceleration is
needed. Save exported model artifacts and benchmark logs under `reports/` before
using any speedup metric on a resume.

## Resume-safe claim

Built an ML optimization scaffold for benchmarking baseline inference, export
paths, and optional TensorRT/ONNX acceleration, with CI tests and a Colab-first
GPU workflow for reproducible measurements.

Do not claim a TensorRT speedup until a saved Colab or local GPU report exists.
