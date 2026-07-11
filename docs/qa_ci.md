# QA and CI Plan

## Test strategy

- Unit tests cover benchmark configuration and report structure.
- Local fallback benchmark must run without GPU dependencies.
- Optional GPU/TensorRT benchmarks must write hardware, model, input shape, and
  trial count to `reports/`.
- Benchmark reports must include the claim boundary that prevents local CPU
  fallback numbers from being described as TensorRT speedups.

## Metric policy

Only use speedup numbers if the report includes:

- baseline runtime
- optimized runtime
- hardware
- batch size
- input shape
- number of trials
- model version

## CI

CI runs the standard-library unit tests. GPU acceleration is intentionally not
required for CI.
