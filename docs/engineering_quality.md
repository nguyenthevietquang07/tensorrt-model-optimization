# Engineering Quality

This project focuses on reproducible inference benchmarking and careful
separation between CPU, GPU, and TensorRT evidence.

## Delivered Practices

- Local CPU benchmark harness with JSON report output.
- Real UCI Iris benchmark with latency and accuracy measurements.
- PyTorch-to-ONNX export path with ONNX Runtime CPU benchmark.
- Correctness comparison between PyTorch logits and ONNX Runtime logits.
- Comparable-report validation before using any speedup metric.
- Colab GPU notebook and artifact validator for CUDA/TensorRT evidence.
- Saved Colab TensorRT artifact set with TensorRT provider, correctness, and
  comparable-setting validation.
- Unit tests and GitHub Actions CI.

## Evidence Rules

- CPU reports prove local CPU behavior only.
- CUDA provider reports prove GPU inference behavior, not TensorRT speedup.
- TensorRT wording requires a saved report with
  `selected_provider: "TensorrtExecutionProvider"`, comparable settings, and
  passing correctness validation.
