# Colab GPU Runbook

Use this runbook to produce the GPU artifact that closes the remaining launch
gap for this project.

## Runtime

1. Open `notebooks/colab_training_plan.ipynb` in Google Colab.
2. Select **Runtime > Change runtime type > GPU**.
3. Run all cells.
4. Download `colab_gpu_artifacts.zip`.
5. Extract the JSON files into this repo's `reports/` directory.
6. Run:

```bash
python scripts/validate_colab_gpu_artifacts.py
```

## Expected Artifacts

- `reports/colab_gpu_environment.json`
- `reports/colab_gpu_pytorch_report.json`
- `reports/colab_gpu_candidate_report.json`
- `reports/colab_gpu_correctness_report.json`
- `reports/colab_gpu_comparison_report.json`
- `reports/colab_gpu_validation.json`

## Claim Boundary

GPU evidence is not automatically TensorRT evidence.

Use TensorRT-specific wording only when:

- `reports/colab_gpu_environment.json` has
  `selected_provider: "TensorrtExecutionProvider"`;
- `reports/colab_gpu_comparison_report.json` has
  `comparable_settings: true`;
- `reports/colab_gpu_correctness_report.json` has `passed: true`;
- `reports/colab_gpu_validation.json` has `passed: true`.

If the selected provider is `CUDAExecutionProvider`, describe the result as
ONNX Runtime GPU evidence and keep TensorRT wording as a planned extension.
