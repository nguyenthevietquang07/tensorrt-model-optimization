from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .real_data import IrisSample, score_accuracy
from .reporting import build_report


@dataclass(frozen=True)
class TorchOnnxConfig:
    epochs: int = 160
    trials: int = 100
    batch_size: int = 30
    learning_rate: float = 0.03
    onnx_path: str = "models/iris_mlp.onnx"


class IrisMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 12, output_dim: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class PreparedIrisData:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    label_names: list[str]
    mean: torch.Tensor
    std: torch.Tensor


def prepare_iris_tensors(train: list[IrisSample], test: list[IrisSample]) -> PreparedIrisData:
    label_names = sorted({sample.label for sample in train + test})
    label_to_index = {label: index for index, label in enumerate(label_names)}
    train_features = torch.tensor([sample.features for sample in train], dtype=torch.float32)
    test_features = torch.tensor([sample.features for sample in test], dtype=torch.float32)
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return PreparedIrisData(
        train_features=(train_features - mean) / std,
        train_labels=torch.tensor([label_to_index[sample.label] for sample in train], dtype=torch.long),
        test_features=(test_features - mean) / std,
        test_labels=torch.tensor([label_to_index[sample.label] for sample in test], dtype=torch.long),
        label_names=label_names,
        mean=mean,
        std=std,
    )


def train_iris_mlp(data: PreparedIrisData, config: TorchOnnxConfig) -> IrisMLP:
    torch.manual_seed(7)
    model = IrisMLP(output_dim=len(data.label_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        logits = model(data.train_features)
        loss = loss_fn(logits, data.train_labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def pytorch_accuracy(model: IrisMLP, data: PreparedIrisData) -> float:
    with torch.no_grad():
        predictions = model(data.test_features).argmax(dim=1)
    return float((predictions == data.test_labels).float().mean().item())


def export_onnx_model(model: IrisMLP, data: PreparedIrisData, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        data.test_features[:1],
        str(path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def benchmark_pytorch(model: IrisMLP, data: PreparedIrisData, config: TorchOnnxConfig) -> dict[str, object]:
    batch = data.test_features[: config.batch_size]
    durations_ms: list[float] = []
    checksum = 0.0
    with torch.no_grad():
        for _ in range(config.trials):
            start = time.perf_counter()
            logits = model(batch)
            durations_ms.append((time.perf_counter() - start) * 1000)
            checksum += float(logits.sum().item())
    return build_report(
        backend="pytorch_cpu_mlp",
        durations_ms=durations_ms,
        metadata={
            "hardware": "local_cpu",
            "trials": config.trials,
            "vector_size": 4,
            "batch_size": int(batch.shape[0]),
            "model": "iris_mlp",
            "accuracy": round(pytorch_accuracy(model, data), 6),
            "checksum": round(checksum, 6),
        },
    )


def benchmark_onnxruntime(onnx_path: str | Path, data: PreparedIrisData, config: TorchOnnxConfig) -> dict[str, object]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = data.test_features[: config.batch_size].numpy().astype(np.float32)
    durations_ms: list[float] = []
    checksum = 0.0
    for _ in range(config.trials):
        start = time.perf_counter()
        logits = session.run(["logits"], {"features": batch})[0]
        durations_ms.append((time.perf_counter() - start) * 1000)
        checksum += float(logits.sum())
    return build_report(
        backend="onnxruntime_cpu",
        durations_ms=durations_ms,
        metadata={
            "hardware": "local_cpu",
            "trials": config.trials,
            "vector_size": 4,
            "batch_size": int(batch.shape[0]),
            "model": "iris_mlp",
            "accuracy": round(onnxruntime_accuracy(session, data), 6),
            "checksum": round(checksum, 6),
        },
    )


def onnxruntime_accuracy(session: ort.InferenceSession, data: PreparedIrisData) -> float:
    logits = session.run(["logits"], {"features": data.test_features.numpy().astype(np.float32)})[0]
    predictions = logits.argmax(axis=1)
    return float((predictions == data.test_labels.numpy()).mean())


def compare_correctness(model: IrisMLP, onnx_path: str | Path, data: PreparedIrisData) -> dict[str, object]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_logits = model(data.test_features).numpy()
    ort_logits = session.run(["logits"], {"features": data.test_features.numpy().astype(np.float32)})[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    torch_predictions = torch_logits.argmax(axis=1)
    ort_predictions = ort_logits.argmax(axis=1)
    prediction_agreement = float((torch_predictions == ort_predictions).mean())
    return {
        "project": "tensorrt-model-optimization",
        "stage": "onnx_correctness",
        "baseline_backend": "pytorch_cpu_mlp",
        "candidate_backend": "onnxruntime_cpu",
        "max_abs_diff": round(max_abs_diff, 8),
        "prediction_agreement": round(prediction_agreement, 6),
        "passed": max_abs_diff <= 1e-4 and prediction_agreement == 1.0,
        "claim_boundary": "Correctness report compares PyTorch and ONNX Runtime logits on the same local CPU test batch.",
    }
