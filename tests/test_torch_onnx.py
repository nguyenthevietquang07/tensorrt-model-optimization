import tempfile
import unittest
from pathlib import Path

from src.modelopt.real_data import parse_iris_csv, split_by_class
from src.modelopt.torch_onnx import (
    TorchOnnxConfig,
    benchmark_pytorch,
    compare_correctness,
    export_onnx_model,
    prepare_iris_tensors,
    train_iris_mlp,
)


def iris_fixture() -> str:
    rows: list[str] = []
    for index in range(14):
        rows.append(f"{5.0 + index * 0.01},3.5,1.4,0.2,Iris-setosa")
        rows.append(f"{6.0 + index * 0.01},2.8,4.5,1.4,Iris-versicolor")
        rows.append(f"{6.5 + index * 0.01},3.0,5.5,2.0,Iris-virginica")
    return "\n".join(rows)


class TorchOnnxTests(unittest.TestCase):
    def test_pytorch_baseline_report_contains_model_metadata(self):
        train, test = split_by_class(parse_iris_csv(iris_fixture()), holdout_per_class=2)
        data = prepare_iris_tensors(train, test)
        model = train_iris_mlp(data, TorchOnnxConfig(epochs=5, trials=3, batch_size=6))
        report = benchmark_pytorch(model, data, TorchOnnxConfig(epochs=5, trials=3, batch_size=6))

        self.assertEqual(report["backend"], "pytorch_cpu_mlp")
        self.assertEqual(report["metadata"]["model"], "iris_mlp")
        self.assertEqual(report["metadata"]["batch_size"], 6)

    def test_onnx_export_and_correctness_report(self):
        train, test = split_by_class(parse_iris_csv(iris_fixture()), holdout_per_class=2)
        data = prepare_iris_tensors(train, test)
        model = train_iris_mlp(data, TorchOnnxConfig(epochs=5, trials=3, batch_size=6))
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = export_onnx_model(model, data, Path(temp_dir) / "iris.onnx")
            report = compare_correctness(model, onnx_path, data)

        self.assertTrue(report["passed"])
        self.assertEqual(report["prediction_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
