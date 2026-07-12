import unittest

from src.modelopt.real_data import (
    IrisBenchmarkConfig,
    parse_iris_csv,
    run_iris_real_data_benchmark,
    score_accuracy,
    split_by_class,
    train_centroid_model,
)


def iris_fixture() -> str:
    rows: list[str] = []
    for index in range(12):
        rows.append(f"{5.0 + index * 0.01},3.5,1.4,0.2,Iris-setosa")
        rows.append(f"{6.0 + index * 0.01},2.8,4.5,1.4,Iris-versicolor")
        rows.append(f"{6.5 + index * 0.01},3.0,5.5,2.0,Iris-virginica")
    return "\n".join(rows)


class RealDataBenchmarkTests(unittest.TestCase):
    def test_iris_parser_and_split(self):
        samples = parse_iris_csv(iris_fixture())
        train, test = split_by_class(samples, holdout_per_class=2)

        self.assertEqual(len(samples), 36)
        self.assertEqual(len(train), 30)
        self.assertEqual(len(test), 6)

    def test_centroid_model_scores_fixture(self):
        samples = parse_iris_csv(iris_fixture())
        train, test = split_by_class(samples, holdout_per_class=2)
        model = train_centroid_model(train)

        self.assertEqual(score_accuracy(model, test), 1.0)

    def test_real_data_benchmark_report_has_measurements(self):
        report = run_iris_real_data_benchmark(
            IrisBenchmarkConfig(trials=5),
            samples=parse_iris_csv(iris_fixture()),
        )

        self.assertEqual(report["pipeline"], "iris_real_data_inference_benchmark")
        self.assertEqual(report["metadata"]["sample_count"], 36)
        self.assertEqual(report["metadata"]["accuracy"], 1.0)
        self.assertGreater(report["mean_ms"], 0)


if __name__ == "__main__":
    unittest.main()
