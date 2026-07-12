from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modelopt.real_data import (  # noqa: E402
    IrisBenchmarkConfig,
    load_iris_dataset,
    run_iris_real_data_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-data local inference benchmark on UCI Iris.")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--input-csv", help="Optional local Iris CSV fixture for offline verification.")
    parser.add_argument("--output", default="reports/iris_real_data_benchmark.json")
    args = parser.parse_args()

    samples = load_iris_dataset(args.input_csv) if args.input_csv else None
    config = IrisBenchmarkConfig(trials=args.trials, output_path=args.output)
    report = run_iris_real_data_benchmark(config, samples=samples)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report_path": str(output_path), "summary": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
