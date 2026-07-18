from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modelopt.real_data import DigitsBenchmarkConfig, run_digits_real_data_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local handwritten-digits inference benchmark.")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--holdout-per-class", type=int, default=25)
    parser.add_argument("--output", default="reports/digits_real_data_benchmark.json")
    args = parser.parse_args()

    config = DigitsBenchmarkConfig(
        trials=args.trials,
        holdout_per_class=args.holdout_per_class,
        output_path=args.output,
    )
    report = run_digits_real_data_benchmark(config)
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(output_path), "summary": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
