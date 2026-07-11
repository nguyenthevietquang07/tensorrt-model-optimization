from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modelopt.comparison import compare_reports, load_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two benchmark reports.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="reports/comparison_report.json")
    args = parser.parse_args()

    report = compare_reports(load_report(args.baseline), load_report(args.candidate))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
