"""Compute classification metrics from a CSV of saved labels and predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="CSV with true_label and predicted_label columns.")
    parser.add_argument("--labels", nargs="+", required=True, help="Class labels in the desired metric order.")
    parser.add_argument("--output-json", help="Optional output path; otherwise print JSON to stdout.")
    args = parser.parse_args()

    with Path(args.input_csv).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"true_label", "predicted_label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain 'true_label' and 'predicted_label' columns.")
        rows = list(reader)

    results = classification_metrics(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        args.labels,
    )
    rendered = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
