#!/usr/bin/env python3
"""
Append a single lm_eval task result to a CSV summary.
"""
import argparse
import csv
import json
import os
from datetime import datetime

FIELDS = [
    "timestamp",
    "model",
    "quant",
    "tier",
    "task",
    "n_shot",
    "metric",
    "value",
    "stderr",
    "samples_original",
    "samples_effective",
    "result_json",
]


def pick_metric(result_block):
    metric_key = None
    for key, value in result_block.items():
        if (
            key != "alias"
            and not key.endswith("_stderr,none")
            and "_stderr" not in key
            and isinstance(value, (int, float))
        ):
            metric_key = key
            break
    if metric_key is None:
        raise ValueError("Could not find a primary metric in results block")

    stderr_key = None
    if "," in metric_key:
        base, suffix = metric_key.split(",", 1)
        stderr_key = f"{base}_stderr,{suffix}"
    else:
        stderr_key = f"{metric_key}_stderr"

    return metric_key, result_block[metric_key], result_block.get(stderr_key, "N/A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to lm_eval results JSON")
    parser.add_argument("--task", required=True, help="Task key to extract")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--quant", required=True, help="Quantization label")
    parser.add_argument("--tier", required=True, help="Benchmark tier label")
    parser.add_argument(
        "--csv",
        default="/home/saketh-msc/quantization/results/accuracy/quality_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    if args.task not in data["results"]:
        raise KeyError(f"Task '{args.task}' not found in {args.json}")

    metric, value, stderr = pick_metric(data["results"][args.task])
    n_shot = data.get("n-shot", {}).get(args.task, "N/A")
    samples = data.get("n-samples", {}).get(args.task, {})

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "quant": args.quant,
        "tier": args.tier,
        "task": args.task,
        "n_shot": n_shot,
        "metric": metric,
        "value": value,
        "stderr": stderr,
        "samples_original": samples.get("original", "N/A"),
        "samples_effective": samples.get("effective", "N/A"),
        "result_json": args.json,
    }

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    write_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged {args.task} to {args.csv}")


if __name__ == "__main__":
    main()
