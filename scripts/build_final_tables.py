#!/usr/bin/env python3
"""Build final CSV tables from real benchmark outputs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path("/home/saketh-msc/quantization")
DATA_DIR = ROOT / "data"
ACCURACY_DIR = ROOT / "results" / "accuracy"
THROUGHPUT_DIR = ROOT / "results" / "throughput"

QUALITY_TASKS = {
    "mmlu_abstract_algebra": "tier1",
    "hellaswag": "tier1",
    "winogrande": "tier1",
    "arc_challenge": "tier1",
    "gsm8k": "tier2",
    "mmlu": "tier2",
}

ACCURACY_FIELDS = [
    "timestamp",
    "model",
    "model_family",
    "model_size",
    "quant",
    "tier",
    "task",
    "n_shot",
    "metric",
    "accuracy",
    "stderr",
    "samples_original",
    "samples_effective",
]


def model_family(model: str) -> str:
    lowered = model.lower()
    if "deepseek-r1-distill-qwen" in lowered:
        return "DeepSeek-R1-Distill-Qwen"
    if "gemma" in lowered:
        return "Gemma"
    if "llama-3.1" in lowered:
        return "Llama-3.1"
    if "llama-3.2" in lowered:
        return "Llama-3.2"
    if "qwen2.5" in lowered:
        return "Qwen2.5"
    if "qwen3" in lowered:
        return "Qwen3"
    return "Other"


def model_size(model: str) -> str:
    lowered = model.lower()
    for size in ["32b", "31b", "14b", "8b", "7b", "3b"]:
        if size in lowered:
            return size.upper()
    return "unknown"


def pick_metric(result_block: dict) -> tuple[str, object, object]:
    metric_key = None
    for key, value in result_block.items():
        if key != "alias" and "_stderr" not in key and isinstance(value, (int, float)):
            metric_key = key
            break
    if metric_key is None:
        raise ValueError("Could not find numeric primary metric")

    if "," in metric_key:
        base, suffix = metric_key.split(",", 1)
        stderr_key = f"{base}_stderr,{suffix}"
    else:
        stderr_key = f"{metric_key}_stderr"
    return metric_key, result_block[metric_key], result_block.get(stderr_key, "N/A")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_accuracy_rows() -> list[dict]:
    evals_dir = ACCURACY_DIR / "evals"
    latest: dict[tuple[str, str, str], Path] = {}

    for path in evals_dir.glob("*/*/results_*.json"):
        with path.open() as f:
            data = json.load(f)

        model = data.get("model_name") or data.get("config", {}).get("model_args", {}).get("model")
        if not model:
            raise ValueError(f"Could not determine model for {path}")

        quant = path.relative_to(evals_dir).parts[0]
        for task in QUALITY_TASKS.keys() & data.get("results", {}).keys():
            key = (model, quant, task)
            if key not in latest or path.name > latest[key].name:
                latest[key] = path

    rows = []
    for model, quant, task in sorted(latest):
        path = latest[(model, quant, task)]
        with path.open() as f:
            data = json.load(f)

        metric, value, stderr = pick_metric(data["results"][task])
        samples = data.get("n-samples", {}).get(task, {})
        rows.append(
            {
                "timestamp": data.get("date", path.stem.removeprefix("results_")),
                "model": model,
                "model_family": model_family(model),
                "model_size": model_size(model),
                "quant": quant,
                "tier": QUALITY_TASKS[task],
                "task": task,
                "n_shot": data.get("n-shot", {}).get(task, "N/A"),
                "metric": metric,
                "accuracy": value,
                "stderr": stderr,
                "samples_original": samples.get("original", "N/A"),
                "samples_effective": samples.get("effective", "N/A"),
            }
        )
    return rows


def build_model_inventory(accuracy_rows: list[dict], throughput_rows: list[dict]) -> list[dict]:
    keys = sorted(
        set((row["model"], row["quant"]) for row in accuracy_rows)
        | set((row["model"], row["quant"]) for row in throughput_rows)
    )
    accuracy_keys = set((row["model"], row["quant"]) for row in accuracy_rows)
    throughput_keys = set((row["model"], row["quant"]) for row in throughput_rows)

    inventory = []
    for model, quant in keys:
        inventory.append(
            {
                "model": model,
                "model_family": model_family(model),
                "model_size": model_size(model),
                "quant": quant,
                "has_accuracy": str((model, quant) in accuracy_keys).lower(),
                "has_throughput": str((model, quant) in throughput_keys).lower(),
            }
        )
    return inventory


def build_accuracy_leaderboard(accuracy_rows: list[dict]) -> list[dict]:
    rows = []
    for task in sorted({row["task"] for row in accuracy_rows}):
        task_rows = [row for row in accuracy_rows if row["task"] == task]
        best = max(task_rows, key=lambda row: float(row["accuracy"]))
        rows.append(
            {
                "task": task,
                "model": best["model"],
                "model_family": best["model_family"],
                "model_size": best["model_size"],
                "quant": best["quant"],
                "accuracy": best["accuracy"],
                "stderr": best["stderr"],
            }
        )
    return rows


def build_throughput_leaderboard(throughput_rows: list[dict]) -> list[dict]:
    clean_rows = [row for row in throughput_rows if int(float(row["failed_reqs"])) == 0]
    rows = []

    for rate in sorted({row["request_rate"] for row in clean_rows}, key=float):
        rate_rows = [row for row in clean_rows if row["request_rate"] == rate]
        best = max(rate_rows, key=lambda row: float(row["output_tok_throughput"]))
        rows.append(
            {
                "leaderboard_type": "best_by_request_rate",
                "group": rate,
                "model": best["model"],
                "quant": best["quant"],
                "request_rate": best["request_rate"],
                "output_tok_throughput": best["output_tok_throughput"],
                "total_tok_throughput": best["total_tok_throughput"],
                "mean_ttft_ms": best["mean_ttft_ms"],
                "mean_tpot_ms": best["mean_tpot_ms"],
            }
        )

    for quant in sorted({row["quant"] for row in clean_rows}):
        quant_rows = [row for row in clean_rows if row["quant"] == quant]
        best = max(quant_rows, key=lambda row: float(row["output_tok_throughput"]))
        rows.append(
            {
                "leaderboard_type": "best_by_quant",
                "group": quant,
                "model": best["model"],
                "quant": best["quant"],
                "request_rate": best["request_rate"],
                "output_tok_throughput": best["output_tok_throughput"],
                "total_tok_throughput": best["total_tok_throughput"],
                "mean_ttft_ms": best["mean_ttft_ms"],
                "mean_tpot_ms": best["mean_tpot_ms"],
            }
        )

    return rows


def main() -> None:
    accuracy_rows = build_accuracy_rows()
    final_accuracy = ACCURACY_DIR / "final" / "accuracy.csv"
    write_csv(final_accuracy, accuracy_rows, ACCURACY_FIELDS)

    DATA_DIR.mkdir(exist_ok=True)
    shutil.copyfile(final_accuracy, DATA_DIR / "accuracy.csv")
    shutil.copyfile(THROUGHPUT_DIR / "results.csv", THROUGHPUT_DIR / "final" / "throughput.csv")
    shutil.copyfile(THROUGHPUT_DIR / "results.csv", DATA_DIR / "throughput.csv")

    throughput_rows = read_csv(DATA_DIR / "throughput.csv")
    inventory = build_model_inventory(accuracy_rows, throughput_rows)
    write_csv(
        DATA_DIR / "model_inventory.csv",
        inventory,
        ["model", "model_family", "model_size", "quant", "has_accuracy", "has_throughput"],
    )
    write_csv(
        DATA_DIR / "accuracy_leaderboard.csv",
        build_accuracy_leaderboard(accuracy_rows),
        ["task", "model", "model_family", "model_size", "quant", "accuracy", "stderr"],
    )
    write_csv(
        DATA_DIR / "throughput_leaderboard.csv",
        build_throughput_leaderboard(throughput_rows),
        [
            "leaderboard_type",
            "group",
            "model",
            "quant",
            "request_rate",
            "output_tok_throughput",
            "total_tok_throughput",
            "mean_ttft_ms",
            "mean_tpot_ms",
        ],
    )

    print(f"Wrote {len(accuracy_rows)} real accuracy rows")
    print("Wrote final throughput tables")
    print("Wrote model inventory and leaderboard tables")


if __name__ == "__main__":
    main()
