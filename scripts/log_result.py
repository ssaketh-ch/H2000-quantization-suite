#!/usr/bin/env python3
"""
Parse vLLM bench serve stdout and append to results CSV.
Usage: docker exec ... bench serve | tee /tmp/bench_out.txt
       python3 log_result.py --file /tmp/bench_out.txt --model X --quant Y --rate Z
"""
import argparse
import csv
import os
import re
import sys
from datetime import datetime

FIELDS = [
    "timestamp", "model", "quant", "request_rate",
    "successful_reqs", "failed_reqs", "duration_s",
    "input_tokens", "output_tokens",
    "req_throughput", "output_tok_throughput",
    "peak_output_tok_throughput", "total_tok_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
]

PATTERNS = {
    "successful_reqs":            r"Successful requests:\s+([\d.]+)",
    "failed_reqs":                r"Failed requests:\s+([\d.]+)",
    "duration_s":                 r"Benchmark duration \(s\):\s+([\d.]+)",
    "input_tokens":               r"Total input tokens:\s+([\d.]+)",
    "output_tokens":              r"Total generated tokens:\s+([\d.]+)",
    "req_throughput":             r"Request throughput \(req/s\):\s+([\d.]+)",
    "output_tok_throughput":      r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "peak_output_tok_throughput": r"Peak output token throughput \(tok/s\):\s+([\d.]+)",
    "total_tok_throughput":       r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "mean_ttft_ms":               r"Mean TTFT \(ms\):\s+([\d.]+)",
    "median_ttft_ms":             r"Median TTFT \(ms\):\s+([\d.]+)",
    "p99_ttft_ms":                r"P99 TTFT \(ms\):\s+([\d.]+)",
    "mean_tpot_ms":               r"Mean TPOT \(ms\):\s+([\d.]+)",
    "median_tpot_ms":             r"Median TPOT \(ms\):\s+([\d.]+)",
    "p99_tpot_ms":                r"P99 TPOT \(ms\):\s+([\d.]+)",
    "mean_itl_ms":                r"Mean ITL \(ms\):\s+([\d.]+)",
    "median_itl_ms":              r"Median ITL \(ms\):\s+([\d.]+)",
    "p99_itl_ms":                 r"P99 ITL \(ms\):\s+([\d.]+)",
}

def parse(text):
    result = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        result[key] = m.group(1) if m else "N/A"
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  required=True, help="Path to benchmark stdout file")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--quant", required=True, help="Quantization format e.g. bf16, fp8, gptq_int4, awq_int4")
    parser.add_argument("--rate",  required=True, help="Request rate used")
    parser.add_argument("--csv",   default="/home/saketh-msc/quantization/results/throughput/results.csv")
    args = parser.parse_args()

    with open(args.file) as f:
        text = f.read()

    parsed = parse(text)
    parsed["timestamp"]    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parsed["model"]        = args.model
    parsed["quant"]        = args.quant
    parsed["request_rate"] = args.rate

    write_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: parsed.get(k, "N/A") for k in FIELDS})

    print(f"Logged to {args.csv}")
    for k in FIELDS[3:]:
        print(f"   {k:35s} {parsed.get(k, 'N/A')}")

if __name__ == "__main__":
    main()
