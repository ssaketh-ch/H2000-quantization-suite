# Final Throughput

This folder contains the final serving benchmark table.

## File

- `throughput.csv`: 275 parsed vLLM serving benchmark rows.

## Schema

| Column | Meaning |
| --- | --- |
| `timestamp` | Timestamp when the row was logged. |
| `model` | Hugging Face model ID. |
| `quant` | Quantization label. |
| `request_rate` | Target request rate in requests per second. |
| `successful_reqs` | Completed requests. |
| `failed_reqs` | Failed requests. |
| `duration_s` | Benchmark duration in seconds. |
| `input_tokens` | Total prompt tokens. |
| `output_tokens` | Total generated tokens. |
| `req_throughput` | Completed requests per second. |
| `output_tok_throughput` | Generated output tokens per second. |
| `peak_output_tok_throughput` | Peak output tokens per second. |
| `total_tok_throughput` | Input plus output tokens per second. |
| `mean_ttft_ms` | Mean time to first token in milliseconds. |
| `median_ttft_ms` | Median time to first token in milliseconds. |
| `p99_ttft_ms` | P99 time to first token in milliseconds. |
| `mean_tpot_ms` | Mean time per output token in milliseconds. |
| `median_tpot_ms` | Median time per output token in milliseconds. |
| `p99_tpot_ms` | P99 time per output token in milliseconds. |
| `mean_itl_ms` | Mean inter-token latency in milliseconds. |
| `median_itl_ms` | Median inter-token latency in milliseconds. |
| `p99_itl_ms` | P99 inter-token latency in milliseconds. |

Regenerate with:

```bash
python3 scripts/build_final_tables.py
```
