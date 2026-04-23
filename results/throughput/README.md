# Throughput Results

This folder stores vLLM serving benchmark outputs.

## Final Table

Use this file for final throughput analysis:

```text
results/throughput/final/throughput.csv
```

The same table is copied to:

```text
data/throughput.csv
```

The final table has 275 rows across 40 model-quantization configurations and request rates 1, 2, 4, 8, and 16.

## Raw Files

- `results.csv`: append-only parsed throughput table.
- `<quant>/<model>/rate*.txt`: raw vLLM benchmark output.
- `logs/`: Slurm logs from throughput sweeps.
- `final/`: final throughput CSV and its folder README.

## Metrics

- `successful_reqs`: completed requests.
- `failed_reqs`: failed requests.
- `duration_s`: benchmark duration.
- `input_tokens`: total prompt tokens.
- `output_tokens`: total generated tokens.
- `req_throughput`: completed requests per second.
- `output_tok_throughput`: generated tokens per second.
- `peak_output_tok_throughput`: peak generated tokens per second.
- `total_tok_throughput`: input plus output tokens per second.
- `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms`: time to first token.
- `mean_tpot_ms`, `median_tpot_ms`, `p99_tpot_ms`: time per output token.
- `mean_itl_ms`, `median_itl_ms`, `p99_itl_ms`: inter-token latency.
