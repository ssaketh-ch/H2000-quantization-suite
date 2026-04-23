# Final Dataset Tables

This folder is the clean data surface for GitHub, notebooks, and Hugging Face.

## Files

| File | Rows | Description |
| --- | ---: | --- |
| `accuracy.csv` | 240 | Final quality benchmark scores. |
| `throughput.csv` | 275 | Final vLLM serving throughput measurements. |
| `model_inventory.csv` | 40 | One row per model-quantization configuration. |
| `accuracy_leaderboard.csv` | 6 | Best model per quality task. |
| `throughput_leaderboard.csv` | 10 | Best model by request rate and by quantization. |

## Accuracy Table

`accuracy.csv` is built from raw `lm_eval` JSON outputs in `results/accuracy/evals/`.

Each row is a unique `(model, quant, task)` result.

Columns:

- `timestamp`: timestamp from the selected result JSON.
- `model`: Hugging Face model ID.
- `model_family`: normalized family label.
- `model_size`: normalized size label.
- `quant`: quantization label.
- `tier`: benchmark tier.
- `task`: benchmark task.
- `n_shot`: few-shot setting.
- `metric`: metric key.
- `accuracy`: score.
- `stderr`: standard error when available.
- `samples_original`: original sample count.
- `samples_effective`: effective sample count.

## Throughput Table

`throughput.csv` is copied from the parsed vLLM benchmark log table.

Each row is one serving benchmark run at a specific request rate.

Columns include request counts, token counts, request throughput, output token throughput, total token throughput, TTFT, TPOT, and ITL statistics.

## Regeneration

```bash
python3 scripts/build_final_tables.py
```
