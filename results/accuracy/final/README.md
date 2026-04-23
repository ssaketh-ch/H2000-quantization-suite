# Final Accuracy

This folder contains the final quality benchmark table.

## File

- `accuracy.csv`: 240 real benchmark rows, one row per model, quantization, and task.

## Schema

| Column | Meaning |
| --- | --- |
| `timestamp` | Timestamp from the selected `lm_eval` JSON result. |
| `model` | Hugging Face model ID used by vLLM and `lm_eval`. |
| `model_family` | Normalized family label. |
| `model_size` | Normalized parameter size. |
| `quant` | Quantization label: `bf16`, `fp8`, `awq`, `gptq`, or `nvfp4`. |
| `tier` | Benchmark tier. |
| `task` | `lm_eval` task name. |
| `n_shot` | Few-shot count. |
| `metric` | Primary metric key, usually `acc,none`. |
| `accuracy` | Final score for the row. |
| `stderr` | Standard error from `lm_eval` when available. |
| `samples_original` | Original sample count from `lm_eval`. |
| `samples_effective` | Effective sample count after filters. |

Regenerate with:

```bash
python3 scripts/build_final_tables.py
```
