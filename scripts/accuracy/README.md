# Accuracy Sweep Scripts

This folder contains Slurm scripts that start a vLLM server for each model and run quality benchmarks through `scripts/run_quality_eval.sh`.

## Benchmark Tiers

- `tier1`: `mmlu_abstract_algebra`, `hellaswag`, `winogrande`, and `arc_challenge`.
- `tier2`: `gsm8k` and full `mmlu`.
- `all`: runs both tiers.

## Script Groups

- `sweep_quality_deepseek_qwen7b.sh`, `sweep_quality_deepseek_qwen14b.sh`, `sweep_quality_deepseek_qwen32b.sh`: DeepSeek-R1-Distill-Qwen family.
- `sweep_quality_gemma4_31b.sh`: Gemma 4 31B family.
- `sweep_quality_llama.sh`, `sweep_quality_llama32_3b.sh`: Llama 3.1 8B and Llama 3.2 3B families.
- `sweep_quality_qwen25_7b.sh`, `sweep_quality_qwen25_32b.sh`: Qwen2.5 family.
- `sweep_quality_qwen3_8b.sh`, `sweep_quality_qwen3_32b.sh`: Qwen3 family.
- `sweep_quality_missing.sh`: targeted repair script for missing `(model, quant, task)` rows.

## Run Pattern

```bash
sbatch scripts/accuracy/sweep_quality_qwen3_8b.sh
```

To run a narrower quality tier:

```bash
QUALITY_TIER=tier1 sbatch scripts/accuracy/sweep_quality_qwen3_8b.sh
```

The scripts write logs to `results/accuracy/logs/`, JSON outputs to `results/accuracy/evals/`, and append run metadata to `results/accuracy/quality_results.csv`.
