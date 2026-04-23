# Throughput Sweep Scripts

This folder contains Slurm scripts for vLLM serving throughput sweeps.

## What They Do

Each script starts a vLLM OpenAI-compatible server in Docker, then runs `scripts/run_bench.sh` for request rates:

```text
1, 2, 4, 8, 16 requests per second
```

The benchmark uses ShareGPT prompts from `datasets/ShareGPT_V3_unfiltered_cleaned_split.json`.

## Script Groups

- `sweep_deepseek_qwen7b.sh`, `sweep_deepseek_qwen14b.sh`, `sweep_deepseek_qwen32b.sh`: DeepSeek-R1-Distill-Qwen family.
- `sweep_gemma4_31b.sh`: Gemma 4 31B family.
- `sweep_llama.sh`, `sweep_llama32_3b.sh`: Llama 3.1 8B and Llama 3.2 3B families.
- `sweep_qwen25_7b.sh`, `sweep_qwen25_32b.sh`: Qwen2.5 family.
- `sweep_qwen3_8b.sh`, `sweep_qwen3_32b.sh`: Qwen3 family.

## Outputs

- Raw benchmark text files are written under `results/throughput/<quant>/<model>/rate*.txt`.
- Parsed rows are appended to `results/throughput/results.csv`.
- The final table is `results/throughput/final/throughput.csv` and is also copied to `data/throughput.csv`.
