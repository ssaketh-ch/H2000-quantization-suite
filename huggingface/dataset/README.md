---
license: mit
task_categories:
- text-generation
tags:
- benchmark
- quantization
- vllm
- lm-eval
- h200
- llm
- awq
- gptq
- fp8
- nvfp4
pretty_name: H200 Quantization Benchmarks
size_categories:
- n<1K
---

# H200 Quantization Benchmarks

This dataset contains final benchmark tables for 40 quantized and non-quantized instruction-tuned LLM configurations evaluated on an NVIDIA H200 MIG setup.

It is designed as a compact, programmatic benchmark dataset. The raw benchmark project includes scripts, logs, and run artifacts, while this Hugging Face dataset contains the clean CSV surface for analysis.

## Files

| File | Rows | Description |
| --- | ---: | --- |
| `data/accuracy.csv` | 240 | Quality benchmark scores from `lm_eval`. |
| `data/throughput.csv` | 275 | vLLM serving throughput measurements. |
| `data/model_inventory.csv` | 40 | Model and quantization inventory. |
| `data/accuracy_leaderboard.csv` | 6 | Best model per quality task. |
| `data/throughput_leaderboard.csv` | 10 | Best throughput by request rate and quantization. |

## Benchmark Stack

- Hardware: NVIDIA H200 NVL.
- GPU partitioning: H200 MIG, mainly `mig-3g.71gb`.
- Serving engine: vLLM OpenAI-compatible server in Docker.
- Quality harness: `lm_eval` with `local-completions`.
- Throughput harness: `vllm bench serve`.
- Prompt dataset for throughput: ShareGPT.
- Offline model cache: Hugging Face cache mounted into the vLLM container.

## Model Coverage

| Family | Configurations |
| --- | ---: |
| DeepSeek-R1-Distill-Qwen | 11 |
| Qwen3 | 8 |
| Qwen2.5 | 8 |
| Gemma | 4 |
| Llama-3.1 | 4 |
| Llama-3.2 | 4 |
| Other DeepSeek/Qwen variant | 1 |

Model sizes: 3B, 7B, 8B, 14B, 31B, and 32B.

Quantization labels: `bf16`, `fp8`, `awq`, `gptq`, and `nvfp4`.

## Quality Tasks

| Task | Few-shot | Description |
| --- | ---: | --- |
| `mmlu_abstract_algebra` | 5 | Fast MMLU STEM smoke test |
| `hellaswag` | 10 | Commonsense continuation |
| `winogrande` | 5 | Commonsense reasoning |
| `arc_challenge` | 25 | Science QA |
| `gsm8k` | 5 | Multi-step math reasoning |
| `mmlu` | 5 | Broad academic knowledge |

## Throughput Measurements

Throughput was measured at request rates 1, 2, 4, 8, and 16 requests per second.

The table includes request throughput, output token throughput, total token throughput, time to first token, time per output token, and inter-token latency.

## Usage

```python
from huggingface_hub import hf_hub_download
import pandas as pd

repo_id = "ssakethch/h200-quantization-benchmarks"

accuracy_path = hf_hub_download(repo_id, "data/accuracy.csv", repo_type="dataset")
throughput_path = hf_hub_download(repo_id, "data/throughput.csv", repo_type="dataset")

accuracy = pd.read_csv(accuracy_path)
throughput = pd.read_csv(throughput_path)

print(accuracy.head())
print(throughput.head())
```

## Recommended Analysis

- Use `accuracy.csv` for task-level quality comparisons.
- Use `throughput.csv` for latency and token-throughput comparisons.
- Use `model_inventory.csv` to verify model and quantization coverage.
- Use the leaderboard CSVs for quick summaries.

## License

MIT.
