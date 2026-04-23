# Quantisation Benchmark Suite

A systematic comparison of 40 quantised instruction-tuned LLMs measured on an NVIDIA H200 NVL server.
The suite covers both **serving throughput** (vLLM, ShareGPT prompts) and **model quality** (lm-eval, six standard tasks)
across five quantisation formats and six model families.

---

## Contents

| File | Rows | Description |
| --- | ---: | --- |
| `data/accuracy.csv` | 240 | Quality scores |
| `data/throughput.csv` | 275 | vLLM serving metrics|
| `data/model_inventory.csv` | 40 | Model and quantisation coverage |
| `data/accuracy_leaderboard.csv` | 6 | Best result per benchmark task |
| `data/throughput_leaderboard.csv` | 10 | Best throughput by request rate and by format |

---

## Hardware and Stack

| Component | Details |
| --- | --- |
| GPU | NVIDIA H200 NVL |
| MIG profile | `mig-3g.71gb` |
| NVIDIA driver | 570.133.20 |
| CUDA | 12.8 |
| PyTorch | 2.7.1+cu126 |
| Serving engine | vLLM OpenAI-compatible server in Docker |
| vLLM image | `vllm/vllm-openai:v0.19.0` |
| Quality harness | `lm_eval` v0.4.11, `local-completions` backend |
| Throughput harness | `vllm bench serve` with ShareGPT dataset |
| Dataset/model cache | Local HuggingFace cache, `HF_HUB_OFFLINE=1` |
| CPU | 2× Intel Xeon Platinum 8480+ (224 threads total) |

---

## Model Coverage

40 model–quantisation configurations across 6 families and 5 formats.

| Family | Size | bf16 | fp8 | awq | gptq | nvfp4 |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| Llama-3.1 | 8B | ✓ | ✓ | ✓ | ✓ | |
| Llama-3.2 | 3B | ✓ | ✓ | ✓ | ✓ | |
| DeepSeek-R1-Distill-Qwen | 7B | ✓ | ✓ | ✓ | ✓ | |
| DeepSeek-R1-Distill-Qwen | 14B | ✓ | ✓ | ✓ | ✓ | |
| DeepSeek-R1-Distill-Qwen | 32B | ✓ | ✓ | ✓ | ✓ | |
| Qwen2.5 | 7B | ✓ | ✓ | ✓ | ✓ | |
| Qwen2.5 | 32B | ✓ | ✓ | ✓ | ✓ | |
| Qwen3 | 8B | ✓ | ✓ | ✓ | ✓ | |
| Qwen3 | 32B | ✓ | ✓ | ✓ | ✓ | |
| Gemma-4 | 31B | ✓ | ✓ | ✓ | | ✓ |

# Full Model List

| Family | Size | Quant | Model |
|--------|------|-------|-------|
| Llama 3.1 | 8B | bf16 | `meta-llama/Llama-3.1-8B-Instruct` |
| Llama 3.1 | 8B | fp8 | `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8` |
| Llama 3.1 | 8B | awq | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |
| Llama 3.1 | 8B | gptq | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` |
| Llama 3.2 | 3B | bf16 | `meta-llama/Llama-3.2-3B-Instruct` |
| Llama 3.2 | 3B | fp8 | `RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic` |
| Llama 3.2 | 3B | awq | `AMead10/Llama-3.2-3B-Instruct-AWQ` |
| Llama 3.2 | 3B | gptq | `shuyuej/Llama-3.2-3B-Instruct-GPTQ` |
| DeepSeek R1 Distill | 7B | bf16 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| DeepSeek R1 Distill | 7B | fp8 | `neuralmagic/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic` |
| DeepSeek R1 Distill | 7B | awq | `AngelSlim/Deepseek_r1_distill_qwen-7b_int4_awq` |
| DeepSeek R1 Distill | 7B | gptq | `jakiAJK/DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4` |
| DeepSeek R1 Distill | 14B | bf16 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| DeepSeek R1 Distill | 14B | fp8 | `RedHatAI/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic` |
| DeepSeek R1 Distill | 14B | awq | `Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ` |
| DeepSeek R1 Distill | 14B | gptq | `RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16` |
| DeepSeek R1 Distill | 32B | bf16 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| DeepSeek R1 Distill | 32B | fp8 | `neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic` |
| DeepSeek R1 Distill | 32B | awq | `inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ` |
| DeepSeek R1 Distill | 32B | gptq | `dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4` |
| Qwen 2.5 | 7B | bf16 | `Qwen/Qwen2.5-7B-Instruct` |
| Qwen 2.5 | 7B | fp8 | `neuralmagic/Qwen2.5-7B-Instruct-FP8-Dynamic` |
| Qwen 2.5 | 7B | awq | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| Qwen 2.5 | 7B | gptq | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` |
| Qwen 2.5 | 32B | bf16 | `Qwen/Qwen2.5-32B-Instruct` |
| Qwen 2.5 | 32B | fp8 | `neuralmagic/Qwen2.5-32B-Instruct-FP8-Dynamic` |
| Qwen 2.5 | 32B | awq | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| Qwen 2.5 | 32B | gptq | `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` |
| Qwen 3 | 8B | bf16 | `Qwen/Qwen3-8B` |
| Qwen 3 | 8B | fp8 | `Qwen/Qwen3-8B-FP8` |
| Qwen 3 | 8B | awq | `Qwen/Qwen3-8B-AWQ` |
| Qwen 3 | 8B | gptq | `JunHowie/Qwen3-8B-GPTQ-Int4` |
| Qwen 3 | 32B | bf16 | `Qwen/Qwen3-32B` |
| Qwen 3 | 32B | fp8 | `Qwen/Qwen3-32B-FP8` |
| Qwen 3 | 32B | awq | `Qwen/Qwen3-32B-AWQ` |
| Qwen 3 | 32B | gptq | `JunHowie/Qwen3-32B-GPTQ-Int4` |
| Gemma 4 | 31B | bf16 | `google/gemma-4-31B-it` |
| Gemma 4 | 31B | fp8 | `RedHatAI/gemma-4-31B-it-FP8-block` |
| Gemma 4 | 31B | awq | `QuantTrio/gemma-4-31B-it-AWQ` |
| Gemma 4 | 31B | nvfp4 | `RedHatAI/gemma-4-31B-it-NVFP4` |

---
## Throughput Benchmarks

### Setup

Served via vLLM OpenAI-compatible server using the ShareGPT prompt dataset.
Request rates swept: **1, 2, 4, 8, 16 req/s**.
All 40 models completed every rate with zero failed requests.

Primary metrics:

| Metric | Description |
| --- | --- |
| `output_tok_throughput` | Generated output tokens per second |
| `total_tok_throughput` | Input + output tokens per second |
| `mean_ttft_ms` | Mean time to first token (ms) |
| `mean_tpot_ms` | Mean time per output token (ms) |
| `p99_ttft_ms` | p99 time to first token (ms) |
| `p99_tpot_ms` | p99 time per output token (ms) |

## Throughput Leaderboard by Request Rate

Best output tokens per second at each request rate

| Rate (req/s) | Model | Quant | Output tok/s | TTFT (ms) | TPOT (ms) |
|-------------:|-------|-------|-------------:|----------:|----------:|
| 1  | Gemma-4-31B | fp8 | 216.93 | 2702.92 | 29.69 |
| 2  | Qwen3-8B | fp8 | 424.53 | 28.45 | 6.01 |
| 4  | Qwen3-8B | fp8 | 867.83 | 29.48 | 6.23 |
| 8  | Qwen3-8B | fp8 | 1594.12 | 32.50 | 6.69 |
| 16 | Qwen3-8B | fp8 | 3150.57 | 46.73 | 8.37 |

---

## Throughput Leaderboard by Quantization

Best model per quant format

| Format | Model | Rate (req/s) | Output tok/s | Total tok/s |
|--------|-------|-------------:|-------------:|-------------:|
| bf16 | DeepSeek-R1-Distill-Qwen-7B | 16 | 3102.81 | 6577.31 |
| fp8  | Qwen3-8B | 16 | 3150.57 | 6606.96 |
| awq  | DeepSeek-R1-Distill-Qwen-7B (AWQ) | 16 | 3141.13 | 6649.51 |
| gptq | DeepSeek-R1-Distill-Qwen-7B (GPTQ) | 16 | 3126.08 | 6635.00 |
| nvfp4 | Gemma-4-31B | 4 | 639.36 | 1270.88 |

---

## Peak Throughput (All Models)

Best observed performance across all runs

| Model | Quant | Peak tok/s | Best TTFT (ms) | Best TPOT (ms) |
|------|-------|-----------:|---------------:|---------------:|
| Qwen3-8B | fp8 | 3150.57 | 28.45 | 5.91 |
| DeepSeek-R1-Distill-Qwen-7B (AWQ) | awq | 3141.13 | 24.67 | 5.88 |
| DeepSeek-R1-Distill-Qwen-7B (FP8) | fp8 | 3132.08 | 22.41 | 5.38 |
| DeepSeek-R1-Distill-Qwen-7B (GPTQ) | gptq | 3126.08 | 24.21 | 5.88 |
| Qwen3-8B (AWQ) | awq | 3103.91 | 33.26 | 7.60 |
| DeepSeek-R1-Distill-Qwen-7B | bf16 | 3102.81 | 27.62 | 7.88 |
| Qwen3-8B | bf16 | 3094.58 | 35.65 | 9.27 |
| Qwen3-8B (GPTQ) | gptq | 3075.86 | 33.20 | 7.48 |
| Qwen2.5-7B-Instruct (GPTQ) | gptq | 3012.66 | 37.91 | 6.82 |
| Qwen2.5-7B-Instruct (FP8) | fp8 | 3008.81 | 28.31 | 5.47 |

---

## Key Observations

- Qwen3-8B (FP8) dominates throughput across all request rates
- 7B distilled models cluster tightly across quantization formats
- AWQ, GPTQ, and FP8 perform similarly at peak for 7B models
- Larger models show severe TTFT degradation at high concurrency
- Smaller models provide more stable latency but lower peak throughput

---

## Mean TTFT Summary (Condensed)

| Model | 1 req/s | 16 req/s |
|------|--------:|---------:|
| Llama-3.2-3B | 23.75 | 21.44 |
| Qwen3-8B | 54.88 | 64.60 |
| DeepSeek-7B | 45.38 | 38.29 |
| DeepSeek-32B | 142.97 | 172286.19 |

## Throughput Table (All Models)

| Model | Size | Quant | Peak tok/s | Best TTFT (ms) | Best TPOT (ms) |
|------|------|-------|-----------:|---------------:|---------------:|
| Qwen3 | 8B | fp8 | 3150.57 | 28.45 | 5.91 |
| DeepSeek-R1-Distill-Qwen | 7B | awq | 3141.13 | 24.67 | 5.88 |
| DeepSeek-R1-Distill-Qwen | 7B | fp8 | 3132.08 | 22.41 | 5.38 |
| DeepSeek-R1-Distill-Qwen | 7B | gptq | 3126.08 | 24.21 | 5.88 |
| Qwen3 | 8B | awq | 3103.91 | 33.26 | 7.60 |
| DeepSeek-R1-Distill-Qwen | 7B | bf16 | 3102.81 | 27.62 | 7.88 |
| Qwen3 | 8B | bf16 | 3094.58 | 35.65 | 9.27 |
| Qwen3 | 8B | gptq | 3075.86 | 33.20 | 7.48 |
| Qwen2.5-Instruct | 7B | gptq | 3012.66 | 37.91 | 6.82 |
| Qwen2.5-Instruct | 7B | fp8 | 3008.81 | 28.31 | 5.47 |
| Qwen2.5-Instruct | 7B | awq | 2997.44 | 33.36 | 6.50 |
| Qwen2.5-Instruct | 7B | bf16 | 2978.69 | 33.85 | 8.01 |
| Llama-3.2-Instruct | 3B | awq | 2966.60 | 17.06 | 3.80 |
| DeepSeek-R1-Distill-Qwen | 14B | fp8 | 2957.94 | 38.60 | 10.04 |
| Llama-3.1-Instruct | 8B | fp8 | 2948.59 | 35.08 | 5.64 |
| Llama-3.2-Instruct | 3B | fp8 | 2945.36 | 16.35 | 3.20 |
| Llama-3.2-Instruct | 3B | bf16 | 2944.68 | 17.70 | 4.43 |
| Llama-3.2-Instruct | 3B | gptq | 2940.03 | 16.90 | 3.74 |
| Llama-3.1-Instruct | 8B | awq | 2912.63 | 34.11 | 6.93 |
| Llama-3.1-Instruct | 8B | bf16 | 2900.35 | 38.96 | 9.21 |
| Llama-3.1-Instruct | 8B | gptq | 2889.80 | 34.94 | 7.18 |
| DeepSeek-R1-Distill-Qwen | 14B | gptq | 2809.96 | 45.10 | 11.57 |
| DeepSeek-R1-Distill-Qwen | 14B | bf16 | 2787.78 | 54.30 | 16.62 |
| DeepSeek-R1-Distill-Qwen | 14B | awq | 2778.40 | 44.99 | 11.72 |
| Qwen3 | 32B | fp8 | 2146.57 | 89.14 | 19.77 |
| DeepSeek-R1-Distill-Qwen | 32B | fp8 | 2036.66 | 79.14 | 20.28 |
| Qwen2.5-Instruct | 32B | fp8 | 2029.69 | 88.81 | 20.55 |
| DeepSeek-R1-Distill-Qwen | 32B | gptq | 1589.17 | 92.07 | 24.38 |
| DeepSeek-R1-Distill-Qwen | 32B | awq | 1582.26 | 92.65 | 24.63 |
| Qwen3 | 32B | gptq | 1530.18 | 108.62 | 24.67 |
| Qwen2.5-Instruct | 32B | awq | 1507.41 | 108.68 | 24.47 |
| Qwen3 | 32B | awq | 1507.06 | 101.19 | 24.68 |
| Qwen2.5-Instruct | 32B | gptq | 1503.55 | 97.28 | 23.63 |
| Gemma-4 | 31B | fp8 | 1164.88 | 162.32 | 29.69 |
| Gemma-4 | 31B | awq | 927.55 | 165.86 | 28.30 |
| DeepSeek-R1-Distill-Qwen | 32B | bf16 | 801.95 | 129.19 | 35.21 |
| Qwen3 | 32B | bf16 | 776.52 | 139.17 | 35.46 |
| Qwen2.5-Instruct | 32B | bf16 | 769.75 | 147.81 | 35.34 |
| Gemma-4 | 31B | nvfp4 | 639.36 | 249.81 | 27.88 |
| Gemma-4 | 31B | bf16 | 365.86 | 166.42 | 41.40 |
---
## Quality Benchmarks

Each model–quantisation configuration is evaluated on five tasks.

| Task | Tier | Few-shot | Metric | Samples | What it measures |
| --- | --- | ---: | --- | ---: | --- |
| `hellaswag` | tier 1 | 10 | acc | 10 042 | Commonsense sentence completion |
| `winogrande` | tier 1 | 5 | acc | 1 267 | Commonsense pronoun resolution |
| `arc_challenge` | tier 1 | 25 | acc | 1 172 | Science multiple-choice QA |
| `gsm8k` | tier 2 | 5 | exact_match | 1 319 | Multi-step grade-school math |
| `mmlu` | tier 2 | 5 | acc | ~14 000 | Broad academic knowledge (57 subjects) |

Evaluation command:

```bash
lm_eval \
  --model local-completions \
  --model_args model=<MODEL>,base_url=http://localhost:<PORT>/v1/completions,\
tokenizer_backend=huggingface,tokenizer=<MODEL> \
  --apply_chat_template \
  --tasks <TASK> \
  --num_fewshot <N> \
  --batch_size 1
```

### Note on GSM8K scores for reasoning models

DeepSeek-R1-Distill and Qwen3 models use a chat template that prepends `<｜Assistant｜><think>\n`
to every generation, placing the model in extended chain of thought mode.
Combined with the 2 047 token context limit used in this sweep, the models exhausted the context window
mid-reasoning trace and never produced a final `#### <answer>` token, causing `strict-match` to return zero.
GSM8K scores for those families are therefore omitted from all tables.
The scores reported for Gemma-4, Llama-3.1, and Llama-3.2 are not affected.

---

## Quality Results

### Accuracy leaderboard - best result per task

| Task | Best model | Quant | Score |
| --- | --- | --- | ---: |
| `arc_challenge` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | awq | 0.7090 |
| `gsm8k` | `RedHatAI/gemma-4-31B-it-NVFP4` | nvfp4 | 0.9340 |
| `hellaswag` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | awq | 0.6844 |
| `mmlu` | `RedHatAI/gemma-4-31B-it-FP8-block` | fp8 | 0.8754 |
| `mmlu_abstract_algebra` | `google/gemma-4-31B-it` | bf16 | 0.8732 |
| `winogrande` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | awq | 0.7782 |

# Accuracy Results

Scores are percentages (higher is better). GSM8K uses `exact_match,flexible-extract`. `—` indicates omitted rows due to the reasoning-model context issue.

| Family | Size | Quant | HellaSwag | Winogrande | ARC-C | GSM8K | Full MMLU |
|--------|------|-------|----------:|-----------:|------:|------:|----------:|
| Llama 3.1 | 8B | bf16 | 60.74% | 77.03% | 62.12% | 81.73% | 68.79% |
| Llama 3.1 | 8B | fp8 | 60.92% | 77.27% | 61.60% | 82.18% | 68.39% |
| Llama 3.1 | 8B | awq | 60.35% | 77.82% | 59.56% | 77.56% | 66.53% |
| Llama 3.1 | 8B | gptq | 60.46% | 74.66% | 59.73% | 76.19% | 66.81% |
| Llama 3.2 | 3B | bf16 | 53.68% | 69.93% | 51.11% | 72.71% | 61.71% |
| Llama 3.2 | 3B | fp8 | 53.58% | 70.32% | 50.00% | 68.92% | 61.53% |
| Llama 3.2 | 3B | awq | 52.85% | 68.82% | 47.53% | 63.91% | 60.43% |
| Llama 3.2 | 3B | gptq | 52.59% | 68.03% | 48.12% | 60.58% | 57.75% |
| DeepSeek R1 Distill | 7B | bf16 | 46.01% | 61.25% | 33.96% | — | 30.56% |
| DeepSeek R1 Distill | 7B | fp8 | 45.96% | 59.67% | 33.62% | — | 29.96% |
| DeepSeek R1 Distill | 7B | awq | 44.97% | 58.25% | 33.02% | — | 27.59% |
| DeepSeek R1 Distill | 7B | gptq | 46.90% | 61.01% | 44.37% | 5.16% | 49.21% |
| DeepSeek R1 Distill | 14B | bf16 | 56.14% | 67.09% | 40.19% | — | 22.97% |
| DeepSeek R1 Distill | 14B | fp8 | 56.01% | 66.77% | 40.02% | — | 23.08% |
| DeepSeek R1 Distill | 14B | awq | 59.74% | 60.22% | 50.17% | — | 49.36% |
| DeepSeek R1 Distill | 14B | gptq | 55.66% | 67.64% | 40.27% | — | 22.95% |
| DeepSeek R1 Distill | 32B | bf16 | 60.00% | 73.09% | 48.29% | — | 26.73% |
| DeepSeek R1 Distill | 32B | fp8 | 59.97% | 72.69% | 48.72% | — | 27.35% |
| DeepSeek R1 Distill | 32B | awq | 62.29% | 70.64% | 62.71% | 53.68% | 68.66% |
| DeepSeek R1 Distill | 32B | gptq | 62.54% | 70.48% | 62.97% | 65.66% | 65.95% |
| Qwen 2.5 | 7B | bf16 | 59.65% | 62.19% | 57.59% | 16.76% | 73.51% |
| Qwen 2.5 | 7B | fp8 | 60.00% | 62.75% | 57.68% | 13.65% | 73.41% |
| Qwen 2.5 | 7B | awq | 58.04% | 64.09% | 58.79% | 13.72% | 71.96% |
| Qwen 2.5 | 7B | gptq | 59.45% | 63.54% | 57.85% | 6.90% | 72.15% |
| Qwen 2.5 | 32B | bf16 | 68.04% | 66.61% | 70.22% | 5.46% | 82.50% |
| Qwen 2.5 | 32B | fp8 | 68.32% | 67.09% | 70.82% | 4.32% | 82.48% |
| Qwen 2.5 | 32B | awq | 68.44% | 68.90% | 70.90% | 11.30% | 81.86% |
| Qwen 2.5 | 32B | gptq | 66.59% | 66.38% | 69.54% | 3.79% | 82.00% |
| Qwen 3 | 8B | bf16 | 57.33% | 68.19% | 54.18% | — | 28.66% |
| Qwen 3 | 8B | fp8 | 57.20% | 68.43% | 54.44% | — | 27.73% |
| Qwen 3 | 8B | awq | 56.83% | 68.51% | 55.80% | — | 23.05% |
| Qwen 3 | 8B | gptq | 56.05% | 68.03% | 53.33% | — | 23.47% |
| Qwen 3 | 32B | bf16 | 63.07% | 75.14% | 63.14% | — | 44.94% |
| Qwen 3 | 32B | fp8 | 63.14% | 73.56% | 61.95% | — | 38.51% |
| Qwen 3 | 32B | awq | 62.74% | 74.03% | 62.20% | — | 41.80% |
| Qwen 3 | 32B | gptq | 63.01% | 75.14% | 61.77% | — | 40.66% |
| Gemma 4 | 31B | bf16 | 57.88% | 73.96% | 67.88% | 93.03% | 87.32% |
| Gemma 4 | 31B | fp8 | 58.07% | 74.11% | 67.83% | 93.03% | 87.54% |
| Gemma 4 | 31B | awq | 59.44% | 73.64% | 68.17% | 92.57% | 86.68% |
| Gemma 4 | 31B | nvfp4 | 57.87% | 73.96% | 66.47% | 93.40% | 86.12% |
---



## Key Findings

### Throughput

- **Small models (3B–8B) scale linearly.** Output throughput doubles with each doubling of request rate up through 16 req/s, reaching ~3 000 tok/s for 7B/8B models on a single MIG-3g.71gb slice.
- **Large models (30B–32B) saturate early.** Throughput for BF16 32B models plateaus around 770–800 tok/s after 4 req/s, indicating GPU memory bandwidth saturation.
- **FP8 quantisation unlocks 2–3× more throughput for 32B models.** `Qwen3-32B-FP8` peaks at 2 147 tok/s vs 777 tok/s for BF16; `DeepSeek-R1-Distill-Qwen-32B-FP8` peaks at 2 037 tok/s vs 802 tok/s for BF16.
- **TTFT explodes for unquantised 32B models under load.** BF16 Qwen2.5-32B reaches 167 seconds mean TTFT at 16 req/s; BF16 Gemma-4-31B reaches 469 seconds. FP8 cuts this dramatically (e.g. Qwen2.5-32B-FP8: 15 seconds at 16 req/s).
- **AWQ and GPTQ match FP8 throughput for 8B-class models.** All four formats deliver 2 900–3 150 tok/s for 7B/8B models, so format choice for small models should be driven by quality, not speed.
- **Llama-3.2-3B has the lowest TTFT of any model.** Sub-25 ms at all request rates across all four formats, making it the fastest first-token experience in the sweep.
- **Zero failed requests** across all 40 models and all request rates.

### Quality

- **Gemma-4-31B leads on MMLU (0.8754) and GSM8K (0.9340)**, with quantisation degradation under 1% across FP8, AWQ, and NVFP4.
- **Qwen2.5-32B is the strongest general-knowledge model** after Gemma, scoring 0.8250 full MMLU with near-identical results across all four quant formats.
- **Qwen2.5-32B-AWQ achieves the best ARC-Challenge (0.7090) and HellaSwag (0.6844)** scores in the entire sweep.
- **FP8 preserves accuracy within ~0.5% of BF16** for all standard instruction-tuned models. AWQ and GPTQ typically degrade by 1–3%.
- **DeepSeek-R1-Distill and Qwen3 score low on MMLU in this harness** due to their chain-of-thought output format being evaluated without sufficient context. These scores do not reflect the models' true knowledge capability.
- **Llama-3.1-8B has the strongest GSM8K score** among 8B-class models (0.82 BF16 / fp8), clearly ahead of Qwen2.5-7B (0.17 BF16) on this task.

---

## Rebuild Final Tables

```bash
cd /home/saketh-msc/quantization
python3 scripts/build_final_tables.py
```

Scans `results/accuracy/evals/`, keeps the latest real JSON result for each `(model, quant, task)`, and writes all files under `data/` and `results/accuracy/final/` and `results/throughput/final/`.

---

## Repository Layout

```
data/                     Final CSV files
datasets/                 ShareGPT serving benchmark dataset
models/                   Local model and dataset cache (not tracked)
results/accuracy/         Raw and final lm_eval quality artifacts
results/throughput/       Raw and final vLLM serving artifacts
scripts/                  Runners, parsers, and Slurm sweep scripts
scripts/accuracy/         Quality sweep scripts by model family
scripts/throughput/       Throughput sweep scripts by model family
scripts/archive/          Older scripts retained for reproducibility
