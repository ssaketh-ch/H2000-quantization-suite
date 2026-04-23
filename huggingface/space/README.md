---
title: H200 Quantization Dashboard
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# H200 Quantization Dashboard

Interactive Gradio dashboard for the H200 quantization benchmark tables.

The app reads local CSV files bundled with the Space:

- `data/accuracy.csv`
- `data/throughput.csv`
- `data/model_inventory.csv`
- `data/accuracy_leaderboard.csv`
- `data/throughput_leaderboard.csv`

The associated dataset repo is expected to be:

```text
ssakethch/h200-quantization-benchmarks
```
