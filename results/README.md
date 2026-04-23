# Results

This folder contains raw and final benchmark artifacts.

For analysis, use the final CSVs in `data/` first:

- `data/accuracy.csv`
- `data/throughput.csv`
- `data/model_inventory.csv`
- `data/accuracy_leaderboard.csv`
- `data/throughput_leaderboard.csv`

The `results/` folder is kept for traceability. It lets you audit how the final CSVs were produced.

## Layout

- `accuracy/`: quality benchmark outputs from `lm_eval`.
- `throughput/`: serving benchmark outputs from vLLM.

## Raw Versus Final

Raw artifacts are useful when debugging failed runs, checking a model's exact `lm_eval` output, or validating a parser. Final CSVs are the intended surface for GitHub tables, Hugging Face datasets, notebooks, and plots.

The final accuracy CSV is also mirrored at:

```text
results/accuracy/final/accuracy.csv
```

The final throughput CSV is also mirrored at:

```text
results/throughput/final/throughput.csv
```
