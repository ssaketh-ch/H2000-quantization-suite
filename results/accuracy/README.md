# Accuracy Results

This folder stores quality benchmark outputs produced by `lm_eval`.

## Final Table

Use this file for final accuracy analysis:

```text
results/accuracy/final/accuracy.csv
```

The same table is copied to:

```text
data/accuracy.csv
```

The final table has 240 rows: 40 model-quantization configurations times 6 benchmark tasks.

## Raw Files

- `quality_results.csv`: append-only run log created during quality sweeps.
- `evals/`: raw `lm_eval` JSON output files grouped by quantization and model.
- `logs/`: Slurm logs and task logs.
- `final/`: final accuracy CSV and its folder README.

## Why The Final Table Is Built From JSON

The append-only run log is useful for provenance, but the final table is built from raw JSON files. This is more reliable because:

- it avoids stale paths from older runs;
- it avoids duplicate run-log rows;
- it reads the actual numeric metric from each `lm_eval` result block;
- it keeps the latest result for each `(model, quant, task)` triple.

Regenerate the final table with:

```bash
python3 scripts/build_final_tables.py
```
