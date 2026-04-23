# Raw Accuracy JSONs

This folder contains raw `lm_eval` JSON result files grouped as:

```text
evals/<quant>/<safe_model_name>/results_*.json
```

These files are the source of truth for measured accuracy metrics, but the cleaned analysis tables live in `../final/`.

The folder is ignored by Git because raw result artifacts can grow quickly. Keep the final CSVs tracked instead.
