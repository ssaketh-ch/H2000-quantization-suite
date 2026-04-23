# Scripts

This folder contains the runners and parsers for the benchmark suite.

## Shared Scripts

- `run_quality_eval.sh`: runs `lm_eval` against an already-running vLLM OpenAI-compatible server and logs the result.
- `run_bench.sh`: runs vLLM's serving benchmark inside a running container and appends parsed throughput metrics.
- `log_eval_result.py`: parses one `lm_eval` JSON result and appends to `results/accuracy/quality_results.csv`.
- `log_result.py`: parses one vLLM throughput text output and appends to `results/throughput/results.csv`.
- `build_final_tables.py`: rebuilds final CSVs in `data/` from real raw benchmark artifacts.
- `benchmark_serving.py`: local benchmark helper retained with the project.
- `backend_request_func.py`: helper module used by serving benchmark code.

## Subfolders

- `accuracy/`: Slurm sweep scripts for quality benchmarks.
- `throughput/`: Slurm sweep scripts for vLLM serving throughput.
- `archive/`: older or narrow scripts kept for reproducibility.

## Assumptions

- Docker is available on the target node.
- The requested model is cached under `models/`.
- `lm_eval` is installed in the host conda environment.
- vLLM containers expose OpenAI-compatible endpoints on the selected host port.

## Finalization

After new raw results are added, run:

```bash
python3 scripts/build_final_tables.py
```

This writes `data/accuracy.csv`, `data/throughput.csv`, `data/model_inventory.csv`, and leaderboard tables.
