#!/bin/bash
#SBATCH --job-name=quality_missing
#SBATCH --output=/home/saketh-msc/quantization/results/accuracy/logs/slurm_%j.log
#SBATCH --error=/home/saketh-msc/quantization/results/accuracy/logs/slurm_%j.err
#SBATCH --partition=mig_nodes
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=sssihl_h200

set -uo pipefail

MIG_UUID="${MIG_UUID:-MIG-71c1d4b3-76d9-5fe2-9ed0-ec8b95113573}"
MODELS_DIR="/home/saketh-msc/quantization/models"
SCRIPTS_DIR="/home/saketh-msc/quantization/scripts"
RESULTS_DIR="/home/saketh-msc/quantization/results/accuracy"
RESULTS_CSV="$RESULTS_DIR/quality_results.csv"
LOG_DIR="$RESULTS_DIR/logs"
PORT="${PORT:-8001}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_quality_missing}"
SWEEP_LOG="$LOG_DIR/sweep_quality_missing.log"
ERRORS=0

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SWEEP_LOG"
}

safe_model_name() {
  echo "$1" | sed 's|/|__|g'
}

latest_result_json() {
  local model_dir=$1
  find "$model_dir" -maxdepth 1 -type f -name 'results_*.json' -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | awk '{print $2}'
}

result_exists() {
  local model=$1
  local quant=$2
  local task=$3

  [ -f "$RESULTS_CSV" ] || return 1
  awk -F, -v model="$model" -v quant="$quant" -v task="$task" \
    'NR > 1 && $2 == model && $3 == quant && $5 == task { found = 1 } END { exit !found }' \
    "$RESULTS_CSV"
}

start_container() {
  local model=$1
  local image=$2
  local max_len=$3
  local extra_args=$4

  log "Starting container for $model"
  docker stop "$CONTAINER_NAME" 2>/dev/null && docker rm "$CONTAINER_NAME" 2>/dev/null || true
  docker run -d \
    --gpus "device=$MIG_UUID" \
    --ipc=host \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "$MODELS_DIR:/root/.cache/huggingface" \
    -p "$PORT:8000" \
    --name "$CONTAINER_NAME" \
    "$image" \
    --model "$model" \
    --gpu-memory-utilization 0.93 \
    --max-model-len "$max_len" \
    $extra_args

  log "Waiting for $model on port $PORT"
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null | grep -q "200"; then
      log "Ready: $model"
      return 0
    fi
    sleep 10
  done

  log "Timeout waiting for $model"
  return 1
}

stop_container() {
  docker stop "$CONTAINER_NAME" 2>/dev/null && docker rm "$CONTAINER_NAME" 2>/dev/null || true
  sleep 5
}

run_eval_task() {
  local model=$1
  local quant=$2
  local task=$3
  local fewshot=$4
  local tier=$5
  local safe_model
  local evals_dir
  local model_dir
  local before_json
  local after_json
  local task_log
  local lm_status

  if result_exists "$model" "$quant" "$task"; then
    log "Already present, skipping: $model | $quant | $task"
    return 0
  fi

  safe_model=$(safe_model_name "$model")
  evals_dir="$RESULTS_DIR/evals/$quant"
  model_dir="$evals_dir/$safe_model"
  task_log="$LOG_DIR/${safe_model}_${quant}_${task}.log"
  mkdir -p "$evals_dir"

  before_json=""
  if [ -d "$model_dir" ]; then
    before_json=$(latest_result_json "$model_dir" || true)
  fi

  log "Running missing task: model=$model quant=$quant task=$task fewshot=$fewshot tier=$tier"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  lm_eval \
    --model local-completions \
    --model_args "model=$model,base_url=http://localhost:$PORT/v1/completions,tokenizer_backend=huggingface,tokenizer=$model" \
    --apply_chat_template \
    --tasks "$task" \
    --num_fewshot "$fewshot" \
    --batch_size 1 \
    --output_path "$evals_dir" \
    2>&1 | tee "$task_log"
  lm_status=${PIPESTATUS[0]}

  if [ "$lm_status" -ne 0 ]; then
    log "lm_eval failed for $model | $quant | $task"
    return "$lm_status"
  fi

  after_json=$(latest_result_json "$model_dir" || true)
  if [ -z "$after_json" ]; then
    log "Could not find lm_eval results JSON for $model | $quant | $task"
    return 1
  fi
  if [ -n "$before_json" ] && [ "$before_json" = "$after_json" ]; then
    log "lm_eval did not produce a new results file for $model | $quant | $task"
    return 1
  fi

  python3 "$SCRIPTS_DIR/log_eval_result.py" \
    --json "$after_json" \
    --task "$task" \
    --model "$model" \
    --quant "$quant" \
    --tier "$tier"
}

run_model() {
  local model=$1
  local quant=$2
  local image=$3
  local max_len=$4
  local extra_args=$5
  shift 5

  local pending=0
  local spec
  local task
  local fewshot
  local tier

  for spec in "$@"; do
    IFS=: read -r task fewshot tier <<<"$spec"
    if ! result_exists "$model" "$quant" "$task"; then
      pending=1
      break
    fi
  done

  if [ "$pending" -eq 0 ]; then
    log "No missing tasks remain for $model | $quant"
    return 0
  fi

  if ! start_container "$model" "$image" "$max_len" "$extra_args"; then
    ERRORS=$((ERRORS + 1))
    stop_container
    return 1
  fi

  for spec in "$@"; do
    IFS=: read -r task fewshot tier <<<"$spec"
    if ! run_eval_task "$model" "$quant" "$task" "$fewshot" "$tier"; then
      ERRORS=$((ERRORS + 1))
    fi
  done

  stop_container
}

log "===== Missing Quality Sweep Started ====="

run_model "Qwen/Qwen3-32B" "bf16" "vllm/vllm-openai:v0.19.0" 8192 "" \
  "mmlu:5:tier2"

run_model "Qwen/Qwen3-32B-FP8" "fp8" "vllm/vllm-openai:v0.19.0" 8192 "" \
  "mmlu:5:tier2"

run_model "Qwen/Qwen3-32B-AWQ" "awq" "vllm/vllm-openai:v0.19.0" 8192 "" \
  "gsm8k:5:tier2" \
  "mmlu:5:tier2"

run_model "JunHowie/Qwen3-32B-GPTQ-Int4" "gptq" "vllm/vllm-openai:v0.19.0" 8192 "" \
  "gsm8k:5:tier2" \
  "mmlu:5:tier2"

run_model "Qwen/Qwen3-8B-FP8" "fp8" "vllm/vllm-openai:v0.19.0" 8192 "" \
  "mmlu:5:tier2"

run_model "Qwen/Qwen3-8B-AWQ" "awq" "vllm/vllm-openai:v0.19.0" 8192 "" \
  "gsm8k:5:tier2" \
  "mmlu:5:tier2"

run_model "JunHowie/Qwen3-8B-GPTQ-Int4" "gptq" "vllm/vllm-openai:v0.19.0" 8192 "" \
  "gsm8k:5:tier2" \
  "mmlu:5:tier2"

log "===== Missing Quality Sweep Complete: errors=$ERRORS ====="
exit "$ERRORS"
