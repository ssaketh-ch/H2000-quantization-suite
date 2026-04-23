#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <model_id> <quant_label> <port> [tier1|tier2|all]" >&2
  exit 1
fi

MODEL=$1
QUANT=$2
PORT=$3
TIER=${4:-all}

RESULTS_DIR="/home/saketh-msc/quantization/results/accuracy"
EVALS_DIR="$RESULTS_DIR/evals/$QUANT"
LOG_DIR="$RESULTS_DIR/logs"
SCRIPT_DIR="/home/saketh-msc/quantization/scripts"

mkdir -p "$EVALS_DIR" "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
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

run_task() {
  local TASK=$1
  local FEWSHOT=$2
  local TIER_LABEL=$3
  local SAFE_MODEL
  local MODEL_DIR
  local BEFORE_JSON
  local AFTER_JSON
  local TASK_LOG

  SAFE_MODEL=$(safe_model_name "$MODEL")
  MODEL_DIR="$EVALS_DIR/$SAFE_MODEL"
  TASK_LOG="$LOG_DIR/${SAFE_MODEL}_${QUANT}_${TASK}.log"

  BEFORE_JSON=""
  if [ -d "$MODEL_DIR" ]; then
    BEFORE_JSON=$(latest_result_json "$MODEL_DIR" || true)
  fi

  log "Running task=$TASK fewshot=$FEWSHOT tier=$TIER_LABEL model=$MODEL"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  lm_eval \
    --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,tokenizer_backend=huggingface,tokenizer=$MODEL" \
    --apply_chat_template \
    --tasks "$TASK" \
    --num_fewshot "$FEWSHOT" \
    --batch_size 1 \
    --output_path "$EVALS_DIR" \
    2>&1 | tee "$TASK_LOG"

  AFTER_JSON=$(latest_result_json "$MODEL_DIR")
  if [ -z "$AFTER_JSON" ]; then
    echo "Could not find lm_eval results JSON for $TASK" >&2
    exit 1
  fi
  if [ -n "$BEFORE_JSON" ] && [ "$BEFORE_JSON" = "$AFTER_JSON" ]; then
    echo "lm_eval did not produce a new results file for $TASK" >&2
    exit 1
  fi

  python3 "$SCRIPT_DIR/log_eval_result.py" \
    --json "$AFTER_JSON" \
    --task "$TASK" \
    --model "$MODEL" \
    --quant "$QUANT" \
    --tier "$TIER_LABEL"
}

if ! curl -fsS "http://localhost:$PORT/health" >/dev/null; then
  echo "vLLM health check failed on port $PORT" >&2
  exit 1
fi

case "$TIER" in
  tier1)
    run_task "mmlu_abstract_algebra" 5 "tier1"
    run_task "hellaswag" 10 "tier1"
    run_task "winogrande" 5 "tier1"
    run_task "arc_challenge" 25 "tier1"
    ;;
  tier2)
    #run_task "gsm8k" 5 "tier2"
    run_task "mmlu" 5 "tier2"
    ;;
  all)
    run_task "mmlu_abstract_algebra" 5 "tier1"
    run_task "hellaswag" 10 "tier1"
    run_task "winogrande" 5 "tier1"
    run_task "arc_challenge" 25 "tier1"
    run_task "gsm8k" 5 "tier2"
    run_task "mmlu" 5 "tier2"
    ;;
  *)
    echo "Unknown tier '$TIER'. Use tier1, tier2, or all." >&2
    exit 1
    ;;
esac

log "Completed quality eval for $MODEL | $QUANT | $TIER"
