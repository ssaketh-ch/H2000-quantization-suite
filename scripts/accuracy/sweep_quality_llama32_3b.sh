#!/bin/bash
#SBATCH --job-name=llama32_3b_quality
#SBATCH --output=/home/saketh-msc/quantization/results/accuracy/logs/slurm_%j.log
#SBATCH --error=/home/saketh-msc/quantization/results/accuracy/logs/slurm_%j.err
#SBATCH --partition=mig_nodes
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=sssihl_h200

export CUDA_VISIBLE_DEVICES=MIG-71c1d4b3-76d9-5fe2-9ed0-ec8b95113573
set -euo pipefail

MIG_UUID="MIG-71c1d4b3-76d9-5fe2-9ed0-ec8b95113573"
MODELS_DIR="/home/saketh-msc/quantization/models"
SCRIPTS_DIR="/home/saketh-msc/quantization/scripts"
PORT="${PORT:-8001}"
QUALITY_TIER="${QUALITY_TIER:-tier2}"
MODEL_FILTER="${MODEL_FILTER:-}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_quality}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ~/quantization/results/accuracy/logs/sweep_quality_llama32_3b.log; }

should_run() {
  local MODEL=$1
  local QUANT=$2
  [ -z "$MODEL_FILTER" ] || [[ "$MODEL" == *"$MODEL_FILTER"* ]] || [[ "$QUANT" == *"$MODEL_FILTER"* ]]
}

run_container() {
  local MODEL=$1
  local MAX_LEN=$2
  log "Starting: $MODEL"
  docker stop "$CONTAINER_NAME" 2>/dev/null && docker rm "$CONTAINER_NAME" 2>/dev/null || true

  docker run -d \
    --gpus "device=$MIG_UUID" \
    --ipc=host \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "$MODELS_DIR:/root/.cache/huggingface" \
    -p "$PORT:8000" \
    --name "$CONTAINER_NAME" \
    vllm/vllm-openai:v0.19.0 \
      --model "$MODEL" \
      --gpu-memory-utilization 0.95 \
      --max-model-len "$MAX_LEN"

  log "Waiting for model to be ready..."
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null | grep -q "200"; then
      log "Ready"
      return 0
    fi
    sleep 10
  done
  log "Timeout - skipping"
  return 1
}

run_quality() {
  local MODEL=$1
  local QUANT=$2
  if should_run "$MODEL" "$QUANT"; then
    cd "$SCRIPTS_DIR" && ./run_quality_eval.sh "$MODEL" "$QUANT" "$PORT" "$QUALITY_TIER"
  else
    log "Skipping: $MODEL | $QUANT (MODEL_FILTER=$MODEL_FILTER)"
  fi
  docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME" || true
  sleep 5
}

log "===== Llama-3.2-3B Quality Sweep Started ($QUALITY_TIER) ====="
run_container "meta-llama/Llama-3.2-3B-Instruct" 8192 && run_quality "meta-llama/Llama-3.2-3B-Instruct" "bf16"
run_container "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic" 8192 && run_quality "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic" "fp8"
run_container "AMead10/Llama-3.2-3B-Instruct-AWQ" 8192 && run_quality "AMead10/Llama-3.2-3B-Instruct-AWQ" "awq"
run_container "shuyuej/Llama-3.2-3B-Instruct-GPTQ" 8192 && run_quality "shuyuej/Llama-3.2-3B-Instruct-GPTQ" "gptq"
log "===== Quality Sweep Complete ====="
