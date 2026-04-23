#!/bin/bash
#SBATCH --job-name=llama32_3b_sweep
#SBATCH --output=/home/saketh-msc/quantization/results/throughput/logs/slurm_%j.log
#SBATCH --error=/home/saketh-msc/quantization/results/throughput/logs/slurm_%j.err
#SBATCH --partition=mig_nodes
#SBATCH --gres=gpu:mig-3g.71gb:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=sssihl_h200

MIG_UUID="MIG-71c1d4b3-76d9-5fe2-9ed0-ec8b95113573"
MODELS_DIR="/home/saketh-msc/quantization/models"
DATASET="/home/saketh-msc/quantization/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
SCRIPTS_DIR="/home/saketh-msc/quantization/scripts"
RATES=(1 2 4 8 16)
PORT=8001

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ~/quantization/results/throughput/logs/sweep_llama32_3b.log; }

run_container() {
  local MODEL=$1
  local MAX_LEN=$2
  log "Starting: $MODEL"
  docker stop vllm_bench 2>/dev/null && docker rm vllm_bench 2>/dev/null || true

  docker run -d \
    --gpus "device=$MIG_UUID" \
    --ipc=host \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v $MODELS_DIR:/root/.cache/huggingface \
    -v $DATASET:/tmp/sharegpt.json:ro \
    -p $PORT:8000 \
    --name vllm_bench \
    vllm/vllm-openai:v0.19.0 \
      --model $MODEL \
      --gpu-memory-utilization 0.95 \
      --max-model-len $MAX_LEN

  log "Waiting for model to be ready..."
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null | grep -q "200"; then
      log "Ready"
      return 0
    fi
    sleep 10
  done
  log "Timeout - skipping"
  return 1
}

run_sweep() {
  local MODEL=$1
  local QUANT=$2
  for RATE in "${RATES[@]}"; do
    PROMPTS=$((RATE * 120))
    log "  rate=$RATE prompts=$PROMPTS"
    cd $SCRIPTS_DIR && ./run_bench.sh vllm_bench $MODEL $QUANT $RATE $PROMPTS
  done
  log "Done: $MODEL | $QUANT"
  docker stop vllm_bench && docker rm vllm_bench || true
  sleep 5
}

log "===== Llama-3.2-3B Sweep Started ====="

run_container "meta-llama/Llama-3.2-3B-Instruct" 8192 && run_sweep "meta-llama/Llama-3.2-3B-Instruct" "bf16"
run_container "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic" 8192 && run_sweep "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic" "fp8"
run_container "AMead10/Llama-3.2-3B-Instruct-AWQ" 8192 && run_sweep "AMead10/Llama-3.2-3B-Instruct-AWQ" "awq"
run_container "shuyuej/Llama-3.2-3B-Instruct-GPTQ" 8192 && run_sweep "shuyuej/Llama-3.2-3B-Instruct-GPTQ" "gptq"

log "===== Sweep Complete ====="