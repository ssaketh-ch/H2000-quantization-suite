#!/bin/bash
#SBATCH --job-name=deepseek_qwen7b_sweep
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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ~/quantization/results/throughput/logs/sweep_deepseek_qwen7b.log; }

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

log "===== DeepSeek-R1-Distill-Qwen-7B Sweep Started ====="

run_container "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 8192 && run_sweep "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "bf16"
run_container "neuralmagic/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic" 8192 && run_sweep "neuralmagic/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic" "fp8"
run_container "AngelSlim/Deepseek_r1_distill_qwen-7b_int4_awq" 8192 && run_sweep "AngelSlim/Deepseek_r1_distill_qwen-7b_int4_awq" "awq"
run_container "jakiAJK/DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4" 8192 && run_sweep "jakiAJK/DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4" "gptq"

log "===== Sweep Complete ====="