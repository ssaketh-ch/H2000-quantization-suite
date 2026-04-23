#!/bin/bash
# Usage: ./run_bench.sh <container_name> <model_id> <quant_label> <request_rate> [num_prompts]
CONTAINER=$1
MODEL=$2
QUANT=$3
RATE=$4
NUM_PROMPTS=${5:-200}

# Clean model name (strip org prefix: deepseek-ai/Model-Name → Model-Name)
MODEL_NAME=$(echo $MODEL | sed 's|.*/||')

# Create per-model subfolder under quant label
OUTDIR="/home/saketh-msc/quantization/results/throughput/${QUANT}/${MODEL_NAME}"
mkdir -p $OUTDIR

OUTFILE="${OUTDIR}/rate${RATE}.txt"

echo "Benchmarking: $MODEL | $QUANT | rate=$RATE | prompts=$NUM_PROMPTS"
echo "   Saving raw output to: $OUTFILE"

docker exec $CONTAINER \
  python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai-chat \
  --host localhost --port 8000 \
  --endpoint /v1/chat/completions \
  --model $MODEL \
  --dataset-name sharegpt \
  --dataset-path /tmp/sharegpt.json \
  --num-prompts $NUM_PROMPTS \
  --request-rate $RATE \
  --temperature 0 | tee $OUTFILE

python3 /home/saketh-msc/quantization/scripts/log_result.py \
  --file $OUTFILE \
  --model $MODEL \
  --quant $QUANT \
  --rate $RATE

echo ""
