#!/bin/bash
# Usage: ./train.sh <policy> <dataset>

policy="$1"
dataset="$2"
steps="${3:-50_000}"

if [ -z "$dataset"  ] || [ -z "$policy" ]; then
    echo "Usage: $0 <policy> <dataset>"
    exit 1
fi

accelerate launch src/lerobot/scripts/lerobot_train.py \
  --job_name="${policy}-${dataset}" \
  --dataset.repo_id="sorel/${dataset}" \
  --policy.type=${policy} \
  --policy.repo_id="sorel/${policy}-${dataset}" \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=${steps}
