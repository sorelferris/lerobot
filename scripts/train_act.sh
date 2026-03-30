#!/bin/bash


dataset="$1"

if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

policy_type="act"

lerobot-train \
  --job_name="${policy_type}-${dataset}" \
  --dataset.repo_id="sorel/${dataset}" \
  --dataset.root="data/train_data/lerobot_v3.0/${dataset}" \
  --policy.type=${policy_type} \
  --policy.repo_id="sorel/${policy_type}-${dataset}" \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=100_000 \
