#!/bin/bash

repo_id="sorel/so101-record-0121"
data_name="${repo_id#*/}"

policy_type="act"
policy_repo_id="sorel/${policy_type}_${data_name}"


lerobot-train \
  --dataset.repo_id=${repo_id} \
  --policy.type=${policy_type} \
  --policy.repo_id=${policy_repo_id} \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=100_000 \
  --job_name="train_${policy_type}_${repo_id}" 
#   --output_dir="outputs/train/${policy_type}_${data_name}" \
