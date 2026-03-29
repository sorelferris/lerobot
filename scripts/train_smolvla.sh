#!/bin/bash

repo_id="sorel/so101-record-0121"
policy_type="smolvla"

data_name="${repo_id#*/}"
policy_repo_id="sorel/${policy_type}-${data_name}"

rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

# For RTX 4000 series GPUs to avoid NCCL errors
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

lerobot-train \
  --job_name="train-smolvla-${data_name}" \
  --dataset.repo_id=${repo_id} \
  --rename_map="${rename_map}" \
  --policy.repo_id=${policy_repo_id} \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --wandb.enable=true \
  --batch_size=32 \
  --steps=50_000
