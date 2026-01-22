#!/bin/bash

repo_id="sorel/so101-record-0121"
data_name="${repo_id#*/}"

policy_type="smolvla"
policy_repo_id="sorel/smolvla-${data_name}"

rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

lerobot-train \
  --dataset.repo_id=${repo_id} \
  --rename_map="${rename_map}" \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=${policy_repo_id} \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=50_000 \
  --job_name="train-smolvla-${data_name}"
