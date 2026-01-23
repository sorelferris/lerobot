#!/bin/bash

repo_id="sorel/so101-record-0121"
policy_type="pi05"

data_name="${repo_id#*/}"
policy_repo_id="sorel/${policy_type}-${data_name}"

rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

# For RTX 4000 series GPUs to avoid NCCL errors
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python src/lerobot/scripts/lerobot_train.py\
    --job_name="train-pi05-${data_name}" \
    --dataset.repo_id=${repo_id} \
    --policy.type=pi05 \
    --policy.repo_id=${policy_repo_id} \
    --policy.pretrained_path="lerobot/pi05_base" \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=true \
    --wandb.enable=false \
    --batch_size=32 \
    --steps=50_000