#!/bin/bash

# For RTX 4000 series doesn't support faster communication braodband via P2P or IB
# export NCCL_P2P_DISABLE="1"
# export NCCL_IB_DISABLE="1"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_ENDPOINT="https://hf-mirror.com"

rm -rf ./outputs/pi05_training
python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id="record_1212" \
    --dataset.root="./tmp/record_1212" \
    --policy.type=pi05 \
    --output_dir="./outputs/pi05_training" \
    --job_name="pi05_training" \
    --policy.repo_id="finetune_pi05" \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32