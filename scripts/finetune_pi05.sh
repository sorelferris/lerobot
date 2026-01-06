#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_ENDPOINT="https://hf-mirror.com"


repo_id="record_1212"

output_dir="/root/autodl-tmp/outputs/pi05_train"

rm -rf $output_dir

nohup python src/lerobot/scripts/lerobot_train.py\
    --dataset.repo_id=$repo_id \
    --dataset.root="/root/autodl-tmp/data/$repo_id" \
    --dataset.video_backend="pyav" \
    --policy.type=pi05 \
    --output_dir=$output_dir \
    --job_name="pi05_train" \
    --policy.repo_id="finetune_pi05" \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --steps=50000 \
    --policy.device=cuda \
    --batch_size=32 \
    > log/train.log 2>&1 &
