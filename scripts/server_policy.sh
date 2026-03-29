#!/bin/bash

python -m lerobot.async_inference.policy_server_zmq \
    --host=0.0.0.0 \
    --port=8000 \
    --policy_type="pi05" \
    --pretrained_name_or_path="data/ckpt/2026-03-03/16-40-42_train-pi05-record_1212/checkpoints/050000/pretrained_model" \
    --policy_device=cuda \
    --actions_per_chunk=50
