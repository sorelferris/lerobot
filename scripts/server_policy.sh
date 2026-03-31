#!/bin/bash

# Rename robot observation keys to policy input keys if necessary. 
# For example, if the robot provides "observation.images.right_camera" but the policy expects "observation.images.right", we can set:
rename_map='{"observation.images.right": "observation.images.right_camera"}'


# Uncomment below if no rename is needed.
rename_map='{}'


python -m lerobot.async_inference.policy_server_zmq \
    --host=0.0.0.0 \
    --port=8001 \
    --policy_type="act" \
    --pretrained_name_or_path="outputs/train/2026-03-31/13-37-49_act-record_0330_right_only/checkpoints/020000/pretrained_model" \
    --policy_device=cuda \
    --rename_map="${rename_map}" \
    --actions_per_chunk=50
