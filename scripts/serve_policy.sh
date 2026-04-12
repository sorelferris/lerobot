#!/bin/bash

usage="$0 <policy_type> <pretrained_name_or_path>"

policy_type="$1"
ckpt="$2"

if [ -z "$ckpt" ] || [ -z "$policy_type" ]; then
    echo "Usage: $usage"
    exit 1
fi

# Rename robot observation keys to policy input keys if necessary. 
# For example, if the robot provides "observation.images.right_camera" but the policy expects "observation.images.right", we can set:
rename_map='{"observation.images.right": "observation.images.right_camera"}'


# Uncomment below if no rename is needed.
rename_map='{}'


python scripts/tools/policy_server.py \
    --host=0.0.0.0 \
    --port=8001 \
    --policy_type="$policy_type" \
    --pretrained_name_or_path="$ckpt" \
    --policy_device=cuda \
    --rename_map="${rename_map}" \
    --actions_per_chunk=50
