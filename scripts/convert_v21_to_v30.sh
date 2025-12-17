#!/bin/bash

python ./src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
 --repo-id="record_1212" \
<<<<<<< HEAD
 --root="/root/autodl-tmp/data" \
 --push-to-hub=False
=======
 --root="/root/lerobot/tmp" \
 --push-to-hub=False
>>>>>>> 893d823 (feat(scripts): update dataset paths and add augment dataset script; enhance training logging)
