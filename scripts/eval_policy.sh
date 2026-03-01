#!/bin/bash

task="Grab the tape and put it in the box."

cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, \n
           wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" 

rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

rm -rf "/home/sorel/.cache/huggingface/lerobot/sorel/eval_so101"


lerobot-record  \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="$cameras" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id="sorel/eval_so101" \
  --dataset.push_to_hub=false \
  --dataset.single_task="$task" \
  --dataset.rename_map="$rename_map" \
  --dataset.episode_time_s=3600 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_awesome_leader_arm \
  --policy.path="sorel/smolvla-so101-0121"

#   --policy.path="sorel/act_so101-record-0121"
