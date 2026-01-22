#!/bin/bash

task="Grab the tape and put it in the box."

cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
    wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}"

rm -rf "/home/sorel/.cache/huggingface/lerobot/sorel/eval_so101"

lerobot-record  \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.cameras="$cameras" \
  --robot.id=my_awesome_follower_arm \
  --display_data=false \
  --dataset.repo_id="sorel/eval_so101" \
  --dataset.push_to_hub=false \
  --dataset.single_task="$task" \
  --policy.path="sorel/act_so101-record-0121"
  # <- Teleop optional if you want to teleoperate in between episodes \
  # --teleop.type=so100_leader \
  # --teleop.port=/dev/ttyACM0 \
  # --teleop.id=my_awesome_leader_arm \
